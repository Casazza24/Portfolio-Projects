# Data quality: what the event stream got wrong, and what the pipeline did about it

Phase 3 deliverable. Covers the defects present in (and injected into) the
RetailRocket event stream, how the Spark cleaning layer detects and handles
each one, and the trade-offs that were chosen deliberately rather than by
default.

Figures below are from a full run over the complete dataset on 2026-07-30.

---

## 1. Headline numbers

| Measure | Count | % of messages |
|---|---:|---:|
| Source events in `events.csv` | 2,756,101 | — |
| Messages produced to Redpanda | 2,811,351 | 100% |
| Clean events written | 2,742,263 | 97.54% |
| Quarantined (rejected) | 13,838 | 0.49% |
| Duplicates collapsed by dedupe | 55,250 | 1.97% |

These reconcile exactly: `2,811,351 − 13,838 − 55,250 = 2,742,263`, and
`2,756,101 − 13,838 = 2,742,263`. Every source event is accounted for as
either cleaned or quarantined, and every duplicate delivery was collapsed.
Nothing was silently dropped — which is the property that actually matters.

---

## 2. Where the messiness comes from

Two different origins, and the distinction matters when reading the numbers:

**Injected faults (deliberate).** RetailRocket's raw data is fairly clean, so
a cleaning layer built against it as-shipped would have almost nothing to do
and would prove nothing. The replay producer therefore injects realistic
faults at controlled, seeded rates (`ingestion/replay_retailrocket.py`):

| Fault | Rate | Models |
|---|---:|---|
| Duplicate delivery | 2% | An at-least-once broker redelivering on ack timeout — same `event_id`, same body |
| Late / out-of-order arrival | 1% | A slow partition or a buffering client; held back up to 5,000 stream positions |
| `null_item` | ~0.1% | An upstream service writing a null foreign key |
| `bad_timestamp` | ~0.1% | A client sending an ISO-8601 string into a milliseconds field |
| `missing_field` | ~0.1% | Schema drift — a producer stops emitting `event_type` |
| `truncated_json` | ~0.1% | A payload cut off mid-write |
| `epoch_zero` | ~0.1% | An uninitialised clock |

**Structural properties of the real data (not injected).** Out-of-order
arrival is also inherent, not just injected: the topic has 3 partitions keyed
by `visitor_id`, so ordering is guaranteed *per visitor* but never globally.
Roughly two-thirds of messages arrive out of global event-time order for this
reason alone. Any pipeline that assumes a globally sorted stream is wrong, and
this one is built to demonstrate that.

### Detection is not allowed to cheat

The producer tags each injected fault with a `_fault` field, but
`spark/transforms.py` **never reads it**. Validity is judged from the payload
alone, exactly as it would have to be against a real stream that carries no
such marker. `_fault` is joined back only *afterwards*, to score whether
detection caught what was injected:

| Injected fault | Detected as | Count |
|---|---|---:|
| `null_item` | `missing_item_id` | 2,805 |
| `bad_timestamp` | `event_time_not_numeric` | 2,773 |
| `truncated_json` | `unparseable_json` | 2,759 |
| `epoch_zero` | `event_time_implausible` | 2,755 |
| `missing_field` | `missing_event_type` | 2,746 |

Every injected fault was caught, each classified with the correct root cause,
and no clean event was falsely rejected — the matrix is perfectly diagonal.
Because the totals reconcile exactly, the source data contained **no
additional defects** of these kinds beyond what was injected.

---

## 3. Handling decisions

### 3.1 Rejects are quarantined, not dropped

Failed rows go to a dead-letter Parquet dataset (`data/quarantine/events`)
holding the **original bytes verbatim**, the Kafka partition/offset, the
specific `dq_reason`, and a quarantine timestamp.

Keeping the raw payload is the point: a quarantine record that has already
been reshaped can't be used to diagnose why it was rejected, and can't be
replayed once the upstream bug is fixed. The Kafka coordinates make each
rejection traceable back to an exact position in the topic.

### 3.2 Classification reports root cause, not symptom

Checks are ordered most-fundamental-first, so a truncated payload is reported
as `unparseable_json` rather than as five separate missing fields.

This needed a non-obvious fix. Spark's `from_json` in PERMISSIVE mode does
**not** reliably return a null struct for malformed input — for a payload
truncated mid-write it returns a struct with every field null. Checking only
`e.isNull()` therefore misreported truncated payloads as `missing_event_id`,
pointing a reader at the wrong upstream owner. The dependable signal is an
all-null struct arising from a *non-empty* payload.

### 3.3 Late arrivals: a bounded-state trade-off

Streaming dedupe uses `dropDuplicatesWithinWatermark(["event_id"])` over a
**3-day watermark**. This is a genuine trade-off, chosen not defaulted:

- Holding every `event_id` ever seen would dedupe perfectly but grow state
  without bound — untenable for a long-running stream.
- A watermark bounds that state, at the cost that a redelivery arriving more
  than 3 days (in event time) after its original will slip through.

The 3 days is **derived, not guessed**: the verification consumer measured a
maximum observed lateness of ~27 hours, driven by the producer holding events
back up to 5,000 stream positions across the sparse overnight stretches of the
data. 3 days leaves roughly 2.6× headroom.

It worked: the streaming output contains **0 residual duplicates** across
2,742,263 events — identical to the batch path, which dedupes globally.
`numRowsDroppedByWatermark` is surfaced every micro-batch by the job's
monitor, because beyond-watermark drops are otherwise silent data loss.

### 3.4 Missing item metadata: unknown-member, not delete or impute

**9.27% of clean events (254,336) reference an item with no metadata** in
RetailRocket's `item_properties` feed. Breakdown by event type:

| Event type | Events | Unknown item |
|---|---:|---:|
| `view` | 2,650,959 | 9.54% |
| `transaction` | 22,333 | **2.12%** |
| `addtocart` | 68,971 | **1.20%** |

**This distribution is the finding.** The gap is concentrated almost entirely
in `view` events; the commercially significant events are far better covered.
Revenue-by-category analysis in Phase 4 is therefore only marginally affected,
and the impact is confined to top-of-funnel reach metrics.

Three options were considered:

- **Delete the events** — rejected. These are real user actions; only a
  *different* file fails to describe the item. Deleting them biases the
  dataset against poorly-documented (likely long-tail) inventory and makes
  funnel counts silently wrong.
- **Impute a category** — rejected, and worth being precise about why: the
  value doesn't exist anywhere in the source, so imputation cannot *recover*
  it, only invent it. That would move real transactions into categories those
  items were never in, corrupting the exact metric the marts exist to produce.
- **Unknown member (chosen).** The event is kept with its real `item_id`,
  and its category keys resolve to the sentinel `-1`, which exists as a real
  row in `dim_items`.

A `has_item_metadata` boolean is carried on every fact row, so an analyst can
exclude unknowns as a deliberate choice rather than having that choice baked
in irreversibly upstream.

Why a sentinel beats leaving nulls:

1. **Power BI silently drops nulls from slicers.** A null category makes
   filtered visuals fail to sum to the unfiltered total, with no visible cause.
   A 9%-tall `Unknown` bar is self-documenting.
2. **The Phase 4 `relationships` test can run strict.** With nulls, the test
   would have to tolerate them and could no longer catch a genuinely broken
   join later.
3. **The gap becomes a tracked metric.** `% events on unknown items` belongs
   on the pipeline-health dashboard page; a jump from 9% to 30% means an
   upstream metadata feed has broken.

### 3.5 Current-state vs point-in-time category

`dim_items` is built from an EAV change-log (~20M rows) by taking the **latest
known** value per `(item_id, property)`. A strictly correct SCD-2 treatment
would join each event to the category in effect *at that event's timestamp*.

Current-state was chosen because it can be broadcast into a streaming join,
whereas a range join against a versioned dimension cannot. The cost: events
occurring before a re-categorisation are attributed to the item's newer
category. This is bounded and small, and is the natural Phase 5+ upgrade if
category-level trend accuracy ever matters more than streaming simplicity.

Also noted: **132 category IDs referenced by items do not appear in
`category_tree.csv`**, so those items have a `category_id` but no resolvable
parent or root. They are left as-is rather than force-fitted into the tree.

### 3.6 Weather dimension: single point per state

`ingestion/fetch_weather.py` produces one row per `(state, date)` — 20,898
rows across 27 states and 774 days, no nulls. Each state's coordinates are the
**median lat/lng of that state's Olist geolocation rows**, clipped to Brazil's
bounding box first because the source contains mis-geocoded points outside the
country.

Using the data's own median rather than hardcoded capitals puts the point near
the population centre. It remains an approximation — a single point cannot
represent a large, climatically mixed state like Amazonas — but ~90% of orders
originate in compact southeastern states, so it is proportionate to the
question being asked ("did revenue dip in bad weather?").

---

## 4. Proving the two code paths agree

The streaming job and the batch backfill share every transform
(`spark/transforms.py`); they differ only in how data is read and written. That
is what stops them drifting — but it's a claim that has to be *verified*, not
asserted. `spark/compare_outputs.py` does that:

```
batch      : 2,742,263 rows, 2,742,263 distinct event_ids
streaming  : 2,742,263 rows, 2,742,263 distinct event_ids
in batch only     : 0
in streaming only : 0

column agreement over 2,742,263 shared event_ids
  event_time_ms / visitor_id / event_type / item_id /
  transaction_id / category_id / root_category_id / has_item_metadata
  -> 0 mismatches on every column

PASS
```

The watermarked streaming dedupe and the global batch dedupe produced
**identical** results on this dataset, which is the empirical confirmation
that the 3-day watermark in §3.3 was sized correctly.

---

## 5. Known limitations

- **Small-file fragmentation.** The streaming sink partitions by `event_date`
  across ~150 dates and 14 micro-batches, producing many small Parquet files.
  Fine for a local Snowflake load; a production deployment would need a
  compaction step.
- **Watermark applies to dedupe only.** Beyond-watermark redeliveries are
  reported via `numRowsDroppedByWatermark` but not separately quarantined.
- **`transaction_id` is not validated** beyond type-casting; referential
  integrity against Olist orders is a Phase 4 concern.
- **Current-state dimension** — see §3.5.
