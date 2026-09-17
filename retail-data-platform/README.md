# Retail Data Platform

End-to-end ELT pipeline combining relational order data (Olist) and simulated real-time
event streams (RetailRocket), processed through Redpanda + Spark, modeled in
Snowflake with dbt, and surfaced in a Power BI dashboard.

Full scope, phased build plan, and architecture: [`docs/project_scope.md`](docs/project_scope.md)

**Status:** Phases 1–5 complete as of 2026-08-11 — Postgres + Olist loaded, RetailRocket
events streaming through Redpanda, PySpark cleaning layer producing validated Parquet,
warehouse extract + dim_date built, dbt models written, and Airflow orchestrating the
full pipeline end-to-end. Snowflake load pending account provisioning.

## Stack

- **Source (OLTP):** Olist Brazilian E-Commerce dataset → Postgres
- **Source (events):** RetailRocket clickstream → replayed → Redpanda
- **Processing:** PySpark (Structured Streaming, with a batch backfill path)
- **Warehouse:** Snowflake (raw → staging → marts) *(Phase 4)*
- **Transformation:** dbt-core *(Phase 4)*
- **Orchestration:** Airflow (LocalExecutor) *(Phase 5)*
- **BI:** Power BI *(Phase 6)*

## Structure

- `ingestion/` — Postgres load scripts (Olist), Redpanda producer + verification consumer
- `spark/` — cleaning transforms, streaming job, batch backfill, reconciliation check
- `warehouse/` — Postgres extract, Snowflake raw DDL, stage/COPY loader, `dim_date` generator *(Phase 4)*
- `dbt/` — dbt-core project (staging + marts models, tests) *(Phase 4)*
- `airflow/` — DAG(s) sequencing the pipeline *(Phase 5)*
- `docs/` — scope doc, data quality write-up, architecture diagram, ERD

## Prerequisites

- Docker (Postgres + Redpanda)
- **Java 11 or 17.** Spark 3.5 does not support newer JDKs; if your default `java` is
  newer, the jobs locate a supported JDK automatically via `/usr/libexec/java_home`
  (see [`spark/common.py`](spark/common.py)). Amazon Corretto 11 is what this was built against.
- Python 3.9+

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Setup

**1. Infrastructure (Postgres + Redpanda + Redpanda Console):**

```
docker compose up -d
./ingestion/load_olist.sh
./ingestion/create_topics.sh
```

Postgres 16 on `localhost:5432` (db/user/pass all `retail`) with the Olist CSVs loaded
into the `olist` schema. Redpanda on `localhost:19092` with a 3-partition
`retail.events` topic, and the Redpanda Console UI on
[localhost:8080](http://localhost:8080) for browsing topics and messages.

**2. Produce the event stream (Phase 2):**

```
.venv/bin/python ingestion/replay_retailrocket.py --sink kafka --speed 0
```

Reads `data/raw/retailrocket/events.csv` (2.76M events), sorts by timestamp, and
produces to Redpanda keyed by `visitor_id`. `--speed 0` fires as fast as possible
(~109k msg/s, full replay in ~26s); `--speed 1` replays at the original real-time pace
and `--speed 10000` is a good middle ground for a screen recording.

The producer **deliberately injects data quality faults** so the Phase 3 cleaning layer
has real work to do — 2% duplicate deliveries, 1% late/out-of-order arrivals, and 0.5%
malformed payloads across five fault types. Rates are configurable (`--dup-rate`,
`--late-rate`, `--malformed-rate`) and seeded (`--seed`) so runs are reproducible.
See [`docs/data_quality.md`](docs/data_quality.md).

**3. Verify the stream:**

```
.venv/bin/python ingestion/consume_events.py --from-beginning --summary
```

Independent consumer that proves the producer → Redpanda → consumer loop works and
reports what the stream actually contains (validity, duplicates, out-of-order rate,
partition balance).

**4. Weather dimension (external API ingestion, feeds Phase 4):**

```
.venv/bin/python ingestion/fetch_weather.py
```

Derives each Brazilian state's coordinates from the median of Olist's own
geolocation rows, then pulls daily historical weather from Open-Meteo (no API key)
for the order date range. Produces `data/clean/dim_weather.csv` — 20,898 rows,
27 states × 774 days.

**5. Clean the events (Phase 3):**

```
.venv/bin/python spark/build_item_metadata.py     # item/category dimension (run once)
.venv/bin/python spark/streaming_events.py --reset --stop-when-idle   # live path
.venv/bin/python spark/batch_backfill.py                              # backfill path
.venv/bin/python spark/compare_outputs.py                             # reconcile the two
```

Outputs Parquet partitioned by `event_date`, plus a quarantine dataset holding every
rejected payload verbatim with the reason it was rejected.

## Phase 3 results

Full run over the complete dataset (2026-07-30):

| Measure | Count | % of messages |
|---|---:|---:|
| Messages produced to Redpanda | 2,811,351 | 100% |
| Clean events written | 2,742,263 | 97.54% |
| Quarantined | 13,838 | 0.49% |
| Duplicates collapsed | 55,250 | 1.97% |

Every injected fault was detected and classified with the correct root cause, with no
clean event falsely rejected. Streaming and batch outputs reconcile **exactly** — same
row count, same event_id set, zero column mismatches across 2,742,263 events.

Full write-up, including the trade-offs taken deliberately:
[`docs/data_quality.md`](docs/data_quality.md)

## Phase 3 design notes

**Two paths, one set of transforms.** All cleaning logic lives in
[`spark/transforms.py`](spark/transforms.py) as pure DataFrame functions. The streaming
job and the batch backfill differ only in how they read and write — which is what stops
the two paths from drifting. `spark/compare_outputs.py` verifies they actually agree.

**Rejects are quarantined, not dropped.** Bad payloads go to a dead-letter Parquet
dataset with the original bytes, the Kafka partition/offset, and a specific
`dq_reason`, so they can be diagnosed and replayed once the upstream bug is fixed.

**Detection doesn't cheat.** The producer tags injected faults with a `_fault` field,
but the cleaning logic never reads it — validity is judged from the payload alone, the
way it would have to be against a real stream. `_fault` is only joined back afterwards
to *score* whether detection caught what was injected.

**Watermark sizing is derived, not guessed.** Streaming dedupe uses
`dropDuplicatesWithinWatermark` over a 3-day watermark, sized from the producer's
observed maximum lateness. The trade-off (bounded state vs perfect dedupe) and the
measured cost of it are in [`docs/data_quality.md`](docs/data_quality.md).

## Phase 4 setup — warehouse load

Everything Snowflake-facing lives in `warehouse/`. It is deliberately split into a
Postgres/local half and a Snowflake half, so the source side can be run and checked
without a warehouse account existing.

**6. Build the date dimension:**

```
.venv/bin/python warehouse/build_dim_date.py
```

Derives its range from both sources — the order dates in Postgres and the
`event_date=` partition names Spark wrote — then widens to whole calendar years so
YTD and prior-year comparisons don't fall off the edge. Produces
`data/clean/dim_date.csv`: **1,462 rows** covering 2015-01-01 → 2018-12-31, plus the
Kimball `-1` unknown member (Phase 3 already resolves unknown categories to `-1`, and
the Phase 4 `relationships` tests can only run strict if every key resolves).

**7. Extract Olist from Postgres:**

```
.venv/bin/python warehouse/extract_postgres.py
```

Streams all 9 tables out with `COPY ... TO STDOUT WITH (FORMAT csv)` into gzipped CSV
under `data/warehouse/olist/` — **1,550,922 rows, 41.2 MB**. Postgres's own serializer
rather than a Python row loop, because `order_reviews`' free-text comments contain
embedded newlines, commas and quotes, and because the NULL-vs-empty-string distinction
survives the round trip into Snowflake unchanged.

**8. Stand up Snowflake and load:**

```
cp .env.example .env      # fill in SNOWFLAKE_ACCOUNT / USER / PASSWORD
.venv/bin/python warehouse/load_snowflake.py --dry-run     # inspect the statements
.venv/bin/python warehouse/load_snowflake.py --init        # DDL, then load everything
```

`warehouse/raw_schema.sql` creates the database, warehouse, `raw`/`staging`/`marts`
schemas, file formats, internal stage and all raw tables in one shot.
`warehouse/load_snowflake.py` then `PUT`s the Olist CSVs, the events Parquet and the
three dimensions to the internal stage and `COPY INTO`s them. Credentials come from
the environment or `.env` (gitignored) — never from the code.

## Phase 4 design notes

**Small files fixed at the source.** The batch backfill now repartitions on
`event_date` before writing, so each date produces one file instead of one per
upstream task: **1,112 files → 139**, 177 MB → 171 MB for the same 2,742,263 rows.
The target file size is derived, not hardcoded — `bytes_per_row()` measures how this
schema actually compressed on the previous run and sizes `maxRecordsPerFile` from it,
falling back to a constant only on a cold start. Snowflake's 100–250 MB guidance is
still unreachable *while* partitioning daily (184 MB over 139 days is ~1.3 MB/day);
monthly partitioning would land ~40 MB files. Daily was kept because `event_date` is
what every mart filters on and Snowflake prunes on its own micro-partitions anyway —
the file count was the part of the problem worth fixing at this volume.

**PUT + COPY INTO, not Snowpipe or S3.** An internal stage is free, needs no cloud
account, and is one scriptable step. Snowpipe is for continuous arrival, which a local
batch does not have; an external stage would add a bucket and IAM for no gain.

**Every load is a full reload.** Truncate, then `COPY ... FORCE = TRUE`. Snowflake
skips files it has already loaded within 64 days, so without `FORCE` a re-run of an
unchanged extract silently loads nothing and looks like it worked. At ~215 MB total a
full reload is always correct and always cheap.

**`event_date` is rederived, not parsed out of the path.** It is a Hive partition
column, so Spark left it out of the Parquet files. The `COPY` recomputes it as
`to_date(event_time)` — the same expression `spark/transforms.py` used. Parsing
`METADATA$FILENAME` is the fallback for a partition key that *can't* be recovered from
a retained column; deriving from the data can't be broken by reorganising the stage.

**Verified vs. pending a Snowflake account.** There is no Snowflake account yet, so
the split matters:

| | Status |
|---|---|
| Parquet compaction (`spark/batch_backfill.py`) | **Run** — full backfill, counts unchanged from Phase 3 |
| `warehouse/build_dim_date.py` | **Run** — 1,462 rows, 2015-01-01 → 2018-12-31 |
| `warehouse/extract_postgres.py` | **Run** — 9 tables, 1,550,922 rows |
| `warehouse/load_snowflake.py --dry-run` | **Run** — 177 statements generated |
| `warehouse/raw_schema.sql` | **Not executed** — no account to run it against |
| `warehouse/load_snowflake.py` (live) | **Not executed** — same |

## Phase 5 — Airflow orchestration

Airflow 2.10 (LocalExecutor, containerised via `airflow/docker-compose.yml`) sequences
the full pipeline. The work itself runs on the host, not inside the containers — see
`airflow/host_runner.py` for the bridge and the alternatives it names.

**9. Start the host runner, then trigger the DAG:**

```
.venv/bin/python airflow/host_runner.py          # terminal 1
docker exec retail_platform_airflow_webserver \
    airflow dags trigger retail_platform_pipeline   # terminal 2
```

The host runner is a fixed registry of named jobs, each pinned to its interpreter and
working directory (`.venv` for pyspark/snowflake, `dbt/.venv` for dbt-core), exposed
over HTTP so the containerised task can start one and stream its stdout into the Airflow
log in real time. The DAG owns the order; the runner owns the runtime.

**DAG structure:**

```
[extract_postgres, spark_backfill] >> build_dim_date >> load_snowflake >> dbt_run >> dbt_test
```

`extract_postgres` and `spark_backfill` run in parallel (one reads Postgres, the other
reads the Redpanda topic — no dependency). `build_dim_date` waits on both because its
date range is derived from the order dates in Postgres and the `event_date=` partitions
Spark just wrote. Everything downstream is linear.

**Run results (2026-08-11):**

| Task | State | Duration |
|---|---|---|
| extract_postgres | **success** | ~10s |
| spark_backfill | **success** | ~4m 24s |
| build_dim_date | **success** | ~4s |
| load_snowflake | **failed** | ~1s (after 2 retries) |
| dbt_run | **upstream_failed** | — |
| dbt_test | **upstream_failed** | — |

`load_snowflake` fails at the Snowflake connector import — no account is provisioned.
The three pre-warehouse tasks complete cleanly, which is the verifiable half of the
pipeline and exactly the expected outcome for now.

## Phase 5 design notes

**Host runner, not DockerOperator or SSH.** The Spark jobs need a Java 11 JDK, the
host's local Parquet output, and two mutually incompatible virtualenvs — none of which
exist inside an Airflow image. DockerOperator moves the problem (a second image with
Spark, Java, both venvs and 363 MB of Parquet). SSHOperator is closest in spirit and is
the standard production answer, but needs macOS Remote Login and a key provisioned into
the image — more moving parts for the same shape. What replaces this in production is a
Celery/Kubernetes executor whose image ships the runtime, so the question stops existing.

**One task per script.** Finer (a task per Olist table) buys nothing — `extract_postgres`
holds one Postgres connection and either completes or doesn't, so nine tasks would always
go green or red together. Coarser (one task for the whole pipeline) is a shell script
with a web UI. Per-script is the smallest unit where "which stage broke, and can I
resume from there" stays answerable.

**Manual trigger, no schedule.** The sources are static Kaggle downloads, so a schedule
would produce identical runs on a timer. The interval this would carry in production is
the Postgres sync's, not the event stream's.

## What I'd change for production

TBD — fill in once the build is far enough along to compare against Fivetran /
managed Kafka / Databricks equivalents.
