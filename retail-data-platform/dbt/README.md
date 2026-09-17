# dbt — warehouse modeling layer (Phase 4)

Transforms the Snowflake `raw` schema (Olist OLTP tables, cleaned RetailRocket
events, Open-Meteo weather) into a tested star schema the Power BI report reads.

**Status: written, parsed, and not yet executed.** There is no Snowflake
account for this project yet, so `dbt run` / `dbt test` / `dbt docs generate`
have never been run against a warehouse. What *has* been verified locally is in
[Verified so far](#verified-so-far) below. Treat every row count and every
threshold quoted in a model comment as a Phase 3 measurement carried forward,
not as something this layer has confirmed.

## Layout

```
models/staging/   one model per source table -- rename, type, light cleanup only
models/marts/     the star schema (3 facts, 6 dimensions)
tests/            singular tests: the four checks generic tests cannot express
```

| Model | Grain |
|---|---|
| `fct_orders` | one order line — `(order_id, order_item_id)` |
| `fct_order_payments` | one payment instrument — `(order_id, payment_sequential)` |
| `fct_events` | one clickstream event — `event_id` |
| `dim_customers` | one `customer_id` (Olist mints one **per order**, not per person) |
| `dim_products` | one Olist `product_id` |
| `dim_sellers` | one `seller_id` |
| `dim_items` | one RetailRocket `item_id`, plus the `-1` unknown member |
| `dim_date` | one calendar day, generated in-warehouse |
| `dim_weather` | one `(state, date)` |

Two stars — Olist and RetailRocket describe two different retailers, so they
share `dim_date` and nothing else. `models/marts/_marts.yml` explains why
inventing a link between them would be worse than leaving them separate.

## Setup, once a Snowflake account exists

```bash
python3 -m venv dbt/.venv
dbt/.venv/bin/pip install -r dbt/requirements-dbt.txt

cp dbt/profiles.yml.example dbt/profiles.yml    # gitignored
cp dbt/.env.example dbt/.env                    # gitignored -- fill in
set -a && source dbt/.env && set +a
export DBT_PROFILES_DIR="$PWD/dbt"

cd dbt
.venv/bin/dbt debug          # confirms the connection before anything else
.venv/bin/dbt build          # run + test in dependency order
.venv/bin/dbt docs generate && .venv/bin/dbt docs serve
```

`dbt build` rather than `dbt run && dbt test`: build interleaves them, so a
model whose upstream test failed is never built on top of bad data.

Nothing in the profile is a literal credential — every value comes from the
environment, so `profiles.yml` stays safe even if the gitignore is wrong.

## Things to expect on the first real run

- **Table names.** The `identifier:` on every entry in
  `models/staging/_sources.yml` is this project's assumption about what the
  Snowflake loader calls things (`RAW.CUSTOMERS`, `RAW.EVENTS`,
  `RAW.DIM_ITEMS`, `RAW.DIM_WEATHER`). If the loader lands them under different
  names, `_sources.yml` is the only file that changes.
- **`assert_order_payments_reconcile`** has a first-guess `error_if: >500`.
  Re-set it from the number the first run actually produces.
- **`dim_weather` relationship** runs at `severity: warn`, so an order date or
  customer state the weather pull never covered warns rather than failing the
  build. Everything else runs strict.

## Design calls worth knowing about

**The unknown member is `-1`, and it is a real row.** 9.27% of events reference
an item RetailRocket's metadata feed never described. Rather than dropping
those events or inventing a category for them, Phase 3 resolved them to a
sentinel that exists as an actual row in `dim_items`. `fct_events` carries both
`item_key` (the sentinel-resolved key that joins) and `item_id` (the real id
that does not). That is what lets the `relationships` test run strict instead
of being weakened with a null tolerance it could then never catch a real break
with. Full argument in `docs/data_quality.md` §3.4.

**The 132 orphan categories keep their nulls.** Some items reference a category
id that `category_tree.csv` never defines, so they have a category but no
parent or root. They are *not* folded into the `-1` sentinel: "the item was
never described" and "the category tree is incomplete" are different defects
with different upstream owners, and merging them makes a fix to either one
invisible. Nulls are acceptable here specifically because parent/root are
descriptive attributes of a dimension row, not keys leaving a fact — nothing
joins on them, so no test is weakened. The count is pinned at 132 by
`tests/assert_orphan_categories_bounded.sql`, because tolerating a known gap is
only honest if the size of it is asserted.

**No dbt packages.** `dbt_utils` would have been used for exactly two things —
`date_spine` and a surrogate-key hash. Snowflake's `GENERATOR` does the first
in three lines, and the keys here are two short columns where a readable
`'SP-2017-03-14'` beats an md5 nobody can reverse in a failing test. A package
that saves two calls still brings its own pinning, a `dbt deps` step and an
upgrade surface. If a third use appeared, adding it would be the right call.

**Staging is views, marts are tables.** Power BI issues many small interactive
queries; paying the transform cost once per run beats paying it per slicer
click. No incremental models — the largest mart is 2.7M rows and rebuilds in
seconds, so incrementality would add state to manage for no useful saving.
That changes at roughly two orders of magnitude more data.

## The four singular tests

Generic tests cover keys, enums and referential integrity. These cover the
things that would go wrong *without* breaking any of those:

| Test | Catches |
|---|---|
| `assert_unknown_item_rate_within_bounds` | A broken upstream metadata feed. Every event still resolves — to `-1` — so no key breaks and no generic test fires, while revenue quietly migrates into an Unknown bucket. The rate is the only thing that moves. |
| `assert_orphan_categories_bounded` | The category tree degrading past its known 132-category gap. |
| `assert_fct_orders_line_count_matches_source` | Fan-out **and** row loss in `fct_orders`. Some Olist orders carry two reviews; joining reviews in naively duplicates those orders' lines and inflates revenue. A `unique` test catches that, but not the opposite failure — an inner join quietly dropping every line whose order has no review. |
| `assert_order_payments_reconcile` | The two facts disagreeing about what an order cost. They are built from independent sources, so this is the only check either one cannot perform on itself. Thresholded, because a residue of genuinely mismatched orders (refunds, vouchers) exists in the source. |

## Verified so far

Run against dbt-core 1.10.22 / dbt-snowflake 1.10.2 in `dbt/.venv`, with dummy
credentials and no warehouse:

```
dbt parse   -> OK. 21 models, 155 data tests, 12 sources, 492 macros.
dbt list    -> all 21 models and all tests resolve.
dbt compile -> parses and builds the graph, then fails at the connection
               (Snowflake 08001) as expected. No SQL was compiled server-side.
```

Not verified, and not claimed: that the SQL is valid *Snowflake* SQL, that any
model produces the expected rows, or that any test passes.
