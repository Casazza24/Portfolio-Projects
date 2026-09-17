# Project scope: retail data platform (OLTP + streaming events → Snowflake/dbt → BI dashboard)

**One-liner for a resume/portfolio:** *Built an end-to-end ELT pipeline combining relational order data and simulated real-time event streams, with distributed processing, warehouse modeling, automated data quality testing, and a published BI dashboard.*

This scopes the exact architecture we talked through (Postgres/Fivetran-style source + Kafka-style event stream → Spark → Snowflake/dbt → BI), but sized for a real multi-month build instead of a 24-hour sprint, with a few choices for you to lock in as you go. It's designed to slot into the existing three categories of your portfolio — it touches the streaming/event-driven side, the warehouse/BI side, and (lightly) the orchestration/infra side, so it's a good "capstone" piece that ties the other projects together in interviews.

---

## 1. Data sources

You need two source types to mirror the architecture: a relational (OLTP-style) dataset and an event-stream dataset. Both below are real, free, public datasets — no scraping or API keys needed to get started, which keeps phase 1 fast.

| Source | What it is | Why it fits | Role in the pipeline |
|---|---|---|---|
| **Olist Brazilian E-Commerce dataset** (Kaggle) | ~100k real e-commerce orders (2016–2018) split across normalized tables: orders, customers, order_items, products, sellers, payments, reviews, geolocation | Already relational and reasonably clean — perfect stand-in for "the Postgres database Fivetran would sync" | Load into a local Postgres instance → this becomes your simulated OLTP source |
| **RetailRocket E-commerce events dataset** (Kaggle) | Real anonymized clickstream: view / addtocart / transaction events, with timestamps, plus item/category metadata | Messy, high-volume, event-shaped — exactly the kind of data Kafka + Spark were built for | Replay row-by-row (ordered by timestamp) through a Python producer into Kafka (or a queue/file-drop if you decide to skip Kafka itself) — this becomes your simulated POS/clickstream source |
| **Open-Meteo API** *(optional enrichment)* | Free historical weather API, no key required | Adds a believable "why did sales spike that week" join dimension and gives you an excuse to demo an external API ingestion step | Pulled in during the warehouse modeling phase as a small dimension table |

If at any point the Olist/RetailRocket data feels too clean or too small for what you want to demonstrate, a synthetic generator (Python + `Faker`) is a fine fallback for either side — it just costs you build time you'd otherwise save by using real data.

---

## 2. Target architecture (recap, with your concrete tool choices)

```
Olist data → Postgres (OLTP source)
RetailRocket events → replayed → Redpanda topic (event source)
                                      ↓
                      Spark (clean, dedupe, reshape events)
                                      ↓
        Postgres raw sync  +  cleaned events  →  Snowflake (raw schema)
                                      ↓
                    dbt (staging → marts, + tests)
                                      ↓
                Snowflake (gold: fact/dim tables)
                                      ↓
              Power BI dashboard
                                      ↑
            Airflow orchestrates ingestion → Spark → dbt → refresh
```

Tool lock-ins for this build (decided 2026-07-27): **Redpanda** instead of Kafka (Kafka-API compatible, single binary via docker-compose, much lighter to run locally — same "Kafka-compatible streaming" resume line without the local resource overhead), and **Power BI** instead of Tableau Public (fits Matt's target Microsoft-stack roles better; public sharing is more limited than Tableau Public, so the dashboard will be shown via a screen-recorded walkthrough rather than a cold public link).

Everything above the dashboard is exactly the pattern from the original walkthrough — replacing "Fivetran" with a manual/scripted Postgres sync (free), and "Databricks" with local PySpark (free), since paying for managed infra isn't worth it for a portfolio piece. Snowflake's free trial credits and dbt-core (free, open source) cover the rest at no cost.

---

## 3. Phased build plan (~10–12 weeks, flexible)

Each phase is broken into small, mobile-friendly chunks where possible — planning, schema design, and SQL writing travel well to a phone/laptop-light workflow; the heavier coding sessions (Spark, Airflow) are flagged as needing real laptop time.

### Phase 1 — Foundations (week 1–2)
- Pick and lock your project name and scope (done — see below)
- Download Olist dataset, stand up a local Postgres instance (Docker), load the raw CSVs in
- Download RetailRocket dataset, write a small Python script that replays events in timestamp order at an adjustable speed (this *is* your "Redpanda producer" logic even before Redpanda is wired up)
- Deliverable: a GitHub repo with a clear README skeleton and both datasets loaded/staged locally

### Phase 2 — Streaming layer (week 3–4, needs real laptop time)
- Stand up Redpanda via `docker-compose` (Kafka-API compatible, lighter local footprint)
- Point your replay script at it as a producer; write a simple consumer to confirm events are flowing
- Deliverable: a working producer → Redpanda → consumer loop you can demo with a screen recording

### Phase 3 — Processing layer (week 5–6, laptop time)
- Write the PySpark job: dedupe events, handle out-of-order timestamps, join against the item/category metadata, output cleaned Parquet (or write straight to a Snowflake stage)
- This is the phase to deliberately introduce/handle realistic messiness — duplicate events, malformed rows, late arrivals — so the cleaning logic actually has something to do and is worth talking about in an interview
- Deliverable: cleaned event dataset, plus a short write-up of the specific data quality issues you found and fixed

### Phase 4 — Warehouse + modeling (week 7–8)
- Spin up a Snowflake trial, create raw/staging/marts schemas
- Load: Postgres → Snowflake raw (a simple scripted sync is fine), cleaned events → Snowflake raw, Open-Meteo weather → a small dimension table
- Build dbt-core models: staging models per source, then marts (e.g. `fct_orders`, `fct_events`, `dim_customers`, `dim_products`, `dim_date`)
- Add dbt tests: `not_null`, `unique`, `relationships` between fact and dimension keys, plus at least one custom test
- Deliverable: a working `dbt run` + `dbt test` pipeline, and the auto-generated dbt docs lineage graph (great portfolio screenshot)

### Phase 5 — Orchestration (week 9, optional but recommended)
- Minimal Airflow setup via `docker-compose`, one DAG that sequences: Postgres sync → Spark job → dbt run → dbt test → (trigger dashboard refresh, if your BI tool supports an API-triggered refresh)
- You don't need Kubernetes/Celery executors for this — `LocalExecutor` is fine for a portfolio project
- Deliverable: a DAG screenshot/recording showing a full successful run, plus one showing a deliberately broken run and how Airflow surfaces the failure

### Phase 6 — Dashboard + polish (week 10–12)
- See section 4 below for the dashboard build itself
- Write the README properly: architecture diagram, ERD, setup instructions, what you'd change for a "real" production version (this is where you mention Fivetran/Databricks/managed Kafka as the production equivalents of what you simplified)
- Record a 2–3 minute demo video walking through the dashboard and the pipeline
- Deliverable: polished repo + dashboard link/recording + demo video, ready to link from your resume/LinkedIn

---

## 4. Dashboard / visualization step

**Tool choice:** Power BI (locked in 2026-07-27) — fits a Microsoft-stack-leaning target role set better than Tableau Public. Sharing publicly is more limited than Tableau Public's free public link, so the portfolio deliverable will be a screen-recorded walkthrough of the Power BI report rather than a cold public link.

**Connection:** Power BI connects to Snowflake natively — point the dashboard at your gold-layer marts, not the raw tables. This is also a good talking point in interviews: *"the dashboard only ever queries modeled, tested gold tables, never raw data."*

**Suggested pages/views:**
1. **Revenue overview** — daily/weekly revenue trend, broken out by product category, with the weather dimension overlaid as a secondary line (does revenue dip on certain weather conditions?)
2. **Funnel view** — view → add-to-cart → purchase conversion from the RetailRocket event data, by category
3. **Customer view** — basic cohort/repeat-purchase view from the Olist order data
4. **Pipeline health view** — this one's the differentiator: a small dashboard page showing dbt test pass/fail counts over time, row counts loaded per run, last successful pipeline run timestamp. Most portfolio projects only show the *business* dashboard; showing the *pipeline's own* health is a strong signal you're thinking like a data engineer, not just an analyst.

---

## 5. Stretch goals (only if the core build finishes early)
- A simple demand-forecasting or anomaly-detection model on top of the gold tables
- Swap the local PySpark step for a real Databricks community-edition cluster, to get the genuine Databricks line on your resume
- Add a second BI tool (Tableau Public) reading the same gold tables, just to demonstrate tool flexibility and get the free public-link version too

---

## 6. Final deliverables checklist
- [ ] GitHub repo, clean structure (`/ingestion`, `/spark`, `/dbt`, `/airflow`, `/docs`)
- [ ] README with architecture diagram + ERD + setup steps + "what I'd change for production" section
- [ ] dbt docs site (can be hosted free via GitHub Pages)
- [ ] Power BI report + screen-recorded walkthrough (or Tableau Public link if the stretch goal happens)
- [ ] 2–3 minute demo video
- [ ] One paragraph case-study write-up you can paste directly into applications or talk through in an interview

This is intentionally a lot — treat the phase breakdown as the actual plan and don't try to do it faster than the weeks allotted. The pacing above assumes evenings/weekends around your schedule, not full-time hours.
