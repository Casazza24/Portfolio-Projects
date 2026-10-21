# Nonprofit Fundraising Analytics Pipeline

BI portfolio project: ETL pipeline + dimensional warehouse + parameterized reporting
for IRS 990 nonprofit financial data.

**Stack:** Python, PostgreSQL, Tableau Public

## What This Demonstrates

- **ETL (SSIS-equivalent):** Python pipeline that extracts IRS 990 data, stages it,
  transforms it into a star schema, and loads it into PostgreSQL
- **Reporting (SSRS-equivalent):** Parameterized Tableau dashboards with drill-down
  by org type, geography, and fiscal year
- **SQL:** Complex queries for donor retention cohorts, YoY giving trends, and
  segment analysis

## Setup

```bash
pip install -r requirements.txt
createdb nonprofit_analytics
python etl/run_pipeline.py
```

## Project Structure

```
etl/           - Extract, transform, load scripts
sql/           - Schema DDL + analytical queries
data/raw/      - Raw 990 data (not committed, see etl/extract.py)
docs/          - Data dictionary, schema diagram
```
