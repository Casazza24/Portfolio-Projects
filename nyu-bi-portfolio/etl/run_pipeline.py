"""
Nonprofit Analytics ETL Pipeline
Demonstrates: staged ingestion, cleaning, dimensional modeling, load.
Functional equivalent of an SSIS package.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from config import DB_URL, RAW_DIR


def get_engine():
    return create_engine(DB_URL)


# --- EXTRACT ---
def extract():
    """
    Load raw 990 data from CSV files in data/raw/.
    Download the GivingTuesday 990 Basic Fields data mart CSV from:
    https://nonprofitecosystem.givingtuesday.org/datamarts/

    ponytail: no auto-download, user drops CSV in data/raw/. Add requests fetch if needed.
    """
    csvs = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files in {RAW_DIR}/. Download the 990 Basic Fields "
            "data mart from https://nonprofitecosystem.givingtuesday.org/datamarts/"
        )
    frames = [pd.read_csv(os.path.join(RAW_DIR, f), low_memory=False) for f in csvs]
    raw = pd.concat(frames, ignore_index=True)
    print(f"Extracted {len(raw):,} rows from {len(csvs)} file(s)")
    return raw


# --- TRANSFORM ---
def transform(raw: pd.DataFrame) -> dict:
    """Clean and shape raw 990 data into dimensional tables."""

    # Standardize column names (990 data has inconsistent casing)
    raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "_")

    # --- dim_organization ---
    org_cols = {
        "ein": "ein",
        "organization_name": "name",
        "city": "city",
        "state": "state",
        "zip": "zip",
        "ntee_code": "ntee_code",
        "organization_type": "org_type",
        "ruling_date": "ruling_date",
    }
    # ponytail: best-effort column matching, real 990 headers vary by vintage
    available = {k: v for k, v in org_cols.items() if k in raw.columns}
    dim_org = raw[list(available.keys())].rename(columns=available).drop_duplicates("ein")
    dim_org["ein"] = dim_org["ein"].astype(str).str.zfill(9)

    # --- dim_fiscal_year ---
    if "tax_period" in raw.columns:
        periods = raw["tax_period"].dropna().astype(str).unique()
        dim_fy = pd.DataFrame({"tax_period": periods})
        dim_fy["fiscal_year"] = dim_fy["tax_period"].str[:4].astype(int)
        dim_fy["fiscal_month"] = dim_fy["tax_period"].str[4:6].astype(int)
    else:
        raise KeyError("Expected 'tax_period' column in 990 data")

    # --- fact_filing ---
    fact_cols = {
        "ein": "ein",
        "tax_period": "tax_period",
        "total_revenue": "total_revenue",
        "total_expenses": "total_expenses",
        "net_assets": "net_assets",
        "contributions_and_grants": "contributions",
        "program_service_revenue": "program_revenue",
        "investment_income": "investment_income",
        "total_compensation": "total_compensation",
        "number_of_employees": "num_employees",
        "number_of_volunteers": "num_volunteers",
    }
    available_fact = {k: v for k, v in fact_cols.items() if k in raw.columns}
    fact = raw[list(available_fact.keys())].rename(columns=available_fact)
    fact["ein"] = fact["ein"].astype(str).str.zfill(9)
    fact["tax_period"] = fact["tax_period"].astype(str)

    # Coerce numeric columns
    num_cols = [c for c in fact.columns if c not in ("ein", "tax_period")]
    for c in num_cols:
        fact[c] = pd.to_numeric(fact[c], errors="coerce")

    print(f"Transformed: {len(dim_org):,} orgs, {len(dim_fy)} fiscal periods, {len(fact):,} filings")
    return {"dim_organization": dim_org, "dim_fiscal_year": dim_fy, "fact_filing": fact}


# --- LOAD ---
def load(tables: dict):
    """Load dimensional tables into PostgreSQL."""
    engine = get_engine()

    # Create schema
    with open(os.path.join(os.path.dirname(__file__), "..", "sql", "schema.sql")) as f:
        with engine.begin() as conn:
            conn.execute(text(f.read()))

    # Load dimensions first, then resolve keys for fact table
    tables["dim_organization"].to_sql(
        "dim_organization", engine, if_exists="append", index=False,
        method="multi", chunksize=5000,
    )

    tables["dim_fiscal_year"].to_sql(
        "dim_fiscal_year", engine, if_exists="append", index=False,
    )

    # Resolve surrogate keys for fact table
    with engine.connect() as conn:
        org_keys = pd.read_sql("SELECT org_key, ein FROM dim_organization", conn)
        fy_keys = pd.read_sql("SELECT fy_key, tax_period FROM dim_fiscal_year", conn)

    fact = tables["fact_filing"]
    fact = fact.merge(org_keys, on="ein", how="inner")
    fact = fact.merge(fy_keys, on="tax_period", how="inner")
    fact = fact.drop(columns=["ein", "tax_period"])

    fact.to_sql(
        "fact_filing", engine, if_exists="append", index=False,
        method="multi", chunksize=5000,
    )

    print(f"Loaded {len(fact):,} filings into PostgreSQL")


# --- RUN ---
if __name__ == "__main__":
    raw = extract()
    tables = transform(raw)
    load(tables)
    print("Pipeline complete.")
