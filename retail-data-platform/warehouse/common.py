"""Shared paths, credentials, and connections for the Snowflake load.

Mirrors spark/common.py: one place that decides where things live and how
connections are made, so the extract half and the load half cannot drift on
either.

Credentials are read from environment variables, falling back to a `.env` file
at the project root (see .env.example). Nothing here is ever hardcoded --
`.env` is gitignored, and every value has either a safe local-Docker default
(Postgres) or no default at all (Snowflake).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"

# Extract output. Separate from data/clean/ because data/clean/ is Spark's
# curated output and this is a staging area for a load -- mixing them makes it
# ambiguous which files a rebuild is allowed to delete.
EXTRACT_DIR = DATA_DIR / "warehouse"

EVENTS_DIR = CLEAN_DIR / "events"
DIM_WEATHER_CSV = CLEAN_DIR / "dim_weather.csv"
DIM_DATE_CSV = CLEAN_DIR / "dim_date.csv"

# Load order matters only for readability here -- Snowflake does not enforce
# the foreign keys declared in raw_schema.sql -- but parents first keeps a
# partially-failed load easier to reason about.
OLIST_TABLES = (
    "product_category_name_translation",
    "customers",
    "sellers",
    "products",
    "geolocation",
    "orders",
    "order_items",
    "order_payments",
    "order_reviews",
)

SNOWFLAKE_STAGE = "raw.retail_stage"


def load_dotenv(path: Path | None = None) -> None:
    """Loads KEY=VALUE lines from .env into os.environ without overriding it.

    Deliberately not python-dotenv: this is a dozen lines against a file format
    we control, and it keeps the warehouse scripts runnable with only the two
    database drivers installed.
    """
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def pg_connect():
    """Connects to the local Olist Postgres.

    Defaults match docker-compose.yml, so a fresh clone needs no .env at all to
    run the extract half.
    """
    import psycopg2

    load_dotenv()
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "retail"),
        user=os.environ.get("PGUSER", "retail"),
        password=os.environ.get("PGPASSWORD", "retail"),
    )


def snowflake_connect():
    """Connects to Snowflake. No defaults -- every value must be supplied."""
    import snowflake.connector

    load_dotenv()
    required = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            f"missing Snowflake credentials: {', '.join(missing)} -- "
            "copy .env.example to .env and fill it in"
        )

    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        role=os.environ.get("SNOWFLAKE_ROLE", "SYSADMIN"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "RETAIL_WH"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "RETAIL_PLATFORM"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "RAW"),
    )
