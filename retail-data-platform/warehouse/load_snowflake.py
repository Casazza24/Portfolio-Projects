#!/usr/bin/env python3
"""Loads the local datasets into Snowflake's RAW schema via PUT + COPY INTO.

The load half of the sync. `warehouse/extract_postgres.py` produces the Olist
CSVs; Spark produces the events Parquet; `build_dim_date.py` and
`ingestion/fetch_weather.py` produce the dimensions. This puts all of it into an
internal stage and copies it into the tables declared in raw_schema.sql.

Why PUT/COPY and not Snowpipe or an external S3 stage: an internal stage costs
nothing, needs no cloud account, and is a single scriptable step. Snowpipe is
for continuous arrival, which the local batch does not have, and an external
stage would add an S3 bucket and IAM to a pipeline that gains nothing from it.
Named here because it is the first thing an interviewer asks about a COPY INTO.

Every load is a full reload -- truncate, then COPY with FORCE = TRUE. Snowflake
skips files it has already loaded within 64 days, so without FORCE a re-run of
an unchanged extract silently loads nothing and looks like it worked. At ~215 MB
total, a full reload is always correct and always cheap; incremental loading is
what you would do instead once reloading stopped being free.

Usage:
    python3 warehouse/load_snowflake.py --dry-run          # print every statement
    python3 warehouse/load_snowflake.py --init             # run raw_schema.sql first
    python3 warehouse/load_snowflake.py --targets events   # load one group
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLEAN_DIR,
    DIM_WEATHER_CSV,
    EVENTS_DIR,
    EXTRACT_DIR,
    OLIST_TABLES,
    SNOWFLAKE_STAGE,
    snowflake_connect,
)

SCHEMA_SQL = Path(__file__).resolve().parent / "raw_schema.sql"
DIM_ITEMS_DIR = CLEAN_DIR / "dim_items"

# event_date is a Hive partition column, so Spark left it out of the Parquet
# files. It is rederived here with the same expression transforms.py used
# (to_date of event_time), which is exact rather than approximate.
#
# ponytail: the alternative is parsing it back out of the path with
# REGEXP_SUBSTR(METADATA$FILENAME, 'event_date=(\\d{4}-\\d{2}-\\d{2})', ...),
# which is what you need when the partition column *cannot* be recovered from a
# retained column (a bucketed or hashed partition key). Deriving from the data
# is preferred whenever it is possible, because it cannot be broken by someone
# reorganising the stage layout.
EVENTS_COLUMNS = [
    ("event_id", "$1:event_id::varchar"),
    ("event_time", "$1:event_time::timestamp_ntz"),
    ("event_time_ms", "$1:event_time_ms::number"),
    ("event_date", "to_date($1:event_time::timestamp_ntz)"),
    ("visitor_id", "$1:visitor_id::number"),
    ("event_type", "$1:event_type::varchar"),
    ("item_id", "$1:item_id::number"),
    ("transaction_id", "$1:transaction_id::number"),
    ("category_id", "$1:category_id::number"),
    ("parent_category_id", "$1:parent_category_id::number"),
    ("root_category_id", "$1:root_category_id::number"),
    ("is_available", "$1:is_available::boolean"),
    ("has_item_metadata", "$1:has_item_metadata::boolean"),
    ("producer_id", "$1:producer_id::varchar"),
    ("ingested_at", "$1:ingested_at::timestamp_ntz"),
    ("kafka_partition", "$1:kafka_partition::number"),
    ("kafka_offset", "$1:kafka_offset::number"),
]

DIM_ITEMS_COLUMNS = [
    ("item_id", "$1:item_id::number"),
    ("category_id", "$1:category_id::number"),
    ("parent_category_id", "$1:parent_category_id::number"),
    ("root_category_id", "$1:root_category_id::number"),
    ("is_available", "$1:is_available::boolean"),
    ("category_valid_from", "$1:category_valid_from::number"),
]


def put(local: Path, stage_path: str, auto_compress: bool) -> str:
    """A PUT statement.

    The file URI is single-quoted because this project's own path contains
    spaces; an unquoted PUT target would be truncated at the first one.
    PARALLEL raises the default 4 upload threads -- the events dataset is 139
    files and the round trips, not the bytes, are what take the time.
    """
    return (
        f"put 'file://{local}' @{stage_path} "
        f"auto_compress = {'true' if auto_compress else 'false'} "
        "overwrite = true parallel = 8"
    )


def copy_csv(table: str, stage_path: str) -> str:
    return (
        f"copy into raw.{table} from @{stage_path} "
        "file_format = (format_name = raw.csv_format) "
        "on_error = abort_statement force = true"
    )


def copy_parquet(table: str, stage_path: str, columns) -> str:
    targets = ", ".join(name for name, _ in columns)
    projection = ",\n           ".join(f"{expr} as {name}" for name, expr in columns)
    return (
        f"copy into raw.{table} ({targets})\n"
        f"    from (select {projection}\n"
        f"          from @{stage_path})\n"
        "    file_format = (format_name = raw.parquet_format)\n"
        "    on_error = abort_statement force = true"
    )


def olist_statements(extract_dir: Path, tables) -> list[str]:
    statements = []
    for table in tables:
        source = extract_dir / f"{table}.csv.gz"
        if not source.exists():
            raise SystemExit(
                f"{source} missing -- run `python3 warehouse/extract_postgres.py` first"
            )
        stage_path = f"{SNOWFLAKE_STAGE}/olist/{table}"
        statements += [
            # Already gzipped by the extract; letting PUT compress again would
            # cost CPU for nothing.
            put(source, stage_path, auto_compress=False),
            f"truncate table raw.{table}",
            copy_csv(table, stage_path),
        ]
    return statements


def events_statements(events_dir: Path) -> list[str]:
    partitions = sorted(p for p in events_dir.glob("event_date=*") if any(p.glob("*.parquet")))
    if not partitions:
        raise SystemExit(
            f"no event partitions under {events_dir} -- "
            "run `python3 spark/batch_backfill.py` first"
        )

    # One PUT per partition rather than a recursive wildcard: PUT matches a
    # single directory level only, and preserving the event_date=... directory
    # in the stage keeps the layout self-describing if it ever has to be read
    # back by hand.
    statements = [
        put(part / "*.parquet", f"{SNOWFLAKE_STAGE}/events/{part.name}", auto_compress=False)
        for part in partitions
    ]
    statements += [
        "truncate table raw.events",
        copy_parquet("events", f"{SNOWFLAKE_STAGE}/events/", EVENTS_COLUMNS),
    ]
    return statements


def dimension_statements() -> list[str]:
    """PUT + COPY for the loaded dimensions.

    dim_date is not among them. The calendar is generated in-warehouse by dbt's
    `dim_date` model, which is the canonical one -- models/staging/_sources.yml
    declares no dim_date source, so a raw.dim_date loaded from
    build_dim_date.py's CSV was a second calendar nothing read. Loading it
    anyway would mean two definitions of the same dimension that can disagree.
    build_dim_date.py still runs (it derives the real date span from both
    sources, which is what dbt_project.yml's spine bounds are checked against);
    only the load of its output is retired.
    """
    missing = [p for p in (DIM_WEATHER_CSV, DIM_ITEMS_DIR) if not p.exists()]
    if missing:
        raise SystemExit("missing dimension inputs: " + ", ".join(str(p) for p in missing))

    statements = []
    for path, table in ((DIM_WEATHER_CSV, "dim_weather"),):
        stage_path = f"{SNOWFLAKE_STAGE}/{table}"
        statements += [
            put(path, stage_path, auto_compress=True),
            f"truncate table raw.{table}",
            copy_csv(table, stage_path),
        ]

    stage_path = f"{SNOWFLAKE_STAGE}/dim_items"
    statements += [
        put(DIM_ITEMS_DIR / "*.parquet", stage_path, auto_compress=False),
        "truncate table raw.dim_items",
        copy_parquet("dim_items", f"{stage_path}/", DIM_ITEMS_COLUMNS),
    ]
    return statements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="*",
        choices=["olist", "events", "dims"],
        default=["olist", "events", "dims"],
    )
    parser.add_argument("--extract-dir", type=Path, default=EXTRACT_DIR / "olist")
    parser.add_argument("--events", type=Path, default=EVENTS_DIR)
    parser.add_argument(
        "--init",
        action="store_true",
        help=f"run {SCHEMA_SQL.name} first (creates database, schemas, stage, tables)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the statements without connecting to Snowflake",
    )
    args = parser.parse_args()

    statements = []
    if "olist" in args.targets:
        statements += olist_statements(args.extract_dir, OLIST_TABLES)
    if "events" in args.targets:
        statements += events_statements(args.events)
    if "dims" in args.targets:
        statements += dimension_statements()

    if args.dry_run:
        if args.init:
            print(f"-- execute_string({SCHEMA_SQL})\n")
        for statement in statements:
            print(f"{statement};\n")
        print(f"-- {len(statements)} statements")
        return

    conn = snowflake_connect()
    try:
        if args.init:
            print(f"running {SCHEMA_SQL.name} ...")
            # execute_string splits the file itself, which beats splitting on
            # ';' and getting it wrong on a semicolon inside a string literal.
            conn.execute_string(SCHEMA_SQL.read_text(encoding="utf-8"))

        with conn.cursor() as cur:
            for i, statement in enumerate(statements, 1):
                print(f"[{i}/{len(statements)}] {statement.splitlines()[0][:100]}")
                cur.execute(statement)

            print("\nloaded row counts:")
            loaded = list(OLIST_TABLES) if "olist" in args.targets else []
            loaded += ["events"] if "events" in args.targets else []
            loaded += ["dim_weather", "dim_items"] if "dims" in args.targets else []
            for table in loaded:
                cur.execute(f"select count(*) from raw.{table}")
                print(f"  {table:<40}{cur.fetchone()[0]:>12,}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
