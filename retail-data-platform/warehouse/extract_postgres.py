#!/usr/bin/env python3
"""Extracts the 9 Olist tables from Postgres to gzipped CSV.

The Postgres half of the Postgres -> Snowflake sync, deliberately split out as
its own step: it needs no Snowflake account, so the source side can be run,
diffed and row-count-checked long before the warehouse exists -- and when a
load does go wrong later, "did the extract produce the right rows?" is
answerable without touching Snowflake at all.

Extraction is `COPY ... TO STDOUT WITH (FORMAT csv)`, which is Postgres's own
serializer rather than a Python row loop: it streams server-side, and it gets
the hard cases right without help -- order_reviews' free-text comments contain
embedded newlines, commas and quotes.

The NULL-vs-empty-string distinction survives the round trip: Postgres writes
NULL as a bare empty field and an empty string as `""`, and the Snowflake file
format in raw_schema.sql reads them back the same way.

Usage:
    python3 warehouse/extract_postgres.py
    python3 warehouse/extract_postgres.py --tables orders order_items
"""
from __future__ import annotations

import argparse
import gzip
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXTRACT_DIR, OLIST_TABLES, pg_connect  # noqa: E402


def extract_table(conn, table: str, out_dir: Path) -> tuple[int, int]:
    """Writes olist.<table> to <out_dir>/<table>.csv.gz. Returns (rows, bytes)."""
    path = out_dir / f"{table}.csv.gz"
    sql = f"COPY olist.{table} TO STDOUT WITH (FORMAT csv, HEADER true)"

    with conn.cursor() as cur, gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        cur.copy_expert(sql, f)

    # Counted separately rather than trusting cur.rowcount, which COPY TO
    # STDOUT does not populate reliably across psycopg2 versions.
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM olist.{table}")
        rows = cur.fetchone()[0]

    return rows, path.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="*", default=list(OLIST_TABLES))
    parser.add_argument("--output", type=Path, default=EXTRACT_DIR / "olist")
    args = parser.parse_args()

    unknown = [t for t in args.tables if t not in OLIST_TABLES]
    if unknown:
        raise SystemExit(f"not an Olist table: {', '.join(unknown)}")

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    conn = pg_connect()

    total_rows = total_bytes = 0
    print(f"{'table':<40}{'rows':>12}{'size':>12}")
    for table in args.tables:
        rows, size = extract_table(conn, table, args.output)
        total_rows += rows
        total_bytes += size
        print(f"{table:<40}{rows:>12,}{size / 1024**2:>10.1f} MB")

    conn.close()
    print(f"{'':-<64}")
    print(f"{'total':<40}{total_rows:>12,}{total_bytes / 1024**2:>10.1f} MB")
    print(f"\n-> {args.output}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
