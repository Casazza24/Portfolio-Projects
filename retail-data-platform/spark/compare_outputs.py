#!/usr/bin/env python3
"""Reconciles the streaming output against the batch backfill output.

Two code paths over one dataset is only defensible if you can show they agree.
This reads both Parquet datasets and reports where they diverge -- row counts,
event_id set difference, and per-column disagreement on the overlap.

Some divergence is expected and correct: batch dedupes globally while
streaming dedupes within a watermark, so streaming may retain a small number
of duplicate event_ids that batch collapses. Anything beyond that is a bug in
one of the two paths.

Usage:
    python3 spark/compare_outputs.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN_DIR, EVENTS_OUT, build_spark  # noqa: E402

COMPARED_COLUMNS = [
    "event_time_ms",
    "visitor_id",
    "event_type",
    "item_id",
    "transaction_id",
    "category_id",
    "root_category_id",
    "has_item_metadata",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=EVENTS_OUT)
    parser.add_argument("--streaming", type=Path, default=CLEAN_DIR / "events_streaming")
    args = parser.parse_args()

    for path in (args.batch, args.streaming):
        if not path.exists():
            raise SystemExit(f"missing dataset: {path}")

    spark = build_spark("compare_outputs")
    batch = spark.read.parquet(str(args.batch))
    stream = spark.read.parquet(str(args.streaming))

    batch_count = batch.count()
    stream_count = stream.count()
    batch_ids = batch.select("event_id").distinct()
    stream_ids = stream.select("event_id").distinct()
    batch_unique = batch_ids.count()
    stream_unique = stream_ids.count()

    only_batch = batch_ids.subtract(stream_ids).count()
    only_stream = stream_ids.subtract(batch_ids).count()

    print("\n=== row counts ===")
    print(f"batch      : {batch_count:,} rows, {batch_unique:,} distinct event_ids")
    print(f"streaming  : {stream_count:,} rows, {stream_unique:,} distinct event_ids")
    print(f"duplicates left in streaming : {stream_count - stream_unique:,}")
    print(f"duplicates left in batch     : {batch_count - batch_unique:,}")

    print("\n=== event_id set difference ===")
    print(f"in batch only     : {only_batch:,}")
    print(f"in streaming only : {only_stream:,}")

    # Compare field-by-field on the shared event_ids. Dedupe each side first
    # so a duplicate row cannot fan the join out and inflate the mismatch
    # count into meaninglessness.
    b = batch.dropDuplicates(["event_id"]).select("event_id", *COMPARED_COLUMNS)
    s = stream.dropDuplicates(["event_id"]).select(
        "event_id", *[F.col(c).alias(f"s_{c}") for c in COMPARED_COLUMNS]
    )
    joined = b.join(s, on="event_id", how="inner").cache()
    overlap = joined.count()

    print(f"\n=== column agreement over {overlap:,} shared event_ids ===")
    mismatches = 0
    for col in COMPARED_COLUMNS:
        differs = joined.filter(~F.col(col).eqNullSafe(F.col(f"s_{col}"))).count()
        mismatches += differs
        flag = "OK" if differs == 0 else "MISMATCH"
        print(f"  {col:<20} {differs:>10,}  {flag}")

    print()
    if mismatches == 0 and only_batch == 0:
        print("PASS: streaming and batch agree on every shared row, and batch "
              "contains no event the stream missed.")
    else:
        print(f"FAIL: {mismatches:,} column mismatches, {only_batch:,} events "
              "present in batch but absent from streaming.")

    spark.stop()
    sys.exit(0 if (mismatches == 0 and only_batch == 0) else 1)


if __name__ == "__main__":
    main()
