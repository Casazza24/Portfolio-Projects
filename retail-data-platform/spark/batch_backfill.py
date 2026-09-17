#!/usr/bin/env python3
"""Batch reprocessing of the event history.

The streaming job is the live path; this is the one you run when the cleaning
logic changes and the existing clean dataset has to be rebuilt from source, or
when the stream was down and a window of the topic needs reprocessing.

It reads the Kafka topic in *batch* mode (bounded by explicit offsets) and
runs the exact same transforms as the streaming job -- see spark/transforms.py.
The one intentional divergence is dedupe: batch sees the whole range at once
and can dedupe globally, where streaming is bounded by a watermark.

Usage:
    # rebuild everything currently in the topic
    python3 spark/batch_backfill.py

    # reprocess a bounded offset range
    python3 spark/batch_backfill.py --starting-offsets earliest --ending-offsets latest
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_BROKERS,
    DEFAULT_TOPIC,
    DIM_ITEMS_OUT,
    EVENTS_OUT,
    EVENTS_QUARANTINE,
    build_spark,
)
from transforms import clean_events  # noqa: E402

# Snowflake's COPY INTO guidance is 100-250 MB compressed per file: smaller and
# per-file overhead dominates the load, larger and a single file stops splitting
# across load threads.
DEFAULT_TARGET_FILE_MB = 200

# Only reached on a cold first run, when there is no previous output to measure.
FALLBACK_BYTES_PER_ROW = 70


def bytes_per_row(spark, path: Path) -> float:
    """Compressed bytes per row, measured from the previous run's output.

    Sizing a Parquet file needs to know how well *this* schema actually
    compresses, and the last run already answered that -- so measure it rather
    than hardcode a row count that silently goes wrong when the schema changes.
    Self-calibrating after the first run; constant before it.
    """
    if not path.exists():
        return FALLBACK_BYTES_PER_ROW
    written = sum(f.stat().st_size for f in path.rglob("*.parquet"))
    if not written:
        return FALLBACK_BYTES_PER_ROW
    rows = spark.read.parquet(str(path)).count()
    return written / rows if rows else FALLBACK_BYTES_PER_ROW


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brokers", default=DEFAULT_BROKERS)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--starting-offsets", default="earliest")
    parser.add_argument("--ending-offsets", default="latest")
    parser.add_argument("--output", type=Path, default=EVENTS_OUT)
    parser.add_argument("--quarantine", type=Path, default=EVENTS_QUARANTINE)
    parser.add_argument("--dim-items", type=Path, default=DIM_ITEMS_OUT)
    parser.add_argument(
        "--mode",
        choices=["overwrite", "append"],
        default="overwrite",
        help="overwrite rebuilds the dataset; append adds a reprocessed slice",
    )
    parser.add_argument(
        "--target-file-mb",
        type=int,
        default=DEFAULT_TARGET_FILE_MB,
        help="target compressed size per output Parquet file",
    )
    args = parser.parse_args()

    spark = build_spark("batch_backfill", with_kafka=True)
    started = time.time()

    if not args.dim_items.exists():
        raise SystemExit(
            f"item dimension not found at {args.dim_items} -- "
            "run `python3 spark/build_item_metadata.py` first"
        )
    dim_items = spark.read.parquet(str(args.dim_items))

    raw = (
        spark.read.format("kafka")
        .option("kafka.bootstrap.servers", args.brokers)
        .option("subscribe", args.topic)
        .option("startingOffsets", args.starting_offsets)
        .option("endingOffsets", args.ending_offsets)
        .load()
    )

    clean, quarantined = clean_events(raw, dim_items, streaming=False)

    # Both branches derive from the same Kafka scan; without caching, writing
    # them sequentially re-reads and re-parses the whole topic twice.
    clean = clean.cache()
    quarantined = quarantined.cache()

    # Measured before the write, because an overwrite destroys what it measures.
    rows_per_file = max(1, int(args.target_file_mb * 1024**2 / bytes_per_row(spark, args.output)))

    # The repartition is what compacts: hashing on event_date lands every row of
    # a date in one task, so each date writes one file instead of one per
    # upstream task (previously ~8 per date, 1,112 files for 184 MB).
    # maxRecordsPerFile is the safety valve for a date large enough to blow past
    # the target on its own.
    #
    # ponytail: the 100-250 MB target is unreachable *while* partitioning daily
    # -- 184 MB over ~139 days is ~1.3 MB/day, so one file per date is the
    # ceiling. Monthly partitioning would land ~40 MB files and no partitioning
    # would hit the band exactly. Daily is kept because Snowflake prunes on its
    # own micro-partitions regardless, and event_date is the column every mart
    # filters on; the file count is the part of the problem actually worth
    # fixing at this volume.
    (
        clean.repartition("event_date")
        .write.mode(args.mode)
        .option("maxRecordsPerFile", rows_per_file)
        .partitionBy("event_date")
        .parquet(str(args.output))
    )
    quarantined.write.mode(args.mode).parquet(str(args.quarantine))

    total = raw.count()
    clean_count = clean.count()
    quarantine_count = quarantined.count()
    duplicates_removed = total - quarantine_count - clean_count
    unmatched = clean.filter(~F.col("has_item_metadata")).count()

    print(f"\n=== batch backfill ({time.time() - started:.0f}s) ===")
    print(f"messages read       : {total:,}")
    print(f"clean events        : {clean_count:,}")
    print(f"quarantined         : {quarantine_count:,} ({quarantine_count / total:.2%})")
    print(f"duplicates removed  : {duplicates_removed:,} ({duplicates_removed / total:.2%})")
    print(f"no item metadata    : {unmatched:,} ({unmatched / clean_count:.2%} of clean)")
    files = list(args.output.rglob("*.parquet"))
    written_bytes = sum(f.stat().st_size for f in files)

    print(f"\nclean      -> {args.output}")
    print(f"quarantine -> {args.quarantine}")
    print(
        f"output files        : {len(files):,} "
        f"({written_bytes / 1024**2:.0f} MB, "
        f"{written_bytes / len(files) / 1024**2:.1f} MB avg, "
        f"{written_bytes / clean_count:.0f} B/row)"
    )

    # How the metadata gap distributes across event types decides how much it
    # actually matters: concentrated in `view` it only dents top-of-funnel,
    # but concentrated in `transaction` it distorts revenue by category.
    print("\nitem metadata coverage by event type:")
    clean.groupBy("event_type").agg(
        F.count("*").alias("events"),
        F.sum(F.col("has_item_metadata").cast("int")).alias("with_metadata"),
        F.round(
            100 * (1 - F.avg(F.col("has_item_metadata").cast("int"))), 2
        ).alias("pct_unknown"),
    ).orderBy(F.col("events").desc()).show(truncate=False)

    print("\nquarantine reasons:")
    quarantined.groupBy("dq_reason").count().orderBy(F.col("count").desc()).show(
        truncate=False
    )

    print("detection vs injected fault (off-diagonal = real source defects, not injected):")
    quarantined.groupBy("injected_fault", "dq_reason").count().orderBy(
        F.col("count").desc()
    ).show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
