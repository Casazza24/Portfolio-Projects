#!/usr/bin/env python3
"""Structured Streaming job: Redpanda -> cleaned Parquet.

This is the live path. It runs the same transforms as the batch backfill
(spark/transforms.py); the differences are that dedupe is watermark-bounded
rather than global, and progress is checkpointed so a restart resumes at the
last committed offset instead of reprocessing the topic.

Two independent queries read the topic rather than one query fanning out to
two sinks. That costs a second broker read, and buys separate checkpoints and
separate failure domains -- a poison payload that wedges the quarantine writer
cannot stall the clean path, and either can be restarted or replayed alone.

Usage:
    # run until the topic is drained, then stop (what the demo recording uses)
    python3 spark/streaming_events.py --stop-when-idle

    # run continuously
    python3 spark/streaming_events.py

    # start over from the beginning of the topic
    python3 spark/streaming_events.py --reset --starting-offsets earliest
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHECKPOINT_DIR,
    DEFAULT_BROKERS,
    DEFAULT_TOPIC,
    DIM_ITEMS_OUT,
    build_spark,
)
from transforms import clean_events  # noqa: E402

# Sized from the producer's observed behaviour, not guessed: the replay holds
# late events back by up to 5,000 stream positions, which during the sparse
# overnight stretches of the RetailRocket data spans a little over a day of
# event time. 3 days leaves headroom without making dedupe state unbounded.
DEFAULT_WATERMARK = "3 days"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brokers", default=DEFAULT_BROKERS)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--starting-offsets", default="earliest")
    parser.add_argument("--watermark", default=DEFAULT_WATERMARK)
    parser.add_argument("--dim-items", type=Path, default=DIM_ITEMS_OUT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to data/clean/events_streaming and data/quarantine/events_streaming",
    )
    parser.add_argument(
        "--max-offsets-per-trigger",
        type=int,
        default=200_000,
        help="Caps micro-batch size so the first batch off a full topic stays bounded",
    )
    parser.add_argument(
        "--stop-when-idle",
        action="store_true",
        help="Stop once the topic is drained (batches stop producing rows)",
    )
    parser.add_argument(
        "--idle-batches",
        type=int,
        default=3,
        help="Consecutive empty batches before --stop-when-idle triggers",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete checkpoints and existing output before starting",
    )
    args = parser.parse_args()

    from common import CLEAN_DIR, QUARANTINE_DIR

    clean_out = (
        args.output_root / "events" if args.output_root else CLEAN_DIR / "events_streaming"
    )
    quarantine_out = (
        args.output_root / "quarantine"
        if args.output_root
        else QUARANTINE_DIR / "events_streaming"
    )
    clean_ckpt = CHECKPOINT_DIR / "clean"
    quarantine_ckpt = CHECKPOINT_DIR / "quarantine"

    if args.reset:
        for path in (clean_out, quarantine_out, clean_ckpt, quarantine_ckpt):
            if path.exists():
                shutil.rmtree(path)
        print("reset: cleared checkpoints and output")

    spark = build_spark("streaming_events", with_kafka=True)

    if not args.dim_items.exists():
        raise SystemExit(
            f"item dimension not found at {args.dim_items} -- "
            "run `python3 spark/build_item_metadata.py` first"
        )
    # Static side of the stream-static join. Cached so it is broadcast once
    # rather than re-read from Parquet on every micro-batch.
    dim_items = spark.read.parquet(str(args.dim_items)).cache()
    dim_items.count()

    def source():
        return (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", args.brokers)
            .option("subscribe", args.topic)
            .option("startingOffsets", args.starting_offsets)
            .option("maxOffsetsPerTrigger", str(args.max_offsets_per_trigger))
            .option("failOnDataLoss", "false")
            .load()
        )

    clean, _ = clean_events(source(), dim_items, streaming=True, watermark=args.watermark)
    _, quarantined = clean_events(source(), dim_items, streaming=True, watermark=args.watermark)

    clean_query = (
        clean.writeStream.format("parquet")
        .outputMode("append")
        .option("path", str(clean_out))
        .option("checkpointLocation", str(clean_ckpt))
        .partitionBy("event_date")
        .queryName("clean_events")
        .start()
    )
    quarantine_query = (
        quarantined.writeStream.format("parquet")
        .outputMode("append")
        .option("path", str(quarantine_out))
        .option("checkpointLocation", str(quarantine_ckpt))
        .queryName("quarantined_events")
        .start()
    )

    print(f"\nstreaming from {args.brokers}/{args.topic}")
    print(f"  clean      -> {clean_out}")
    print(f"  quarantine -> {quarantine_out}")
    print(f"  watermark  : {args.watermark}")
    print("  Ctrl-C to stop\n")

    try:
        monitor(
            [clean_query, quarantine_query],
            stop_when_idle=args.stop_when_idle,
            idle_batches=args.idle_batches,
        )
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        for q in (clean_query, quarantine_query):
            if q.isActive:
                q.stop()

    report(clean_query, quarantine_query)
    spark.stop()


def monitor(queries, stop_when_idle: bool, idle_batches: int) -> None:
    """Polls query progress and prints throughput.

    `numRowsDroppedByWatermark` is the number that matters here: it is the
    count of events that arrived too late for the dedupe state to still hold
    their key. Silent data loss unless it is watched, which is exactly why it
    is surfaced every batch rather than buried in the Spark UI.
    """
    totals = {q.name: 0 for q in queries}
    dropped = {q.name: 0 for q in queries}
    idle_polls = {q.name: 0 for q in queries}
    last_batch = {q.name: -1 for q in queries}

    while any(q.isActive for q in queries):
        time.sleep(2)
        for q in queries:
            progress = q.lastProgress
            if progress and progress["batchId"] != last_batch[q.name]:
                last_batch[q.name] = progress["batchId"]
                rows = progress.get("numInputRows", 0)
                totals[q.name] += rows
                for op in progress.get("stateOperators", []):
                    dropped[q.name] += op.get("numRowsDroppedByWatermark", 0)
                if rows:
                    rate = progress.get("processedRowsPerSecond") or 0
                    print(
                        f"  [{q.name}] batch {progress['batchId']}: "
                        f"{rows:,} rows in ({rate:,.0f}/s), total {totals[q.name]:,}"
                        + (f", dropped-late {dropped[q.name]:,}" if dropped[q.name] else "")
                    )

            # Idleness cannot be inferred from batch counts. When a topic
            # drains, Spark stops scheduling batches altogether rather than
            # running empty ones -- so "no new batch" means either "still
            # working" or "nothing left to do", and counting empty batches
            # waits forever because none are ever produced. The query's own
            # status distinguishes the two.
            status = q.status
            working = status["isDataAvailable"] or status["isTriggerActive"]
            has_run = last_batch[q.name] >= 0
            idle_polls[q.name] = 0 if (working or not has_run) else idle_polls[q.name] + 1

        if stop_when_idle and all(idle_polls[q.name] >= idle_batches for q in queries):
            print("\ntopic drained, stopping")
            for q in queries:
                if q.isActive:
                    q.stop()
            break

    for q in queries:
        if dropped.get(q.name):
            print(
                f"WARNING [{q.name}]: {dropped[q.name]:,} rows arrived beyond the "
                "watermark and were dropped from the dedupe state"
            )


def report(*queries) -> None:
    print("\n=== final query status ===")
    for q in queries:
        progress = q.lastProgress
        batches = progress["batchId"] + 1 if progress else 0
        print(f"{q.name}: {batches} batches, exception={q.exception()}")


if __name__ == "__main__":
    main()
