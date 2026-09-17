#!/usr/bin/env python3
"""Replays RetailRocket events.csv in timestamp order at an adjustable speed.

Phase 1 emitted JSON lines to stdout. Phase 2 adds a Redpanda (Kafka-API) sink
and, importantly, *deliberate messiness* -- duplicate deliveries, out-of-order
arrivals, and malformed payloads -- so the Phase 3 Spark job has real data
quality problems to solve rather than a synthetically clean stream.

The messiness is seeded, so a given --seed reproduces byte-identical output.
That matters: the batch backfill job has to be able to reproduce the same
result as the streaming job over the same input.

Usage:
    # stdout, as in Phase 1
    python3 replay_retailrocket.py --speed 10000 --limit 1000

    # produce to Redpanda
    python3 replay_retailrocket.py --sink kafka --speed 0 --limit 200000

    # clean stream, no injected faults
    python3 replay_retailrocket.py --sink kafka --dup-rate 0 --late-rate 0 \
        --malformed-rate 0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterator, NamedTuple

DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent / "data" / "raw" / "retailrocket" / "events.csv"
)
DEFAULT_BROKERS = "localhost:19092"
DEFAULT_TOPIC = "retail.events"

# Injected-fault rates. Tuned to be visible in aggregate (a dedupe/quarantine
# step has obvious work to do) without swamping the real signal.
DEFAULT_DUP_RATE = 0.02
DEFAULT_LATE_RATE = 0.01
DEFAULT_MALFORMED_RATE = 0.005

# How far behind a "late" event is allowed to arrive, in positions in the
# output stream. The Phase 3 watermark is sized against this.
LATE_MIN_LAG = 50
LATE_MAX_LAG = 5000


class SourceEvent(NamedTuple):
    """A row of events.csv, kept as a tuple rather than a dict -- there are
    2.75M of them and dicts roughly triple the resident memory."""

    event_time: int
    visitor_id: int
    event_type: str
    item_id: int
    transaction_id: int | None
    row_index: int


def read_events(path: Path, limit: int | None) -> list[SourceEvent]:
    """Loads events.csv, sorted by timestamp.

    events.csv is not guaranteed to be timestamp-sorted on disk, so we sort
    once up front -- the replay is only meaningful if it follows real event
    order. Any disorder downstream is disorder *we* injected, which keeps the
    Phase 3 late-arrival handling honest.
    """
    rows: list[SourceEvent] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected = ["timestamp", "visitorid", "event", "itemid", "transactionid"]
        if header != expected:
            raise ValueError(f"unexpected events.csv header: {header!r}")

        for i, row in enumerate(reader):
            ts, visitor, event, item, txn = row
            rows.append(
                SourceEvent(
                    event_time=int(ts),
                    visitor_id=int(visitor),
                    event_type=event,
                    item_id=int(item),
                    transaction_id=int(txn) if txn else None,
                    row_index=i,
                )
            )

    rows.sort(key=lambda r: r.event_time)
    return rows[:limit] if limit else rows


def event_id_for(e: SourceEvent) -> str:
    """Stable synthetic id for a source row.

    Real producers stamp an id so consumers can dedupe at-least-once
    redeliveries. We derive one deterministically from the row so that a
    duplicate delivery carries the *same* id (a genuine redelivery) while two
    coincidentally identical events keep distinct ids -- row_index is what
    separates them.
    """
    raw = f"{e.row_index}|{e.event_time}|{e.visitor_id}|{e.item_id}".encode()
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def envelope(e: SourceEvent, producer_id: str) -> dict:
    return {
        "event_id": event_id_for(e),
        "event_time": e.event_time,
        "visitor_id": e.visitor_id,
        "event_type": e.event_type,
        "item_id": e.item_id,
        "transaction_id": e.transaction_id,
        "ingested_at": int(time.time() * 1000),
        "producer_id": producer_id,
    }


def corrupt(payload: dict, rng: random.Random) -> str:
    """Returns a deliberately broken serialization of an event.

    Each variant models a failure this pipeline could genuinely see:
      null_item      -- upstream service wrote a null FK
      bad_timestamp  -- a client sent an ISO string into a millis field
      missing_field  -- schema drift; a producer stopped sending event_type
      truncated_json -- a payload cut off mid-write
      epoch_zero     -- an uninitialized timestamp, far outside any watermark
    """
    variant = rng.choice(
        ["null_item", "bad_timestamp", "missing_field", "truncated_json", "epoch_zero"]
    )
    bad = dict(payload)
    bad["_fault"] = variant

    if variant == "null_item":
        bad["item_id"] = None
    elif variant == "bad_timestamp":
        bad["event_time"] = "2015-06-02T05:02:12Z"
    elif variant == "missing_field":
        bad.pop("event_type")
    elif variant == "epoch_zero":
        bad["event_time"] = 0
    elif variant == "truncated_json":
        text = json.dumps(bad)
        return text[: len(text) // 2]

    return json.dumps(bad)


def build_stream(
    rows: list[SourceEvent],
    producer_id: str,
    dup_rate: float,
    late_rate: float,
    malformed_rate: float,
    rng: random.Random,
) -> Iterator[tuple[bytes, str, int]]:
    """Yields (key, serialized_payload, event_time) in delivery order.

    Delivery order deliberately differs from event-time order: `late` events
    are held in a min-heap and released thousands of positions later, which is
    what an at-least-once broker with a slow partition actually looks like.
    """
    # (release_position, sequence, key, payload, event_time)
    delayed: list[tuple[int, int, bytes, str, int]] = []
    seq = 0
    position = 0

    def flush_due(upto: int) -> Iterator[tuple[bytes, str, int]]:
        while delayed and delayed[0][0] <= upto:
            _, _, key, payload, ts = heapq.heappop(delayed)
            yield key, payload, ts

    for e in rows:
        # Key by visitor so one visitor's events land on one partition. This
        # preserves per-visitor ordering while leaving the stream globally
        # out-of-order -- the realistic case.
        key = str(e.visitor_id).encode()
        payload = envelope(e, producer_id)

        if rng.random() < malformed_rate:
            yield key, corrupt(payload, rng), e.event_time
            position += 1
            yield from flush_due(position)
            continue

        text = json.dumps(payload)
        emissions = [text]
        # An at-least-once broker redelivers on ack timeout: same id, same body.
        if rng.random() < dup_rate:
            emissions.append(text)

        for body in emissions:
            if rng.random() < late_rate:
                lag = rng.randint(LATE_MIN_LAG, LATE_MAX_LAG)
                heapq.heappush(delayed, (position + lag, seq, key, body, e.event_time))
                seq += 1
            else:
                yield key, body, e.event_time
                position += 1
                yield from flush_due(position)

    # Drain anything still held back at end of stream.
    while delayed:
        _, _, key, payload, ts = heapq.heappop(delayed)
        yield key, payload, ts


class StdoutSink:
    def __init__(self, out):
        self.out = out

    def send(self, key: bytes, payload: str) -> None:
        self.out.write(payload + "\n")
        self.out.flush()

    def close(self) -> None:
        pass


class KafkaSink:
    """Thin wrapper over confluent_kafka.Producer.

    linger/batch settings favour throughput -- a full replay is 2.8M messages
    and the default per-message flush would take hours.
    """

    def __init__(self, brokers: str, topic: str):
        from confluent_kafka import Producer

        self.topic = topic
        self.delivered = 0
        self.failed = 0
        self.producer = Producer(
            {
                "bootstrap.servers": brokers,
                "linger.ms": 50,
                "batch.size": 1 << 20,
                "compression.type": "lz4",
                "acks": "all",
                "enable.idempotence": True,
            }
        )

    def _on_delivery(self, err, _msg) -> None:
        if err is None:
            self.delivered += 1
        else:
            self.failed += 1
            print(f"delivery failed: {err}", file=sys.stderr)

    def send(self, key: bytes, payload: str) -> None:
        while True:
            try:
                self.producer.produce(
                    self.topic, key=key, value=payload.encode(), callback=self._on_delivery
                )
                break
            except BufferError:
                # Local queue full -- let librdkafka drain, then retry.
                self.producer.poll(0.5)
        self.producer.poll(0)

    def close(self) -> None:
        self.producer.flush(30)


def replay(stream, sink, speed: float, report_every: int) -> int:
    """speed: real-time multiplier. 0 = no delay (fire as fast as possible).
    1.0 = replay at the pace the events originally occurred.
    10000 = 10,000x faster than real time (a day of events in ~8.6s).
    """
    prev_ts = None
    count = 0
    for key, payload, event_time in stream:
        if speed > 0 and prev_ts is not None:
            gap_seconds = (event_time - prev_ts) / 1000.0
            delay = gap_seconds / speed
            if delay > 0:
                time.sleep(min(delay, 5.0))  # cap so one huge gap doesn't stall a demo
        prev_ts = event_time
        sink.send(key, payload)
        count += 1
        if report_every and count % report_every == 0:
            print(f"  ... {count:,} messages sent", file=sys.stderr)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to events.csv")
    parser.add_argument("--sink", choices=["stdout", "kafka"], default="stdout")
    parser.add_argument("--brokers", default=DEFAULT_BROKERS)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument(
        "--speed",
        type=float,
        default=10000.0,
        help="Real-time speed multiplier (0 = as fast as possible, 1 = real-time)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only replay the first N events")
    parser.add_argument("--seed", type=int, default=42, help="Seed for injected-fault randomness")
    parser.add_argument("--producer-id", default="replay-1")
    parser.add_argument("--dup-rate", type=float, default=DEFAULT_DUP_RATE)
    parser.add_argument("--late-rate", type=float, default=DEFAULT_LATE_RATE)
    parser.add_argument("--malformed-rate", type=float, default=DEFAULT_MALFORMED_RATE)
    parser.add_argument("--report-every", type=int, default=100_000)
    args = parser.parse_args()

    print(f"loading {args.input} ...", file=sys.stderr)
    rows = read_events(args.input, args.limit)
    print(f"loaded {len(rows):,} source events", file=sys.stderr)

    rng = random.Random(args.seed)
    stream = build_stream(
        rows, args.producer_id, args.dup_rate, args.late_rate, args.malformed_rate, rng
    )

    if args.sink == "kafka":
        sink = KafkaSink(args.brokers, args.topic)
        print(f"producing to {args.brokers} topic={args.topic}", file=sys.stderr)
    else:
        sink = StdoutSink(sys.stdout)

    started = time.time()
    try:
        count = replay(stream, sink, args.speed, args.report_every)
    finally:
        sink.close()

    elapsed = time.time() - started
    rate = count / elapsed if elapsed else 0
    print(
        f"sent {count:,} messages from {len(rows):,} source events "
        f"in {elapsed:.1f}s ({rate:,.0f} msg/s)",
        file=sys.stderr,
    )
    if isinstance(sink, KafkaSink):
        print(f"acked={sink.delivered:,} failed={sink.failed:,}", file=sys.stderr)
        if sink.failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
