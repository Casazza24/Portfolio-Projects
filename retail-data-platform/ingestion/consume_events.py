#!/usr/bin/env python3
"""Consumer that verifies the producer -> Redpanda -> consumer loop.

This is a Phase 2 demo/verification tool, not part of the production path --
Phase 3's Spark job is the real consumer. Its job is to prove events are
flowing and to report what the stream actually looks like, including the
injected faults, so the Phase 3 cleaning logic has a target to hit.

Usage:
    # tail the topic live
    python3 consume_events.py

    # read the whole topic from the start and print a summary
    python3 consume_events.py --from-beginning --summary --max 300000
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections import Counter

DEFAULT_BROKERS = "localhost:19092"
DEFAULT_TOPIC = "retail.events"

_stop = False


def _handle_sigint(_sig, _frame):
    global _stop
    _stop = True
    print("\nstopping...", file=sys.stderr)


class StreamStats:
    """Tracks what the consumer sees, without trusting the producer's `_fault`
    marker to decide what is broken -- validity is judged from the payload
    itself, the same way the Spark job will have to. `_fault` is only read back
    afterwards to score whether detection actually caught what was injected.
    """

    def __init__(self) -> None:
        self.total = 0
        self.unparseable = 0
        self.invalid = 0
        self.valid = 0
        self.event_types: Counter[str] = Counter()
        self.partitions: Counter[int] = Counter()
        self.injected_faults: Counter[str] = Counter()
        self.detected_but_unmarked = 0
        self.seen_ids: set[str] = set()
        self.duplicates = 0
        self.out_of_order = 0
        self.max_event_time = 0
        self.max_lateness_ms = 0

    def observe(self, raw: bytes, partition: int) -> None:
        self.total += 1
        self.partitions[partition] += 1

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.unparseable += 1
            return

        fault = payload.get("_fault")
        if fault:
            self.injected_faults[fault] += 1

        problem = self._validate(payload)
        if problem:
            self.invalid += 1
            if not fault:
                self.detected_but_unmarked += 1
            return

        self.valid += 1
        self.event_types[payload["event_type"]] += 1

        event_id = payload["event_id"]
        if event_id in self.seen_ids:
            self.duplicates += 1
        else:
            self.seen_ids.add(event_id)

        ts = payload["event_time"]
        if ts < self.max_event_time:
            self.out_of_order += 1
            self.max_lateness_ms = max(self.max_lateness_ms, self.max_event_time - ts)
        else:
            self.max_event_time = ts

    @staticmethod
    def _validate(payload: dict) -> str | None:
        for field in ("event_id", "event_time", "visitor_id", "event_type", "item_id"):
            if payload.get(field) is None:
                return f"missing:{field}"
        if not isinstance(payload["event_time"], int):
            return "event_time:not_int"
        # RetailRocket covers May-Sep 2015; anything near epoch is uninitialized.
        if payload["event_time"] < 1_000_000_000_000:
            return "event_time:implausible"
        if payload["event_type"] not in ("view", "addtocart", "transaction"):
            return "event_type:unknown"
        return None

    def report(self) -> None:
        pct = lambda n: f"{n / self.total * 100:.2f}%" if self.total else "-"
        print("\n=== stream summary ===", file=sys.stderr)
        print(f"messages consumed : {self.total:,}", file=sys.stderr)
        print(f"  valid           : {self.valid:,} ({pct(self.valid)})", file=sys.stderr)
        print(f"  invalid         : {self.invalid:,} ({pct(self.invalid)})", file=sys.stderr)
        print(f"  unparseable     : {self.unparseable:,} ({pct(self.unparseable)})", file=sys.stderr)
        print(f"duplicate ids     : {self.duplicates:,} ({pct(self.duplicates)})", file=sys.stderr)
        print(f"out-of-order      : {self.out_of_order:,} ({pct(self.out_of_order)})", file=sys.stderr)
        print(f"max lateness      : {self.max_lateness_ms / 1000:,.0f}s", file=sys.stderr)
        print(f"event types       : {dict(self.event_types)}", file=sys.stderr)
        print(f"per partition     : {dict(sorted(self.partitions.items()))}", file=sys.stderr)
        print(f"injected faults   : {dict(self.injected_faults)}", file=sys.stderr)
        if self.detected_but_unmarked:
            print(
                f"NOTE: {self.detected_but_unmarked:,} messages failed validation but "
                "carried no _fault marker -- these are real defects in the source data.",
                file=sys.stderr,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brokers", default=DEFAULT_BROKERS)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--group", default="phase2-verify")
    parser.add_argument(
        "--from-beginning",
        action="store_true",
        help="Read the topic from offset 0 (uses a fresh consumer group each run)",
    )
    parser.add_argument("--max", type=int, default=None, help="Stop after N messages")
    parser.add_argument("--summary", action="store_true", help="Print a summary on exit")
    parser.add_argument(
        "--print", dest="do_print", action="store_true", help="Print each message to stdout"
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=10.0,
        help="Exit after this many seconds with no new messages (0 = wait forever)",
    )
    args = parser.parse_args()

    from confluent_kafka import Consumer, KafkaError

    group = f"{args.group}-{int(time.time())}" if args.from_beginning else args.group
    consumer = Consumer(
        {
            "bootstrap.servers": args.brokers,
            "group.id": group,
            "auto.offset.reset": "earliest" if args.from_beginning else "latest",
            "enable.auto.commit": True,
        }
    )
    consumer.subscribe([args.topic])
    signal.signal(signal.SIGINT, _handle_sigint)

    stats = StreamStats()
    print(
        f"consuming {args.topic} from {args.brokers} (group={group}) -- Ctrl-C to stop",
        file=sys.stderr,
    )

    last_message_at = time.time()
    started = time.time()
    try:
        while not _stop:
            msg = consumer.poll(1.0)
            if msg is None:
                if args.idle_timeout and time.time() - last_message_at > args.idle_timeout:
                    print(f"idle for {args.idle_timeout}s, stopping", file=sys.stderr)
                    break
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"consumer error: {msg.error()}", file=sys.stderr)
                continue

            last_message_at = time.time()
            stats.observe(msg.value(), msg.partition())
            if args.do_print:
                sys.stdout.write(msg.value().decode("utf-8", "replace") + "\n")
            if stats.total % 100_000 == 0:
                print(f"  ... {stats.total:,} consumed", file=sys.stderr)
            if args.max and stats.total >= args.max:
                break
    finally:
        consumer.close()

    elapsed = time.time() - started
    print(
        f"consumed {stats.total:,} messages in {elapsed:.1f}s "
        f"({stats.total / elapsed if elapsed else 0:,.0f} msg/s)",
        file=sys.stderr,
    )
    if args.summary:
        stats.report()


if __name__ == "__main__":
    main()
