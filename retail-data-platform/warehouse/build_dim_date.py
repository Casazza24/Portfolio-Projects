#!/usr/bin/env python3
"""Generates the shared date dimension for the Phase 4 marts.

One row per calendar day, covering *both* sources -- Olist orders (2016-2018)
and RetailRocket events (May-Sep 2015) -- so `fct_orders` and `fct_events` can
conform on a single `dim_date` instead of each carrying its own calendar.

The range is derived from the data rather than declared: the low/high order
dates come from Postgres, and the event range comes from the `event_date=`
partition directory names Spark wrote (reading directory names beats reading
2.7M rows of Parquet, and needs no Spark session to do it). If neither source
is reachable, it falls back to a fixed range that covers both datasets -- this
script has to stay runnable on a laptop with Docker stopped, because a date
dimension is exactly the thing you want to be able to rebuild anywhere.

The derived span is then widened to whole calendar years. A date dimension with
a ragged first and last year makes YTD and prior-year comparisons quietly wrong
at the edges, and the extra rows cost nothing.

Row `-1` is the Kimball unknown member. Phase 3 already resolves unmatched item
categories to `-1` rather than to null (see spark/transforms.py), and any fact
row with an unparseable or absent date has to resolve somewhere; without this
row a strict dbt `relationships` test on the date key fails on rows that are
correctly modelled.

Usage:
    python3 warehouse/build_dim_date.py
    python3 warehouse/build_dim_date.py --start 2014-01-01 --end 2020-12-31
"""
from __future__ import annotations

import argparse
import calendar
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIM_DATE_CSV, EVENTS_DIR  # noqa: E402

# Used only when neither Postgres nor the event partitions can be read. Chosen
# to contain both datasets outright, so a fallback build is still correct --
# just less tightly fitted than a derived one.
FALLBACK_RANGE = (date(2015, 1, 1), date(2018, 12, 31))

UNKNOWN_KEY = -1

FIELDS = [
    "date_key",
    "date_day",
    "year",
    "quarter",
    "month",
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "iso_year",
    "day_of_week",
    "day_name",
    "day_abbr",
    "month_name",
    "month_abbr",
    "year_month",
    "year_quarter",
    "first_day_of_month",
    "last_day_of_month",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_end",
    "is_year_end",
]


def olist_range() -> tuple[date, date] | None:
    """Order date span from Postgres, or None if it is not reachable.

    Bounded by the estimated delivery date as well as the purchase date: the
    marts measure delivery lateness, and estimated delivery runs months past
    the last purchase. A dimension that stopped at the last purchase would
    leave those rows with an unresolvable date key.
    """
    try:
        from common import pg_connect

        conn = pg_connect()
    except Exception as exc:  # noqa: BLE001 -- any failure means "fall back"
        print(f"  Postgres unavailable ({exc.__class__.__name__}); skipping", file=sys.stderr)
        return None

    with conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT min(order_purchase_timestamp)::date,
                   greatest(max(order_purchase_timestamp)::date,
                            max(order_estimated_delivery_date)::date,
                            max(order_delivered_customer_date)::date)
            FROM olist.orders
            """
        )
        low, high = cur.fetchone()
    conn.close()
    return (low, high) if low and high else None


def events_range(events_dir: Path) -> tuple[date, date] | None:
    """Event date span, read from Spark's Hive-style partition directory names."""
    days = []
    for part in events_dir.glob("event_date=*"):
        if not any(part.glob("*.parquet")):
            continue  # empty partition dir, nothing was written for that date
        try:
            days.append(date.fromisoformat(part.name.split("=", 1)[1]))
        except ValueError:
            continue
    return (min(days), max(days)) if days else None


def unknown_row() -> dict:
    """The Kimball unknown member: a resolvable key with no calendar meaning.

    Numeric attributes are -1 and text attributes read 'Unknown' rather than
    being left blank, so an unknown date is visible in a BI slicer instead of
    silently disappearing from it -- the same reasoning as the unknown item
    category in spark/transforms.py.
    """
    row = {field: UNKNOWN_KEY for field in FIELDS}
    row.update(
        {
            "date_day": "",
            "day_name": "Unknown",
            "day_abbr": "Unknown",
            "month_name": "Unknown",
            "month_abbr": "Unknown",
            "year_month": "Unknown",
            "year_quarter": "Unknown",
            "first_day_of_month": "",
            "last_day_of_month": "",
        }
    )
    row.update({f: "false" for f in FIELDS if f.startswith("is_")})
    return row


def day_row(day: date) -> dict:
    iso_year, iso_week, iso_weekday = day.isocalendar()
    last_dom = calendar.monthrange(day.year, day.month)[1]
    quarter = (day.month - 1) // 3 + 1
    return {
        # yyyymmdd rather than a surrogate sequence: it sorts correctly, it is
        # readable in a raw table, and it survives a partial reload unchanged.
        "date_key": int(day.strftime("%Y%m%d")),
        "date_day": day.isoformat(),
        "year": day.year,
        "quarter": quarter,
        "month": day.month,
        "day_of_month": day.day,
        "day_of_year": day.timetuple().tm_yday,
        "week_of_year": iso_week,
        "iso_year": iso_year,
        "day_of_week": iso_weekday,  # ISO: 1 = Monday .. 7 = Sunday
        "day_name": calendar.day_name[day.weekday()],
        "day_abbr": calendar.day_abbr[day.weekday()],
        "month_name": calendar.month_name[day.month],
        "month_abbr": calendar.month_abbr[day.month],
        "year_month": f"{day.year}-{day.month:02d}",
        "year_quarter": f"{day.year}-Q{quarter}",
        "first_day_of_month": day.replace(day=1).isoformat(),
        "last_day_of_month": day.replace(day=last_dom).isoformat(),
        "is_weekend": str(iso_weekday >= 6).lower(),
        "is_month_start": str(day.day == 1).lower(),
        "is_month_end": str(day.day == last_dom).lower(),
        "is_quarter_end": str(day.month in (3, 6, 9, 12) and day.day == last_dom).lower(),
        "is_year_end": str(day.month == 12 and day.day == 31).lower(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--events", type=Path, default=EVENTS_DIR)
    parser.add_argument("--output", type=Path, default=DIM_DATE_CSV)
    args = parser.parse_args()

    spans = []
    olist = olist_range()
    if olist:
        print(f"  olist orders     : {olist[0]} -> {olist[1]}")
        spans.append(olist)
    events = events_range(args.events)
    if events:
        print(f"  retailrocket     : {events[0]} -> {events[1]}")
        spans.append(events)
    if not spans:
        print("  no source reachable; using fallback range", file=sys.stderr)
        spans.append(FALLBACK_RANGE)

    low = args.start or min(s[0] for s in spans).replace(month=1, day=1)
    high = args.end or max(s[1] for s in spans).replace(month=12, day=31)
    if low > high:
        raise SystemExit(f"empty range: {low} -> {high}")
    print(f"  dimension covers : {low} -> {high}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(unknown_row())
        day, written = low, 1
        while day <= high:
            writer.writerow(day_row(day))
            day += timedelta(days=1)
            written += 1

    print(f"\nwrote {args.output}")
    print(f"  rows       : {written:,} ({written - 1:,} days + 1 unknown member)")
    print(f"  range      : {low} -> {high}")


if __name__ == "__main__":
    main()
