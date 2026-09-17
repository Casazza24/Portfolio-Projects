#!/usr/bin/env python3
"""Fetches daily historical weather per Brazilian state from Open-Meteo.

This is the external-API ingestion step, and the source for the weather
dimension the Phase 4 marts join against ("did revenue dip in bad weather?").

Grain: one row per (state, date). Olist records a customer state but not
coordinates, so each state needs a single representative point. Rather than
hardcoding capital cities, the coordinates are derived from the Olist
geolocation table itself -- the median lat/lng of that state's zip-code
prefixes. Zip prefixes are denser where people are, so the median lands near
the population centre, which is the right place to measure weather that would
plausibly affect orders.

That is still an approximation, and a deliberate one: for a large, climatically
mixed state (AM, BA, MT) a single point cannot represent the whole state. It is
proportionate here because 90% of orders come from a handful of compact
southeastern states, and it is recorded in docs/data_quality.md rather than
left implicit.

No API key required. Open-Meteo's archive endpoint accepts many locations per
request, so all 27 states cost only a few calls.

Usage:
    python3 ingestion/fetch_weather.py
    python3 ingestion/fetch_weather.py --start 2016-09-01 --end 2018-10-31
"""
from __future__ import annotations

import argparse
import csv
import json
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "clean" / "dim_weather.csv"
PG_CONTAINER = "retail_platform_postgres"

DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "weather_code",
]

# Olist's geolocation table contains a number of coordinates outside Brazil
# (bad geocoding upstream). Clip to Brazil's bounding box before taking a
# median so a handful of junk points cannot drag a state's centroid.
BRAZIL_BOUNDS = {"lat_min": -34.0, "lat_max": 5.5, "lng_min": -74.0, "lng_max": -34.0}

# Requests are batched by location count to keep URLs and responses a sane
# size; Open-Meteo is generous but not unlimited.
LOCATIONS_PER_REQUEST = 10
RETRIES = 4


def ssl_context() -> ssl.SSLContext:
    """Builds an SSL context with a usable CA bundle.

    The python.org macOS builds ship without wiring up system root
    certificates, so HTTPS fails with CERTIFICATE_VERIFY_FAILED until either
    `Install Certificates.command` is run or certifi's bundle is pointed at
    explicitly. Doing it here keeps the script working on a fresh clone
    instead of failing with an error that looks like a network problem.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def query_postgres(sql: str) -> list[list[str]]:
    """Runs SQL in the project's Postgres container and returns parsed rows.

    Shells out via `docker exec` rather than taking a psycopg2 dependency --
    this script is the only thing in ingestion/ that touches Postgres from
    Python, and it is not worth a driver install for one query.
    """
    result = subprocess.run(
        ["docker", "exec", PG_CONTAINER, "psql", "-U", "retail", "-d", "retail",
         "-t", "-A", "-F", "\t", "-c", sql],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Postgres query failed (is `docker compose up -d` running?):\n{result.stderr}"
        )
    return [line.split("\t") for line in result.stdout.strip().splitlines() if line.strip()]


def state_centroids() -> list[tuple[str, float, float, int]]:
    """Returns (state, lat, lng, order_count), busiest state first."""
    sql = f"""
        WITH clean_geo AS (
            SELECT geolocation_state AS state,
                   geolocation_lat   AS lat,
                   geolocation_lng   AS lng
            FROM olist.geolocation
            WHERE geolocation_lat BETWEEN {BRAZIL_BOUNDS['lat_min']} AND {BRAZIL_BOUNDS['lat_max']}
              AND geolocation_lng BETWEEN {BRAZIL_BOUNDS['lng_min']} AND {BRAZIL_BOUNDS['lng_max']}
        ),
        centroid AS (
            SELECT state,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY lat) AS lat,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY lng) AS lng
            FROM clean_geo
            GROUP BY state
        ),
        volume AS (
            SELECT c.customer_state AS state, count(*) AS orders
            FROM olist.orders o
            JOIN olist.customers c USING (customer_id)
            GROUP BY 1
        )
        SELECT centroid.state,
               round(centroid.lat::numeric, 4),
               round(centroid.lng::numeric, 4),
               coalesce(volume.orders, 0)
        FROM centroid
        LEFT JOIN volume USING (state)
        ORDER BY coalesce(volume.orders, 0) DESC;
    """
    return [(r[0], float(r[1]), float(r[2]), int(r[3])) for r in query_postgres(sql)]


def order_date_range() -> tuple[str, str]:
    rows = query_postgres(
        "SELECT min(order_purchase_timestamp)::date, max(order_purchase_timestamp)::date "
        "FROM olist.orders;"
    )
    return rows[0][0], rows[0][1]


def fetch_batch(batch, start: str, end: str) -> list[dict]:
    """Fetches one batch of locations. Open-Meteo returns a bare object for a
    single location and a list for several, so both shapes are normalised."""
    params = {
        "latitude": ",".join(str(lat) for _, lat, _, _ in batch),
        "longitude": ",".join(str(lng) for _, _, lng, _ in batch),
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "America/Sao_Paulo",
    }
    url = f"{ARCHIVE_URL}?{urllib.parse.urlencode(params)}"
    context = ssl_context()

    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=120, context=context) as response:
                payload = json.load(response)
            return payload if isinstance(payload, list) else [payload]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            if attempt == RETRIES:
                raise SystemExit(f"Open-Meteo request failed after {RETRIES} attempts: {exc}")
            backoff = 2**attempt
            print(f"  request failed ({exc}); retrying in {backoff}s", file=sys.stderr)
            time.sleep(backoff)
    return []


def rows_from_response(state: str, orders: int, block: dict):
    daily = block.get("daily")
    if not daily:
        return
    for i, day in enumerate(daily["time"]):
        yield {
            "state": state,
            "weather_date": day,
            "latitude": round(block["latitude"], 4),
            "longitude": round(block["longitude"], 4),
            "orders_in_state": orders,
            **{var: daily[var][i] for var in DAILY_VARIABLES},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (defaults to first order date)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (defaults to last order date)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    start, end = order_date_range()
    start = args.start or start
    end = args.end or end
    print(f"date range: {start} -> {end}")

    states = state_centroids()
    print(f"states: {len(states)} (from Olist geolocation medians)")

    all_rows = []
    for i in range(0, len(states), LOCATIONS_PER_REQUEST):
        batch = states[i : i + LOCATIONS_PER_REQUEST]
        labels = ", ".join(s for s, _, _, _ in batch)
        print(f"  fetching {labels} ...")
        blocks = fetch_batch(batch, start, end)
        if len(blocks) != len(batch):
            raise SystemExit(
                f"expected {len(batch)} location blocks, got {len(blocks)} -- "
                "Open-Meteo response shape changed"
            )
        for (state, _, _, orders), block in zip(batch, blocks):
            all_rows.extend(rows_from_response(state, orders, block))
        time.sleep(1)  # be polite to a free API

    if not all_rows:
        raise SystemExit("no weather rows returned")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    days = len({r["weather_date"] for r in all_rows})
    missing = sum(1 for r in all_rows if r["temperature_2m_mean"] is None)
    print(f"\nwrote {args.output}")
    print(f"  rows        : {len(all_rows):,}")
    print(f"  states      : {len({r['state'] for r in all_rows})}")
    print(f"  days        : {days:,}")
    print(f"  null temps  : {missing:,}")


if __name__ == "__main__":
    main()
