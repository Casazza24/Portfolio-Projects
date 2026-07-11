#!/usr/bin/env python3
"""Phase 3A — one-shot data pull for the NYC Transit Efficiency Analysis.

Pulls three raw CSVs into data/:
  - mta_ridership_raw.csv      MTA Daily Ridership, subway only (Socrata, data.ny.gov)
  - weather_raw.csv            NOAA CDO daily summaries for NYC Central Park
  - mta_delays_monthly_raw.csv MTA Subway Delay-Causing Incidents, monthly (Socrata)

Run once: `python3 pull_data.py`. Prints a sanity summary and exits non-zero
if any of the three files is empty. The daily ridership and weather pulls also
get an overlap check (they're joined downstream); the monthly delay pull is an
independent secondary signal, so it only gets a row-count + month-range report.

# ponytail: no retry/backoff. If either API flakes, just re-run the script.
# ponytail: requests + stdlib only, raw csv writing, no pandas.
"""

import csv
import json
import os
import socket
import sys
import time
from datetime import date, datetime, timedelta

import requests

# ponytail: force IPv4 — on Matt's network, IPv6 routes to NOAA/Socrata hang
# past the requests timeout (no Happy-Eyeballs fallback in urllib3), while
# IPv4 answers instantly. Patch getaddrinfo to drop IPv6 results globally.
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4_only_getaddrinfo

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# MTA Daily Ridership and Traffic: Beginning 2020 (data.ny.gov, Socrata).
# Switched here from the "Delay-Causing Incidents" dataset (g937-7k7c) and all
# the related delay/performance datasets (Major Incidents ereg-mcvp, Service
# Delivered 32ch-sei3, Terminal OTP f6rf-2a3t, 4-5min Late x7nj-r656, Customer
# Journey r7qk-6tcy) because EVERY one of those is aggregated by month +
# day_type -- there is no true per-calendar-day delay record anywhere in the
# MTA catalog. This dataset has a genuine per-day `date` field (one row per
# mode per day), so it joins cleanly against daily NOAA weather. It's a
# ridership signal, not a delay-cause breakdown -- the deliberate tradeoff
# Matt chose: daily granularity over delay-cause richness. We filter to
# mode='Subway'; `count` is total estimated subway ridership that day.
MTA_DOMAIN = "https://data.ny.gov"
MTA_DATASET = "sayj-mze2"
MTA_MODE = "Subway"          # long-format dataset; one row per mode per day
# Genuine per-day floating timestamp (not a monthly bucket). Fixed, not probed.
MTA_DATE_FIELD = "date"

# MTA Subway Delay-Causing Incidents: Beginning 2020 (g937-7k7c). Secondary
# signal — "which lines have the most delays, by category". This dataset is
# aggregated by `month` (floating timestamp, first-of-month) + `day_type`
# (weekday/weekend) + line + category, so there's no daily join here; it's
# its own independent pull. The `month` field is the month bucket.
MTA_DELAYS_DATASET = "g937-7k7c"
MTA_DELAYS_DATE_FIELD = "month"

# NOAA CDO daily summaries, NY City Central Park.
NOAA_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
NOAA_STATION = "GHCND:USW00094728"
NOAA_DATATYPES = ("TMAX", "TMIN", "PRCP", "SNOW")


def load_env(path):
    """Parse simple KEY=VALUE lines from .env (stdlib only)."""
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def one_year_window():
    """Trailing 1-year daily window ending at the last day of last month.

    Matt chose a 1-year (not multi-year) span so the daily ridership/weather
    join has a full seasonal cycle to control for day-of-week and month effects
    and to surface one-off event spikes. End is the last day of last month (the
    most recent complete data); start is the first of that same month one year
    earlier — i.e. 12 full calendar months.
    """
    today = date.today()
    first_of_this_month = today.replace(day=1)
    end = first_of_this_month - timedelta(days=1)          # last day of last month
    end_month_first = end.replace(day=1)
    start = end_month_first.replace(year=end_month_first.year - 1)  # 12 months back
    return start, end


def pull_mta(token, start, end):
    url = f"{MTA_DOMAIN}/resource/{MTA_DATASET}.json"
    headers = {"X-App-Token": token} if token else {}
    # Real per-day timestamps; range-filter lexically and pin to subway rows.
    where = (
        f"mode = '{MTA_MODE}' AND "
        f"{MTA_DATE_FIELD} >= '{start.isoformat()}T00:00:00' AND "
        f"{MTA_DATE_FIELD} <= '{end.isoformat()}T23:59:59'"
    )
    params = {"$where": where, "$order": MTA_DATE_FIELD, "$limit": 50000}
    r = requests.get(url, headers=headers, params=params, timeout=120)
    r.raise_for_status()
    rows = r.json()
    out = os.path.join(DATA_DIR, "mta_ridership_raw.csv")
    write_csv(out, rows)
    return out, rows, MTA_DATE_FIELD


def delays_window(end):
    """First-of-month 12 months back through `end`'s month, for the monthly pull.

    The delay dataset is monthly-bucketed, so a 2-month window would yield only
    ~2 rows per line — too thin for a "which lines have the most delays" story.
    We use a 12-month trailing window so each line/category has enough months to
    rank meaningfully, while still ending in the same overall period as the
    daily pull. Returns (start_first_of_month, end_first_of_month).
    """
    end_month_first = end.replace(day=1)
    # Step back 12 months from the end month's first day.
    y, m = end_month_first.year, end_month_first.month - 11
    while m <= 0:
        m += 12
        y -= 1
    return date(y, m, 1), end_month_first


def pull_mta_delays(token, start, end):
    url = f"{MTA_DOMAIN}/resource/{MTA_DELAYS_DATASET}.json"
    headers = {"X-App-Token": token} if token else {}
    where = (
        f"{MTA_DELAYS_DATE_FIELD} >= '{start.isoformat()}T00:00:00' AND "
        f"{MTA_DELAYS_DATE_FIELD} <= '{end.isoformat()}T23:59:59'"
    )
    params = {"$where": where, "$order": MTA_DELAYS_DATE_FIELD, "$limit": 50000}
    r = requests.get(url, headers=headers, params=params, timeout=120)
    r.raise_for_status()
    rows = r.json()
    out = os.path.join(DATA_DIR, "mta_delays_monthly_raw.csv")
    write_csv(out, rows)
    return out, rows, MTA_DELAYS_DATE_FIELD


def pull_weather(token, start, end):
    headers = {"token": token}
    out_rows = []
    offset = 1  # NOAA CDO offset is 1-based
    while True:
        params = {
            "datasetid": "GHCND",
            "stationid": NOAA_STATION,
            "startdate": start.isoformat(),
            "enddate": end.isoformat(),
            "units": "metric",
            "limit": 1000,
            "offset": offset,
        }
        # datatypeid repeated for each requested type
        params_list = list(params.items()) + [("datatypeid", d) for d in NOAA_DATATYPES]
        r = requests.get(NOAA_BASE, headers=headers, params=params_list, timeout=120)
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results", [])
        if not results:
            break
        out_rows.extend(results)
        meta = payload.get("metadata", {}).get("resultset", {})
        total = int(meta.get("count", len(out_rows)))
        offset += len(results)
        if offset > total or len(results) < 1000:
            break
        # ponytail: ~1460 rows = 2 pages over a year; tiny sleep to stay clear
        # of NOAA's 5 req/sec cap between pages. Nothing fancier.
        time.sleep(0.25)
    out = os.path.join(DATA_DIR, "weather_raw.csv")
    write_csv(out, out_rows)
    return out, out_rows


def write_csv(path, rows):
    """Write list-of-dicts to CSV. Union of keys as header (Socrata rows vary)."""
    if not rows:
        # Still create the (empty) file so the caller's sanity check fires cleanly.
        open(path, "w").close()
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def date_range_of(rows, field):
    """Min/max date (as date objects) over rows, reading `field`, ISO-ish."""
    seen = []
    for row in rows:
        raw = row.get(field)
        if not raw:
            continue
        # Take just the date part of an ISO timestamp.
        try:
            seen.append(datetime.fromisoformat(raw.replace("Z", "")).date())
        except ValueError:
            try:
                seen.append(date.fromisoformat(raw[:10]))
            except ValueError:
                continue
    if not seen:
        return None, None
    return min(seen), max(seen)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    env = load_env(os.path.join(HERE, ".env"))
    mta_token = env.get("MTA_SOCRATA_TOKEN")
    noaa_token = env.get("NOAA_CDO_TOKEN")
    if not noaa_token:
        sys.exit("NOAA_CDO_TOKEN missing from .env")

    start, end = one_year_window()
    print(f"Window: {start} .. {end}")

    delay_start, delay_end = delays_window(end)

    print("Pulling MTA daily subway ridership ...")
    mta_path, mta_rows, mta_date_field = pull_mta(mta_token, start, end)
    print("Pulling NOAA daily weather ...")
    wx_path, wx_rows = pull_weather(noaa_token, start, end)
    print(f"Pulling MTA monthly delay incidents ({delay_start}..{delay_end}) ...")
    dly_path, dly_rows, dly_date_field = pull_mta_delays(mta_token, delay_start, delay_end)

    mta_min, mta_max = date_range_of(mta_rows, mta_date_field)
    wx_min, wx_max = date_range_of(wx_rows, "date")
    dly_min, dly_max = date_range_of(dly_rows, dly_date_field)

    print("\n--- sanity summary ---")
    print(f"Ridership {mta_path}")
    print(f"          rows={len(mta_rows)}  date_field='{mta_date_field}'  range={mta_min}..{mta_max}")
    print(f"Weather   {wx_path}")
    print(f"          rows={len(wx_rows)}  range={wx_min}..{wx_max}")
    print(f"Delays    {dly_path}")
    print(f"          rows={len(dly_rows)}  date_field='{dly_date_field}'  month_range={dly_min}..{dly_max}")

    problems = []
    if not mta_rows:
        problems.append("Ridership file is empty.")
    if not wx_rows:
        problems.append("Weather file is empty.")
    if not dly_rows:
        problems.append("Delays file is empty.")
    if mta_min and wx_min:
        overlap_start = max(mta_min, wx_min)
        overlap_end = min(mta_max, wx_max)
        if overlap_start > overlap_end:
            problems.append(
                f"Date ranges do not overlap: MTA {mta_min}..{mta_max} vs "
                f"weather {wx_min}..{wx_max}."
            )
    if problems:
        print("\nFAIL:")
        for p in problems:
            print("  - " + p)
        sys.exit(1)

    print("\nOK: all three files non-empty; ridership/weather ranges overlap.")


# ponytail: the runnable self-check IS just running this file end-to-end.
if __name__ == "__main__":
    main()
