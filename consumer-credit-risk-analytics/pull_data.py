#!/usr/bin/env python3
"""Phase 3B — download and filter the Lending Club loan dataset from Kaggle.

Setup (one-time):
  1. pip install kaggle
  2. Go to kaggle.com -> Account -> Create New API Token
  3. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
  4. chmod 600 ~/.kaggle/kaggle.json

Run:
  python3 pull_data.py

Downloads the full Lending Club dataset (~1.7GB zip), extracts CSVs to data/,
filters to only fully-paid and charged-off loans (the two definitive outcome
classes), and saves data/lending_club_clean.csv. Drops current, in-grace-period,
late, and issued loans since their outcome is unknown.

# ponytail: kaggle CLI does the heavy lifting. No retry logic — re-run if it fails.
# ponytail: pandas only for the filter step; could be csv module but the file is
#   huge and pandas chunked reading handles memory better for 2.2M rows.
"""

import os
import subprocess
import sys
import glob

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _find_raw_csvs():
    """Find raw CSV files in data/ — handles Kaggle's nested unzip layout.
    # ponytail: Kaggle --unzip on .csv.gz creates dirs like data/foo.csv/Foo.csv
    """
    found = []
    for pattern in [os.path.join(DATA, "*.csv"), os.path.join(DATA, "**", "*.csv")]:
        for f in glob.glob(pattern, recursive=True):
            if os.path.isfile(f) and "clean" not in os.path.basename(f) and "scored" not in os.path.basename(f):
                found.append(f)
    return list(set(found))


def download():
    """Download and unzip the Lending Club dataset via Kaggle CLI."""
    os.makedirs(DATA, exist_ok=True)

    raw_csvs = _find_raw_csvs()
    if raw_csvs:
        print(f"Raw CSV(s) already present: {[os.path.basename(f) for f in raw_csvs]}")
        print("Delete them and re-run to force a fresh download.")
        return raw_csvs

    print("Downloading Lending Club dataset from Kaggle...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "wordsforthewise/lending-club",
        "-p", DATA, "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"kaggle CLI failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(result.stdout)

    raw_csvs = _find_raw_csvs()
    if not raw_csvs:
        print("ERROR: no CSV files found after download.", file=sys.stderr)
        sys.exit(1)
    print(f"Downloaded: {[os.path.basename(f) for f in raw_csvs]}")
    return raw_csvs


def filter_loans(raw_csvs):
    """Keep only Fully Paid and Charged Off loans. Save to lending_club_clean.csv."""
    out_path = os.path.join(DATA, "lending_club_clean.csv")

    # The dataset may come as one or two CSVs (accepted.csv / rejected.csv).
    # We only want accepted loans with known outcomes.
    KEEP_STATUSES = {"Fully Paid", "Charged Off"}

    frames = []
    for path in raw_csvs:
        print(f"Reading {os.path.basename(path)}...")
        # ponytail: low_memory=False avoids mixed-type warnings on big files
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as e:
            print(f"  Skipping {os.path.basename(path)}: {e}")
            continue

        if "loan_status" not in df.columns:
            print(f"  No 'loan_status' column — skipping (likely rejected loans file).")
            continue

        before = len(df)
        df = df[df["loan_status"].isin(KEEP_STATUSES)]
        after = len(df)
        print(f"  {before:,} rows -> {after:,} after filtering to {KEEP_STATUSES}")
        frames.append(df)

    if not frames:
        print("ERROR: no frames with loan_status found.", file=sys.stderr)
        sys.exit(1)

    clean = pd.concat(frames, ignore_index=True)
    clean.to_csv(out_path, index=False)

    n_default = (clean["loan_status"] == "Charged Off").sum()
    n_paid = (clean["loan_status"] == "Fully Paid").sum()
    rate = n_default / len(clean) * 100

    print(f"\nSaved {out_path}")
    print(f"  Total: {len(clean):,} loans")
    print(f"  Fully Paid: {n_paid:,}  |  Charged Off: {n_default:,}")
    print(f"  Default rate: {rate:.1f}%")
    return out_path


if __name__ == "__main__":
    raw = download()
    filter_loans(raw)
