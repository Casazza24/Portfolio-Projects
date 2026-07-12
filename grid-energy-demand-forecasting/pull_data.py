"""Download PJM hourly energy consumption data from Kaggle and clean it.

Setup (one-time):
    pip install kaggle
    # Create a Kaggle API token at https://www.kaggle.com/settings
    # Place kaggle.json in ~/.kaggle/ (chmod 600)

Usage:
    python3 pull_data.py

Downloads the robikscube/hourly-energy-consumption dataset, extracts
PJME_hourly.csv (PJM East — largest, most complete region), cleans it,
and saves data/pjm_clean.csv.
"""

import os
import subprocess
import sys
import zipfile

import pandas as pd

DATA = os.path.join(os.path.dirname(__file__) or ".", "data")
DATASET = "robikscube/hourly-energy-consumption"
ZIP_NAME = "hourly-energy-consumption.zip"
RAW_FILE = "PJME_hourly.csv"
OUT_FILE = os.path.join(DATA, "pjm_clean.csv")


def download():
    """Pull the dataset zip from Kaggle CLI into data/."""
    os.makedirs(DATA, exist_ok=True)
    zip_path = os.path.join(DATA, ZIP_NAME)
    if os.path.exists(zip_path):
        print(f"  {ZIP_NAME} already exists, skipping download")
        return zip_path
    print(f"  Downloading {DATASET} ...")
    subprocess.check_call([
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", DATASET, "-p", DATA,
    ])
    return zip_path


def extract(zip_path):
    """Extract only PJME_hourly.csv from the zip."""
    raw_path = os.path.join(DATA, RAW_FILE)
    if os.path.exists(raw_path):
        print(f"  {RAW_FILE} already exists, skipping extraction")
        return raw_path
    print(f"  Extracting {RAW_FILE} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(RAW_FILE, DATA)
    return raw_path


def clean(raw_path):
    """Parse, sort, deduplicate, interpolate nulls, save cleaned CSV."""
    print("  Cleaning ...")
    df = pd.read_csv(raw_path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").drop_duplicates(subset="Datetime")
    df = df.set_index("Datetime")

    # Interpolate any missing values (rare but possible)
    nulls = df["PJME_MW"].isna().sum()
    if nulls:
        print(f"  Interpolating {nulls} missing values")
        df["PJME_MW"] = df["PJME_MW"].interpolate(method="time")

    df = df.reset_index()
    df.to_csv(OUT_FILE, index=False)
    print(f"  Saved {OUT_FILE}: {len(df):,} rows, "
          f"{df['Datetime'].min().date()} to {df['Datetime'].max().date()}")
    return df


def main():
    print("=== PJM East Hourly Energy Data Pull ===")
    zip_path = download()
    raw_path = extract(zip_path)
    clean(raw_path)
    print("Done.")


if __name__ == "__main__":
    main()
