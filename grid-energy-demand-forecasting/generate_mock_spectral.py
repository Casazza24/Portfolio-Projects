"""Generate synthetic spectral features from pjm_clean.csv timestamps.

TEST HELPER — lets the full Python pipeline (analysis.py, build_dashboard.py)
run without MATLAB installed. Produces data/spectral_features.csv with
sin/cos pairs at the three dominant energy demand periods (24h daily, 168h
weekly, 8760h annual) using realistic amplitudes derived from typical PJM
East demand swings.

Usage:
    python3 generate_mock_spectral.py

Replace with real MATLAB output (spectral_analysis.m) for final results.
"""

import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__) or ".", "data")
IN_FILE = os.path.join(DATA, "pjm_clean.csv")
OUT_FILE = os.path.join(DATA, "spectral_features.csv")

# Approximate amplitudes (MW) from typical PJM East FFT peaks
# ponytail: eyeballed from PJM data characteristics, good enough for pipeline testing
COMPONENTS = [
    ("daily",  24,   3500),   # ~3500 MW daily swing
    ("weekly", 168,  1500),   # ~1500 MW weekday/weekend difference
    ("annual", 8760, 5000),   # ~5000 MW summer/winter difference
]


def main():
    df = pd.read_csv(IN_FILE, parse_dates=["Datetime"])
    t0 = df["Datetime"].iloc[0]
    t_hours = (df["Datetime"] - t0).dt.total_seconds() / 3600.0

    out = pd.DataFrame({"Datetime": df["Datetime"]})
    for name, period_h, amplitude in COMPONENTS:
        omega = 2 * np.pi / period_h
        # ponytail: random phase offset so mock data isn't suspiciously aligned
        phase = np.random.default_rng(42).uniform(0, 2 * np.pi)
        out[f"spectral_{name}_sin"] = amplitude * np.sin(omega * t_hours + phase)
        out[f"spectral_{name}_cos"] = amplitude * np.cos(omega * t_hours + phase)

    out.to_csv(OUT_FILE, index=False)
    print(f"Mock spectral features: {OUT_FILE} ({len(out):,} rows, "
          f"{len(COMPONENTS)} components x 2 = {len(COMPONENTS)*2} feature cols)")
    print("NOTE: These are synthetic. Run spectral_analysis.m for real FFT features.")


if __name__ == "__main__":
    main()
