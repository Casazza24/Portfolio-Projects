# NYC Transit Efficiency Analysis

An analysis of what actually drives NYC subway ridership, using 13 months of daily ridership data joined with NOAA weather observations. The project starts with the intuitive hypothesis that bad weather keeps riders off the subway, then systematically dismantles it: the naive rain effect (-9% on a small sample) collapses to roughly -1% once day-of-week confounding is removed, and a multivariate regression reveals that calendar structure explains nearly all ridership variation.

## Key Findings

- **Day of week is the dominant driver.** Weekends see approximately 39% fewer riders than weekdays (a gap of roughly 1.5 million daily trips). This single factor dwarfs everything else.
- **Holidays behave like extra weekends.** A federal holiday drops ridership by approximately 1.52 million riders, landing between Saturday and Sunday in magnitude. This effect is invisible in one-factor-at-a-time analysis.
- **Rain is real but small.** After controlling for all other factors, rain reduces ridership by roughly 123,000 riders per day (about 3% of a weekday). The original 9% estimate was inflated by weekend confounding.
- **Temperature has no independent effect.** Max temperature is not statistically significant (p = 0.14) once month/season is in the model. The intuition that extreme temperatures empty the subway does not survive the controls.
- **The full regression model explains 88% of daily variance** (R-squared = 0.88, n = 396).
- **Worst delay lines**: the 6, 2, N, A, and F lines had the most delay incidents. Police and medical incidents (20,415) outpaced infrastructure and equipment failures (17,220) as the top system-wide cause.

## Methodology

1. Pull daily subway ridership from MTA Open Data (Socrata API) and daily weather from NOAA Climate Data Online (Central Park station)
2. Join datasets on date across a 13-month window (May 2025 to May 2026, 396 observations)
3. Demonstrate the confounding problem: naive rain-vs-dry comparison, then the same comparison within weekdays only
4. Fit a multivariate regression (ridership ~ weekday + month + rain + temperature + holiday) to isolate each factor's effect while holding others constant
5. Separately analyze monthly delay incidents by line and cause
6. Flag anomalous high-ridership days as candidate event days for cross-referencing

## Tech Stack

- Python (pandas, NumPy, scikit-learn, matplotlib, Plotly)
- Jupyter Notebook for exploratory analysis
- MTA Socrata API and NOAA CDO API for data acquisition

## Project Structure

| File | Description |
|---|---|
| `analysis.ipynb` | Full analysis notebook: data exploration, confound demonstration, regression, delay analysis |
| `build_dashboard.py` | Generates a self-contained interactive dashboard |
| `pull_data.py` | API data acquisition for ridership and weather |
| `dashboard.html` | Interactive dashboard with day-of-week, seasonal, and driver-ranking charts |
| `requirements.txt` | Python dependencies |

## Running Locally

```bash
pip install -r requirements.txt
python pull_data.py
jupyter notebook analysis.ipynb
python build_dashboard.py
# Open dashboard.html in your browser
```

## Live Demo

[Interactive Dashboard](https://casazza24.github.io/Portfolio-Projects/nyc-transit-efficiency-analysis/dashboard.html)
