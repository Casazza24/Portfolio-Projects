# Grid Energy Demand Forecasting

A time series forecasting project that uses spectral analysis (FFT) to engineer cyclical features for predicting hourly electricity demand on the PJM Interconnection, the largest US wholesale electricity market serving 65 million people. The core question: does explicitly extracting periodic components via Fourier analysis improve forecasts compared to standard calendar features and lag variables?

## Key Findings

- **Lag features dominate.** Recent demand (t-1, t-24, rolling mean) is the strongest predictor, confirming that yesterday's demand is the best predictor of today's.
- **Spectral features provide measurable improvement.** FFT-derived sin/cos pairs reduce forecasting error compared to calendar dummies alone. The improvement is modest because lag features already carry much of the cyclical signal implicitly.
- **Three dominant cycles confirmed.** FFT on approximately 145,000 hourly observations reveals clear peaks at 24 hours (daily, strongest), 168 hours (weekly), and approximately 8,760 hours (annual).
- **Spectral features have practical advantages over lags.** They require no lookback window, are model-agnostic, and work at the first timestamp of a new series, which lag features cannot do.

## Methodology

1. **Data pull**: PJM East hourly consumption data (2002-2018) from Kaggle, cleaned and deduplicated
2. **Spectral analysis** (MATLAB): FFT on the full demand signal, power spectral density computation, extraction of sin/cos feature pairs for the three dominant frequency components
3. **Feature engineering and modeling** (Python): Calendar features (hour, day of week, month, weekend/holiday flags), autoregressive lags (t-1, t-24, t-168), rolling statistics, and six spectral features. Temporal train/test split with the last three months held out. Gradient boosting regressor as the primary model; seasonal naive baseline for comparison.
4. **Ablation study**: Model runs with and without spectral features to isolate their marginal contribution

## Tech Stack

- MATLAB for FFT computation and spectral feature extraction
- Python (pandas, NumPy, scikit-learn, matplotlib, Plotly) for data cleaning, feature engineering, modeling, and visualization

## Project Structure

| File | Description |
|---|---|
| `spectral_analysis.m` | MATLAB script for FFT, power spectral density, and sin/cos feature extraction |
| `analysis.py` | Feature engineering, model training, evaluation, and ablation study |
| `build_dashboard.py` | Generates a self-contained interactive Plotly dashboard |
| `pull_data.py` | Data acquisition script |
| `generate_mock_spectral.py` | Generates mock spectral features for testing without MATLAB |
| `requirements.txt` | Python dependencies |

## Running Locally

```bash
# Run spectral analysis in MATLAB first (or use mock data)
python generate_mock_spectral.py  # if MATLAB is not available

pip install -r requirements.txt
python pull_data.py
python analysis.py
python build_dashboard.py
# Open dashboard.html in your browser
```

## Live Demo

[Interactive Dashboard](https://casazza24.github.io/Portfolio-Projects/grid-energy-demand-forecasting/dashboard.html)
