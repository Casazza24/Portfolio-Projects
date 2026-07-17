"""Grid energy demand forecasting — feature engineering, modeling, ablation.

Loads cleaned PJM East hourly data, engineers calendar/lag/rolling features,
optionally merges MATLAB spectral features, trains a GradientBoostingRegressor,
compares against a seasonal naive baseline, and runs an ablation study to
measure the impact of spectral features on forecast accuracy.

Usage:
    python3 analysis.py

Inputs:  data/pjm_clean.csv
         data/spectral_features.csv  (optional — from spectral_analysis.m or
                                      generate_mock_spectral.py)
Outputs: outputs/feature_importance.png
         outputs/actual_vs_predicted.png
         outputs/residuals.png
         outputs/ablation_comparison.png
         outputs/model_metrics.csv
         outputs/feature_importances.csv
         outputs/predictions_sample.csv
         data/pjm_predictions.csv     (for Tableau)
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = os.path.dirname(__file__) or "."
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, RED, GREEN = "#4C72B0", "#DD8452", "#E45756", "#2C9E6B"


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(os.path.join(DATA, "pjm_clean.csv"), parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    print(f"Loaded {len(df):,} rows: {df['Datetime'].min()} to {df['Datetime'].max()}")

    spectral_path = os.path.join(DATA, "spectral_features.csv")
    spectral_cols = []
    if os.path.exists(spectral_path):
        sf = pd.read_csv(spectral_path, parse_dates=["Datetime"])
        spectral_cols = [c for c in sf.columns if c.startswith("spectral_")]
        df = df.merge(sf, on="Datetime", how="left")
        # ponytail: forward-fill any edge NaNs from merge alignment
        for c in spectral_cols:
            df[c] = df[c].ffill().bfill()
        print(f"Merged {len(spectral_cols)} spectral features from {spectral_path}")
    else:
        print("WARNING: spectral_features.csv not found — running without spectral features.")
        print("  Run spectral_analysis.m (MATLAB) or generate_mock_spectral.py first.")

    return df, spectral_cols


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df, spectral_cols):
    """Add calendar, lag, rolling, and holiday features."""
    dt = df["Datetime"]

    # Calendar
    df["hour_of_day"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Holidays
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=dt.min(), end=dt.max())
    df["is_holiday"] = dt.dt.normalize().isin(holidays).astype(int)

    # Lags
    df["lag_1"] = df["PJME_MW"].shift(1)
    df["lag_24"] = df["PJME_MW"].shift(24)
    df["lag_168"] = df["PJME_MW"].shift(168)

    # Rolling stats (24h window)
    df["rolling_24h_mean"] = df["PJME_MW"].shift(1).rolling(24).mean()
    df["rolling_24h_std"] = df["PJME_MW"].shift(1).rolling(24).std()

    # Drop rows with NaN from lags/rolling (first 168 hours)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
        "lag_1", "lag_24", "lag_168", "rolling_24h_mean", "rolling_24h_std",
    ] + spectral_cols

    print(f"Features: {len(feature_cols)} total ({len(spectral_cols)} spectral)")
    return df, feature_cols


# ---------------------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------------------

def temporal_split(df, test_months=3):
    """Last `test_months` months as test set."""
    cutoff = df["Datetime"].max() - pd.DateOffset(months=test_months)
    train = df[df["Datetime"] < cutoff].copy()
    test = df[df["Datetime"] >= cutoff].copy()
    print(f"Train: {len(train):,} rows ({train['Datetime'].min().date()} to "
          f"{train['Datetime'].max().date()})")
    print(f"Test:  {len(test):,} rows ({test['Datetime'].min().date()} to "
          f"{test['Datetime'].max().date()})")
    return train, test


# ---------------------------------------------------------------------------
# 4. Modeling
# ---------------------------------------------------------------------------

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def seasonal_naive(train, test):
    """Predict same hour last week. Simple but surprisingly competitive."""
    # ponytail: just shift test's own lag_168 column — it's already computed
    preds = test["lag_168"].values
    return preds


def train_gbr(X_train, y_train, X_test):
    """GradientBoostingRegressor with sensible defaults."""
    # ponytail: no hyperparameter search — these defaults are fine for a portfolio demo
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds


def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    m = mape(y_true, y_pred)
    print(f"  {label:30s}  RMSE={rmse:,.0f}  MAE={mae:,.0f}  MAPE={m:.2f}%")
    return {"model": label, "RMSE": round(rmse, 1), "MAE": round(mae, 1), "MAPE": round(m, 2)}


# ---------------------------------------------------------------------------
# 5. Ablation
# ---------------------------------------------------------------------------

def run_ablation(train, test, feature_cols, spectral_cols, y_train, y_test):
    """Train with and without spectral features, report delta."""
    if not spectral_cols:
        print("\n  Ablation skipped — no spectral features available.")
        return None, None, None

    # WITH spectral
    model_with, preds_with = train_gbr(
        train[feature_cols], y_train, test[feature_cols])
    metrics_with = evaluate(y_test, preds_with, "GBR + spectral")

    # WITHOUT spectral
    base_cols = [c for c in feature_cols if c not in spectral_cols]
    model_without, preds_without = train_gbr(
        train[base_cols], y_train, test[base_cols])
    metrics_without = evaluate(y_test, preds_without, "GBR - spectral")

    delta_rmse = metrics_without["RMSE"] - metrics_with["RMSE"]
    delta_pct = delta_rmse / metrics_without["RMSE"] * 100
    print(f"\n  Spectral features reduce RMSE by {delta_rmse:,.0f} MW ({delta_pct:.1f}%)")

    return metrics_with, metrics_without, model_with


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_cols):
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(5, len(imp) * 0.35)))
    imp.plot.barh(ax=ax, color=BLUE)
    ax.set_xlabel("Feature Importance (impurity decrease)")
    ax.set_title("Gradient Boosting — Feature Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
    plt.close(fig)
    # Save CSV
    imp_df = imp.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(OUT, "feature_importances.csv"), index=False)


def plot_actual_vs_predicted(test, preds, label="GBR"):
    # ponytail: subsample for readability — 2 weeks of hourly data
    n_show = min(24 * 14, len(test))
    t = test["Datetime"].iloc[:n_show]
    y = test["PJME_MW"].iloc[:n_show]
    p = preds[:n_show]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, y, label="Actual", color=BLUE, linewidth=0.8)
    ax.plot(t, p, label=f"Predicted ({label})", color=ORANGE, linewidth=0.8, alpha=0.85)
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (MW)")
    ax.set_title("Actual vs Predicted — Test Period (first 2 weeks)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "actual_vs_predicted.png"), dpi=150)
    plt.close(fig)

    # Save sample CSV
    sample = pd.DataFrame({
        "Datetime": test["Datetime"].iloc[:n_show].values,
        "actual_MW": y.values,
        "predicted_MW": np.round(p, 1),
    })
    sample.to_csv(os.path.join(OUT, "predictions_sample.csv"), index=False)


def plot_residuals(test, preds):
    residuals = test["PJME_MW"].values - preds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(residuals, bins=60, color=BLUE, edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color=RED, linestyle="--")
    axes[0].set_xlabel("Residual (MW)")
    axes[0].set_title("Residual Distribution")

    axes[1].scatter(preds, residuals, s=1, alpha=0.15, color=BLUE)
    axes[1].axhline(0, color=RED, linestyle="--")
    axes[1].set_xlabel("Predicted (MW)")
    axes[1].set_ylabel("Residual (MW)")
    axes[1].set_title("Residuals vs Predicted")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "residuals.png"), dpi=150)
    plt.close(fig)


def plot_ablation(metrics_with, metrics_without):
    if metrics_with is None:
        return
    labels = ["RMSE", "MAE"]
    with_vals = [metrics_with["RMSE"], metrics_with["MAE"]]
    without_vals = [metrics_without["RMSE"], metrics_without["MAE"]]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w/2, without_vals, w, label="Without Spectral", color=ORANGE)
    ax.bar(x + w/2, with_vals, w, label="With Spectral", color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (MW)")
    ax.set_title("Ablation: Impact of Spectral Features")
    ax.legend()

    # Annotate deltas
    for i, (wo, wi) in enumerate(zip(without_vals, with_vals)):
        delta = wo - wi
        pct = delta / wo * 100
        ax.annotate(f"-{delta:,.0f}\n({pct:.1f}%)",
                    xy=(x[i] + w/2, wi), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, color=RED)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "ablation_comparison.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Grid Energy Demand Forecasting — PJM East")
    print("=" * 60)

    # Load
    df, spectral_cols = load_data()
    df, feature_cols = engineer_features(df, spectral_cols)
    train, test = temporal_split(df)

    y_train = train["PJME_MW"].values
    y_test = test["PJME_MW"].values

    # --- Baselines ---
    print("\n--- Model Evaluation ---")
    naive_preds = seasonal_naive(train, test)
    m_naive = evaluate(y_test, naive_preds, "Seasonal Naive (same hour last wk)")

    # --- Full GBR ---
    model, gbr_preds = train_gbr(train[feature_cols], y_train, test[feature_cols])
    m_gbr = evaluate(y_test, gbr_preds, "Gradient Boosting (all features)")

    # --- Ablation ---
    print("\n--- Ablation Study ---")
    m_with, m_without, model_ablation = run_ablation(
        train, test, feature_cols, spectral_cols, y_train, y_test)

    # --- Save metrics ---
    metrics_rows = [m_naive, m_gbr]
    if m_with:
        metrics_rows.extend([m_with, m_without])
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(OUT, "model_metrics.csv"), index=False)

    # --- Plots ---
    print("\n--- Generating Plots ---")
    # Use the ablation model (with spectral) if available, else the full model
    best_model = model_ablation if model_ablation else model
    best_preds = gbr_preds  # full model preds for the plots

    plot_feature_importance(best_model, feature_cols)
    print("  Saved feature_importance.png + feature_importances.csv")

    plot_actual_vs_predicted(test, best_preds)
    print("  Saved actual_vs_predicted.png + predictions_sample.csv")

    plot_residuals(test, best_preds)
    print("  Saved residuals.png")

    plot_ablation(m_with, m_without)
    if m_with:
        print("  Saved ablation_comparison.png")

    # --- Tableau export ---
    tableau = test[["Datetime", "PJME_MW"]].copy()
    tableau["predicted_MW"] = np.round(best_preds, 1)
    tableau["residual_MW"] = np.round(y_test - best_preds, 1)
    tableau.to_csv(os.path.join(DATA, "pjm_predictions.csv"), index=False)
    print(f"\n  Saved data/pjm_predictions.csv ({len(tableau):,} rows) for Tableau")

    print("\nDone.")


if __name__ == "__main__":
    main()
