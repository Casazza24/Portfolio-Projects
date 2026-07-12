"""Build a self-contained interactive Plotly dashboard (dashboard.html).

Reads output CSVs from analysis.py and data CSVs, generates a single HTML
file with client-side tabbed navigation. No server needed.

Usage:
    python3 build_dashboard.py

Inputs:  data/pjm_clean.csv
         data/spectral_features.csv (optional)
         outputs/model_metrics.csv
         outputs/feature_importances.csv
         outputs/predictions_sample.csv
Output:  dashboard.html
"""

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(__file__) or "."
DATA = os.path.join(BASE, "data")
OUTS = os.path.join(BASE, "outputs")
HTML_OUT = os.path.join(BASE, "dashboard.html")

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
BLUE, ORANGE, RED, GREEN, TEAL = "#4C72B0", "#DD8452", "#E45756", "#2C9E6B", "#72B7B2"


def load_clean():
    df = pd.read_csv(os.path.join(DATA, "pjm_clean.csv"), parse_dates=["Datetime"])
    return df


def load_spectral():
    path = os.path.join(DATA, "spectral_features.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Datetime"])
    return None


def load_metrics():
    return pd.read_csv(os.path.join(OUTS, "model_metrics.csv"))


def load_importances():
    return pd.read_csv(os.path.join(OUTS, "feature_importances.csv"))


def load_predictions():
    return pd.read_csv(os.path.join(OUTS, "predictions_sample.csv"), parse_dates=["Datetime"])


# ponytail: one function per tab trace, dump to JSON, template into HTML string


def tab1_demand_timeseries(df):
    """Raw demand — downsample to every 6th hour for browser performance."""
    ds = df.iloc[::6].copy()
    return json.dumps([{
        "x": ds["Datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        "y": ds["PJME_MW"].tolist(),
        "type": "scatter", "mode": "lines",
        "line": {"color": BLUE, "width": 0.7},
        "name": "PJM East Demand",
    }])


def tab1_layout():
    return json.dumps({
        "title": "PJM East Hourly Energy Demand",
        "xaxis": {
            "title": "Date",
            "rangeslider": {"visible": True},
            "rangeselector": {"buttons": [
                {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                {"count": 5, "label": "5Y", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ]},
        },
        "yaxis": {"title": "Demand (MW)"},
        "margin": {"t": 50, "b": 60},
    })


def tab2_psd(df):
    """Recreate PSD from clean data via numpy FFT."""
    demand = df["PJME_MW"].values
    N = len(demand)
    centered = demand - demand.mean()
    Y = np.fft.fft(centered)
    P2 = np.abs(Y / N)
    P1 = P2[:N // 2 + 1].copy()
    P1[1:-1] *= 2
    PSD = P1 ** 2
    freq = np.arange(N // 2 + 1) / N  # cycles per hour
    period = np.where(freq > 0, 1.0 / freq, 0)

    # Filter to interesting range
    mask = (period >= 6) & (period <= 10000)
    p_filt = period[mask].tolist()
    psd_filt = PSD[mask].tolist()

    return json.dumps([{
        "x": p_filt, "y": psd_filt,
        "type": "scatter", "mode": "lines",
        "line": {"color": BLUE, "width": 0.8},
        "name": "PSD",
    }])


def tab2_layout():
    return json.dumps({
        "title": "Power Spectral Density — Dominant Cyclical Periods",
        "xaxis": {"title": "Period (hours)", "type": "log"},
        "yaxis": {"title": "Power", "type": "log"},
        "margin": {"t": 50, "b": 60},
        "shapes": [
            {"type": "line", "x0": 24, "x1": 24, "y0": 0, "y1": 1,
             "yref": "paper", "line": {"dash": "dash", "color": "#999"}},
            {"type": "line", "x0": 168, "x1": 168, "y0": 0, "y1": 1,
             "yref": "paper", "line": {"dash": "dash", "color": "#999"}},
            {"type": "line", "x0": 8760, "x1": 8760, "y0": 0, "y1": 1,
             "yref": "paper", "line": {"dash": "dash", "color": "#999"}},
        ],
        "annotations": [
            {"x": np.log10(24), "y": 1, "yref": "paper", "text": "24h (daily)",
             "showarrow": False, "yanchor": "bottom", "xref": "x"},
            {"x": np.log10(168), "y": 1, "yref": "paper", "text": "168h (weekly)",
             "showarrow": False, "yanchor": "bottom", "xref": "x"},
            {"x": np.log10(8760), "y": 1, "yref": "paper", "text": "8760h (annual)",
             "showarrow": False, "yanchor": "bottom", "xref": "x"},
        ],
    })


def tab3_predictions(pred_df):
    return json.dumps([
        {"x": pred_df["Datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
         "y": pred_df["actual_MW"].tolist(),
         "type": "scatter", "mode": "lines",
         "line": {"color": BLUE, "width": 1}, "name": "Actual"},
        {"x": pred_df["Datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
         "y": pred_df["predicted_MW"].tolist(),
         "type": "scatter", "mode": "lines",
         "line": {"color": ORANGE, "width": 1}, "name": "Predicted (GBR)"},
    ])


def tab3_layout():
    return json.dumps({
        "title": "Actual vs Predicted Demand — Test Period",
        "xaxis": {"title": "Date"},
        "yaxis": {"title": "Demand (MW)"},
        "margin": {"t": 50, "b": 60},
    })


def tab4_importance(imp_df):
    imp_df = imp_df.sort_values("importance")
    return json.dumps([{
        "x": imp_df["importance"].tolist(),
        "y": imp_df["feature"].tolist(),
        "type": "bar", "orientation": "h",
        "marker": {"color": BLUE},
    }])


def tab4_layout(imp_df):
    return json.dumps({
        "title": "Feature Importance (Gradient Boosting)",
        "xaxis": {"title": "Importance"},
        "margin": {"l": 180, "t": 50, "b": 60},
        "height": max(400, len(imp_df) * 28),
    })


def tab5_heatmap(df):
    """Hour x day_of_week mean demand heatmap."""
    df = df.copy()
    df["hour"] = df["Datetime"].dt.hour
    df["dow"] = df["Datetime"].dt.dayofweek
    pivot = df.groupby(["dow", "hour"])["PJME_MW"].mean().unstack(fill_value=0)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return json.dumps([{
        "z": pivot.values.tolist(),
        "x": list(range(24)),
        "y": days,
        "type": "heatmap",
        "colorscale": "YlOrRd",
        "colorbar": {"title": "MW"},
    }])


def tab5_layout():
    return json.dumps({
        "title": "Average Demand Heatmap — Hour x Day of Week",
        "xaxis": {"title": "Hour of Day", "dtick": 1},
        "yaxis": {"title": ""},
        "margin": {"t": 50, "b": 60, "l": 60},
    })


def tab6_ablation(metrics_df):
    """Bar chart comparing with/without spectral."""
    with_row = metrics_df[metrics_df["model"].str.contains("\\+ spectral", na=False)]
    without_row = metrics_df[metrics_df["model"].str.contains("- spectral", na=False)]
    if with_row.empty or without_row.empty:
        # No ablation data — show a placeholder
        return json.dumps([{
            "x": ["RMSE", "MAE"], "y": [0, 0],
            "type": "bar", "name": "No ablation data",
        }])

    wr = with_row.iloc[0]
    wor = without_row.iloc[0]
    return json.dumps([
        {"x": ["RMSE", "MAE"], "y": [wor["RMSE"], wor["MAE"]],
         "type": "bar", "name": "Without Spectral", "marker": {"color": ORANGE}},
        {"x": ["RMSE", "MAE"], "y": [wr["RMSE"], wr["MAE"]],
         "type": "bar", "name": "With Spectral", "marker": {"color": BLUE}},
    ])


def tab6_layout():
    return json.dumps({
        "title": "Ablation: Spectral Features Impact on Forecast Error",
        "yaxis": {"title": "Error (MW)"},
        "barmode": "group",
        "margin": {"t": 50, "b": 60},
    })


def build_html(tab_data):
    """Assemble the full dashboard HTML with client-side tab switching."""
    tabs = [
        ("demand", "Demand Time Series"),
        ("psd", "Spectral Analysis"),
        ("predictions", "Actual vs Predicted"),
        ("importance", "Feature Importance"),
        ("heatmap", "Demand Heatmap"),
        ("ablation", "Spectral Ablation"),
    ]

    tab_buttons = "\n".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" '
        f'onclick="showTab(\'{tid}\')">{label}</button>'
        for i, (tid, label) in enumerate(tabs)
    )

    tab_divs = "\n".join(
        f'<div id="tab-{tid}" class="tab-content" '
        f'style="display:{"block" if i == 0 else "none"}">'
        f'<div id="plot-{tid}"></div></div>'
        for i, (tid, _) in enumerate(tabs)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grid Energy Demand Forecasting — PJM East</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f8f9fa; color: #333; }}
  header {{ background: #2c3e50; color: white; padding: 20px 30px; }}
  header h1 {{ font-size: 1.4rem; font-weight: 600; }}
  header p {{ font-size: 0.85rem; opacity: 0.8; margin-top: 4px; }}
  .tab-bar {{ display: flex; flex-wrap: wrap; gap: 4px; padding: 12px 20px;
              background: #fff; border-bottom: 1px solid #dee2e6; }}
  .tab-btn {{ border: none; background: #e9ecef; padding: 8px 16px; cursor: pointer;
              border-radius: 6px; font-size: 0.85rem; transition: all 0.15s; }}
  .tab-btn:hover {{ background: #dee2e6; }}
  .tab-btn.active {{ background: {BLUE}; color: white; }}
  .tab-content {{ padding: 16px 20px; }}
</style>
</head>
<body>
<header>
  <h1>Grid Energy Demand Forecasting &mdash; PJM East</h1>
  <p>FFT-based periodicity analysis + gradient boosting | Phase 3D Portfolio Project</p>
</header>
<div class="tab-bar">{tab_buttons}</div>
{tab_divs}
<script>
function showTab(id) {{
  document.querySelectorAll('.tab-content').forEach(d => d.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).style.display = 'block';
  event.target.classList.add('active');
  Plotly.Plots.resize(document.getElementById('plot-' + id));
}}

var cfg = {{responsive: true}};

Plotly.newPlot('plot-demand', {tab_data['demand_traces']}, {tab_data['demand_layout']}, cfg);
Plotly.newPlot('plot-psd', {tab_data['psd_traces']}, {tab_data['psd_layout']}, cfg);
Plotly.newPlot('plot-predictions', {tab_data['pred_traces']}, {tab_data['pred_layout']}, cfg);
Plotly.newPlot('plot-importance', {tab_data['imp_traces']}, {tab_data['imp_layout']}, cfg);
Plotly.newPlot('plot-heatmap', {tab_data['heat_traces']}, {tab_data['heat_layout']}, cfg);
Plotly.newPlot('plot-ablation', {tab_data['abl_traces']}, {tab_data['abl_layout']}, cfg);
</script>
</body>
</html>"""


def main():
    print("Building dashboard ...")
    df = load_clean()
    metrics = load_metrics()
    imp = load_importances()
    pred = load_predictions()

    tab_data = {
        "demand_traces": tab1_demand_timeseries(df),
        "demand_layout": tab1_layout(),
        "psd_traces": tab2_psd(df),
        "psd_layout": tab2_layout(),
        "pred_traces": tab3_predictions(pred),
        "pred_layout": tab3_layout(),
        "imp_traces": tab4_importance(imp),
        "imp_layout": tab4_layout(imp),
        "heat_traces": tab5_heatmap(df),
        "heat_layout": tab5_layout(),
        "abl_traces": tab6_ablation(metrics),
        "abl_layout": tab6_layout(),
    }

    html = build_html(tab_data)
    with open(HTML_OUT, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(HTML_OUT) / 1024
    print(f"Saved {HTML_OUT} ({size_kb:.0f} KB)")
    print("Open in browser to view. Plotly loaded from CDN.")


if __name__ == "__main__":
    main()
