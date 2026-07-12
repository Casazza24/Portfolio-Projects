#!/usr/bin/env python3
"""Phase 3B — Build a self-contained Plotly dashboard from analysis outputs.

    python3 build_dashboard.py

Reads the CSVs that analysis.py wrote to outputs/ and data/, generates
dashboard.html with client-side tab navigation. Same pattern as Phase 3A.

# ponytail: static Plotly HTML, no Dash server. CDN plotly.js, data inlined
#   as JSON in the page. Re-run after analysis.py to refresh.
"""

import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
DATA = os.path.join(HERE, "data")
DASHBOARD = os.path.join(HERE, "dashboard.html")

BLUE = "#4C72B0"
ORANGE = "#DD8452"
RED = "#E45756"
GREEN = "#2C9E6B"
TEAL = "#72B7B2"
PURPLE = "#9D7DB8"


def read_csv(folder, name):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def build_grade_chart(grade_df):
    """Tab 1: Default rate by loan grade."""
    grades = grade_df.sort_values("grade")
    trace = {
        "x": grades["grade"].tolist(),
        "y": [round(r * 100, 1) for r in grades["default_rate"]],
        "text": [f"{r*100:.1f}% ({n:,} loans)" for r, n in
                 zip(grades["default_rate"], grades["loan_count"])],
        "type": "bar",
        "marker": {"color": [BLUE, BLUE, TEAL, ORANGE, ORANGE, RED, RED][:len(grades)]},
        "hovertemplate": "Grade %{x}<br>Default rate: %{y:.1f}%<br>%{text}<extra></extra>",
    }
    layout = {
        "title": {"text": "Default Rate by Loan Grade"},
        "xaxis": {"title": "Loan Grade"},
        "yaxis": {"title": "Default Rate (%)", "range": [0, max(trace["y"]) * 1.15]},
        "margin": {"t": 60, "b": 50},
    }
    return trace, layout


def build_roc_chart(roc_df):
    """Tab 2: ROC curve comparison."""
    traces = []

    # Read model comparison for AUC values
    comp_df = read_csv(OUT, "model_comparison.csv")
    lr_auc = comp_df.loc[comp_df["model"] == "Logistic Regression", "roc_auc"].values[0] if comp_df is not None else 0.5
    xgb_auc = comp_df.loc[comp_df["model"] == "XGBoost", "roc_auc"].values[0] if comp_df is not None else 0.5

    lr_fpr = roc_df["lr_fpr"].dropna().tolist()
    lr_tpr = roc_df["lr_tpr"].dropna().tolist()
    xgb_fpr = roc_df["xgb_fpr"].dropna().tolist()
    xgb_tpr = roc_df["xgb_tpr"].dropna().tolist()

    traces.append({
        "x": lr_fpr, "y": lr_tpr, "type": "scatter", "mode": "lines",
        "name": f"Logistic Regression (AUC={lr_auc:.3f})",
        "line": {"color": BLUE, "width": 2},
    })
    traces.append({
        "x": xgb_fpr, "y": xgb_tpr, "type": "scatter", "mode": "lines",
        "name": f"XGBoost (AUC={xgb_auc:.3f})",
        "line": {"color": ORANGE, "width": 2},
    })
    traces.append({
        "x": [0, 1], "y": [0, 1], "type": "scatter", "mode": "lines",
        "name": "Random (AUC=0.500)",
        "line": {"color": "#999", "width": 1, "dash": "dash"},
        "showlegend": True,
    })
    layout = {
        "title": {"text": "ROC Curve — Model Comparison"},
        "xaxis": {"title": "False Positive Rate"},
        "yaxis": {"title": "True Positive Rate"},
        "legend": {"x": 0.55, "y": 0.15},
        "margin": {"t": 60},
    }
    return traces, layout


def build_importance_chart(imp_df):
    """Tab 3: Top 15 feature importances."""
    # imp_df has index=feature name, column="importance"
    top15 = imp_df.head(15).iloc[::-1]  # reverse for horizontal bar
    names = top15.index.tolist()
    # Clean up feature names for display
    display_names = [n.replace("_", " ").replace("home ownership ", "Home: ")
                      .replace("purpose ", "Purpose: ").title() for n in names]
    vals = top15["importance"].tolist()
    trace = {
        "y": display_names, "x": vals, "type": "bar",
        "orientation": "h",
        "marker": {"color": BLUE},
        "hovertemplate": "%{y}<br>Importance: %{x:.4f}<extra></extra>",
    }
    layout = {
        "title": {"text": "Top 15 Feature Importances — XGBoost"},
        "xaxis": {"title": "Feature Importance (gain)"},
        "margin": {"l": 180, "t": 60, "r": 30},
        "height": 500,
    }
    return trace, layout


def build_dti_income_chart(dti_df):
    """Tab 4: Default rate by DTI bucket and income bracket."""
    # Grouped bar chart: x = DTI bucket, color = income bracket
    dti_order = ["low", "medium", "high", "very_high"]
    inc_order = ["<40k", "40-75k", "75-120k", "120k+"]
    colors = [RED, ORANGE, TEAL, BLUE]

    traces = []
    for inc, color in zip(inc_order, colors):
        sub = dti_df[dti_df["income_bracket"] == inc].set_index("dti_bucket")
        sub = sub.reindex(dti_order)
        rates = [round(r * 100, 1) if pd.notna(r) else 0 for r in sub["default_rate"]]
        counts = [int(n) if pd.notna(n) else 0 for n in sub["loan_count"]]
        traces.append({
            "x": ["Low (0-10)", "Medium (10-20)", "High (20-30)", "Very High (30+)"],
            "y": rates, "type": "bar", "name": inc,
            "marker": {"color": color},
            "text": [f"{n:,} loans" for n in counts],
            "hovertemplate": "DTI: %{x}<br>Income: " + inc +
                             "<br>Default rate: %{y:.1f}%<br>%{text}<extra></extra>",
        })
    layout = {
        "title": {"text": "Default Rate by DTI Bucket & Income Bracket"},
        "xaxis": {"title": "DTI Ratio Bucket"},
        "yaxis": {"title": "Default Rate (%)"},
        "barmode": "group",
        "legend": {"title": {"text": "Annual Income"}},
        "margin": {"t": 60},
    }
    return traces, layout


def build_time_series(scored_df):
    """Tab 5: Loan volume & default rate over time (if issue_d available)."""
    if scored_df is None or "issue_d" not in scored_df.columns:
        return None, None

    df = scored_df.copy()
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=["issue_d"])
    if len(df) < 100:
        return None, None

    df["month"] = df["issue_d"].dt.to_period("M").astype(str)
    monthly = df.groupby("month").agg(
        volume=("default", "count"),
        default_rate=("default", "mean"),
    ).reset_index()

    traces = [
        {
            "x": monthly["month"].tolist(),
            "y": monthly["volume"].tolist(),
            "type": "bar", "name": "Loan Volume",
            "marker": {"color": BLUE, "opacity": 0.6},
            "yaxis": "y",
        },
        {
            "x": monthly["month"].tolist(),
            "y": [round(r * 100, 1) for r in monthly["default_rate"]],
            "type": "scatter", "mode": "lines+markers",
            "name": "Default Rate (%)",
            "line": {"color": RED, "width": 2},
            "yaxis": "y2",
        },
    ]
    layout = {
        "title": {"text": "Loan Volume & Default Rate Over Time"},
        "xaxis": {"title": "Issue Month", "tickangle": -45},
        "yaxis": {"title": "Loan Volume", "side": "left"},
        "yaxis2": {
            "title": "Default Rate (%)", "side": "right",
            "overlaying": "y", "showgrid": False,
        },
        "legend": {"x": 0.01, "y": 0.99},
        "margin": {"t": 60, "b": 80},
    }
    return traces, layout


def generate_html(tabs):
    """Build the full dashboard HTML with client-side tab navigation."""
    tab_buttons = []
    tab_divs = []

    for i, (label, traces, layout) in enumerate(tabs):
        active = "active" if i == 0 else ""
        display = "block" if i == 0 else "none"
        tab_buttons.append(
            f'<button class="tab-btn {active}" onclick="showTab({i})">{label}</button>'
        )
        # traces can be a single dict or a list
        if isinstance(traces, dict):
            traces = [traces]
        tab_divs.append(f"""
    <div class="tab-content" id="tab-{i}" style="display:{display}">
      <div id="chart-{i}" style="width:100%;height:500px;"></div>
      <script>
        Plotly.newPlot('chart-{i}', {json.dumps(traces)}, {json.dumps(layout)},
          {{responsive: true, displayModeBar: false}});
      </script>
    </div>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Consumer Credit Risk Analytics — Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f8f9fa; color: #333; }}
  .header {{ background: #2c3e50; color: white; padding: 20px 30px; }}
  .header h1 {{ font-size: 1.5rem; font-weight: 600; }}
  .header p {{ font-size: 0.85rem; opacity: 0.8; margin-top: 4px; }}
  .tab-bar {{ display: flex; gap: 0; background: #eee; border-bottom: 2px solid #ddd;
              padding: 0 20px; overflow-x: auto; }}
  .tab-btn {{ padding: 12px 20px; border: none; background: transparent; cursor: pointer;
              font-size: 0.85rem; color: #666; white-space: nowrap;
              border-bottom: 3px solid transparent; transition: all 0.15s; }}
  .tab-btn:hover {{ color: #333; background: #f0f0f0; }}
  .tab-btn.active {{ color: #2c3e50; font-weight: 600; border-bottom-color: #4C72B0; }}
  .tab-content {{ padding: 20px; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
</style>
</head>
<body>
<div class="header">
  <div class="container">
    <h1>Consumer Credit Risk Analytics</h1>
    <p>Lending Club loan default prediction &mdash; Logistic Regression vs. XGBoost</p>
  </div>
</div>
<div class="container">
  <div class="tab-bar">
    {''.join(tab_buttons)}
  </div>
  {''.join(tab_divs)}
</div>
<script>
function showTab(idx) {{
  document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + idx).style.display = 'block';
  document.querySelectorAll('.tab-btn')[idx].classList.add('active');
  // Relayout for responsive sizing
  var chartEl = document.getElementById('chart-' + idx);
  if (chartEl && chartEl.data) Plotly.Plots.resize(chartEl);
}}
</script>
</body>
</html>"""


def main():
    print("Building dashboard...")

    tabs = []

    # Tab 1: Grade default rates
    grade_df = read_csv(OUT, "grade_default_rates.csv")
    if grade_df is not None:
        trace, layout = build_grade_chart(grade_df)
        tabs.append(("Default by Grade", trace, layout))
    else:
        print("  WARN: grade_default_rates.csv not found, skipping tab")

    # Tab 2: ROC comparison
    roc_df = read_csv(OUT, "roc_data.csv")
    if roc_df is not None:
        traces, layout = build_roc_chart(roc_df)
        tabs.append(("ROC Comparison", traces, layout))
    else:
        print("  WARN: roc_data.csv not found, skipping tab")

    # Tab 3: Feature importances
    imp_df = read_csv(OUT, "feature_importances.csv")
    if imp_df is not None:
        imp_df = imp_df.set_index(imp_df.columns[0])
        trace, layout = build_importance_chart(imp_df)
        tabs.append(("Feature Importance", trace, layout))
    else:
        print("  WARN: feature_importances.csv not found, skipping tab")

    # Tab 4: DTI x income
    dti_df = read_csv(OUT, "dti_income_default_rates.csv")
    if dti_df is not None:
        traces, layout = build_dti_income_chart(dti_df)
        tabs.append(("DTI vs Income", traces, layout))
    else:
        print("  WARN: dti_income_default_rates.csv not found, skipping tab")

    # Tab 5: Time series
    scored_df = read_csv(DATA, "lending_club_scored.csv")
    traces, layout = build_time_series(scored_df)
    if traces is not None:
        tabs.append(("Volume Over Time", traces, layout))
    else:
        print("  WARN: issue_d not available or too few rows, skipping time-series tab")

    if not tabs:
        print("ERROR: no data available to build dashboard. Run analysis.py first.")
        return

    html = generate_html(tabs)
    with open(DASHBOARD, "w") as f:
        f.write(html)
    size_kb = os.path.getsize(DASHBOARD) / 1024
    print(f"Dashboard saved to {DASHBOARD} ({size_kb:.0f} KB, {len(tabs)} tabs)")


if __name__ == "__main__":
    main()
