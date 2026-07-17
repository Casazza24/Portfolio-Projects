"""Build an interactive Plotly dashboard (dashboard.html) for the NYC subway
ridership-drivers analysis. Standalone: reads the three raw CSVs, replicates the
notebook's feature engineering + OLS fit, and writes one self-contained HTML file
with tabbed views. No Dash server needed — client-side interactivity only.

    python3 build_dashboard.py
"""

import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from pandas.tseries.holiday import USFederalHolidayCalendar

DATA = "data"
OUT = "dashboard.html"

BLUE, ORANGE, RED, GREEN, TEAL = "#4C72B0", "#DD8452", "#E45756", "#2C9E6B", "#72B7B2"


def load():
    """Reproduce the notebook's ridership+weather+delay frames and calendar features."""
    rider = pd.read_csv(f"{DATA}/mta_ridership_raw.csv", parse_dates=["date"])
    rider["date"] = rider["date"].dt.normalize()
    rider = rider[["date", "count"]].rename(columns={"count": "ridership"})

    wx_long = pd.read_csv(f"{DATA}/weather_raw.csv", parse_dates=["date"])
    wx_long["date"] = wx_long["date"].dt.normalize()
    wx = wx_long.pivot_table(index="date", columns="datatype", values="value").reset_index()
    for col in ("TMAX", "TMIN"):
        if col in wx:
            wx[col + "_F"] = wx[col] * 9 / 5 + 32

    df = rider.merge(wx, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["rain_day"] = df["PRCP"] > 0

    def temp_band(f):
        if f < 55: return "Cold (<55F)"
        if f <= 75: return "Mild (55-75F)"
        return "Warm (>75F)"
    df["temp_band"] = df["TMAX_F"].apply(temp_band)

    df["weekday"] = df["date"].dt.day_name()
    df["weekday_num"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["weekday_num"] >= 5
    hol = USFederalHolidayCalendar().holidays(start=df["date"].min(), end=df["date"].max())
    df["holiday"] = df["date"].isin(hol)

    dl = pd.read_csv(f"{DATA}/mta_delays_monthly_raw.csv", parse_dates=["month"])
    return df, dl


def fmt_riders(v):
    return f"{v/1e6:.2f}M"


def fig_dow(df):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    m = df.groupby("weekday")["ridership"].mean().reindex(order)
    colors = [BLUE if d not in ("Saturday", "Sunday") else ORANGE for d in order]
    wk = df.loc[~df["is_weekend"], "ridership"].mean()
    we = df.loc[df["is_weekend"], "ridership"].mean()
    fig = go.Figure(go.Bar(
        x=order, y=m.values, marker_color=colors,
        text=[fmt_riders(v) for v in m.values], textposition="outside",
        hovertemplate="%{x}<br>%{y:,.0f} riders<extra></extra>"))
    fig.update_layout(
        title=f"Mean ridership by day of week — weekend runs {(we-wk)/wk*100:+.0f}% vs weekdays",
        yaxis_title="Mean daily riders", margin=dict(t=60))
    return fig


def fig_month(df):
    m = df.groupby("month")["ridership"].mean().sort_index()
    labels = [pd.Timestamp(2025, i, 1).strftime("%b") for i in m.index]
    swing = (m.max() - m.min()) / m.min() * 100
    fig = go.Figure(go.Scatter(
        x=labels, y=m.values, mode="lines+markers", line_color=BLUE,
        hovertemplate="%{x}<br>%{y:,.0f} riders<extra></extra>"))
    fig.update_layout(
        title=f"Mean ridership by calendar month — {swing:.0f}% peak-to-trough (Apr peak, Jan trough)",
        yaxis_title="Mean daily riders", margin=dict(t=60))
    return fig


def fig_rain(df):
    """Naive vs weekday-controlled rain effect — surfaces the day-of-week confound."""
    wd = df[~df["is_weekend"]]
    all_m = df.groupby("rain_day")["ridership"].mean()
    wd_m = wd.groupby("rain_day")["ridership"].mean()
    all_pct = (all_m[True] - all_m[False]) / all_m[False] * 100
    wd_pct = (wd_m[True] - wd_m[False]) / wd_m[False] * 100
    cats = ["Dry", "Rain"]
    fig = go.Figure()
    fig.add_bar(name=f"All days (naive: {all_pct:+.1f}%)", x=cats,
                y=[all_m[False], all_m[True]], marker_color=BLUE,
                hovertemplate="%{x} (all days)<br>%{y:,.0f}<extra></extra>")
    fig.add_bar(name=f"Weekdays only (controlled: {wd_pct:+.1f}%)", x=cats,
                y=[wd_m[False], wd_m[True]], marker_color=TEAL,
                hovertemplate="%{x} (weekdays)<br>%{y:,.0f}<extra></extra>")
    fig.update_layout(
        barmode="group",
        title="Rain effect shrinks once day-of-week is controlled (rain days skew toward weekends)",
        yaxis_title="Mean daily riders", margin=dict(t=60))
    return fig


def fig_temp(df):
    order = ["Cold (<55F)", "Mild (55-75F)", "Warm (>75F)"]
    all_m = df.groupby("temp_band")["ridership"].mean().reindex(order)
    wd_m = df[~df["is_weekend"]].groupby("temp_band")["ridership"].mean().reindex(order)
    fig = go.Figure()
    fig.add_bar(name="All days", x=order, y=all_m.values, marker_color=BLUE,
                hovertemplate="%{x} (all)<br>%{y:,.0f}<extra></extra>")
    fig.add_bar(name="Weekdays only", x=order, y=wd_m.values, marker_color=TEAL,
                hovertemplate="%{x} (weekdays)<br>%{y:,.0f}<extra></extra>")
    fig.update_layout(
        barmode="group",
        title="Ridership by temperature band — warm-day lift is mostly season, not heat",
        yaxis_title="Mean daily riders", margin=dict(t=60))
    return fig


LABELS = {"rain_day[T.True]": "Rain day", "holiday[T.True]": "Holiday",
          "TMAX_F": "Max temp (per °F)"}


def clean_label(name):
    if name in LABELS:
        return LABELS[name]
    if name.startswith("C(weekday)[T."):
        return name[len("C(weekday)[T."):-1] + " (vs Fri)"
    if name.startswith("C(month)[T."):
        n = int(name[len("C(month)[T."):-1])
        return pd.Timestamp(2025, n, 1).strftime("%b") + " (vs Jan)"
    return name


def fig_regression(df):
    """OLS coefficient tornado — the centerpiece ranking of ridership drivers."""
    model = smf.ols("ridership ~ C(weekday) + C(month) + rain_day + TMAX_F + holiday",
                    data=df).fit()
    coef = model.params.drop("Intercept")
    pval = model.pvalues.drop("Intercept")
    rank = pd.DataFrame({"eff": coef, "p": pval})
    rank["abs"] = rank["eff"].abs()
    rank = rank.sort_values("abs")  # ascending so largest lands on top of a horizontal bar
    labels = [clean_label(i) for i in rank.index]
    stars = pd.cut(rank["p"], [-1, .001, .01, .05, 1], labels=["***", "**", "*", "ns"])
    text = [f"{v:+,.0f} {s}" for v, s in zip(rank["eff"], stars)]
    colors = [GREEN if v > 0 else RED for v in rank["eff"]]
    fig = go.Figure(go.Bar(
        x=rank["eff"], y=labels, orientation="h", marker_color=colors,
        text=text, textposition="outside",
        customdata=rank["p"],
        hovertemplate="%{y}<br>%{x:+,.0f} riders/day<br>p=%{customdata:.3g}<extra></extra>"))
    fig.update_layout(
        title=f"Ridership drivers ranked by daily-rider effect (OLS, R²={model.rsquared:.2f}, n={int(model.nobs)})",
        xaxis_title="Effect on daily riders vs. baseline (Fri / Jan / dry)",
        margin=dict(t=60, l=140), height=560)
    return fig


def fig_delays(dl):
    """Top offender lines with stacked delay-cause breakdown."""
    by_line = dl.groupby("line")["incidents"].sum().sort_values(ascending=False)
    top_lines = by_line.head(12).index.tolist()
    by_cat = dl.groupby("reporting_category")["incidents"].sum().sort_values(ascending=False)
    top_cats = [c for c in by_cat.head(6).index if isinstance(c, str)]
    pivot = (dl[dl["line"].isin(top_lines)]
             .pivot_table(index="line", columns="reporting_category",
                          values="incidents", aggfunc="sum", fill_value=0)
             .reindex(top_lines))
    order = top_lines[::-1]  # smallest total on top for horizontal stack
    fig = go.Figure()
    for cat in top_cats:
        if cat in pivot.columns:
            fig.add_bar(name=cat, x=pivot.loc[order, cat].values,
                        y=[str(l) for l in order], orientation="h",
                        hovertemplate=f"Line %{{y}} — {cat}<br>%{{x:,.0f}} incidents<extra></extra>")
    fig.update_layout(
        barmode="stack",
        title="Delay incidents by line (top 12, trailing 12mo) — worst: 6, 2, N, A, F",
        xaxis_title="Incidents", yaxis_title="Line", legend_title="Cause",
        margin=dict(t=60), height=560)
    return fig


TABS = [
    ("Day of week", fig_dow),
    ("Seasonal", fig_month),
    ("Rain effect", fig_rain),
    ("Temperature", fig_temp),
    ("Driver ranking", fig_regression),
    ("Delays by line", fig_delays),
]


def build(df, dl):
    panels, buttons = [], []
    for i, (name, fn) in enumerate(TABS):
        fig = fn(dl if fn is fig_delays else df)
        fig.update_layout(template="plotly_white", font=dict(family="system-ui, sans-serif"),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        # ponytail: "cdn" over True — loads plotly.js from unpkg once instead of
        # inlining ~4.7MB into every build. Shrinks dashboard.html ~4.9MB -> ~50KB.
        # Tradeoff: needs internet to open. Fine for a hosted GitHub portfolio piece.
        div = fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False), div_id=f"tab{i}")
        panels.append(f'<div class="panel" id="panel{i}" '
                      f'style="display:{"block" if i == 0 else "none"}">'
                      f'<div class="card">{div}</div></div>')
        buttons.append(f'<button class="tab{" active" if i == 0 else ""}" '
                       f'onclick="show({i})">{name}</button>')
    span = f"{df['date'].min().date()} to {df['date'].max().date()}"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>NYC Subway Ridership Drivers</title>
<style>
 :root{{--brand:#4C72B0;--brand-dark:#33507e;--ink:#1f2933;--muted:#64748b;--line:#e2e8f0;--bg:#eef1f6;--card:#ffffff}}
 *{{box-sizing:border-box}}
 body{{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;margin:0;background:var(--bg);color:var(--ink);line-height:1.5}}
 .wrap{{max-width:1080px;margin:0 auto;padding:0 20px 56px}}
 header{{background:linear-gradient(135deg,var(--brand-dark),var(--brand));color:#fff;padding:30px 0 28px;box-shadow:0 2px 12px rgba(15,23,42,.12)}}
 header .wrap{{padding-bottom:0}}
 header h1{{margin:0;font-size:25px;font-weight:650;letter-spacing:-.2px}}
 header p{{margin:8px 0 0;color:#dbe3f0;font-size:15px;max-width:70ch}}
 .tabs{{display:flex;flex-wrap:wrap;gap:8px;margin:26px 0 18px}}
 .tab{{padding:9px 16px;border:1px solid var(--line);background:var(--card);color:var(--muted);border-radius:999px;cursor:pointer;font-size:14px;font-weight:550;transition:all .15s ease}}
 .tab:hover{{border-color:var(--brand);color:var(--brand);box-shadow:0 1px 4px rgba(76,114,176,.18)}}
 .tab.active{{background:var(--brand);color:#fff;border-color:var(--brand);box-shadow:0 2px 8px rgba(76,114,176,.35)}}
 .card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;box-shadow:0 4px 20px rgba(15,23,42,.06)}}
 footer{{margin-top:22px;color:var(--muted);font-size:12.5px;text-align:center}}
</style></head><body>
<header><div class="wrap"><h1>NYC Transit Efficiency Analysis</h1>
<p>What actually drives subway ridership — daily ridership vs. weather, calendar, and delays, {span}. Calendar structure dominates; weather only trims the edges.</p></div></header>
<div class="wrap">
<div class="tabs">{''.join(buttons)}</div>
{''.join(panels)}
<footer>Data: MTA daily ridership &amp; delays, NOAA Central Park weather. Built with Plotly.</footer>
</div>
<script>
function show(n){{
 document.querySelectorAll('.panel').forEach((p,i)=>p.style.display=i==n?'block':'none');
 document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',i==n));
 window.dispatchEvent(new Event('resize')); // let hidden Plotly divs size correctly on reveal
}}
</script></body></html>"""


if __name__ == "__main__":
    df, dl = load()
    html = build(df, dl)
    with open(OUT, "w") as f:
        f.write(html)
    print(f"Wrote {OUT} ({len(html):,} bytes, {len(TABS)} tabs, {len(df)} ridership days)")
