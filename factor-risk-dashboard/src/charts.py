import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from scipy import stats


COLORS = {
    "portfolio": "#636EFA",
    "benchmark": "#EF553B",
    "positive": "#00CC96",
    "negative": "#EF553B",
    "neutral": "#AB63FA",
}

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    margin=dict(l=40, r=20, t=40, b=40),
    height=380,
    font=dict(size=12),
    hovermode="x unified",
)


def rolling_beta_chart(rolling_beta: pd.Series, window: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_beta.index,
        y=rolling_beta.values,
        mode="lines",
        name=f"{window}-day Rolling Beta",
        line=dict(color=COLORS["portfolio"], width=2),
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.6)
    fig.update_layout(
        title=f"Rolling Beta ({window}-day)",
        yaxis_title="Beta",
        **LAYOUT_DEFAULTS,
    )
    return fig


def cumulative_return_chart(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_name: str = "Portfolio",
    benchmark_name: str = "Benchmark",
) -> go.Figure:
    port_cum = (1 + portfolio_returns).cumprod() * 100
    bench_cum = (1 + benchmark_returns).cumprod() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port_cum.index, y=port_cum.values,
        mode="lines", name=portfolio_name,
        line=dict(color=COLORS["portfolio"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=bench_cum.index, y=bench_cum.values,
        mode="lines", name=benchmark_name,
        line=dict(color=COLORS["benchmark"], width=2),
    ))
    fig.update_layout(
        title="Cumulative Return (rebased to 100)",
        yaxis_title="Value",
        **LAYOUT_DEFAULTS,
    )
    return fig


def return_distribution_chart(
    daily_returns: pd.Series, var_95: float, cvar_95: float
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=daily_returns.values,
        nbinsx=60,
        name="Daily Returns",
        marker_color=COLORS["portfolio"],
        opacity=0.7,
        histnorm="probability density",
    ))

    x_range = np.linspace(daily_returns.min(), daily_returns.max(), 200)
    mu, sigma = daily_returns.mean(), daily_returns.std()
    normal_curve = stats.norm.pdf(x_range, mu, sigma)
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_curve,
        mode="lines", name="Normal Distribution",
        line=dict(color="gray", width=2, dash="dash"),
    ))

    fig.add_vline(x=var_95, line_dash="dot", line_color=COLORS["negative"],
                  annotation_text=f"VaR 95%: {var_95:.2%}")
    fig.add_vline(x=cvar_95, line_dash="dot", line_color="#FFA15A",
                  annotation_text=f"CVaR 95%: {cvar_95:.2%}")

    fig.update_layout(
        title="Return Distribution",
        xaxis_title="Daily Return",
        yaxis_title="Density",
        **LAYOUT_DEFAULTS,
    )
    return fig


def factor_exposure_chart(factor_betas: dict[str, float]) -> go.Figure:
    factors = list(factor_betas.keys())
    values = list(factor_betas.values())
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=factors,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="gray", line_width=1)
    fig.update_layout(
        title="Factor Exposures (FF5 + Momentum + Sentiment)",
        xaxis_title="Factor Beta",
        **LAYOUT_DEFAULTS,
    )
    return fig


def drawdown_chart(drawdown: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill="tozeroy",
        mode="lines",
        name="Drawdown",
        line=dict(color=COLORS["negative"], width=1),
        fillcolor="rgba(239, 85, 59, 0.2)",
    ))
    fig.update_layout(
        title="Drawdown from Peak",
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        **LAYOUT_DEFAULTS,
    )
    return fig


MULTI_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]


def multi_cumulative_return_chart(
    portfolio_series: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    benchmark_name: str = "Benchmark",
) -> go.Figure:
    fig = go.Figure()

    for i, (name, returns) in enumerate(portfolio_series.items()):
        cum = (1 + returns).cumprod() * 100
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            mode="lines", name=name,
            line=dict(color=MULTI_COLORS[i % len(MULTI_COLORS)], width=2),
        ))

    bench_cum = (1 + benchmark_returns).cumprod() * 100
    fig.add_trace(go.Scatter(
        x=bench_cum.index, y=bench_cum.values,
        mode="lines", name=benchmark_name,
        line=dict(color="gray", width=2, dash="dash"),
    ))

    fig.update_layout(
        title="Cumulative Returns — Portfolio Comparison (rebased to 100)",
        yaxis_title="Value",
        **LAYOUT_DEFAULTS,
        height=450,
    )
    return fig


def multi_drawdown_chart(
    drawdown_series_dict: dict[str, pd.Series],
) -> go.Figure:
    fig = go.Figure()
    for i, (name, dd) in enumerate(drawdown_series_dict.items()):
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", name=name,
            line=dict(color=MULTI_COLORS[i % len(MULTI_COLORS)], width=1.5),
        ))
    fig.update_layout(
        title="Drawdown Comparison",
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        **LAYOUT_DEFAULTS,
        height=400,
    )
    return fig


def correlation_heatmap(returns_df: pd.DataFrame) -> go.Figure:
    corr = returns_df.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Correlation Matrix",
        height=400,
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig
