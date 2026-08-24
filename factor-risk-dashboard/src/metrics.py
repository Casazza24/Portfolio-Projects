import numpy as np
import pandas as pd

from config.settings import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE_ANNUAL

RF_DAILY = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR


def annualized_return(daily_returns: pd.Series) -> float:
    return float(daily_returns.mean() * TRADING_DAYS_PER_YEAR)


def annualized_volatility(daily_returns: pd.Series) -> float:
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_returns: pd.Series) -> float:
    excess = daily_returns - RF_DAILY
    ann_excess = excess.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if ann_vol == 0:
        return 0.0
    return float(ann_excess / ann_vol)


def sortino_ratio(daily_returns: pd.Series) -> float:
    excess = daily_returns - RF_DAILY
    ann_excess = excess.mean() * TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    downside_dev = np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside_dev == 0:
        return 0.0
    return float(ann_excess / downside_dev)


def max_drawdown(daily_returns: pd.Series) -> float:
    cumulative = (1 + daily_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def var_95(daily_returns: pd.Series) -> float:
    """Historical VaR at 95% confidence (5th percentile of returns)."""
    return float(np.percentile(daily_returns.dropna(), 5))


def cvar_95(daily_returns: pd.Series) -> float:
    """Conditional VaR (Expected Shortfall) at 95% confidence."""
    threshold = var_95(daily_returns)
    tail = daily_returns[daily_returns <= threshold]
    return float(tail.mean())


def tracking_error(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    active_returns = portfolio_returns - benchmark_returns
    return float(active_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def information_ratio(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    active_returns = portfolio_returns - benchmark_returns
    ann_active = active_returns.mean() * TRADING_DAYS_PER_YEAR
    te = active_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if te == 0:
        return 0.0
    return float(ann_active / te)


def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    cumulative = (1 + daily_returns).cumprod()
    peak = cumulative.cummax()
    return (cumulative - peak) / peak


def rolling_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    cov = portfolio_returns.rolling(window).cov(benchmark_returns)
    var = benchmark_returns.rolling(window).var()
    return (cov / var).dropna()


def regime_conditioned_risk(
    portfolio_returns: pd.Series,
    sent_factor: pd.Series,
) -> dict:
    """Compute VaR and CVaR split by high/low sentiment regimes."""
    aligned_port, aligned_sent = portfolio_returns.align(sent_factor, join="inner")
    median_sent = aligned_sent.median()

    high_mask = aligned_sent >= median_sent
    low_mask = aligned_sent < median_sent

    high_returns = aligned_port[high_mask]
    low_returns = aligned_port[low_mask]

    return {
        "high_sentiment_var": var_95(high_returns) if len(high_returns) > 20 else None,
        "high_sentiment_cvar": cvar_95(high_returns) if len(high_returns) > 20 else None,
        "low_sentiment_var": var_95(low_returns) if len(low_returns) > 20 else None,
        "low_sentiment_cvar": cvar_95(low_returns) if len(low_returns) > 20 else None,
        "n_high_days": int(high_mask.sum()),
        "n_low_days": int(low_mask.sum()),
    }


def sentiment_stress_impact(
    sent_beta: float,
    sigma_shock: float,
    sent_factor_vol: float,
) -> float:
    daily_vol = sent_factor_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    return sent_beta * sigma_shock * daily_vol


def compute_all_kpis(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    """Compute all dashboard KPIs and return as a dict."""
    return {
        "Annualized Alpha": alpha,
        "Beta": beta,
        "Sharpe Ratio": sharpe_ratio(portfolio_returns),
        "Sortino Ratio": sortino_ratio(portfolio_returns),
        "Max Drawdown": max_drawdown(portfolio_returns),
        "VaR (95%)": var_95(portfolio_returns),
        "CVaR (95%)": cvar_95(portfolio_returns),
        "Information Ratio": information_ratio(portfolio_returns, benchmark_returns),
    }
