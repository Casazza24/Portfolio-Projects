import pandas as pd
import numpy as np


def compute_portfolio_returns(
    returns: pd.DataFrame, weights: dict[str, float]
) -> pd.Series:
    """Compute weighted portfolio daily returns from constituent returns.

    Args:
        returns: DataFrame of daily returns with ticker columns.
        weights: {ticker: weight} mapping, weights should sum to 1.

    Returns:
        Series of daily portfolio returns.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    aligned = returns[tickers].dropna()
    portfolio_returns = aligned.values @ w
    return pd.Series(portfolio_returns, index=aligned.index, name="portfolio")


def align_series(*series: pd.Series | pd.DataFrame) -> list[pd.Series | pd.DataFrame]:
    """Align multiple time series to their common date index."""
    common_index = series[0].index
    for s in series[1:]:
        common_index = common_index.intersection(s.index)
    common_index = common_index.sort_values()
    return [s.loc[common_index] for s in series]
