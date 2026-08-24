import pandas as pd
import numpy as np

from config.sentiment import QUINTILE_MIN_TICKERS


def build_daily_factor_return(
    daily_z: pd.DataFrame,
    daily_returns: pd.Series,
) -> float:
    merged = daily_z.set_index("ticker")[["z_sentiment"]].join(
        daily_returns.rename("ret"), how="inner"
    ).dropna()

    if len(merged) < QUINTILE_MIN_TICKERS:
        return 0.0

    quintile_size = len(merged) // 5
    sorted_df = merged.sort_values("z_sentiment")
    bottom = sorted_df.head(quintile_size)
    top = sorted_df.tail(quintile_size)

    long_ret = top["ret"].mean()
    short_ret = bottom["ret"].mean()
    return float(long_ret - short_ret)


def construct_sentiment_factor(
    z_scores: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    dates = sorted(returns.index)
    z_dates = sorted(z_scores["date"].unique())
    factor_returns = []
    factor_dates = []

    for i in range(1, len(dates)):
        current_date = dates[i]
        prev_date = dates[i - 1]

        # Use prior-day sentiment for ranking (look-ahead bias protection)
        prev_z = z_scores[z_scores["date"] == prev_date]
        if prev_z.empty:
            closest_idx = np.searchsorted(z_dates, prev_date, side="right") - 1
            if closest_idx >= 0:
                prev_z = z_scores[z_scores["date"] == z_dates[closest_idx]]
            if prev_z.empty:
                continue

        if current_date not in returns.index:
            continue

        day_returns = returns.loc[current_date]
        factor_ret = build_daily_factor_return(prev_z, day_returns)
        factor_returns.append(factor_ret)
        factor_dates.append(current_date)

    return pd.Series(factor_returns, index=factor_dates, name="SENT")
