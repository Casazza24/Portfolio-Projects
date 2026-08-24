import pandas as pd
import numpy as np
import pytest
from src.sentiment_factor import build_daily_factor_return, construct_sentiment_factor


@pytest.fixture
def daily_z_scores():
    """20 tickers, 10 days of z-scored sentiment."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-02", periods=10)
    tickers = [f"T{i:02d}" for i in range(20)]
    rows = []
    for d in dates:
        z_vals = np.random.normal(0, 1, len(tickers))
        for t, z in zip(tickers, z_vals):
            rows.append({"date": d, "ticker": t, "z_sentiment": z})
    return pd.DataFrame(rows)


@pytest.fixture
def price_returns():
    """Daily returns for 20 tickers, 10 days."""
    np.random.seed(99)
    dates = pd.bdate_range("2024-01-02", periods=10)
    tickers = [f"T{i:02d}" for i in range(20)]
    data = np.random.normal(0.0005, 0.015, (len(dates), len(tickers)))
    return pd.DataFrame(data, index=dates, columns=tickers)


class TestBuildDailyFactorReturn:
    def test_returns_scalar(self, daily_z_scores, price_returns):
        one_day = daily_z_scores[daily_z_scores["date"] == daily_z_scores["date"].iloc[0]]
        one_day_returns = price_returns.iloc[0]
        result = build_daily_factor_return(one_day, one_day_returns)
        assert isinstance(result, float)

    def test_long_minus_short(self, daily_z_scores, price_returns):
        one_day = daily_z_scores[daily_z_scores["date"] == daily_z_scores["date"].iloc[0]]
        one_day_returns = price_returns.iloc[0]
        result = build_daily_factor_return(one_day, one_day_returns)
        # With 20 tickers and quintiles, top/bottom 4 each
        # Result should be finite
        assert np.isfinite(result)


class TestConstructSentimentFactor:
    def test_returns_series(self, daily_z_scores, price_returns):
        factor = construct_sentiment_factor(daily_z_scores, price_returns)
        assert isinstance(factor, pd.Series)
        assert factor.name == "SENT"

    def test_length_matches_dates(self, daily_z_scores, price_returns):
        factor = construct_sentiment_factor(daily_z_scores, price_returns)
        assert len(factor) <= len(price_returns)
        assert len(factor) > 0

    def test_values_are_finite(self, daily_z_scores, price_returns):
        factor = construct_sentiment_factor(daily_z_scores, price_returns)
        assert factor.notna().all()
        assert np.isfinite(factor.values).all()
