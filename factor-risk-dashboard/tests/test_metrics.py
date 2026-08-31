import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    var_95,
    cvar_95,
    tracking_error,
    information_ratio,
    drawdown_series,
    rolling_beta,
)


@pytest.fixture
def daily_returns():
    np.random.seed(42)
    return pd.Series(
        np.random.normal(0.0004, 0.01, 252),
        index=pd.bdate_range("2023-01-01", periods=252),
        name="portfolio",
    )


@pytest.fixture
def benchmark_returns():
    np.random.seed(99)
    return pd.Series(
        np.random.normal(0.0003, 0.009, 252),
        index=pd.bdate_range("2023-01-01", periods=252),
        name="benchmark",
    )


class TestAnnualizedReturn:
    def test_positive_mean(self, daily_returns):
        result = annualized_return(daily_returns)
        assert isinstance(result, float)
        expected = daily_returns.mean() * 252
        assert abs(result - expected) < 1e-10

    def test_zero_returns(self):
        zeros = pd.Series(np.zeros(100))
        assert annualized_return(zeros) == 0.0


class TestAnnualizedVolatility:
    def test_known_vol(self, daily_returns):
        result = annualized_volatility(daily_returns)
        expected = daily_returns.std() * np.sqrt(252)
        assert abs(result - expected) < 1e-10

    def test_constant_returns(self):
        constant = pd.Series(np.full(100, 0.001))
        assert annualized_volatility(constant) < 1e-10


class TestSharpeRatio:
    def test_positive_sharpe(self, daily_returns):
        result = sharpe_ratio(daily_returns)
        assert isinstance(result, float)

    def test_zero_vol_returns_zero(self):
        constant = pd.Series(np.zeros(100))
        assert sharpe_ratio(constant) == 0.0


class TestSortinoRatio:
    def test_positive_sortino(self, daily_returns):
        result = sortino_ratio(daily_returns)
        assert isinstance(result, float)

    def test_sortino_geq_sharpe_for_right_skew(self):
        np.random.seed(7)
        right_skew = pd.Series(np.abs(np.random.normal(0.001, 0.01, 252)))
        sortino = sortino_ratio(right_skew)
        sharpe = sharpe_ratio(right_skew)
        assert sortino >= sharpe


class TestMaxDrawdown:
    def test_negative_value(self, daily_returns):
        result = max_drawdown(daily_returns)
        assert result <= 0

    def test_always_positive_returns(self):
        always_up = pd.Series(np.full(100, 0.01))
        assert max_drawdown(always_up) == 0.0

    def test_known_drawdown(self):
        # Price goes 100 -> 120 -> 60 -> 90: drawdown from 120 to 60 = -50%
        returns = pd.Series([0.20, -0.50, 0.50])
        result = max_drawdown(returns)
        assert abs(result - (-0.50)) < 1e-10


class TestVaR:
    def test_var_negative(self, daily_returns):
        result = var_95(daily_returns)
        assert result < 0

    def test_var_is_5th_percentile(self, daily_returns):
        result = var_95(daily_returns)
        expected = np.percentile(daily_returns, 5)
        assert abs(result - expected) < 1e-10


class TestCVaR:
    def test_cvar_worse_than_var(self, daily_returns):
        v = var_95(daily_returns)
        cv = cvar_95(daily_returns)
        assert cv <= v

    def test_cvar_is_tail_mean(self, daily_returns):
        threshold = var_95(daily_returns)
        tail = daily_returns[daily_returns <= threshold]
        expected = tail.mean()
        assert abs(cvar_95(daily_returns) - expected) < 1e-10


class TestTrackingError:
    def test_positive(self, daily_returns, benchmark_returns):
        result = tracking_error(daily_returns, benchmark_returns)
        assert result > 0

    def test_identical_returns_zero_te(self, daily_returns):
        result = tracking_error(daily_returns, daily_returns)
        assert abs(result) < 1e-10


class TestInformationRatio:
    def test_returns_float(self, daily_returns, benchmark_returns):
        result = information_ratio(daily_returns, benchmark_returns)
        assert isinstance(result, float)

    def test_identical_returns_zero_ir(self, daily_returns):
        result = information_ratio(daily_returns, daily_returns)
        assert result == 0.0


class TestDrawdownSeries:
    def test_all_nonpositive(self, daily_returns):
        dd = drawdown_series(daily_returns)
        assert (dd <= 0).all()

    def test_starts_at_zero_or_negative(self, daily_returns):
        dd = drawdown_series(daily_returns)
        assert dd.iloc[0] <= 0


class TestRollingBeta:
    def test_output_length(self, daily_returns, benchmark_returns):
        window = 60
        rb = rolling_beta(daily_returns, benchmark_returns, window=window)
        assert len(rb) > 0
        assert len(rb) <= len(daily_returns) - window + 1
