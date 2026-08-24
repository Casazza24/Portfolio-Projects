# tests/test_validation.py
import numpy as np
import pandas as pd
import pytest
from src.validation import (
    incremental_r_squared,
    factor_orthogonality_test,
    factor_significance_test,
)


@pytest.fixture
def factor_data():
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2022-01-01", periods=n)
    return pd.DataFrame({
        "Mkt-RF": np.random.normal(0.0003, 0.01, n),
        "SMB": np.random.normal(0.0001, 0.005, n),
        "HML": np.random.normal(0.0001, 0.004, n),
        "RMW": np.random.normal(0.00005, 0.003, n),
        "CMA": np.random.normal(0.00005, 0.003, n),
        "MOM": np.random.normal(0.0002, 0.006, n),
        "SENT": np.random.normal(0.0001, 0.004, n),
        "RF": np.full(n, 0.0002),
    }, index=dates)


@pytest.fixture
def portfolio_returns(factor_data):
    np.random.seed(7)
    mkt = factor_data["Mkt-RF"].values
    sent = factor_data["SENT"].values
    noise = np.random.normal(0, 0.005, len(mkt))
    returns = 0.0001 + 1.1 * mkt + 0.3 * sent + noise
    return pd.Series(returns, index=factor_data.index, name="portfolio")


class TestIncrementalRSquared:
    def test_returns_positive_delta(self, portfolio_returns, factor_data):
        result = incremental_r_squared(portfolio_returns, factor_data)
        assert "r2_without_sent" in result
        assert "r2_with_sent" in result
        assert "delta_r2" in result
        assert result["delta_r2"] >= 0

    def test_f_test_result(self, portfolio_returns, factor_data):
        result = incremental_r_squared(portfolio_returns, factor_data)
        assert "f_stat" in result
        assert "f_pvalue" in result


class TestOrthogonality:
    def test_returns_alpha(self, factor_data):
        result = factor_orthogonality_test(factor_data)
        assert "alpha" in result
        assert "alpha_tstat" in result
        assert "r_squared" in result


class TestSignificance:
    def test_returns_results_per_portfolio(self, factor_data):
        np.random.seed(99)
        test_portfolios = {}
        for i in range(5):
            noise = np.random.normal(0, 0.01, len(factor_data))
            test_portfolios[f"port_{i}"] = pd.Series(
                0.0001 + 1.0 * factor_data["Mkt-RF"].values + noise,
                index=factor_data.index,
            )
        result = factor_significance_test(test_portfolios, factor_data)
        assert len(result) == 5
        assert "sent_beta" in result[0]
        assert "sent_tstat" in result[0]
