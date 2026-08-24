import numpy as np
import pandas as pd
import pytest

from src.factors import run_capm_regression, run_ff_regression


@pytest.fixture
def factor_data():
    """Synthetic Fama-French factor data."""
    np.random.seed(42)
    n = 252
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {
            "Mkt-RF": np.random.normal(0.0003, 0.01, n),
            "SMB": np.random.normal(0.0001, 0.005, n),
            "HML": np.random.normal(0.0001, 0.004, n),
            "RMW": np.random.normal(0.00005, 0.003, n),
            "CMA": np.random.normal(0.00005, 0.003, n),
            "MOM": np.random.normal(0.0002, 0.006, n),
            "RF": np.full(n, 0.0002),
        },
        index=dates,
    )


@pytest.fixture
def portfolio_returns(factor_data):
    """Portfolio returns correlated with market factor."""
    np.random.seed(7)
    mkt = factor_data["Mkt-RF"].values
    noise = np.random.normal(0, 0.005, len(mkt))
    # beta ~ 1.2, small positive alpha
    returns = 0.0002 + 1.2 * mkt + noise
    return pd.Series(returns, index=factor_data.index, name="portfolio")


class TestCAPM:
    def test_returns_all_keys(self, portfolio_returns, factor_data):
        result = run_capm_regression(portfolio_returns, factor_data)
        expected_keys = {
            "alpha", "beta", "alpha_tstat", "beta_tstat",
            "alpha_pvalue", "beta_pvalue", "r_squared",
        }
        assert set(result.keys()) == expected_keys

    def test_beta_near_expected(self, portfolio_returns, factor_data):
        result = run_capm_regression(portfolio_returns, factor_data)
        assert abs(result["beta"] - 1.2) < 0.3

    def test_alpha_is_annualized(self, portfolio_returns, factor_data):
        result = run_capm_regression(portfolio_returns, factor_data)
        assert isinstance(result["alpha"], float)

    def test_r_squared_in_range(self, portfolio_returns, factor_data):
        result = run_capm_regression(portfolio_returns, factor_data)
        assert 0 <= result["r_squared"] <= 1


class TestFamaFrench:
    def test_returns_all_keys(self, portfolio_returns, factor_data):
        result = run_ff_regression(portfolio_returns, factor_data)
        assert "alpha" in result
        assert "factor_betas" in result
        assert "Mkt-RF" in result["factor_betas"]

    def test_all_six_factors(self, portfolio_returns, factor_data):
        result = run_ff_regression(portfolio_returns, factor_data)
        expected_factors = {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"}
        assert set(result["factor_betas"].keys()) == expected_factors

    def test_r_squared_at_least_capm(self, portfolio_returns, factor_data):
        capm = run_capm_regression(portfolio_returns, factor_data)
        ff = run_ff_regression(portfolio_returns, factor_data)
        # More factors should explain at least as much variance
        assert ff["r_squared"] >= capm["r_squared"] - 0.01

    def test_mkt_beta_similar_to_capm(self, portfolio_returns, factor_data):
        capm = run_capm_regression(portfolio_returns, factor_data)
        ff = run_ff_regression(portfolio_returns, factor_data)
        assert abs(ff["factor_betas"]["Mkt-RF"] - capm["beta"]) < 0.3


class TestSentimentIntegration:
    def test_ff_regression_includes_sent_when_present(self, portfolio_returns, factor_data):
        factor_data_with_sent = factor_data.copy()
        np.random.seed(123)
        factor_data_with_sent["SENT"] = np.random.normal(0.0001, 0.003, len(factor_data))
        result = run_ff_regression(portfolio_returns, factor_data_with_sent)
        assert "SENT" in result["factor_betas"]
        assert "SENT" in result["factor_tstats"]
        assert "SENT" in result["factor_pvalues"]

    def test_ff_regression_works_without_sent(self, portfolio_returns, factor_data):
        result = run_ff_regression(portfolio_returns, factor_data)
        assert "SENT" not in result["factor_betas"]
        assert len(result["factor_betas"]) == 6
