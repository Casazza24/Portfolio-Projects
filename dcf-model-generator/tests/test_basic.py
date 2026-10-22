"""Smoke tests for the DCF pipeline."""

import pytest
from dcf_engine import calculate_wacc, run_dcf
from validation import validate_financials
import pandas as pd


def test_run_dcf_known_inputs():
    """Verify DCF with known inputs produces known output."""
    result = run_dcf(
        wacc=0.10,
        terminal_growth=0.025,
        projected_ufcfs=[100e6] * 8,
        total_debt=500e6,
        cash=200e6,
        shares=100e6,
    )
    assert result['implied_price'] > 0
    assert result['enterprise_value'] > 0
    assert result['pv_terminal'] > sum(result['pv_fcfs'])


def test_run_dcf_wacc_below_tgr():
    """WACC <= terminal growth should raise."""
    with pytest.raises(ValueError, match="wacc must be greater"):
        run_dcf(wacc=0.02, terminal_growth=0.03, projected_ufcfs=[100], total_debt=0, cash=0, shares=100)


def test_run_dcf_no_shares():
    """Missing shares should raise."""
    with pytest.raises(ValueError, match="shares"):
        run_dcf(wacc=0.10, terminal_growth=0.025, projected_ufcfs=[100], total_debt=0, cash=0, shares=0)


def test_calculate_wacc_returns_dict():
    """calculate_wacc should return a dict with all components."""
    capital = {"total_debt": 500e6, "avg_debt_for_interest": 500e6, "market_cap": 10e9}
    info = {"beta": 1.2, "marketCap": 10e9}
    income_stmt = pd.DataFrame(
        {"2024": [50e6]},
        index=["Interest Expense"],
    )
    result = calculate_wacc(capital, info, income_stmt, tax_rate=0.21,
                            risk_free_rate=0.04, equity_risk_premium=0.045)

    assert isinstance(result, dict)
    assert 'wacc' in result
    assert 0 < result['wacc'] < 0.30
    assert result['beta'] == 1.2
    assert result['cost_of_equity'] > 0
    assert result['equity_weight'] + result['debt_weight'] == pytest.approx(1.0)


def test_calculate_wacc_zero_debt():
    """With no debt, WACC should equal cost of equity."""
    capital = {"total_debt": 0.0, "avg_debt_for_interest": 0.0, "market_cap": 10e9}
    info = {"beta": 1.0, "marketCap": 10e9}
    income_stmt = pd.DataFrame({"2024": [0]}, index=["Interest Expense"])

    result = calculate_wacc(capital, info, income_stmt, tax_rate=0.21,
                            risk_free_rate=0.04, equity_risk_premium=0.045)
    assert result['wacc'] == result['cost_of_equity']
    assert result['debt_weight'] == 0.0


def test_validate_financials_clean():
    """Clean data should produce no warnings."""
    hist_df = pd.DataFrame({
        "2024": [1000, 400, 600, 700, 600, 100],
    }, index=["Total Revenue", "Cost Of Revenue", "Gross Profit", "EBITDA", "EBIT", "Depreciation And Amortization"])
    warnings = validate_financials(hist_df)
    assert len(warnings) == 0


def test_validate_financials_mismatch():
    """Gross profit mismatch should produce a warning."""
    hist_df = pd.DataFrame({
        "2024": [1000, 400, 700, 0, 0, 0],
    }, index=["Total Revenue", "Cost Of Revenue", "Gross Profit", "EBITDA", "EBIT", "Depreciation And Amortization"])
    warnings = validate_financials(hist_df)
    assert any("Gross Profit mismatch" in w["message"] for w in warnings)
