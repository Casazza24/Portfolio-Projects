"""Regression tests for the DCF financial-calculation fixes.

Each test uses a small hand-computed example so the expected value can be
checked on paper. None of these touch the network.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from config import assumptions as CONFIG, revenue_growth_rates
from data_fetcher import build_historical, effective_tax_rate, _safe_loc
from dcf_engine import calculate_wacc, run_dcf
from projections import derive_assumptions, project_financials, apply_overrides
from sensitivity import build_sensitivity_tables

YEARS = [pd.Timestamp(f"{y}-09-30") for y in (2025, 2024, 2023)]  # newest first, like yfinance


def make_statements(capex_sign=-1, **overrides):
    """Three identical years of a simple company.

    Revenue 1000, COGS 400, GP 600, SG&A 150, R&D 100, Operating income 350,
    D&A 50 (embedded in COGS/opex), pretax 340, tax 68.
    """
    inc = {
        "Total Revenue": 1000.0, "Cost Of Revenue": 400.0, "Gross Profit": 600.0,
        "Selling General And Administration": 150.0, "Research And Development": 100.0,
        "Operating Income": 350.0, "EBITDA": 400.0, "EBIT": 350.0,
        "Pretax Income": 340.0, "Tax Provision": 68.0, "Net Income": 272.0,
    }
    cf = {
        "Depreciation And Amortization": 50.0,
        "Capital Expenditure": capex_sign * 80.0,
        "Change In Working Capital": -20.0,
        "Operating Cash Flow": 300.0,
    }
    bal = {
        "Current Assets": 500.0, "Current Liabilities": 300.0, "Current Debt": 50.0,
        "Total Debt": 200.0, "Cash And Cash Equivalents": 100.0,
    }
    for key, val in overrides.items():
        for d in (inc, cf, bal):
            if key in d:
                d[key] = val
    mk = lambda d: pd.DataFrame({y: d for y in YEARS})
    return mk(inc), mk(bal), mk(cf)


def make_hist(**overrides):
    hist_df, missing = build_historical(*make_statements(**overrides))
    return hist_df, missing


def flat_config(**extra):
    cfg = copy.deepcopy(CONFIG)
    cfg.update({"gross_margin_expansion": 0.0}, **extra)
    return cfg


# --- 1. Free cash flow sign conventions -----------------------------------

def test_historical_fcf_subtracts_negative_capex():
    hist_df, _ = make_hist()
    # tax rate 68/340 = 20%; 350*0.8 + 50 - 80 - 20 = 230
    assert hist_df.loc["UFCF", YEARS[0]] == pytest.approx(230.0)


def test_historical_fcf_same_for_positive_capex_input():
    hist_df, _ = make_hist(capex_sign=+1)
    assert hist_df.loc["UFCF", YEARS[0]] == pytest.approx(230.0)
    assert hist_df.loc["Capital Expenditure", YEARS[0]] == -80.0


def test_projected_rows_use_cash_flow_signs():
    hist_df, _ = make_hist()
    d = derive_assumptions(hist_df, flat_config())
    d["revenue_growth_rates"] = [0.10] * d["projection_years"]
    p = project_financials(1000.0, d)
    y1 = p["Year 1"]
    assert y1["Capital Expenditure"] < 0
    # NWC = 500 - 100 - (300 - 50) = 150 -> 15% of revenue; revenue +100 -> NWC +15 -> cash -15
    assert y1["Change In Working Capital"] == pytest.approx(-15.0)
    assert y1["UFCF"] == pytest.approx(
        y1["NOPAT"] + y1["Depreciation And Amortization"]
        + y1["Capital Expenditure"] + y1["Change In Working Capital"]
    )


# --- 2. EBIT reconciles to operating income, D&A not double counted --------

def test_projected_ebit_reconciles_to_operating_income():
    hist_df, _ = make_hist()
    d = derive_assumptions(hist_df, flat_config())
    d["revenue_growth_rates"] = [0.0] * d["projection_years"]
    p = project_financials(1000.0, d)
    # Old formula: GP - SG&A - D&A = 600 - 150 - 50 = 400 (drops R&D, double-counts D&A)
    assert p.loc["EBIT", "Year 1"] == pytest.approx(350.0)
    assert p.loc["EBITDA", "Year 1"] == pytest.approx(400.0)
    assert p.loc["Research And Development", "Year 1"] == pytest.approx(100.0)


def test_other_opex_absorbs_unlisted_operating_costs():
    # Operating income 300 with SG&A 150 and R&D 100 -> 50 of other opex
    hist_df, _ = make_hist(**{"Operating Income": 300.0})
    d = derive_assumptions(hist_df, flat_config())
    assert d["other_opex_pct_revenue"] == pytest.approx(0.05)
    d["revenue_growth_rates"] = [0.0] * d["projection_years"]
    assert project_financials(1000.0, d).loc["EBIT", "Year 1"] == pytest.approx(300.0)


# --- 3. Tax rate uses pretax income, abnormal years excluded ---------------

def test_tax_rate_uses_pretax_income_not_ebit():
    hist_df, _ = make_hist()
    d = derive_assumptions(hist_df, flat_config())
    assert d["tax_rate"] == pytest.approx(68 / 340)  # 20.0%, not 68/350 = 19.4%


@pytest.mark.parametrize("tax, pretax", [(10.0, -50.0), (80.0, 100.0), (-5.0, 100.0), (np.nan, 100.0)])
def test_abnormal_tax_years_are_nan(tax, pretax):
    assert np.isnan(effective_tax_rate(tax, pretax))


def test_normalized_tax_rate_is_median_of_normal_years():
    inc, bal, cf = make_statements()
    inc.loc["Tax Provision"] = [68.0, 102.0, 51.0]   # 20%, 30%, 15%
    inc.loc["Pretax Income"] = [340.0, 340.0, 340.0]
    hist_df, _ = build_historical(inc, bal, cf)
    assert derive_assumptions(hist_df, flat_config())["tax_rate"] == pytest.approx(0.20)


def test_tax_rate_fallback_when_no_normal_year():
    inc, bal, cf = make_statements()
    inc.loc["Pretax Income"] = -10.0
    hist_df, _ = build_historical(inc, bal, cf)
    d = derive_assumptions(hist_df, flat_config())
    assert d["tax_rate"] == CONFIG["fallback_tax_rate"]
    assert "fallback" in d["tax_rate_source"]


# --- 4. Missing values are not silently zero -------------------------------

def test_safe_loc_defaults_to_nan():
    assert np.isnan(_safe_loc(pd.Series({"a": 1.0}), "b"))


def test_missing_required_field_raises():
    inc, bal, cf = make_statements()
    inc = inc.drop(index="Operating Income")
    with pytest.raises(ValueError, match="Operating Income"):
        build_historical(inc, bal, cf)


def test_missing_capex_everywhere_raises_instead_of_zero():
    inc, bal, cf = make_statements()
    cf = cf.drop(index="Capital Expenditure")
    hist_df, missing = build_historical(inc, bal, cf)
    assert "Capital Expenditure" in missing
    with pytest.raises(ValueError, match="capex"):
        derive_assumptions(hist_df, flat_config())


def test_missing_debt_is_zero_but_flagged():
    inc, bal, cf = make_statements()
    bal = bal.drop(index="Current Debt")
    hist_df, missing = build_historical(inc, bal, cf)
    assert hist_df.loc["Current Debt", YEARS[0]] == 0.0
    assert "Current Debt" in missing


def test_all_nan_padding_columns_are_dropped():
    inc, bal, cf = make_statements()
    pad = pd.Timestamp("2022-09-30")
    inc[pad] = np.nan
    hist_df, _ = build_historical(inc, bal, cf)
    assert pad not in hist_df.columns


# --- 5. Projection horizon passed consistently -----------------------------

@pytest.mark.parametrize("n, hold", [(5, 3), (3, 3), (1, 3), (10, 0)])
def test_growth_path_respects_horizon(n, hold):
    path = revenue_growth_rates(0.10, 0.02, n, hold)
    assert len(path) == n
    assert path[-1] == pytest.approx(0.02)


def test_custom_horizon_flows_through_projection_and_sensitivity():
    hist_df, _ = make_hist()
    d = derive_assumptions(hist_df, flat_config(projection_years=5, hold_years=2))
    assert len(d["revenue_growth_rates"]) == len(d["gross_margins"]) == 5
    p = project_financials(1000.0, d)
    assert list(p.columns) == [f"Year {i}" for i in range(1, 6)]
    ufcfs = list(p.loc["UFCF"])
    sens, sens2 = build_sensitivity_tables(ufcfs, 200, 100, 10, 0.09, 0.025, 1000.0, d)
    assert sens.notna().any().any() and sens2.notna().all().all()


def test_project_financials_rejects_short_schedules():
    hist_df, _ = make_hist()
    d = derive_assumptions(hist_df, flat_config())
    d["revenue_growth_rates"] = d["revenue_growth_rates"][:3]
    with pytest.raises(ValueError, match="growth rates"):
        project_financials(1000.0, d)


# --- 6. WACC uses configured inputs ----------------------------------------

def _wacc_inputs(beta=1.2):
    capital = {"total_debt": 500e6, "avg_debt_for_interest": 500e6, "market_cap": 1.5e9}
    income_stmt = pd.DataFrame({"2024": [25e6]}, index=["Interest Expense"])
    return capital, {"beta": beta, "marketCap": 1.5e9}, income_stmt


def test_wacc_requires_explicit_rates():
    with pytest.raises(TypeError):
        calculate_wacc(*_wacc_inputs(), tax_rate=0.2)


def test_wacc_known_value():
    r = calculate_wacc(*_wacc_inputs(), tax_rate=0.20, risk_free_rate=0.05, equity_risk_premium=0.06)
    # Ke = 5% + 1.2*6% = 12.2%; Kd = 25/500 = 5%; weights 75/25
    assert r["wacc"] == pytest.approx(0.75 * 0.122 + 0.25 * 0.05 * 0.8)


def test_wacc_flags_missing_beta():
    r = calculate_wacc(*_wacc_inputs(beta=None), tax_rate=0.2, risk_free_rate=0.04, equity_risk_premium=0.05)
    assert r["beta"] == 1.0 and "assumed" in r["beta_source"]


def test_run_dcf_hand_computed():
    r = run_dcf(0.10, 0.02, [100.0, 110.0], total_debt=50.0, cash=20.0, shares=10.0)
    pv = 100 / 1.1 + 110 / 1.21
    tv = 110 * 1.02 / 0.08
    assert r["terminal_value"] == pytest.approx(tv)
    assert r["enterprise_value"] == pytest.approx(pv + tv / 1.21)
    assert r["implied_price"] == pytest.approx((pv + tv / 1.21 - 50 + 20) / 10)


# --- 7. Company-specific sensitivity grids ---------------------------------

def test_sensitivity_grids_centered_on_base_case():
    hist_df, _ = make_hist()
    d = derive_assumptions(hist_df, CONFIG)
    p = project_financials(1000.0, d)
    ufcfs = list(p.loc["UFCF"])
    wacc, tgr = 0.0731, 0.025
    base = run_dcf(wacc, tgr, ufcfs, 200, 100, 10)["implied_price"]
    sens, sens2 = build_sensitivity_tables(ufcfs, 200, 100, 10, wacc, tgr, 1000.0, d)
    assert sens.iloc[len(sens) // 2, len(sens.columns) // 2] == pytest.approx(base, abs=0.01)
    assert sens2.iloc[len(sens2) // 2, len(sens2.columns) // 2] == pytest.approx(base, abs=0.01)


def test_sensitivity_marks_wacc_below_tgr_undefined():
    hist_df, _ = make_hist()
    d = derive_assumptions(hist_df, CONFIG)
    ufcfs = list(project_financials(1000.0, d).loc["UFCF"])
    sens, _ = build_sensitivity_tables(ufcfs, 0, 0, 10, 0.03, 0.025, 1000.0, d)
    assert sens.isna().any().any()


# --- D. Dashboard overrides propagate consistently --------------------------

def test_override_terminal_growth_rebuilds_growth_path():
    hist_df, _ = make_hist()
    d = derive_assumptions(hist_df, CONFIG)
    out = apply_overrides(d, {"terminal_growth_rate": 0.01})
    assert out["revenue_growth_rates"][-1] == pytest.approx(0.01)
    assert d["revenue_growth_rates"][-1] == pytest.approx(d["terminal_growth_rate"])  # original untouched


def test_override_year1_growth_keeps_hold_and_fade_shape():
    hist_df, _ = make_hist()
    d = derive_assumptions(hist_df, CONFIG)
    out = apply_overrides(d, {"latest_growth": 0.12})
    expected = revenue_growth_rates(0.12, d["terminal_growth_rate"], d["projection_years"], d["hold_years"])
    assert out["revenue_growth_rates"] == pytest.approx(expected)


def test_wacc_overrides_are_known_keys():
    hist_df, _ = make_hist()
    d = derive_assumptions(hist_df, CONFIG)
    out = apply_overrides(d, {"risk_free_rate": 0.05, "equity_risk_premium": 0.06})
    assert out["risk_free_rate"] == 0.05 and out["equity_risk_premium"] == 0.06


def test_unknown_override_raises():
    hist_df, _ = make_hist()
    with pytest.raises(KeyError):
        apply_overrides(derive_assumptions(hist_df, CONFIG), {"wacc_typo": 0.1})


def test_gross_margin_override_rebuilds_ramp():
    hist_df, _ = make_hist()
    d = derive_assumptions(hist_df, CONFIG)
    out = apply_overrides(d, {"gross_margin": 0.50, "gross_margin_expansion": 0.0})
    assert out["gross_margins"] == pytest.approx([0.50] * d["projection_years"])
