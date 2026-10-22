"""Regression tests for capital-structure consistency, UFCF definition, and
valuation sanity checks. Offline; app.run_model is exercised with its data
sources stubbed out.
"""

import numpy as np
import pandas as pd
import pytest

from config import assumptions as CONFIG
from data_fetcher import build_capital_structure, build_historical, conventional_fcf
from dcf_engine import calculate_wacc, run_dcf
from projections import derive_assumptions, project_financials
from validation import valuation_sanity_checks
from tests.test_financials import make_statements, YEARS, flat_config

Q1, Q2 = pd.Timestamp("2026-06-30"), pd.Timestamp("2026-03-31")
FY1, FY2 = pd.Timestamp("2025-09-30"), pd.Timestamp("2024-09-30")
INFO = {"sharesOutstanding": 100.0, "marketCap": 5000.0, "beta": 1.0, "currentPrice": 50.0, "country": "United States"}


def quarterly_bs(**rows):
    base = {
        "Total Debt": [84.0, 85.0],
        "Cash Cash Equivalents And Short Term Investments": [62.0, 68.0],
        "Cash And Cash Equivalents": [40.0, 46.0],
    }
    base.update(rows)
    return pd.DataFrame(base, index=[Q1, Q2]).T


def annual_bs():
    return pd.DataFrame({"Total Debt": [98.0, 106.0]}, index=[FY1, FY2]).T


def quarterly_is(diluted=101.5, basic=100.5):
    return pd.DataFrame({"Diluted Average Shares": [diluted], "Basic Average Shares": [basic]}, index=[Q1]).T


def capital(**kw):
    return build_capital_structure(
        kw.pop("qbs", quarterly_bs()), kw.pop("qis", quarterly_is()), annual_bs(), INFO,
        as_of_today="2026-10-21",
    )


# --- Capital structure snapshot -------------------------------------------

def test_snapshot_uses_latest_quarter_not_fiscal_year_end():
    c = capital()
    assert c["as_of"] == Q1
    assert c["total_debt"] == 84.0          # not the 98.0 FY-end figure
    assert c["cash"] == 62.0                # cash + short-term investments
    assert c["age_days"] == (pd.Timestamp("2026-10-21") - Q1).days


def test_cost_of_debt_uses_average_fiscal_year_debt():
    c = capital()
    assert c["avg_debt_for_interest"] == pytest.approx(102.0)
    income_stmt = pd.DataFrame({FY1: [5.1]}, index=["Interest Expense"])
    w = calculate_wacc(c, INFO, income_stmt, 0.2, risk_free_rate=0.04, equity_risk_premium=0.05)
    assert w["cost_of_debt"] == pytest.approx(5.1 / 102.0)
    assert w["total_debt"] == 84.0


def test_debt_falls_back_to_components_including_finance_leases():
    qbs = quarterly_bs(**{"Total Debt": [np.nan, np.nan],
                          "Current Debt And Capital Lease Obligation": [9.0, 9.0],
                          "Long Term Debt And Capital Lease Obligation": [47.0, 48.0]})
    c = capital(qbs=qbs)
    assert c["total_debt"] == 56.0


def test_diluted_shares_add_latest_quarter_dilution():
    c = capital()
    assert c["shares_diluted"] == pytest.approx(101.0)
    assert c["shares_diluted"] > c["shares_basic"]


def test_antidilution_is_clipped_and_missing_dilution_is_flagged():
    assert capital(qis=quarterly_is(diluted=99.0, basic=100.0))["dilutive_shares"] == 0.0
    c = capital(qis=pd.DataFrame())
    assert c["shares_diluted"] == c["shares_basic"] and "unavailable" in c["share_source"]


def test_wacc_and_bridge_use_the_same_debt():
    c = capital()
    w = calculate_wacc(c, INFO, pd.DataFrame(), 0.2, risk_free_rate=0.04, equity_risk_premium=0.05)
    r = run_dcf(w["wacc"], 0.02, [100.0] * 3, c["total_debt"], c["cash"], c["shares_diluted"])
    assert w["total_debt"] == r["total_debt"] == c["total_debt"]
    assert w["debt_weight"] == pytest.approx(84.0 / (5000.0 + 84.0))


def test_run_model_end_to_end_uses_one_snapshot(monkeypatch):
    """The dashboard pipeline must bridge with the snapshot, not info['totalDebt']."""
    import app

    inc, bal, cf = make_statements()
    hist_df, missing = build_historical(inc, bal, cf)
    info = dict(INFO, totalDebt=999.0, totalCash=1.0)  # deliberately wrong legacy fields
    c = build_capital_structure(quarterly_bs(), quarterly_is(), annual_bs(), info, as_of_today="2026-10-21")
    data = {"info": info, "hist_df": hist_df, "income_stmt": inc, "missing": missing,
            "capital": c, "warning": None}
    monkeypatch.setattr(app, "fetch_data", lambda symbol: data)
    monkeypatch.setattr(app, "fetch_xbrl", lambda symbol: None)
    monkeypatch.setattr(app, "analyze_filing", lambda *a: None)

    res = app.run_model("TEST")
    assert res["dcf_results"]["total_debt"] == res["wacc_result"]["total_debt"] == 84.0
    assert res["dcf_results"]["cash"] == 62.0
    assert res["dcf_results"]["shares"] == pytest.approx(101.0)

    res2 = app.run_model("TEST", {"risk_free_rate": 0.06})
    assert res2["wacc_result"]["risk_free_rate"] == 0.06
    assert res2["implied_price"] < res["implied_price"]


# --- UFCF definition and working capital ----------------------------------

def test_ufcf_differs_from_ocf_minus_capex():
    hist_df, _ = build_historical(*make_statements())
    # UFCF = 350*(1-0.2) + 50 - 80 - 20 = 230; OCF - CapEx = 300 - 80 = 220
    assert hist_df.loc["UFCF", YEARS[0]] == pytest.approx(230.0)
    assert hist_df.loc["OCF - CapEx", YEARS[0]] == pytest.approx(220.0)
    assert conventional_fcf(300.0, -80.0) == conventional_fcf(300.0, 80.0) == 220.0


def test_nwc_excludes_short_term_investments():
    inc, bal, cf = make_statements()
    bal.loc["Cash Cash Equivalents And Short Term Investments"] = 130.0  # 100 cash + 30 STI
    hist_df, _ = build_historical(inc, bal, cf)
    d = derive_assumptions(hist_df, flat_config())
    # NWC = 500 - 130 - (300 - 50) = 120, not 150
    assert d["nwc_pct_revenue"] == pytest.approx(0.12)


def test_working_capital_build_reduces_ufcf():
    """Growing revenue with positive NWC% must consume cash (negative dNWC)."""
    hist_df, _ = build_historical(*make_statements())
    d = derive_assumptions(hist_df, flat_config())
    d["revenue_growth_rates"] = [0.10] * d["projection_years"]
    p = project_financials(1000.0, d)
    assert d["nwc_pct_revenue"] > 0
    assert (p.loc["Change In Working Capital"] < 0).all()
    d0 = dict(d, nwc_pct_revenue=0.0)
    assert (p.loc["UFCF"] < project_financials(1000.0, d0).loc["UFCF"]).all()


# --- Valuation sanity checks ----------------------------------------------

def projected_frame(nopat=100.0, capex=-60.0, da=40.0, dnwc=-5.0, margin=0.30):
    return pd.DataFrame({"Year 1": {"NOPAT": nopat, "Capital Expenditure": capex,
                                    "Depreciation And Amortization": da,
                                    "Change In Working Capital": dnwc, "EBIT Margin": margin}})


def checks(projected=None, pv_terminal=600.0, ev=1000.0, g=0.025, hist_margin=0.30, **derived_extra):
    dcf = {"enterprise_value": ev, "pv_terminal": pv_terminal}
    derived = {"terminal_growth_rate": g, "operating_margin_hist": hist_margin,
               "tax_rate_source": "median of 3 normal year(s)", "tax_rate": 0.2, **derived_extra}
    return valuation_sanity_checks(dcf, derived, projected if projected is not None else projected_frame(),
                                   capital(), derived_extra.pop("country", "United States"))


def by_check(issues, name):
    return [i for i in issues if i["check"] == name]


def test_reinvestment_and_implied_ronic():
    issues, m = checks()
    # Net reinvestment = 60 - 40 + 5 = 25 -> 25% of NOPAT; RONIC = 2.5% / 25% = 10%
    assert m["reinvestment_rate"] == pytest.approx(0.25)
    assert m["implied_ronic"] == pytest.approx(0.10)
    assert by_check(issues, "reinvestment")[0]["severity"] == "info"


def test_growth_without_reinvestment_is_flagged():
    issues, _ = checks(projected_frame(capex=-30.0, da=40.0, dnwc=5.0))
    assert by_check(issues, "reinvestment")[0]["severity"] == "warning"


def test_terminal_value_share():
    issues, m = checks(pv_terminal=900.0)
    assert m["tv_pct_ev"] == pytest.approx(0.90)
    assert by_check(issues, "terminal_value_share")[0]["severity"] == "warning"
    assert by_check(checks()[0], "terminal_value_share")[0]["severity"] == "info"


def test_negative_ev_is_critical():
    issues, _ = checks(ev=-10.0)
    assert by_check(issues, "negative_ev")[0]["severity"] == "critical"


def test_margin_drift_warning():
    issues, _ = checks(projected_frame(margin=0.45), hist_margin=0.30)
    assert by_check(issues, "operating_margin")[0]["severity"] == "warning"


def test_us_fallback_rate_flagged_for_non_us_company():
    dcf = {"enterprise_value": 1000.0, "pv_terminal": 600.0}
    derived = {"terminal_growth_rate": 0.025, "operating_margin_hist": 0.3,
               "tax_rate_source": "fallback (no normal historical year)", "tax_rate": 0.21}
    issues, _ = valuation_sanity_checks(dcf, derived, projected_frame(), capital(), "Japan")
    assert by_check(issues, "tax_rate_jurisdiction")
    issues_us, _ = valuation_sanity_checks(dcf, derived, projected_frame(), capital(), "United States")
    assert not by_check(issues_us, "tax_rate_jurisdiction")
