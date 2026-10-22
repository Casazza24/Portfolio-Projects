"""Tests for XBRL period alignment, verification status, and validation severity."""

import pandas as pd
import pytest

from xbrl_fetcher import extract_annual_financials, validate_against_yfinance, summarize_xbrl_report
from validation import validate_financials, xbrl_issues, is_reliable, input_provenance


def fact(start, end, val, fy, filed, form="10-K"):
    e = {"end": end, "val": val, "fy": fy, "fp": "FY", "form": form, "filed": filed}
    if start:
        e["start"] = start
    return e


def facts(**concepts):
    return {"facts": {"us-gaap": {c: {"units": {"USD": rows}} for c, rows in concepts.items()}}}


# Mirrors real AAPL data: the FY2025 10-K tags its comparative years with fy=2025.
REVENUE_ROWS = [
    fact("2022-09-25", "2023-09-30", 383.0, 2024, "2024-11-01"),
    fact("2023-10-01", "2024-09-28", 391.0, 2024, "2024-11-01"),
    fact("2022-09-25", "2023-09-30", 383.0, 2025, "2025-10-31"),
    fact("2023-10-01", "2024-09-28", 391.0, 2025, "2025-10-31"),
    fact("2024-09-29", "2025-09-27", 416.0, 2025, "2025-10-31"),
    fact("2025-06-29", "2025-09-27", 102.0, 2025, "2025-10-31"),  # Q4 duration inside the 10-K
    fact("2024-09-29", "2025-06-28", 300.0, 2025, "2025-08-01", form="10-Q"),
]


def test_keys_by_period_end_not_filing_fy():
    out = extract_annual_financials(facts(RevenueFromContractWithCustomerExcludingAssessedTax=REVENUE_ROWS))
    rev = out["Revenues"]
    assert rev[pd.Timestamp("2025-09-27")] == 416.0
    assert rev[pd.Timestamp("2024-09-28")] == 391.0
    assert rev[pd.Timestamp("2023-09-30")] == 383.0
    assert 102.0 not in rev.values() and 300.0 not in rev.values()


def test_merges_concepts_when_company_switches_tags():
    old = [fact("2017-10-01", "2018-09-29", 265.0, 2018, "2018-11-05")]
    out = extract_annual_financials(facts(
        Revenues=old, RevenueFromContractWithCustomerExcludingAssessedTax=REVENUE_ROWS,
    ))
    # The stale 'Revenues' tag must not hide recent years
    assert pd.Timestamp("2025-09-27") in out["Revenues"]


def test_amendment_supersedes_original_filing():
    rows = [
        fact("2024-09-29", "2025-09-27", 416.0, 2025, "2025-10-31"),
        fact("2024-09-29", "2025-09-27", 410.0, 2025, "2026-01-15", form="10-K/A"),
    ]
    out = extract_annual_financials(facts(Revenues=rows))
    assert out["Revenues"][pd.Timestamp("2025-09-27")] == 410.0


def test_instant_facts_for_balance_sheet():
    rows = [{"end": "2025-09-27", "val": 36.0, "fy": 2025, "fp": "FY", "form": "10-K", "filed": "2025-10-31"}]
    out = extract_annual_financials(facts(CashAndCashEquivalentsAtCarryingValue=rows))
    assert out["Cash"] == {pd.Timestamp("2025-09-27"): 36.0}


def _hist(rev=(416.0, 391.0), cols=("2025-09-30", "2024-09-30")):
    return pd.DataFrame({pd.Timestamp(c): {"Total Revenue": r} for c, r in zip(cols, rev)})


def test_52_53_week_period_end_matches_yfinance_column():
    xbrl = {"Revenues": {pd.Timestamp("2025-09-27"): 416.0, pd.Timestamp("2024-09-28"): 391.0}}
    report = validate_against_yfinance("X", xbrl, _hist())
    rev = [r for r in report if r["field"] == "Revenues"]
    assert {r["status"] for r in rev} == {"MATCH"}


def test_no_comparisons_is_not_verified():
    # The old code printed "passed" here because there were no mismatches.
    xbrl = {"Revenues": {pd.Timestamp("2018-09-29"): 265.0}}
    summary = summarize_xbrl_report(validate_against_yfinance("X", xbrl, _hist()))
    assert summary["status"] == "UNAVAILABLE"


def test_empty_report_is_unavailable():
    assert summarize_xbrl_report([])["status"] == "UNAVAILABLE"


def test_low_coverage_is_inconclusive():
    hist = _hist()
    hist.loc["Gross Profit"] = [195.0, 180.0]
    hist.loc["Operating Income"] = [133.0, 123.0]
    xbrl = {"Revenues": {pd.Timestamp("2025-09-27"): 416.0}}
    summary = summarize_xbrl_report(validate_against_yfinance("X", xbrl, hist))
    assert summary["status"] == "INCONCLUSIVE"


def test_mismatch_on_latest_revenue_is_critical():
    xbrl = {"Revenues": {pd.Timestamp("2025-09-27"): 500.0, pd.Timestamp("2024-09-28"): 391.0}}
    report = validate_against_yfinance("X", xbrl, _hist())
    summary = summarize_xbrl_report(report)
    assert summary["status"] == "DISCREPANCIES"
    issues = xbrl_issues(report, summary)
    assert not is_reliable(issues)


def test_zero_xbrl_value_is_noncomparable():
    xbrl = {"Revenues": {pd.Timestamp("2025-09-27"): 0.0}}
    report = validate_against_yfinance("X", xbrl, _hist())
    assert any(r["status"] == "NONCOMPARABLE" for r in report)


def test_large_gross_profit_gap_is_critical_small_is_warning():
    big = pd.DataFrame({"2024": [1000, 400, 700]}, index=["Total Revenue", "Cost Of Revenue", "Gross Profit"])
    small = pd.DataFrame({"2024": [1000, 400, 620]}, index=["Total Revenue", "Cost Of Revenue", "Gross Profit"])
    assert not is_reliable(validate_financials(big))
    sev = [i["severity"] for i in validate_financials(small)]
    assert sev == ["warning"] and is_reliable(validate_financials(small))


def test_provenance_distinguishes_missing_and_verified():
    latest = pd.Timestamp("2025-09-30")
    hist = pd.DataFrame({latest: {"Total Revenue": 416.0, "Total Debt": 0.0, "Pretax Income": float("nan")}})
    report = [{"field": "Revenues", "period_end": latest, "status": "MATCH"}]
    prov = input_provenance(hist, {"Total Debt": [latest], "Pretax Income": [latest]}, report).set_index("Input")
    assert prov.loc["Total Revenue", "Verification"] == "XBRL verified"
    assert prov.loc["Total Debt", "Source"] == "assumed 0"
    assert prov.loc["Pretax Income", "Source"] == "missing"
