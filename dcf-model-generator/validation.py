"""Accounting identity and sanity checks on fetched financial data.

Every issue carries a severity:
    critical -- the valuation should not be presented as reliable
    warning  -- an input is estimated, missing in some years, or suspicious
    info     -- context worth showing, no effect on reliability
"""

import pandas as pd

CRITICAL, WARNING, INFO = "critical", "warning", "info"

# Inputs shown in the provenance table: (label, hist_df row or None, XBRL field)
PROVENANCE_INPUTS = [
    ("Total Revenue", "Total Revenue", "Revenues"),
    ("Gross Profit", "Gross Profit", "GrossProfit"),
    ("Operating Income", "Operating Income", "OperatingIncome"),
    ("Pretax Income", "Pretax Income", None),
    ("Tax Provision", "Tax Provision", None),
    ("Depreciation And Amortization", "Depreciation And Amortization", "DepreciationAmortization"),
    ("Capital Expenditure", "Capital Expenditure", "CapEx"),
    ("Change In Working Capital", "Change In Working Capital", None),
    ("Cash And Cash Equivalents", "Cash And Cash Equivalents", "Cash"),
    ("Total Debt", "Total Debt", None),
]

# XBRL mismatches on these fields in the latest year undermine the forecast base.
KEY_XBRL_FIELDS = {"Revenues", "OperatingIncome"}


def _issue(severity, check, year, message):
    return {"severity": severity, "check": check, "year": year, "message": message}


def validate_financials(hist_df, missing=None, derived=None, wacc_result=None):
    """Run accounting identity and input checks. Return a list of issue dicts."""
    issues = []
    missing = missing or {}

    for col in hist_df.columns:
        year_label = col.year if hasattr(col, 'year') else str(col)

        revenue = _safe(hist_df, 'Total Revenue', col)
        cogs = _safe(hist_df, 'Cost Of Revenue', col)
        gross_profit_reported = _safe(hist_df, 'Gross Profit', col)
        if revenue and cogs and gross_profit_reported:
            computed = revenue - cogs
            gap = abs(computed - gross_profit_reported)
            if gap > abs(revenue) * 0.01:
                issues.append(_issue(
                    CRITICAL if gap > abs(revenue) * 0.05 else WARNING,
                    "gross_profit_identity", year_label,
                    f"{year_label}: Gross Profit mismatch: "
                    f"Revenue - COGS = {computed:,.0f}, "
                    f"reported = {gross_profit_reported:,.0f}",
                ))

        if revenue and revenue < 0:
            issues.append(_issue(CRITICAL, "negative_revenue", year_label,
                                 f"{year_label}: Negative revenue ({revenue:,.0f})"))

        ebit = _safe(hist_df, 'EBIT', col)
        ebitda = _safe(hist_df, 'EBITDA', col)
        da = _safe(hist_df, 'Depreciation And Amortization', col)
        if ebitda and da and ebit:
            computed_ebit = ebitda - abs(da)
            if abs(computed_ebit - ebit) > abs(ebitda) * 0.05:
                issues.append(_issue(
                    INFO, "ebitda_bridge", year_label,
                    f"{year_label}: EBIT vs EBITDA-D&A gap: "
                    f"computed = {computed_ebit:,.0f}, reported EBIT = {ebit:,.0f}",
                ))

    latest = hist_df.columns[0] if len(hist_df.columns) else None
    for field, years in missing.items():
        zero_filled = field in ("Current Debt", "Total Debt", "Research And Development")
        in_latest = latest in years
        if zero_filled:
            severity, verb = INFO, "not reported; treated as 0"
        else:
            severity, verb = WARNING, "missing; excluded from averages"
        year_txt = ", ".join(str(y.year if hasattr(y, 'year') else y) for y in years)
        issues.append(_issue(severity, "missing_input", None,
                             f"{field} {verb} ({year_txt})" + (" including latest year" if in_latest else "")))

    if derived:
        for year, rate in derived.get("effective_tax_rates", {}).items():
            if pd.isna(rate):
                y = year.year if hasattr(year, 'year') else year
                issues.append(_issue(INFO, "abnormal_tax_year", y,
                                     f"{y}: effective tax rate abnormal or loss year; excluded from normalized rate"))
        if "fallback" in derived.get("tax_rate_source", ""):
            issues.append(_issue(WARNING, "tax_rate_estimated", None,
                                 f"No normal historical tax year; using fallback rate {derived['tax_rate']:.1%}. "
                                 "This ignores loss carryforwards and non-US rates; set the tax rate manually."))
        if derived.get("other_opex_pct_revenue", 0) < -0.02:
            issues.append(_issue(WARNING, "opex_reconciliation", None,
                                 "SG&A + R&D exceed (Gross Profit - Operating Income); "
                                 "reported operating lines may overlap"))

    if wacc_result:
        if "assumed" in wacc_result.get("beta_source", ""):
            issues.append(_issue(WARNING, "beta_estimated", None, "Beta unavailable; assumed 1.0"))
        if "estimated" in wacc_result.get("interest_expense_source", ""):
            issues.append(_issue(WARNING, "interest_estimated", None,
                                 "Interest expense not reported; cost of debt estimated at 3.5%"))

    return issues


def xbrl_issues(report, summary):
    """Translate an XBRL cross-check into validation issues."""
    issues = []
    status = summary["status"]
    if status == "DISCREPANCIES":
        latest_end = max((r['period_end'] for r in report if r['period_end'] is not None), default=None)
        for r in report:
            if r['status'] != 'MISMATCH':
                continue
            key = r['field'] in KEY_XBRL_FIELDS and r['period_end'] == latest_end
            issues.append(_issue(
                CRITICAL if key else WARNING, "xbrl_mismatch", r['year'],
                f"{r['field']} ({r['year']}): SEC XBRL = {r['xbrl_val']:,.0f} vs "
                f"Yahoo = {r['yf_val']:,.0f} ({r['pct_diff']:.1%} diff)",
            ))
    elif status in ("INCONCLUSIVE", "UNAVAILABLE"):
        issues.append(_issue(WARNING, "xbrl_unverified", None,
                             f"SEC XBRL verification {status.lower()}: {summary['reason']}"))
    else:
        issues.append(_issue(INFO, "xbrl_verified", None, f"SEC XBRL verified: {summary['reason']}"))
    return issues


def valuation_sanity_checks(dcf_results, derived, projected, capital, country=None):
    """Sanity checks on the finished valuation. Returns (issues, metrics).

    Covers terminal value share of EV, projected vs historical operating
    margin, terminal-year reinvestment vs growth, diluted share count, the
    age of the capital-structure snapshot, and the tax-rate fallback.
    """
    issues, m = [], {}
    ev = dcf_results["enterprise_value"]
    g = derived["terminal_growth_rate"]

    if ev <= 0:
        issues.append(_issue(CRITICAL, "negative_ev", None, f"Enterprise value is not positive ({ev:,.0f})"))
        m["tv_pct_ev"] = float("nan")
    else:
        m["tv_pct_ev"] = dcf_results["pv_terminal"] / ev
        sev = WARNING if m["tv_pct_ev"] > 0.85 else INFO
        issues.append(_issue(sev, "terminal_value_share", None,
                             f"Terminal value is {m['tv_pct_ev']:.0%} of enterprise value"
                             + (" (most of the value sits beyond the forecast window)" if sev == WARNING else "")))

    last = projected.columns[-1]
    m["op_margin_hist"] = derived.get("operating_margin_hist", float("nan"))
    m["op_margin_terminal"] = projected.loc["EBIT Margin", last]
    gap = m["op_margin_terminal"] - m["op_margin_hist"]
    issues.append(_issue(WARNING if abs(gap) > 0.10 else INFO, "operating_margin", None,
                         f"Terminal-year operating margin {m['op_margin_terminal']:.1%} vs "
                         f"historical average {m['op_margin_hist']:.1%} ({gap:+.1%})"))

    # Net reinvestment = CapEx beyond D&A plus NWC build (both sign-flipped
    # from cash-flow convention). Sustainable growth g = reinvestment rate x
    # RONIC, so the implied return on new capital is g / reinvestment rate.
    nopat = projected.loc["NOPAT", last]
    net_reinvest = (-projected.loc["Capital Expenditure", last]
                    - projected.loc["Depreciation And Amortization", last]
                    - projected.loc["Change In Working Capital", last])
    m["reinvestment_rate"] = net_reinvest / nopat if nopat else float("nan")
    if nopat <= 0:
        issues.append(_issue(WARNING, "reinvestment", None, "Terminal-year NOPAT is not positive"))
    elif g > 0 and m["reinvestment_rate"] <= 0:
        issues.append(_issue(WARNING, "reinvestment", None,
                             f"Terminal growth {g:.1%} with net reinvestment of {m['reinvestment_rate']:.1%} "
                             "of NOPAT: growth is assumed without investment (implied RONIC unbounded)"))
    else:
        m["implied_ronic"] = g / m["reinvestment_rate"] if m["reinvestment_rate"] else float("nan")
        sev = WARNING if m["implied_ronic"] > 0.50 else INFO
        issues.append(_issue(sev, "reinvestment", None,
                             f"Terminal reinvestment {m['reinvestment_rate']:.1%} of NOPAT implies "
                             f"{m['implied_ronic']:.0%} return on new capital at {g:.1%} growth"))

    m["dilution_pct"] = capital["dilutive_shares"] / capital["shares_basic"]
    if "unavailable" in capital["share_source"]:
        issues.append(_issue(WARNING, "diluted_shares", None,
                             "Diluted share data unavailable; implied price uses basic shares"))
    else:
        issues.append(_issue(INFO, "diluted_shares", None,
                             f"Diluted shares {capital['shares_diluted']:,.0f} "
                             f"(+{m['dilution_pct']:.2%} over basic; {capital['share_source']})"))

    if capital["age_days"] > 135:
        issues.append(_issue(WARNING, "stale_balance_sheet", None,
                             f"Debt and cash are as of {capital['as_of'].date()} "
                             f"({capital['age_days']} days before the share price date)"))

    if "fallback" in derived.get("tax_rate_source", "") and country and country != "United States":
        issues.append(_issue(WARNING, "tax_rate_jurisdiction", None,
                             f"Fallback tax rate {derived['tax_rate']:.1%} is the US federal statutory "
                             f"rate; company is domiciled in {country}. Override the tax rate."))

    return issues, m


def is_reliable(issues):
    """A valuation is presentable as reliable only with no critical issues."""
    return not any(i["severity"] == CRITICAL for i in issues)


def input_provenance(hist_df, missing=None, xbrl_report=None):
    """Latest-year status of each key input: reported, missing, assumed 0, XBRL verified/mismatch."""
    missing = missing or {}
    latest = hist_df.columns[0]
    xbrl_status = {}
    for r in xbrl_report or []:
        if r['period_end'] is not None and pd.Timestamp(r['period_end']) == pd.Timestamp(latest):
            xbrl_status[r['field']] = r['status']

    rows = []
    for label, hist_label, xbrl_field in PROVENANCE_INPUTS:
        value = hist_df.loc[hist_label, latest] if hist_label in hist_df.index else None
        if latest in missing.get(hist_label, []):
            source = "assumed 0" if value == 0 else "missing"
        else:
            source = "reported (Yahoo Finance)"
        if xbrl_field is None:
            check = "not checked"
        else:
            check = {
                "MATCH": "XBRL verified", "MISMATCH": "XBRL mismatch",
                "NONCOMPARABLE": "not comparable",
            }.get(xbrl_status.get(xbrl_field), "XBRL unavailable")
        rows.append({"Input": label, "Value": value, "Source": source, "Verification": check})
    return pd.DataFrame(rows)


def _safe(df, label, col, default=None):
    """Safely get a value from a DataFrame."""
    try:
        val = df.loc[label, col]
        return val if pd.notna(val) and val != 0 else default
    except (KeyError, TypeError):
        return default
