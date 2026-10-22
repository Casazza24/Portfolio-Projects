import numpy as np
import yfinance as yf
import pandas as pd

# Effective tax rates outside this band (or with non-positive pretax income)
# are treated as abnormal: one-off charges, NOL releases, loss years.
NORMAL_TAX_RATE_BAND = (0.0, 0.50)

# Fields the model cannot run without. Anything else missing is reported,
# not silently zeroed.
REQUIRED_FIELDS = ("Total Revenue", "Operating Income")

# Fields where a missing row legitimately means "none" (e.g. a debt-free
# company has no Current Debt line). These default to 0 but are still flagged.
ZERO_IF_MISSING = ("Current Debt", "Total Debt", "Research And Development")


def _safe_loc(statement, label, default=np.nan):
    """Return a value from a financial statement, or default if missing."""
    try:
        val = statement.loc[label]
    except KeyError:
        return default
    return default if pd.isna(val) else val


def effective_tax_rate(tax_provision, pretax_income):
    """Historical effective tax rate, or NaN when the year is abnormal.

    Uses pretax income (not EBIT) as the base. Loss years and rates outside
    NORMAL_TAX_RATE_BAND return NaN so they never drive the forecast rate.
    """
    if pd.isna(tax_provision) or pd.isna(pretax_income) or pretax_income <= 0:
        return np.nan
    rate = tax_provision / pretax_income
    lo, hi = NORMAL_TAX_RATE_BAND
    return rate if lo <= rate <= hi else np.nan


def unlevered_fcf(operating_income, tax_rate, da, capex, change_wc):
    """UFCF using cash-flow-statement sign conventions.

    capex:     negative = cash outflow (sign of the input is normalized).
    change_wc: cash impact of working capital (negative = NWC build).
    """
    return operating_income * (1 - tax_rate) + da - abs(capex) + change_wc


def build_historical(income_stmt, balance_sheet, cashflow, fallback_tax_rate=0.21):
    """Structure raw statements into hist_df plus a dict of missing fields."""
    historical = {}
    missing = {}

    for year in income_stmt.columns:
        inc = income_stmt[year]
        bal = balance_sheet[year] if year in balance_sheet.columns else pd.Series(dtype=float)
        cf = cashflow[year] if year in cashflow.columns else pd.Series(dtype=float)

        total_revenue = _safe_loc(inc, "Total Revenue")
        if pd.isna(total_revenue):
            continue  # yfinance pads older years with all-NaN columns

        gross_profit = _safe_loc(inc, "Gross Profit")
        cost_of_revenue = _safe_loc(inc, "Cost Of Revenue")
        if pd.isna(gross_profit) and not pd.isna(cost_of_revenue):
            gross_profit = total_revenue - cost_of_revenue

        operating_income = _safe_loc(inc, "Operating Income")
        tax_provision = _safe_loc(inc, "Tax Provision")
        pretax_income = _safe_loc(inc, "Pretax Income")
        da = _safe_loc(cf, "Depreciation And Amortization")
        capex = _safe_loc(cf, "Capital Expenditure")
        change_wc = _safe_loc(cf, "Change In Working Capital")

        eff_tax = effective_tax_rate(tax_provision, pretax_income)
        fcf_tax = fallback_tax_rate if pd.isna(eff_tax) else eff_tax

        row = {
            "Total Revenue": total_revenue,
            "Cost Of Revenue": cost_of_revenue,
            "Gross Profit": gross_profit,
            "SG&A": _safe_loc(inc, "Selling General And Administration"),
            "Research And Development": _safe_loc(inc, "Research And Development"),
            "Operating Income": operating_income,
            "EBITDA": _safe_loc(inc, "EBITDA"),
            "EBIT": _safe_loc(inc, "EBIT"),
            "Pretax Income": pretax_income,
            "Tax Provision": tax_provision,
            "Effective Tax Rate": eff_tax,
            "Net Income": _safe_loc(inc, "Net Income"),
            "Capital Expenditure": -abs(capex) if not pd.isna(capex) else np.nan,
            "Depreciation And Amortization": da,
            "Change In Working Capital": change_wc,
            "Current Assets": _safe_loc(bal, "Current Assets"),
            "Current Liabilities": _safe_loc(bal, "Current Liabilities"),
            "Current Debt": _safe_loc(bal, "Current Debt"),
            "Total Debt": _safe_loc(bal, "Total Debt"),
            "Cash And Cash Equivalents": _safe_loc(bal, "Cash And Cash Equivalents"),
            "Gross Margin": gross_profit / total_revenue if total_revenue else np.nan,
            # Operating income is the EBIT concept used in the projections, so
            # historical and projected margins are comparable.
            "EBIT Margin": operating_income / total_revenue if total_revenue else np.nan,
            "Free Cash Flow": unlevered_fcf(operating_income, fcf_tax, da, capex, change_wc),
        }

        for field in ZERO_IF_MISSING:
            if pd.isna(row[field]):
                row[field] = 0.0
                missing.setdefault(field, []).append(year)
        for field, val in row.items():
            if field in ZERO_IF_MISSING or field == "Effective Tax Rate":
                continue
            if pd.isna(val):
                missing.setdefault(field, []).append(year)

        historical[year] = row

    if not historical:
        raise ValueError("No fiscal years with reported revenue.")

    hist_df = pd.DataFrame(historical)
    hist_df.loc["Revenue Growth"] = hist_df.loc["Total Revenue"].pct_change(periods=-1, fill_method=None)

    latest = hist_df.columns[0]
    for field in REQUIRED_FIELDS:
        if pd.isna(hist_df.loc[field, latest]):
            raise ValueError(f"Required field '{field}' is missing for the latest fiscal year ({latest}).")

    return hist_df, missing


def fetch_financial_data(symbol, fallback_tax_rate=0.21):
    """Fetch and structure 4 years of financial data for a given ticker."""
    ticker = yf.Ticker(symbol)
    info = ticker.info

    if not info or "currentPrice" not in info:
        raise ValueError(
            f"No valid data for '{symbol}'. "
            "Ticker may be delisted, misspelled, or missing from yfinance."
        )

    DCF_UNFRIENDLY_INDUSTRIES = {
        "banks", "insurance", "reits", "real estate investment trusts",
        "diversified banks", "regional banks", "thrifts & mortgage finance",
        "life & health insurance", "property & casualty insurance",
        "multi-line insurance", "reinsurance", "mortgage reits",
    }
    industry = (info.get("industry") or "").lower()
    sector = (info.get("sector") or "").lower()
    warning = None
    if industry in DCF_UNFRIENDLY_INDUSTRIES or sector == "financial services":
        warning = (
            f"Warning: {symbol.upper()} is in the {info.get('industry', 'financial')} industry. "
            "Traditional DCF models are not well-suited for financial companies "
            "(banks, insurance, REITs) because their cash flows, debt, and working "
            "capital behave fundamentally differently. Results should be interpreted with caution."
        )

    income_stmt = ticker.income_stmt.iloc[:, :4]
    balance_sheet = ticker.balance_sheet.iloc[:, :4]
    cashflow = ticker.cashflow.iloc[:, :4]

    hist_df, missing = build_historical(income_stmt, balance_sheet, cashflow, fallback_tax_rate)

    return {
        "symbol": symbol,
        "ticker": ticker,
        "info": info,
        "hist_df": hist_df,
        "income_stmt": income_stmt,
        "missing": missing,
        "warning": warning,
    }
