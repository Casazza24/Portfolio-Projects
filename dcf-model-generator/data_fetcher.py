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


def conventional_fcf(operating_cash_flow, capex):
    """Cash flow from operations minus CapEx (levered, after interest)."""
    return operating_cash_flow - abs(capex)


def _first_available(statement, *labels):
    for label in labels:
        val = _safe_loc(statement, label)
        if not pd.isna(val):
            return val
    return np.nan


def build_capital_structure(quarterly_bs, quarterly_is, annual_bs, info, as_of_today=None):
    """One dated snapshot of debt, cash and shares for WACC and the equity bridge.

    Debt is balance-sheet Total Debt (borrowings plus finance leases).
    Operating lease liabilities are excluded because operating lease cost is
    already deducted in operating income, so counting them as debt too would
    double count. Cash includes short-term investments. Both come from the
    most recent balance sheet, the closest available date to the market cap
    and share price. Book debt stands in for the market value of debt.

    Shares = current basic shares + latest-quarter dilution (diluted minus
    basic weighted-average shares), a treasury-stock-method approximation.
    """
    sheet = quarterly_bs if quarterly_bs is not None and not quarterly_bs.empty else annual_bs
    source = "quarterly balance sheet" if sheet is quarterly_bs else "annual balance sheet"
    as_of = None
    for col in sheet.columns:
        if not pd.isna(_first_available(
                sheet[col], "Total Debt", "Total Assets", "Long Term Debt And Capital Lease Obligation",
                "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")):
            as_of = col
            break
    if as_of is None:
        raise ValueError("No balance sheet available for the capital structure snapshot.")
    bal = sheet[as_of]

    total_debt = _safe_loc(bal, "Total Debt")
    debt_source = f"Total Debt, {source}"
    if pd.isna(total_debt):
        parts = [_safe_loc(bal, "Current Debt And Capital Lease Obligation"),
                 _safe_loc(bal, "Long Term Debt And Capital Lease Obligation")]
        if all(pd.isna(x) for x in parts):
            total_debt, debt_source = 0.0, "not reported; assumed 0"
        else:
            total_debt = float(np.nansum(parts))
            debt_source = f"current + long-term debt, {source}"

    cash = _first_available(bal, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")
    cash_source = f"cash + short-term investments, {source}"
    if pd.isna(cash):
        cash, cash_source = 0.0, "not reported; assumed 0"

    # Cost of debt pairs a full-year interest flow with average debt over the
    # same fiscal year, using the same Total Debt definition.
    annual_debt = annual_bs.loc["Total Debt"].dropna() if "Total Debt" in annual_bs.index else pd.Series(dtype=float)
    avg_debt = float(annual_debt.iloc[:2].mean()) if len(annual_debt) else total_debt

    basic = info.get("sharesOutstanding")
    if not basic:
        raise ValueError("sharesOutstanding is missing")
    dilution, share_source = 0.0, "basic only (diluted share data unavailable)"
    if quarterly_is is not None and not quarterly_is.empty and {
        "Diluted Average Shares", "Basic Average Shares"} <= set(quarterly_is.index):
        both = quarterly_is.loc[["Diluted Average Shares", "Basic Average Shares"]].dropna(axis=1)
        if not both.empty:
            col = both.columns[0]
            dilution = max(0.0, float(both.loc["Diluted Average Shares", col] - both.loc["Basic Average Shares", col]))
            share_source = f"basic + dilution from quarter ending {pd.Timestamp(col).date()}"

    today = pd.Timestamp(as_of_today) if as_of_today is not None else pd.Timestamp.today()
    return {
        "as_of": pd.Timestamp(as_of),
        "age_days": (today.normalize() - pd.Timestamp(as_of).normalize()).days,
        "total_debt": float(total_debt),
        "debt_source": debt_source,
        "avg_debt_for_interest": avg_debt,
        "cash": float(cash),
        "cash_source": cash_source,
        "shares_basic": float(basic),
        "dilutive_shares": dilution,
        "shares_diluted": float(basic) + dilution,
        "share_source": share_source,
        "market_cap": info.get("marketCap"),
    }


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
            # Non-operating liquid assets, excluded from operating working capital
            "Cash And Short Term Investments": _first_available(
                bal, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents",
            ),
            "Gross Margin": gross_profit / total_revenue if total_revenue else np.nan,
            # Operating income is the EBIT concept used in the projections, so
            # historical and projected margins are comparable.
            "EBIT Margin": operating_income / total_revenue if total_revenue else np.nan,
            "UFCF": unlevered_fcf(operating_income, fcf_tax, da, capex, change_wc),
            # Conventional (levered) FCF, shown for comparison only. It is after
            # interest, uses cash taxes, and adds back stock-based compensation,
            # so it is not the cash flow discounted at WACC.
            "OCF - CapEx": conventional_fcf(_safe_loc(cf, "Operating Cash Flow"), capex),
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
    capital = build_capital_structure(
        ticker.quarterly_balance_sheet, ticker.quarterly_income_stmt, balance_sheet, info,
    )

    return {
        "symbol": symbol,
        "ticker": ticker,
        "info": info,
        "hist_df": hist_df,
        "income_stmt": income_stmt,
        "missing": missing,
        "capital": capital,
        "warning": warning,
    }
