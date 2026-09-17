import yfinance as yf
import pandas as pd


def _safe_loc(statement, label, default=0):
    """Return a row from a financial statement, or default if missing."""
    try:
        return statement.loc[label]
    except KeyError:
        return default


def fetch_financial_data(symbol):
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

    historical = {}
    for year in income_stmt.columns:
        total_revenue = _safe_loc(income_stmt[year], "Total Revenue")
        gross_profit = _safe_loc(income_stmt[year], "Gross Profit")
        ebit = _safe_loc(income_stmt[year], "EBIT")
        tax_provision = _safe_loc(income_stmt[year], "Tax Provision")
        pretax_income = _safe_loc(income_stmt[year], "Pretax Income")
        da = _safe_loc(cashflow[year], "Depreciation And Amortization")
        capex = _safe_loc(cashflow[year], "Capital Expenditure")
        change_wc = _safe_loc(cashflow[year], "Change In Working Capital")

        tax_rate = (
            tax_provision / pretax_income
            if pretax_income != 0
            else 0
        )

        historical[year] = {
            "Total Revenue": total_revenue,
            "Cost Of Revenue": _safe_loc(income_stmt[year], "Cost Of Revenue"),
            "Gross Profit": gross_profit,
            "EBITDA": _safe_loc(income_stmt[year], "EBITDA"),
            "EBIT": ebit,
            "Tax Provision": tax_provision,
            "Net Income": _safe_loc(income_stmt[year], "Net Income"),
            "SG&A": _safe_loc(income_stmt[year], "Selling General And Administration"),
            "Capital Expenditure": capex,
            "Depreciation And Amortization": da,
            "Change In Working Capital": change_wc,
            "Current Assets": _safe_loc(balance_sheet[year], "Current Assets"),
            "Current Liabilities": _safe_loc(balance_sheet[year], "Current Liabilities"),
            "Current Debt": _safe_loc(balance_sheet[year], "Current Debt", default=0),
            "Total Debt": _safe_loc(balance_sheet[year], "Total Debt", default=0),
            "Cash And Cash Equivalents": _safe_loc(
                balance_sheet[year], "Cash And Cash Equivalents"
            ),
            "Gross Margin": gross_profit / total_revenue if total_revenue != 0 else 0,
            "EBIT Margin": ebit / total_revenue if total_revenue != 0 else 0,
            "Free Cash Flow": (
                ebit * (1 - tax_rate) + da - capex + change_wc
            ),
        }

    hist_df = pd.DataFrame(historical)
    hist_df.loc["Revenue Growth"] = hist_df.loc["Total Revenue"].pct_change(periods=-1)

    return {
        "symbol": symbol,
        "ticker": ticker,
        "info": info,
        "hist_df": hist_df,
        "income_stmt": income_stmt,
        "warning": warning,
    }
