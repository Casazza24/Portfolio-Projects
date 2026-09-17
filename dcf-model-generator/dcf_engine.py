"""WACC calculation and DCF valuation engine."""


def calculate_wacc(hist_df, info, income_stmt, tax_rate, risk_free_rate=0.04, equity_risk_premium=0.045):
    """Calculate weighted average cost of capital."""
    beta = info.get("beta", 1.0)
    market_cap = info.get("marketCap")
    if not market_cap:
        raise ValueError("market_cap is missing or zero")

    total_debt = hist_df.loc["Total Debt"].iloc[0]

    try:
        interest_series = income_stmt.loc["Interest Expense"].dropna()
        interest_expense = interest_series.iloc[0] if len(interest_series) > 0 else total_debt * 0.035
    except KeyError:
        interest_expense = total_debt * 0.035

    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    if total_debt == 0:
        return cost_of_equity

    cost_of_debt = abs(interest_expense) / total_debt
    total_capital = market_cap + total_debt
    equity_weight = market_cap / total_capital
    debt_weight = total_debt / total_capital

    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)
    return wacc


def run_dcf(wacc, terminal_growth, projected_ufcfs, total_debt, cash, shares):
    """Run discounted cash flow valuation and return results dict."""
    if not shares:
        raise ValueError("shares is missing or zero")
    if wacc <= terminal_growth:
        raise ValueError("wacc must be greater than terminal_growth")

    pv_fcfs = [ufcf / (1 + wacc) ** i for i, ufcf in enumerate(projected_ufcfs, 1)]
    terminal_value = projected_ufcfs[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** len(projected_ufcfs)
    enterprise_value = sum(pv_fcfs) + pv_terminal
    equity_value = enterprise_value - total_debt + cash
    implied_price = equity_value / shares

    return {
        "pv_fcfs": pv_fcfs,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "implied_price": implied_price,
    }
