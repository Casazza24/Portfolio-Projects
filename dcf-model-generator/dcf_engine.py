"""WACC calculation and DCF valuation engine."""


def calculate_wacc(capital, info, income_stmt, tax_rate, *, risk_free_rate, equity_risk_premium):
    """Calculate weighted average cost of capital.

    capital is the snapshot from data_fetcher.build_capital_structure; its
    total_debt is the same figure deducted in the equity bridge, so WACC
    weights and the EV-to-equity bridge use one debt definition and date.
    Cost of debt = annual interest / average debt over that fiscal year.

    risk_free_rate and equity_risk_premium are required so the configured or
    user-overridden values always flow through (no hidden defaults).
    Returns a dict with the WACC value and all component assumptions.
    """
    beta = info.get("beta")
    beta_source = "reported"
    if beta is None or beta != beta:
        beta = 1.0
        beta_source = "assumed (1.0, beta unavailable)"
    market_cap = capital.get("market_cap") or info.get("marketCap")
    if not market_cap:
        raise ValueError("market_cap is missing or zero")

    total_debt = capital["total_debt"]
    avg_debt = capital.get("avg_debt_for_interest") or total_debt

    interest_expense_source = 'reported'
    try:
        interest_series = income_stmt.loc["Interest Expense"].dropna()
        interest_expense = interest_series.iloc[0] if len(interest_series) > 0 else None
    except KeyError:
        interest_expense = None

    if interest_expense is None:
        interest_expense = avg_debt * 0.035
        interest_expense_source = 'estimated (3.5% of debt)'

    cost_of_equity = risk_free_rate + beta * equity_risk_premium

    if total_debt == 0:
        return {
            'wacc': cost_of_equity,
            'cost_of_equity': cost_of_equity,
            'cost_of_debt': 0.0,
            'equity_weight': 1.0,
            'debt_weight': 0.0,
            'beta': beta,
            'beta_source': beta_source,
            'risk_free_rate': risk_free_rate,
            'equity_risk_premium': equity_risk_premium,
            'market_cap': market_cap,
            'total_debt': 0,
            'interest_expense': 0,
            'interest_expense_source': interest_expense_source,
            'tax_rate': tax_rate,
        }

    cost_of_debt = abs(interest_expense) / avg_debt if avg_debt else 0.0
    total_capital = market_cap + total_debt
    equity_weight = market_cap / total_capital
    debt_weight = total_debt / total_capital

    wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)
    return {
        'wacc': wacc,
        'cost_of_equity': cost_of_equity,
        'cost_of_debt': cost_of_debt,
        'equity_weight': equity_weight,
        'debt_weight': debt_weight,
        'beta': beta,
        'beta_source': beta_source,
        'risk_free_rate': risk_free_rate,
        'equity_risk_premium': equity_risk_premium,
        'market_cap': market_cap,
        'total_debt': total_debt,
        'interest_expense': interest_expense,
        'interest_expense_source': interest_expense_source,
        'tax_rate': tax_rate,
    }


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
        "total_debt": total_debt,
        "cash": cash,
        "shares": shares,
        "pv_fcfs": pv_fcfs,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "implied_price": implied_price,
    }
