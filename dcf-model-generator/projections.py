"""Derive assumptions from historical data and project financials."""

import numpy as np
import pandas as pd
from config import revenue_growth_rates as build_growth_path


def _ratio(numerator, denominator):
    """Mean of numerator/denominator across years, skipping zero denominators and NaNs.

    Returns NaN (not 0.0) when no year is usable, so callers must decide how
    to handle the gap instead of silently forecasting a zero.
    """
    valid = (denominator != 0) & np.isfinite(numerator) & np.isfinite(denominator)
    if not valid.any():
        return np.nan
    return float(np.nanmean(numerator[valid] / denominator[valid]))


def _require(name, value):
    if not np.isfinite(value):
        raise ValueError(f"Cannot derive '{name}': no historical year has the required data.")
    return value


def build_gross_margin_ramp(start, target, projection_years):
    """Linear ramp from start to target over the projection window."""
    return [
        start + (target - start) * i / max(projection_years - 1, 1)
        for i in range(projection_years)
    ]


def derive_assumptions(hist_df, config_assumptions):
    """Build derived assumptions dict from historical DataFrame and config."""
    cols = list(hist_df.columns)
    ordered = list(reversed(cols))  # oldest-first

    def row(label):
        if label not in hist_df.index:
            return np.full(len(ordered), np.nan)
        return hist_df.loc[label, ordered].values.astype(float)

    rev = row("Total Revenue")
    gp = row("Gross Profit")
    sga = row("SG&A")
    rnd = row("Research And Development")
    op_inc = row("Operating Income")
    da = row("Depreciation And Amortization")
    capex = row("Capital Expenditure")
    eff_tax = row("Effective Tax Rate")
    cur_assets = row("Current Assets")
    cash = row("Cash And Cash Equivalents")
    cur_liab = row("Current Liabilities")
    cur_debt = row("Current Debt")

    if len(rev) < 2:
        raise ValueError("At least two fiscal years of revenue are needed to derive growth.")
    yoy = (rev[1:] - rev[:-1]) / rev[:-1]
    latest_growth = float(yoy[-1])
    historical_cagr = float((rev[-1] / rev[0]) ** (1.0 / (len(rev) - 1)) - 1)

    projection_years = config_assumptions["projection_years"]
    hold_years = config_assumptions["hold_years"]
    terminal_growth_rate = min(config_assumptions["terminal_growth_rate"], 0.04)

    # Operating working capital: excludes cash and short-term debt. Missing
    # balance-sheet lines propagate as NaN and are skipped by _ratio.
    nwc = cur_assets - cash - (cur_liab - cur_debt)

    # Operating expenses are everything between gross profit and reported
    # operating income, so EBIT reconciles to the income statement. D&A is
    # already embedded in COGS/opex and is NOT subtracted again.
    hist_gm = _require("gross_margin", _ratio(gp, rev))
    opex_pct = _require("operating expenses", _ratio(gp - op_inc, rev))
    sga_pct = _ratio(sga, rev)
    sga_pct = 0.0 if not np.isfinite(sga_pct) else sga_pct
    rnd_pct = _ratio(rnd, rev)
    rnd_pct = 0.0 if not np.isfinite(rnd_pct) else rnd_pct
    other_opex_pct = opex_pct - sga_pct - rnd_pct

    expansion = config_assumptions.get("gross_margin_expansion", 0.05)
    gross_margin_start = hist_gm
    gross_margin_target = min(hist_gm + expansion, 0.90)

    # Historical effective rates (tax / pretax income) vs the normalized rate
    # used for forecasting. The median is robust to one-off years such as a
    # large discrete tax charge; abnormal years are already NaN.
    valid_tax = eff_tax[np.isfinite(eff_tax)]
    if len(valid_tax):
        tax_rate = float(np.median(valid_tax))
        tax_rate_source = f"median of {len(valid_tax)} normal year(s)"
    else:
        tax_rate = config_assumptions["fallback_tax_rate"]
        tax_rate_source = "fallback (no normal historical year)"

    return {
        "projection_years": projection_years,
        "hold_years": hold_years,
        "revenue_growth_rates": build_growth_path(
            latest_growth, terminal_growth_rate, projection_years, hold_years,
        ),
        "gross_margin": hist_gm,
        "gross_margin_expansion": expansion,
        "gross_margins": build_gross_margin_ramp(gross_margin_start, gross_margin_target, projection_years),
        "sga_pct_revenue": sga_pct,
        "rnd_pct_revenue": rnd_pct,
        "other_opex_pct_revenue": other_opex_pct,
        "operating_margin_hist": _ratio(op_inc, rev),
        "da_pct_revenue": _require("da_pct_revenue", _ratio(da, rev)),
        "capex_pct_revenue": abs(_require("capex_pct_revenue", _ratio(capex, rev))),
        "tax_rate": tax_rate,
        "tax_rate_source": tax_rate_source,
        "effective_tax_rates": dict(zip(ordered, eff_tax)),
        "nwc_pct_revenue": _require("nwc_pct_revenue", _ratio(nwc, rev)),
        "terminal_growth_rate": terminal_growth_rate,
        "risk_free_rate": config_assumptions["risk_free_rate"],
        "equity_risk_premium": config_assumptions["equity_risk_premium"],
        "latest_growth": latest_growth,
        "historical_cagr": historical_cagr,
    }


def apply_overrides(derived, overrides):
    """Return a copy of derived with user overrides applied consistently.

    Dependent schedules are rebuilt so an override can't silently conflict
    with them: changing Year-1 growth, terminal growth, the horizon or hold
    period rebuilds the growth path, and changing the gross margin rebuilds
    the margin ramp.
    """
    result = dict(derived)
    if not overrides:
        return result
    unknown = set(overrides) - set(derived)
    if unknown:
        raise KeyError(f"Unknown assumption override(s): {sorted(unknown)}")
    result.update(overrides)

    n = result["projection_years"]
    growth_keys = {"latest_growth", "terminal_growth_rate", "projection_years", "hold_years"}
    if growth_keys & set(overrides) and "revenue_growth_rates" not in overrides:
        result["revenue_growth_rates"] = build_growth_path(
            result["latest_growth"], result["terminal_growth_rate"], n, result["hold_years"],
        )
    margin_keys = {"gross_margin", "gross_margin_expansion", "projection_years"}
    if margin_keys & set(overrides) and "gross_margins" not in overrides:
        start = result["gross_margin"]
        target = min(start + result["gross_margin_expansion"], 0.90)
        result["gross_margins"] = build_gross_margin_ramp(start, target, n)
    return result


def project_financials(base_revenue, assumptions):
    """Project financials for each year and return a DataFrame.

    Sign conventions match the cash flow statement: Capital Expenditure is
    negative (outflow) and Change In Working Capital is the cash impact
    (negative when working capital grows), so UFCF = NOPAT + D&A + CapEx + dNWC
    for both historical and projected columns.
    """
    n = assumptions["projection_years"]
    growth_rates = assumptions["revenue_growth_rates"]
    gross_margins = assumptions.get("gross_margins") or [assumptions["gross_margin"]] * n
    if len(growth_rates) < n or len(gross_margins) < n:
        raise ValueError(
            f"projection_years={n} but got {len(growth_rates)} growth rates "
            f"and {len(gross_margins)} gross margins"
        )
    sga_pct = assumptions["sga_pct_revenue"]
    rnd_pct = assumptions.get("rnd_pct_revenue", 0.0)
    other_pct = assumptions.get("other_opex_pct_revenue", 0.0)
    da_pct = assumptions["da_pct_revenue"]
    capex_pct = assumptions["capex_pct_revenue"]
    tax_rate = assumptions["tax_rate"]
    nwc_pct = assumptions["nwc_pct_revenue"]

    data = {}
    prev_revenue = base_revenue

    for i in range(n):
        col = f"Year {i + 1}"
        g = growth_rates[i]
        revenue = prev_revenue * (1 + g)
        gm = gross_margins[i]
        gp = revenue * gm
        sga = revenue * sga_pct
        rnd = revenue * rnd_pct
        other_opex = revenue * other_pct
        da = revenue * da_pct
        ebit = gp - sga - rnd - other_opex
        ebitda = ebit + da
        ebit_margin = ebit / revenue if revenue != 0 else 0
        tax_provision = ebit * tax_rate
        nopat = ebit - tax_provision
        capex_val = -revenue * capex_pct
        change_wc = -(revenue - prev_revenue) * nwc_pct
        ufcf = nopat + da + capex_val + change_wc

        data[col] = {
            "Total Revenue": revenue,
            "Gross Profit": gp,
            "Gross Margin": gm,
            "SG&A": sga,
            "Research And Development": rnd,
            "Other Operating Expenses": other_opex,
            "Depreciation And Amortization": da,
            "EBITDA": ebitda,
            "EBIT": ebit,
            "EBIT Margin": ebit_margin,
            "Revenue Growth": g,
            "Tax Provision": tax_provision,
            "NOPAT": nopat,
            "Capital Expenditure": capex_val,
            "Change In Working Capital": change_wc,
            "UFCF": ufcf,
        }
        prev_revenue = revenue

    rows = list(next(iter(data.values())).keys())
    return pd.DataFrame(
        {col: [data[col][r] for r in rows] for col in data},
        index=rows,
    )
