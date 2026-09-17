"""Derive assumptions from historical data and project financials."""

import numpy as np
import pandas as pd
from config import revenue_growth_rates as build_growth_path


def _ratio(numerator, denominator):
    """Mean of numerator/denominator across years, skipping zero denominators."""
    mask = denominator != 0
    if not mask.any():
        return 0.0
    return float(np.mean(numerator[mask] / denominator[mask]))


def derive_assumptions(hist_df, config_assumptions):
    """Build derived assumptions dict from historical DataFrame and config."""
    cols = list(hist_df.columns)
    ordered = list(reversed(cols))  # oldest-first

    rev = hist_df.loc["Total Revenue", ordered].values.astype(float)
    gp = hist_df.loc["Gross Profit", ordered].values.astype(float)
    sga = hist_df.loc["SG&A", ordered].values.astype(float)
    da = hist_df.loc["Depreciation And Amortization", ordered].values.astype(float)
    capex = hist_df.loc["Capital Expenditure", ordered].values.astype(float)
    tax = hist_df.loc["Tax Provision", ordered].values.astype(float)
    ebit = hist_df.loc["EBIT", ordered].values.astype(float)
    cur_assets = hist_df.loc["Current Assets", ordered].values.astype(float)
    cash = hist_df.loc["Cash And Cash Equivalents", ordered].values.astype(float)
    cur_liab = hist_df.loc["Current Liabilities", ordered].values.astype(float)
    cur_debt = hist_df.loc["Current Debt", ordered].values.astype(float)

    yoy = (rev[1:] - rev[:-1]) / rev[:-1]
    latest_growth = float(yoy[-1])
    historical_cagr = float((rev[-1] / rev[0]) ** (1.0 / (len(rev) - 1)) - 1)

    projection_years = config_assumptions["projection_years"]
    terminal_growth_rate = min(config_assumptions["terminal_growth_rate"], 0.04)

    nwc = cur_assets - cash - (cur_liab - cur_debt)

    hist_gm = _ratio(gp, rev)
    gross_margin_start = hist_gm if hist_gm > 0 else config_assumptions.get("gross_margin_start", 0.45)
    gross_margin_target = min(hist_gm + 0.05, 0.90) if hist_gm > 0 else config_assumptions.get("gross_margin_target", 0.50)
    gross_margins = [
        gross_margin_start + (gross_margin_target - gross_margin_start) * i / max(projection_years - 1, 1)
        for i in range(projection_years)
    ]

    return {
        "projection_years": projection_years,
        "revenue_growth_rates": build_growth_path(latest_growth, terminal_growth_rate),
        "gross_margin": _ratio(gp, rev),
        "gross_margins": gross_margins,
        "sga_pct_revenue": _ratio(sga, rev),
        "da_pct_revenue": _ratio(da, rev),
        "capex_pct_revenue": abs(_ratio(capex, rev)),
        "tax_rate": _ratio(tax, ebit),
        "nwc_pct_revenue": _ratio(nwc, rev),
        "terminal_growth_rate": terminal_growth_rate,
        "latest_growth": latest_growth,
        "historical_cagr": historical_cagr,
    }


def project_financials(base_revenue, assumptions):
    """Project financials for each year and return a DataFrame."""
    n = assumptions["projection_years"]
    growth_rates = assumptions["revenue_growth_rates"]
    gross_margins = assumptions.get("gross_margins") or [assumptions["gross_margin"]] * n
    sga_pct = assumptions["sga_pct_revenue"]
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
        gp = revenue * gross_margins[i]
        gm = gross_margins[i]
        sga = revenue * sga_pct
        da = revenue * da_pct
        ebitda = gp - sga
        ebit = ebitda - da
        ebit_margin = ebit / revenue if revenue != 0 else 0
        tax_provision = ebit * tax_rate
        nopat = ebit - tax_provision
        capex_val = -revenue * capex_pct
        delta_nwc = (revenue - prev_revenue) * nwc_pct
        ufcf = nopat + da - abs(capex_val) - delta_nwc

        data[col] = {
            "Total Revenue": revenue,
            "Gross Profit": gp,
            "Gross Margin": gm,
            "SG&A": sga,
            "Depreciation And Amortization": da,
            "EBITDA": ebitda,
            "EBIT": ebit,
            "EBIT Margin": ebit_margin,
            "Revenue Growth": g,
            "Tax Provision": tax_provision,
            "NOPAT": nopat,
            "Capital Expenditure": capex_val,
            "Change In Working Capital": delta_nwc,
            "UFCF": ufcf,
        }
        prev_revenue = revenue

    rows = [
        "Total Revenue", "Gross Profit", "Gross Margin", "SG&A",
        "Depreciation And Amortization", "EBITDA", "EBIT", "EBIT Margin",
        "Revenue Growth", "Tax Provision", "NOPAT", "Capital Expenditure",
        "Change In Working Capital", "UFCF",
    ]
    return pd.DataFrame(
        {col: [data[col][r] for r in rows] for col in data},
        index=rows,
    )
