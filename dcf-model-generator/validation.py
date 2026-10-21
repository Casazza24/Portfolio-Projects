"""Accounting identity and sanity checks on fetched financial data."""

import pandas as pd


def validate_financials(hist_df):
    """Run accounting identity checks on hist_df. Return list of warning strings."""
    warnings = []

    for col in hist_df.columns:
        year_label = col.year if hasattr(col, 'year') else str(col)

        revenue = _safe(hist_df, 'Total Revenue', col)
        cogs = _safe(hist_df, 'Cost Of Revenue', col)
        gross_profit_reported = _safe(hist_df, 'Gross Profit', col)
        if revenue and cogs and gross_profit_reported:
            computed = revenue - cogs
            if abs(computed - gross_profit_reported) > abs(revenue) * 0.01:
                warnings.append(
                    f"{year_label}: Gross Profit mismatch: "
                    f"Revenue - COGS = {computed:,.0f}, "
                    f"reported = {gross_profit_reported:,.0f}"
                )

        if revenue and revenue < 0:
            warnings.append(f"{year_label}: Negative revenue ({revenue:,.0f})")

        ebit = _safe(hist_df, 'EBIT', col)
        ebitda = _safe(hist_df, 'EBITDA', col)
        da = _safe(hist_df, 'Depreciation And Amortization', col)
        if ebitda and da and ebit:
            computed_ebit = ebitda - abs(da)
            if abs(computed_ebit - ebit) > abs(ebitda) * 0.05:
                warnings.append(
                    f"{year_label}: EBIT vs EBITDA-D&A gap: "
                    f"computed = {computed_ebit:,.0f}, reported EBIT = {ebit:,.0f}"
                )

    return warnings


def _safe(df, label, col, default=None):
    """Safely get a value from a DataFrame."""
    try:
        val = df.loc[label, col]
        return val if pd.notna(val) and val != 0 else default
    except (KeyError, TypeError):
        return default
