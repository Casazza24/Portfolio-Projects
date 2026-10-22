import numpy as np
import pandas as pd

from config import revenue_growth_rates as build_growth_path
from dcf_engine import run_dcf
from projections import project_financials


def centered_range(center, step, n_each_side):
    """Values center - n*step ... center + n*step (center included exactly)."""
    return [center + step * k for k in range(-n_each_side, n_each_side + 1)]


def build_sensitivity_tables(projected_ufcfs, total_debt, total_cash, shares, wacc, terminal_growth, base_revenue, derived_assumptions):
    """Build WACC-vs-TGR and RevGrowth-vs-GM sensitivity tables.

    Both grids are centered on this company's own base-case assumptions, so
    the middle cell of each table equals the base-case implied price. Cells
    where WACC <= terminal growth are undefined and left as NaN.
    """

    # Table 1: WACC vs Terminal Growth Rate
    wacc_range = centered_range(wacc, 0.005, 4)
    tgr_range = centered_range(terminal_growth, 0.005, 2)

    rows = {}
    for wacc_val in wacc_range:
        row = {}
        for tgr in tgr_range:
            if wacc_val <= tgr or wacc_val <= 0:
                row[f"TGR {tgr:.1%}"] = np.nan
                continue
            result = run_dcf(wacc_val, tgr, projected_ufcfs, total_debt, total_cash, shares)
            row[f"TGR {tgr:.1%}"] = round(result["implied_price"], 2)
        rows[f"{wacc_val:.2%}"] = row

    sens_df = pd.DataFrame(rows).T
    sens_df.index.name = "WACC"

    # Table 2: Year-1 Revenue Growth vs Gross Margin. The margin axis shifts
    # the whole base-case ramp, so the center column is the base case.
    projection_years = derived_assumptions["projection_years"]
    hold_years = derived_assumptions["hold_years"]
    base_growth = derived_assumptions["revenue_growth_rates"][0]
    base_margins = derived_assumptions["gross_margins"][:projection_years]
    growth_range = centered_range(base_growth, 0.02, 2)
    margin_shifts = centered_range(0.0, 0.02, 2)

    rows2 = {}
    for growth in growth_range:
        row = {}
        for shift in margin_shifts:
            test_assumptions = dict(derived_assumptions)
            test_assumptions["gross_margins"] = [m + shift for m in base_margins]
            test_assumptions["revenue_growth_rates"] = build_growth_path(
                growth, terminal_growth, projection_years, hold_years,
            )
            projections = project_financials(base_revenue, test_assumptions)
            ufcfs = [projections.loc["UFCF", f"Year {i}"] for i in range(1, projection_years + 1)]
            result = run_dcf(wacc, terminal_growth, ufcfs, total_debt, total_cash, shares)
            row[f"GM {base_margins[0] + shift:.0%}"] = round(result["implied_price"], 2)
        rows2[f"{growth:.1%}"] = row

    sens2_df = pd.DataFrame(rows2).T
    sens2_df.index.name = "Rev Growth Y1"

    return sens_df, sens2_df
