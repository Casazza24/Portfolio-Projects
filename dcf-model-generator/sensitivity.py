import pandas as pd

from config import revenue_growth_rates as build_growth_path
from dcf_engine import run_dcf
from projections import project_financials


def build_sensitivity_tables(projected_ufcfs, total_debt, total_cash, shares, wacc, terminal_growth, base_revenue, derived_assumptions):
    """Build WACC-vs-TGR and RevGrowth-vs-GM sensitivity tables."""

    # Table 1: WACC vs Terminal Growth Rate
    wacc_range = [0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115, 0.12]
    tgr_range = [0.015, 0.020, 0.025, 0.030, 0.035]

    rows = {}
    for wacc_val in wacc_range:
        row = {}
        for tgr in tgr_range:
            result = run_dcf(wacc_val, tgr, projected_ufcfs, total_debt, total_cash, shares)
            row[f"TGR {tgr:.1%}"] = round(result["implied_price"], 2)
        rows[f"{wacc_val:.1%}"] = row

    sens_df = pd.DataFrame(rows).T
    sens_df.index.name = "WACC"

    # Table 2: Revenue Growth vs Gross Margin
    growth_range = [0.04, 0.06, 0.08, 0.10, 0.12]
    margin_range = [0.43, 0.45, 0.47, 0.49, 0.51]
    projection_years = len(derived_assumptions["gross_margins"])

    rows2 = {}
    for growth in growth_range:
        row = {}
        for margin in margin_range:
            test_assumptions = dict(derived_assumptions)
            test_assumptions["gross_margins"] = [margin] * projection_years
            test_assumptions["revenue_growth_rates"] = build_growth_path(growth, terminal_growth)
            projections = project_financials(base_revenue, test_assumptions)
            ufcfs = [projections.loc["UFCF", f"Year {i}"] for i in range(1, projection_years + 1)]
            result = run_dcf(wacc, terminal_growth, ufcfs, total_debt, total_cash, shares)
            row[f"GM {margin:.0%}"] = round(result["implied_price"], 2)
        rows2[f"{growth:.0%}"] = row

    sens2_df = pd.DataFrame(rows2).T
    sens2_df.index.name = "Rev Growth Y1"

    return sens_df, sens2_df
