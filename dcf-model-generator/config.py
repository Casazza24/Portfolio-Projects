# Tunable inputs. Everything not listed here is derived from the historical
# statements in dcf.py.
assumptions = {
    # Projection window. Longer window + longer hold = slower fade to terminal.
    "projection_years": 8,
    "hold_years": 3,          # years at the latest growth rate before fading

    # Gross margin ramps linearly from start to target over the window,
    # reflecting the shift in mix toward higher-margin services.
    "gross_margin_start": 0.45,
    "gross_margin_target": 0.50,

    # Cost of equity input. Street practice runs 4.5%-5.5%; the low end
    # reflects the compressed premium on mega-cap quality names.
    "equity_risk_premium": 0.045,
    "risk_free_rate": 0.04,

    # Legacy flat-rate fallbacks, kept for reference / sensitivity overrides.
    "gross_margin": 0.451,
    "sga_pct_revenue": 0.065,
    "da_pct_revenue": 0.029,
    "capex_pct_revenue": 0.028,
    "tax_rate": 0.176,
    "nwc_pct_revenue": -0.065,
    "terminal_growth_rate": 0.025,
}


def revenue_growth_rates(
    latest_growth,
    terminal_growth_rate,
    projection_years=None,
    hold_years=None,
):
    """Hold at latest_growth, then fade linearly to the terminal rate.

    Year `hold_years + fade_years` lands exactly on terminal_growth_rate.
    Called by dcf.py once the historical growth figures are known, and again
    per cell of the growth sensitivity grid with a different starting rate.
    """
    if projection_years is None:
        projection_years = assumptions["projection_years"]
    if hold_years is None:
        hold_years = assumptions["hold_years"]
    fade_years = projection_years - hold_years

    return [
        latest_growth
        if i < hold_years
        else latest_growth
        + (terminal_growth_rate - latest_growth)
        * ((i - hold_years + 1) / fade_years)
        for i in range(projection_years)
    ]


assumptions["revenue_growth_rates"] = revenue_growth_rates
