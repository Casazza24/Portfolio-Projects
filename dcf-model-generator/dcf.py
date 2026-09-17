"""CLI entry point: run a DCF analysis for any ticker."""

import sys
from datetime import datetime

import pandas as pd

from config import assumptions
from data_fetcher import fetch_financial_data
from projections import derive_assumptions, project_financials
from dcf_engine import calculate_wacc, run_dcf
from sensitivity import build_sensitivity_tables
from edgar import analyze_filing, empty_filing, print_report
from excel_output import generate_excel


def smart_format(x):
    if abs(x) < 1:
        return f"{x:.1%}"
    return f"{x:,.0f}"


def run_analysis(symbol):
    """Run the full DCF pipeline for a ticker and save an Excel workbook."""
    data = fetch_financial_data(symbol)
    info = data["info"]
    hist_df = data["hist_df"]
    income_stmt = data["income_stmt"]

    if data["warning"]:
        print(f"\n  ⚠ {data['warning']}")

    derived = derive_assumptions(hist_df, assumptions)

    print("\n=== DERIVED ASSUMPTIONS ===")
    print(f"  Latest YoY Growth:     {derived['latest_growth']:.1%}")
    print(f"  Historical CAGR:       {derived['historical_cagr']:.1%}")
    print(f"  Projection Years:      {derived['projection_years']}")
    print("  Revenue Growth Rates:  " + ", ".join(f"{g:.1%}" for g in derived["revenue_growth_rates"]))
    print(f"  Gross Margin (hist):   {derived['gross_margin']:.1%}")
    print("  Gross Margin Ramp:     " + ", ".join(f"{m:.1%}" for m in derived["gross_margins"]))
    print(f"  SG&A % Revenue:        {derived['sga_pct_revenue']:.1%}")
    print(f"  D&A % Revenue:         {derived['da_pct_revenue']:.1%}")
    print(f"  CapEx % Revenue:       {derived['capex_pct_revenue']:.1%}")
    print(f"  Tax Rate:              {derived['tax_rate']:.1%}")
    print(f"  NWC % Revenue (delta): {derived['nwc_pct_revenue']:.1%}")
    print(f"  Terminal Growth Rate:  {derived['terminal_growth_rate']:.1%}")

    base_revenue = hist_df.loc["Total Revenue"].iloc[0]
    projected = project_financials(base_revenue, derived)
    full_df = pd.concat([hist_df, projected], axis=1)

    wacc = calculate_wacc(hist_df, info, income_stmt, derived["tax_rate"])
    projection_years = derived["projection_years"]
    projected_ufcfs = [projected.loc["UFCF", f"Year {i}"] for i in range(1, projection_years + 1)]

    total_debt = info.get("totalDebt", 0)
    total_cash = info.get("totalCash", 0)
    shares = info.get("sharesOutstanding")
    current_price = info.get("currentPrice")

    dcf_results = run_dcf(wacc, derived["terminal_growth_rate"], projected_ufcfs, total_debt, total_cash, shares)
    implied_price = dcf_results["implied_price"]

    print("\n=== DCF Valuation ===")
    for year_num, (ufcf, pv) in enumerate(zip(projected_ufcfs, dcf_results["pv_fcfs"]), 1):
        print(f"  Year {year_num}: UFCF = ${ufcf:,.0f}  |  PV = ${pv:,.0f}")
    print(f"  Total PV of FCFs: ${sum(dcf_results['pv_fcfs']):,.0f}")
    print(f"  Terminal Value: ${dcf_results['terminal_value']:,.0f}")
    print(f"  PV of Terminal Value: ${dcf_results['pv_terminal']:,.0f}")
    print(f"  Enterprise Value: ${dcf_results['enterprise_value']:,.0f}")
    print(f"  Less Debt: ${total_debt:,.0f}")
    print(f"  Plus Cash: ${total_cash:,.0f}")
    print(f"  Equity Value: ${dcf_results['equity_value']:,.0f}")
    print(f"  Implied Price: ${implied_price:.2f}")
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Upside/Downside: {((implied_price / current_price) - 1):.1%}")

    sens_df, sens2_df = build_sensitivity_tables(
        projected_ufcfs, total_debt, total_cash, shares, wacc,
        derived["terminal_growth_rate"], base_revenue, derived,
    )

    try:
        filing_analysis = analyze_filing(symbol, "10-Q")
        print_report(filing_analysis)
    except Exception as e:
        print(f"\n  Warning: Could not fetch EDGAR filing: {e}")
        filing_analysis = empty_filing(symbol)

    filename = f"{symbol.upper()}_{datetime.now().strftime('%Y-%m-%d')}_dcf_model.xlsx"
    generate_excel(
        ticker=symbol,
        company_name=info.get("shortName", symbol),
        current_price=current_price,
        implied_price=implied_price,
        wacc=wacc,
        assumptions=derived,
        full_df=full_df,
        dcf_results=dcf_results,
        filing_analysis=filing_analysis,
        sens_df=sens_df,
        sens2_df=sens2_df,
        filename=filename,
    )


if __name__ == "__main__":
    pd.set_option("display.float_format", smart_format)
    if len(sys.argv) > 1:
        ticker_input = sys.argv[1].strip().upper()
    else:
        ticker_input = input("Enter ticker symbol: ").strip().upper()
    try:
        run_analysis(ticker_input)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
