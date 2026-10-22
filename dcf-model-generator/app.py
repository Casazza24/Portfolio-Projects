"""Streamlit dashboard for the DCF model generator."""

import math

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from config import assumptions
from data_fetcher import fetch_financial_data
from projections import derive_assumptions, project_financials, apply_overrides
from dcf_engine import calculate_wacc, run_dcf
from sensitivity import build_sensitivity_tables
from edgar import analyze_filing, empty_filing
from excel_output import generate_excel_bytes
from validation import validate_financials, xbrl_issues, is_reliable, input_provenance, valuation_sanity_checks
from xbrl_fetcher import (
    fetch_xbrl_facts, extract_annual_financials, validate_against_yfinance, summarize_xbrl_report,
)

st.set_page_config(page_title="DCF Model Generator", page_icon="📊", layout="wide")


@st.cache_resource(ttl=300)
def fetch_data(symbol):
    """Fetch raw financial data (cached separately from model run)."""
    return fetch_financial_data(symbol, assumptions["fallback_tax_rate"])


@st.cache_data(ttl=300)
def fetch_xbrl(symbol):
    """Fetch XBRL data from SEC EDGAR (cached)."""
    try:
        raw = fetch_xbrl_facts(symbol)
        return extract_annual_financials(raw)
    except Exception:
        return None


def run_model(symbol, overrides=None):
    """Run the full DCF pipeline with optional assumption overrides."""
    data = fetch_data(symbol)
    info = data["info"]
    hist_df = data["hist_df"]
    income_stmt = data["income_stmt"]
    capital = data["capital"]

    derived = apply_overrides(derive_assumptions(hist_df, assumptions), overrides)

    base_revenue = hist_df.loc["Total Revenue"].iloc[0]
    projected = project_financials(base_revenue, derived)
    full_df = pd.concat([hist_df, projected], axis=1)

    wacc_result = calculate_wacc(
        capital, info, income_stmt, derived["tax_rate"],
        risk_free_rate=derived["risk_free_rate"],
        equity_risk_premium=derived["equity_risk_premium"],
    )
    wacc = wacc_result['wacc']

    projection_years = derived["projection_years"]
    projected_ufcfs = [projected.loc["UFCF", f"Year {i}"] for i in range(1, projection_years + 1)]

    # Same dated snapshot as the WACC weights (see build_capital_structure)
    total_debt = capital["total_debt"]
    total_cash = capital["cash"]
    shares = capital["shares_diluted"]
    current_price = info.get("currentPrice")

    dcf_results = run_dcf(wacc, derived["terminal_growth_rate"], projected_ufcfs, total_debt, total_cash, shares)

    sens_df, sens2_df = build_sensitivity_tables(
        projected_ufcfs, total_debt, total_cash, shares, wacc,
        derived["terminal_growth_rate"], base_revenue, derived,
    )

    try:
        filing_analysis = analyze_filing(symbol, "10-Q")
    except Exception:
        filing_analysis = None

    xbrl_data = fetch_xbrl(symbol)
    xbrl_report = validate_against_yfinance(symbol, xbrl_data, hist_df) if xbrl_data else []
    xbrl_summary = summarize_xbrl_report(xbrl_report)
    sanity_issues, sanity_metrics = valuation_sanity_checks(
        dcf_results, derived, projected, capital, info.get("country"),
    )
    issues = (validate_financials(hist_df, data.get("missing"), derived, wacc_result)
              + xbrl_issues(xbrl_report, xbrl_summary) + sanity_issues)

    return {
        "info": info,
        "hist_df": hist_df,
        "income_stmt": income_stmt,
        "derived": derived,
        "full_df": full_df,
        "wacc": wacc,
        "wacc_result": wacc_result,
        "dcf_results": dcf_results,
        "current_price": current_price,
        "implied_price": dcf_results["implied_price"],
        "total_debt": total_debt,
        "total_cash": total_cash,
        "shares": shares,
        "projected_ufcfs": projected_ufcfs,
        "sens_df": sens_df,
        "sens2_df": sens2_df,
        "filing_analysis": filing_analysis,
        "warning": data.get("warning"),
        "issues": issues,
        "capital": capital,
        "sanity_metrics": sanity_metrics,
        "reliable": is_reliable(issues),
        "xbrl_report": xbrl_report,
        "xbrl_summary": xbrl_summary,
        "provenance": input_provenance(hist_df, data.get("missing"), xbrl_report),
    }


def render_sidebar_overrides(derived):
    """Render assumption override controls in the sidebar. Return overrides dict."""
    overrides = {}

    with st.sidebar:
        st.header("Assumption Overrides")
        st.caption("Adjust any input below to re-run the model.")

        st.subheader("WACC Inputs")
        overrides["risk_free_rate"] = st.number_input(
            "Risk-Free Rate", value=derived["risk_free_rate"],
            min_value=0.0, max_value=0.15, step=0.005, format="%.3f",
        )
        overrides["equity_risk_premium"] = st.number_input(
            "Equity Risk Premium", value=derived["equity_risk_premium"],
            min_value=0.0, max_value=0.15, step=0.005, format="%.3f",
        )

        st.subheader("Growth")
        overrides["terminal_growth_rate"] = st.number_input(
            "Terminal Growth Rate", value=derived["terminal_growth_rate"],
            min_value=0.0, max_value=0.05, step=0.005, format="%.3f",
        )
        # Growth path (hold, then fade to terminal) is rebuilt by apply_overrides.
        overrides["latest_growth"] = st.number_input(
            "Year 1 Revenue Growth", value=derived["latest_growth"],
            min_value=-0.50, max_value=1.0, step=0.01, format="%.3f",
        )

        st.subheader("Margins & Costs")
        overrides["gross_margin"] = st.number_input(
            "Gross Margin", value=derived["gross_margin"],
            min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
        )
        overrides["gross_margin_expansion"] = st.number_input(
            "Gross Margin Expansion (over window)", value=derived["gross_margin_expansion"],
            min_value=-0.20, max_value=0.20, step=0.01, format="%.3f",
        )

        overrides["sga_pct_revenue"] = st.number_input(
            "SG&A % Revenue", value=derived["sga_pct_revenue"],
            min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
        )
        overrides["rnd_pct_revenue"] = st.number_input(
            "R&D % Revenue", value=derived["rnd_pct_revenue"],
            min_value=0.0, max_value=1.0, step=0.01, format="%.3f",
        )
        overrides["da_pct_revenue"] = st.number_input(
            "D&A % Revenue", value=derived["da_pct_revenue"],
            min_value=0.0, max_value=0.50, step=0.005, format="%.3f",
        )
        overrides["capex_pct_revenue"] = st.number_input(
            "CapEx % Revenue", value=derived["capex_pct_revenue"],
            min_value=0.0, max_value=0.50, step=0.005, format="%.3f",
        )
        overrides["tax_rate"] = st.number_input(
            "Tax Rate", value=derived["tax_rate"],
            min_value=0.0, max_value=0.50, step=0.01, format="%.3f",
        )
        overrides["nwc_pct_revenue"] = st.number_input(
            "NWC % Revenue", value=derived["nwc_pct_revenue"],
            min_value=-0.50, max_value=0.50, step=0.005, format="%.3f",
        )

    changed = {k: v for k, v in overrides.items() if k in derived and v != derived.get(k)}
    return changed if changed else None


def render_summary(results):
    """Render the valuation summary section."""
    info = results["info"]
    implied = results["implied_price"]
    current = results["current_price"]
    upside = (implied - current) / current

    company_name = info.get("shortName", "")
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")

    st.markdown(f"## {company_name}")
    st.caption(f"{sector}  ·  {industry}  ·  {datetime.now().strftime('%B %d, %Y')}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${current:,.2f}")
    col2.metric("Implied Price", f"${implied:,.2f}")
    col3.metric("Upside / Downside", f"{upside:+.1%}")
    col4.metric("WACC", f"{results['wacc']:.2%}")

    critical = [i for i in results["issues"] if i["severity"] == "critical"]
    if critical:
        st.error(
            "**Valuation not reliable.** Critical data-validation issues were found; "
            "the implied price above is diagnostic only. See the Data Validation tab.\n\n"
            + "\n".join(f"- {i['message']}" for i in critical)
        )


def render_wacc_decomposition(wacc_result):
    """Render the WACC decomposition panel."""
    st.subheader("WACC Decomposition")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Cost of Equity", f"{wacc_result['cost_of_equity']:.2%}",
        help=f"CAPM: {wacc_result['risk_free_rate']:.1%} + {wacc_result['beta']:.2f} × {wacc_result['equity_risk_premium']:.1%}",
    )
    col2.metric(
        "Cost of Debt (after-tax)",
        f"{wacc_result['cost_of_debt'] * (1 - wacc_result['tax_rate']):.2%}",
    )
    col3.metric("WACC", f"{wacc_result['wacc']:.2%}")

    st.caption(
        f"Capital structure: {wacc_result['equity_weight']:.1%} equity, "
        f"{wacc_result['debt_weight']:.1%} debt  ·  "
        f"Beta: {wacc_result['beta']:.2f}"
    )
    if 'estimated' in wacc_result['interest_expense_source']:
        st.warning("Interest expense not found in filings. Using 3.5% of total debt as estimate.")


def render_assumptions(derived):
    """Render the derived assumptions panel."""
    st.subheader("Derived Assumptions")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Growth**")
        st.write(f"Latest YoY Growth: {derived['latest_growth']:.1%}")
        st.write(f"Historical CAGR: {derived['historical_cagr']:.1%}")
        st.write(f"Terminal Growth: {derived['terminal_growth_rate']:.1%}")

    with col2:
        st.markdown("**Margins & Costs**")
        st.write(f"Gross Margin (hist avg): {derived['gross_margin']:.1%}")
        st.write(f"SG&A % Revenue: {derived['sga_pct_revenue']:.1%}")
        st.write(f"R&D % Revenue: {derived['rnd_pct_revenue']:.1%}")
        st.write(f"Other Opex % Revenue: {derived['other_opex_pct_revenue']:.1%}")
        st.write(f"D&A % Revenue: {derived['da_pct_revenue']:.1%}")

    with col3:
        st.markdown("**Capital**")
        st.write(f"CapEx % Revenue: {derived['capex_pct_revenue']:.1%}")
        st.write(f"Tax Rate: {derived['tax_rate']:.1%} ({derived['tax_rate_source']})")
        st.write(f"NWC % Revenue: {derived['nwc_pct_revenue']:.1%}")


def render_financials(full_df):
    """Render the historical + projected financials table."""
    st.subheader("Historical & Projected Financials")
    st.caption("$ in millions")

    pct_rows = {"Gross Margin", "EBIT Margin", "Revenue Growth"}
    display = full_df.copy().astype(object)

    for col in display.columns:
        for idx in display.index:
            val = full_df.loc[idx, col]
            if pd.isna(val):
                display.loc[idx, col] = ""
            elif idx in pct_rows:
                display.loc[idx, col] = f"{val:.1%}"
            else:
                display.loc[idx, col] = f"{val / 1e6:,.0f}"

    display.columns = [str(c.year) if hasattr(c, "year") else str(c) for c in display.columns]
    st.dataframe(display, use_container_width=True, height=600)


def render_dcf_bridge(results):
    """Render the DCF valuation bridge."""
    st.subheader("DCF Valuation Bridge")

    dcf = results["dcf_results"]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Equity Bridge**")
        bridge_data = {
            "Sum of PV of FCFs": sum(dcf["pv_fcfs"]) / 1e6,
            "+ PV of Terminal Value": dcf["pv_terminal"] / 1e6,
            "= Enterprise Value": dcf["enterprise_value"] / 1e6,
            "- Total Debt": results["total_debt"] / 1e6,
            "+ Cash & ST Investments": results["total_cash"] / 1e6,
            "= Equity Value": dcf["equity_value"] / 1e6,
        }
        bridge_df = pd.DataFrame({"$ Millions": bridge_data.values()}, index=bridge_data.keys())
        bridge_df["$ Millions"] = bridge_df["$ Millions"].map(lambda x: f"{x:,.0f}")
        st.dataframe(bridge_df, use_container_width=True)
        cap = results["capital"]
        st.caption(
            f"Debt and cash as of {cap['as_of'].date()} (same snapshot as WACC weights). "
            f"Diluted shares: {cap['shares_diluted']:,.0f} ({cap['share_source']})."
        )

    with col2:
        st.markdown("**Discounted Cash Flows**")
        years = list(range(1, len(results["projected_ufcfs"]) + 1))
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Year {y}" for y in years],
            y=[u / 1e6 for u in results["projected_ufcfs"]],
            name="UFCF",
            marker_color="#4472C4",
        ))
        fig.add_trace(go.Bar(
            x=[f"Year {y}" for y in years],
            y=[pv / 1e6 for pv in dcf["pv_fcfs"]],
            name="PV of UFCF",
            marker_color="#1F4E79",
        ))
        fig.update_layout(
            yaxis_title="$ Millions",
            barmode="group",
            height=350,
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_sensitivity_heatmap(df, title, x_label, y_label, current_price):
    """Render a sensitivity table as a Plotly heatmap."""
    z = df.values.astype(float)
    text = [[f"${v:,.2f}" for v in row] for row in z]

    green = [0, 128, 0]
    white = [255, 255, 255]
    red = [200, 0, 0]

    colors = []
    for row in z:
        row_colors = []
        for val in row:
            if math.isnan(val) or not current_price:
                row_colors.append("rgb(128,128,128)")
                continue
            ratio = float((val - current_price) / current_price)
            if ratio >= 0:
                t = min(ratio / 0.5, 1.0)
                c = [round(white[i] + (green[i] - white[i]) * t) for i in range(3)]
            else:
                t = min(abs(ratio) / 0.5, 1.0)
                c = [round(white[i] + (red[i] - white[i]) * t) for i in range(3)]
            row_colors.append(f"rgb({c[0]},{c[1]},{c[2]})")
        colors.append(row_colors)

    fig = go.Figure(data=go.Table(
        header=dict(
            values=[y_label] + list(df.columns),
            fill_color="#1F4E79",
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=[list(df.index)] + [df[col].map(lambda x: "n/a" if pd.isna(x) else f"${x:,.2f}").tolist() for col in df.columns],
            fill_color=[["#2E75B6"] * len(df.index)] + [colors_col for colors_col in
                        [[colors[r][c] for r in range(len(df.index))] for c in range(len(df.columns))]],
            font=dict(
                color=[["white"] * len(df.index)] + [["black"] * len(df.index)] * len(df.columns),
                size=11,
                family="Calibri",
            ),
            align="center",
            height=30,
        ),
    ))
    fig.update_layout(title=title, height=50 + 35 * (len(df.index) + 1), margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)


def render_sensitivity(results):
    """Render both sensitivity tables."""
    st.subheader("Sensitivity Analysis")
    current = results["current_price"]

    col1, col2 = st.columns(2)
    with col1:
        render_sensitivity_heatmap(results["sens_df"], "WACC vs Terminal Growth Rate", "TGR", "WACC", current)
    with col2:
        render_sensitivity_heatmap(results["sens2_df"], "Revenue Growth vs Gross Margin", "GM", "Rev Growth Y1", current)

    st.caption("Green = at or above current price, Red = below current price")


def render_validation(results):
    """Render data validation section: issues by severity, provenance, XBRL cross-check."""
    issues = results["issues"]
    if results["reliable"]:
        st.success("No critical validation issues.")
    else:
        st.error("Critical validation issues: valuation should not be presented as reliable.")

    for severity, show in (("critical", st.error), ("warning", st.warning), ("info", st.info)):
        group = [i for i in issues if i["severity"] == severity]
        if group:
            st.markdown(f"**{severity.title()} ({len(group)})**")
            for i in group:
                show(i["message"])

    st.subheader("Input Provenance (latest fiscal year)")
    prov = results["provenance"].copy()
    prov["Value"] = prov["Value"].map(lambda x: "N/A" if x is None or pd.isna(x) else f"{x:,.0f}")
    st.dataframe(prov, use_container_width=True, hide_index=True)

    summary = results["xbrl_summary"]
    with st.expander(f"Data Source Validation (SEC XBRL vs Yahoo Finance): {summary['status']}"):
        st.caption(summary["reason"])
        report = results["xbrl_report"]
        if not report:
            return

        display_df = pd.DataFrame(report).drop(columns=["period_end"])
        fmt = lambda x: "N/A" if x is None or pd.isna(x) else f"{x:,.0f}"
        display_df['xbrl_val'] = display_df['xbrl_val'].map(fmt)
        display_df['yf_val'] = display_df['yf_val'].map(fmt)
        display_df['pct_diff'] = display_df['pct_diff'].map(lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.1%}")

        colors = {"MATCH": "#c6efce", "MISMATCH": "#ffc7ce"}

        def highlight_status(row):
            return [f"background-color: {colors.get(row['status'], '#eeeeee')}"] * len(row)

        styled = display_df.style.apply(highlight_status, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(
            f"{summary['n_match']} matches, {summary['n_mismatch']} mismatches (>5% difference), "
            f"{summary['n_missing']} missing, {summary['n_noncomparable']} not comparable"
        )


def render_earnings(filing):
    """Render earnings analysis from SEC filing."""
    if filing is None:
        st.info("SEC filing analysis unavailable for this ticker.")
        return

    st.subheader("SEC Filing Analysis")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Filing Details**")
        st.write(f"Type: {filing['filing_type']}")
        st.write(f"Date: {filing['filing_date']}")

        tone = filing["tone_score"]
        if tone is not None:
            label = "Net Positive" if tone > 0.55 else ("Net Negative" if tone < 0.45 else "Neutral")
            st.write(f"Tone: {tone:.3f} ({label})")
        else:
            st.write("Tone: N/A")

    with col2:
        pos = filing.get("positive_counts", {})
        neg = filing.get("negative_counts", {})
        if pos or neg:
            st.markdown("**Keyword Analysis**")
            kw_col1, kw_col2 = st.columns(2)
            with kw_col1:
                st.markdown("*Positive*")
                for word, count in sorted(pos.items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.write(f"{word}: {count}")
            with kw_col2:
                st.markdown("*Negative*")
                for word, count in sorted(neg.items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.write(f"{word}: {count}")

    guidance = filing.get("guidance_sentences", [])
    if guidance:
        with st.expander(f"Management Guidance ({len(guidance)} excerpts)"):
            for i, sentence in enumerate(guidance[:10], 1):
                st.markdown(f"{i}. {sentence}")


def main():
    st.title("DCF Model Generator")
    st.markdown("Enter a stock ticker to generate a full discounted cash flow analysis.")

    col1, col2 = st.columns([2, 5])
    with col1:
        symbol = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).strip().upper()
        run_button = st.button("Run Analysis", type="primary", use_container_width=True)

    if run_button or st.session_state.get("last_ticker"):
        if run_button:
            st.session_state["last_ticker"] = symbol
        ticker = st.session_state.get("last_ticker", symbol)

        with st.spinner(f"Running DCF analysis for {ticker}..."):
            try:
                base_results = run_model(ticker)
            except ValueError as e:
                st.error(str(e))
                return

        overrides = render_sidebar_overrides(base_results["derived"])

        if overrides:
            with st.spinner("Recalculating with overrides..."):
                results = run_model(ticker, overrides)
        else:
            results = base_results

        if results.get("warning"):
            st.warning(results["warning"])

        render_summary(results)

        filing = results.get("filing_analysis")
        excel_bytes = generate_excel_bytes(
            ticker=ticker,
            company_name=results["info"].get("shortName", ticker),
            current_price=results["current_price"],
            implied_price=results["implied_price"],
            wacc=results["wacc"],
            assumptions=results["derived"],
            full_df=results["full_df"],
            dcf_results=results["dcf_results"],
            filing_analysis=filing or empty_filing(ticker),
            sens_df=results["sens_df"],
            sens2_df=results["sens2_df"],
        )

        st.download_button(
            label="Download Excel Report",
            data=excel_bytes,
            file_name=f"{ticker}_{datetime.now().strftime('%Y-%m-%d')}_dcf_model.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.divider()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Financials", "DCF Valuation", "Sensitivity",
            "Data Validation", "Earnings Analysis",
        ])

        with tab1:
            render_assumptions(results["derived"])
            render_financials(results["full_df"])
        with tab2:
            render_wacc_decomposition(results["wacc_result"])
            render_dcf_bridge(results)
        with tab3:
            render_sensitivity(results)
        with tab4:
            render_validation(results)
        with tab5:
            render_earnings(filing)


if __name__ == "__main__":
    main()
