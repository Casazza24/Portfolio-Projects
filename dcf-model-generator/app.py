"""Streamlit dashboard for the DCF model generator."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from config import assumptions
from data_fetcher import fetch_financial_data
from projections import derive_assumptions, project_financials
from dcf_engine import calculate_wacc, run_dcf
from sensitivity import build_sensitivity_tables
from edgar import analyze_filing, empty_filing
from excel_output import generate_excel_bytes

st.set_page_config(page_title="DCF Model Generator", page_icon="📊", layout="wide")


@st.cache_data(ttl=300)
def run_model(symbol):
    """Run the full DCF pipeline and return all results."""
    data = fetch_financial_data(symbol)
    info = data["info"]
    hist_df = data["hist_df"]
    income_stmt = data["income_stmt"]

    derived = derive_assumptions(hist_df, assumptions)
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

    sens_df, sens2_df = build_sensitivity_tables(
        projected_ufcfs, total_debt, total_cash, shares, wacc,
        derived["terminal_growth_rate"], base_revenue, derived,
    )

    try:
        filing_analysis = analyze_filing(symbol, "10-Q")
    except Exception:
        filing_analysis = None

    return {
        "info": info,
        "hist_df": hist_df,
        "derived": derived,
        "full_df": full_df,
        "wacc": wacc,
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
    }


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
        st.write(f"D&A % Revenue: {derived['da_pct_revenue']:.1%}")

    with col3:
        st.markdown("**Capital**")
        st.write(f"CapEx % Revenue: {derived['capex_pct_revenue']:.1%}")
        st.write(f"Tax Rate: {derived['tax_rate']:.1%}")
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
            "+ Cash": results["total_cash"] / 1e6,
            "= Equity Value": dcf["equity_value"] / 1e6,
        }
        bridge_df = pd.DataFrame({"$ Millions": bridge_data.values()}, index=bridge_data.keys())
        bridge_df["$ Millions"] = bridge_df["$ Millions"].map(lambda x: f"{x:,.0f}")
        st.dataframe(bridge_df, use_container_width=True)

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
            ratio = float((val - current_price) / current_price) if current_price else 0.0
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
            values=[list(df.index)] + [df[col].map(lambda x: f"${x:,.2f}").tolist() for col in df.columns],
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
                results = run_model(ticker)
            except ValueError as e:
                st.error(str(e))
                return

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

        tab1, tab2, tab3, tab4 = st.tabs(["Financials", "DCF Valuation", "Sensitivity", "Earnings Analysis"])

        with tab1:
            render_assumptions(results["derived"])
            render_financials(results["full_df"])
        with tab2:
            render_dcf_bridge(results)
        with tab3:
            render_sensitivity(results)
        with tab4:
            render_earnings(filing)


if __name__ == "__main__":
    main()
