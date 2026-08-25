import streamlit as st

st.set_page_config(
    page_title="Factor Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

import pandas as pd

from config.portfolios import MODEL_PORTFOLIOS
from config.benchmarks import BENCHMARKS
from config.settings import (
    DEFAULT_WINDOW,
    DEFAULT_BENCHMARK,
    DEFAULT_ROLLING_PERIOD,
    WINDOW_MAP,
)
from src.data_loader import (
    fetch_prices,
    compute_returns,
    fetch_fama_french_factors,
    trim_to_window,
    validate_ticker,
    fetch_sentiment_factor,
    merge_sentiment_with_ff,
)
from src.portfolio import compute_portfolio_returns, align_series
from src.metrics import (
    compute_all_kpis,
    drawdown_series,
    rolling_beta,
    var_95,
    cvar_95,
)
from src.factors import run_capm_regression, run_ff_regression
from src.charts import (
    rolling_beta_chart,
    cumulative_return_chart,
    return_distribution_chart,
    factor_exposure_chart,
    drawdown_chart,
    correlation_heatmap,
    multi_cumulative_return_chart,
    multi_drawdown_chart,
)
from components.kpi_cards import render_kpi_row
from components.search_bar import render_search_bar
from components.comparison import render_comparison_table, render_comparison_selector
from components.objective_section import render_objective_section
from components.sentiment_panel import render_sentiment_panel


# ── Sidebar ──────────────────────────────────────────────────────────────

st.sidebar.title("Controls")

st.sidebar.subheader("Benchmark")
benchmark_label = st.sidebar.selectbox(
    "Benchmark index",
    options=list(BENCHMARKS.keys()) + ["Custom"],
    index=list(BENCHMARKS.keys()).index(
        next(k for k, v in BENCHMARKS.items() if v == DEFAULT_BENCHMARK)
    ),
)
if benchmark_label == "Custom":
    raw_benchmark = st.sidebar.text_input("Custom benchmark ticker", value="SPY")
    benchmark_ticker = validate_ticker(raw_benchmark)
    if not benchmark_ticker:
        st.sidebar.error("Enter a valid ticker symbol (letters only, max 10 chars).")
        st.stop()
else:
    benchmark_ticker = BENCHMARKS[benchmark_label]

st.sidebar.divider()

st.sidebar.subheader("Time Window")
window_label = st.sidebar.selectbox(
    "Analysis period",
    options=list(WINDOW_MAP.keys()),
    index=list(WINDOW_MAP.keys()).index(DEFAULT_WINDOW),
)
trading_days = WINDOW_MAP[window_label]

st.sidebar.subheader("Rolling Window")
rolling_period = st.sidebar.selectbox(
    "Rolling period (days)",
    options=[30, 60, 90, 120],
    index=[30, 60, 90, 120].index(DEFAULT_ROLLING_PERIOD),
)


# ── Main Content ─────────────────────────────────────────────────────────

st.title("Factor Risk Management Dashboard")

tab_single, tab_compare, tab_sentiment = st.tabs(["Single Portfolio", "Compare Portfolios", "Sentiment Factor"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Single Portfolio Analysis
# ═══════════════════════════════════════════════════════════════════════════

with tab_single:
    st.subheader("Select a portfolio")

    col_model, col_search = st.columns([1, 2])

    with col_model:
        portfolio_choice = st.radio(
            "Model portfolios",
            options=["None (use search)"] + list(MODEL_PORTFOLIOS.keys()),
            index=0,
            key="portfolio_radio",
        )

    active_portfolio = None
    portfolio_label = None

    if portfolio_choice != "None (use search)":
        active_portfolio = MODEL_PORTFOLIOS[portfolio_choice]
        portfolio_label = portfolio_choice
    else:
        with col_search:
            search_result = render_search_bar()
            if search_result:
                active_portfolio = search_result
                portfolio_label = list(search_result.keys())[0]

    if active_portfolio is None:
        st.info("Select a model portfolio or search for an ETF to get started.")
        render_objective_section()
    else:
        # ── Data Loading ─────────────────────────────────────────
        tickers = list(active_portfolio.keys())
        all_tickers = sorted(set(tickers + [benchmark_ticker]))

        with st.spinner(f"Fetching price data for {', '.join(all_tickers)}..."):
            try:
                prices = fetch_prices(all_tickers, period="5y")
            except ValueError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(
                    f"Could not fetch price data. Try again in a moment.\n\nDetails: {e}"
                )
                st.stop()

        returns = compute_returns(prices)

        missing = [t for t in tickers if t not in returns.columns]
        if missing:
            st.error(f"No data for: **{', '.join(missing)}**. May be delisted or misspelled.")
            st.stop()
        if benchmark_ticker not in returns.columns:
            st.error(f"No data for benchmark **{benchmark_ticker}**.")
            st.stop()

        returns = trim_to_window(returns, trading_days)

        if len(returns) < 30:
            st.warning(f"Only {len(returns)} trading days available — metrics may be unreliable.")

        if len(tickers) == 1 and tickers[0] in returns.columns:
            portfolio_returns = returns[tickers[0]].rename("portfolio")
        else:
            portfolio_returns = compute_portfolio_returns(returns, active_portfolio)

        benchmark_returns_s = returns[benchmark_ticker].rename("benchmark")
        portfolio_returns, benchmark_returns_s = align_series(portfolio_returns, benchmark_returns_s)

        # ── Factor Regressions ───────────────────────────────────
        with st.spinner("Running factor regressions..."):
            factor_error = None
            try:
                factor_data = fetch_fama_french_factors()
                capm_result = run_capm_regression(portfolio_returns, factor_data)
                ff_result = run_ff_regression(portfolio_returns, factor_data)
            except Exception as e:
                factor_error = str(e)
                capm_result = {"alpha": 0.0, "beta": 0.0}
                ff_result = {"factor_betas": {}}

        # ── KPI Row ──────────────────────────────────────────────
        st.divider()
        st.subheader(f"{portfolio_label}  vs  {benchmark_ticker}  —  {window_label}")

        kpis = compute_all_kpis(
            portfolio_returns, benchmark_returns_s,
            alpha=capm_result["alpha"],
            beta=capm_result["beta"],
        )
        render_kpi_row(kpis)

        if factor_error:
            st.warning(f"Factor regression unavailable: {factor_error}")
        else:
            st.caption(
                f"CAPM Alpha t-stat: {capm_result.get('alpha_tstat', 0):.2f} "
                f"(p={capm_result.get('alpha_pvalue', 1):.3f})  |  "
                f"Beta t-stat: {capm_result.get('beta_tstat', 0):.2f} "
                f"(p={capm_result.get('beta_pvalue', 1):.3f})  |  "
                f"R²: {capm_result.get('r_squared', 0):.3f}"
            )

        # ── Charts ───────────────────────────────────────────────
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            rb = rolling_beta(portfolio_returns, benchmark_returns_s, window=rolling_period)
            if len(rb) > 0:
                st.plotly_chart(rolling_beta_chart(rb, rolling_period), use_container_width=True)
            else:
                st.info(f"Not enough data for {rolling_period}-day rolling beta.")

        with col2:
            v = var_95(portfolio_returns)
            cv = cvar_95(portfolio_returns)
            st.plotly_chart(
                return_distribution_chart(portfolio_returns, v, cv),
                use_container_width=True,
            )

        if ff_result.get("factor_betas"):
            st.plotly_chart(
                factor_exposure_chart(ff_result["factor_betas"]),
                use_container_width=True,
            )

        st.plotly_chart(
            cumulative_return_chart(
                portfolio_returns, benchmark_returns_s,
                portfolio_name=portfolio_label,
                benchmark_name=benchmark_ticker,
            ),
            use_container_width=True,
        )

        dd = drawdown_series(portfolio_returns)
        st.plotly_chart(drawdown_chart(dd), use_container_width=True)

        st.divider()
        render_objective_section()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Portfolio Comparison
# ═══════════════════════════════════════════════════════════════════════════

with tab_compare:
    st.subheader("Compare Portfolios Side by Side")

    selected_names = render_comparison_selector()

    if len(selected_names) < 2:
        st.info("Select at least 2 portfolios to compare.")
    else:
        # Gather all tickers needed
        all_comparison_tickers = set([benchmark_ticker])
        for name in selected_names:
            all_comparison_tickers.update(MODEL_PORTFOLIOS[name].keys())
        all_comparison_tickers = sorted(all_comparison_tickers)

        with st.spinner(f"Fetching data for {len(all_comparison_tickers)} tickers..."):
            try:
                comp_prices = fetch_prices(all_comparison_tickers, period="5y")
            except Exception as e:
                st.error(f"Failed to fetch comparison data: {e}")
                st.stop()

        comp_returns = compute_returns(comp_prices)
        comp_returns = trim_to_window(comp_returns, trading_days)

        comp_benchmark = comp_returns[benchmark_ticker].rename("benchmark")

        # Compute per-portfolio returns and KPIs
        portfolio_return_series = {}
        all_kpis = {}
        all_drawdowns = {}

        with st.spinner("Running regressions for all portfolios..."):
            try:
                factor_data = fetch_fama_french_factors()
            except Exception:
                factor_data = None

            for name in selected_names:
                weights = MODEL_PORTFOLIOS[name]
                tickers = list(weights.keys())

                port_missing = [t for t in tickers if t not in comp_returns.columns]
                if port_missing:
                    st.warning(f"Skipping **{name}** — missing data for {', '.join(port_missing)}")
                    continue

                if len(tickers) == 1:
                    port_ret = comp_returns[tickers[0]].rename("portfolio")
                else:
                    port_ret = compute_portfolio_returns(comp_returns, weights)

                port_ret, bench_aligned = align_series(port_ret, comp_benchmark)
                portfolio_return_series[name] = port_ret

                # Factor regressions
                alpha, beta = 0.0, 0.0
                if factor_data is not None:
                    try:
                        capm = run_capm_regression(port_ret, factor_data)
                        alpha, beta = capm["alpha"], capm["beta"]
                    except Exception:
                        pass

                all_kpis[name] = compute_all_kpis(port_ret, bench_aligned, alpha=alpha, beta=beta)
                all_drawdowns[name] = drawdown_series(port_ret)

        if len(all_kpis) < 2:
            st.error("Not enough valid portfolios to compare.")
        else:
            # ── Comparison Table ─────────────────────────────────
            st.divider()
            st.subheader(f"KPI Comparison  —  {window_label}  vs  {benchmark_ticker}")
            render_comparison_table(all_kpis)

            # ── Overlay Charts ───────────────────────────────────
            st.divider()

            st.plotly_chart(
                multi_cumulative_return_chart(
                    portfolio_return_series, comp_benchmark,
                    benchmark_name=benchmark_ticker,
                ),
                use_container_width=True,
            )

            st.plotly_chart(
                multi_drawdown_chart(all_drawdowns),
                use_container_width=True,
            )

            # ── Correlation Heatmap ──────────────────────────────
            corr_df = pd.DataFrame(portfolio_return_series)
            corr_df[benchmark_ticker] = comp_benchmark
            st.plotly_chart(
                correlation_heatmap(corr_df),
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Sentiment Factor Analysis
# ═══════════════════════════════════════════════════════════════════════════

with tab_sentiment:
    st.subheader("Sentiment Risk Factor Analysis")

    st.markdown(
        "This tab integrates a custom NLP-derived sentiment factor (SENT) into the "
        "Fama-French regression framework. SENT is constructed as a daily long-short "
        "portfolio: long top-quintile sentiment, short bottom-quintile sentiment, "
        "scored using FinBERT on financial news and social media text."
    )

    # Reuse the portfolio from tab 1 if selected
    if "portfolio_radio" in st.session_state:
        choice = st.session_state.portfolio_radio
        if choice != "None (use search)":
            sent_portfolio = MODEL_PORTFOLIOS[choice]
            sent_label = choice
        elif "search_input" in st.session_state and st.session_state.search_input:
            ticker = st.session_state.search_input.strip().upper()
            sent_portfolio = {ticker: 1.0}
            sent_label = ticker
        else:
            st.info("Select a portfolio in the Single Portfolio tab first.")
            st.stop()
    else:
        st.info("Select a portfolio in the Single Portfolio tab first.")
        st.stop()

    sent_tickers = list(sent_portfolio.keys())
    sent_all_tickers = sorted(set(sent_tickers + [benchmark_ticker]))

    with st.spinner("Loading data for sentiment analysis..."):
        try:
            sent_prices = fetch_prices(sent_all_tickers, period="5y")
        except Exception as e:
            st.error(f"Failed to fetch price data: {e}")
            st.stop()

    sent_returns = compute_returns(sent_prices)
    sent_returns = trim_to_window(sent_returns, trading_days)

    if len(sent_tickers) == 1 and sent_tickers[0] in sent_returns.columns:
        sent_port_ret = sent_returns[sent_tickers[0]].rename("portfolio")
    else:
        sent_port_ret = compute_portfolio_returns(sent_returns, sent_portfolio)

    sent_bench_ret = sent_returns[benchmark_ticker].rename("benchmark")
    sent_port_ret, sent_bench_ret = align_series(sent_port_ret, sent_bench_ret)

    # Load factors + sentiment
    with st.spinner("Loading factor data with sentiment..."):
        try:
            ff_data = fetch_fama_french_factors()
            sent_series = fetch_sentiment_factor()
            augmented_factors = merge_sentiment_with_ff(ff_data, sent_series)
            sent_ff_result = run_ff_regression(sent_port_ret, augmented_factors)
        except Exception as e:
            st.warning(f"Factor regression error: {e}")
            sent_ff_result = {"factor_betas": {}}
            augmented_factors = ff_data if "ff_data" in dir() else pd.DataFrame()
            sent_series = None

    render_sentiment_panel(
        sent_port_ret, augmented_factors, sent_ff_result, sent_series
    )
