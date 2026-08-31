# Factor Risk Management Dashboard

## Project spec -- Streamlit application

---

## 1. Objective

### What this project is

An interactive portfolio risk analytics dashboard built in Streamlit that lets a user search for any publicly available portfolio (ETFs, index funds, or curated model portfolios), select a benchmark, and view a comprehensive breakdown of risk-adjusted performance metrics, factor exposures, and tail-risk indicators, all rendered in real time with a responsive caching layer underneath.

### Why it matters

Portfolio managers, risk analysts, and traders don't evaluate portfolios on raw returns alone. A portfolio that returned 15% is meaningless without knowing how much risk was taken to get there, what systematic factors drove the return, and how the portfolio would behave in a drawdown. This dashboard surfaces those answers in a single view.

The project demonstrates three distinct competencies:

1. **Quantitative finance** -- factor modeling (CAPM, Fama-French), tail-risk measurement (VaR/CVaR), and risk-adjusted return attribution are the analytical core of any markets or portfolio management role.
2. **Data engineering** -- the caching architecture, API integration, and pipeline from raw price data to computed metrics mirrors the kind of data plumbing that supports trading desks and risk systems in production.
3. **Product thinking** -- presenting complex analytics in a clean, scannable interface that a non-technical stakeholder (a PM, a client, or an interviewer) can actually use without explanation.

### Who it's for

- Interviewers at quant/trading desks who want to see that you understand factor risk beyond textbook definitions.
- Anyone evaluating portfolio construction decisions and wanting to compare risk profiles across different strategies.
- Personal use as a research tool when screening ETFs or model portfolios.

---

## 2. Feature overview

### 2.1 Portfolio search

- A text search bar in the main content area (not the sidebar) that lets a user search for publicly traded ETFs or index funds by ticker or name.
- Search queries hit a local lookup table of known tickers first (for instant results), then fall back to a live API call if the ticker is not in the local cache.
- Selecting a result loads the portfolio's constituent holdings (for ETFs that publish daily holdings) or, if holdings are unavailable, treats the ETF/fund itself as a single-asset portfolio and computes metrics on its price series.
- A "recent searches" section persists the last 5 searched portfolios in `st.session_state` for quick re-selection.

### 2.2 Curated model portfolios

- A dropdown or button group of pre-built model portfolios for users who want to explore without searching. These are defined in a config file and serve as sensible defaults on first load.
- Suggested set:
  - **Growth tilt** -- QQQ-heavy allocation with tech/growth ETF mix
  - **Dividend / value** -- VYM, SCHD, VTV weighted blend
  - **Balanced 60/40** -- 60% VTI / 40% BND classic allocation
  - **Sector concentrated** -- XLK or XLE single-sector bet
  - **Small cap momentum** -- IWM / MTUM factor tilt
- Each model portfolio is defined as a dictionary of `{ticker: weight}` pairs so the analytics pipeline treats them identically to a searched portfolio.

### 2.3 Benchmark selection

- A dropdown selector (sidebar) offering common benchmarks:
  - S&P 500 (SPY)
  - Russell 2000 (IWM)
  - MSCI World (URTH)
  - Bloomberg US Aggregate Bond (AGG)
  - Nasdaq 100 (QQQ)
  - Custom ticker input as a fallback
- Benchmark return series is fetched and cached with the same pipeline as portfolio data.
- All relative metrics (alpha, beta, tracking error, information ratio) are computed against the selected benchmark.

### 2.4 Time window control

- A selector for analysis period: 6M, 1Y, 2Y, 3Y, 5Y.
- All charts and metrics recompute when the window changes.
- Rolling window size for rolling beta/correlation adjusts proportionally (e.g., 60-day rolling for 1Y, 120-day for 3Y+).

### 2.5 KPI summary row

- A horizontal row of metric cards at the top of the dashboard displaying the headline numbers at a glance.
- Each card shows the metric name, computed value, and a color indicator (green for favorable, red for unfavorable).
- Metrics displayed (see section 4 for definitions and formulas):
  - Annualized alpha
  - Beta
  - Sharpe ratio
  - Sortino ratio
  - Max drawdown
  - VaR (95%)
  - CVaR (95%)
  - Information ratio

### 2.6 Charts section

All charts are rendered with Plotly (via `plotly.graph_objects` or `plotly.express`) for interactivity (hover, zoom, pan). Layout:

1. **Rolling beta** -- line chart showing how beta evolves over the selected window. Configurable rolling period (30, 60, 90 days).
2. **Cumulative return vs benchmark** -- dual-line chart, portfolio vs benchmark, rebased to 100.
3. **Return distribution** -- histogram of daily returns with overlaid normal distribution curve and VaR/CVaR threshold lines marked.
4. **Fama-French factor exposures** -- horizontal bar chart of factor betas (Mkt-RF, SMB, HML, MOM, RMW, CMA).
5. **Drawdown chart** -- time series of drawdown from peak, useful for visualizing recovery periods.
6. **Correlation heatmap** (if comparing multiple portfolios) -- pairwise correlation matrix across selected portfolios.

### 2.7 Objective and KPI guide section

- An expandable section (`st.expander`) titled "About this dashboard" that contains:
  - The project objective (condensed version of section 1 above).
  - A KPI reference guide explaining each metric, why it matters, and how to interpret the number in context. Written for someone who knows basic finance but may not have a quant background.
  - A note on data sources and methodology (factor data provenance, regression approach, return calculation assumptions).

### 2.8 Portfolio comparison mode (stretch)

- Ability to select 2-3 portfolios simultaneously and compare them side by side.
- Shared benchmark, shared time window.
- Metrics rendered in a comparison table rather than individual cards.
- Overlay on cumulative return chart with one line per portfolio.

---

## 3. Architecture

### 3.1 Project structure

```
factor-risk-dashboard/
|-- app.py                     # Streamlit entry point
|-- requirements.txt
|-- README.md
|-- config/
|   |-- portfolios.py          # Model portfolio definitions ({ticker: weight})
|   |-- benchmarks.py          # Benchmark ticker map
|   |-- settings.py            # Default window, rolling period, cache TTL
|-- data/
|   |-- cache/                 # Local cache directory (SQLite or parquet files)
|   |-- ff_factors.csv         # Fama-French factor data (refreshed monthly)
|-- src/
|   |-- data_loader.py         # API calls, data fetching, cache management
|   |-- portfolio.py           # Portfolio construction (weights, returns)
|   |-- metrics.py             # All risk/return metric calculations
|   |-- factors.py             # Factor regression logic (CAPM, FF3, FF5+MOM)
|   |-- charts.py              # Plotly chart builders
|-- components/
|   |-- kpi_cards.py           # Streamlit metric card rendering
|   |-- search_bar.py          # Portfolio search component
|   |-- objective_section.py   # Expandable objective + KPI guide content
|-- tests/
|   |-- test_metrics.py        # Unit tests for metric calculations
|   |-- test_factors.py        # Unit tests for factor regressions
```

### 3.2 Data flow

```
User selects portfolio + benchmark + window
        |
        v
data_loader.py
   |-- Check local cache (SQLite/parquet) for each ticker
   |   |-- Cache hit + fresh (< 1 trading day old) --> return cached data
   |   |-- Cache miss or stale --> fetch from API, write to cache, return
   |-- Batch download via yfinance (multiple tickers in one call)
   |-- Download Fama-French factors (cached monthly)
        |
        v
portfolio.py
   |-- Compute weighted portfolio return series from constituent returns
   |-- Align dates across portfolio, benchmark, and factor series
        |
        v
metrics.py + factors.py
   |-- Compute all KPIs from aligned return series
   |-- Run OLS regressions for alpha/beta (CAPM) and factor betas (FF)
   |-- Compute rolling statistics (rolling beta, rolling Sharpe)
        |
        v
charts.py + kpi_cards.py
   |-- Build Plotly figures from computed data
   |-- Render metric cards with st.metric or custom HTML
        |
        v
Streamlit renders dashboard
```

### 3.3 Caching layer

**Goal**: first load of a new portfolio takes 3-8 seconds (API fetch); subsequent loads and parameter changes (window, benchmark swap) feel near-instant.

**Implementation**:

1. **Price data cache** -- use `@st.cache_data(ttl=3600)` on the data fetching function. This means a given ticker's price history is fetched at most once per hour per session. Underlying storage can be in-memory (Streamlit's default) or written to a local parquet file for persistence across app restarts.

2. **Fama-French factor cache** -- factor data from Ken French's data library updates monthly. Download once on app startup, cache to `data/ff_factors.csv`, and reload from disk on subsequent runs. Only re-download if the file is older than 30 days.

3. **Computed metrics cache** -- use `@st.cache_data` on the metrics computation functions with the portfolio weights, ticker list, window, and benchmark as hash keys. If the user switches from 1Y to 3Y and back, the 1Y results are still in cache.

4. **Session state for UI** -- `st.session_state` stores recent searches, current selections, and comparison portfolio list so the app doesn't reset on every interaction.

**Cache invalidation**: TTL-based. Price data expires after 1 hour (covers intraday re-visits). Factor data expires after 30 days. Metrics expire whenever their input data expires (Streamlit handles this automatically since cached functions re-run when upstream inputs change).

### 3.4 Data sources

| Data | Source | Method | Rate limits |
|---|---|---|---|
| Historical prices (daily OHLCV) | Yahoo Finance via `yfinance` | `yf.download(tickers, period, interval)` | Unofficial API, no hard limit, but batch calls recommended |
| ETF holdings/weights | ETF provider websites or `etfdb-api` / manual CSV | Scrape or static config | Varies |
| Fama-French factor returns | Kenneth French Data Library via `pandas_datareader` or direct CSV download | `pdr.get_data_famafrench()` | None (static academic data) |
| Benchmark prices | Same yfinance pipeline | Same as above | Same |

**Fallback strategy**: if yfinance fails or rate-limits, the app should display a clear error message ("Could not fetch data for [ticker]. Try again in a moment.") rather than crashing. Wrap all API calls in try/except blocks.

---

## 4. Metrics reference

### 4.1 KPI definitions and formulas

Each metric below is computed from daily return series unless otherwise noted. All annualization assumes 252 trading days per year.

---

**Alpha (annualized)**

- **What it is**: the portfolio's excess return above what CAPM predicts given its market exposure. Measures the value added (or destroyed) beyond passive benchmark tracking.
- **Formula**: from the CAPM regression `R_p - R_f = alpha + beta * (R_m - R_f) + epsilon`, the intercept term `alpha` is annualized by multiplying daily alpha by 252.
- **Interpretation**: positive alpha means the portfolio outperformed its risk-adjusted benchmark. A growth portfolio with alpha = 2.4% generated 2.4% of annual return that can't be explained by market exposure alone.
- **Why it matters**: this is the single most important number for evaluating active management skill. A portfolio with high returns but negative alpha was just taking more market risk, not generating real value.

---

**Beta**

- **What it is**: the portfolio's sensitivity to benchmark movements. A beta of 1.2 means the portfolio moves 1.2% for every 1% move in the benchmark, on average.
- **Formula**: slope coefficient from the CAPM regression above, or equivalently `Cov(R_p, R_m) / Var(R_m)`.
- **Interpretation**: beta > 1 means the portfolio amplifies market moves (more aggressive); beta < 1 means it dampens them (more defensive). Beta near 0 means low correlation to the benchmark.
- **Why it matters**: a trader or PM needs to know how much market risk a portfolio carries. A beta-1.3 portfolio will lose ~13% if the market drops 10%, all else equal. It's also the foundation for understanding whether returns are coming from skill (alpha) or from levered market exposure.

---

**Sharpe ratio**

- **What it is**: risk-adjusted return, measuring how much excess return (over the risk-free rate) the portfolio earns per unit of total volatility.
- **Formula**: `(annualized_return - risk_free_rate) / annualized_volatility`, where annualized volatility = `std(daily_returns) * sqrt(252)` and annualized return = `mean(daily_returns) * 252`.
- **Interpretation**: higher is better. Sharpe > 1.0 is generally considered good; > 2.0 is exceptional. Below 0.5 suggests the risk isn't being well compensated.
- **Why it matters**: raw returns are meaningless without context. A 20% return with 40% volatility (Sharpe ~0.4) is worse risk-adjusted than a 10% return with 8% volatility (Sharpe ~1.0). This is the universal language for comparing strategies across asset classes.

---

**Sortino ratio**

- **What it is**: a modification of the Sharpe ratio that only penalizes downside volatility. Investors don't mind upside variance; they mind drawdowns.
- **Formula**: `(annualized_return - risk_free_rate) / downside_deviation`, where downside deviation = `sqrt(mean(min(R_p - target, 0)^2)) * sqrt(252)`, with target typically set to 0 or the risk-free rate.
- **Interpretation**: higher is better. A portfolio with high upside variance but controlled drawdowns will have a Sortino much higher than its Sharpe, which is a positive signal.
- **Why it matters**: Sharpe penalizes a portfolio for "too much" upside volatility, which is a flaw. Sortino isolates the risk that actually hurts, making it a more accurate measure of risk-adjusted performance for asymmetric return profiles.

---

**Max drawdown**

- **What it is**: the largest peak-to-trough decline in portfolio value over the selected window. Measures the worst-case loss an investor would have experienced.
- **Formula**: `max over all t of (peak_value_up_to_t - value_at_t) / peak_value_up_to_t`, expressed as a negative percentage.
- **Interpretation**: a max drawdown of -22% means the portfolio lost 22% from its highest point before recovering. Deeper drawdowns take longer to recover from (a -50% loss requires a +100% gain to break even).
- **Why it matters**: volatility is symmetric and abstract; max drawdown is concrete and asymmetric. It answers the question every investor actually cares about: "How bad could it get?" It's also a key input for position sizing and leverage decisions on a trading desk.

---

**Value at Risk (VaR) at 95%**

- **What it is**: the maximum expected daily loss at the 95th percentile confidence level, assuming normal market conditions. "On 95% of days, the portfolio will not lose more than X%."
- **Formula (historical)**: the 5th percentile of the empirical daily return distribution.
- **Formula (parametric)**: `mean(daily_returns) - 1.645 * std(daily_returns)`.
- **Interpretation**: a VaR of -2.1% means that on 95% of trading days, the portfolio's loss is expected to be 2.1% or less. On the remaining 5% of days, losses could exceed this.
- **Why it matters**: VaR is the industry-standard risk metric reported to regulators and risk committees. Every bank and fund calculates it daily. It translates portfolio risk into a single dollar (or percentage) number that senior management and compliance can act on.

---

**Conditional VaR (CVaR / Expected Shortfall) at 95%**

- **What it is**: the average loss in the worst 5% of days. It answers the question VaR leaves open: "When things go past the VaR threshold, how bad does it actually get?"
- **Formula**: `mean(daily_returns where daily_return <= VaR_95)`.
- **Interpretation**: CVaR is always worse (more negative) than VaR. A VaR of -2.1% with a CVaR of -3.4% means that while 95% of days lose less than 2.1%, the average loss on the bad tail days is 3.4%. The gap between VaR and CVaR reveals tail risk.
- **Why it matters**: VaR is blind to what happens beyond the threshold. Two portfolios can have the same VaR but very different CVaRs if one has fatter tails. CVaR is increasingly preferred by risk managers (and is required under Basel III/IV) because it captures the severity of tail events, not just their frequency.

---

**Information ratio**

- **What it is**: the portfolio's active return (alpha) per unit of active risk (tracking error). Measures how consistently the portfolio outperforms its benchmark.
- **Formula**: `(annualized_portfolio_return - annualized_benchmark_return) / tracking_error`, where tracking error = `std(R_p - R_b) * sqrt(252)`.
- **Interpretation**: IR > 0.5 is generally considered good; > 1.0 is exceptional. A high IR means the portfolio beats the benchmark consistently, not just in a few lucky periods.
- **Why it matters**: alpha alone doesn't tell you if outperformance was consistent or a one-time fluke. A portfolio with 3% alpha and 10% tracking error (IR = 0.3) is far less reliable than one with 2% alpha and 3% tracking error (IR = 0.67). For asset allocators deciding whether a manager has genuine skill, IR is the go-to metric.

---

### 4.2 Factor model details

**CAPM (single-factor)**

Regression: `R_p - R_f = alpha + beta * (R_m - R_f) + epsilon`

Produces alpha and beta. Run using `statsmodels.api.OLS` with Newey-West standard errors (to correct for autocorrelation in daily returns). Report t-statistics and p-values alongside coefficients so the user can assess statistical significance.

**Fama-French five-factor + momentum**

Regression: `R_p - R_f = alpha + b1*(Mkt-RF) + b2*(SMB) + b3*(HML) + b4*(RMW) + b5*(CMA) + b6*(MOM) + epsilon`

Factor definitions:
- **Mkt-RF**: market excess return (market return minus risk-free rate)
- **SMB** (small minus big): small-cap stocks minus large-cap stocks. Positive loading means the portfolio tilts toward small caps.
- **HML** (high minus low): value stocks (high book-to-market) minus growth stocks. Positive loading means value tilt.
- **RMW** (robust minus weak): profitable firms minus unprofitable firms. Positive loading means quality/profitability tilt.
- **CMA** (conservative minus aggressive): low-investment firms minus high-investment firms. Positive loading means the portfolio favors capital-light businesses.
- **MOM** (momentum): past winners minus past losers. Positive loading means the portfolio tilts toward recent outperformers.

Factor data source: Kenneth French Data Library, downloaded via `pandas_datareader` or direct URL (`https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html`).

**Rolling factor analysis**: run the factor regression on a rolling window (default 120 trading days) to show how factor exposures shift over time. Display as a stacked area chart or multi-line chart.

---

## 5. Tech stack

| Layer | Tool | Rationale |
|---|---|---|
| Frontend / UI | Streamlit | Single-file Python app, built-in widgets (search, dropdown, slider, metric cards), fast prototyping, free deployment via Streamlit Community Cloud |
| Charts | Plotly (`plotly.graph_objects`) | Interactive (hover, zoom, pan), integrates natively with Streamlit via `st.plotly_chart`, better than matplotlib for dashboard use cases |
| Data fetching | `yfinance` | Free, no API key required, supports batch ticker downloads, widely used in quant projects |
| Factor data | `pandas_datareader` or direct CSV | Standard method for accessing Fama-French factor returns from Ken French's library |
| Statistics | `statsmodels` (OLS), `numpy`, `pandas` | `statsmodels.OLS` gives regression coefficients, t-stats, p-values, and R-squared in one call. NumPy/pandas for vectorized return and risk calculations |
| Caching | `st.cache_data` + local parquet files | Streamlit's built-in caching decorator handles in-session memoization; parquet files persist across restarts |
| Deployment | Streamlit Community Cloud | Free hosting, GitHub integration, shareable URL for portfolio site / interviews |

### Python dependencies (requirements.txt)

```
streamlit>=1.30.0
yfinance>=0.2.30
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
statsmodels>=0.14.0
pandas-datareader>=0.10.0
```

---

## 6. UI layout

```
+-----------------------------------------------------------------------+
|  Factor Risk Management Dashboard                          [About] btn|
+-----------------------------------------------------------------------+
|                                                                       |
|  SIDEBAR                          MAIN CONTENT                        |
|  +--------------+                 +----------------------------------+|
|  | Model port-  |                 | Search portfolio  [___________]  ||
|  | folios       |                 | Recent: SPY  QQQ  VTI            ||
|  | [Growth    ] |                 +----------------------------------+|
|  | [Dividend  ] |                 |                                   |
|  | [60/40     ] |                 | +------+ +------+ +------+ +---+ |
|  | [Sector    ] |                 | |Alpha | |Beta  | |Sharpe| |...| |
|  | [SmallCap  ] |                 | |+2.4% | |1.18  | |0.92  | |   | |
|  |              |                 | +------+ +------+ +------+ +---+ |
|  | Benchmark    |                 |                                   |
|  | [S&P 500  v] |                 | +---------------++--------------+|
|  |              |                 | | Rolling beta   || Return dist  ||
|  | Window       |                 | | [line chart]   || [histogram]  ||
|  | [1Y       v] |                 | +---------------++--------------+|
|  |              |                 |                                   |
|  | Rolling win  |                 | +-------------------------------+|
|  | [60 days  v] |                 | | Factor exposures (FF 5+MOM)   ||
|  |              |                 | | [horizontal bar chart]        ||
|  +--------------+                 | +-------------------------------+|
|                                   |                                   |
|                                   | +-------------------------------+|
|                                   | | Cumulative return vs benchmark||
|                                   | | [dual line chart]             ||
|                                   | +-------------------------------+|
|                                   |                                   |
|                                   | +-------------------------------+|
|                                   | | Drawdown from peak            ||
|                                   | | [area chart]                  ||
|                                   | +-------------------------------+|
|                                   |                                   |
|                                   | [v] About this dashboard         |
|                                   |   Objective, KPI guide,          |
|                                   |   methodology notes              |
|                                   +----------------------------------+|
+-----------------------------------------------------------------------+
```

---

## 7. Implementation phases

### Phase 1 -- Core pipeline (day 1-2)

- Set up project structure and `requirements.txt`.
- Implement `data_loader.py` with yfinance batch download and `@st.cache_data` caching.
- Implement `metrics.py` with all KPI calculations (alpha, beta, Sharpe, Sortino, max drawdown, VaR, CVaR, tracking error, information ratio).
- Implement `factors.py` with CAPM and Fama-French regressions using `statsmodels.OLS`.
- Write unit tests for metric calculations against known values.

### Phase 2 -- Streamlit UI (day 3-4)

- Build `app.py` with sidebar controls (benchmark, window, rolling period).
- Implement model portfolio selector and portfolio search bar.
- Render KPI summary row using `st.columns` and `st.metric`.
- Build all Plotly charts in `charts.py` and wire them to the UI.
- Add the objective/KPI guide expandable section.

### Phase 3 -- Caching and polish (day 5)

- Add parquet-based persistent cache for price data.
- Add Fama-French factor data download and monthly refresh logic.
- Add loading spinners (`st.spinner`) during first-time data fetches.
- Error handling for invalid tickers, API failures, and missing data.
- Responsive layout testing.

### Phase 4 -- Comparison mode and deployment (day 6-7)

- Multi-portfolio comparison table and overlay charts.
- Correlation heatmap across selected portfolios.
- Deploy to Streamlit Community Cloud.
- Add shareable link to portfolio website and GitHub README.
- Final documentation pass on README and code comments.

---

## 8. Stretch goals (post-MVP)

- **Stress testing**: let the user apply scenario shocks (rates +100bp, SPX -20%, VIX spike) and see how metrics change under the stressed return distribution.
- **Monte Carlo simulation**: forward-looking return distribution using bootstrapped historical returns, displayed as a fan chart of possible portfolio paths.
- **Sector/industry decomposition**: break down portfolio exposure by GICS sector and show which sectors are driving risk.
- **PDF report export**: generate a one-page PDF summary of the dashboard for a selected portfolio, suitable for sending to a client or attaching to a job application.
- **Real-time price integration**: use a WebSocket or polling connection to update the dashboard intraday rather than end-of-day only.

---

## 9. Deployment checklist

- [ ] All dependencies pinned in `requirements.txt`
- [ ] `app.py` runs locally with `streamlit run app.py`
- [ ] GitHub repo is public at `github.com/Casazza24/factor-risk-dashboard`
- [ ] Streamlit Community Cloud app is live and accessible via shareable URL
- [ ] README includes project description, screenshots, live link, and local setup instructions
- [ ] Portfolio website links to the live dashboard
- [ ] All unit tests pass
- [ ] Objective/KPI guide section is complete and proofread
