import streamlit as st


def render_objective_section() -> None:
    with st.expander("About this dashboard"):
        st.markdown("""
### Objective

This dashboard provides interactive portfolio risk analytics for any publicly available
ETF or index fund. Select a portfolio, choose a benchmark, and get a comprehensive
breakdown of risk-adjusted performance metrics, factor exposures, and tail-risk
indicators — all computed in real time.

---

### KPI Reference Guide

**Annualized Alpha** — The portfolio's excess return above what CAPM predicts given its
market exposure. Positive alpha means the portfolio outperformed on a risk-adjusted basis.
A value of +2.4% means 2.4% of annual return can't be explained by market exposure alone.

**Beta** — Sensitivity to benchmark movements. Beta of 1.2 means the portfolio moves 1.2%
for every 1% benchmark move. Above 1 is more aggressive; below 1 is more defensive.

**Sharpe Ratio** — Excess return per unit of total volatility. Above 1.0 is generally good;
above 2.0 is exceptional. Compares strategies on an apples-to-apples risk-adjusted basis.

**Sortino Ratio** — Like Sharpe, but only penalizes downside volatility. A Sortino much
higher than Sharpe indicates the portfolio's volatility comes mostly from upside moves
(a positive signal).

**Max Drawdown** — Largest peak-to-trough decline over the period. A -22% drawdown means
the portfolio lost 22% from its high before recovering. This answers "how bad could it get?"

**VaR (95%)** — The maximum expected daily loss at 95% confidence. A VaR of -2.1% means
on 95% of trading days, losses won't exceed 2.1%. Computed using the historical method
(5th percentile of actual returns).

**CVaR / Expected Shortfall (95%)** — Average loss on the worst 5% of days. Always worse
than VaR. The gap between VaR and CVaR reveals how fat the tail is. Required under
Basel III/IV for regulatory risk reporting.

**Information Ratio** — Active return per unit of tracking error. Measures consistency of
outperformance vs the benchmark. Above 0.5 is good; above 1.0 is exceptional.

---

### Factor Model

**CAPM** regresses portfolio excess returns against the market factor to produce alpha and
beta. **Fama-French 5-Factor + Momentum** adds size (SMB), value (HML), profitability (RMW),
investment (CMA), and momentum (MOM) factors to decompose what's really driving returns.

All regressions use Newey-West standard errors (5 lags) to correct for autocorrelation in
daily returns. T-statistics and p-values are reported alongside coefficients.

---

### Data Sources & Methodology

- **Price data**: Yahoo Finance via `yfinance`. Daily adjusted close prices.
- **Factor data**: Kenneth French Data Library (Fama-French 5 factors + momentum, daily).
- **Returns**: Simple daily returns (not log returns). Annualized assuming 252 trading days.
- **Risk-free rate**: Sourced from the Fama-French RF series for regression; a configurable
  annual rate (default 5%) for Sharpe/Sortino calculations.
- **Caching**: Price data cached for 1 hour; factor data refreshed monthly.
        """)
