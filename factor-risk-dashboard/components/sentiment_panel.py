# components/sentiment_panel.py
import streamlit as st
import pandas as pd
import numpy as np

from src.validation import incremental_r_squared, factor_orthogonality_test
from src.metrics import regime_conditioned_risk, sentiment_stress_impact, annualized_volatility
from src.charts import regime_risk_chart, rolling_sentiment_chart


def render_sentiment_panel(
    portfolio_returns: pd.Series,
    factor_data: pd.DataFrame,
    ff_result: dict,
    sent_factor: pd.Series | None,
) -> None:
    if sent_factor is None or sent_factor.empty:
        st.info(
            "No sentiment factor data available. Run the ingestion and scoring pipeline "
            "to populate sentiment data. See the About section for details."
        )
        return

    sent_beta = ff_result.get("factor_betas", {}).get("SENT", None)

    # ── Sentiment Beta Card ──────────────────────────────────
    if sent_beta is not None:
        sent_tstat = ff_result.get("factor_tstats", {}).get("SENT", 0)
        sent_pval = ff_result.get("factor_pvalues", {}).get("SENT", 1)
        significant = abs(sent_tstat) > 1.96

        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment Beta", f"{sent_beta:.4f}")
        col2.metric("t-statistic", f"{sent_tstat:.2f}")
        col3.metric("p-value", f"{sent_pval:.4f}", delta="significant" if significant else "not significant")

        if significant:
            direction = "positive" if sent_beta > 0 else "negative"
            st.caption(
                f"Your portfolio has a statistically significant {direction} sentiment loading. "
                f"A 1σ positive sentiment shock is associated with a "
                f"{abs(sent_beta * annualized_volatility(sent_factor) / np.sqrt(252)) * 100:.1f} bps return impact."
            )

    # ── Incremental R² ───────────────────────────────────────
    st.subheader("Incremental Explanatory Power")
    inc_r2 = incremental_r_squared(portfolio_returns, factor_data)

    col1, col2, col3 = st.columns(3)
    col1.metric("R² without SENT", f"{inc_r2['r2_without_sent']:.4f}")
    col2.metric("R² with SENT", f"{inc_r2['r2_with_sent']:.4f}")
    col3.metric("ΔR²", f"{inc_r2['delta_r2']:.4f}",
                delta=f"F={inc_r2['f_stat']:.2f}, p={inc_r2['f_pvalue']:.4f}")

    # ── Orthogonality ────────────────────────────────────────
    st.subheader("Factor Orthogonality")
    orth = factor_orthogonality_test(factor_data)
    st.caption(
        f"SENT regressed on FF5+MOM: alpha={orth['alpha']:.4f} "
        f"(t={orth['alpha_tstat']:.2f}, p={orth['alpha_pvalue']:.4f}), "
        f"R²={orth['r_squared']:.4f}. "
        f"{'SENT carries independent information.' if orth['alpha_pvalue'] < 0.05 else 'SENT may be partially redundant with existing factors.'}"
    )

    # ── Regime-Conditioned Risk ──────────────────────────────
    st.subheader("Sentiment Regime Risk")
    regime = regime_conditioned_risk(portfolio_returns, sent_factor)
    st.plotly_chart(regime_risk_chart(regime), use_container_width=True)

    # ── Stress Test ──────────────────────────────────────────
    st.subheader("Sentiment Stress Test")
    if sent_beta is not None:
        shock = st.slider("Sentiment shock (σ)", min_value=-3.0, max_value=3.0, value=-2.0, step=0.5,
                          key="sent_shock_slider")
        sent_vol = annualized_volatility(sent_factor)
        impact = sentiment_stress_impact(sent_beta, shock, sent_vol)
        color = "#00CC96" if impact > 0 else "#EF553B"
        st.markdown(
            f"A **{shock:.1f}σ** sentiment shock implies a "
            f"<span style='color:{color};font-weight:bold'>{impact*100:.2f} bps</span> "
            f"daily portfolio impact.",
            unsafe_allow_html=True,
        )

    # ── Rolling Sentiment Overlay ────────────────────────────
    st.subheader("Rolling Sentiment vs Portfolio")
    port_cum = (1 + portfolio_returns).cumprod() * 100
    st.plotly_chart(
        rolling_sentiment_chart(sent_factor, port_cum, window=60),
        use_container_width=True,
    )
