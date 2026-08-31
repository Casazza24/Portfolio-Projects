import streamlit as st


KPI_CONFIG = {
    "Annualized Alpha": {"format": "{:+.2%}", "good": "positive"},
    "Beta": {"format": "{:.2f}", "good": None},
    "Sharpe Ratio": {"format": "{:.2f}", "good": "positive"},
    "Sortino Ratio": {"format": "{:.2f}", "good": "positive"},
    "Max Drawdown": {"format": "{:.2%}", "good": "negative"},
    "VaR (95%)": {"format": "{:.2%}", "good": "negative"},
    "CVaR (95%)": {"format": "{:.2%}", "good": "negative"},
    "Information Ratio": {"format": "{:.2f}", "good": "positive"},
}


def render_kpi_row(kpis: dict[str, float]) -> None:
    cols = st.columns(len(kpis))
    for col, (name, value) in zip(cols, kpis.items()):
        config = KPI_CONFIG.get(name, {"format": "{:.4f}", "good": None})
        formatted = config["format"].format(value)

        if config["good"] == "positive":
            color = "#00CC96" if value > 0 else "#EF553B"
        elif config["good"] == "negative":
            color = "#00CC96" if value > -0.10 else "#EF553B"
        else:
            color = "#636EFA"

        col.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 16px 12px;
                text-align: center;
            ">
                <div style="color: #8892b0; font-size: 11px; text-transform: uppercase;
                            letter-spacing: 0.5px; margin-bottom: 6px;">{name}</div>
                <div style="color: {color}; font-size: 22px; font-weight: 700;">{formatted}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
