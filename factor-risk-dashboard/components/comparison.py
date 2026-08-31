import streamlit as st
import pandas as pd

from components.kpi_cards import KPI_CONFIG


def render_comparison_table(
    all_kpis: dict[str, dict[str, float]],
) -> None:
    """Render a side-by-side KPI comparison table for multiple portfolios."""
    rows = []
    metric_names = list(next(iter(all_kpis.values())).keys())

    for metric in metric_names:
        config = KPI_CONFIG.get(metric, {"format": "{:.4f}", "good": None})
        row = {"Metric": metric}
        for port_name, kpis in all_kpis.items():
            row[port_name] = config["format"].format(kpis[metric])
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Metric")

    st.dataframe(
        df,
        use_container_width=True,
        height=35 * len(rows) + 38,
    )


def render_comparison_selector() -> list[str]:
    """Render multi-select for comparison portfolios. Returns list of selected names."""
    from config.portfolios import MODEL_PORTFOLIOS

    options = list(MODEL_PORTFOLIOS.keys())
    selected = st.multiselect(
        "Select portfolios to compare (2-4)",
        options=options,
        default=options[:2],
        max_selections=4,
        key="comparison_select",
    )
    return selected
