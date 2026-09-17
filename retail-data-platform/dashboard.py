"""
Retail Data Platform -- Streamlit Dashboard
============================================
Connects to Snowflake (key-pair auth) and displays retail analytics
across four tabs: Revenue, Clickstream, Operations, Pipeline Health.

Additional pip installs (beyond project requirements.txt):
    pip install streamlit plotly cryptography pandas
"""

import os
import pathlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

import snowflake.connector

from warehouse.common import load_dotenv
load_dotenv()

# Default key path if not set in .env
if not os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH"):
    os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"] = str(pathlib.Path.home() / "snowflake_key.p8")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Data Platform",
    layout="wide",
    page_icon=":bar_chart:",
)

# ---------------------------------------------------------------------------
# Snowflake connection
# ---------------------------------------------------------------------------
SCHEMA_MARTS = "DBT_DEV_MARTS"
SCHEMA_STAGING = "DBT_DEV_STAGING"


@st.cache_resource
def get_snowflake_connection():
    """Return a cached Snowflake connection using key-pair auth."""
    key_path = os.environ.get(
        "SNOWFLAKE_PRIVATE_KEY_PATH",
        str(pathlib.Path.home() / "snowflake_key.p8"),
    )
    try:
        with open(key_path, "rb") as f:
            p_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        pkb = p_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        conn = snowflake.connector.connect(
            account=os.environ.get("SNOWFLAKE_ACCOUNT", ""),
            user=os.environ.get("SNOWFLAKE_USER", ""),
            private_key=pkb,
            role=os.environ.get("SNOWFLAKE_ROLE", ""),
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", ""),
            database=os.environ.get("SNOWFLAKE_DATABASE", ""),
        )
        return conn
    except Exception as exc:
        st.error(f"Snowflake connection failed: {exc}")
        st.stop()


@st.cache_data(ttl=300)
def run_query(sql: str) -> pd.DataFrame:
    """Execute *sql* against Snowflake and return a DataFrame."""
    conn = get_snowflake_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [desc[0].upper() for desc in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    except Exception as exc:
        st.error(f"Query failed: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Plotly defaults
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PALETTE = px.colors.qualitative.Set2

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Retail Data Platform")
    st.caption("Phase 4A -- End-to-end analytics pipeline")
    st.divider()
    st.subheader("Architecture")
    st.markdown(
        """
        ```
        Sources
        ├─ Olist (orders/products)
        ├─ Clickstream (events)
        └─ Open-Meteo (weather)
              │
              ▼
        Ingestion (Python / Kafka)
              │
              ▼
        Staging  (Snowflake RAW_STAGING)
              │   dbt models
              ▼
        Marts    (Snowflake RAW_MARTS)
              │
              ▼
        Dashboard (Streamlit + Plotly)
        ```
        """
    )
    st.divider()
    st.caption("Source: Olist Brazilian E-Commerce (Kaggle, 2016–2018) + RetailRocket Clickstream (Kaggle, 2015)")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_revenue, tab_funnel, tab_ops, tab_health = st.tabs(
    [
        "Revenue Overview",
        "Clickstream Funnel",
        "Operations",
        "Pipeline Health",
    ]
)

# ========== TAB 1 -- Revenue Overview ======================================
with tab_revenue:
    st.header("Revenue Overview")

    # KPIs
    kpi_df = run_query(f"""
        SELECT
            SUM(item_revenue)                               AS total_revenue,
            COUNT(DISTINCT order_id)                        AS total_orders,
            SUM(item_revenue) / NULLIF(COUNT(DISTINCT order_id), 0)
                                                            AS avg_order_value,
            AVG(review_score)                               AS avg_review_score
        FROM {SCHEMA_MARTS}.fct_orders
        WHERE order_status != 'canceled'
    """)

    if not kpi_df.empty:
        r = kpi_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revenue", f"R$ {float(r['TOTAL_REVENUE'] or 0):,.2f}")
        c2.metric("Total Orders", f"{int(r['TOTAL_ORDERS'] or 0):,}")
        c3.metric("Avg Order Value", f"R$ {float(r['AVG_ORDER_VALUE'] or 0):,.2f}")
        c4.metric("Avg Review Score", f"{float(r['AVG_REVIEW_SCORE'] or 0):.2f} / 5", help="Average customer review across all delivered orders (1 = worst, 5 = best)")

    st.divider()

    # Monthly revenue trend
    monthly_df = run_query(f"""
        SELECT
            d.year_month,
            SUM(o.item_revenue) AS revenue
        FROM {SCHEMA_MARTS}.fct_orders o
        JOIN {SCHEMA_MARTS}.dim_date d
          ON o.order_date = d.date_day
        WHERE o.order_status != 'canceled'
        GROUP BY d.year_month
        ORDER BY d.year_month
    """)

    if not monthly_df.empty:
        fig = px.line(
            monthly_df,
            x="YEAR_MONTH",
            y="REVENUE",
            title="Monthly Revenue Trend",
            labels={"YEAR_MONTH": "Month", "REVENUE": "Revenue (R$)"},
            template=PLOTLY_TEMPLATE,
            markers=True,
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Olist launched in late 2016 as a Brazilian e-commerce marketplace. "
        "The revenue ramp reflects marketplace growth (sellers: 2 → 1,266), "
        "not seasonality. The Nov 2017 spike is Black Friday. "
        "Sep 2018 has one order — the dataset export cutoff, not a collapse."
    )

    # Revenue drivers — what's behind the growth?
    st.subheader("What's Driving Revenue Growth?")
    drivers_df = run_query(f"""
        SELECT
            d.year_month,
            COUNT(DISTINCT o.order_id)                          AS orders,
            COUNT(DISTINCT o.customer_id)                       AS unique_customers,
            COUNT(DISTINCT o.seller_id)                         AS active_sellers,
            ROUND(SUM(o.item_revenue) / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)
                                                                AS avg_order_value
        FROM {SCHEMA_MARTS}.fct_orders o
        JOIN {SCHEMA_MARTS}.dim_date d ON o.order_date = d.date_day
        WHERE o.order_status != 'canceled'
        GROUP BY d.year_month
        ORDER BY d.year_month
    """)

    if not drivers_df.empty:
        dc1, dc2 = st.columns(2)

        with dc1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drivers_df["YEAR_MONTH"], y=drivers_df["ORDERS"],
                name="Orders", mode="lines+markers", line=dict(color="#2ecc71"),
            ))
            fig.add_trace(go.Scatter(
                x=drivers_df["YEAR_MONTH"], y=drivers_df["UNIQUE_CUSTOMERS"],
                name="Unique Customers", mode="lines+markers", line=dict(color="#3498db"),
            ))
            fig.add_trace(go.Scatter(
                x=drivers_df["YEAR_MONTH"], y=drivers_df["ACTIVE_SELLERS"],
                name="Active Sellers", mode="lines+markers", line=dict(color="#e67e22"),
            ))
            fig.update_layout(
                title="Marketplace Growth: Orders, Customers & Sellers",
                xaxis_title="Month", yaxis_title="Count",
                template=PLOTLY_TEMPLATE, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

        with dc2:
            fig = px.line(
                drivers_df,
                x="YEAR_MONTH", y="AVG_ORDER_VALUE",
                title="Average Order Value Over Time",
                labels={"YEAR_MONTH": "Month", "AVG_ORDER_VALUE": "Avg Order Value (R$)"},
                template=PLOTLY_TEMPLATE, markers=True,
            )
            fig.update_traces(line_color="#9b59b6")
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Revenue growth is driven by supply-side expansion (more sellers → more products → more customers), "
            "not by increasing order values. AOV stays relatively flat around R$150-170 while order volume 3-4x'd."
        )

    st.divider()

    col_left, col_right = st.columns(2)

    # Top 10 categories
    with col_left:
        cat_df = run_query(f"""
            SELECT
                p.product_category AS category,
                SUM(o.item_revenue)             AS revenue
            FROM {SCHEMA_MARTS}.fct_orders o
            JOIN {SCHEMA_MARTS}.dim_products p
              ON o.product_id = p.product_id
            WHERE o.order_status != 'canceled'
              AND p.product_category != 'unknown'
            GROUP BY p.product_category
            ORDER BY revenue DESC
            LIMIT 10
        """)
        if not cat_df.empty:
            fig = px.bar(
                cat_df.sort_values("REVENUE"),
                x="REVENUE",
                y="CATEGORY",
                orientation="h",
                title="Top 10 Categories by Revenue",
                labels={"REVENUE": "Revenue (R$)", "CATEGORY": "Category"},
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=COLOR_PALETTE,
            )
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)

    # Revenue by state
    with col_right:
        BR_STATES = {
            "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá",
            "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal",
            "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
            "MG": "Minas Gerais", "MS": "Mato Grosso do Sul",
            "MT": "Mato Grosso", "PA": "Pará", "PB": "Paraíba",
            "PE": "Pernambuco", "PI": "Piauí", "PR": "Paraná",
            "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
            "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul",
            "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo",
            "TO": "Tocantins",
        }
        state_df = run_query(f"""
            SELECT
                c.customer_state AS state,
                SUM(o.item_revenue) AS revenue
            FROM {SCHEMA_MARTS}.fct_orders o
            JOIN {SCHEMA_MARTS}.dim_customers c
              ON o.customer_id = c.customer_id
            WHERE o.order_status != 'canceled'
            GROUP BY c.customer_state
            ORDER BY revenue DESC
            LIMIT 10
        """)
        if not state_df.empty:
            state_df["STATE_NAME"] = state_df["STATE"].map(BR_STATES).fillna(state_df["STATE"])
            fig = px.bar(
                state_df.sort_values("REVENUE"),
                x="REVENUE",
                y="STATE_NAME",
                orientation="h",
                title="Revenue by Customer State (Top 10)",
                labels={"REVENUE": "Revenue (R$)", "STATE_NAME": "State"},
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=[COLOR_PALETTE[1]],
            )
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)


# ========== TAB 2 -- Clickstream Funnel ====================================
with tab_funnel:
    st.header("Clickstream Funnel")

    # KPIs
    funnel_kpi = run_query(f"""
        SELECT
            COUNT(*)                                           AS total_events,
            COUNT(DISTINCT visitor_id)                         AS unique_visitors,
            COUNT(DISTINCT CASE WHEN event_type = 'transaction'
                                THEN visitor_id END)           AS txn_visitors,
            COUNT(DISTINCT CASE WHEN event_type = 'view'
                                THEN visitor_id END)           AS view_visitors
        FROM {SCHEMA_MARTS}.fct_events
    """)

    if not funnel_kpi.empty:
        r = funnel_kpi.iloc[0]
        view_v = int(r["VIEW_VISITORS"] or 1)
        txn_v = int(r["TXN_VISITORS"] or 0)
        conv_rate = (txn_v / view_v) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Events", f"{int(r['TOTAL_EVENTS']):,}")
        c2.metric("Unique Visitors", f"{int(r['UNIQUE_VISITORS']):,}")
        c3.metric(
            "View-to-Purchase Rate",
            f"{conv_rate:.2f}%",
            help=f"{txn_v:,} visitors who purchased out of {view_v:,} who viewed a product",
        )

    st.divider()

    col_left, col_right = st.columns([1, 1])

    # Funnel chart (vertical tapered)
    with col_left:
        funnel_df = run_query(f"""
            SELECT
                event_type,
                funnel_step,
                COUNT(DISTINCT visitor_id) AS visitors
            FROM {SCHEMA_MARTS}.fct_events
            GROUP BY event_type, funnel_step
            ORDER BY funnel_step
        """)
        if not funnel_df.empty:
            funnel_df = funnel_df.sort_values("FUNNEL_STEP")
            step_names = {"view": "View", "addtocart": "Add to Cart", "transaction": "Purchase"}
            labels = [step_names.get(t, t) for t in funnel_df["EVENT_TYPE"].tolist()]
            values = funnel_df["VISITORS"].tolist()

            fig = go.Figure(go.Funnel(
                y=labels,
                x=values,
                textinfo="value+percent previous",
                marker=dict(color=["#2ecc71", "#f1c40f", "#e74c3c"]),
            ))
            fig.update_layout(
                title="Visitor Funnel: View → Add to Cart → Purchase",
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Daily event volume
    with col_right:
        daily_events = run_query(f"""
            SELECT
                event_date,
                COUNT(*) AS events
            FROM {SCHEMA_MARTS}.fct_events
            GROUP BY event_date
            ORDER BY event_date
        """)
        if not daily_events.empty:
            fig = px.line(
                daily_events,
                x="EVENT_DATE",
                y="EVENTS",
                title="Daily Event Volume",
                labels={"EVENT_DATE": "Date", "EVENTS": "Events"},
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)


# ========== TAB 3 -- Operations ============================================
with tab_ops:
    st.header("Operations")

    col_left, col_right = st.columns(2)

    # Delivery SLA distribution
    with col_left:
        sla_df = run_query(f"""
            SELECT
                delivery_timeliness_band,
                COUNT(*) AS order_items
            FROM {SCHEMA_MARTS}.fct_orders
            WHERE delivery_timeliness_band IS NOT NULL
              AND order_status = 'delivered'
            GROUP BY delivery_timeliness_band
        """)
        if not sla_df.empty:
            sla_order = [
                ">10d early",
                "4-10d early",
                "0-3d early",
                "1-3d late",
                "4-10d late",
                ">10d late",
            ]
            color_map = {
                ">10d early": "#27ae60",
                "4-10d early": "#2ecc71",
                "0-3d early": "#a3d977",
                "1-3d late": "#e67e22",
                "4-10d late": "#e74c3c",
                ">10d late": "#c0392b",
            }
            sla_df["SORT"] = sla_df["DELIVERY_TIMELINESS_BAND"].map(
                {v: i for i, v in enumerate(sla_order)}
            )
            sla_df = sla_df.sort_values("SORT")
            sla_df["COLOR"] = sla_df["DELIVERY_TIMELINESS_BAND"].map(color_map)

            fig = go.Figure(
                go.Bar(
                    x=sla_df["DELIVERY_TIMELINESS_BAND"],
                    y=sla_df["ORDER_ITEMS"],
                    marker_color=sla_df["COLOR"],
                    text=sla_df["ORDER_ITEMS"],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Delivery SLA Distribution",
                xaxis_title="SLA Band",
                yaxis_title="Order Items",
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Payment method breakdown
    with col_right:
        pay_df = run_query(f"""
            SELECT
                payment_type,
                SUM(payment_value) AS total_value
            FROM {SCHEMA_MARTS}.fct_order_payments
            GROUP BY payment_type
            ORDER BY total_value DESC
        """)
        if not pay_df.empty:
            fig = px.pie(
                pay_df,
                names="PAYMENT_TYPE",
                values="TOTAL_VALUE",
                title="Payment Method Breakdown",
                template=PLOTLY_TEMPLATE,
                hole=0.4,
                color_discrete_sequence=COLOR_PALETTE,
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Average review by category -- top 10 and bottom 10 in one vertical chart
    st.subheader("Customer Satisfaction by Product Category")
    st.caption("Average review score (1–5 stars). Only categories with 30+ reviews shown to avoid noise from small samples.")
    review_df = run_query(f"""
        SELECT
            p.product_category AS category,
            ROUND(AVG(o.review_score), 2)   AS avg_review,
            COUNT(*)                        AS n_reviews
        FROM {SCHEMA_MARTS}.fct_orders o
        JOIN {SCHEMA_MARTS}.dim_products p
          ON o.product_id = p.product_id
        WHERE o.review_score IS NOT NULL
          AND o.is_order_header_line = TRUE
          AND p.product_category != 'unknown'
        GROUP BY p.product_category
        HAVING COUNT(*) >= 30
        ORDER BY avg_review DESC
    """)

    if not review_df.empty:
        top10 = review_df.head(10).copy()
        bot10 = review_df.tail(10).copy()
        top10["GROUP"] = "Top 10"
        bot10["GROUP"] = "Bottom 10"
        combined = pd.concat([top10, bot10], ignore_index=True)
        combined = combined.sort_values("AVG_REVIEW", ascending=True)

        fig = go.Figure()

        for _, row in combined.iterrows():
            is_top = row["GROUP"] == "Top 10"
            fig.add_trace(go.Bar(
                y=[row["CATEGORY"]],
                x=[row["AVG_REVIEW"]],
                orientation="h",
                marker_color="#2ecc71" if is_top else "#e74c3c",
                text=f"  {row['AVG_REVIEW']:.2f}  ({int(row['N_REVIEWS']):,} reviews)",
                textposition="outside",
                textfont=dict(size=12),
                showlegend=False,
            ))

        # Add invisible traces for the legend
        fig.add_trace(go.Bar(
            y=[None], x=[None], marker_color="#2ecc71",
            name="Top 10", showlegend=True,
        ))
        fig.add_trace(go.Bar(
            y=[None], x=[None], marker_color="#e74c3c",
            name="Bottom 10", showlegend=True,
        ))

        fig.update_layout(
            title="Best & Worst Rated Product Categories (Avg Review out of 5)",
            xaxis=dict(range=[0, 5.5], title="Avg Review Score", dtick=1),
            yaxis=dict(categoryorder="array", categoryarray=combined["CATEGORY"].tolist()),
            template=PLOTLY_TEMPLATE,
            height=600,
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)

# ========== TAB 4 -- Pipeline Health =======================================
with tab_health:
    st.header("Pipeline Health")

    # Row counts per mart table
    st.subheader("Mart Table Row Counts")
    mart_tables = [
        "fct_orders",
        "fct_order_payments",
        "fct_events",
        "dim_customers",
        "dim_products",
        "dim_sellers",
        "dim_items",
        "dim_date",
        "dim_weather",
    ]
    counts = []
    for tbl in mart_tables:
        df = run_query(f"SELECT COUNT(*) AS cnt FROM {SCHEMA_MARTS}.{tbl}")
        cnt = int(df.iloc[0]["CNT"]) if not df.empty else 0
        counts.append({"Table": tbl, "Row Count": f"{cnt:,}"})

    st.dataframe(
        pd.DataFrame(counts),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Data freshness
    st.subheader("Data Freshness")
    freshness_df = run_query(f"""
        SELECT
            'fct_orders'  AS source,
            MIN(order_date) AS min_date,
            MAX(order_date) AS max_date
        FROM {SCHEMA_MARTS}.fct_orders
        UNION ALL
        SELECT
            'fct_events',
            MIN(event_date),
            MAX(event_date)
        FROM {SCHEMA_MARTS}.fct_events
    """)
    if not freshness_df.empty:
        st.dataframe(
            freshness_df.rename(
                columns={
                    "SOURCE": "Table",
                    "MIN_DATE": "Earliest Date",
                    "MAX_DATE": "Latest Date",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # Unknown item rate
    st.subheader("Clickstream Data Quality")
    unknown_df = run_query(f"""
        SELECT
            COUNT(*)                                                AS total_events,
            SUM(CASE WHEN item_key = -1 THEN 1 ELSE 0 END)        AS unknown_items,
            ROUND(
                100.0 * SUM(CASE WHEN item_key = -1 THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0), 2
            )                                                       AS unknown_pct
        FROM {SCHEMA_MARTS}.fct_events
    """)
    if not unknown_df.empty:
        r = unknown_df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Events", f"{int(r['TOTAL_EVENTS'] or 0):,}")
        c2.metric("Unknown Items (key = -1)", f"{int(r['UNKNOWN_ITEMS'] or 0):,}")
        c3.metric("Unknown Item Rate", f"{float(r['UNKNOWN_PCT'] or 0)}%")

    st.divider()

    # Schema listing
    st.subheader("Schema Object Listing")
    schema_df = run_query(f"""
        SELECT
            table_schema,
            table_name,
            table_type,
            row_count,
            created
        FROM information_schema.tables
        WHERE table_schema IN ('{SCHEMA_MARTS}', '{SCHEMA_STAGING}')
        ORDER BY table_schema, table_type, table_name
    """)
    if not schema_df.empty:
        st.dataframe(
            schema_df.rename(
                columns={
                    "TABLE_SCHEMA": "Schema",
                    "TABLE_NAME": "Table",
                    "TABLE_TYPE": "Type",
                    "ROW_COUNT": "Row Count",
                    "CREATED": "Created",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
