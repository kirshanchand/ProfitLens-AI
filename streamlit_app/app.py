import config as _config
import pandas as pd

import streamlit as st
import sqlite3
from pathlib import Path

st.set_page_config(
    page_title="ProfitLens AI",
    page_icon="📊",
    layout="wide"
)

PROJECT_ROOT = _config.PROJECT_ROOT
DB_PATH = _config.DB_PATH

st.title("📊 ProfitLens AI")
st.caption("Revenue, Sales Analytics & Business Decision Intelligence")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Business Overview",
        "Sales Analytics",
        "Anomaly Detection",
        "Financial Simulator",
        "AI Business Copilot"
    ]
)

@st.cache_data
def load_overview():
    if not DB_PATH.is_file():
        return None

    with sqlite3.connect(str(DB_PATH)) as conn:
        sales = conn.execute("""
            SELECT
                COALESCE(SUM(oi.price), 0),
                COUNT(DISTINCT oi.order_id),
                COALESCE(SUM(oi.freight_value), 0)
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.order_id
            WHERE o.order_status = 'delivered'
        """).fetchone()

        cancellations = conn.execute("""
            SELECT
                COUNT(*)
            FROM orders
            WHERE order_status = 'canceled'
        """).fetchone()[0]

        total_orders = conn.execute(
            "SELECT COUNT(*) FROM orders"
        ).fetchone()[0]

    return {
        "sales": sales[0],
        "delivered_orders": sales[1],
        "freight": sales[2],
        "cancellations": cancellations,
        "total_orders": total_orders
    }

if page == "Business Overview":

    st.header("Business Overview")

    try:
        data = load_overview()
    except sqlite3.Error as error:
        st.error(f"Unable to load database metrics: {error}")
        st.stop()

    if data is None:
        st.error("Saved SQLite database not found.")
        st.stop()

    cancellation_rate = (
        100 * data["cancellations"] / data["total_orders"]
        if data["total_orders"] else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Product Sales",
        f"R${data['sales']:,.2f}"
    )

    col2.metric(
        "Delivered Orders",
        f"{data['delivered_orders']:,}"
    )

    col3.metric(
        "Cancellation Rate",
        f"{cancellation_rate:.2f}%"
    )

    col4.metric(
        "Customer Freight Charged",
        f"R${data['freight']:,.2f}"
    )

    st.info(
        "Sales and freight are based on delivered orders "
        "with item records. Freight charged to customers "
        "is not the actual shipping expense."
    )

elif page == "Sales Analytics":
    st.header("Sales Analytics")
    st.subheader("Top 10 Product Categories by Sales")

    @st.cache_data
    def load_category_sales():
        with sqlite3.connect(str(DB_PATH)) as conn:
            return pd.read_sql_query("""
                SELECT
                    COALESCE(
                        p.product_category_name,
                        'Unknown'
                    ) AS category,
                    SUM(oi.price) AS product_sales
                FROM order_items oi
                JOIN orders o
                    ON oi.order_id = o.order_id
                JOIN products p
                    ON oi.product_id = p.product_id
                WHERE o.order_status = 'delivered'
                GROUP BY 1
                ORDER BY product_sales DESC
                LIMIT 10
            """, conn)

    try:
        category_sales = load_category_sales()

        # Translate displayed category names.
        translations = {
            "beleza_saude": "Health & Beauty",
            "relogios_presentes": "Watches & Gifts",
            "cama_mesa_banho": "Bed, Bath & Table",
            "esporte_lazer": "Sports & Leisure",
            "informatica_acessorios": "Computers & Accessories",
            "moveis_decoracao": "Furniture & Decor",
            "utilidades_domesticas": "Housewares",
            "cool_stuff": "Cool Stuff",
            "automotivo": "Automotive",
            "brinquedos": "Toys"
        }

        category_sales["category"] = (
            category_sales["category"]
            .replace(translations)
        )

        # Sort from highest to lowest product sales.
        category_sales = category_sales.sort_values(
            "product_sales", ascending=False
        )

        # Horizontal bar chart with readable labels.
        import plotly.express as px

        fig = px.bar(
            category_sales,
            x="product_sales",
            y="category",
            orientation="h",
            text="product_sales",
            labels={
                "product_sales": "Product Sales (R$)",
                "category": "Product Category"
            }
        )

        fig.update_traces(
            texttemplate="R$%{x:,.0f}",
            textposition="outside"
        )

        fig.update_layout(
            yaxis={
                "categoryorder": "array",
                "categoryarray": category_sales["category"].iloc[::-1].tolist()
            },
            height=520,
            margin=dict(l=20, r=90, t=20, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            category_sales,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Product sales from delivered orders. "
            "These are not net profit figures."
        )

    except Exception as error:
        st.error(f"Unable to load sales chart: {error}")


elif page == "Anomaly Detection":
    st.header("Anomaly Detection")
    st.caption("Isolation Forest | Order Investigation Monitor")

    report_path = (
        PROJECT_ROOT
        / "reports/anomalies/anomaly_investigation_report.csv"
    )

    if not report_path.exists():
        st.error("Saved anomaly investigation report not found.")
        st.stop()

    @st.cache_data
    def load_anomalies():
        return pd.read_csv(report_path)

    anomaly_data = load_anomalies()

    st.metric("Orders in Investigation Report", len(anomaly_data))

    st.warning(
        "Anomalies are unusual transactions identified by a "
        "machine learning model. They are not confirmed fraud "
        "or financial losses."
    )

    st.subheader("Flagged Orders")

    st.dataframe(
        anomaly_data,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Anomaly Investigation Visualization")

    if {"order_value", "anomaly_score"}.issubset(anomaly_data.columns):

        import plotly.express as px

        fig = px.scatter(
            anomaly_data,
            x="order_value",
            y="anomaly_score",
            hover_data=["order_id"]
                if "order_id" in anomaly_data.columns else None,
            title="Order Value vs Anomaly Score",
            labels={
                "order_value": "Product Order Value (R$)",
                "anomaly_score": "Isolation Forest Anomaly Score"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "The saved report does not contain the columns "
            "needed for the scatter chart."
        )


elif page == "Financial Simulator":
    st.header("Financial What-If Simulator")
    st.caption("Compare hypothetical business decisions")

    scenario_path = (
        PROJECT_ROOT / "reports/decisions/scenario_comparison.csv"
    )

    if not scenario_path.is_file():
        st.error("Saved financial scenarios not found.")
        st.stop()

    @st.cache_data
    def load_scenarios():
        return pd.read_csv(scenario_path)

    scenarios = load_scenarios()

    required = {
        "Scenario", "Product Sales",
        "Variable Cost", "Contribution"
    }

    if not required.issubset(scenarios.columns):
        st.error("Required scenario columns are missing.")
        st.stop()

    # Validate saved financial calculations.
    difference = (
        scenarios["Product Sales"]
        - scenarios["Variable Cost"]
        - scenarios["Contribution"]
    ).abs()

    if (difference > 0.01).any():
        st.error("Scenario arithmetic validation failed.")
        st.stop()

    st.warning(
        "These are hypothetical financial scenarios. "
        "Contribution is product sales minus assumed variable costs, "
        "not verified net profit."
    )

    st.subheader("Saved Business Scenarios")

    st.dataframe(
        scenarios,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Compare Two Strategies")

    names = scenarios["Scenario"].tolist()

    col1, col2 = st.columns(2)

    with col1:
        strategy_a = st.selectbox("Strategy A", names, index=0)

    with col2:
        strategy_b = st.selectbox(
            "Strategy B", names,
            index=min(1, len(names) - 1)
        )

    a = scenarios.loc[
        scenarios["Scenario"] == strategy_a
    ].iloc[0]

    b = scenarios.loc[
        scenarios["Scenario"] == strategy_b
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Strategy A Contribution",
        f"R${a['Contribution']:,.2f}"
    )

    col2.metric(
        "Strategy B Contribution",
        f"R${b['Contribution']:,.2f}"
    )

    col3.metric(
        "Contribution Difference (A - B)",
        f"R${a['Contribution'] - b['Contribution']:,.2f}"
    )

    import plotly.express as px

    fig = px.bar(
        scenarios.sort_values("Contribution", ascending=False),
        x="Scenario",
        y="Contribution",
        title="Hypothetical Contribution by Scenario",
        labels={"Contribution": "Contribution (R$)"}
    )

    st.plotly_chart(fig, use_container_width=True)


elif page == "AI Business Copilot":
    st.header("AI Business Copilot")
    st.caption("Qwen AI with verified ProfitLens business data")

    from utils.copilot_service import ask_profitlens

    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = []

    st.subheader("Suggested questions")

    suggestions = [
        "What are our total product sales?",
        "Which categories generate the most sales?",
        "How many orders were canceled?",
        "Which scenario has the highest contribution?",
        "Show the most unusual orders.",
        "What is our actual net profit?"
    ]

    selected_question = None

    # Display suggestions in two columns.
    columns = st.columns(2)

    for i, suggestion in enumerate(suggestions):
        with columns[i % 2]:
            if st.button(
                suggestion,
                key=f"copilot_suggestion_{i}",
                use_container_width=True
            ):
                selected_question = suggestion

    if st.button("Clear conversation"):
        st.session_state.copilot_messages = []
        st.rerun()

    st.divider()

    # Display conversation history.
    for message in st.session_state.copilot_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_question = st.chat_input(
        "Ask ProfitLens AI about your business..."
    )

    # Accept either a suggested or typed question.
    question = selected_question or typed_question

    if question:
        st.session_state.copilot_messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing business data..."):
                try:
                    answer = ask_profitlens(question)

                    st.markdown(answer)

                    st.session_state.copilot_messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:
                    st.error(
                        "The Copilot could not answer this question."
                    )
                    st.caption(
                        f"{type(error).__name__}: {error}"
                    )
