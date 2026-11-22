import config as _config
"""ProfitLens AI Copilot functions extracted from Colab."""

import re
import pandas as pd
import torch

def ask_free_llm(question, max_new_tokens=150):

    messages = [
        {
            "role": "system",
            "content": (
                "You are ProfitLens AI, a helpful business "
                "analytics assistant. Explain concepts "
                "in simple English."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()

def route_business_question(question):

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                "Category:"
            )
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    # Restrict generation to approved category labels.
    # Compare model likelihoods instead of parsing free-form text.
    candidate_scores = {}

    with torch.inference_mode():
        for label in ROUTE_LABELS:

            label_ids = tokenizer(
                label,
                add_special_tokens=False
            ).input_ids

            candidate = torch.tensor(
                [label_ids],
                device=inputs["input_ids"].device
            )

            full_ids = torch.cat(
                [inputs["input_ids"], candidate],
                dim=1
            )

            full_mask = torch.ones_like(full_ids)

            output = model(
                input_ids=full_ids,
                attention_mask=full_mask
            )

            logits = output.logits

            start = inputs["input_ids"].shape[1] - 1
            end = start + len(label_ids)

            log_probs = torch.log_softmax(
                logits[:, start:end, :],
                dim=-1
            )

            token_scores = log_probs.gather(
                2,
                candidate.unsqueeze(-1)
            ).squeeze(-1)

            candidate_scores[label] = (
                token_scores.sum().item()
            )

    return max(candidate_scores, key=candidate_scores.get)

def execute_business_tool(route):

    approved_tools = {
        "sales": lambda: get_sales_summary(),
        "categories": lambda: get_top_categories(10),
        "cancellations": lambda: get_cancellation_summary(),
        "anomalies": lambda: get_top_anomalies(5),
        "scenarios": lambda: get_best_scenario()
    }

    if route not in approved_tools:
        return None

    result = approved_tools[route]()

    return result

def brl(value):
    return f"R${float(value):,.2f}"

def generate_verified_answer(route, result):

    if result is None:
        return "No verified data available."

    if isinstance(result, pd.DataFrame) and result.empty:
        return "No verified data available."

    if route == "sales":
        row = result.iloc[0]

        return (
            "Delivered-order sales summary:\n"
            f"• Delivered orders with items: "
            f"{int(row['total_orders']):,}\n"
            f"• Product sales: {brl(row['product_sales'])}\n"
            f"• Customer freight charged: "
            f"{brl(row['freight_charged'])}\n\n"
            "Freight charged is not shipping expense. "
            "These amounts do not represent net profit."
        )

    elif route == "categories":
        df = result.sort_values(
            "product_sales", ascending=False
        ).head(3)

        lines = ["Top three categories by product sales:"]

        for i, (_, row) in enumerate(df.iterrows(), 1):
            name = str(row["category"]).replace("_", " ").title()

            lines.append(
                f"{i}. {name}: {brl(row['product_sales'])}"
            )

        lines.append(
            "\nRankings are based on product sales value, "
            "not units sold or profitability."
        )

        return "\n".join(lines)

    elif route == "cancellations":
        row = result.iloc[0]

        return (
            f"Total orders: {int(row['total_orders']):,}\n"
            f"Canceled orders: {int(row['canceled_orders']):,}\n"
            f"Cancellation rate: "
            f"{float(row['cancellation_rate_pct']):.2f}%\n\n"
            "The rate measures the percentage of orders "
            "recorded as canceled."
        )

    elif route == "anomalies":
        lines = ["Top flagged orders for investigation:"]

        for _, row in result.head(5).iterrows():
            lines.append(
                f"• Order {row['order_id']}: "
                f"product sales {brl(row['order_value'])}, "
                f"customer freight {brl(row['freight_charged'])}, "
                f"priority {row['priority']}"
            )

        lines.append(
            "\nThese are statistical anomalies, "
            "not confirmed fraud or financial losses."
        )

        return "\n".join(lines)

    elif route == "scenarios":
        row = result.iloc[0]

        return (
            f"Highest-ranked saved scenario: {row['Scenario']}\n"
            f"Price change: {row['Price Change (%)']:+g}%\n"
            f"Assumed demand change: "
            f"{row['Demand Change (%)']:+g}%\n"
            f"Assumed cost change: "
            f"{row['Cost Change (%)']:+g}%\n"
            f"Modeled product sales: "
            f"{brl(row['Product Sales'])}\n"
            f"Assumed variable cost: "
            f"{brl(row['Variable Cost'])}\n"
            f"Modeled contribution: "
            f"{brl(row['Contribution'])}\n"
            f"Contribution change: "
            f"{brl(row['Contribution Change'])}\n\n"
            "This is a hypothetical contribution scenario, "
            "not actual net profit. Its ranking depends "
            "on the modeled assumptions."
        )

    return "This question is not supported by the approved tools."

def safe_route_business_question(question):

    q = question.lower().strip()

    # Reject actual profitability questions
    profit_terms = re.search(
        r"\b(profit|profits|profitable|profitability|"
        r"net income|earnings)\b",
        q
    )

    if profit_terms:
        return "unsupported"

    # Reject requests for a new, specific pricing simulation
    new_scenario = re.search(
        r"\b(simulate|calculate|model|what if)\b",
        q
    )

    price_terms = re.search(
        r"\b(price|prices|pricing)\b",
        q
    )

    if new_scenario and price_terms:
        return "unsupported"

    # Otherwise use the existing Qwen router
    return route_business_question(question)

def profitlens_ai(question, show_source=True):

    print("\n" + "=" * 65)
    print("PROFITLENS AI — BUSINESS DECISION COPILOT")
    print("=" * 65)

    print("\nQUESTION:")
    print(question)

    # Step 1: Let Qwen classify the question
    route = safe_route_business_question(question)

    print("\nSELECTED ANALYTICS TOOL:")
    print(route)

    # Step 2: Reject unsupported questions
    if route == "unsupported":
        print(
            "\nANSWER:\n"
            "I cannot reliably answer this question "
            "using the currently available verified data."
        )
        return

    # Step 3: Execute approved analytics function
    try:
        result = execute_business_tool(route)
    except Exception as error:
        print(
            "\nThe analytics tool encountered an error:",
            type(error).__name__
        )
        return

    # Step 4: Check for missing results
    if result is None or (
        isinstance(result, pd.DataFrame) and result.empty
    ):
        print("\nNo business data was returned.")
        return

    # Step 5: Generate factual answer using Python
    answer = generate_verified_answer(route, result)

    print("\nBUSINESS ANSWER:")
    print(answer)

    # Step 6: Display original data for verification
    if show_source:
        print("\nSOURCE DATA:")
        display(result)

    return {
        "question": question,
        "route": route,
        "answer": answer,
        "source": result
    }


# Original verified business dependencies

def get_sales_summary():
    query = """
    SELECT
        COUNT(DISTINCT oi.order_id) AS total_orders,
        ROUND(SUM(oi.price), 2) AS product_sales,
        ROUND(SUM(oi.freight_value), 2) AS freight_charged
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered';
    """

    return pd.read_sql_query(query, conn_ai)

def get_top_categories(limit=10):
    limit = max(1, min(int(limit), 20))

    query = """
    SELECT
        COALESCE(
            c.product_category_name_english,
            p.product_category_name,
            'Unknown'
        ) AS category,
        ROUND(SUM(oi.price), 2) AS product_sales
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    LEFT JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN categories c
        ON p.product_category_name = c.product_category_name
    WHERE o.order_status = 'delivered'
    GROUP BY category
    ORDER BY product_sales DESC
    LIMIT ?;
    """

    return pd.read_sql_query(
        query, conn_ai, params=(limit,)
    )

def get_cancellation_summary():
    query = """
    SELECT
        COUNT(*) AS total_orders,
        SUM(
            CASE WHEN order_status = 'canceled'
            THEN 1 ELSE 0 END
        ) AS canceled_orders,
        ROUND(
            100.0 * SUM(
                CASE WHEN order_status = 'canceled'
                THEN 1 ELSE 0 END
            ) / NULLIF(COUNT(*), 0),
            2
        ) AS cancellation_rate_pct
    FROM orders;
    """

    return pd.read_sql_query(query, conn_ai)

def get_top_anomalies(limit=5):
    if anomaly_report.empty:
        return pd.DataFrame()

    limit = max(1, min(int(limit), 20))

    return (
        anomaly_report
        .sort_values("anomaly_score")
        .head(limit)
        [
            [
                "order_id",
                "order_value",
                "freight_charged",
                "anomaly_score",
                "priority",
                "investigation_summary"
            ]
        ]
    )

def get_best_scenario():
    if scenario_report.empty:
        return pd.DataFrame()

    return (
        scenario_report
        .sort_values("Contribution", ascending=False)
        .head(1)
    )

ROUTER_SYSTEM = """
You classify questions for ProfitLens AI.

Available categories:

sales: Total product sales, order counts, freight charged.
categories: Top-selling product categories, category rankings.
cancellations: Canceled orders, cancellation rates.
anomalies: Unusual orders, anomaly detection.
scenarios: Pricing changes, assumed costs, contribution scenarios.
unsupported: Actual net profit, unrelated or unavailable data.

Examples:
Question: Which categories have the highest sales?
Category: categories

Question: Show the top 10 product categories.
Category: categories

Question: Which pricing scenario gives the highest contribution?
Category: scenarios

Question: What happens if prices increase?
Category: scenarios

Question: What are total product sales?
Category: sales

Question: How many orders were canceled?
Category: cancellations

Question: Show unusual orders.
Category: anomalies

Question: What was the actual net profit?
Category: unsupported

Return exactly one category.
"""

ROUTE_LABELS = [
    "sales",
    "categories",
    "cancellations",
    "anomalies",
    "scenarios",
    "unsupported"
]


# Streamlit-compatible business data initialization

import sqlite3
from pathlib import Path

PROJECT_ROOT = _config.PROJECT_ROOT

DB_PATH = _config.DB_PATH

ANOMALY_PATH = _config.ANOMALY_PATH

SCENARIO_PATH = _config.SCENARIO_PATH


def initialize_business_data():
    """Connect the Copilot to saved, verified business data."""

    global conn_ai, anomaly_report, scenario_report

    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    if not ANOMALY_PATH.is_file():
        raise FileNotFoundError(
            f"Anomaly report not found: {ANOMALY_PATH}"
        )

    if not SCENARIO_PATH.is_file():
        raise FileNotFoundError(
            f"Scenario report not found: {SCENARIO_PATH}"
        )

    conn_ai = sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False
    )

    anomaly_report = pd.read_csv(ANOMALY_PATH)
    scenario_report = pd.read_csv(SCENARIO_PATH)

    return {
        "database": "Connected",
        "anomaly_records": len(anomaly_report),
        "scenario_records": len(scenario_report)
    }


def configure_qwen(tokenizer_instance, model_instance):
    """Connect an already-loaded Qwen model to the Copilot."""
    global tokenizer, model
    tokenizer = tokenizer_instance
    model = model_instance



def answer_business_question(question):
    """Answer business questions using verified ProfitLens tools."""

    if not question or not question.strip():
        return "Please enter a business question."

    q = question.lower().strip()

    # Protect against unsupported actual-profit claims.
    if any(term in q for term in [
        "net profit",
        "actual profit",
        "true profit",
        "net income"
    ]):
        return (
            "Actual net profit cannot be calculated from "
            "the available verified data.\n\n"
            "ProfitLens AI has verified product sales and "
            "customer freight charges, but complete verified "
            "business costs and expenses are unavailable.\n\n"
            "The What-If Simulator estimates hypothetical "
            "contribution, not actual net profit."
        )

    if "model" not in globals() or "tokenizer" not in globals():
        raise RuntimeError("Qwen model has not been configured.")

    if "conn_ai" not in globals():
        raise RuntimeError("Business data has not been initialized.")

    # Use existing Qwen routing and approved business tools.
    route = safe_route_business_question(question)

    result = execute_business_tool(route)

    answer = generate_verified_answer(route, result)

    if answer is None or not str(answer).strip():
        return (
            "I couldn't find a verified answer to that question. "
            "Try asking about sales, product categories, "
            "cancellations, unusual orders, or saved scenarios."
        )

    answer = str(answer)

    if answer.strip().lower() == "no verified data available.":
        return (
            "I don't have enough verified information to answer "
            "that question. You can ask about product sales, "
            "categories, cancellations, unusual orders, "
            "or saved financial scenarios."
        )

    return answer
