---
title: ProfitLens AI
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# ProfitLens AI
### Retail & E-commerce Analytics, Forecasting, and AI Business Copilot

ProfitLens AI is a data science portfolio project that explores how retail and e-commerce data can support business decisions. It combines SQL-based reporting, exploratory analytics, anomaly detection, sales forecasting experiments, a financial what-if simulator, and an AI-assisted interface in a Streamlit dashboard.

> **Project status:** Research and portfolio prototype. The repository contains the analysis notebook and a Streamlit application. Public deployment and reproducibility across environments are still being validated. This is not a production financial decision system.

## The business problem

Retail teams often need to answer connected questions: Which categories generate the most sales? Are unusual orders worth investigating? How reliable are future sales estimates? How might changes in price, volume, or costs affect modeled contribution? ProfitLens AI brings these analyses into one workflow rather than treating them as isolated charts.

## Dashboard features

| Feature | Purpose |
| --- | --- |
| **Business Overview** | Review high-level business and sales indicators. |
| **Sales Analytics** | Explore trends, product/category performance, and sales distribution. |
| **Anomaly Detection** | Identify unusual orders and provide investigation context. |
| **Financial What-If Simulator** | Compare hypothetical business scenarios using explicit assumptions. |
| **AI Business Copilot** | Interpret supported business questions and retrieve figures from approved Python/SQL analytics tools. |

## Data sources and scope

The research notebook works with several **separate** retail datasets. They are not merged into a single company's financial records:

- **Olist Brazilian E-commerce:** Orders, order items, payments, products, customers, sellers, reviews, and category translations. Used for core e-commerce analytics and the packaged SQLite database.
- **Walmart M5:** Sales history, calendar, and selling-price tables, explored for retail demand data.
- **UCI Online Retail II:** Transaction-level retail data used for exploratory research.
- **dunnhumby The Complete Journey:** Household transactions, products, coupons, and campaigns, explored for promotional and customer analytics.

The repository includes a reduced Olist-derived SQLite database for the application. Raw source datasets used in the notebook are **not all included**. Dataset owners retain their respective rights; review each source's terms before redistribution or commercial use.

## Analytics and machine learning

### SQL and financial analytics

The notebook uses SQLite and Python to calculate order, product, and category metrics; examine monthly trends; assess freight charges; and cross-check product/category sales totals. **Sales and delivered-order revenue are not equivalent to net profit.** Missing cost and refund information limits profitability conclusions.

### Anomaly detection

An **Isolation Forest** model uses order-level features to flag unusual observations for review. The notebook adds business investigation rules, explanations, and priority labels. Anomaly flags are **not proof of fraud** or confirmed operational problems.

### Sales forecasting

The notebook builds monthly sales features such as lagged values and rolling averages and compares forecasting approaches, including a naive baseline and machine-learning models. It uses time-aware evaluation rather than randomly mixing future observations into training.

In one saved notebook comparison, the **naive model had the lowest reported MAE (R$ 41,682.46)**. This is an experimental result from that specific evaluation, not a general claim that a naive forecast is always best. The notebook discusses limitations of the short and irregular time series.

### Financial what-if analysis

The simulator compares hypothetical changes to business assumptions and ranks modeled outcomes. Its costs and scenario inputs are assumptions; **modeled contribution is not audited or actual net profit**. Results should be used for learning and exploratory decision support.

### AI Business Copilot

The project experiments with **Qwen/Qwen2.5-1.5B-Instruct**. The language model helps interpret user questions, while approved SQL/Python tools supply the business figures. The design includes checks to reject unsupported profitability or simulation requests instead of allowing the model to invent financial numbers.

## Architecture

```text
Retail datasets / Olist-derived SQLite database
                    |
             SQL + Python analytics
                    |
        +-----------+------------+
        |           |            |
   Sales trends  Anomalies   Scenario reports
        |           |            |
        +-----------+------------+
                    |
             Streamlit dashboard
                    |
          AI Copilot (Qwen +
          approved business tools)
```

## Technology

**Python · Pandas · NumPy · SQLite/SQL · scikit-learn · Streamlit · Plotly/Matplotlib · PyTorch · Hugging Face Transformers · Qwen**

## Repository structure

```text
ProfitLens-AI/
├── ProfitLens_AI.ipynb          # Research, data preparation, modeling, experiments
├── database/
│   └── profitlens_olist.db     # Application database
├── reports/
│   ├── anomalies/
│   └── decisions/
├── streamlit_app/
│   ├── app.py                  # Streamlit entry point
│   ├── config.py               # Paths and configuration
│   ├── requirements-deploy.txt
│   └── utils/
│       ├── copilot_core.py
│       ├── copilot_service.py
│       └── qwen_loader.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Run locally (development)

**Prerequisites:** Git, Python, enough RAM to load the selected AI model, and an internet connection for its first download. GPU availability may affect model performance. Dependencies and model loading have primarily been developed in Google Colab; the local setup below is a starting point and is **not yet verified across operating systems**.

```bash
git clone https://github.com/kirshanchand/ProfitLens-AI.git
cd ProfitLens-AI
python -m venv .venv
```

Activate the environment:

- **Windows:** `.venv\Scripts\activate`
- **macOS/Linux:** `source .venv/bin/activate`

Install dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app/app.py
```

If a dependency or model-loading error occurs, check your Python/PyTorch compatibility and available RAM. Do not assume the AI Copilot will run on every free-tier host.

## Deployment

The repository includes a Dockerfile and Hugging Face Spaces metadata (`sdk: docker`, port `7860`). **A live Space has not yet been confirmed.** Deployment may require changes to dependency versions, CPU/GPU settings, and model loading to fit the selected host.

## Limitations and responsible use

- Different retail datasets represent different sources and should not be interpreted as one unified company.
- Historical observations, missing periods, and limited test windows affect forecasting reliability.
- Unsupervised anomaly scores do not establish fraud or business loss.
- The financial simulator uses assumptions and does not calculate audited net profit.
- AI responses should be checked against the underlying approved analytics; unsupported questions may be rejected.
- The included database was reduced for sharing, but data provenance, licensing, and privacy should be reviewed before broader distribution.
- The notebook includes development experiments and may require Google Drive paths or source datasets not included in this repository.

## Next improvements

- Validate reproducible installation and Docker deployment.
- Improve forecasting evaluation with more time periods and clearer baselines.
- Add dashboard screenshots and a verified live-demo link.
- Document dataset licenses and automate selected quality checks.
- Evaluate the AI Copilot's accuracy and resource requirements on deployment hardware.

## Author

**Kirshan Chand**  
Master's in Data Science portfolio project  
[GitHub: kirshanchand](https://github.com/kirshanchand)
