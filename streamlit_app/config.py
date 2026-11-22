
"""ProfitLens AI portable configuration."""

import os
from pathlib import Path

# This file lives inside streamlit_app.
APP_DIR = Path(__file__).resolve().parent

# Project folder is one level above streamlit_app.
LOCAL_PROJECT_ROOT = APP_DIR.parent

# Original Google Colab project location.
COLAB_PROJECT_ROOT = Path(
    "/content/drive/MyDrive/ProfitLens_AI"
)

# Priority:
# 1. Environment variable
# 2. Google Colab Drive, when available
# 3. Local project directory

configured_root = os.environ.get(
    "PROFITLENS_PROJECT_ROOT"
)

if configured_root:
    PROJECT_ROOT = Path(configured_root).expanduser().resolve()

elif COLAB_PROJECT_ROOT.is_dir():
    PROJECT_ROOT = COLAB_PROJECT_ROOT.resolve()

else:
    PROJECT_ROOT = LOCAL_PROJECT_ROOT

DB_PATH = (
    PROJECT_ROOT / "database/profitlens_olist.db"
)

ANOMALY_PATH = (
    PROJECT_ROOT
    / "reports/anomalies/anomaly_investigation_report.csv"
)

SCENARIO_PATH = (
    PROJECT_ROOT
    / "reports/decisions/scenario_comparison.csv"
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
