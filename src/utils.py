from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"
REPORT_DIR = PROJECT_ROOT / "report"

RAW_DATA_PATH = RAW_DIR / "ai4i2020.csv"
REPORT_PATH = REPORT_DIR / "高级机器学习理论课程报告_董艺玮_预测性维护.docx"
REPORT_NOTE_PATH = REPORT_DIR / "报告生成说明.md"

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
AI_CASES_URL = "https://ai-cases.com/predictive-maintenance/"

TARGET_COLUMN = "Machine failure"
ID_COLUMNS = ["UDI", "UID", "Product ID"]
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
INPUT_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def ensure_directories() -> None:
    """Create all project directories used by the pipeline."""
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        REPORT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_joblib(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    return joblib.load(path)


def safe_model_filename(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")


def format_percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)

