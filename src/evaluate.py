from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .utils import (
        MODELS_DIR,
        PROCESSED_DIR,
        TABLES_DIR,
        TARGET_COLUMN,
        ensure_directories,
        load_joblib,
        load_json,
    )
except ImportError:  # pragma: no cover
    from utils import (
        MODELS_DIR,
        PROCESSED_DIR,
        TABLES_DIR,
        TARGET_COLUMN,
        ensure_directories,
        load_joblib,
        load_json,
    )


def _load_test() -> tuple[pd.DataFrame, pd.Series]:
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")[TARGET_COLUMN].astype(int)
    return X_test, y_test


def _predict_probability(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    raise AttributeError(f"Model {type(model).__name__} does not provide probabilities.")


def evaluate_models() -> pd.DataFrame:
    """Evaluate all trained models on the held-out test set."""
    ensure_directories()

    manifest = load_json(MODELS_DIR / "model_manifest.json")
    X_test, y_test = _load_test()
    y_true = y_test.to_numpy()

    metrics_rows: list[dict[str, object]] = []
    cm_rows: list[dict[str, object]] = []
    prediction_frame = pd.DataFrame({"y_true": y_true})

    for model_info in manifest["models"]:
        model_name = model_info["name"]
        threshold = float(model_info.get("threshold", 0.50))
        model = load_joblib(model_info["path"])
        scores = _predict_probability(model, X_test)
        y_pred = (scores >= threshold).astype(int)

        try:
            roc_auc = roc_auc_score(y_true, scores)
        except ValueError:
            roc_auc = float("nan")

        try:
            pr_auc = average_precision_score(y_true, scores)
        except ValueError:
            pr_auc = float("nan")

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics_rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
        cm_rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )
        prediction_frame[f"{model_name}_score"] = scores
        prediction_frame[f"{model_name}_pred"] = y_pred

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(TABLES_DIR / "model_metrics.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(TABLES_DIR / "confusion_matrix.csv", index=False)
    prediction_frame.to_csv(TABLES_DIR / "test_predictions.csv", index=False)

    print("[evaluate] Saved metrics and predictions to results/tables")
    return metrics_df


if __name__ == "__main__":
    evaluate_models()

