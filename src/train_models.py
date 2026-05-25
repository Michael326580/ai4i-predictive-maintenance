from __future__ import annotations

import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    from .utils import (
        MODELS_DIR,
        PROCESSED_DIR,
        RANDOM_STATE,
        TABLES_DIR,
        TARGET_COLUMN,
        ensure_directories,
        load_json,
        safe_model_filename,
        save_joblib,
        save_json,
    )
except ImportError:  # pragma: no cover
    from utils import (
        MODELS_DIR,
        PROCESSED_DIR,
        RANDOM_STATE,
        TABLES_DIR,
        TARGET_COLUMN,
        ensure_directories,
        load_json,
        safe_model_filename,
        save_joblib,
        save_json,
    )


def _load_split(split: str) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(PROCESSED_DIR / f"X_{split}.csv")
    y = pd.read_csv(PROCESSED_DIR / f"y_{split}.csv")[TARGET_COLUMN].astype(int)
    return X, y


def _xgboost_classifier() -> object | None:
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None

    return XGBClassifier(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=2,
        tree_method="hist",
        verbosity=0,
    )


def _boosting_model() -> tuple[str, object]:
    xgb_model = _xgboost_classifier()
    if xgb_model is not None:
        return "XGBoost", xgb_model

    return (
        "GradientBoosting",
        GradientBoostingClassifier(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    )


def _predict_probability(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    raise AttributeError(f"Model {type(model).__name__} does not provide probabilities.")


def _threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def optimize_threshold(
    y_true: pd.Series,
    scores: np.ndarray,
    min_recall: float = 0.80,
) -> dict[str, float | str]:
    """Search a validation threshold with a recall-first F1 objective."""
    y_array = y_true.to_numpy()
    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = _threshold_metrics(y_array, scores, float(threshold))
        rows.append({"threshold": float(threshold), **metrics})

    candidates = [row for row in rows if row["recall"] >= min_recall]
    objective = f"maximize F1 under Recall >= {min_recall:.2f}"
    if not candidates:
        candidates = rows
        objective = "maximize F1 because recall constraint is infeasible"

    best = max(candidates, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    return {
        "objective": objective,
        "threshold": best["threshold"],
        "validation_precision": best["precision"],
        "validation_recall": best["recall"],
        "validation_f1": best["f1"],
    }


def _build_models(boosting_name: str, boosting_model: object) -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=320,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        boosting_name: boosting_model,
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        ),
    }


def train_models() -> dict[str, object]:
    """Train baseline models and a cost-sensitive boosting model."""
    ensure_directories()

    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    feature_meta = load_json(PROCESSED_DIR / "feature_names.json")

    boosting_name, boosting_model = _boosting_model()
    models = _build_models(boosting_name, boosting_model)

    pos_count = int(y_train.sum())
    neg_count = int((y_train == 0).sum())
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    sample_weight = np.asarray(sample_weight, dtype=float)
    sample_weight[y_train.to_numpy() == 1] *= 2.0

    cost_boosting_name, cost_boosting_model = _boosting_model()
    cost_name = f"CostSensitive_{cost_boosting_name}"

    manifest: dict[str, object] = {
        "random_state": RANDOM_STATE,
        "feature_names": feature_meta["feature_names"],
        "boosting_backend": boosting_name,
        "xgboost_available": boosting_name == "XGBoost",
        "models": [],
        "class_distribution_train": {"negative": neg_count, "positive": pos_count},
        "cost_sensitive_weighting": {
            "method": "compute_sample_weight('balanced') with positive-class multiplier",
            "positive_multiplier": 2.0,
            "motivation": "fault samples are rare and false negatives are costly",
        },
    }

    training_rows: list[dict[str, object]] = []

    for model_name, model in models.items():
        print(f"[train] Training {model_name}")
        start = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        elapsed = perf_counter() - start

        model_path = MODELS_DIR / f"{safe_model_filename(model_name)}.joblib"
        save_joblib(model, model_path)
        val_scores = _predict_probability(model, X_val)
        val_default = _threshold_metrics(y_val.to_numpy(), val_scores, 0.50)

        manifest["models"].append(
            {
                "name": model_name,
                "path": str(model_path),
                "threshold": 0.50,
                "is_cost_sensitive": False,
                "validation_metrics_at_threshold": val_default,
            }
        )
        training_rows.append(
            {
                "model": model_name,
                "seconds": elapsed,
                "threshold": 0.50,
                "validation_precision": val_default["precision"],
                "validation_recall": val_default["recall"],
                "validation_f1": val_default["f1"],
            }
        )

    print(f"[train] Training {cost_name} with sample weights")
    start = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cost_boosting_model.fit(X_train, y_train, sample_weight=sample_weight)
    elapsed = perf_counter() - start

    cost_model_path = MODELS_DIR / f"{safe_model_filename(cost_name)}.joblib"
    save_joblib(cost_boosting_model, cost_model_path)
    cost_val_scores = _predict_probability(cost_boosting_model, X_val)
    threshold_info = optimize_threshold(y_val, cost_val_scores, min_recall=0.80)
    threshold = float(threshold_info["threshold"])

    threshold_df = pd.DataFrame(
        [
            {
                "model": cost_name,
                **threshold_info,
            }
        ]
    )
    threshold_df.to_csv(TABLES_DIR / "best_threshold.csv", index=False)

    manifest["models"].append(
        {
            "name": cost_name,
            "path": str(cost_model_path),
            "threshold": threshold,
            "is_cost_sensitive": True,
            "sample_weight_positive_mean": float(sample_weight[y_train.to_numpy() == 1].mean()),
            "sample_weight_negative_mean": float(sample_weight[y_train.to_numpy() == 0].mean()),
            "threshold_optimization": threshold_info,
        }
    )
    training_rows.append(
        {
            "model": cost_name,
            "seconds": elapsed,
            "threshold": threshold,
            "validation_precision": threshold_info["validation_precision"],
            "validation_recall": threshold_info["validation_recall"],
            "validation_f1": threshold_info["validation_f1"],
        }
    )

    pd.DataFrame(training_rows).to_csv(TABLES_DIR / "training_log.csv", index=False)
    save_json(manifest, MODELS_DIR / "model_manifest.json")

    summary_text = [
        "Training completed.",
        f"Boosting backend: {boosting_name}",
        f"Training samples: {len(y_train)}, positives: {pos_count}, negatives: {neg_count}",
        f"Cost-sensitive model: {cost_name}",
        f"Best validation threshold: {threshold:.4f}",
        f"Validation recall at best threshold: {threshold_info['validation_recall']:.4f}",
        f"Validation F1 at best threshold: {threshold_info['validation_f1']:.4f}",
    ]
    (MODELS_DIR / "training_summary.txt").write_text("\n".join(summary_text), encoding="utf-8")
    print("[train] Saved models and training logs to results/models and results/tables")
    return manifest


if __name__ == "__main__":
    train_models()

