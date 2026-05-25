from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

try:
    from .utils import (
        FIGURES_DIR,
        LEAKAGE_COLUMNS,
        MODELS_DIR,
        NUMERIC_FEATURES,
        RAW_DATA_PATH,
        TABLES_DIR,
        TARGET_COLUMN,
        ensure_directories,
        load_joblib,
        load_json,
    )
except ImportError:  # pragma: no cover
    from utils import (
        FIGURES_DIR,
        LEAKAGE_COLUMNS,
        MODELS_DIR,
        NUMERIC_FEATURES,
        RAW_DATA_PATH,
        TABLES_DIR,
        TARGET_COLUMN,
        ensure_directories,
        load_joblib,
        load_json,
    )


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    labels = ["Normal", "Failure"]
    colors = ["#4C78A8", "#D65F5F"]
    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(labels, counts.values, color=colors, edgecolor="#333333", linewidth=0.8)
    plt.ylabel("Samples")
    plt.title("Class Distribution")
    for bar, count in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(count)),
                 ha="center", va="bottom", fontsize=10)
    _savefig(FIGURES_DIR / "class_distribution.png")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    axes = axes.ravel()
    for idx, feature in enumerate(NUMERIC_FEATURES):
        ax = axes[idx]
        normal = df.loc[df[TARGET_COLUMN] == 0, feature]
        failure = df.loc[df[TARGET_COLUMN] == 1, feature]
        ax.hist(normal, bins=35, alpha=0.65, label="Normal", color="#4C78A8", density=True)
        ax.hist(failure, bins=35, alpha=0.65, label="Failure", color="#D65F5F", density=True)
        ax.set_title(feature)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.10))
    _savefig(FIGURES_DIR / "feature_distributions.png")


def plot_model_metrics(metrics_df: pd.DataFrame) -> None:
    metric_cols = ["precision", "recall", "f1_score", "pr_auc", "roc_auc"]
    plot_df = metrics_df.set_index("model")[metric_cols]
    ax = plot_df.plot(kind="bar", figsize=(11.5, 5.8), width=0.78)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", ncol=3)
    plt.xticks(rotation=20, ha="right")
    _savefig(FIGURES_DIR / "model_metrics_comparison.png")


def plot_roc_pr_curves(pred_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    y_true = pred_df["y_true"].to_numpy()

    plt.figure(figsize=(7.0, 5.4))
    for model_name in metrics_df["model"]:
        scores = pred_df[f"{model_name}_score"].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, scores)
        curve_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=1.8, label=f"{model_name} (AUC={curve_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.25)
    _savefig(FIGURES_DIR / "roc_curve.png")

    plt.figure(figsize=(7.0, 5.4))
    baseline = y_true.mean()
    for model_name in metrics_df["model"]:
        scores = pred_df[f"{model_name}_score"].to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, scores)
        curve_auc = auc(recall, precision)
        plt.plot(recall, precision, linewidth=1.8, label=f"{model_name} (AUC={curve_auc:.3f})")
    plt.axhline(baseline, linestyle="--", color="#777777", linewidth=1.0, label="Positive rate")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(alpha=0.25)
    _savefig(FIGURES_DIR / "pr_curve.png")


def plot_cost_sensitive_confusion(metrics_df: pd.DataFrame) -> None:
    row = metrics_df[metrics_df["model"].str.contains("CostSensitive", regex=False)].iloc[0]
    matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)

    plt.figure(figsize=(5.8, 4.8))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"Confusion Matrix: {row['model']}")
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_labels = ["Normal", "Failure"]
    plt.xticks([0, 1], tick_labels)
    plt.yticks([0, 1], tick_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    max_value = matrix.max()
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > max_value * 0.55 else "#222222"
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=12)
    _savefig(FIGURES_DIR / "confusion_matrix_cost_sensitive.png")


def plot_feature_importance() -> None:
    manifest = load_json(MODELS_DIR / "model_manifest.json")
    feature_names = manifest["feature_names"]
    cost_model_info = next(
        (item for item in manifest["models"] if item.get("is_cost_sensitive")), manifest["models"][0]
    )
    model = load_joblib(cost_model_info["path"])

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_)).ravel()

    if importances is None:
        return

    top_idx = np.argsort(importances)[::-1][:12]
    top_features = [feature_names[i] for i in top_idx]
    top_values = importances[top_idx]

    plt.figure(figsize=(8.2, 5.4))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_values, color="#59A14F", edgecolor="#333333", linewidth=0.6)
    plt.yticks(y_pos, top_features)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Feature Importance: {cost_model_info['name']}")
    plt.grid(axis="x", alpha=0.25)
    _savefig(FIGURES_DIR / "feature_importance.png")


def plot_results() -> None:
    ensure_directories()

    df = pd.read_csv(RAW_DATA_PATH)
    metrics_df = pd.read_csv(TABLES_DIR / "model_metrics.csv")
    pred_df = pd.read_csv(TABLES_DIR / "test_predictions.csv")

    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_model_metrics(metrics_df)
    plot_roc_pr_curves(pred_df, metrics_df)
    plot_cost_sensitive_confusion(metrics_df)
    plot_feature_importance()

    print("[plot] Saved figures to results/figures")


if __name__ == "__main__":
    plot_results()

