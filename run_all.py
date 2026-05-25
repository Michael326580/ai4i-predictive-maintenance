from __future__ import annotations

from src.download_data import download_data
from src.evaluate import evaluate_models
from src.plot_results import plot_results
from src.preprocess import preprocess_data
from src.train_models import train_models
from src.utils import FIGURES_DIR, PROJECT_ROOT, TABLES_DIR, ensure_directories


def main() -> None:
    ensure_directories()

    print("=" * 72)
    print("AI4I predictive maintenance experiment pipeline")
    print("=" * 72)

    download_data()
    preprocess_data()
    train_models()
    evaluate_models()
    plot_results()

    print("=" * 72)
    print("Experiment pipeline finished successfully.")
    print(f"Model metrics table path: {TABLES_DIR / 'model_metrics.csv'}")
    print(f"Figures folder path: {FIGURES_DIR}")
    print(f"README path: {PROJECT_ROOT / 'README.md'}")
    print("=" * 72)


if __name__ == "__main__":
    main()