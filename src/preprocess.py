from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .utils import (
        INPUT_FEATURES,
        LEAKAGE_COLUMNS,
        NUMERIC_FEATURES,
        PROCESSED_DIR,
        RANDOM_STATE,
        RAW_DATA_PATH,
        TABLES_DIR,
        TARGET_COLUMN,
        CATEGORICAL_FEATURES,
        ensure_directories,
        save_joblib,
        save_json,
    )
except ImportError:  # pragma: no cover
    from utils import (
        INPUT_FEATURES,
        LEAKAGE_COLUMNS,
        NUMERIC_FEATURES,
        PROCESSED_DIR,
        RANDOM_STATE,
        RAW_DATA_PATH,
        TABLES_DIR,
        TARGET_COLUMN,
        CATEGORICAL_FEATURES,
        ensure_directories,
        save_joblib,
        save_json,
    )


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # compatibility with older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _sanitize_feature_name(name: str) -> str:
    """Make feature names compatible with XGBoost and CSV headers."""
    return (
        str(name)
        .replace("[", "(")
        .replace("]", ")")
        .replace("<", "less_than")
        .replace(">", "greater_than")
    )


def _validate_columns(df: pd.DataFrame) -> None:
    required = set(INPUT_FEATURES + [TARGET_COLUMN])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")

    leakage_present = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    if len(leakage_present) != len(LEAKAGE_COLUMNS):
        print(f"[preprocess] Warning: leakage label columns present: {leakage_present}")


def preprocess_data(raw_path: Path = RAW_DATA_PATH) -> dict[str, Path]:
    """Clean, split, encode, scale, and save processed AI4I data."""
    ensure_directories()

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {raw_path}. Run src/download_data.py first."
        )

    df = pd.read_csv(raw_path)
    _validate_columns(df)

    leakage_cols = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    if leakage_cols:
        print(f"[preprocess] Excluding leakage columns from model inputs: {leakage_cols}")

    X = df[INPUT_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", _make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names = [_sanitize_feature_name(name) for name in preprocessor.get_feature_names_out()]

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    output_paths: dict[str, Path] = {}
    for split_name, (X_split, y_split) in splits.items():
        X_path = PROCESSED_DIR / f"X_{split_name}.csv"
        y_path = PROCESSED_DIR / f"y_{split_name}.csv"
        raw_path_out = PROCESSED_DIR / f"raw_{split_name}.csv"

        pd.DataFrame(X_split, columns=feature_names).to_csv(X_path, index=False)
        pd.DataFrame({TARGET_COLUMN: y_split.to_numpy()}).to_csv(y_path, index=False)
        raw_split = X.loc[y_split.index].copy()
        raw_split[TARGET_COLUMN] = y_split
        raw_split.to_csv(raw_path_out, index=False)

        output_paths[f"X_{split_name}"] = X_path
        output_paths[f"y_{split_name}"] = y_path
        output_paths[f"raw_{split_name}"] = raw_path_out

    selected_data = X.copy()
    selected_data[TARGET_COLUMN] = y
    selected_data.to_csv(PROCESSED_DIR / "selected_features_with_target.csv", index=False)

    save_joblib(preprocessor, PROCESSED_DIR / "preprocessor.joblib")
    save_json({"feature_names": feature_names}, PROCESSED_DIR / "feature_names.json")

    failure_count = int(y.sum())
    normal_count = int((y == 0).sum())
    summary_rows = [
        {"item": "total_samples", "value": int(len(df))},
        {"item": "normal_samples", "value": normal_count},
        {"item": "failure_samples", "value": failure_count},
        {"item": "failure_ratio", "value": failure_count / len(df)},
        {"item": "input_feature_count_after_encoding", "value": len(feature_names)},
        {"item": "train_samples", "value": int(len(y_train))},
        {"item": "validation_samples", "value": int(len(y_val))},
        {"item": "test_samples", "value": int(len(y_test))},
        {"item": "train_failure_samples", "value": int(y_train.sum())},
        {"item": "validation_failure_samples", "value": int(y_val.sum())},
        {"item": "test_failure_samples", "value": int(y_test.sum())},
    ]
    dataset_summary = pd.DataFrame(summary_rows)
    dataset_summary.to_csv(TABLES_DIR / "dataset_summary.csv", index=False)

    save_json(
        {
            "random_state": RANDOM_STATE,
            "split_ratio": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "target_column": TARGET_COLUMN,
            "input_features": INPUT_FEATURES,
            "excluded_identifier_columns": ["UDI/UID", "Product ID"],
            "excluded_leakage_columns": leakage_cols,
            "numeric_features_scaled": NUMERIC_FEATURES,
            "categorical_features_one_hot": CATEGORICAL_FEATURES,
            "feature_names_after_encoding": feature_names,
        },
        PROCESSED_DIR / "preprocess_meta.json",
    )

    print("[preprocess] Saved processed data to data/processed")
    return output_paths


if __name__ == "__main__":
    preprocess_data()
