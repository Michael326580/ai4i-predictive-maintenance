from __future__ import annotations

from pathlib import Path

import requests

try:
    from .utils import DATA_URL, RAW_DATA_PATH, ensure_directories
except ImportError:  # pragma: no cover - supports direct script execution
    from utils import DATA_URL, RAW_DATA_PATH, ensure_directories


def download_data(force: bool = False) -> Path:
    """Download AI4I 2020 data to data/raw if it is not already present."""
    ensure_directories()

    if RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size > 0 and not force:
        print(f"[download] Raw data already exists: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    print(f"[download] Downloading data from: {DATA_URL}")
    try:
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        message = (
            "Unable to download AI4I 2020 data automatically.\n"
            f"Reason: {exc}\n"
            f"Manual download URL: {DATA_URL}\n"
            f"Please save the file as: {RAW_DATA_PATH}"
        )
        raise RuntimeError(message) from exc

    RAW_DATA_PATH.write_bytes(response.content)
    print(f"[download] Saved raw data to: {RAW_DATA_PATH}")
    return RAW_DATA_PATH


if __name__ == "__main__":
    download_data()

