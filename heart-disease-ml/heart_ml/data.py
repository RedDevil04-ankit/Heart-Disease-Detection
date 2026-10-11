"""Data acquisition and inspection module (report section 5.1.1).

Accepts two on-disk formats:

* a CSV with a header row (the common ``heart.csv`` copy of the Cleveland
  data, columns ``age ... thal target``), and
* the raw UCI ``processed.cleveland.data`` file (no header, ``?`` = missing,
  last column ``num`` in 0..4, which is binarised to 0 / 1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import COLUMNS, DEFAULT_DATA_PATH, FEATURES, TARGET

_TARGET_ALIASES = ("target", "num", "condition", "output", "disease")

MISSING_FILE_HELP = (
    "Dataset not found at {path}.\n"
    "Get the UCI Heart Disease (Cleveland) data and place it there:\n"
    "  python scripts/download_data.py        # downloads from the UCI repository\n"
    "or copy a 303-row heart.csv (columns: {cols}) into data/.\n"
    "For a quick smoke test only, run: python scripts/make_synthetic_data.py"
)


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the dataset, returning columns in ``COLUMNS`` order."""
    path = Path(path) if path else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(MISSING_FILE_HELP.format(path=path, cols=", ".join(COLUMNS)))

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline().strip().lower()
    try:
        float(first_line.split(",")[0])
        has_header = False  # first field is a number -> raw UCI file
    except ValueError:
        has_header = True

    if has_header:
        df = pd.read_csv(path, na_values=["?", "NA", "nan", ""])
        df.columns = [c.strip().lower() for c in df.columns]
    else:
        df = pd.read_csv(path, header=None, names=COLUMNS[:-1] + ["num"], na_values=["?"])

    df = _normalise_target(df)
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    df = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing / non-numeric values.")
    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"Target must be binary (0/1); found {sorted(df[TARGET].unique())}")
    return df.reset_index(drop=True)


def _normalise_target(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the label column to ``target`` and binarise multi-class labels."""
    if TARGET not in df.columns:
        for alias in _TARGET_ALIASES:
            if alias in df.columns:
                df = df.rename(columns={alias: TARGET})
                break
    if TARGET in df.columns:
        labels = pd.to_numeric(df[TARGET], errors="coerce")
        if labels.max() > 1:  # UCI "num": 0 = no disease, 1..4 = disease
            labels = (labels > 0).astype(int)
        df[TARGET] = labels
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Shape, dtypes, missing values, duplicates and class balance (flowchart step 2)."""
    counts = df[TARGET].value_counts().sort_index()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {c: int(n) for c, n in df.isna().sum().items() if n},
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "positive_rate": float(df[TARGET].mean()),
        "feature_count": len(FEATURES),
    }


def format_summary(summary: dict) -> str:
    lines = [
        f"Rows x columns : {summary['rows']} x {summary['columns']}",
        f"Missing values : {summary['missing_total']}"
        + (f" {summary['missing_by_column']}" if summary["missing_by_column"] else ""),
        f"Duplicate rows : {summary['duplicate_rows']}",
        f"Class balance  : {summary['class_counts']} "
        f"(positive rate {summary['positive_rate']:.1%})",
    ]
    return "\n".join(lines)


def to_numpy(x) -> np.ndarray:
    """Helper used by the Keras wrapper: DataFrame or array -> float32 array."""
    return np.asarray(x, dtype="float32")
