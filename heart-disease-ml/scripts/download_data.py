#!/usr/bin/env python
"""Download the UCI Heart Disease (Cleveland) data and write data/heart.csv.

Usage:  python scripts/download_data.py

The raw UCI file has 303 rows, 6 missing values (``ca`` / ``thal``) and a
5-level label (0 = no disease, 1-4 = disease). It is binarised to 0/1 here.
Missing values are kept as blanks and imputed inside the modelling pipeline
(fitted on training data only).

Note: the widely shared Kaggle ``heart.csv`` is a re-coded copy of the same
Cleveland records (class counts 165 / 138, as quoted in the report). If you
want that exact copy, download it from Kaggle and save it as data/heart.csv;
the loader accepts both formats.
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heart_ml.config import DEFAULT_DATA_PATH, UCI_CLEVELAND_URL  # noqa: E402
from heart_ml.data import dataset_summary, format_summary, load_dataset  # noqa: E402


def main() -> int:
    data_dir = DEFAULT_DATA_PATH.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "processed.cleveland.data"

    print(f"Downloading {UCI_CLEVELAND_URL}")
    try:
        urllib.request.urlretrieve(UCI_CLEVELAND_URL, raw_path)
    except Exception as exc:  # network / policy errors
        print(f"Download failed: {exc}\n"
              "Download the file manually from the UCI repository (or a heart.csv from Kaggle) "
              f"and place it in {data_dir}.", file=sys.stderr)
        return 1

    df = load_dataset(raw_path)
    df.to_csv(DEFAULT_DATA_PATH, index=False)
    print(f"Wrote {DEFAULT_DATA_PATH}")
    print(format_summary(dataset_summary(df)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
