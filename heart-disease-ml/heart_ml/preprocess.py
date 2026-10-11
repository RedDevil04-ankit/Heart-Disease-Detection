"""Preprocessing module (report section 5.1.2) and train/test split.

* cp, restecg, slope, thal  -> dummy (one-hot) variables
* age, trestbps, chol, thalach, oldpeak -> standardised (zero mean, unit variance)
* sex, fbs, exang, ca       -> passed through unchanged

The transformer is fitted on the training portion only, so no information from
the test set leaks into scaling or encoding. Median / most-frequent imputers are
included as a safety net (the Kaggle-style CSV has no missing values; the raw
UCI file has 6 missing ``ca`` / ``thal`` entries).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    PASSTHROUGH_FEATURES,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
)


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer implementing the report's encoding/scaling."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    passthrough = SimpleImputer(strategy="most_frequent")
    transformer = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("bin", passthrough, PASSTHROUGH_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform="pandas")
    return transformer


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
):
    """Stratified train/test split with a fixed seed (all models share it)."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
