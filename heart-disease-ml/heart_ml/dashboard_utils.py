"""Helpers for the Streamlit dashboard (``app.py``).

Kept free of any Streamlit import so they can be unit-tested. They reuse the
same preprocessing, models and metrics as ``run_pipeline.py``, so the numbers
shown in the dashboard come from the identical pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import FEATURES, MODEL_LABELS, RANDOM_SEED, TARGET, TEST_SIZE
from .evaluate import evaluate_on_test, results_table, select_best
from .models import get_models
from .preprocess import build_preprocessor, split_data

# Human-readable meaning of the coded categorical features (UCI documentation).
CATEGORY_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"},
    "fbs": {0: "No (<= 120 mg/dl)", 1: "Yes (> 120 mg/dl)"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "thal": {1: "Fixed defect", 2: "Normal", 3: "Reversible defect"},
}


# Raw UCI coding (what ``scripts/download_data.py`` writes): cp 1-4, slope 1-3,
# thal 3 / 6 / 7. The CATEGORY_LABELS above follow the 0-based Kaggle copy.
UCI_CATEGORY_LABELS = {
    **CATEGORY_LABELS,
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}


def category_labels(df: pd.DataFrame) -> dict:
    """Pick the label set that matches how the loaded file codes its categories."""
    uci_style = df["cp"].max() == 4 or df["thal"].max() >= 6
    return UCI_CATEGORY_LABELS if uci_style else CATEGORY_LABELS


@dataclass
class TrainedRun:
    """Everything the dashboard needs after one train/evaluate pass."""

    preprocessor: object
    models: dict
    results: list
    table: pd.DataFrame
    best_key: str
    feature_names: list
    X_test: pd.DataFrame           # preprocessed test features
    y_test: pd.Series
    n_train: int
    n_test: int
    roc: dict = field(default_factory=dict)


def roc_points(model, X_test, y_test) -> tuple[np.ndarray, np.ndarray] | None:
    """FPR / TPR arrays for one fitted model, or None if it has no score."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        return None
    fpr, tpr, _ = roc_curve(y_test, scores)
    return fpr, tpr


def train_and_evaluate(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
    include_ann: bool = False,
    model_keys: list[str] | None = None,
) -> TrainedRun:
    """Split, preprocess (fit on train only), train and evaluate the models."""
    X_train, X_test, y_train, y_test = split_data(df, test_size, seed)
    prep = build_preprocessor()
    Xtr = prep.fit_transform(X_train)
    Xte = prep.transform(X_test)

    models = get_models(seed=seed, include_ann=include_ann)
    if model_keys:
        models = {k: m for k, m in models.items() if k in model_keys}

    results, roc = [], {}
    for key, model in models.items():
        model.fit(Xtr, y_train)
        results.append(evaluate_on_test(key, model, Xte, y_test))
        pts = roc_points(model, Xte, y_test)
        if pts is not None:
            roc[key] = pts
    best = select_best(results)
    return TrainedRun(
        preprocessor=prep,
        models=models,
        results=results,
        table=results_table(results),
        best_key=best["key"],
        feature_names=list(Xtr.columns),
        X_test=Xte,
        y_test=y_test,
        n_train=len(X_train),
        n_test=len(X_test),
        roc=roc,
    )


def patient_frame(values: dict) -> pd.DataFrame:
    """One-row DataFrame in the raw feature schema, ready for the preprocessor."""
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError(f"Missing patient values: {missing}")
    return pd.DataFrame([{f: values[f] for f in FEATURES}])


def predict_patient(run: TrainedRun, values: dict) -> pd.DataFrame:
    """Prediction of every trained model for one patient.

    Returns a table with the predicted class and, where the model provides one,
    the probability of disease. Linear SVM has no calibrated probability, so
    that cell is left empty rather than invented.
    """
    X = run.preprocessor.transform(patient_frame(values))
    rows = []
    for key, model in run.models.items():
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else np.nan
        rows.append({
            "key": key,
            "model": MODEL_LABELS.get(key, key),
            "prediction": "Disease" if pred == 1 else "No disease",
            "probability": prob,
        })
    return pd.DataFrame(rows)


def feature_defaults(df: pd.DataFrame) -> dict:
    """Median of every feature, used as the starting point of the patient form."""
    return {f: float(df[f].median()) for f in FEATURES}


__all__ = [
    "CATEGORY_LABELS", "UCI_CATEGORY_LABELS", "category_labels", "TrainedRun", "train_and_evaluate", "predict_patient",
    "patient_frame", "feature_defaults", "roc_points", "TARGET",
]
