"""Evaluation and comparison modules (report sections 3.5.9, 5.1.6, 6.1).

* ``evaluate_on_test``   - accuracy, precision, recall, F1, confusion matrix,
                           classification report on the held-out test set.
* ``cross_validate_all`` - repeated stratified k-fold estimate. The report's
                           limitations section notes that a single 61-patient
                           test split is noisy and that cross-validation gives
                           a more stable estimate, so this is provided as an
                           additional, clearly separate result.
* ``select_best``        - highest accuracy, ties broken by F1 (Algorithm 1, step 9).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .config import FEATURES, MODEL_LABELS, RANDOM_SEED, TARGET
from .preprocess import build_preprocessor


def _scores(model, X) -> np.ndarray | None:
    """Continuous score for ROC-AUC, if the model offers one."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate_on_test(name, model, X_test, y_test) -> dict:
    """Score a fitted model on the held-out test set."""
    y_pred = model.predict(X_test)
    scores = _scores(model, X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "key": name,
        "model": MODEL_LABELS.get(name, name),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores) if scores is not None else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "confusion_matrix": cm,
        "report": classification_report(
            y_test, y_pred, target_names=["No disease (0)", "Disease (1)"], zero_division=0
        ),
    }


def results_table(results: list[dict]) -> pd.DataFrame:
    cols = ["key", "model", "accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]
    return pd.DataFrame([{c: r[c] for c in cols} for r in results])


def select_best(results: list[dict]) -> dict:
    """Highest accuracy; ties broken by F1, then by recall (fewer missed patients)."""
    return max(results, key=lambda r: (round(r["accuracy"], 10), r["f1"], r["recall"]))


def cross_validate_all(
    models: dict,
    df: pd.DataFrame,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = RANDOM_SEED,
    ann_repeats: int | None = None,
) -> pd.DataFrame:
    """Repeated stratified k-fold CV with the preprocessor refit inside every fold.

    ``ann_repeats`` lets the (slower) neural network use fewer repeats.
    """
    X, y = df[FEATURES], df[TARGET]
    scoring = ["accuracy", "precision", "recall", "f1"]
    rows = []
    for key, model in models.items():
        repeats = ann_repeats if (key == "ANN" and ann_repeats is not None) else n_repeats
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=seed)
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clone(model))])
        out = cross_validate(
            pipe, X, y, cv=cv, scoring=scoring,
            n_jobs=1 if key in ("ANN", "XGB") else -1,
        )
        row = {"key": key, "model": MODEL_LABELS.get(key, key), "folds": len(out["test_accuracy"])}
        for s in scoring:
            row[f"{s}_mean"] = out[f"test_{s}"].mean()
            row[f"{s}_std"] = out[f"test_{s}"].std()
        rows.append(row)
    return pd.DataFrame(rows)
