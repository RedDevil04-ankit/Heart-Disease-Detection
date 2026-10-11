"""Model training module (report sections 3.5 and 5.1.4-5.1.5).

Seven Scikit-learn / XGBoost classifiers plus a Keras ANN with one hidden
layer. The report does not list hyper-parameters, so the values below are
sensible, documented defaults; change them here in one place.

All models expose the same ``fit`` / ``predict`` interface, which is what
keeps the comparison identical across families.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import MODEL_ORDER, RANDOM_SEED


class KerasANN(ClassifierMixin, BaseEstimator):
    """Sequential(Dense(hidden, relu) -> Dense(1, sigmoid)), Adam, binary cross-entropy.

    Wrapped in the scikit-learn estimator API so it can be dropped into the same
    evaluation and cross-validation code as the other seven models. Keras is
    imported lazily so the rest of the project runs without TensorFlow.
    """

    def __init__(
        self,
        hidden_units: int = 16,
        epochs: int = 100,
        batch_size: int = 16,
        validation_split: float = 0.2,
        learning_rate: float = 1e-3,
        seed: int = RANDOM_SEED,
        verbose: int = 0,
    ):
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y):
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import keras

        keras.utils.set_random_seed(self.seed)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.classes_ = np.array([0, 1])

        model = keras.Sequential([
            keras.layers.Input(shape=(X.shape[1],)),
            keras.layers.Dense(self.hidden_units, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.history_ = model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose,
        ).history
        self.model_ = model
        return self

    def predict_proba(self, X):
        p = self.model_.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def keras_available() -> bool:
    try:
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import keras  # noqa: F401
        return True
    except Exception:
        return False


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def get_models(seed: int = RANDOM_SEED, include_ann: bool = True) -> "OrderedDict[str, BaseEstimator]":
    """Return the eight models in report Table 5 order (ANN optional)."""
    models: "OrderedDict[str, BaseEstimator]" = OrderedDict()
    models["LR"] = LogisticRegression(max_iter=1000, random_state=seed)
    models["NB"] = GaussianNB()
    models["SVM"] = LinearSVC(C=1.0, max_iter=20000, random_state=seed)
    models["KNN"] = KNeighborsClassifier(n_neighbors=5)
    models["DT"] = DecisionTreeClassifier(random_state=seed)
    models["RF"] = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)

    if xgboost_available():
        from xgboost import XGBClassifier

        models["XGB"] = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    else:
        print("[warn] xgboost not installed - skipping XGBoost (pip install xgboost)")

    if include_ann:
        if keras_available():
            models["ANN"] = KerasANN(seed=seed)
        else:
            print("[warn] keras/tensorflow not installed - skipping ANN (pip install tensorflow)")

    return OrderedDict((k, models[k]) for k in MODEL_ORDER if k in models)
