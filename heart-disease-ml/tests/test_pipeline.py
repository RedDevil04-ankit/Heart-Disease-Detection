"""Smoke and unit tests. They run on SYNTHETIC data with the real schema, so they
verify that the pipeline works, not that it reproduces the report's accuracies."""
import numpy as np
import pandas as pd
import pytest

from heart_ml.config import COLUMNS, FEATURES, TARGET
from heart_ml.data import dataset_summary, load_dataset
from heart_ml.evaluate import evaluate_on_test, select_best
from heart_ml.models import get_models
from heart_ml.preprocess import build_preprocessor, split_data
from make_synthetic_data import make_synthetic


@pytest.fixture(scope="module")
def df():
    return make_synthetic(303, seed=1)


def test_loader_reads_csv_and_validates(tmp_path, df):
    path = tmp_path / "heart.csv"
    df.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert list(loaded.columns) == COLUMNS
    assert dataset_summary(loaded)["rows"] == 303


def test_loader_reads_raw_uci_format_and_binarises(tmp_path, df):
    raw = df.copy()
    raw[TARGET] = np.where(raw[TARGET] == 1, 3, 0)  # UCI "num" levels
    raw.loc[0, "ca"] = np.nan
    path = tmp_path / "processed.cleveland.data"
    raw.to_csv(path, header=False, index=False, na_rep="?")
    loaded = load_dataset(path)
    assert set(loaded[TARGET].unique()) <= {0, 1}
    assert loaded["ca"].isna().sum() == 1


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_dataset(tmp_path / "nope.csv")


def test_preprocessor_scales_and_encodes_without_leakage(df):
    X_train, X_test, y_train, _ = split_data(df)
    prep = build_preprocessor()
    Xtr = prep.fit_transform(X_train)
    Xte = prep.transform(X_test)
    # standardised numeric columns: mean ~0 / std ~1 on TRAIN only
    assert abs(Xtr["age"].mean()) < 1e-6
    assert abs(Xtr["age"].std(ddof=0) - 1) < 1e-6
    # dummy columns replace the categorical codes
    assert "cp" not in Xtr.columns and any(c.startswith("cp_") for c in Xtr.columns)
    assert list(Xtr.columns) == list(Xte.columns)
    assert not Xte.isna().any().any()


def test_split_is_stratified_and_reproducible(df):
    a = split_data(df, seed=3)
    b = split_data(df, seed=3)
    assert a[1].index.equals(b[1].index)
    assert abs(a[3].mean() - a[2].mean()) < 0.05
    assert len(a[1]) == 61  # 20 % of 303, as in the report


def test_all_classical_models_fit_and_predict(df):
    X_train, X_test, y_train, y_test = split_data(df)
    prep = build_preprocessor()
    Xtr, Xte = prep.fit_transform(X_train), prep.transform(X_test)
    models = get_models(include_ann=False)
    assert {"LR", "NB", "SVM", "KNN", "DT", "RF"} <= set(models)
    results = []
    for key, model in models.items():
        model.fit(Xtr, y_train)
        res = evaluate_on_test(key, model, Xte, y_test)
        assert 0.0 <= res["accuracy"] <= 1.0
        assert res["tn"] + res["fp"] + res["fn"] + res["tp"] == len(y_test)
        results.append(res)
    best = select_best(results)
    assert best["accuracy"] == max(r["accuracy"] for r in results)


def test_ann_trains_if_keras_available(df):
    pytest.importorskip("keras")
    from heart_ml.models import KerasANN

    X_train, X_test, y_train, y_test = split_data(df)
    prep = build_preprocessor()
    Xtr, Xte = prep.fit_transform(X_train), prep.transform(X_test)
    ann = KerasANN(epochs=5).fit(Xtr, y_train)
    proba = ann.predict_proba(Xte)
    assert proba.shape == (len(X_test), 2)
    assert set(np.unique(ann.predict(Xte))) <= {0, 1}
    assert "val_loss" in ann.history_
