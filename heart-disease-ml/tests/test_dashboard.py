"""Tests for the dashboard helpers and a headless run of the Streamlit app (synthetic data)."""
import numpy as np
import pytest

from heart_ml.config import FEATURES
from heart_ml.dashboard_utils import feature_defaults, patient_frame, predict_patient, train_and_evaluate
from make_synthetic_data import make_synthetic


@pytest.fixture(scope="module")
def df():
    return make_synthetic(303, seed=2)


@pytest.fixture(scope="module")
def run(df):
    return train_and_evaluate(df, include_ann=False)


def test_train_and_evaluate_matches_split_sizes(run):
    assert run.n_test == 61 and run.n_train == 242
    assert run.best_key in run.models
    assert {"LR", "RF"} <= set(run.roc)  # models with scores have ROC curves


def test_patient_frame_requires_all_features(df):
    values = feature_defaults(df)
    assert list(patient_frame(values).columns) == FEATURES
    values.pop("age")
    with pytest.raises(ValueError, match="age"):
        patient_frame(values)


def test_predict_patient_returns_one_row_per_model(run, df):
    out = predict_patient(run, feature_defaults(df))
    assert len(out) == len(run.models)
    assert set(out["prediction"]) <= {"Disease", "No disease"}
    probs = out["probability"].dropna()
    assert ((probs >= 0) & (probs <= 1)).all()
    # Linear SVM has no calibrated probability -> left empty, not invented
    assert np.isnan(out.loc[out["key"] == "SVM", "probability"]).all()


def test_streamlit_app_runs_headless():
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(app), default_timeout=180).run()
    at.sidebar.radio[0].set_value("Synthetic demo data").run()
    assert not at.exception
    assert len(at.tabs) == 5
