"""Interactive dashboard for the heart disease prediction project.

Run with:   streamlit run app.py

Tabs: Overview, Explore data, Model comparison, Cross-validation, Predict.
Everything is computed with the same code as ``run_pipeline.py``.
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from heart_ml.config import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    FEATURES,
    HIGHLIGHT,
    MODEL_LABELS,
    NEUTRAL_BAR,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
)
from heart_ml.dashboard_utils import (
    category_labels,
    feature_defaults,
    predict_patient,
    train_and_evaluate,
)
from heart_ml.data import dataset_summary, load_dataset
from heart_ml.models import keras_available

st.set_page_config(page_title="Heart Disease ML Dashboard", page_icon="🫀", layout="wide")

ORANGE = "#eb6834"
FEATURE_HELP = {
    "age": "Age in years",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Maximum heart rate achieved (bpm)",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thalassemia",
}


# --------------------------------------------------------------- data -------
@st.cache_data(show_spinner=False)
def _read_path(path: str) -> pd.DataFrame:
    return load_dataset(path)


@st.cache_data(show_spinner=False)
def _read_upload(raw: bytes, name: str) -> pd.DataFrame:
    tmp = Path(DEFAULT_OUTPUT_DIR).parent / f".upload_{abs(hash(name))}.tmp"
    tmp.write_bytes(raw)
    try:
        return load_dataset(tmp)
    finally:
        tmp.unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def _synthetic() -> pd.DataFrame:
    from scripts.make_synthetic_data import make_synthetic  # noqa: PLC0415

    return make_synthetic(303, seed=0)


@st.cache_resource(show_spinner="Training models...")
def _train(df: pd.DataFrame, seed: int, test_size: float, include_ann: bool):
    return train_and_evaluate(df, seed=seed, test_size=test_size, include_ann=include_ann)


def load_selected_data(source: str, upload) -> tuple[pd.DataFrame | None, str]:
    """Return (dataframe, banner message). Banner is '' for real data."""
    try:
        if source == "Project file (data/heart.csv)":
            if not DEFAULT_DATA_PATH.exists():
                return None, "missing"
            return _read_path(str(DEFAULT_DATA_PATH)), ""
        if source == "Upload a CSV":
            if upload is None:
                return None, "no_upload"
            return _read_upload(upload.getvalue(), upload.name), ""
        return _synthetic(), "synthetic"
    except Exception as exc:  # noqa: BLE001 - show any loader problem to the user
        return None, f"error:{exc}"


# ------------------------------------------------------------ sidebar -------
st.sidebar.title("🫀 Heart Disease ML")
source = st.sidebar.radio(
    "Data source",
    ["Project file (data/heart.csv)", "Upload a CSV", "Synthetic demo data"],
    help="Use the real UCI / Kaggle file for any result you intend to report.",
)
upload = None
if source == "Upload a CSV":
    upload = st.sidebar.file_uploader("heart.csv or processed.cleveland.data", type=["csv", "data", "txt"])

st.sidebar.markdown("### Experiment settings")
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=RANDOM_SEED, step=1)
test_size = st.sidebar.slider("Test fraction", 0.1, 0.4, float(TEST_SIZE), 0.05)
ann_ok = keras_available()
include_ann = st.sidebar.checkbox(
    "Include Keras ANN (slower)", value=False, disabled=not ann_ok,
    help="TensorFlow is not installed." if not ann_ok else "Adds the one-hidden-layer neural network.",
)

df, status = load_selected_data(source, upload)

if df is None:
    st.title("Heart Disease Prediction Dashboard")
    if status == "missing":
        st.warning(
            f"No dataset found at `{DEFAULT_DATA_PATH}`. Run `python scripts/download_data.py`, "
            "copy a Kaggle `heart.csv` into `data/`, upload a CSV from the sidebar, "
            "or switch to the synthetic demo data."
        )
    elif status == "no_upload":
        st.info("Upload a CSV in the sidebar to continue.")
    else:
        st.error(status.removeprefix("error:"))
    st.stop()

if status == "synthetic":
    st.warning(
        "**Synthetic demo data.** These records are randomly generated. Accuracies shown here say "
        "nothing about heart disease or about the report's results."
    )

LABELS = category_labels(df)
run = _train(df, int(seed), float(test_size), bool(include_ann))
summary = dataset_summary(df)

st.title("Heart Disease Prediction Dashboard")
st.caption(
    "Comparative study of classical and neural-network classifiers on the UCI Heart Disease "
    "(Cleveland) dataset. Research and teaching tool, not a medical device."
)

tab_over, tab_eda, tab_models, tab_cv, tab_predict = st.tabs(
    ["Overview", "Explore data", "Model comparison", "Cross-validation", "Predict"]
)

# ----------------------------------------------------------- Overview -------
with tab_over:
    best = next(r for r in run.results if r["key"] == run.best_key)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", summary["rows"])
    c2.metric("Disease rate", f"{summary['positive_rate']:.1%}")
    c3.metric("Best model (single split)", MODEL_LABELS[run.best_key])
    c4.metric("Best test accuracy", f"{best['accuracy']:.1%}",
              help=f"{run.n_test} test patients: one patient is {100 / run.n_test:.1f} points.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Dataset check")
        check = pd.DataFrame({
            "Item": ["Rows x columns", "Missing values", "Duplicate rows", "Class counts (0 / 1)",
                     "Train / test split"],
            "Value": [
                f"{summary['rows']} x {summary['columns']}",
                f"{summary['missing_total']}" + (f" {summary['missing_by_column']}" if summary["missing_by_column"] else ""),
                str(summary["duplicate_rows"]),
                f"{summary['class_counts'].get(0, 0)} / {summary['class_counts'].get(1, 0)}",
                f"{run.n_train} / {run.n_test} (seed {int(seed)})",
            ],
        })
        st.dataframe(check, hide_index=True, width="stretch")
    with right:
        st.subheader("Class balance")
        cb = pd.DataFrame({
            "Class": ["No disease (0)", "Disease (1)"],
            "Patients": [summary["class_counts"].get(0, 0), summary["class_counts"].get(1, 0)],
        })
        st.altair_chart(
            alt.Chart(cb).mark_bar(color=HIGHLIGHT, size=48).encode(
                x=alt.X("Class:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)),
                y=alt.Y("Patients:Q"), tooltip=["Class", "Patients"],
            ).properties(height=220),
            width="stretch",
        )
    with st.expander("Preview the data"):
        st.dataframe(df.head(25), width="stretch")
    st.info(
        f"**Read single-split results with care.** With {run.n_test} test patients, changing the seed "
        "can move accuracy by several points. Use the Cross-validation tab for a steadier estimate."
    )

# -------------------------------------------------------------- EDA ---------
with tab_eda:
    st.subheader("Feature distributions")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        feat = st.selectbox("Feature", FEATURES, format_func=lambda f: f"{f} - {FEATURE_HELP[f]}")
        split_by_class = st.toggle("Split by outcome", value=True)
    plot_df = df[[feat, TARGET]].dropna().copy()
    plot_df["Outcome"] = plot_df[TARGET].map({0: "No disease", 1: "Disease"})
    color = alt.Color("Outcome:N", scale=alt.Scale(domain=["No disease", "Disease"], range=[NEUTRAL_BAR, ORANGE]))
    with col_b:
        if feat in NUMERIC_FEATURES:
            base = alt.Chart(plot_df).mark_bar(opacity=0.8).encode(
                x=alt.X(f"{feat}:Q", bin=alt.Bin(maxbins=25), title=FEATURE_HELP[feat]),
                y=alt.Y("count():Q", title="Patients", stack=None),
                **({"color": color} if split_by_class else {"color": alt.value(HIGHLIGHT)}),
            )
        else:
            plot_df["Level"] = plot_df[feat].map(lambda v: LABELS.get(feat, {}).get(int(v), str(int(v))))
            base = alt.Chart(plot_df).mark_bar().encode(
                x=alt.X("Level:N", title=FEATURE_HELP[feat], axis=alt.Axis(labelAngle=0)),
                y=alt.Y("count():Q", title="Patients"),
                **({"color": color, "xOffset": "Outcome:N"} if split_by_class else {"color": alt.value(HIGHLIGHT)}),
            )
        st.altair_chart(base.properties(height=320), width="stretch")

    st.subheader("Disease rate by category")
    cat = st.selectbox("Categorical feature", CATEGORICAL_FEATURES + ["sex", "fbs", "exang", "ca"], key="cat_rate")
    rate = df.groupby(cat)[TARGET].agg(["mean", "count"]).reset_index()
    rate["Level"] = rate[cat].map(lambda v: LABELS.get(cat, {}).get(int(v), str(int(v))))
    rate = rate.rename(columns={"mean": "Disease rate", "count": "Patients"})
    st.altair_chart(
        alt.Chart(rate).mark_bar(color=HIGHLIGHT).encode(
            x=alt.X("Level:N", title=FEATURE_HELP[cat], axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Disease rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            tooltip=["Level", alt.Tooltip("Disease rate:Q", format=".1%"), "Patients"],
        ).properties(height=260),
        width="stretch",
    )
    st.caption("Small groups give unreliable rates; check the Patients count in the tooltip.")

    st.subheader("Correlation heatmap")
    corr = df[FEATURES + [TARGET]].corr().round(2).stack().rename("r").reset_index()
    corr.columns = ["a", "b", "r"]
    heat = alt.Chart(corr).mark_rect().encode(
        x=alt.X("a:N", title=None, sort=FEATURES + [TARGET]),
        y=alt.Y("b:N", title=None, sort=FEATURES + [TARGET]),
        color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[HIGHLIGHT, "#f4f3ef", ORANGE]), title="Pearson r"),
        tooltip=["a", "b", "r"],
    )
    text = alt.Chart(corr).mark_text(size=9).encode(
        x=alt.X("a:N", sort=FEATURES + [TARGET]), y=alt.Y("b:N", sort=FEATURES + [TARGET]),
        text=alt.Text("r:Q", format=".2f"),
        color=alt.condition("abs(datum.r) > 0.6", alt.value("white"), alt.value("#0b0b0b")),
    )
    st.altair_chart((heat + text).properties(height=480), width="stretch")

# ---------------------------------------------------- Model comparison ------
with tab_models:
    st.subheader("Held-out test set results")
    st.caption(f"{run.n_train} training / {run.n_test} test patients. Preprocessing fitted on training data only.")

    t = run.table.copy()
    t["Best"] = np.where(t["key"] == run.best_key, "✓", "")
    show = t[["Best", "model", "accuracy", "precision", "recall", "f1", "roc_auc", "tn", "fp", "fn", "tp"]]
    st.dataframe(
        show.rename(columns={"model": "Model", "accuracy": "Accuracy", "precision": "Precision",
                             "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
                             "tn": "TN", "fp": "FP", "fn": "FN", "tp": "TP"}),
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.3f")
                       for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]},
    )

    metric = st.radio("Compare by", ["accuracy", "precision", "recall", "f1", "roc_auc"],
                      horizontal=True, format_func=lambda m: m.replace("_", "-").title() if m != "f1" else "F1")
    bar_df = t[["key", "model", metric]].rename(columns={metric: "Value"})
    bar_df["is_best"] = bar_df["key"] == run.best_key
    st.altair_chart(
        alt.Chart(bar_df).mark_bar().encode(
            x=alt.X("key:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), title=metric.replace("_", "-")),
            color=alt.condition("datum.is_best", alt.value(HIGHLIGHT), alt.value(NEUTRAL_BAR)),
            tooltip=["model", alt.Tooltip("Value:Q", format=".3f")],
        ).properties(height=280),
        width="stretch",
    )
    st.caption("Highlighted bar: best model by accuracy, ties broken by F1 then recall.")

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Confusion matrix")
        pick = st.selectbox("Model", list(run.models), index=list(run.models).index(run.best_key),
                            format_func=lambda k: MODEL_LABELS[k])
        res = next(r for r in run.results if r["key"] == pick)
        cm = res["confusion_matrix"]
        cm_df = pd.DataFrame(
            [(a, p, int(cm[i, j])) for i, a in enumerate(["No disease", "Disease"])
             for j, p in enumerate(["No disease", "Disease"])],
            columns=["Actual", "Predicted", "Patients"],
        )
        base = alt.Chart(cm_df).encode(
            x=alt.X("Predicted:N", sort=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Actual:N", sort=None))
        st.altair_chart(
            (base.mark_rect().encode(color=alt.Color("Patients:Q", scale=alt.Scale(range=["#eaf1fb", HIGHLIGHT]), legend=None))
             + base.mark_text(size=22).encode(text="Patients:Q",
                                              color=alt.condition("datum.Patients > 20", alt.value("white"), alt.value("#0b0b0b")))
             ).properties(height=280),
            width="stretch",
        )
        st.caption(f"{int(res['fn'])} patients with disease were missed (false negatives); "
                   f"{int(res['fp'])} healthy patients were flagged.")
        with st.expander("Classification report"):
            st.code(res["report"])
    with c_right:
        st.subheader("ROC curves")
        roc_rows = []
        for k, (fpr, tpr) in run.roc.items():
            roc_rows += [{"Model": k, "FPR": float(a), "TPR": float(b)} for a, b in zip(fpr, tpr)]
        roc_df = pd.DataFrame(roc_rows)
        diag = pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]})
        st.altair_chart(
            (alt.Chart(diag).mark_line(strokeDash=[4, 4], color="#999").encode(x="FPR:Q", y="TPR:Q")
             + alt.Chart(roc_df).mark_line().encode(
                x=alt.X("FPR:Q", title="False positive rate"), y=alt.Y("TPR:Q", title="True positive rate"),
                color=alt.Color("Model:N"), tooltip=["Model", alt.Tooltip("FPR:Q", format=".2f"), alt.Tooltip("TPR:Q", format=".2f")])
             ).properties(height=280),
            width="stretch",
        )

    if "RF" in run.models:
        st.subheader("Random Forest feature importance")
        imp = pd.DataFrame({"Feature": run.feature_names, "Importance": run.models["RF"].feature_importances_})
        imp = imp.sort_values("Importance", ascending=False).head(12)
        st.altair_chart(
            alt.Chart(imp).mark_bar(color=HIGHLIGHT).encode(
                x=alt.X("Importance:Q", title="Mean decrease in impurity"),
                y=alt.Y("Feature:N", sort="-x", title=None),
                tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")],
            ).properties(height=320),
            width="stretch",
        )

# ------------------------------------------------------ Cross-validation ----
with tab_cv:
    st.subheader("Repeated stratified k-fold cross-validation")
    st.caption("A steadier estimate than one 61-patient split. Preprocessing is refit inside every fold.")
    cv_path = DEFAULT_OUTPUT_DIR / "metrics_cross_validation.csv"
    cv_df = None
    if cv_path.exists():
        cv_df = pd.read_csv(cv_path)
        st.success(f"Loaded `{cv_path.relative_to(cv_path.parents[1])}` from a previous `run_pipeline.py` run.")
        st.caption("These come from the saved file, so they reflect the data and seed of that run, not the sidebar.")
    with st.expander("Run cross-validation now" if cv_df is not None else "Run cross-validation", expanded=cv_df is None):
        k = st.slider("Folds", 3, 10, 5)
        reps = st.slider("Repeats", 1, 10, 3)
        if st.button("Run", type="primary"):
            from heart_ml.evaluate import cross_validate_all  # noqa: PLC0415
            from heart_ml.models import get_models  # noqa: PLC0415

            with st.spinner("Cross-validating..."):
                cv_df = cross_validate_all(get_models(seed=int(seed), include_ann=False), df,
                                           n_splits=k, n_repeats=reps, seed=int(seed))
            st.session_state["cv_live"] = cv_df
        elif "cv_live" in st.session_state:
            cv_df = st.session_state["cv_live"]

    if cv_df is not None:
        cv_show = cv_df.copy()
        st.dataframe(
            cv_show[["model", "accuracy_mean", "accuracy_std", "precision_mean", "recall_mean", "f1_mean"]].rename(
                columns={"model": "Model", "accuracy_mean": "Accuracy (mean)", "accuracy_std": "Accuracy (std)",
                         "precision_mean": "Precision", "recall_mean": "Recall", "f1_mean": "F1"}),
            hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.3f")
                           for c in ["Accuracy (mean)", "Accuracy (std)", "Precision", "Recall", "F1"]},
        )
        bars = alt.Chart(cv_show).mark_bar(color=HIGHLIGHT).encode(
            x=alt.X("key:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("accuracy_mean:Q", scale=alt.Scale(domain=[0, 1]), title="CV accuracy"),
            tooltip=["model", alt.Tooltip("accuracy_mean:Q", format=".3f"), alt.Tooltip("accuracy_std:Q", format=".3f")],
        )
        err = alt.Chart(cv_show.assign(lo=cv_show.accuracy_mean - cv_show.accuracy_std,
                                       hi=cv_show.accuracy_mean + cv_show.accuracy_std)).mark_errorbar(color="#0b0b0b").encode(
            x=alt.X("key:N", sort=None), y=alt.Y("lo:Q"), y2="hi:Q")
        st.altair_chart((bars + err).properties(height=300), width="stretch")
        st.caption("Error bars show one standard deviation across folds.")

# ------------------------------------------------------------ Predict -------
with tab_predict:
    st.subheader("Single-patient prediction")
    st.warning("Illustration only. These models were trained on 303 records from one source and are "
               "**not** validated for clinical use.")
    defaults = feature_defaults(df)
    vals: dict = {}
    cols = st.columns(3)
    for i, f in enumerate(FEATURES):
        with cols[i % 3]:
            series = df[f].dropna()
            if f in LABELS or f in CATEGORICAL_FEATURES or f == "ca":
                options = sorted(int(v) for v in series.unique())
                labels = LABELS.get(f, {})
                default_val = int(round(defaults[f]))
                idx = options.index(default_val) if default_val in options else 0
                vals[f] = st.selectbox(FEATURE_HELP[f], options, index=idx, key=f"in_{f}",
                                       format_func=lambda v, lab=labels: f"{v} - {lab[v]}" if v in lab else str(v))
            elif f == "oldpeak":
                vals[f] = st.number_input(FEATURE_HELP[f], float(series.min()), float(series.max()) + 1.0,
                                          float(defaults[f]), 0.1, key=f"in_{f}")
            else:
                vals[f] = st.number_input(FEATURE_HELP[f], int(series.min()), int(series.max()) + 20,
                                          int(round(defaults[f])), 1, key=f"in_{f}")

    out = predict_patient(run, vals)
    n_pos = int((out["prediction"] == "Disease").sum())
    st.markdown(f"### {n_pos} of {len(out)} models predict **disease present**")
    probs = out["probability"].dropna()
    if len(probs):
        st.progress(float(probs.mean()), text=f"Mean predicted probability (models with probabilities): {probs.mean():.0%}")
    st.dataframe(
        out[["model", "prediction", "probability"]].rename(
            columns={"model": "Model", "prediction": "Prediction", "probability": "P(disease)"}),
        hide_index=True, width="stretch",
        column_config={"P(disease)": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0)},
    )
    st.caption("Linear SVM gives a class decision but no calibrated probability, so its cell is empty. "
               "Disagreement between models is a sign the case is borderline.")

    st.download_button(
        "Download predictions (CSV)",
        out[["model", "prediction", "probability"]].to_csv(index=False).encode(),
        file_name="patient_predictions.csv", mime="text/csv",
    )
