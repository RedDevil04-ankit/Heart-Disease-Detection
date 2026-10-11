#!/usr/bin/env python
"""End-to-end pipeline (report Algorithm 1).

    load -> inspect -> EDA -> preprocess -> split -> train 8 models
         -> evaluate on held-out test set -> compare -> select best

Examples
--------
    python run_pipeline.py                        # uses data/heart.csv
    python run_pipeline.py --data path/to/heart.csv --seed 7
    python run_pipeline.py --skip-ann --cv-repeats 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from heart_ml import plots
from heart_ml.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    MODEL_LABELS,
    RANDOM_SEED,
    TEST_SIZE,
)
from heart_ml.data import dataset_summary, format_summary, load_dataset
from heart_ml.evaluate import (
    cross_validate_all,
    evaluate_on_test,
    results_table,
    select_best,
)
from heart_ml.models import get_models
from heart_ml.preprocess import build_preprocessor, split_data


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="path to heart.csv or processed.cleveland.data")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="where tables, plots and models are written")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="random seed for split and models")
    p.add_argument("--test-size", type=float, default=TEST_SIZE, help="fraction held out for testing")
    p.add_argument("--skip-ann", action="store_true", help="skip the Keras neural network")
    p.add_argument("--skip-eda", action="store_true", help="skip exploratory plots")
    p.add_argument("--skip-cv", action="store_true", help="skip repeated cross-validation")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=10, help="repeats of stratified k-fold CV")
    p.add_argument("--ann-cv-repeats", type=int, default=2, help="CV repeats for the (slower) ANN")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Load and inspect --------------------------------------------------
    df = load_dataset(args.data)
    summary = dataset_summary(df)
    print("== Dataset ==")
    print(format_summary(summary))
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    # 2. EDA ---------------------------------------------------------------
    if not args.skip_eda:
        plots.plot_eda(df, out / "eda")
        print(f"EDA plots written to {out / 'eda'}")

    # 3. Preprocess + split (transformer fitted on training data only) -----
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.seed)
    preprocessor = build_preprocessor()
    Xtr = preprocessor.fit_transform(X_train)
    Xte = preprocessor.transform(X_test)
    feature_names = list(Xtr.columns)
    print(f"\n== Split == train {len(X_train)} / test {len(X_test)}  "
          f"({Xtr.shape[1]} model features after encoding), seed {args.seed}")

    # 4-5. Train and evaluate every model on the same split ----------------
    models = get_models(seed=args.seed, include_ann=not args.skip_ann)
    results, fitted = [], {}
    print("\n== Training ==")
    for key, model in models.items():
        t = time.time()
        model.fit(Xtr, y_train)
        fitted[key] = model
        res = evaluate_on_test(key, model, Xte, y_test)
        results.append(res)
        print(f"  {MODEL_LABELS[key]:<24} acc {res['accuracy']:.3f}  f1 {res['f1']:.3f}  "
              f"recall {res['recall']:.3f}  ({time.time() - t:.1f}s)")
        (out / f"classification_report_{key}.txt").write_text(res["report"])

    table = results_table(results)
    table.to_csv(out / "metrics_test_split.csv", index=False)
    best = select_best(results)
    best_key = best["key"]

    # 6. Plots for the comparison -----------------------------------------
    plots.plot_accuracy_comparison(table, best_key, out / "accuracy_comparison.png")
    plots.plot_confusion_matrix(
        best["confusion_matrix"], f"{MODEL_LABELS[best_key]}: confusion matrix", out / "confusion_matrix_best.png"
    )
    if "RF" in fitted:
        plots.plot_feature_importance(feature_names, fitted["RF"].feature_importances_, out / "rf_feature_importance.png")
    if "ANN" in fitted:
        plots.plot_training_curve(fitted["ANN"].history_, out / "ann_training_curve.png")

    # 7. Report ------------------------------------------------------------
    print("\n== Test-set results (single split) ==")
    show = table.assign(**{c: (table[c] * 100).round(1) for c in ["accuracy", "precision", "recall", "f1"]})
    print(show[["model", "accuracy", "precision", "recall", "f1", "fn", "fp"]].to_string(index=False))
    print(f"\nBest model: {MODEL_LABELS[best_key]} ({best['accuracy']:.1%} accuracy, "
          f"{int(best['fn'])} missed patients on {len(y_test)} test records)")
    print("\n" + best["report"])

    # 8. Optional: repeated stratified CV for a more stable estimate --------
    if not args.skip_cv:
        print(f"== Repeated {args.cv_folds}-fold CV ({args.cv_repeats} repeats) ==")
        cv = cross_validate_all(
            models, df, n_splits=args.cv_folds, n_repeats=args.cv_repeats,
            seed=args.seed, ann_repeats=args.ann_cv_repeats,
        )
        cv.to_csv(out / "metrics_cross_validation.csv", index=False)
        for _, r in cv.iterrows():
            print(f"  {r['model']:<24} acc {r['accuracy_mean']:.3f} +/- {r['accuracy_std']:.3f}   "
                  f"recall {r['recall_mean']:.3f}   f1 {r['f1_mean']:.3f}")

    # 9. Persist the preprocessor and the best model ------------------------
    joblib.dump(preprocessor, out / "preprocessor.joblib")
    if best_key == "ANN":
        fitted["ANN"].model_.save(out / "best_model.keras")
    else:
        joblib.dump(fitted[best_key], out / "best_model.joblib")
    (out / "run_info.json").write_text(json.dumps({
        "seed": args.seed, "test_size": args.test_size, "best_model": best_key,
        "best_accuracy": float(best["accuracy"]), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "numpy": np.__version__,
    }, indent=2))
    print(f"\nAll outputs written to {out}  (total {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)
