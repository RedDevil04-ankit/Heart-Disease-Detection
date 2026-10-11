# Heart Disease Prediction Using Machine Learning

A comparative study of classical and neural-network classifiers on the UCI Heart Disease (Cleveland) dataset.
This repository is the runnable code behind the report *"Heart Disease Prediction Using Machine Learning: A Comparative Study of Classical and Neural Network Classifiers"* (Ankit Ghosh and Shreej Aditya, Manipal Institute of Technology, Bengaluru, 2026-27).

Eight models are trained and evaluated under one identical pipeline: Logistic Regression, Naive Bayes, linear SVM, KNN, Decision Tree, Random Forest, XGBoost, and a one-hidden-layer Keras ANN.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py      # fetches the UCI data into data/heart.csv
python run_pipeline.py               # runs everything, writes results/
```

No TensorFlow? `pip install` everything except `tensorflow` and run `python run_pipeline.py --skip-ann`.

Prefer the Kaggle `heart.csv` (the copy with 165 / 138 class counts quoted in the report)? Save it as `data/heart.csv`; the loader accepts both formats. See `data/README.md`.

## What the pipeline does (report Algorithm 1)

| Step | Report section | Code |
|---|---|---|
| Load, inspect shape / types / missing values / duplicates | 5.1.1 | `heart_ml/data.py` |
| EDA: class counts, histograms, correlation heatmap | 5.1.3 | `heart_ml/plots.py` |
| Dummy-encode `cp, restecg, slope, thal`; standardise `age, trestbps, chol, thalach, oldpeak` | 5.1.2 | `heart_ml/preprocess.py` |
| 80/20 stratified split with a fixed seed (61 test patients) | 5.1.2 | `heart_ml/preprocess.py` |
| Train the 8 models on the same training set | 5.1.4-5.1.5 | `heart_ml/models.py` |
| Accuracy, precision, recall, F1, confusion matrix, classification report | 5.1.6 | `heart_ml/evaluate.py` |
| Compare models, select best by accuracy then F1 | 6.1 | `run_pipeline.py` |

The scaler and encoder are fitted on the training data only, so nothing from the test set leaks into preprocessing.

## Outputs (`results/`)

| File | Content |
|---|---|
| `metrics_test_split.csv` | Table 5 equivalent: accuracy, precision, recall, F1, ROC-AUC, TN/FP/FN/TP per model |
| `classification_report_<MODEL>.txt` | Per-class precision / recall / F1 |
| `accuracy_comparison.png` | Fig. 5 equivalent (best model highlighted) |
| `confusion_matrix_best.png` | Fig. 4 equivalent |
| `rf_feature_importance.png`, `ann_training_curve.png` | Supporting figures |
| `eda/` | Class distribution, histograms, correlation heatmap (Fig. 3) |
| `metrics_cross_validation.csv` | Repeated stratified 5-fold CV mean ± std (see below) |
| `best_model.joblib` (or `.keras`), `preprocessor.joblib` | Fitted artefacts |

## Reproducing the report's numbers

Be aware of three things before comparing against Table 5 (Random Forest 95.0 %):

1. **One small test split.** 61 test patients means one patient is about 1.6 percentage points, and 95 % is 58 of 61 correct. Results move noticeably with the seed (`--seed`). The report's own limitations section says the same.
2. **Unreported settings.** The report does not state the split seed, whether the split was stratified, or the hyper-parameters. This code uses documented defaults (see `heart_ml/models.py`) and `--seed 42`, so your single-split numbers may differ from Table 5. The code does not tune a seed to hit a target number.
3. **Dataset copy.** The raw UCI file and the Kaggle `heart.csv` encode the same records differently, so exact numbers can differ between them.

For a figure that does not depend on one lucky or unlucky split, use `metrics_cross_validation.csv` (repeated stratified 5-fold CV, preprocessing refit inside every fold). Cite that alongside the single-split table if the two disagree.

Useful flags: `--seed`, `--test-size`, `--skip-ann`, `--skip-cv`, `--cv-repeats`, `--data`, `--output-dir`.

## Project layout

```
heart-disease-ml/
├── run_pipeline.py          # CLI entry point
├── heart_ml/                # config, data, preprocess, models, evaluate, plots
├── scripts/
│   ├── download_data.py     # UCI download -> data/heart.csv
│   └── make_synthetic_data.py   # random stand-in data for smoke tests ONLY
├── notebooks/heart_disease_prediction.ipynb   # step-by-step version of the pipeline
├── tests/                   # pytest suite (runs on synthetic data)
├── data/                    # put heart.csv here
└── results/                 # generated outputs
```

## Tests

```bash
pytest -q
```

The tests use synthetic data with the real schema. They check that the code runs and behaves correctly (loader, no-leakage preprocessing, split size, all models, metrics); they do not and cannot confirm the report's accuracies.

## Scope and disclaimer

Binary prediction (disease present or absent) on 303 records from a single source. This is a research and teaching project, a decision-support illustration and not a medical device; it has not been validated on hospital data or by clinicians.
