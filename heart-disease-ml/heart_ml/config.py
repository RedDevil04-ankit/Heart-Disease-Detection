"""Central configuration: column roles, seeds, paths and plot colours.

Everything that the report fixes (which columns are categorical, which are
scaled, 8 models, held-out test split with a fixed seed) lives here so the
rest of the code never hard-codes it.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "heart.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"

# --- Dataset schema (report Table 2) ---------------------------------------
TARGET = "target"  # 1 = heart disease present, 0 = absent
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
COLUMNS = FEATURES + [TARGET]

# Report section 5.1.2: dummy-encode cp, restecg, slope, thal;
# standardise age, trestbps, chol, thalach, oldpeak.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
PASSTHROUGH_FEATURES = ["sex", "fbs", "exang", "ca"]  # already 0/1 or small counts

# --- Experiment settings ----------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2  # 20 % of 303 records -> 61 test patients

# Raw UCI file (no header, "?" for missing values, target "num" in 0..4).
UCI_CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# --- Model display names (report Table 5 order) -----------------------------
MODEL_ORDER = ["LR", "NB", "SVM", "KNN", "DT", "RF", "XGB", "ANN"]
MODEL_LABELS = {
    "LR": "Logistic Regression",
    "NB": "Naive Bayes",
    "SVM": "Linear SVM",
    "KNN": "K-Nearest Neighbours",
    "DT": "Decision Tree",
    "RF": "Random Forest",
    "XGB": "XGBoost",
    "ANN": "ANN (1 hidden layer)",
}

# --- Plot palette (single hue + neutral; best model highlighted) ------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL_BAR = "#b9b8b2"
HIGHLIGHT = "#2a78d6"
