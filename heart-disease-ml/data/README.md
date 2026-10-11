# Data

Place the dataset here as `heart.csv` (not committed to git).

**Option A: UCI repository.** `python scripts/download_data.py` downloads `processed.cleveland.data` (303 rows, 6 missing `ca` / `thal` values, label 0 to 4) and writes a binarised `heart.csv`. Missing values are imputed inside the pipeline, fitted on training data only.

**Option B: Kaggle `heart.csv`.** A 303-row, 14-column CSV with header `age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target` and no missing values. This is the copy whose class counts (165 positive / 138 negative) the report quotes.

Both load through `heart_ml.data.load_dataset`. `target = 1` means heart disease present.

| Column | Meaning |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (4 categories) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1/0) |
| restecg | Resting ECG result |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1/0) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Major vessels (0-3) coloured by fluoroscopy |
| thal | Thalassemia |
| target | Heart disease present (1) or absent (0) |

`scripts/make_synthetic_data.py` creates random data with this schema for smoke tests. Never report results from it.
