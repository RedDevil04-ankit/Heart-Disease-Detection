#!/usr/bin/env python
"""Generate a SYNTHETIC stand-in dataset with the same schema as heart.csv.

For smoke tests and CI only. The records are randomly generated, so any
accuracy obtained on them says nothing about heart disease or about the
report's results. Use the real UCI data for the actual study.

Usage:  python scripts/make_synthetic_data.py [--out data/synthetic_heart.csv] [--n 303] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart_ml.config import COLUMNS, PROJECT_ROOT  # noqa: E402


def make_synthetic(n: int = 303, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(54, 9, n).clip(29, 77).round()
    sex = rng.binomial(1, 0.68, n)
    cp = rng.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])
    trestbps = rng.normal(131, 17, n).clip(94, 200).round()
    chol = rng.normal(246, 51, n).clip(126, 564).round()
    fbs = rng.binomial(1, 0.15, n)
    restecg = rng.choice([0, 1, 2], n, p=[0.50, 0.49, 0.01])
    thalach = (200 - 0.6 * age + rng.normal(0, 17, n)).clip(71, 202).round()
    exang = rng.binomial(1, 0.33, n)
    oldpeak = rng.gamma(1.2, 0.9, n).clip(0, 6.2).round(1)
    slope = rng.choice([0, 1, 2], n, p=[0.07, 0.46, 0.47])
    ca = rng.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07])
    thal = rng.choice([1, 2, 3], n, p=[0.06, 0.55, 0.39])

    logit = (
        -1.2
        + 0.9 * (cp == 0)
        - 0.03 * (thalach - 150)
        + 0.9 * exang
        + 0.6 * oldpeak
        + 0.7 * ca
        + 0.8 * (thal == 3)
        + 0.4 * sex
        + 0.015 * (age - 54)
        + rng.normal(0, 0.8, n)
    )
    target = (1 / (1 + np.exp(-logit)) > rng.uniform(0, 1, n)).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
        "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal, "target": target,
    })
    return df[COLUMNS]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(PROJECT_ROOT / "data" / "synthetic_heart.csv"))
    p.add_argument("--n", type=int, default=303)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    df = make_synthetic(args.n, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote SYNTHETIC data ({len(df)} rows, positive rate {df['target'].mean():.0%}) to {args.out}")
    print("Do not report results from this file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
