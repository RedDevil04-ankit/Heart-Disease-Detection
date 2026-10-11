"""EDA module and comparison plots (report sections 3.4, 5.1.3, Figs. 3-5).

Plot style: one hue plus neutral grey (the best model is the only highlighted
bar), light recessive grid, direct value labels, no dual axes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; works on servers and in CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    FEATURES,
    GRID,
    HIGHLIGHT,
    INK,
    INK_SECONDARY,
    MODEL_LABELS,
    NEUTRAL_BAR,
    SURFACE,
    TARGET,
)


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 10,
    })


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- EDA -------
def plot_eda(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    _style()
    out_dir = Path(out_dir)
    paths = []

    # Class distribution (one series -> single hue, no legend needed)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    counts = df[TARGET].value_counts().sort_index()
    bars = ax.bar(["No disease (0)", "Disease (1)"], counts.values, color=HIGHLIGHT, width=0.5)
    ax.bar_label(bars, padding=3, color=INK)
    ax.set_ylabel("Patients")
    ax.set_title("Class distribution")
    ax.grid(axis="x", visible=False)
    paths.append(_save(fig, out_dir / "eda_class_distribution.png"))

    # Histograms of key numeric attributes
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, col, label in zip(
        axes, ["age", "chol", "thalach"], ["Age (years)", "Cholesterol (mg/dl)", "Max heart rate (bpm)"]
    ):
        ax.hist(df[col].dropna(), bins=20, color=HIGHLIGHT, edgecolor=SURFACE, linewidth=1.2)
        ax.set_xlabel(label)
        ax.set_ylabel("Patients")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Distribution of age, cholesterol and maximum heart rate", fontweight="bold", x=0.02, ha="left")
    paths.append(_save(fig, out_dir / "eda_histograms.png"))

    # Correlation heatmap (diverging, grey midpoint)
    fig, ax = plt.subplots(figsize=(8, 6.6))
    corr = df[FEATURES + [TARGET]].corr()
    sns.heatmap(
        corr, ax=ax, cmap=sns.diverging_palette(250, 30, l=55, s=80, center="light", as_cmap=True),
        vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=1, linecolor=SURFACE, cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Correlation heatmap")
    ax.grid(False)
    paths.append(_save(fig, out_dir / "eda_correlation_heatmap.png"))
    return paths


# ------------------------------------------------------- Model comparison ---
def plot_accuracy_comparison(table: pd.DataFrame, best_key: str, path: Path) -> Path:
    """Bar chart of test accuracy (report Fig. 5); the best model is highlighted."""
    _style()
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    labels = table["key"].tolist()
    vals = (table["accuracy"] * 100).tolist()
    colors = [HIGHLIGHT if k == best_key else NEUTRAL_BAR for k in labels]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in vals], padding=3, color=INK)
    ax.set_ylim(0, 105)  # bars always start at zero
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy of the classifiers on the held-out test set")
    ax.grid(axis="x", visible=False)
    ax.text(0.0, -0.16, f"Highlighted: {MODEL_LABELS.get(best_key, best_key)} (highest accuracy).",
            transform=ax.transAxes, fontsize=8, color=INK_SECONDARY)
    return _save(fig, Path(path))


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path) -> Path:
    """Confusion matrix for one model (report Fig. 4)."""
    _style()
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=sns.light_palette(HIGHLIGHT, as_cmap=True), cbar=False,
        xticklabels=["No disease", "Disease"], yticklabels=["No disease", "Disease"],
        linewidths=2, linecolor=SURFACE, annot_kws={"size": 14}, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.grid(False)
    return _save(fig, Path(path))


def plot_feature_importance(names: list[str], importances: np.ndarray, path: Path, top: int = 12) -> Path:
    """Horizontal bar chart of Random Forest feature importances."""
    _style()
    order = np.argsort(importances)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh([names[i] for i in order], importances[order], color=HIGHLIGHT, height=0.6)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest feature importance")
    ax.grid(axis="y", visible=False)
    return _save(fig, Path(path))


def plot_training_curve(history: dict, path: Path) -> Path:
    """Training vs validation loss of the ANN (report section 5.1.5)."""
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(history["loss"], color=HIGHLIGHT, linewidth=2, label="Training loss")
    ax.plot(history["val_loss"], color="#eb6834", linewidth=2, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.set_title("ANN training curve")
    ax.legend(frameon=False)
    return _save(fig, Path(path))
