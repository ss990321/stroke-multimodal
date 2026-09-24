"""Scoring and operating points."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRICS = ["accuracy", "precision", "recall", "f1", "auc", "brier"]


def youden_threshold(y_true, probability) -> float:
    """The point maximising sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def sensitivity_threshold(y_true, probability, target: float = 0.90) -> float:
    """The highest threshold still reaching the target sensitivity."""
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    reached = np.flatnonzero(tpr >= target)
    return float(thresholds[reached[0]]) if reached.size else float(thresholds[-1])


def false_positive_rate(y_true, probability, threshold) -> float:
    predicted = (np.asarray(probability) >= threshold).astype(int)
    negatives = np.asarray(y_true) == 0
    return float(predicted[negatives].mean()) if negatives.any() else np.nan


def score(y_true, probability, threshold) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "auc": roc_auc_score(y_true, probability),
        "brier": float(np.mean((probability - y_true) ** 2)),
    }


def score_with_validation_threshold(y_val, p_val, y_test, p_test,
                                    sensitivity_target: float = 0.90) -> dict:
    """Score a test split at the operating point chosen on its validation split."""
    threshold = youden_threshold(y_val, p_val)
    out = {"threshold": threshold, **score(y_test, p_test, threshold)}
    out["fpr_at_target_sensitivity"] = false_positive_rate(
        y_test, p_test, sensitivity_threshold(y_val, p_val, sensitivity_target))
    return out


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float]:
    """Mean and the half-width of its confidence interval across folds."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float(v.mean()) if v.size else np.nan), np.nan
    low, _ = stats.t.interval(confidence, len(v) - 1, loc=v.mean(),
                              scale=stats.sem(v))
    return float(v.mean()), float(v.mean() - low)


def summarise(per_fold: pd.DataFrame, by: str = "model",
              metrics=METRICS) -> pd.DataFrame:
    """Fold-wise results collapsed to a mean and interval per model."""
    rows = []
    for name, group in per_fold.groupby(by, sort=False):
        row = {by: name, "n_folds": len(group)}
        for metric in metrics:
            if metric not in group:
                continue
            mean, half_width = mean_ci(group[metric].to_numpy())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = half_width
        rows.append(row)
    return pd.DataFrame(rows)


def format_table(summary: pd.DataFrame, by: str = "model") -> pd.DataFrame:
    """A summary rendered as `mean (low-high)`, percentages where conventional."""
    as_percent = {"accuracy", "precision", "recall", "f1",
                  "fpr_at_target_sensitivity"}
    present = [c[:-len("_mean")] for c in summary.columns if c.endswith("_mean")]
    rows = []
    for _, r in summary.iterrows():
        row = {by: r[by]}
        for metric in present:
            if f"{metric}_ci95" not in r:
                continue
            mean, half = r[f"{metric}_mean"], r[f"{metric}_ci95"]
            if metric in as_percent:
                row[metric] = (f"{100 * mean:.1f} "
                               f"({100 * (mean - half):.1f}-{100 * (mean + half):.1f})")
            else:
                row[metric] = f"{mean:.3f} ({mean - half:.3f}-{mean + half:.3f})"
        rows.append(row)
    return pd.DataFrame(rows)
