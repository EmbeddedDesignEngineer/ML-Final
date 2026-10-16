"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH where reasonable).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers or for the baseline models you compare against.
"""

from __future__ import annotations

import numpy as np


def _check_arrays(y_true, y_pred):
    """Validate inputs to avoid accidental broadcasting or invalid numbers."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for values in (y_true, y_pred):
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Inputs must be non-empty one-dimensional arrays.")
        if values.dtype.kind not in "biuf":
            raise ValueError("Inputs must contain real numbers.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Inputs must not contain NaN or infinity.")
    if y_true.size != y_pred.size:
        raise ValueError("Inputs must have the same length.")
    return y_true.astype(float), y_pred.astype(float)


def _check_labels(labels):
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("Classification labels must be 0 or 1.")


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _check_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _check_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = _check_arrays(y_true, y_pred)
    if len(y_true) < 2:
        raise ValueError("R-squared needs at least two observations.")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # Same constant-target policy as sklearn's default force_finite=True.
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]]."""
    y_true, y_pred = _check_arrays(y_true, y_pred)
    _check_labels(y_true)
    _check_labels(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class."""
    cm = confusion_matrix(y_true, y_pred)
    fp, fn, tp = int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    # Undefined ratios return zero, like sklearn with zero_division=0.
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
    return precision, recall, f1


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney) computation is fine.
    """
    y_true, scores = _check_arrays(y_true, scores)
    _check_labels(y_true)
    n_positive = int(np.sum(y_true == 1))
    n_negative = len(y_true) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("ROC-AUC needs both classes in y_true.")
    # Larger scores must mean stronger evidence for class 1 (premium).
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Positions start..end-1 have one-based ranks start+1..end.
        average_rank = (start + 1 + end) / 2
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = np.sum(ranks[y_true == 1])
    baseline = n_positive * (n_positive + 1) / 2
    return float((positive_rank_sum - baseline) / (n_positive * n_negative))


# --- Comparison helper ------------------------------------------------------
def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.
    """
    if not name_to_metrics:
        print("No model results to compare.")
        return
    metric_names = []
    for metrics in name_to_metrics.values():
        for metric in metrics:
            if metric not in metric_names:
                metric_names.append(metric)
    rows = [["Model"] + [str(metric) for metric in metric_names]]
    for name, metrics in name_to_metrics.items():
        row = [str(name)]
        for metric in metric_names:
            value = metrics.get(metric)
            row.append("-" if value is None else f"{float(value):.4f}")
        rows.append(row)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for index, row in enumerate(rows):
        print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))
        if index == 0:
            print("-+-".join("-" * width for width in widths))
