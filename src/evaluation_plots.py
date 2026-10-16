"""Simple plots and regression error summaries. All inputs share row order."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .evaluate import _check_arrays, confusion_matrix, mae, rmse


def _save(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(y_true, y_pred, output_path, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            color = "white" if cm[row, col] > cm.max() / 2 else "black"
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color=color)
    labels = ["Standard (0)", "Premium (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    return _save(fig, output_path)


def plot_actual_vs_predicted(y_true, y_pred, output_path, title="Price predictions", unit="AZN"):
    y_true, y_pred = _check_arrays(y_true, y_pred)
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    if low == high:
        low, high = low - 1, high + 1
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.4, s=12)
    ax.plot([low, high], [low, high], "r--", label="Perfect prediction")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(f"Actual price ({unit})")
    ax.set_ylabel(f"Predicted price ({unit})")
    ax.set_title(title)
    ax.legend()
    return _save(fig, output_path)


def summarize_errors_by_group(y_true, y_pred, groups):
    """Return count, MAE, RMSE and bias (predicted minus actual) per group.

Use original location names. Replace missing names with 'Unknown' first.
R-squared is omitted because a group can contain only one observation.
"""
    y_true, y_pred = _check_arrays(y_true, y_pred)
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(y_true):
        raise ValueError("groups must have one entry per observation.")
    if any(value is None or str(value).strip().lower() in ("", "nan", "<na>") for value in groups):
        raise ValueError("Replace missing group names with 'Unknown' first.")
    groups = groups.astype(str)
    results = []
    for name in np.unique(groups):
        mask = groups == name
        results.append({
            "group": str(name), "count": int(mask.sum()),
            "mae": mae(y_true[mask], y_pred[mask]),
            "rmse": rmse(y_true[mask], y_pred[mask]),
            "bias": float(np.mean(y_pred[mask] - y_true[mask])),
        })
    return results


def summarize_price_errors(y_true, y_pred, expensive_threshold):
    """Supply a threshold chosen from TRAIN prices before test evaluation."""
    y_true, y_pred = _check_arrays(y_true, y_pred)
    if not isinstance(expensive_threshold, (int, float, np.integer, np.floating)):
        raise ValueError("expensive_threshold must be a finite number.")
    if not np.isfinite(expensive_threshold):
        raise ValueError("expensive_threshold must be a finite number.")
    groups = np.where(y_true > expensive_threshold, "Expensive", "Other")
    return summarize_errors_by_group(y_true, y_pred, groups)


def plot_feature_values(feature_names, values, output_path, title="Feature importance", xlabel="Importance", top_n=10):
    """Plot tree importances or LINEAR SVM weights, ranked by absolute value.

For SVM use training-standardized features. RFF weights refer to transformed
features and must not be labelled with original feature names.
"""
    values, _ = _check_arrays(values, values)
    names = np.asarray(feature_names)
    if names.ndim != 1 or len(names) != len(values):
        raise ValueError("feature_names and values must have the same length.")
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    order = np.argsort(np.abs(values))[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.arange(len(order)), values[order])
    ax.set_yticks(np.arange(len(order)), names[order].astype(str))
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return _save(fig, output_path)
