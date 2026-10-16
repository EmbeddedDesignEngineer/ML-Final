"""Compare tree settings on prepared train/validation splits.

The shared runner handles preprocessing and test evaluation.
sklearn provides reference models and validation scores.
"""
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import f1_score, root_mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.decision_tree import DecisionTree


def compare_tree_configs(X_train, y_train, X_val, y_val, *, task,
                         depths=(2, 4, 8, 12), leaf_sizes=(1, 5, 20), seed=42):
    """Return validation results and the best setting per implementation.

    Maximize F1 for class 1 or minimize RMSE in the supplied target units.
    Fit on train only. Ties keep the first configuration.
    """
    if task not in ("classification", "regression"):
        raise ValueError("task must be classification or regression")
    X_train, X_val = np.asarray(X_train, dtype=float), np.asarray(X_val, dtype=float)
    y_train, y_val = np.asarray(y_train), np.asarray(y_val)
    for X, y in ((X_train, y_train), (X_val, y_val)):
        if X.ndim != 2 or not all(X.shape) or not np.isfinite(X).all():
            raise ValueError("Each feature split must be a non-empty finite 2-D array")
        if y.ndim != 1 or len(y) != len(X) or y.dtype.kind not in "biuf" or not np.isfinite(y).all():
            raise ValueError("Each target split must be a finite numeric 1-D array matching X")
        if task == "classification" and not np.isin(y, [0, 1]).all():
            raise ValueError("Price-tier labels must be 0 or 1")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Feature counts must match across splits")
    depths, leaf_sizes = tuple(depths), tuple(leaf_sizes)
    if not depths or not leaf_sizes:
        raise ValueError("Search grids must not be empty")
    if any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 1 for d in depths):
        raise ValueError("Sweep depths must be positive integers for sklearn compatibility")
    if any(isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or s < 1 for s in leaf_sizes):
        raise ValueError("Leaf sizes must be positive integers")
    classification = task == "classification"
    criteria = ("gini", "entropy") if classification else ("mse",)
    metric = "f1" if classification else "rmse"
    score = (lambda actual, pred: f1_score(actual, pred, zero_division=0)) if classification else root_mean_squared_error
    rows, best = [], {}
    for criterion, depth, leaf in product(criteria, depths, leaf_sizes):
        params = dict(max_depth=int(depth), min_samples_leaf=int(leaf), random_state=seed)
        reference = (DecisionTreeClassifier(criterion=criterion, **params) if classification
                     else DecisionTreeRegressor(criterion="squared_error", **params))
        models = {"custom": DecisionTree(task=task, criterion=criterion, **params), "sklearn": reference}
        for name, model in models.items():
            start = perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = perf_counter() - start
            train_score = float(score(y_train, model.predict(X_train)))
            start = perf_counter()
            predicted = model.predict(X_val)
            predict_seconds = perf_counter() - start
            val_score = float(score(y_val, predicted))
            row = dict(implementation=name, task=task, criterion=criterion,
                       max_depth=int(depth), min_samples_leaf=int(leaf), metric=metric,
                       train_score=train_score, validation_score=val_score,
                       fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                       fitted_depth=model.get_depth(), n_leaves=model.get_n_leaves())
            rows.append(row)
            old = best.get(name)
            improves = old is None or (val_score > old["validation_score"] if classification
                                      else val_score < old["validation_score"])
            if improves:
                best[name] = row.copy()
    return rows, best


def save_validation_curves(rows, path):
    """Plot train and validation scores by depth for each setting."""
    import matplotlib.pyplot as plt

    if not rows or len({(r["task"], r["metric"]) for r in rows}) != 1:
        raise ValueError("Provide non-empty rows for exactly one task and metric")
    groups = sorted({(r["implementation"], r["criterion"], r["min_samples_leaf"]) for r in rows})
    fig, axes = plt.subplots(len(groups), 1, figsize=(7, 2.6 * len(groups)), squeeze=False)
    for ax, (implementation, criterion, leaf) in zip(axes[:, 0], groups):
        selected = sorted((r for r in rows if (r["implementation"], r["criterion"], r["min_samples_leaf"])
                           == (implementation, criterion, leaf)), key=lambda r: r["max_depth"])
        for split in ("train", "validation"):
            ax.plot([r["max_depth"] for r in selected], [r[f"{split}_score"] for r in selected],
                    marker="o", label=split)
        ax.set(title=f"{implementation}: {criterion}, min_samples_leaf={leaf}",
               xlabel="Maximum depth", ylabel=rows[0]["metric"].upper())
        ax.legend()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
        fig.savefig(destination)
    finally:
        plt.close(fig)
