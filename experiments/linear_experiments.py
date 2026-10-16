from time import perf_counter

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.linear import RidgeRegression
from src.evaluate import rmse, mae, r2


class ScaledRidge:
    def __init__(self, alpha):
        self.scale = StandardScaler()
        self.ridge = RidgeRegression(alpha=alpha)

    def fit(self, X, y):
        self.ridge.fit(self.scale.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.ridge.predict(self.scale.transform(X))


def create_ridge_model(config):
    if config["implementation"] == "custom":
        return ScaledRidge(config["alpha"])
    estimator = Ridge(alpha=config["alpha"], solver="cholesky")
    return Pipeline([("scale", StandardScaler()), ("ridge", estimator)])


def select_ridge_configs(splits, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    if not alphas or any(isinstance(alpha, bool) or not isinstance(alpha, (int, float))
                         or not np.isfinite(alpha) or alpha < 0 for alpha in alphas):
        raise ValueError("alphas must contain finite non-negative numbers.")
    rows, configs = [], {}
    for implementation in ("custom", "sklearn"):
        best = None
        for alpha in alphas:
            config = {"kind": "ridge", "task": "regression", "implementation": implementation, "alpha": float(alpha)}
            model = create_ridge_model(config)
            start = perf_counter()
            model.fit(splits["X_tr"], splits["y_tr"])
            fit_seconds = perf_counter() - start
            predicted = model.predict(splits["X_val"])
            row = {"implementation": implementation, "alpha": float(alpha),
                   "validation_rmse": rmse(splits["y_val"], predicted),
                   "validation_mae": mae(splits["y_val"], predicted),
                   "validation_r2": r2(splits["y_val"], predicted), "fit_seconds": fit_seconds}
            rows.append(row)
            if best is None or row["validation_rmse"] < best["validation_rmse"]:
                best = row
        configs[f"{implementation}_ridge"] = {
            "kind": "ridge", "task": "regression", "implementation": implementation,
            "alpha": best["alpha"], "validation_score": best["validation_rmse"], "selection_metric": "rmse"}
    return configs, rows


def save_ridge_curve(rows, destination):
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(7, 4))
    for implementation, marker in (("custom", "o"), ("sklearn", "x")):
        selected = [row for row in rows if row["implementation"] == implementation]
        ax.plot([row["alpha"] for row in selected], [row["validation_rmse"] for row in selected],
                marker=marker, label=implementation)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set(xlabel="Ridge alpha", ylabel="Validation RMSE (AZN)")
    ax.legend()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
