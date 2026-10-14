"""SVM experiments on splits exported by the team's preprocessing pipeline.

Run from the repository root:
    python -m experiments.svm_experiments --data data/svm_prepared.npz

No cleaning, encoding, splitting, scaling, or synthetic fallback occurs here.
See SVM_REPORT_NOTES.md for the NPZ contract and current integration blockers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.svm import SVC

from src.svm import PegasosSVM


SPLITS = ("train", "val", "test")
FAMILIES = ("custom_linear", "custom_rff", "sklearn_linear", "sklearn_rbf")
DATA_KEYS = tuple(f"{prefix}_{split}" for split in SPLITS for prefix in ("X", "y"))
DATA_KEYS += tuple(f"{split}_ids" for split in SPLITS) + ("feature_names",)


def load_prepared_data(path):
    """Read numeric/string arrays without allowing pickle or fitting transforms."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared data not found: {path}. Ask Davud/Bailar to export the "
            "train-only-preprocessed splits described in experiments/SVM_REPORT_NOTES.md. "
            "No real experiment was run."
        )
    with np.load(path, allow_pickle=False) as archive:
        missing = set(DATA_KEYS + ("metadata_json",)) - set(archive.files)
        if missing:
            raise ValueError(f"Prepared NPZ is missing keys: {sorted(missing)}")
        data = {name: archive[name] for name in DATA_KEYS}
        metadata = json.loads(str(archive["metadata_json"].item()))
    return data, metadata


def validate_prepared_data(data, metadata):
    """Check observable properties; provenance declarations are not a leakage proof."""
    missing = set(DATA_KEYS) - set(data)
    if missing:
        raise ValueError(f"Missing prepared arrays: {sorted(missing)}")
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must encode a JSON object.")
    for name in ("dataset_name", "preprocessing_source", "split_strategy"):
        if not isinstance(metadata.get(name), str) or not metadata[name].strip():
            raise ValueError(f"Metadata must describe {name}.")
    for name in ("scaling_fit_split", "encoding_fit_split", "tier_threshold_fit_split"):
        if metadata.get(name) != "train":
            raise ValueError(f"Metadata {name} must be 'train'.")
    threshold = metadata.get("tier_threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (float, int))
            or not np.isfinite(threshold)):
        raise ValueError("Metadata tier_threshold must be a finite training-derived number.")

    feature_names = np.asarray(data["feature_names"])
    if feature_names.ndim != 1 or feature_names.dtype.kind != "U" or not feature_names.size:
        raise ValueError("feature_names must be a non-empty 1-D array of Unicode strings.")
    names = [name.strip().lower() for name in feature_names]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("feature_names must be non-empty and unique.")
    # Known target columns from the brief; other derived/renamed leaks need human review.
    forbidden = {"price", "log_price", "unit_price", "total_price", "price_tier", "target"}
    if forbidden.intersection(names):
        raise ValueError(f"Target leakage columns in X: {sorted(forbidden.intersection(names))}")

    id_sets, sizes = {}, {}
    for split in SPLITS:
        X, y = np.asarray(data[f"X_{split}"]), np.asarray(data[f"y_{split}"])
        if X.ndim != 2 or not X.shape[0] or X.shape[1] != len(names):
            raise ValueError(f"X_{split} must be non-empty, 2-D, with {len(names)} features.")
        if X.dtype.kind not in "fiu" or not np.all(np.isfinite(X)):
            raise ValueError(f"X_{split} must contain finite numeric features.")
        if y.ndim != 1 or len(y) != len(X) or y.dtype.kind not in "biuf":
            raise ValueError(f"y_{split} must be a numeric 1-D label per row.")
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError(f"y_{split} must contain both labels 0 and 1 (needed for ROC-AUC).")
        ids = np.asarray(data[f"{split}_ids"])
        if ids.ndim != 1 or len(ids) != len(X) or ids.dtype.kind not in "Uiu":
            raise ValueError(f"{split}_ids must provide one string/integer listing ID per row.")
        id_sets[split] = set(ids.astype(str))
        if len(id_sets[split]) != len(ids) or "" in id_sets[split]:
            raise ValueError(f"{split}_ids contains duplicate or empty listing IDs.")
        sizes[split] = len(X)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if id_sets[left] & id_sets[right]:
            raise ValueError(f"Listing IDs overlap between {left} and {right}.")

    # Check only TRAIN moments. Validation/test must use its statistics, and
    # therefore should NOT be required to have mean zero and variance one.
    train = np.asarray(data["X_train"], dtype=float)
    means, stds = train.mean(axis=0), train.std(axis=0)
    if not np.allclose(means, 0.0, atol=1e-5, rtol=0):
        raise ValueError("X_train is not centred; provide train-standardized features.")
    if not np.all(np.isclose(stds, 1, atol=1e-5) | np.isclose(stds, 0, atol=1e-5)):
        raise ValueError("X_train must have population standard deviation 1 (or 0 if constant).")
    return {
        "split_sizes": sizes, "n_features": len(names), "listing_ids_disjoint": True,
        "max_abs_training_mean": float(np.abs(means).max()),
        "constant_features": int(np.count_nonzero(np.isclose(stds, 0, atol=1e-5))),
        "limitation": "Fit scope, threshold derivation and renamed target leakage require "
                      "review of the declared preprocessing source; arrays alone cannot prove them.",
    }


def build_configurations(lambdas=(1e-4, 1e-3, 1e-2), iterations=(5000, 20000),
                         gammas=(0.01, 0.1), components=(128, 512),
                         constant_rate=0.01, seed=42):
    """Default: 12 linear + 12 RFF + 3 linear SVC + 6 RBF SVC fits."""
    grids = {"lambdas": lambdas, "iterations": iterations,
             "gammas": gammas, "components": components}
    for name, values in grids.items():
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be non-empty and have no duplicate values.")
        for value in values:
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must contain finite positive numbers.")
            if name in ("iterations", "components") and not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must contain integers.")
    if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a non-negative integer.")
    if (isinstance(constant_rate, bool) or not isinstance(constant_rate, (int, float))
            or not np.isfinite(constant_rate) or constant_rate <= 0
            or constant_rate * max(lambdas) > 1):
        raise ValueError("constant_rate must be positive, finite, and constant_rate * lambda <= 1.")

    configurations = []

    def add(family, lambda_, n_iters=None, learning_rate=None, gamma=None, n_components=None):
        configurations.append({
            "config_id": f"{family}_{len(configurations):03d}", "family": family,
            "lambda_": float(lambda_), "n_iters": n_iters, "learning_rate": learning_rate,
            "eta0": constant_rate if learning_rate == "constant" else None,
            "gamma": gamma, "n_components": n_components, "random_state": int(seed),
        })

    for lam, steps, schedule in product(lambdas, iterations, ("pegasos", "constant")):
        add("custom_linear", lam, int(steps), schedule)
    for lam, gamma, size in product(lambdas, gammas, components):
        add("custom_rff", lam, int(max(iterations)), "pegasos", float(gamma), int(size))
    for lam in lambdas:
        add("sklearn_linear", lam)
    for lam, gamma in product(lambdas, gammas):
        add("sklearn_rbf", lam, gamma=float(gamma))
    return configurations


def make_model(config, n_train):
    if config["family"].startswith("custom_"):
        return PegasosSVM(
            lambda_=config["lambda_"], n_iters=config["n_iters"],
            random_state=config["random_state"], learning_rate=config["learning_rate"],
            eta0=config["eta0"] if config["eta0"] is not None else 0.01,
            n_components=config["n_components"],
            gamma=config["gamma"] if config["gamma"] is not None else 1.0,
        )
    # Multiply our mean-hinge objective by 1/lambda to obtain SVC's
    # .5*||w||^2 + C*sum(hinge). Thus C=1/(n_train*lambda), not 1/lambda.
    return SVC(
        C=1.0 / (n_train * config["lambda_"]),
        kernel="linear" if config["family"] == "sklearn_linear" else "rbf",
        gamma=config["gamma"] if config["gamma"] is not None else "scale",
        random_state=config["random_state"], cache_size=256,
    )


def evaluate_model(model, X, y):
    """Times exclude metric calculation; decision score timing is separate."""
    start = perf_counter()
    predictions = model.predict(X)
    prediction_seconds = perf_counter() - start
    start = perf_counter()
    scores = model.decision_function(X)
    decision_seconds = perf_counter() - start
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, average="binary", pos_label=1, zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, predictions)), "precision": float(precision),
        "recall": float(recall), "f1": float(f1), "roc_auc": float(roc_auc_score(y, scores)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "prediction_seconds": prediction_seconds, "decision_seconds": decision_seconds,
    }


def training_diagnostics(model):
    """Do not confuse RKHS support coefficients with explicit RFF weights."""
    diagnostics = dict.fromkeys(("final_objective", "weight_norm", "margin_width",
                                "n_margin_active", "margin_active_fraction", "n_support_vectors"))
    if isinstance(model, PegasosSVM):
        diagnostics.update(
            final_objective=model.loss_history_[-1], weight_norm=model.weight_norm_,
            margin_width=model.margin_width_ if np.isfinite(model.margin_width_) else None,
            n_margin_active=int(model.n_margin_active_),
            margin_active_fraction=model.margin_active_fraction_,
        )
    else:
        diagnostics["n_support_vectors"] = int(model.n_support_.sum())
    return diagnostics


def selection_key(row):
    """Predeclared rule: validation F1, then ROC-AUC; exact ties keep grid order."""
    return row["f1"], row["roc_auc"]


def select_best(validation_rows):
    if not validation_rows or any(row["split"] != "validation" for row in validation_rows):
        raise ValueError("Selection requires validation rows only.")
    best = {}
    for row in validation_rows:
        family = row["family"]
        if family not in best or selection_key(row) > selection_key(best[family]):
            best[family] = row
    return best


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_metrics(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figures(rows, histories, selected, output, dataset_name):
    """Use validation only; no test plot is used to choose a configuration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validation = [row for row in rows if row["split"] == "validation"]
    figures = []

    def save(fig, filename):
        fig.suptitle(dataset_name)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
        figures.append(filename)

    # A controlled comparison at the selected linear lambda shows schedule
    # and iteration effects without mixing objectives with different lambdas.
    fig, ax = plt.subplots(figsize=(7, 4))
    selected_lambda = selected["custom_linear"]["lambda_"]
    for row in validation:
        if row["family"] == "custom_linear" and row["lambda_"] == selected_lambda:
            history = histories[row["config_id"]]
            label = f'{row["learning_rate"]}, T={row["n_iters"]}'
            ax.plot(history["iterations"], history["objective"], label=label)
    ax.set(xlabel="Stochastic updates (iterations)", ylabel="Training objective J(w, b)",
           title=f"Linear convergence, lambda={selected_lambda:g}")
    ax.set_yscale("symlog", linthresh=1)
    ax.legend(fontsize="small")
    save(fig, "objective_vs_iterations.png")

    linear = [row for row in validation if row["family"] == "custom_linear"]
    for field, ylabel, filename in (
        ("f1", "Validation F1 (positive class = premium)", "lambda_vs_validation_f1.png"),
        ("margin_width", "Margin width (standardized feature space)", "lambda_vs_margin.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        groups = sorted({(row["learning_rate"], row["n_iters"]) for row in linear})
        for schedule, steps in groups:
            group = sorted((r for r in linear if (r["learning_rate"], r["n_iters"]) == (schedule, steps)),
                           key=lambda r: r["lambda_"])
            values = [np.nan if r[field] is None else r[field] for r in group]
            ax.plot([r["lambda_"] for r in group], values, "o-", label=f"{schedule}, T={steps}")
        ax.set(xscale="log", xlabel="Regularization lambda", ylabel=ylabel, title="Linear SVM")
        if field == "f1":
            ax.set_ylim(0, 1.02)
        ax.legend(fontsize="small")
        save(fig, filename)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(FAMILIES)), [selected[family]["f1"] for family in FAMILIES])
    ax.set_xticks(range(len(FAMILIES)), ["Pegasos linear", "Pegasos RFF", "SVC linear", "SVC RBF"])
    ax.set(ylabel="Validation F1", ylim=(0, 1.02), title="Best validation configuration per family")
    save(fig, "linear_vs_nonlinear.png")

    rff = [row for row in validation if row["family"] == "custom_rff"]
    lambdas = sorted({row["lambda_"] for row in rff})
    fig, axes = plt.subplots(1, len(lambdas), figsize=(5 * len(lambdas), 4), squeeze=False)
    for ax, lam in zip(axes[0], lambdas):
        for size in sorted({row["n_components"] for row in rff}):
            group = sorted((r for r in rff if r["lambda_"] == lam and r["n_components"] == size),
                           key=lambda r: r["gamma"])
            ax.plot([r["gamma"] for r in group], [r["f1"] for r in group], "o-", label=f"D={size}")
        ax.set(xscale="log", xlabel="RBF gamma (inverse squared standardized distance)",
               ylabel="Validation F1", ylim=(0, 1.02), title=f"RFF lambda={lam:g}")
        ax.legend()
    save(fig, "rff_parameter_effects.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    row = selected["custom_rff"]
    history = histories[row["config_id"]]
    ax.plot(history["iterations"], history["objective"])
    ax.set(xlabel="Stochastic updates (iterations)", ylabel="Training objective in RFF space",
           title=f'Selected RFF: lambda={row["lambda_"]:g}, gamma={row["gamma"]:g}, D={row["n_components"]}')
    ax.set_yscale("symlog", linthresh=1)
    save(fig, "rff_objective_vs_iterations.png")
    return figures


def run_experiments(data, metadata, configurations, output, *, input_sha256=None, verbose=True):
    """Train on train, select on validation, freeze choices, then evaluate test once.

    Retain only each family's current best model, avoiding a second fit after
    selection. Existing result directories are never overwritten. Timings are
    measurements, so repeated seeded runs reproduce metrics, not elapsed time.
    """
    audit = validate_prepared_data(data, metadata)
    if {c["family"] for c in configurations} != set(FAMILIES):
        raise ValueError("Configurations must cover all four SVM families.")
    if len({c["config_id"] for c in configurations}) != len(configurations):
        raise ValueError("Configuration IDs must be unique.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite results in {output}; choose a new --output.")
    # Check the plotting dependency before spending time fitting models.
    import matplotlib

    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "status": "running", "dataset": metadata, "data_audit": audit,
        "input_sha256": input_sha256, "configurations": configurations,
        "selection_rule": "Maximum validation F1, then ROC-AUC, then first in grid order",
        "test_policy": "Evaluate only four selected models after freezing all choices; no refit",
        "regularization": "SVC C = 1 / (n_train * lambda)",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("src/svm.py", "src/data_prep.py", "experiments/svm_experiments.py")},
    }
    _write_json(output / "configurations.json", manifest)
    rows, histories, best_rows, best_models = [], {}, {}, {}
    for index, config in enumerate(configurations, start=1):
        model = make_model(config, len(data["y_train"]))
        start = perf_counter()
        model.fit(data["X_train"], data["y_train"])
        training_seconds = perf_counter() - start
        row = {**config, "split": "validation", "training_seconds": training_seconds,
               **training_diagnostics(model), **evaluate_model(model, data["X_val"], data["y_val"])}
        rows.append(row)
        if isinstance(model, PegasosSVM):
            histories[config["config_id"]] = {
                "iterations": model.loss_iterations_, "objective": model.loss_history_,
            }
        family = config["family"]
        if family not in best_rows or selection_key(row) > selection_key(best_rows[family]):
            best_rows[family], best_models[family] = row, model
        _write_metrics(output / "metrics.csv", rows)
        _write_json(output / "loss_histories.json", histories)
        if verbose:
            print(f'[{index}/{len(configurations)}] {config["config_id"]}: '
                  f'validation F1={row["f1"]:.4f}, fit={training_seconds:.2f}s', flush=True)

    selected = select_best(rows)
    # Persist the decision BEFORE the first test prediction or test metric.
    _write_json(output / "selected_configurations.json", selected)
    manifest["status"] = "selected_before_test"
    _write_json(output / "configurations.json", manifest)
    for family in FAMILIES:
        row = {**selected[family], "split": "test",
               **evaluate_model(best_models[family], data["X_test"], data["y_test"])}
        rows.append(row)
        _write_metrics(output / "metrics.csv", rows)

    manifest["figures"] = save_figures(rows, histories, selected, output / "figures", metadata["dataset_name"])
    manifest["status"] = "complete"
    _write_json(output / "configurations.json", manifest)
    return {"metrics": rows, "selected": selected, "manifest": manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/svm_prepared.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/svm"))
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--iterations", type=int, nargs="+", default=[5000, 20000])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.01, 0.1])
    parser.add_argument("--components", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--constant-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        configurations = build_configurations(args.lambdas, args.iterations, args.gammas,
                                              args.components, args.constant_rate, args.seed)
        data, metadata = load_prepared_data(args.data)
        with args.data.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        result = run_experiments(data, metadata, configurations, args.output, input_sha256=digest)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.exit(2, f"SVM experiments: {error}\n")
    for family, row in result["selected"].items():
        print(f'{family}: selected {row["config_id"]}, validation F1={row["f1"]:.4f}')
    print(f"Saved actual results to {args.output}")


if __name__ == "__main__":
    main()
