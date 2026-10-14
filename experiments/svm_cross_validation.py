"""Five-fold development-set CV for the four frozen SVM configurations.

Run: python -m experiments.svm_cross_validation
Rebuild features from cleaned rows inside every fold. The prepared NPZ supplies
only original split IDs and provenance; its fitted features/labels are not used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import data_prep
from experiments.prepare_svm_data import file_sha256
from experiments.svm_experiments import (
    FAMILIES, _write_json, evaluate_model, make_model, training_diagnostics,
)


N_FOLDS = 5
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
CONFIG_KEYS = ("config_id", "family", "lambda_", "n_iters", "learning_rate",
               "eta0", "gamma", "n_components", "random_state")
OUTPUT_FILES = ("cv_fold_metrics.csv", "cv_summary.csv", "cv_summary.md",
                "cv_manifest.json", "cv_assignments.csv", "cv_preprocessing.json")


def load_development_data(clean_path, prepared_path):
    """Load original train+validation rows; skip final-test rows in the CSV.

    Access test IDs only to enforce exclusion. Never load X_test/y_test or
    reuse the original training encoder, scaler or tier threshold.
    """
    with np.load(prepared_path, allow_pickle=False) as archive:
        ids = [archive[f"{split}_ids"] for split in ("train", "val", "test")]
        metadata = json.loads(str(archive["metadata_json"].item()))
    for part in ids:
        if part.ndim != 1 or part.dtype.kind not in "iu" or not len(part):
            raise ValueError("Original split IDs must be nonempty 1-D integer row positions.")
    joined = np.concatenate(ids)
    if not np.array_equal(np.sort(joined), np.arange(len(joined))):
        raise ValueError("Original split IDs must be disjoint, unique and exhaustive.")
    if file_sha256(clean_path) != metadata.get("cleaned_sha256"):
        raise ValueError("Cleaned CSV does not match the prepared split provenance.")

    development_ids = np.sort(np.concatenate(ids[:2]))
    # Use only known row positions, not final-test feature/price values.
    keep = set(development_ids)
    development = pd.read_csv(
        clean_path, encoding="utf-8", skiprows=lambda row: row > 0 and row - 1 not in keep,
    )
    if len(development) != len(development_ids):
        raise ValueError("Cleaned CSV row count does not match the saved split IDs.")
    development.index = development_ids
    provenance = {
        "cleaned_path": str(clean_path), "cleaned_sha256": file_sha256(clean_path),
        "prepared_path": str(prepared_path), "prepared_sha256": file_sha256(prepared_path),
        "original_split_sizes": dict(zip(("train", "val", "test"), map(len, ids))),
        "development_rows": len(development), "final_test_excluded": True,
        "pool": "Original training + validation rows; final test rows skipped at CSV loading",
    }
    return development, ids[2], provenance


def load_fixed_configurations(path):
    """Read parameters from the saved validation selections, without reselecting."""
    selected = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(selected) != set(FAMILIES):
        raise ValueError("Exactly one saved selection per SVM family is required.")
    configurations = []
    for family in FAMILIES:
        row = selected[family]
        if row.get("family") != family or row.get("split") != "validation":
            raise ValueError("Fixed configurations must be the original validation selections.")
        if not set(CONFIG_KEYS).issubset(row):
            raise ValueError(f"Incomplete fixed configuration for {family}.")
        configurations.append({key: row[key] for key in CONFIG_KEYS})
    return configurations


def make_folds(prices, seed=42):
    """Price-decile stratification within development only; no global tier label.

    Deciles allocate rows to folds. Actual tier thresholds are learned later,
    independently from the training prices in each fold.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or not len(prices) or not np.isfinite(prices).all():
        raise ValueError("Development prices must be a finite nonempty 1-D array.")
    edges = np.quantile(prices, np.linspace(0, 1, 11)[1:-1])
    strata = np.searchsorted(edges, prices, side="right")
    if np.unique(strata, return_counts=True)[1].min() < N_FOLDS:
        raise ValueError("Each development price stratum needs at least five rows.")
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(prices)), strata))


def prepare_fold(train, validation):
    """Fit numeric fills, Davud's encoder, labels and scaler on this train fold.

    Shared cleaning already applies structural fills and removes incomplete
    essentials. Train-only numeric medians are a safeguard for remaining NaNs;
    categorical unknowns follow Davud's existing fixed/other encoding policy.
    """
    numeric = train[data_prep.NUMERIC_FEATURES].to_numpy(dtype=float)
    if np.isinf(numeric).any() or np.isnan(numeric).all(axis=0).any():
        raise ValueError("Training numeric columns must have observed finite values.")
    medians = np.nanmedian(numeric, axis=0)

    def impute(frame):
        filled = frame.copy()
        values = frame[data_prep.NUMERIC_FEATURES].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("Numeric features cannot contain infinity.")
        missing = np.isnan(values)
        values = np.where(missing, medians, values)
        for column, values_column in zip(data_prep.NUMERIC_FEATURES, values.T):
            filled[column] = values_column
        return filled, int(missing.sum())

    filled_train, train_missing = impute(train)
    filled_val, val_missing = impute(validation)
    encoder = data_prep.FeatureEncoder().fit(filled_train)
    X_train, X_val = encoder.transform(filled_train), encoder.transform(filled_val)
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Encoded fold features must be finite.")
    for frame in (train, validation):
        if not np.isfinite(frame["price"].to_numpy(dtype=float)).all():
            raise ValueError("Fold prices must be finite; targets are never imputed.")
    y_train, threshold = data_prep.make_tier_label(train["price"].to_numpy())
    y_val, _ = data_prep.make_tier_label(validation["price"].to_numpy(), threshold)
    if any(not np.array_equal(np.unique(y), [0, 1]) for y in (y_train, y_val)):
        raise ValueError("Each fold needs both tier classes in training and validation.")
    train_scaled, val_scaled = data_prep.standardize(X_train, X_val)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    audit = {
        "fit_scope": "fold_training_only", "tier_threshold": threshold,
        "numeric_imputation_medians": dict(zip(data_prep.NUMERIC_FEATURES, medians.tolist())),
        "imputed_train_cells": train_missing, "imputed_validation_cells": val_missing,
        "frequent_locations": encoder.locations_, "fixed_categories": encoder.categories_,
        "feature_names": encoder.feature_names_, "scaler_mean": mean.tolist(),
        "scaler_std": np.where(std == 0, 1, std).tolist(),
        "training_premium_fraction": float(y_train.mean()),
        "validation_premium_fraction": float(y_val.mean()),
    }
    arrays = dict(X_train=train_scaled, X_val=val_scaled, y_train=y_train, y_val=y_val,
                  unscaled_train=X_train, unscaled_val=X_val)
    return arrays, audit


def summarize_folds(rows):
    """Unweighted fold means and SAMPLE standard deviations (ddof=1)."""
    frame = pd.DataFrame(rows)
    summaries = []
    for family in FAMILIES:
        group = frame[frame["family"] == family]
        if len(group) != N_FOLDS or set(group["fold"]) != set(range(1, N_FOLDS + 1)):
            raise ValueError("Summary requires exactly five distinct folds per family.")
        values = group[list(METRICS)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Fold metrics must be finite values in [0,1].")
        row = {"family": family, "n_folds": N_FOLDS}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summaries.append(row)
    return summaries


def run_cross_validation(development, excluded_test_ids, configurations, output,
                         *, seed=42, provenance=None, verbose=True):
    """Twenty fits, no hyperparameter search and no final-test evaluation."""
    output = Path(output)
    if any((output / name).exists() for name in OUTPUT_FILES):
        raise FileExistsError("CV outputs already exist; use a new --output directory.")
    if (not development.index.is_unique
            or np.intersect1d(development.index, excluded_test_ids).size):
        raise ValueError("Development IDs must be unique and exclude final-test IDs.")
    if len(configurations) != 4 or {c["family"] for c in configurations} != set(FAMILIES):
        raise ValueError("Provide exactly one fixed configuration for each family.")
    folds = make_folds(development["price"].to_numpy(), seed)
    assignment = np.zeros(len(development), dtype=int)
    for fold, (_, val_indices) in enumerate(folds, start=1):
        assignment[val_indices] = fold
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cleaned_row_id": development.index, "validation_fold": assignment}).to_csv(
        output / "cv_assignments.csv", index=False,
    )
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "status": "running", "n_folds": N_FOLDS, "seed": seed,
        "provenance": provenance or {}, "fixed_configurations": configurations,
        "split_policy": "Shuffled stratified 5-fold on development price deciles",
        "label_policy": "Training-fold price median; reused unchanged for its validation fold",
        "regularization": "Fixed lambda; sklearn C=1/(fold_training_size*lambda)",
        "preprocessing": "Training-fold numeric medians, shared FeatureEncoder and standardize; "
                         "baseline scaling inside a fresh StandardScaler/SVC Pipeline",
        "uncertainty": "Unweighted mean and sample standard deviation across five folds (ddof=1); not a confidence interval",
        "limitation": "Fixed configurations were previously selected using this development pool; "
                      "this is not nested CV and may retain selection optimism. Final test is not reused.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "source_sha256": {name: file_sha256(root / name) for name in
                          ("src/data_prep.py", "src/svm.py", "experiments/svm_experiments.py",
                           "experiments/svm_cross_validation.py")},
    }
    _write_json(output / "cv_manifest.json", manifest)
    rows, audits = [], []
    for fold, (train_indices, val_indices) in enumerate(folds, start=1):
        start = perf_counter()
        arrays, audit = prepare_fold(development.iloc[train_indices], development.iloc[val_indices])
        preprocessing_seconds = perf_counter() - start
        audits.append({"fold": fold, "n_train": len(train_indices), "n_validation": len(val_indices), **audit})
        _write_json(output / "cv_preprocessing.json", audits)
        for config in configurations:
            estimator = make_model(config, len(train_indices))
            baseline = config["family"].startswith("sklearn_")
            model = Pipeline([("scale", StandardScaler()), ("svm", estimator)]) if baseline else estimator
            X_train = arrays["unscaled_train"] if baseline else arrays["X_train"]
            X_val = arrays["unscaled_val"] if baseline else arrays["X_val"]
            start = perf_counter()
            model.fit(X_train, arrays["y_train"])
            training_seconds = perf_counter() - start
            if baseline:
                np.testing.assert_allclose(model.named_steps["scale"].mean_, audit["scaler_mean"])
                np.testing.assert_allclose(model.named_steps["scale"].scale_, audit["scaler_std"])
            row = {**config, "fold": fold, "split": "cv_validation",
                   "n_train": len(train_indices), "n_validation": len(val_indices),
                   "n_features": X_train.shape[1], "tier_threshold": audit["tier_threshold"],
                   "C": estimator.C if baseline else None,
                   "preprocessing_seconds": preprocessing_seconds, "training_seconds": training_seconds,
                   **training_diagnostics(estimator), **evaluate_model(model, X_val, arrays["y_val"])}
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / "cv_fold_metrics.csv", index=False)
            if verbose:
                print(f'Fold {fold}/5 {config["family"]}: F1={row["f1"]:.4f}, '
                      f'AUC={row["roc_auc"]:.4f}, fit={training_seconds:.2f}s', flush=True)
    summary = summarize_folds(rows)
    pd.DataFrame(summary).to_csv(output / "cv_summary.csv", index=False)
    table = ["| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        cells = [f'{row[m + "_mean"]:.6f} ± {row[m + "_std"]:.6f}' for m in METRICS]
        table.append("| " + " | ".join([row["family"], *cells]) + " |")
    (output / "cv_summary.md").write_text(
        "Development-only 5-fold CV; mean ± sample standard deviation (ddof=1).\n\n"
        + "\n".join(table) + "\n", encoding="utf-8",
    )
    manifest.update(status="complete", fits=len(rows), summary=summary)
    _write_json(output / "cv_manifest.json", manifest)
    return {"metrics": rows, "summary": summary, "audits": audits, "manifest": manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=Path(data_prep.CLEAN_PATH))
    parser.add_argument("--prepared", type=Path, default=Path("data/svm_prepared.npz"))
    parser.add_argument("--selected", type=Path, default=Path("results/svm/selected_configurations.json"))
    parser.add_argument("--output", type=Path, default=Path("results/svm"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    configurations = load_fixed_configurations(args.selected)
    development, excluded, provenance = load_development_data(args.clean, args.prepared)
    provenance.update(selected_path=str(args.selected), selected_sha256=file_sha256(args.selected))
    result = run_cross_validation(development, excluded, configurations, args.output,
                                  seed=args.seed, provenance=provenance)
    for row in result["summary"]:
        print(f'{row["family"]}: CV F1={row["f1_mean"]:.4f} ± {row["f1_std"]:.4f}')
    print(f"Saved five-fold CV results to {args.output}")


if __name__ == "__main__":
    main()
