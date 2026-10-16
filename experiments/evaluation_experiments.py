import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src import data_prep
from src.decision_tree import DecisionTree
from src.evaluate import rmse, mae, r2, precision_recall_f1, roc_auc, confusion_matrix, compare
from src.evaluation_plots import (
    plot_confusion_matrix, plot_actual_vs_predicted, plot_feature_values,
    summarize_errors_by_group, summarize_classification_by_group, location_groups,
)
from experiments.tree_experiments import compare_tree_configs
from experiments.svm_experiments import build_configurations, make_model
from experiments.svm_cross_validation import make_folds


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_data(path, seed=42):
    frame = pd.read_csv(path, encoding="utf-8")
    splits = data_prep.get_splits(seed=seed, path=str(path))
    ids = data_prep._split_indices(frame["price"].to_numpy(), seed=seed)
    for short, positions in zip(("tr", "val", "te"), ids):
        np.testing.assert_array_equal(splits[f"y_{short}"], frame.iloc[positions]["price"].to_numpy())
    return frame, splits, ids


def select_configs(splits, depths=(2, 4, 8, 12), leaf_sizes=(1, 5, 20)):
    configs, validation_rows = {}, []
    for task, target in (("regression", "y"), ("classification", "t")):
        print(f"Selecting tree {task} configurations on validation only", flush=True)
        rows, best = compare_tree_configs(splits["X_tr"], splits[f"{target}_tr"],
                                         splits["X_val"], splits[f"{target}_val"],
                                         task=task, depths=depths, leaf_sizes=leaf_sizes)
        validation_rows.extend(rows)
        for implementation, row in best.items():
            name = f"{implementation}_tree_{task}"
            configs[name] = {"kind": "tree", "task": task, "implementation": implementation,
                             "criterion": row["criterion"], "max_depth": row["max_depth"],
                             "min_samples_leaf": row["min_samples_leaf"],
                             "validation_score": row["validation_score"], "selection_metric": row["metric"]}
    fixed_ids = {"custom_linear_003", "custom_rff_017", "sklearn_linear_024", "sklearn_rbf_027"}
    for config in build_configurations():
        if config["config_id"] in fixed_ids:
            configs[config["family"]] = {"kind": "svm", "task": "classification", "config": config,
                                         "selection_source": "experiments/SVM_REPORT_NOTES.md: validation winners, 14 October"}
    return configs, validation_rows


def create_model(config, n_train):
    if config["kind"] == "svm":
        return make_model(config["config"], n_train)
    params = {"max_depth": config["max_depth"], "min_samples_leaf": config["min_samples_leaf"], "random_state": 42}
    if config["implementation"] == "custom":
        return DecisionTree(task=config["task"], criterion=config["criterion"], **params)
    if config["task"] == "regression":
        return DecisionTreeRegressor(criterion="squared_error", **params)
    return DecisionTreeClassifier(criterion=config["criterion"], **params)


def classification_scores(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    probabilities = model.predict_proba(X)
    positions = np.flatnonzero(model.classes_ == 1)
    if not len(positions):
        return np.zeros(len(X))
    return probabilities[:, positions[0]]


def metric_row(task, actual, predicted, scores=None):
    if task == "regression":
        return {"rmse": rmse(actual, predicted), "mae": mae(actual, predicted), "r2": r2(actual, predicted)}
    precision, recall, f1 = precision_recall_f1(actual, predicted)
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    return {"precision": precision, "recall": recall, "f1": f1,
            "roc_auc": roc_auc(actual, scores) if len(np.unique(actual)) == 2 else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def evaluate_models(configs, X_train, prices_train, tiers_train, X_test, prices_test, tiers_test):
    scaled_train, scaled_test = data_prep.standardize(X_train, X_test)
    rows, predictions, scores, models = [], {}, {}, {}
    for name, config in configs.items():
        print(f"Evaluating {name}", flush=True)
        svm = config["kind"] == "svm"
        train_X, test_X = (scaled_train, scaled_test) if svm else (X_train, X_test)
        train_y, test_y = (prices_train, prices_test) if config["task"] == "regression" else (tiers_train, tiers_test)
        model = create_model(config, len(X_train))
        start = perf_counter()
        model.fit(train_X, train_y)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        pred = model.predict(test_X)
        predict_seconds = perf_counter() - start
        score, score_seconds = None, None
        if config["task"] == "classification":
            start = perf_counter()
            score = classification_scores(model, test_X)
            score_seconds = perf_counter() - start
        row = {"model": name, "task": config["task"], "count": len(test_y),
               **metric_row(config["task"], test_y, pred, score),
               "fit_seconds": fit_seconds, "predict_seconds": predict_seconds, "score_seconds": score_seconds}
        rows.append(row)
        predictions[name], scores[name], models[name] = pred, score, model
    return rows, predictions, scores, models


def group_analysis(configs, predictions, scores, test_prices, test_tiers, train_frame, test_frame):
    threshold = float(np.quantile(train_frame["price"], 0.90))
    groups = {
        "price": np.where(test_prices > threshold, "Expensive", "Other"),
        "representation": location_groups(train_frame["location"], test_frame["location"]),
        "location": test_frame["location"].fillna("Unknown").to_numpy(),
    }
    rows = []
    for name, config in configs.items():
        for kind, values in groups.items():
            if config["task"] == "regression":
                results = summarize_errors_by_group(test_prices, predictions[name], values)
            else:
                results = summarize_classification_by_group(test_tiers, predictions[name], scores[name], values)
            for row in results:
                rows.append({"model": name, "task": config["task"], "group_type": kind, **row})
    return rows, threshold


def cross_validate(frame, development_ids, configs):
    development = frame.iloc[np.sort(development_ids)].reset_index(drop=True)
    rows, audit = [], []
    for fold, (train_ids, val_ids) in enumerate(make_folds(development["price"].to_numpy()), start=1):
        print(f"Cross-validation fold {fold}/5", flush=True)
        train, validation = development.iloc[train_ids], development.iloc[val_ids]
        encoder = data_prep.FeatureEncoder().fit(train)
        train_X, val_X = encoder.transform(train), encoder.transform(validation)
        train_prices, val_prices = train["price"].to_numpy(), validation["price"].to_numpy()
        train_tiers, threshold = data_prep.make_tier_label(train_prices)
        val_tiers, _ = data_prep.make_tier_label(val_prices, threshold)
        fold_rows, _, _, _ = evaluate_models(configs, train_X, train_prices, train_tiers, val_X, val_prices, val_tiers)
        rows.extend({"fold": fold, **row} for row in fold_rows)
        audit.append({"fold": fold, "train_count": len(train), "validation_count": len(validation),
                      "tier_threshold": float(threshold), "feature_count": train_X.shape[1]})
    summary = []
    for name, config in configs.items():
        metrics = ("rmse", "mae", "r2") if config["task"] == "regression" else ("precision", "recall", "f1", "roc_auc")
        selected = [row for row in rows if row["model"] == name]
        for metric in metrics:
            values = [row[metric] for row in selected if row[metric] is not None]
            summary.append({"model": name, "task": config["task"], "metric": metric, "folds": len(values),
                            "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))})
    return rows, summary, audit


def save_figures(output, configs, models, predictions, prices, tiers, feature_names):
    effects = []
    for name, config in configs.items():
        model = models[name]
        if config["task"] == "regression":
            plot_actual_vs_predicted(prices, predictions[name], output / f"{name}_price.png", title=name)
        else:
            plot_confusion_matrix(tiers, predictions[name], output / f"{name}_confusion.png", title=name)
        values, kind = None, None
        if config["kind"] == "tree":
            values, kind = model.feature_importances_, "Impurity importance"
        elif name == "custom_linear":
            values, kind = model.w, "Standardized linear weight"
        elif name == "sklearn_linear":
            values, kind = model.coef_[0], "Standardized linear weight"
        if values is not None:
            plot_feature_values(feature_names, values, output / f"{name}_features.png", title=name, xlabel=kind)
            effects.extend({"model": name, "feature": str(feature), "kind": kind, "value": float(value)}
                           for feature, value in zip(feature_names, values))
    return effects


def run_evaluation(clean_path, output, include_cv=True, depths=(2, 4, 8, 12), leaf_sizes=(1, 5, 20)):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory must be empty; preserve previous test results.")
    frame, splits, ids = load_data(clean_path)
    output.mkdir(parents=True, exist_ok=True)
    configs, validation = select_configs(splits, depths, leaf_sizes)
    write_json(output / "selected_configs.json", configs)
    write_csv(output / "tree_validation.csv", validation)
    cv_rows, cv_summary, cv_audit = [], [], []
    if include_cv:
        cv_rows, cv_summary, cv_audit = cross_validate(frame, np.concatenate(ids[:2]), configs)
        write_csv(output / "cv_folds.csv", cv_rows)
        write_csv(output / "cv_summary.csv", cv_summary)
        write_json(output / "cv_preprocessing.json", cv_audit)
    test, predictions, scores, models = evaluate_models(configs, splits["X_tr"], splits["y_tr"], splits["t_tr"],
                                                       splits["X_te"], splits["y_te"], splits["t_te"])
    groups, expensive_threshold = group_analysis(configs, predictions, scores, splits["y_te"], splits["t_te"],
                                                 frame.iloc[ids[0]], frame.iloc[ids[2]])
    effects = save_figures(output / "figures", configs, models, predictions, splits["y_te"], splits["t_te"], splits["feature_names"])
    write_csv(output / "test_metrics.csv", test)
    write_csv(output / "error_groups.csv", groups)
    write_csv(output / "feature_effects.csv", effects)
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}
    source_hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                     for path in ("src/data_prep.py", "src/decision_tree.py", "src/svm.py", "src/evaluate.py", "experiments/evaluation_experiments.py")}
    manifest = {"dataset": "Real Bina.az cleaned sale listings", "cleaned_sha256": hashlib.sha256(Path(clean_path).read_bytes()).hexdigest(),
                "seed": 42, "split_counts": dict(zip(("train", "validation", "test"), map(len, ids))),
                "tier_threshold": float(splits["tier_threshold"]), "expensive_threshold": expensive_threshold,
                "sparse_location_training_count_below": 200, "feature_count": len(splits["feature_names"]),
                "target_units": "AZN", "encoding_and_scaling_fit": "training only; refitted inside each CV fold",
                "test_used_for_selection": False, "cv_pool": "original train + validation; final test excluded",
                "cv_note": "Descriptive uncertainty for fixed selected settings; not nested model-selection CV",
                "cv_complete": include_cv, "versions": versions, "source_hashes": source_hashes,
                "timing_note": "Single-run wall-clock timings vary with hardware and load", "status": "complete"}
    write_json(output / "manifest.json", manifest)
    write_csv(output / "split_audit.csv", [{"split": split, "rows": len(part), "ids_sha256": hashlib.sha256(np.asarray(part, dtype=np.int64).tobytes()).hexdigest()}
                                          for split, part in zip(("train", "validation", "test"), ids)])
    compare({row["model"]: {key: row[key] for key in ("rmse", "mae", "r2", "precision", "recall", "f1", "roc_auc") if key in row} for row in test})
    return {"test": test, "groups": groups, "cv_summary": cv_summary, "manifest": manifest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, default=Path(data_prep.CLEAN_PATH))
    parser.add_argument("--output", type=Path, default=Path("evaluation/artifacts"))
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()
    run_evaluation(args.clean, args.output, include_cv=not args.skip_cv)


if __name__ == "__main__":
    main()
