import hashlib
import json
import platform
import shutil
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import sklearn

from src import data_prep
from experiments.prepare_svm_data import export_prepared
from experiments.svm_experiments import build_configurations, run_experiments, load_prepared_data
from experiments.svm_cross_validation import CONFIG_KEYS
from experiments.evaluation_experiments import run_evaluation, write_json
from experiments.tree_experiments import save_validation_curves


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_eda(frame, threshold, output):
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(frame["price"], bins=60)
    axes[0].set(xlabel="Price (AZN)", ylabel="Listings", title="Cleaned price distribution")
    axes[1].hist(np.log(frame["price"]), bins=60)
    axes[1].set(xlabel="Natural log of price (AZN)", ylabel="Listings", title="Log-price distribution")
    fig.tight_layout()
    fig.savefig(output / "price_distribution.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(frame["area_m2"], frame["price"], s=3, alpha=0.15)
    axes[0].set(xlabel="Area (m2)", ylabel="Price (AZN)", title="Area and price")
    frame.groupby("rooms")["price"].median().plot.bar(ax=axes[1])
    axes[1].set(xlabel="Rooms", ylabel="Median price (AZN)", title="Room count and price")
    fig.tight_layout()
    fig.savefig(output / "area_rooms.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    image = axes[0].scatter(frame["lng"], frame["lat"], c=np.log(frame["price"]), s=3, alpha=0.3)
    axes[0].set(xlabel="Longitude", ylabel="Latitude", title="Location and log-price")
    fig.colorbar(image, ax=axes[0], label="Natural log of price (AZN)")
    labels = (frame["price"].to_numpy() > threshold).astype(int)
    axes[1].bar(["Standard", "Premium"], np.bincount(labels, minlength=2))
    axes[1].set(ylabel="Listings", title="Tier balance: training-median threshold")
    fig.tight_layout()
    fig.savefig(output / "location_balance.png", dpi=150)
    plt.close(fig)


def run_pipeline(raw_path, output, include_cv=True, search_svm=True,
                 depths=(2, 4, 8, 12), leaf_sizes=(1, 5, 20), resume=False):
    raw_path, output = Path(raw_path), Path(output)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Place the real dataset at {raw_path}; see data/README.md.")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Output is not empty; choose a new --output to preserve earlier results.")
    if resume:
        old_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        if old_manifest["raw_sha256"] != sha256(raw_path):
            raise ValueError("Cannot resume with a different raw dataset.")
        if old_manifest["status"] == "complete":
            raise FileExistsError("The run is already complete; use a fresh output to reproduce it.")
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    manifest = {"status": "running", "seed": 42, "raw_sha256": sha256(raw_path), "resumed": resume,
                "cv_requested": include_cv, "svm_search_requested": search_svm,
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__}}
    write_json(output / "run_manifest.json", manifest)
    try:
        raw = data_prep.load_raw(str(raw_path))
        frame = data_prep.clean(raw)
        cleaned_path = output / "prepared/house_sale_clean.csv"
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cleaned_path, index=False, encoding="utf-8")
        write_json(output / "cleaning_audit.json", {
            "raw_rows": len(raw), "cleaned_rows": len(frame), "cleaned_columns": list(frame.columns),
            "raw_sha256": manifest["raw_sha256"], "cleaned_sha256": sha256(cleaned_path),
            "removed_leakage_columns": data_prep.LEAKAGE_COLUMNS,
            "rules_source": "src/data_prep.py: clean; fixed domain filters and structural missing-value rules"})
        prepared_path = output / "prepared/svm_prepared.npz"
        if resume:
            arrays, metadata = load_prepared_data(prepared_path)
            if metadata["cleaned_sha256"] != sha256(cleaned_path) or metadata["raw_sha256"] != sha256(raw_path):
                raise ValueError("Prepared data hashes do not match the resumed run.")
        else:
            arrays, metadata = export_prepared(cleaned_path, prepared_path, raw_path=raw_path)
        save_eda(frame, metadata["tier_threshold"], output / "figures/eda")
        selected_svm = None
        if search_svm:
            if resume:
                study_manifest = json.loads((output / "svm/configurations.json").read_text(encoding="utf-8"))
                if study_manifest["status"] != "complete" or study_manifest["dataset"]["cleaned_sha256"] != sha256(cleaned_path):
                    raise ValueError("Only a completed, matching SVM search can be reused.")
                for name, expected_hash in study_manifest["source_sha256"].items():
                    if sha256(name) != expected_hash:
                        raise ValueError("SVM search source changed; start a fresh run.")
                selections = json.loads((output / "svm/selected_configurations.json").read_text(encoding="utf-8"))
            else:
                study = run_experiments(arrays, metadata, build_configurations(), output / "svm")
                selections = study["selected"]
            if any(row["split"] != "validation" for row in selections.values()):
                raise ValueError("SVM settings must be selected on validation, never test.")
            selected_svm = {name: {key: row[key] for key in CONFIG_KEYS} for name, row in selections.items()}
        evaluation_output = output / "evaluation"
        attempt = 1
        while evaluation_output.exists() and any(evaluation_output.iterdir()):
            attempt += 1
            evaluation_output = output / f"evaluation_attempt_{attempt}"
        result = run_evaluation(cleaned_path, evaluation_output, include_cv=include_cv,
                                depths=depths, leaf_sizes=leaf_sizes, include_ridge=True, svm_configs=selected_svm)
        validation = pd.read_csv(evaluation_output / "tree_validation.csv").to_dict("records")
        for task in ("regression", "classification"):
            save_validation_curves([row for row in validation if row["task"] == task],
                                   output / f"figures/tree_{task}_validation.png")
        for name in ("test_metrics.csv", "cv_folds.csv", "cv_summary.csv", "error_groups.csv", "feature_effects.csv", "selected_configs.json"):
            source = evaluation_output / name
            if source.exists():
                shutil.copyfile(source, output / name)
        sources = ("src/linear.py", "src/run_all.py", "experiments/pipeline.py", "experiments/linear_experiments.py",
                   "src/data_prep.py", "src/decision_tree.py", "src/svm.py", "src/evaluate.py", "experiments/evaluation_experiments.py")
        manifest.update({"status": "complete", "models": len(result["test"]),
                         "split_counts": result["manifest"]["split_counts"],
                         "tier_threshold": result["manifest"]["tier_threshold"],
                         "source_hashes": {name: sha256(name) for name in sources},
                         "elapsed_seconds": perf_counter() - start,
                         "elapsed_scope": "This invocation; reused SVM search timings are recorded separately",
                         "test_used_for_selection": False, "ridge_included": True,
                         "evaluation_directory": evaluation_output.name,
                         "timing_policy": "Single-run measured times; not deterministic benchmark averages",
                         "data_policy": "Prepared CSV/NPZ stays local under results; commit only aggregate evidence"})
        write_json(output / "run_manifest.json", manifest)
        print(f"Complete: {len(result['test'])} models. Results saved to {output}", flush=True)
        return result
    except Exception as error:
        manifest.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        write_json(output / "run_manifest.json", manifest)
        raise
