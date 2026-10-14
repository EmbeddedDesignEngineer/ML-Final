"""Export Davud's train-fitted features for the standalone SVM runner.

Run from the repo root: python -m experiments.prepare_svm_data
Cleaning, encoding, splitting, labels and scaling all belong to src.data_prep.
This adapter validates their outputs and writes the agreed NPZ format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from src import data_prep
from experiments.svm_experiments import validate_prepared_data


def file_sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def export_prepared(clean_path, output, seed=42, raw_path=None):
    """Use the shared pipeline, check its contract, and preserve existing exports."""
    clean_path, output = Path(clean_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    frame = pd.read_csv(clean_path, encoding="utf-8")
    splits = data_prep.get_splits(seed=seed, path=str(clean_path))

    # The public splitter also accepts row IDs. Using the same prices/seed
    # recovers membership without duplicating its stratification algorithm.
    indexed = data_prep.train_val_test_split(
        np.arange(len(frame)), frame["price"].to_numpy(), seed=seed,
    )
    ids = {"train": indexed[0], "val": indexed[2], "test": indexed[4]}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(ids.values()))), np.arange(len(frame)))
    encoder = splits["encoder"]
    train_frame = frame.iloc[ids["train"]]
    counts = train_frame["location"].value_counts()
    expected_locations = sorted(counts[counts >= data_prep.MIN_LOCATION_COUNT].index)
    if encoder.locations_ != expected_locations:
        raise ValueError("Encoder locations do not match training-only frequencies.")
    if encoder.categories_ != list(data_prep.CATEGORY_MAP.values()):
        raise ValueError("Encoder categories do not match the fixed domain vocabulary.")

    mean = splits["X_tr"].mean(axis=0)
    std = splits["X_tr"].std(axis=0)
    safe_std = np.where(std == 0, 1, std)
    scaled = data_prep.standardize(splits["X_tr"], splits["X_val"], splits["X_te"])
    threshold = float(splits["tier_threshold"])
    if threshold != float(np.median(splits["y_tr"])):
        raise ValueError("Tier threshold is not the training-price median.")
    data = {"feature_names": np.asarray(splits["feature_names"], dtype=str)}
    for name, short, X_scaled in zip(("train", "val", "test"), ("tr", "val", "te"), scaled):
        rows = frame.iloc[ids[name]]
        np.testing.assert_array_equal(splits[f"y_{short}"], rows["price"].to_numpy())
        np.testing.assert_allclose(splits[f"X_{short}"], encoder.transform(rows), rtol=0, atol=0)
        np.testing.assert_array_equal(splits[f"t_{short}"], (rows["price"].to_numpy() > threshold).astype(int))
        np.testing.assert_allclose(X_scaled, (splits[f"X_{short}"] - mean) / safe_std)
        data[f"X_{name}"] = X_scaled
        data[f"y_{name}"] = splits[f"t_{short}"]
        data[f"{name}_ids"] = ids[name]

    # Held-out transforms must neither learn a new vocabulary nor change train.
    state_before = (list(encoder.locations_), list(encoder.feature_names_))
    probe = frame.iloc[ids["val"][:5]].copy()
    probe["location"] = "__unseen_svm_audit_location__"
    probe["price"] = -1.0  # Target changes must not affect feature computation.
    probe_X = encoder.transform(probe)
    np.testing.assert_array_equal(probe_X[:, -1], np.ones(len(probe)))
    if (encoder.locations_, encoder.feature_names_) != state_before:
        raise ValueError("A held-out transform changed the fitted encoder.")
    np.testing.assert_array_equal(encoder.transform(train_frame), splits["X_tr"])

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "dataset_name": "Bina.az sale listings (local house_sale.csv)",
        "preprocessing_source": f"src/data_prep.py at {commit}; get_splits + standardize",
        "preprocessing_sha256": file_sha256(data_prep.__file__),
        "exporter_sha256": file_sha256(__file__),
        "cleaned_path": str(clean_path), "cleaned_sha256": file_sha256(clean_path),
        "split_strategy": f"Shared price-decile stratification; 70/15/15; seed={seed}",
        "split_seed": seed, "encoding_fit_split": "train", "scaling_fit_split": "train",
        "tier_threshold_fit_split": "train", "tier_threshold": threshold,
        "id_definition": "Deduplicated cleaned-row position, tied to cleaned_sha256",
        "frequent_locations": encoder.locations_, "fixed_categories": encoder.categories_,
        "scaler_mean": mean.tolist(), "scaler_std": safe_std.tolist(),
        "encoding_invariance_check": "Unseen held-out location leaves vocabulary and train features unchanged",
    }
    if raw_path is not None:
        metadata["raw_path"] = str(raw_path)
        metadata["raw_sha256"] = file_sha256(raw_path)
    metadata["array_checks"] = validate_prepared_data(data, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        np.savez_compressed(file, **data, metadata_json=json.dumps(metadata, ensure_ascii=False))
    return data, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=Path(data_prep.CLEAN_PATH))
    parser.add_argument("--raw", type=Path, default=Path(data_prep.DATA_PATH))
    parser.add_argument("--output", type=Path, default=Path("data/svm_prepared.npz"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _, metadata = export_prepared(args.clean, args.output, args.seed, args.raw)
    print(json.dumps(metadata["array_checks"], indent=2))
    print(f'Training median: {metadata["tier_threshold"]:,.0f} AZN')
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
