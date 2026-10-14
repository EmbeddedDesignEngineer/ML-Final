"""Experiment wiring checks; all fixtures are synthetic, never report results."""

import csv
import json
from collections import Counter

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from experiments import svm_experiments as exp
from src.svm import PegasosSVM


@pytest.fixture
def prepared():
    rng = np.random.default_rng(15)
    raw = [rng.normal(size=(n, 3)) for n in (40, 20, 24)]
    mean, std = raw[0].mean(axis=0), raw[0].std(axis=0)
    prices = [100 + 10 * X[:, 0] + 4 * X[:, 1] for X in raw]
    threshold = float(np.median(prices[0]))
    data = {"feature_names": np.array(["area_m2", "rooms", "floor"])}
    for split, X, price in zip(exp.SPLITS, raw, prices):
        data[f"X_{split}"] = (X - mean) / std
        data[f"y_{split}"] = (price > threshold).astype(int)
        data[f"{split}_ids"] = np.array([f"{split}_{i}" for i in range(len(X))])
    metadata = {
        "dataset_name": "SYNTHETIC UNIT TEST FIXTURE (not Bina.az)",
        "preprocessing_source": "tests/test_svm_experiments.py::prepared",
        "split_strategy": "Independent synthetic draws; test fixture only",
        "scaling_fit_split": "train", "encoding_fit_split": "train",
        "tier_threshold_fit_split": "train", "tier_threshold": threshold,
    }
    return data, metadata


@pytest.fixture
def tiny_grid():
    # Multiple values exercise the plot groups and selection; update budgets
    # remain tiny because these are wiring checks, not performance claims.
    return exp.build_configurations(lambdas=(0.01, 0.1), iterations=(30, 60),
                                    gammas=(0.2, 0.8), components=(8, 16))


def test_default_grid_has_controlled_comparisons_without_duplicates():
    configs = exp.build_configurations()
    assert Counter(c["family"] for c in configs) == {
        "custom_linear": 12, "custom_rff": 12, "sklearn_linear": 3, "sklearn_rbf": 6,
    }
    params = [{key: value for key, value in c.items() if key != "config_id"} for c in configs]
    assert len({json.dumps(c, sort_keys=True) for c in params}) == len(configs)
    assert all(c["random_state"] == 42 for c in configs)
    assert {c["n_iters"] for c in configs if c["family"] == "custom_linear"} == {5000, 20000}
    assert {c["learning_rate"] for c in configs if c["family"] == "custom_linear"} == {"constant", "pegasos"}


@pytest.mark.parametrize("kwargs", [
    {"lambdas": ()}, {"lambdas": (0.01, 0.01)}, {"lambdas": (-1,)},
    {"lambdas": (np.nan,)}, {"iterations": (1.5,)}, {"components": (0,)},
    {"gammas": (np.inf,)}, {"constant_rate": 0}, {"constant_rate": 1e9},
    {"seed": -1}, {"seed": True},
])
def test_invalid_grids_fail_before_training(kwargs):
    with pytest.raises(ValueError):
        exp.build_configurations(**kwargs)


def test_baseline_regularization_and_custom_configuration(tiny_grid):
    for config in tiny_grid:
        model = exp.make_model(config, 40)
        if config["family"].startswith("custom"):
            assert isinstance(model, PegasosSVM)
            assert model.lambda_ == config["lambda_"]
            assert model.n_components == config["n_components"]
        else:
            assert model.C == pytest.approx(1 / (40 * config["lambda_"]))
            assert model.kernel == ("rbf" if config["family"] == "sklearn_rbf" else "linear")


def test_prepared_npz_roundtrip_and_no_new_scaling(prepared, tmp_path):
    data, metadata = prepared
    # Large held-out shifts are allowed: fitting a new scaler there would leak.
    data["X_val"] = data["X_val"] * 5 + 20
    path = tmp_path / "prepared.npz"
    np.savez(path, **data, metadata_json=json.dumps(metadata))
    restored, restored_metadata = exp.load_prepared_data(path)
    assert restored_metadata == metadata
    audit = exp.validate_prepared_data(restored, restored_metadata)
    assert audit["split_sizes"] == {"train": 40, "val": 20, "test": 24}
    assert audit["listing_ids_disjoint"] is True
    for name in data:
        assert_array_equal(restored[name], data[name])


def test_missing_and_incomplete_npz_fail_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No real experiment was run"):
        exp.load_prepared_data(tmp_path / "missing.npz")
    np.savez(tmp_path / "incomplete.npz", X_train=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="missing keys"):
        exp.load_prepared_data(tmp_path / "incomplete.npz")


@pytest.mark.parametrize("pair", [("train", "val"), ("train", "test"), ("val", "test")])
def test_split_overlap_rejected(prepared, pair):
    data, metadata = prepared
    left, right = pair
    # Use object-free strings with enough width to avoid fixture truncation.
    data[f"{right}_ids"] = data[f"{right}_ids"].astype("U30")
    data[f"{right}_ids"][0] = data[f"{left}_ids"][0]
    with pytest.raises(ValueError, match="overlap"):
        exp.validate_prepared_data(data, metadata)


@pytest.mark.parametrize("key, value, message", [
    ("X_train", np.zeros((40, 4)), "features"),
    ("X_train", np.full((40, 3), 7.0), "centred"),
    ("X_val", np.full((20, 3), np.nan), "finite"),
    ("y_val", np.zeros(20, dtype=int), "both labels"),
    ("y_train", np.ones((40, 1)), "1-D"),
    ("train_ids", np.repeat("one-listing", 40), "duplicate"),
    ("feature_names", np.array(["area", "rooms", "price"]), "leakage"),
])
def test_invalid_prepared_arrays(prepared, key, value, message):
    data, metadata = prepared
    data[key] = value
    with pytest.raises(ValueError, match=message):
        exp.validate_prepared_data(data, metadata)


def test_nonstandard_variance_rejected(prepared):
    data, metadata = prepared
    data["X_train"] *= 2
    with pytest.raises(ValueError, match="standard deviation"):
        exp.validate_prepared_data(data, metadata)


@pytest.mark.parametrize("field, value", [
    ("scaling_fit_split", "all"), ("encoding_fit_split", "train+val"),
    ("tier_threshold_fit_split", "test"), ("tier_threshold", float("nan")),
    ("preprocessing_source", ""), ("dataset_name", ""),
])
def test_invalid_provenance_rejected(prepared, field, value):
    data, metadata = prepared
    metadata[field] = value
    with pytest.raises(ValueError, match=field):
        exp.validate_prepared_data(data, metadata)


def test_selection_uses_validation_f1_then_auc_never_training_time():
    rows = [
        dict(family="custom_linear", config_id="a", split="validation", f1=0.8, roc_auc=0.9, training_seconds=1),
        dict(family="custom_linear", config_id="b", split="validation", f1=0.8, roc_auc=0.95, training_seconds=100),
        dict(family="custom_linear", config_id="c", split="validation", f1=0.79, roc_auc=1.0, training_seconds=0),
    ]
    assert exp.select_best(rows)["custom_linear"]["config_id"] == "b"
    rows.append({**rows[1], "config_id": "later-tie", "training_seconds": 0})
    assert exp.select_best(rows)["custom_linear"]["config_id"] == "b"
    with pytest.raises(ValueError, match="validation rows only"):
        exp.select_best([{**rows[0], "split": "test"}])


def test_metrics_match_a_hand_computed_confusion_matrix():
    class FixedPredictions:
        def predict(self, X):
            return np.array([0, 1, 0, 1])

        def decision_function(self, X):
            return np.array([-2, 0.2, -0.1, 2])

    metrics = exp.evaluate_model(FixedPredictions(), np.zeros((4, 1)), [0, 0, 1, 1])
    assert [metrics[key] for key in ("tn", "fp", "fn", "tp")] == [1, 1, 1, 1]
    assert metrics["accuracy"] == metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert metrics["roc_auc"] == 0.75  # Uses ranked scores, not thresholded predictions.
    assert metrics["prediction_seconds"] >= 0


def test_end_to_end_separation_reproducibility_and_saved_outputs(prepared, tiny_grid, tmp_path, monkeypatch):
    data, metadata = prepared
    originals = {key: value.copy() for key, value in data.items()}
    output = tmp_path / "synthetic_checks_only"
    fit_calls, evaluations = [], []
    original_custom_fit, original_svc_fit = PegasosSVM.fit, exp.SVC.fit
    original_evaluate = exp.evaluate_model

    def spy_fit(original):
        def fit(model, X, y):
            assert X is data["X_train"] and y is data["y_train"]
            fit_calls.append(type(model).__name__)
            return original(model, X, y)
        return fit

    def spy_evaluate(model, X, y):
        if X is data["X_test"]:
            assert evaluations.count("validation") == len(tiny_grid)
            assert (output / "selected_configurations.json").is_file()
            manifest = json.loads((output / "configurations.json").read_text())
            assert manifest["status"] == "selected_before_test"
            assert y is data["y_test"]
            evaluations.append("test")
        else:
            assert X is data["X_val"] and y is data["y_val"]
            assert "test" not in evaluations
            evaluations.append("validation")
        return original_evaluate(model, X, y)

    monkeypatch.setattr(PegasosSVM, "fit", spy_fit(original_custom_fit))
    monkeypatch.setattr(exp.SVC, "fit", spy_fit(original_svc_fit))
    monkeypatch.setattr(exp, "evaluate_model", spy_evaluate)
    result = exp.run_experiments(data, metadata, tiny_grid, output, verbose=False)
    assert len(fit_calls) == len(tiny_grid)  # Winners are not refitted.
    assert evaluations.count("test") == 4
    assert len(result["metrics"]) == len(tiny_grid) + 4
    for key in data:
        assert_array_equal(data[key], originals[key])

    with (output / "metrics.csv").open() as file:
        saved_rows = list(csv.DictReader(file))
    assert len(saved_rows) == len(tiny_grid) + 4
    for row in result["metrics"]:
        n = len(data["y_val"] if row["split"] == "validation" else data["y_test"])
        assert sum(row[key] for key in ("tn", "fp", "fn", "tp")) == n
        assert all(0 <= row[key] <= 1 for key in ("accuracy", "precision", "recall", "f1", "roc_auc"))
        assert row["training_seconds"] >= 0
        if row["family"].startswith("custom"):
            assert row["final_objective"] is not None
            assert 0 <= row["margin_active_fraction"] <= 1
        else:
            assert row["final_objective"] is None
    assert result["manifest"]["status"] == "complete"
    assert "SYNTHETIC" in result["manifest"]["dataset"]["dataset_name"]
    for name in ("configurations.json", "selected_configurations.json", "loss_histories.json"):
        assert json.loads((output / name).read_text())
    pngs = sorted((output / "figures").glob("*.png"))
    assert len(pngs) == 6
    for path in pngs:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 1000
    with pytest.raises(FileExistsError, match="overwrite"):
        exp.run_experiments(data, metadata, tiny_grid, output, verbose=False)

    # Repeat the seed with inverted test labels. Every validation result and
    # selected configuration must stay identical, while final test metrics change.
    monkeypatch.setattr(exp, "evaluate_model", original_evaluate)
    monkeypatch.setattr(exp, "save_figures", lambda *args: [])
    data["y_test"] = 1 - data["y_test"]
    second = exp.run_experiments(data, metadata, tiny_grid, tmp_path / "repeat", verbose=False)
    without_times = lambda row: {k: v for k, v in row.items() if not k.endswith("_seconds")}
    first_val = [without_times(r) for r in result["metrics"] if r["split"] == "validation"]
    second_val = [without_times(r) for r in second["metrics"] if r["split"] == "validation"]
    assert first_val == second_val
    assert {k: v["config_id"] for k, v in result["selected"].items()} == {
        k: v["config_id"] for k, v in second["selected"].items()
    }
    assert any(a["f1"] != b["f1"] for a, b in zip(result["metrics"][-4:], second["metrics"][-4:]))
    first_histories = json.loads((output / "loss_histories.json").read_text())
    second_histories = json.loads((tmp_path / "repeat/loss_histories.json").read_text())
    assert first_histories == second_histories


def test_cli_missing_data_creates_no_fake_results(tmp_path, capsys):
    output = tmp_path / "results"
    with pytest.raises(SystemExit) as error:
        exp.main(["--data", str(tmp_path / "unavailable.npz"), "--output", str(output)])
    assert error.value.code == 2
    assert "No real experiment was run" in capsys.readouterr().err
    assert not output.exists()
