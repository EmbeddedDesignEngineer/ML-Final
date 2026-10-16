import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src import data_prep
from src.linear import RidgeRegression
from experiments import pipeline, evaluation_experiments as evaluation
from experiments.linear_experiments import create_ridge_model, select_ridge_configs


@pytest.fixture
def cleaned_frame():
    rng = np.random.default_rng(19)
    frame = pd.DataFrame({name: rng.uniform(1, 10, 200) for name in data_prep.NUMERIC_FEATURES})
    frame["price"] = np.arange(200) * 1000 + 50000
    frame["area_m2"] = frame["price"] / 1500
    frame["category"] = np.resize(list(data_prep.CATEGORY_MAP.values()), 200)
    frame["location"] = np.resize(["A", "B", "C"], 200)
    frame["lat"] = rng.uniform(40.3, 40.5, 200)
    frame["lng"] = rng.uniform(49.7, 49.9, 200)
    return frame


def test_ridge_selection_ignores_test_targets(cleaned_frame, tmp_path):
    path = tmp_path / "clean.csv"
    cleaned_frame.to_csv(path, index=False)
    splits = data_prep.get_splits(path=str(path))
    first, rows = select_ridge_configs(splits, alphas=(0.1, 1.0))
    splits["y_te"][:] = 1e12
    splits["X_te"][:] = -9999
    second, _ = select_ridge_configs(splits, alphas=(0.1, 1.0))
    assert first == second
    assert len(rows) == 4
    model = create_ridge_model(first["custom_ridge"])
    model.fit(splits["X_tr"], splits["y_tr"])
    np.testing.assert_allclose(model.scale.mean_, splits["X_tr"].mean(axis=0))


def test_ridge_models_use_fold_training_scaling():
    X = np.array([[1.0, 10], [2, 20], [3, 30], [4, 40]])
    config = {"kind": "ridge", "task": "regression", "implementation": "custom", "alpha": 1.0}
    model = evaluation.create_model(config, len(X))
    model.fit(X, [100, 200, 300, 400])
    assert isinstance(model.ridge, RidgeRegression)
    np.testing.assert_allclose(model.scale.mean_, [2.5, 25])
    before = model.scale.mean_.copy()
    model.predict([[1e6, -1e6]])
    np.testing.assert_array_equal(model.scale.mean_, before)


def test_one_command_pipeline_end_to_end(cleaned_frame, tmp_path, monkeypatch):
    raw = tmp_path / "input.csv"
    cleaned_frame.to_csv(raw, index=False)
    before = raw.read_bytes()
    monkeypatch.setattr(data_prep, "clean", lambda frame: frame.copy())
    output = tmp_path / "results"
    result = pipeline.run_pipeline(raw, output, include_cv=False, search_svm=False,
                                   depths=(1,), leaf_sizes=(2,))
    assert len(result["test"]) == 10
    assert {row["model"] for row in result["test"]} >= {"custom_ridge", "sklearn_ridge"}
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["ridge_included"] and not manifest["cv_requested"]
    assert (output / "test_metrics.csv").is_file()
    assert (output / "figures/eda/price_distribution.png").is_file()
    assert (output / "figures/tree_classification_validation.png").is_file()
    assert raw.read_bytes() == before
    metrics = pd.read_csv(output / "test_metrics.csv").set_index("model")
    assert metrics.loc["custom_ridge", "rmse"] == pytest.approx(metrics.loc["sklearn_ridge", "rmse"], rel=1e-8)
    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(raw, output, include_cv=False, search_svm=False)


def test_default_pipeline_requests_search_cv_and_ridge(cleaned_frame, tmp_path, monkeypatch):
    raw = tmp_path / "raw.csv"
    cleaned_frame.to_csv(raw, index=False)
    monkeypatch.setattr(data_prep, "clean", lambda frame: frame.copy())
    called = {}
    real_run = pipeline.run_evaluation

    def search(arrays, metadata, configs, output):
        called["search"] = len(configs)
        from experiments.svm_experiments import FAMILIES
        selected = {family: {**next(config for config in configs if config["family"] == family), "split": "validation"} for family in FAMILIES}
        output.mkdir(parents=True)
        evaluation.write_json(output / "configurations.json", {"status": "complete", "dataset": metadata,
                              "source_sha256": {"src/svm.py": pipeline.sha256("src/svm.py")}})
        evaluation.write_json(output / "selected_configurations.json", selected)
        return {"selected": selected}

    def checked_run(clean, output, **kwargs):
        called["cv"] = kwargs["include_cv"]
        called["ridge"] = kwargs["include_ridge"]
        assert len(kwargs["svm_configs"]) == 4
        kwargs["include_cv"] = False
        return real_run(clean, output, **kwargs)

    monkeypatch.setattr(pipeline, "run_experiments", search)
    monkeypatch.setattr(pipeline, "run_evaluation", checked_run)
    output = tmp_path / "out"
    pipeline.run_pipeline(raw, output, depths=(1,), leaf_sizes=(2,))
    assert called == {"search": 33, "cv": True, "ridge": True}
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "failed"
    evaluation.write_json(output / "run_manifest.json", manifest)
    monkeypatch.setattr(pipeline, "run_experiments", lambda *args, **kwargs: pytest.fail("Completed search should not repeat"))
    pipeline.run_pipeline(raw, output, depths=(1,), leaf_sizes=(2,), resume=True)
    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(raw, output, depths=(1,), leaf_sizes=(2,), resume=True)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "failed"
    manifest["raw_sha256"] = "wrong"
    evaluation.write_json(output / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="different raw"):
        pipeline.run_pipeline(raw, output, resume=True)


def test_missing_raw_data_creates_no_fake_results(tmp_path):
    output = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline(tmp_path / "missing.csv", output)
    assert not output.exists()


def test_main_interface_and_missing_data_exit(tmp_path):
    command = [sys.executable, "-m", "src.run_all"]
    help_result = subprocess.run(command + ["--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "--skip-cv" in help_result.stdout
    failed = subprocess.run(command + ["--raw", str(tmp_path / "missing.csv"), "--output", str(tmp_path / "out")],
                            capture_output=True, text=True)
    assert failed.returncode != 0
    assert not (tmp_path / "out").exists()
