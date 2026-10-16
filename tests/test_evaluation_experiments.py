import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from sklearn import metrics

from src import data_prep
from src.decision_tree import DecisionTree
from experiments import evaluation_experiments as evaluation


@pytest.fixture
def frame():
    rng = np.random.default_rng(11)
    result = pd.DataFrame({name: rng.uniform(1, 10, 200) for name in data_prep.NUMERIC_FEATURES})
    result["price"] = np.arange(200) * 1000 + 50000
    result["area"] = result["price"] / 2000
    result["category"] = np.resize(list(data_prep.CATEGORY_MAP.values()), 200)
    result["location"] = np.resize(["A", "B", "C"], 200)
    result["lat"] = rng.uniform(40.3, 40.5, 200)
    result["lng"] = rng.uniform(49.7, 49.9, 200)
    return result


def test_alignment_uses_exact_shared_split(frame, tmp_path):
    path = tmp_path / "clean.csv"
    frame.to_csv(path, index=False)
    loaded, splits, ids = evaluation.load_data(path)
    assert sorted(np.concatenate(ids).tolist()) == list(range(len(frame)))
    for short, positions in zip(("tr", "val", "te"), ids):
        np.testing.assert_array_equal(splits[f"y_{short}"], loaded.iloc[positions]["price"].to_numpy())
        np.testing.assert_allclose(splits[f"X_{short}"], splits["encoder"].transform(loaded.iloc[positions]))


def test_tree_score_column_handles_single_class():
    X = np.array([[0.0], [1.0]])
    for label in (0, 1):
        tree = DecisionTree().fit(X, [label, label])
        np.testing.assert_array_equal(evaluation.classification_scores(tree, X), [label, label])


def test_metric_rows_match_sklearn():
    regression = evaluation.metric_row("regression", [100, 200, 300], [110, 180, 330])
    assert regression["rmse"] == pytest.approx(np.sqrt(metrics.mean_squared_error([100, 200, 300], [110, 180, 330])))
    row = evaluation.metric_row("classification", [0, 0, 1, 1], [0, 1, 0, 1], [-2, 1, -1, 2])
    assert row["f1"] == pytest.approx(metrics.f1_score([0, 0, 1, 1], [0, 1, 0, 1]))
    assert row["roc_auc"] == pytest.approx(metrics.roc_auc_score([0, 0, 1, 1], [-2, 1, -1, 2]))
    assert [row[key] for key in ("tn", "fp", "fn", "tp")] == [1, 1, 1, 1]


def test_saved_svm_winners_preserve_original_parameters(frame, tmp_path):
    path = tmp_path / "clean.csv"
    frame.to_csv(path, index=False)
    _, splits, _ = evaluation.load_data(path)
    configs, rows = evaluation.select_configs(splits, depths=(1,), leaf_sizes=(2,))
    assert len(configs) == 8
    assert {row["task"] for row in rows} == {"classification", "regression"}
    assert configs["custom_linear"]["config"]["learning_rate"] == "constant"
    assert configs["custom_linear"]["config"]["lambda_"] == 1e-4
    assert configs["custom_rff"]["config"]["n_components"] == 512
    assert configs["custom_rff"]["config"]["gamma"] == 0.01


def test_cv_never_fits_final_test_rows_and_refits_encoder(frame, monkeypatch):
    train, validation, test = data_prep._split_indices(frame["price"].to_numpy())
    frame.loc[test, "location"] = "FINAL_TEST_ONLY"
    fitted_frames = []
    original_fit = data_prep.FeatureEncoder.fit

    def checked_fit(encoder, fold_frame):
        assert "FINAL_TEST_ONLY" not in set(fold_frame["location"])
        fitted_frames.append(fold_frame.copy())
        return original_fit(encoder, fold_frame)

    def fake_evaluate(configs, X_train, train_prices, train_tiers, X_val, val_prices, val_tiers):
        assert np.median(train_prices) == np.median(fitted_frames[-1]["price"])
        np.testing.assert_array_equal(train_tiers, train_prices > np.median(train_prices))
        np.testing.assert_array_equal(val_tiers, val_prices > np.median(train_prices))
        assert not set(train_prices) & set(val_prices)
        return [{"model": "tree", "task": "regression", "rmse": 1.0, "mae": 1.0, "r2": 0.5}], {}, {}, {}

    monkeypatch.setattr(data_prep.FeatureEncoder, "fit", checked_fit)
    monkeypatch.setattr(evaluation, "evaluate_models", fake_evaluate)
    rows, summary, audit = evaluation.cross_validate(frame, np.concatenate([train, validation]), {"tree": {"task": "regression"}})
    assert len(fitted_frames) == len(rows) == len(audit) == 5
    assert all(row["folds"] == 5 and row["std"] == 0 for row in summary)


def test_real_apis_end_to_end_on_development_fixture(frame, tmp_path):
    path = tmp_path / "clean.csv"
    frame.to_csv(path, index=False)
    source = path.read_bytes()
    output = tmp_path / "evaluation"
    result = evaluation.run_evaluation(path, output, include_cv=False, depths=(1,), leaf_sizes=(2,))
    assert len(result["test"]) == 8
    assert result["manifest"]["test_used_for_selection"] is False
    assert result["manifest"]["cv_complete"] is False
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tier_threshold"] > 0
    assert len(list((output / "figures").glob("*.png"))) == 14
    groups = pd.read_csv(output / "error_groups.csv")
    for name in groups["model"].unique():
        subset = groups[groups["model"] == name]
        for group_type in ("price", "representation", "location"):
            assert subset[subset["group_type"] == group_type]["count"].sum() == manifest["split_counts"]["test"]
    assert path.read_bytes() == source
    with pytest.raises(FileExistsError):
        evaluation.run_evaluation(path, output, include_cv=False)
