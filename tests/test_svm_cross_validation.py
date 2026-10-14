"""Small synthetic fixtures test CV isolation; they are not Bina.az results."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.pipeline import Pipeline

from experiments import svm_cross_validation as cv
from experiments.prepare_svm_data import file_sha256
from experiments.svm_experiments import build_configurations
from src import data_prep


@pytest.fixture
def clean_frame():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({name: rng.uniform(1, 10, 600) for name in data_prep.NUMERIC_FEATURES})
    frame["price"] = 20000 + np.arange(600) * 1000
    frame["area_m2"] = np.arange(600) + 20.0
    frame["category"] = np.resize(list(data_prep.CATEGORY_MAP.values()), 600)
    frame["location"] = ["common"] * 400 + ["rare"] * 200
    frame["lat"] = rng.uniform(40.3, 40.5, 600)
    frame["lng"] = rng.uniform(49.7, 49.9, 600)
    return frame


@pytest.fixture
def small_configs():
    grid = build_configurations(lambdas=(0.01,), iterations=(50,), gammas=(0.01,),
                                components=(16,), seed=42)
    return list({c["family"]: c for c in grid}.values())


def write_source(frame, tmp_path, ids=None):
    clean, prepared = tmp_path / "clean.csv", tmp_path / "ids.npz"
    frame.to_csv(clean, index=False)
    if ids is None:
        ids = {"train_ids": np.arange(350), "val_ids": np.arange(350, 500),
               "test_ids": np.arange(500, 600)}
    # Deliberately no X/y arrays: the loader must need ONLY membership/provenance.
    np.savez(prepared, **ids, metadata_json=json.dumps({"cleaned_sha256": file_sha256(clean)}))
    return clean, prepared


def test_loader_excludes_final_test_and_does_not_load_prepared_features(clean_frame, tmp_path):
    clean_frame.loc[500:, "price"] = np.nan  # Cannot reach CV target checks.
    clean_frame.loc[500:, "location"] = "FINAL-TEST-ONLY"
    clean, prepared = write_source(clean_frame, tmp_path)
    development, excluded, provenance = cv.load_development_data(clean, prepared)
    assert_array_equal(development.index, np.arange(500))
    assert_array_equal(excluded, np.arange(500, 600))
    assert development["price"].notna().all()
    assert "FINAL-TEST-ONLY" not in set(development.location)
    assert provenance["final_test_excluded"] is True


def test_loader_rejects_overlap_and_changed_dataset(clean_frame, tmp_path):
    bad_ids = {"train_ids": np.arange(351), "val_ids": np.arange(350, 500),
               "test_ids": np.arange(500, 600)}
    clean, prepared = write_source(clean_frame, tmp_path, bad_ids)
    with pytest.raises(ValueError, match="disjoint"):
        cv.load_development_data(clean, prepared)
    clean, prepared = write_source(clean_frame, tmp_path)
    clean.write_text(clean.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance"):
        cv.load_development_data(clean, prepared)


def test_folds_are_disjoint_exhaustive_and_seeded(clean_frame):
    prices = clean_frame.price.to_numpy()
    folds = cv.make_folds(prices)
    repeated = cv.make_folds(prices)
    assert len(folds) == 5
    assert_array_equal(np.sort(np.concatenate([val for _, val in folds])), np.arange(len(prices)))
    for (train, val), (train2, val2) in zip(folds, repeated):
        assert not set(train) & set(val)
        assert set(train) | set(val) == set(range(len(prices)))
        assert_array_equal(train, train2)
        assert_array_equal(val, val2)
    assert not np.array_equal(folds[0][1], cv.make_folds(prices, seed=43)[0][1])


@pytest.mark.parametrize("prices", [[], [np.nan] * 100, np.arange(20)])
def test_invalid_fold_prices_raise(prices):
    with pytest.raises(ValueError):
        cv.make_folds(prices)


def test_imputation_threshold_and_scaling_have_manual_training_values(clean_frame):
    train, val = clean_frame.iloc[:4].copy(), clean_frame.iloc[4:6].copy()
    train["area_m2"] = [10.0, np.nan, 30.0, 50.0]
    train["price"] = [10, 20, 30, 40]
    val["area_m2"] = [np.nan, 1000.0]
    val["price"] = [15, 35]
    arrays, audit = cv.prepare_fold(train, val)
    assert audit["tier_threshold"] == 25.0
    assert audit["numeric_imputation_medians"]["area_m2"] == 30.0
    assert audit["imputed_train_cells"] == audit["imputed_validation_cells"] == 1
    assert_array_equal(arrays["unscaled_train"][:, 0], [10, 30, 30, 50])
    assert_array_equal(arrays["unscaled_val"][:, 0], [30, 1000])
    assert_allclose(arrays["X_train"][:, 0], np.array([-20, 0, 0, 20]) / np.sqrt(200))
    assert_allclose(arrays["X_val"][:, 0], np.array([0, 970]) / np.sqrt(200))
    assert_array_equal(arrays["y_train"], [0, 0, 1, 1])
    assert_array_equal(arrays["y_val"], [0, 1])
    assert np.isnan(train.area_m2.iloc[1])  # Source is never modified.


def test_validation_changes_cannot_change_training_preprocessing(clean_frame):
    train = clean_frame.iloc[:400].copy()
    val = clean_frame.iloc[400:].copy()
    val["price"] = np.resize([10000, 1000000], len(val))
    before, audit = cv.prepare_fold(train, val)
    # This location reaches the frequency cutoff only in held-out rows.
    assert audit["frequent_locations"] == ["common"]
    assert "loc_rare" not in audit["feature_names"]
    changed = pd.concat([val] * 3, ignore_index=True)
    changed["location"] = "new-held-out-location"
    changed["area_m2"] = np.resize([np.nan, 1e9], len(changed))
    changed["price"] = np.resize([1, 1e12], len(changed))
    after, new_audit = cv.prepare_fold(train, changed)
    assert_array_equal(before["X_train"], after["X_train"])
    assert_array_equal(before["y_train"], after["y_train"])
    for key in ("tier_threshold", "frequent_locations", "feature_names", "numeric_imputation_medians",
                "scaler_mean", "scaler_std"):
        assert audit[key] == new_audit[key]
    assert_array_equal(after["unscaled_val"][:, -1], np.ones(len(changed)))


@pytest.mark.parametrize("problem", ["all_missing", "infinity", "missing_target", "one_class"])
def test_bad_fold_data_fails_explicitly(clean_frame, problem):
    train, val = clean_frame.iloc[:400].copy(), clean_frame.iloc[400:].copy()
    val["price"] = np.resize([10000, 1000000], len(val))
    if problem == "all_missing":
        train["area_m2"] = np.nan
    elif problem == "infinity":
        val["area_m2"] = np.inf
    elif problem == "missing_target":
        val.loc[val.index[0], "price"] = np.nan
    else:
        val["price"] = 1
    with pytest.raises(ValueError):
        cv.prepare_fold(train, val)


def test_fixed_configuration_loader_ignores_metrics_and_rejects_test_selection(small_configs, tmp_path):
    path = tmp_path / "selected.json"
    records = {c["family"]: {**c, "split": "validation", "f1": 0.999} for c in small_configs}
    path.write_text(json.dumps(records))
    assert cv.load_fixed_configurations(path) == small_configs
    records["custom_linear"]["split"] = "test"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="validation selections"):
        cv.load_fixed_configurations(path)


def test_summary_uses_sample_standard_deviation():
    rows = [{"family": family, "fold": fold, **{m: value for m in cv.METRICS}}
            for family in cv.FAMILIES for fold, value in enumerate([0.5, 0.6, 0.7, 0.8, 0.9], 1)]
    for row in cv.summarize_folds(rows):
        for metric in cv.METRICS:
            assert row[f"{metric}_mean"] == pytest.approx(0.7)
            assert row[f"{metric}_std"] == pytest.approx(np.sqrt(0.025))
    with pytest.raises(ValueError, match="five distinct"):
        cv.summarize_folds(rows[:-1])


def test_cv_fits_and_evaluates_only_the_correct_fold(clean_frame, small_configs, tmp_path, monkeypatch):
    development = clean_frame.iloc[:500].copy()
    folds = cv.make_folds(development.price.to_numpy())
    expected = [cv.prepare_fold(development.iloc[tr], development.iloc[va])[0] for tr, va in folds]
    fit_calls, score_calls, encoder_calls = [], [], []
    original_model, original_evaluate = cv.make_model, cv.evaluate_model
    original_encoder_fit = data_prep.FeatureEncoder.fit

    def checked_encoder_fit(encoder, frame):
        fold = len(encoder_calls)
        assert_array_equal(frame.index, development.iloc[folds[fold][0]].index)
        encoder_calls.append(frame.index.tolist())
        return original_encoder_fit(encoder, frame)

    def checked_model(config, n_train):
        estimator = original_model(config, n_train)
        fit = estimator.fit

        def checked_fit(X, y):
            fold = len(fit_calls) // 4
            assert len(X) == n_train == len(folds[fold][0])
            assert_allclose(X, expected[fold]["X_train"], atol=1e-12)
            assert_array_equal(y, expected[fold]["y_train"])
            if hasattr(estimator, "C"):
                assert estimator.C == pytest.approx(1 / (n_train * config["lambda_"]))
            fit_calls.append(config["family"])
            return fit(X, y)

        estimator.fit = checked_fit
        return estimator

    def checked_evaluate(model, X, y):
        fold = len(score_calls) // 4
        key = "unscaled_val" if isinstance(model, Pipeline) else "X_val"
        assert_allclose(X, expected[fold][key])
        assert_array_equal(y, expected[fold]["y_val"])
        score_calls.append(len(X))
        return original_evaluate(model, X, y)

    monkeypatch.setattr(data_prep.FeatureEncoder, "fit", checked_encoder_fit)
    monkeypatch.setattr(cv, "make_model", checked_model)
    monkeypatch.setattr(cv, "evaluate_model", checked_evaluate)
    (tmp_path / "metrics.csv").write_text("Existing final-test results must survive.\n")
    configs_before = copy.deepcopy(small_configs)
    result = cv.run_cross_validation(development, np.arange(500, 600), small_configs, tmp_path, verbose=False)
    assert len(fit_calls) == len(score_calls) == len(result["metrics"]) == 20
    assert len(encoder_calls) == len(result["audits"]) == 5
    assert small_configs == configs_before
    assert result["manifest"]["status"] == "complete"
    assert all((tmp_path / name).is_file() for name in cv.OUTPUT_FILES)
    assert (tmp_path / "metrics.csv").read_text() == "Existing final-test results must survive.\n"
    assignments = pd.read_csv(tmp_path / "cv_assignments.csv")
    assert set(assignments.cleaned_row_id) == set(range(500))
    assert set(assignments.validation_fold) == {1, 2, 3, 4, 5}
    assert len(pd.read_csv(tmp_path / "cv_fold_metrics.csv")) == 20
    with pytest.raises(FileExistsError):
        cv.run_cross_validation(development, [], small_configs, tmp_path, verbose=False)


def test_cv_reproducibility_and_test_overlap_guard(clean_frame, small_configs, tmp_path):
    development = clean_frame.iloc[:500]
    with pytest.raises(ValueError, match="exclude final-test"):
        cv.run_cross_validation(development, [3], small_configs, tmp_path / "invalid", verbose=False)
    first = cv.run_cross_validation(development, np.arange(500, 600), small_configs, tmp_path / "first", verbose=False)
    second = cv.run_cross_validation(development, np.arange(500, 600), small_configs, tmp_path / "second", verbose=False)
    assert first["summary"] == second["summary"]
    assert first["audits"] == second["audits"]
    for one, two in zip(first["metrics"], second["metrics"]):
        assert {k: v for k, v in one.items() if not k.endswith("_seconds")} == {
            k: v for k, v in two.items() if not k.endswith("_seconds")}
