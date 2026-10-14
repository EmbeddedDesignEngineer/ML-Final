"""SVM export integration checks using the team's existing preprocessing."""

import copy

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from experiments.prepare_svm_data import export_prepared
from experiments.svm_experiments import load_prepared_data, validate_prepared_data
from src import data_prep


@pytest.fixture
def cleaned_frame():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({name: rng.uniform(1, 10, 400) for name in data_prep.NUMERIC_FEATURES})
    frame["price"] = np.arange(400) * 1000 + 20000
    frame["category"] = np.resize(list(data_prep.CATEGORY_MAP.values()), 400)
    frame["location"] = ["common"] * 320 + ["rare"] * 80
    frame["lat"] = rng.uniform(40.3, 40.5, 400)
    frame["lng"] = rng.uniform(49.7, 49.9, 400)
    return frame


def test_export_matches_shared_pipeline_and_preserves_source(cleaned_frame, tmp_path):
    clean = tmp_path / "clean.csv"
    cleaned_frame.to_csv(clean, index=False)
    original = clean.read_bytes()
    output = tmp_path / "prepared.npz"
    arrays, metadata = export_prepared(clean, output)
    loaded, loaded_metadata = load_prepared_data(output)
    audit = validate_prepared_data(loaded, loaded_metadata)
    assert audit["listing_ids_disjoint"] is True
    assert metadata["frequent_locations"] == ["common"]
    assert metadata["fixed_categories"] == list(data_prep.CATEGORY_MAP.values())
    assert metadata["encoding_fit_split"] == metadata["scaling_fit_split"] == "train"
    sp = data_prep.get_splits(path=str(clean))
    assert metadata["tier_threshold"] == np.median(sp["y_tr"])
    for key in arrays:
        assert_array_equal(arrays[key], loaded[key])
    assert clean.read_bytes() == original
    with pytest.raises(FileExistsError, match="overwrite"):
        export_prepared(clean, output)


def test_held_out_features_and_targets_cannot_change_training_encoding(cleaned_frame):
    train = cleaned_frame.iloc[:250]
    encoder = data_prep.FeatureEncoder().fit(train)
    before, state = encoder.transform(train), copy.deepcopy(vars(encoder))
    held_out = cleaned_frame.iloc[250:].copy()
    # Even an arbitrarily frequent unseen held-out location stays in other.
    held_out["location"] = "new-validation-location"
    held_out["price"] = 1e10
    repeated = pd.concat([held_out] * 5, ignore_index=True)
    transformed = encoder.transform(repeated)
    assert vars(encoder) == state
    assert_array_equal(encoder.transform(train), before)
    assert_array_equal(transformed[:, -1], np.ones(len(repeated)))
    assert "loc_new-validation-location" not in encoder.feature_names_
    repeated["price"] = -999
    assert_array_equal(encoder.transform(repeated), transformed)


def test_shared_get_splits_fits_encoder_only_once_on_training_rows(cleaned_frame, tmp_path, monkeypatch):
    path = tmp_path / "clean.csv"
    cleaned_frame.to_csv(path, index=False)
    expected_ids = data_prep._split_indices(cleaned_frame["price"].to_numpy(), seed=42)[0]
    calls = []
    original_fit = data_prep.FeatureEncoder.fit

    def checked_fit(encoder, frame):
        assert_array_equal(frame["price"].to_numpy(), cleaned_frame.iloc[expected_ids]["price"].to_numpy())
        calls.append(len(frame))
        return original_fit(encoder, frame)

    monkeypatch.setattr(data_prep.FeatureEncoder, "fit", checked_fit)
    splits = data_prep.get_splits(path=str(path))
    assert calls == [len(expected_ids)]
    assert splits["tier_threshold"] == np.median(cleaned_frame.iloc[expected_ids]["price"])


def test_export_is_reproducible(cleaned_frame, tmp_path):
    path = tmp_path / "clean.csv"
    cleaned_frame.to_csv(path, index=False)
    first, first_meta = export_prepared(path, tmp_path / "first.npz")
    second, second_meta = export_prepared(path, tmp_path / "second.npz")
    assert first_meta == second_meta
    for key in first:
        assert_array_equal(first[key], second[key])
