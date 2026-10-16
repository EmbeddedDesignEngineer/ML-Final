"""Public tree behavior on hand-calculated examples (no real-data claims)."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from src.decision_tree import DecisionTree


def test_gini_classifier_separates_numeric_threshold_and_returns_self():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    model = DecisionTree(max_depth=1)
    assert model.fit(X, [0, 0, 1, 1]) is model
    assert_array_equal(model.predict([[0], [1.5], [2], [5]]), [0, 0, 1, 1])
    assert_allclose(model.predict_proba([[0], [3]]), [[1, 0], [0, 1]])
    assert model.root.threshold == 1.5


def test_leaf_probabilities_and_ties_follow_sorted_classes():
    model = DecisionTree(max_depth=0).fit([[1], [2], [3], [4]], ["b", "a", "b", "a"])
    assert_array_equal(model.classes_, ["a", "b"])
    assert_array_equal(model.predict([[9]]), ["a"])
    assert_allclose(model.predict_proba([[9]]), [[.5, .5]])


def test_constant_features_and_single_class_are_leaves():
    model = DecisionTree().fit([[1], [1], [1]], [0, 1, 1])
    assert model.root.is_leaf()
    assert_allclose(model.predict_proba([[1]]), [[1 / 3, 2 / 3]])
    model.fit([[0], [1]], [7, 7])
    assert_array_equal(model.predict([[4]]), [7])
    assert_allclose(model.predict_proba([[4]]), [[1]])


def test_regression_uses_variance_splits_and_leaf_means():
    model = DecisionTree(task="regression", criterion="mse", max_depth=1)
    model.fit([[0], [1], [2], [3]], [10., 12., 30., 32.])
    assert_allclose(model.predict([[0], [3]]), [11., 31.])
    assert model.root.threshold == 1.5
    assert model.root.impurity == pytest.approx(101.)
    with pytest.raises(ValueError, match="classification"):
        model.predict_proba([[0]])


def test_regression_preserves_small_variation_at_large_price_offset():
    X = np.arange(4.).reshape(-1, 1)
    y = 1e12 + np.array([0., 2., 10., 12.])
    model = DecisionTree(task="regression", criterion="mse", max_depth=1).fit(X, y)
    assert model.root.threshold == 1.5
    assert_allclose(model.predict(X) - 1e12, [1., 1., 11., 11.])
