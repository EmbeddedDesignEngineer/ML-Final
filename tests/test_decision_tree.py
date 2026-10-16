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


@pytest.mark.parametrize("criterion, impurity", [("gini", .5), ("entropy", 1.)])
def test_criteria_have_known_root_impurity(criterion, impurity):
    model = DecisionTree(criterion=criterion, max_depth=1).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert model.root.impurity == pytest.approx(impurity)
    assert_array_equal(model.predict([[0], [3]]), [0, 1])


@pytest.mark.parametrize("params", [
    {"max_depth": 0}, {"min_samples_split": 5},
    {"min_samples_leaf": 3}, {"min_impurity_decrease": .51},
])
def test_each_stopping_parameter_can_prevent_a_split(params):
    X, y = [[0], [1], [2], [3]], [0, 0, 1, 1]
    assert not DecisionTree().fit(X, y).root.is_leaf()
    assert DecisionTree(**params).fit(X, y).root.is_leaf()


def test_impurity_decrease_uses_fraction_of_all_training_samples():
    # A small global minimum still permits the useful descendant splits.
    model = DecisionTree(min_impurity_decrease=.1).fit([[0], [1], [2], [3]], [0, 1, 1, 0])
    assert not model.root.is_leaf()
    assert_array_equal(model.predict([[0], [1], [2], [3]]), [0, 1, 1, 0])


def test_min_leaf_rejects_invalid_optimum_but_searches_valid_alternative():
    model = DecisionTree(min_samples_leaf=2, max_depth=1).fit(
        [[0], [1], [2], [3], [4]], [0, 1, 1, 1, 1])
    assert model.root.threshold == 1.5
    assert model.root.left.n_samples == 2
    assert model.root.right.n_samples == 3


def test_deeper_split_is_stopped_by_global_not_local_gain():
    model = DecisionTree(task="regression", criterion="mse", min_impurity_decrease=.5)
    model.fit(np.arange(6).reshape(-1, 1), [0, 0, 0, 0, 10, 12])
    assert model.root.threshold == 3.5
    assert model.root.right.impurity == pytest.approx(1.)
    assert model.root.right.is_leaf()  # global gain = (2/6)*1 < .5
    assert_allclose(model.predict([[4], [5]]), [11, 11])


def test_importance_is_normalized_and_refit_resets_metadata():
    model = DecisionTree().fit([[0, 4], [1, 4], [2, 4], [3, 4]], [0, 0, 1, 1])
    assert_allclose(model.feature_importances_, [1., 0.])
    assert model.get_depth() == 1
    assert model.get_n_leaves() == 2
    model.fit([[5], [5]], [1, 1])
    assert_allclose(model.feature_importances_, [0.])
    assert_array_equal(model.classes_, [1])
    assert model.get_depth() == 0
    assert model.get_n_leaves() == 1


@pytest.mark.parametrize("params", [
    {"task": "bad"}, {"criterion": "bad"}, {"task": "regression"},
    {"max_depth": -1}, {"max_depth": 1.5}, {"max_depth": True},
    {"min_samples_split": 1}, {"min_samples_leaf": 0},
    {"min_impurity_decrease": -1}, {"min_impurity_decrease": float("nan")},
])
def test_invalid_parameters_fail_clearly(params):
    with pytest.raises(ValueError):
        DecisionTree(**params).fit([[0], [1]], [0, 1])


@pytest.mark.parametrize("X,y", [
    ([], []), ([1, 2], [0, 1]), ([[0], [1]], [1]),
    ([[0], [1]], [[0], [1]]), ([[float("nan")]], [0]),
    ([[float("inf")]], [0]), ([[0], [1]], [0, float("nan")]),
])
def test_invalid_training_arrays_fail_clearly(X, y):
    with pytest.raises(ValueError):
        DecisionTree().fit(X, y)


def test_prediction_checks_and_empty_batches():
    model = DecisionTree()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict([[0]])
    model.fit([[0], [1]], [0, 1])
    assert model.predict(np.empty((0, 1))).shape == (0,)
    assert model.predict_proba(np.empty((0, 1))).shape == (0, 2)
    for X in ([[0, 1]], [0], [[float("nan")]]):
        with pytest.raises(ValueError):
            model.predict(X)


def test_adjacent_floats_split_without_empty_child():
    low = 1.
    high = np.nextafter(low, np.inf)
    model = DecisionTree().fit([[low], [high]], [0, 1])
    assert_array_equal(model.predict([[low], [high]]), [0, 1])


@pytest.mark.parametrize("task,criterion", [("classification", "gini"), ("classification", "entropy"), ("regression", "mse")])
def test_zero_gain_bridge_can_learn_xor(task, criterion):
    X, y = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]
    model = DecisionTree(task=task, criterion=criterion, max_depth=2).fit(X, y)
    assert_array_equal(model.predict(X), y)
    assert model.get_depth() == 2
    blocked = DecisionTree(task=task, criterion=criterion, min_impurity_decrease=.01).fit(X, y)
    assert blocked.root.is_leaf()


def test_unlimited_depth_handles_pathological_unbalanced_tree():
    X = np.arange(1100.).reshape(-1, 1)
    y = np.arange(1100) % 2
    model = DecisionTree().fit(X, y)
    assert_array_equal(model.predict(X), y)
    assert model.get_depth() > 1000
