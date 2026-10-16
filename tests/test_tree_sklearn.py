"""Independent reference checks; sklearn is never used by the custom model."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.decision_tree import DecisionTree


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_unique_classification_split_matches_sklearn(criterion):
    X = np.array([[0, 7], [1, 7], [2, 7], [3, 7]], dtype=float)
    y = np.array([0, 0, 1, 1])
    ours = DecisionTree(criterion=criterion, max_depth=1).fit(X, y)
    ref = DecisionTreeClassifier(criterion=criterion, max_depth=1, random_state=42).fit(X, y)
    assert ours.root.feature == ref.tree_.feature[0]
    assert ours.root.threshold == ref.tree_.threshold[0]
    assert_allclose(ours.root.impurity, ref.tree_.impurity[0])
    assert_array_equal(ours.predict(X), ref.predict(X))
    assert_allclose(ours.predict_proba(X), ref.predict_proba(X))
    assert_allclose(ours.feature_importances_, ref.feature_importances_)


@pytest.mark.parametrize("depth", [1, 3, 6])
@pytest.mark.parametrize("min_leaf", [1, 4])
def test_regression_predictions_and_importance_match_sklearn(depth, min_leaf):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 3))
    y = 5 * X[:, 0] + X[:, 1] ** 2 + rng.normal(0, .1, 80)
    args = dict(max_depth=depth, min_samples_leaf=min_leaf, random_state=42)
    ours = DecisionTree(task="regression", criterion="mse", **args).fit(X, y)
    ref = DecisionTreeRegressor(**args).fit(X, y)
    assert ours.root.feature == ref.tree_.feature[0]
    assert ours.root.threshold == pytest.approx(ref.tree_.threshold[0], abs=1e-7)
    assert_allclose(ours.predict(X), ref.predict(X), atol=1e-12)
    # Two-point leaves can be separated equally by several features; sklearn
    # randomizes feature order, while ours uses the first feature on a tie.
    if min_leaf > 1 or depth < 6:
        assert_allclose(ours.feature_importances_, ref.feature_importances_, atol=1e-12)
    assert ours.get_depth() == ref.get_depth()
    assert ours.get_n_leaves() == ref.get_n_leaves()


def test_sample_weighted_importance_and_stopping_match_sklearn():
    X = np.column_stack((np.arange(6.), [0, 0, 0, 0, 0, 1]))
    y = [0, 0, 0, 0, 10, 12]
    for minimum in (0., .5):
        ours = DecisionTree(task="regression", criterion="mse", min_impurity_decrease=minimum).fit(X, y)
        ref = DecisionTreeRegressor(min_impurity_decrease=minimum, random_state=42).fit(X, y)
        assert_allclose(ours.predict(X), ref.predict(X))
        # With zero stopping threshold the last split can use tied features.
        assert ours.get_n_leaves() == ref.get_n_leaves()
