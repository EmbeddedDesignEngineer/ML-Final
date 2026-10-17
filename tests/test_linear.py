import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.linear import RidgeRegression


@pytest.mark.parametrize("alpha", [0.0, 0.1, 1.0, 100.0])
@pytest.mark.parametrize("intercept", [True, False])
def test_matches_sklearn(alpha, intercept):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 4)) + 3
    y = 500 + X @ np.array([2, -3, 0.5, 1]) + rng.normal(size=100)
    custom = RidgeRegression(alpha=alpha, fit_intercept=intercept).fit(X, y)
    reference = Ridge(alpha=alpha, fit_intercept=intercept, solver="svd").fit(X, y)
    np.testing.assert_allclose(custom.coef_, reference.coef_, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(custom.predict(X), reference.predict(X), rtol=1e-8, atol=1e-8)
    assert custom.intercept_ == pytest.approx(reference.intercept_)


def test_bias_is_not_penalized():
    X = np.zeros((5, 2))
    model = RidgeRegression(alpha=1e8).fit(X, [10, 20, 30, 40, 50])
    np.testing.assert_array_equal(model.coef_, [0, 0])
    np.testing.assert_array_equal(model.predict(X), np.full(5, 30))


def test_singular_unregularized_data_and_empty_prediction():
    X = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)
    model = RidgeRegression(alpha=0).fit(X, [2, 4, 6])
    np.testing.assert_allclose(model.predict(X), [2, 4, 6], atol=1e-12)
    assert model.predict(np.empty((0, 2))).shape == (0,)


def test_regularization_reduces_weight_norm():
    X = np.arange(10.0).reshape(-1, 1)
    y = 10 * X[:, 0] + 3
    weak = RidgeRegression(alpha=0.1).fit(X, y)
    strong = RidgeRegression(alpha=100).fit(X, y)
    assert np.linalg.norm(strong.coef_) < np.linalg.norm(weak.coef_)


@pytest.mark.parametrize("alpha", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        RidgeRegression(alpha).fit([[1], [2]], [1, 2])


@pytest.mark.parametrize("X,y", [([], []), ([1, 2], [1, 2]), ([[1], [2]], [1]),
                                  ([[1], [float("nan")]], [1, 2]), ([[1], [2]], [1, float("inf")]),
                                  ([["1"], ["2"]], [1, 2])])
def test_invalid_training_inputs(X, y):
    with pytest.raises(ValueError):
        RidgeRegression().fit(X, y)


def test_prediction_validation():
    with pytest.raises(RuntimeError):
        RidgeRegression().predict([[1]])
    model = RidgeRegression().fit([[1], [2]], [1, 2])
    with pytest.raises(ValueError):
        model.predict([[1, 2]])
    with pytest.raises(ValueError):
        model.predict([[float("nan")]])


@pytest.mark.parametrize('scale,alpha', [(1e8,1.),(1.,1e-20)])
def test_collinear_positive_ridge_remains_solvable(scale,alpha):
    X=scale*np.array([[-1.,-1.],[0.,0.],[1.,1.]])
    y=np.array([-2.,0.,2.])
    model=RidgeRegression(alpha=alpha).fit(X,y)
    reference=Ridge(alpha=alpha,solver='svd').fit(X,y)
    np.testing.assert_allclose(model.predict(X),reference.predict(X),atol=1e-10)
