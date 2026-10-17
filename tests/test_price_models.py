import numpy as np
import pytest
from src.price_models import PriceRegressor, BoostedTreeRegressor


def test_price_per_area_round_trip_and_unseen_rows():
    X = np.array([[20., 1], [40., 1], [80., 1]])
    from src.ensemble import BaggedTreeRegressor
    model = PriceRegressor(BaggedTreeRegressor(n_estimators=2), target='log_price_per_area', area_index=0)
    model.fit(X, 1000 * X[:, 0])
    np.testing.assert_allclose(model.predict([[60., 1]]), [60000])
    with pytest.raises(ValueError):
        model.predict([[0., 1]])


def test_boosting_learns_residuals_and_stages_match():
    X = np.arange(40.).reshape(-1, 1)
    y = 100 + X[:, 0] * 2
    model = BoostedTreeRegressor(n_estimators=30, max_depth=2, min_samples_leaf=2).fit(X, y)
    stages = list(model.staged_predict(X))
    assert np.mean((stages[-1] - y) ** 2) < np.mean((stages[0] - y) ** 2) / 10
    np.testing.assert_allclose(model.predict(X), stages[-1])
    np.testing.assert_allclose(BoostedTreeRegressor(n_estimators=30, max_depth=2, min_samples_leaf=2).fit(X, y).predict(X), stages[-1])


@pytest.mark.parametrize('params', [{'n_estimators':0}, {'learning_rate':0}, {'learning_rate':float('nan')}])
def test_boosting_rejects_invalid_parameters(params):
    with pytest.raises(ValueError):
        BoostedTreeRegressor(**params).fit([[1], [2]], [1, 2])


def test_failed_refit_preserves_previous_boosting_predictions():
    model = BoostedTreeRegressor(n_estimators=2, min_samples_leaf=1).fit([[1], [2]], [10, 20])
    previous = model.predict([[1], [2]])
    model.max_depth = -1
    with pytest.raises(ValueError):
        model.fit([[1], [2]], [1000, 2000])
    np.testing.assert_allclose(model.predict([[1], [2]]), previous)
