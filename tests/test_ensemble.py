import numpy as np
import pytest
from numpy.testing import assert_allclose
from src.ensemble import BaggedTreeRegressor

def test_bagged_trees_are_seeded_and_average_staged_predictions():
    rng=np.random.default_rng(4); X=rng.normal(size=(80,4)); y=X[:,0]**2+X[:,1]
    args=dict(n_estimators=5,max_depth=4,min_samples_leaf=2,random_state=42)
    model=BaggedTreeRegressor(**args).fit(X,y)
    repeated=BaggedTreeRegressor(**args).fit(X,y)
    assert_allclose(model.predict(X),repeated.predict(X))
    assert_allclose(list(model.staged_predict(X))[-1],model.predict(X))
    assert len(model.estimators_)==5
    assert_allclose(model.feature_importances_.sum(),1)
    assert model.predict(np.empty((0,4))).shape==(0,)

def test_constant_targets_and_invalid_shapes():
    model=BaggedTreeRegressor(n_estimators=2).fit([[0,0],[1,1],[2,2]],[5,5,5])
    assert_allclose(model.predict([[3,3]]),[5])
    assert_allclose(model.feature_importances_,[0,0])
    with pytest.raises(ValueError): model.predict([[1]])
    with pytest.raises(RuntimeError): BaggedTreeRegressor().predict([[1]])

@pytest.mark.parametrize('params',[{'n_estimators':0},{'max_samples':0},{'max_features':1.1},{'random_state':-1}])
def test_invalid_settings(params):
    with pytest.raises(ValueError): BaggedTreeRegressor(**params).fit([[0],[1]],[0,1])
