"""Residual boosting and reversible price targets using our NumPy trees."""
import numpy as np
from .decision_tree import DecisionTree


def _features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError('X must be a finite 2-D feature matrix')
    return X


class PriceRegressor:
    def __init__(self, estimator, target='price', area_index=None):
        self.estimator = estimator
        self.target = target
        self.area_index = area_index

    def _area(self, X):
        if (isinstance(self.area_index, bool) or not isinstance(self.area_index, (int, np.integer))
                or not 0 <= self.area_index < X.shape[1]):
            raise ValueError('area_index must identify the area feature')
        area = X[:, self.area_index]
        if np.any(area <= 0):
            raise ValueError('Area must be positive')
        return area

    def fit(self, X, y):
        X = _features(X)
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or len(y) != len(X) or not len(y) or not np.isfinite(y).all() or np.any(y <= 0):
            raise ValueError('Prices must be finite and positive with one value per row')
        if self.target not in ('price', 'log_price', 'log_price_per_area'):
            raise ValueError('Unknown price target')
        target = y
        if self.target == 'log_price_per_area':
            target = target / self._area(X)
        if self.target != 'price':
            target = np.log(target)
        self.estimator.fit(X, target)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, 'n_features_in_'):
            raise RuntimeError('Call fit before predict')
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Feature count differs from training')
        area = self._area(X) if self.target == 'log_price_per_area' else 1
        prediction = self.estimator.predict(X)
        if self.target != 'price':
            with np.errstate(over='raise', invalid='raise'):
                prediction = np.exp(prediction) * area
        return prediction


class BoostedTreeRegressor:
    def __init__(self, n_estimators=100, learning_rate=.1, max_depth=4, min_samples_leaf=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = _features(X)
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or len(y) != len(X) or not len(y) or not np.isfinite(y).all():
            raise ValueError('Targets must be finite with one value per row')
        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, (int, np.integer)) or self.n_estimators < 1:
            raise ValueError('n_estimators must be a positive integer')
        if (isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float))
                or not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1):
            raise ValueError('learning_rate must be in (0, 1]')
        initial = float(np.mean(y))
        prediction = np.full(len(y), initial)
        estimators = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(task='regression', criterion='mse', max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf)
            tree.fit(X, y - prediction)
            prediction += self.learning_rate * tree.predict(X)
            estimators.append(tree)
        self.estimators_ = estimators
        self.initial_ = initial
        self.n_features_in_ = X.shape[1]
        return self

    def staged_predict(self, X):
        if not hasattr(self, 'estimators_'):
            raise RuntimeError('Call fit before predict')
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Feature count differs from training')
        prediction = np.full(len(X), self.initial_)
        for tree in self.estimators_:
            prediction += self.learning_rate * tree.predict(X)
            yield prediction.copy()

    def predict(self, X):
        for prediction in self.staged_predict(X):
            pass
        return prediction
