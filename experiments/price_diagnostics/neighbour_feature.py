"""Neighbour log-ppm feature, tested in study.py and rejected (worse in inner CV)."""
import numpy as np
from src.price_models import _features


class NeighbourPriceFeature:
    """Median log price per m2 of the k nearest training listings.

    Training rows get out-of-fold values so no row sees its own price. Rows of
    different groups (e.g. category one-hot columns) are never neighbours.
    """

    def __init__(self, area_index, lat_index, lng_index, k=10, group_indices=(), folds=5, random_state=42):
        self.area_index = area_index
        self.lat_index = lat_index
        self.lng_index = lng_index
        self.k = k
        self.group_indices = group_indices
        self.folds = folds
        self.random_state = random_state

    def _points(self, X):
        X = _features(X)
        lat, lng = X[:, self.lat_index], X[:, self.lng_index]
        points = [(lat - self.lat0_) * 111.0, (lng - self.lng0_) * 111.0 * np.cos(np.radians(self.lat0_))]
        if len(self.group_indices):
            points.append(np.argmax(X[:, list(self.group_indices)], axis=1) * 1e4)
        return np.column_stack(points)

    def _query(self, points, reference, values):
        k = min(self.k, len(reference))
        result = np.empty(len(points))
        squared = (reference ** 2).sum(axis=1)
        for start in range(0, len(points), 256):
            chunk = points[start:start + 256]
            distance = squared[None, :] - 2 * chunk @ reference.T
            nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
            result[start:start + 256] = np.median(values[nearest], axis=1)
        return result

    def fit_transform(self, X, y):
        X = _features(X)
        y = np.asarray(y, dtype=float)
        if isinstance(self.k, bool) or not isinstance(self.k, (int, np.integer)) or self.k < 1:
            raise ValueError('k must be a positive integer')
        if isinstance(self.folds, bool) or not isinstance(self.folds, (int, np.integer)) or not 2 <= self.folds <= len(X):
            raise ValueError('folds must be an integer between 2 and the row count')
        area = X[:, self.area_index]
        if y.shape != (len(X),) or not np.isfinite(y).all() or np.any(y <= 0) or np.any(area <= 0):
            raise ValueError('Prices and areas must be positive with one price per row')
        self.lat0_ = float(np.mean(X[:, self.lat_index]))
        self.lng0_ = float(np.mean(X[:, self.lng_index]))
        self.points_ = self._points(X)
        self.values_ = np.log(y / area)
        fold = np.random.default_rng(self.random_state).permutation(len(X)) % self.folds
        result = np.empty(len(X))
        for index in range(self.folds):
            held = fold == index
            result[held] = self._query(self.points_[held], self.points_[~held], self.values_[~held])
        return result

    def transform(self, X):
        if not hasattr(self, 'points_'):
            raise RuntimeError('Call fit_transform before transform')
        return self._query(self._points(X), self.points_, self.values_)


class NeighbourPriceRegressor:
    """Append a NeighbourPriceFeature column before fitting a price model."""

    def __init__(self, price_model, feature):
        self.price_model = price_model
        self.feature = feature

    def fit(self, X, y):
        X = _features(X)
        self.price_model.fit(np.column_stack([X, self.feature.fit_transform(X, y)]), y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, 'n_features_in_'):
            raise RuntimeError('Call fit before predict')
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Feature count differs from training')
        return self.price_model.predict(np.column_stack([X, self.feature.transform(X)]))
