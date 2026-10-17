"""Bootstrap and random-subspace averaging of our NumPy regression trees."""
import numpy as np
from .decision_tree import DecisionTree


class BaggedTreeRegressor:
    def __init__(self, n_estimators=24, max_depth=16, min_samples_leaf=5,
                 max_samples=.8, max_features=.8, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, (int, np.integer)) or self.n_estimators < 1:
            raise ValueError('n_estimators must be a positive integer')
        for name in ('max_samples', 'max_features'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not 0 < value <= 1:
                raise ValueError(f'{name} must be a fraction in (0, 1]')
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, (int, np.integer)) or self.random_state < 0:
            raise ValueError('random_state must be a non-negative integer')
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim != 2 or not all(X.shape) or not np.isfinite(X).all():
            raise ValueError('X must be a non-empty finite 2-D array')
        if y.ndim != 1 or len(y) != len(X) or not np.isfinite(y).all():
            raise ValueError('y must be finite with one target per row')
        rng = np.random.default_rng(self.random_state)
        n_rows = max(1, int(len(X) * self.max_samples))
        n_columns = max(1, int(X.shape[1] * self.max_features))
        estimators, features = [], []
        importance = np.zeros(X.shape[1])
        for _ in range(self.n_estimators):
            rows = rng.integers(len(X), size=n_rows)
            columns = np.sort(rng.choice(X.shape[1], size=n_columns, replace=False))
            tree = DecisionTree(task='regression', criterion='mse', max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf)
            tree.fit(X[rows][:, columns], y[rows])
            estimators.append(tree)
            features.append(columns)
            importance[columns] += tree.feature_importances_
        self.estimators_, self.features_ = estimators, features
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = importance / importance.sum() if importance.sum() else importance
        return self

    def staged_predict(self, X):
        if not hasattr(self, 'estimators_'):
            raise RuntimeError('Call fit before predict')
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError('X must be finite with the fitted feature count')
        total = np.zeros(len(X))
        for count, (tree, columns) in enumerate(zip(self.estimators_, self.features_), start=1):
            total += tree.predict(X[:, columns])
            yield total / count

    def predict(self, X):
        for prediction in self.staged_predict(X):
            pass
        return prediction
