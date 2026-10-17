import numpy as np


class RidgeRegression:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or not all(X.shape) or X.dtype.kind not in "biuf":
            raise ValueError("X must be a non-empty numeric two-dimensional array.")
        if y.ndim != 1 or len(y) != len(X) or y.dtype.kind not in "biuf":
            raise ValueError("y must be a numeric one-dimensional array matching X.")
        X, y = X.astype(float), y.astype(float)
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain finite values.")
        if isinstance(self.alpha, (bool, np.bool_)) or not isinstance(self.alpha, (int, float, np.integer, np.floating)):
            raise ValueError("alpha must be a finite non-negative number.")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be a finite non-negative number.")
        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be True or False.")
        self.n_features_in_ = X.shape[1]
        mean_X = X.mean(axis=0) if self.fit_intercept else np.zeros(X.shape[1])
        mean_y = y.mean() if self.fit_intercept else 0.0
        centered_X, centered_y = X - mean_X, y - mean_y
        if self.alpha == 0:
            self.coef_ = np.linalg.lstsq(centered_X, centered_y, rcond=None)[0]
        else:
            u, singular, vt = np.linalg.svd(centered_X, full_matrices=False)
            weights = singular / (singular * singular + self.alpha)
            self.coef_ = vt.T @ (weights * (u.T @ centered_y))
        self.intercept_ = float(mean_y - mean_X @ self.coef_)
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before predict.")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or X.dtype.kind not in "biuf":
            raise ValueError("X must be numeric and have the fitted feature count.")
        X = X.astype(float)
        if not np.isfinite(X).all():
            raise ValueError("X must contain finite values.")
        return X @ self.coef_ + self.intercept_
