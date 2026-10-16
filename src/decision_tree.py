"""Deterministic numeric decision tree implemented with NumPy only.

Splits use X[:, feature] <= threshold. Tied gains retain the first feature
and lowest threshold; tied class counts choose the first sorted class.
"""
from __future__ import annotations

import numpy as np


class Node:
    """A split, or a leaf with a prediction and class frequencies."""

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.proba = None
        self.n_samples = 0
        self.impurity = 0.0
        self.gain = 0.0

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    """Exhaustive CART-style tree for finite, preprocessed numeric features.

    ``min_impurity_decrease`` uses gain weighted by node/training sample
    count, as in sklearn. ``max_depth=0`` deliberately permits a stump.
    No random feature sampling is used, so random_state is API-only.
    """
    def __init__(self, task="classification", criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, random_state=42):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        # Exhaustive search is deterministic; retained for the shared API.
        self.random_state = random_state
        self.root = None

    def _impurity(self, y):
        if self.task == "regression":
            return float(np.var(y))
        counts = np.bincount(y, minlength=len(self.classes_))
        p = counts[counts > 0] / len(y)
        if self.criterion == "entropy":
            return float(-np.sum(p * np.log2(p)))
        return float(1.0 - p @ p)

    def _best_split(self, X, y):
        parent = self._impurity(y)
        best, best_gain = None, 0.0
        n = len(y)
        for feature in range(X.shape[1]):
            order = np.argsort(X[:, feature], kind="stable")
            values, targets = X[order, feature], y[order]
            positions = np.flatnonzero(values[:-1] < values[1:]) + 1
            positions = positions[(positions >= self.min_samples_leaf)
                                  & (n - positions >= self.min_samples_leaf)]
            if not len(positions):
                continue
            left_n, right_n = positions, n - positions
            if self.task == "regression":
                # Center before accumulating squares: real prices may be large.
                centered = targets - np.mean(targets)
                sums = np.cumsum(centered)
                squares = np.cumsum(centered * centered)
                left_s = sums[positions - 1]
                right_s = sums[-1] - left_s
                left_sse = squares[positions - 1] - left_s * left_s / left_n
                right_sse = squares[-1] - squares[positions - 1] - right_s * right_s / right_n
                child_impurity = (np.maximum(left_sse, 0) + np.maximum(right_sse, 0)) / n
            else:
                # Prefix class counts evaluate all valid thresholds in one pass.
                counts = np.eye(len(self.classes_), dtype=np.int64)[targets].cumsum(axis=0)
                left_counts = counts[positions - 1]
                right_counts = counts[-1] - left_counts
                left_p = left_counts / left_n[:, None]
                right_p = right_counts / right_n[:, None]
                if self.criterion == "gini":
                    left_i = 1 - np.sum(left_p * left_p, axis=1)
                    right_i = 1 - np.sum(right_p * right_p, axis=1)
                else:
                    left_i = -np.sum(left_p * np.log2(np.maximum(left_p, np.finfo(float).tiny)), axis=1)
                    right_i = -np.sum(right_p * np.log2(np.maximum(right_p, np.finfo(float).tiny)), axis=1)
                child_impurity = (left_n * left_i + right_n * right_i) / n
            gains = parent - child_impurity
            index = int(np.argmax(gains))
            gain = float(gains[index])
            if gain > best_gain:
                position = positions[index]
                lower, upper = values[position - 1], values[position]
                threshold = lower / 2 + upper / 2
                if threshold >= upper:
                    threshold = lower
                best, best_gain = (feature, float(threshold), gain), gain
        return best

    def _grow(self, X, y, depth):
        node = Node()
        node.n_samples = len(y)
        node.impurity = self._impurity(y)
        if self.task == "regression":
            node.value = float(np.mean(y))
        else:
            counts = np.bincount(y, minlength=len(self.classes_))
            node.proba = counts / len(y)
            node.value = self.classes_[np.argmax(counts)]
        if (node.impurity <= 0 or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node
        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold, gain = split
        if gain * len(y) / self.n_samples_ < self.min_impurity_decrease:
            return node
        node.feature, node.threshold, node.gain = feature, threshold, gain
        left = X[:, feature] <= threshold
        node.left = self._grow(X[left], y[left], depth + 1)
        node.right = self._grow(X[~left], y[~left], depth + 1)
        return node

    def fit(self, X, y):
        self._validate_parameters()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or not all(X.shape) or not np.isfinite(X).all():
            raise ValueError("X must be a non-empty finite 2-D numeric array")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be a 1-D array with one target per row")
        if y.dtype.kind not in "biufUS" or (y.dtype.kind in "biuf" and not np.isfinite(y).all()):
            raise ValueError("y must contain finite numbers or string labels")
        self.root = None
        if self.task == "classification":
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            y = np.asarray(y, dtype=float)
            if not np.isfinite(y).all():
                raise ValueError("Regression targets must be finite numbers")
            if hasattr(self, "classes_"):
                del self.classes_
        self.n_features_in_ = X.shape[1]
        self.n_samples_ = len(y)
        self.root = self._grow(X, y, 0)
        self.feature_importances_ = np.zeros(self.n_features_in_)
        for node, _ in self._nodes():
            if not node.is_leaf():
                self.feature_importances_[node.feature] += node.n_samples / self.n_samples_ * node.gain
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _validate_parameters(self):
        valid = {"classification": ("gini", "entropy"), "regression": ("mse",)}
        if self.task not in valid or self.criterion not in valid[self.task]:
            raise ValueError("Use classification with gini/entropy, or regression with mse")
        for name, minimum in (("max_depth", 0), ("min_samples_split", 2), ("min_samples_leaf", 1)):
            value = getattr(self, name)
            if name == "max_depth" and value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        value = self.min_impurity_decrease
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value < 0):
            raise ValueError("min_impurity_decrease must be finite and non-negative")

    def _check_fitted(self):
        if self.root is None:
            raise RuntimeError("Call fit before predicting or inspecting the tree")

    def _prediction_array(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError(f"X must be finite and 2-D with {self.n_features_in_} features")
        return X

    def _nodes(self):
        self._check_fitted()
        stack = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            yield node, depth
            if not node.is_leaf():
                stack.extend(((node.right, depth + 1), (node.left, depth + 1)))

    def get_depth(self):
        return max(depth for _, depth in self._nodes())

    def get_n_leaves(self):
        return sum(node.is_leaf() for node, _ in self._nodes())

    def _leaf(self, x, node):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def _predict_one(self, x, node):
        return self._leaf(x, node).value

    def predict(self, X):
        X = self._prediction_array(X)
        dtype = self.classes_.dtype if self.task == "classification" else float
        return np.asarray([self._predict_one(x, self.root) for x in X], dtype=dtype)

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba is available for classification only")
        X = self._prediction_array(X)
        return np.asarray([self._leaf(x, self.root).proba for x in X]).reshape(len(X), len(self.classes_))
