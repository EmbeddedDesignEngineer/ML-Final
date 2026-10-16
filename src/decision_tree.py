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
        return float(1.0 - p @ p)

    def _best_split(self, X, y):
        parent = self._impurity(y)
        best, best_gain = None, 0.0
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for lower, upper in zip(values[:-1], values[1:]):
                threshold = lower / 2 + upper / 2
                if threshold >= upper:
                    threshold = lower
                left = X[:, feature] <= threshold
                n_left = int(left.sum())
                if min(n_left, len(y) - n_left) < self.min_samples_leaf:
                    continue
                gain = parent - (n_left * self._impurity(y[left])
                                 + (len(y) - n_left) * self._impurity(y[~left])) / len(y)
                if gain > best_gain:
                    best, best_gain = (feature, threshold, gain), gain
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
        X = np.asarray(X, dtype=float)
        if self.task == "classification":
            self.classes_, y = np.unique(np.asarray(y), return_inverse=True)
        else:
            y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.n_samples_ = len(y)
        self.root = self._grow(X, y, 0)
        return self

    def _leaf(self, x):
        node = self.root
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def _predict_one(self, x, node):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return np.asarray([self._predict_one(x, self.root) for x in np.asarray(X, dtype=float)])

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_proba is available for classification only")
        return np.asarray([self._leaf(x).proba for x in np.asarray(X, dtype=float)])
