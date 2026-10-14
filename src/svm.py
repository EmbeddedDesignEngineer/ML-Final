"""NumPy soft-margin SVM with optional random Fourier features (RFF).

For internal labels in {-1, +1}, both modes minimise
    J(w, b) = lambda/2 * ||w||^2 + mean(max(0, 1 - y * (X @ w + b))).
Bias is not regularised. In RFF mode, replace X by z(X) in this objective.
Supply numeric features standardised using TRAIN statistics only; this
module deliberately does not fit a scaler or choose the price-tier threshold.

References: Shalev-Shwartz et al., Pegasos (2011); Rahimi and Recht,
Random Features for Large-Scale Kernel Machines (2007).
"""

from __future__ import annotations

import numpy as np


def _positive_number(value, name, *, integer=False):
    """Validate scalar hyperparameters without silently truncating floats."""
    types = (int, np.integer) if integer else (int, float, np.integer, np.floating)
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, types)
            or not np.isfinite(value) or value <= 0):
        kind = "integer" if integer else "number"
        raise ValueError(f"{name} must be a finite positive {kind}.")


def _validate_seed(random_state):
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, (int, np.integer)) or random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer or None.")


def _validate_X(X, n_features=None, *, allow_empty=False):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] == 0 or (not allow_empty and X.shape[0] == 0):
        raise ValueError("X must have shape (n_samples, n_features), with non-empty training data.")
    if X.dtype.kind not in "biuf":
        raise ValueError("X must contain real numeric values.")
    X = X.astype(float, copy=False)
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values (no NaN or infinity).")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X must have {n_features} features; got {X.shape[1]}.")
    return X


class RandomFourierFeatures:
    """Approximate k(x, x') = exp(-gamma * ||x - x'||^2).

    z(x) = sqrt(2/D) * cos(x @ weights_ + offsets_).
    Each frequency is N(0, 2*gamma*I), so its standard deviation is
    sqrt(2*gamma), and each phase is uniform on [0, 2*pi).
    fit() samples and stores this map; transform() never resamples it.
    """

    def __init__(self, n_components=500, gamma=1.0, random_state=42):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.weights_ = None
        self.offsets_ = None

    def fit(self, X):
        _positive_number(self.n_components, "n_components", integer=True)
        _positive_number(self.gamma, "gamma")
        _validate_seed(self.random_state)
        X = _validate_X(X)
        rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.weights_ = rng.normal(
            scale=np.sqrt(self.gamma) * np.sqrt(2.0),
            size=(self.n_features_in_, self.n_components),
        )
        self.offsets_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        if self.weights_ is None:
            raise RuntimeError("Call fit before transform.")
        X = _validate_X(X, self.n_features_in_, allow_empty=True)
        with np.errstate(over="raise", invalid="raise"):
            projection = X @ self.weights_ + self.offsets_
            return np.sqrt(2.0 / self.weights_.shape[1]) * np.cos(projection)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PegasosSVM:
    """Binary SVM accepting labels {0, 1}; score zero predicts class 1.

    The first three constructor arguments preserve the starter interface.
    n_iters counts sampled updates. If n_epochs is supplied, it overrides
    n_iters: one stochastic epoch means n_samples draws WITH replacement,
    not a guaranteed visit to every row. n_iter_ records the actual updates.

    learning_rate='pegasos' uses 1/(lambda_*t). 'constant' uses eta0 and is
    an experimental comparison; it need not converge to an exact optimum.
    loss_interval=None records roughly 50 full-training objectives plus the
    initial value. Explicit intervals are measured in sampled updates.

    n_components=None selects a linear SVM; a positive integer enables RFF
    with that many features and the supplied RBF gamma. Its margin and weight
    norm then describe the RFF space, not distances in the original input.
    """

    def __init__(
        self,
        lambda_: float = 1e-4,     # equivalent SVC C = 1/(n_samples*lambda_)
        n_iters: int = 20_000,     # number of sub-gradient steps
        random_state: int = 42,
        *,
        n_epochs: int | None = None,
        learning_rate: str = "pegasos",
        eta0: float = 0.01,
        loss_interval: int | None = None,
        n_components: int | None = None,
        gamma: float = 1.0,
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.loss_interval = loss_interval
        self.n_components = n_components
        self.gamma = gamma
        self.w = None              # weight vector
        self.b = 0.0               # bias
        self.rff_ = None

    @staticmethod
    def _to_pm1(y):
        """Validate a non-empty 1-D label array and map 0 -> -1, 1 -> +1."""
        y = np.asarray(y)
        if y.ndim != 1 or y.size == 0:
            raise ValueError("y must be a non-empty 1-D array.")
        if y.dtype.kind not in "biuf" or not np.all((y == 0) | (y == 1)):
            raise ValueError("y must contain only binary labels 0 and 1.")
        return 2.0 * y.astype(float) - 1.0

    def _validate_parameters(self):
        _positive_number(self.lambda_, "lambda_")
        _positive_number(self.n_iters, "n_iters", integer=True)
        _positive_number(self.eta0, "eta0")
        _positive_number(self.gamma, "gamma")
        for name in ("n_epochs", "loss_interval", "n_components"):
            if getattr(self, name) is not None:
                _positive_number(getattr(self, name), name, integer=True)
        _validate_seed(self.random_state)
        if self.learning_rate not in ("pegasos", "constant"):
            raise ValueError("learning_rate must be 'pegasos' or 'constant'.")
        if self.learning_rate == "constant" and self.eta0 > 1.0 / self.lambda_:
            raise ValueError("For constant learning_rate, eta0 * lambda_ must be <= 1.")

    def fit(self, X, y):
        """Fit from zero, using only the supplied training data; return self."""
        self._validate_parameters()
        X = _validate_X(X)
        y_pm1 = self._to_pm1(y)
        if len(y_pm1) != len(X):
            raise ValueError("X and y must have the same number of samples.")
        if np.unique(y_pm1).size != 2:
            raise ValueError("Training y must contain both classes 0 and 1.")

        rff = None
        features = X
        if self.n_components is not None:
            rff = RandomFourierFeatures(self.n_components, self.gamma, self.random_state)
            features = rff.fit_transform(X)
        n_samples, n_features = features.shape
        n_steps = self.n_iters if self.n_epochs is None else self.n_epochs * n_samples
        interval = self.loss_interval or max(1, (n_steps + 49) // 50)
        rng = np.random.default_rng(self.random_state)
        w = np.zeros(n_features, dtype=float)
        b = 0.0
        losses = [1.0]  # At w=0, b=0, every sample has hinge loss 1.
        iterations = [0]
        radius = 1.0 / np.sqrt(self.lambda_)

        # Fail explicitly on overflow instead of returning NaN predictions.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for t in range(1, n_steps + 1):
                i = rng.integers(n_samples)
                eta = (1.0 / self.lambda_) / t if self.learning_rate == "pegasos" else self.eta0
                # Test the margin BEFORE shrinking w: this is the subgradient
                # of the objective at the current (w, b).
                hinge_active = y_pm1[i] * (features[i] @ w + b) < 1.0
                w *= 1.0 - eta * self.lambda_
                if hinge_active:
                    w += eta * y_pm1[i] * features[i]
                    b += eta * y_pm1[i]  # No shrinkage or projection of bias.

                # Euclidean projection onto ||w|| <= 1/sqrt(lambda).
                # The ball contains an optimum; bias is excluded throughout.
                norm = np.linalg.norm(w)
                if not np.isfinite(norm):
                    raise FloatingPointError("Weight norm overflow; check feature scaling and lambda_.")
                if norm > radius:
                    w *= radius / norm

                if t % interval == 0 or t == n_steps:
                    hinge = np.maximum(0.0, 1.0 - y_pm1 * (features @ w + b))
                    losses.append(float(0.5 * self.lambda_ * (w @ w) + hinge.mean()))
                    iterations.append(t)

            margins = y_pm1 * (features @ w + b)
        if not np.isfinite(b) or not np.all(np.isfinite(w)) or not np.all(np.isfinite(losses)):
            raise FloatingPointError("Non-finite SVM state; check feature scaling and hyperparameters.")

        self.w, self.b = w, float(b)
        self.rff_ = rff
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = n_steps
        self.loss_history_ = losses
        self.loss_iterations_ = iterations
        # These are margin-active points, NOT coefficients of a dual solution.
        self.margin_active_indices_ = np.flatnonzero(margins <= 1.0)
        self.n_margin_active_ = self.margin_active_indices_.size
        self.margin_active_fraction_ = self.n_margin_active_ / n_samples
        return self

    def _check_fitted(self):
        if self.w is None:
            raise RuntimeError("Call fit before using this SVM.")

    def decision_function(self, X):
        """Return signed scores, shape (n_samples,); these are not probabilities."""
        self._check_fitted()
        X = _validate_X(X, self.n_features_in_, allow_empty=True)
        features = X if self.rff_ is None else self.rff_.transform(X)
        with np.errstate(over="raise", invalid="raise"):
            return features @ self.w + self.b

    def predict(self, X):
        """Return the original labels: negative score -> 0, otherwise -> 1."""
        return (self.decision_function(X) >= 0.0).astype(int)

    def hinge_loss(self, X, y):
        """Mean hinge loss only, excluding regularisation; y uses labels {0,1}."""
        y_pm1 = self._to_pm1(y)
        scores = self.decision_function(X)
        if len(y_pm1) != len(scores):
            raise ValueError("X and y must have the same number of samples.")
        return float(np.maximum(0.0, 1.0 - y_pm1 * scores).mean())

    def objective(self, X, y):
        """Evaluate the same regularised objective recorded during training."""
        hinge = self.hinge_loss(X, y)
        return float(0.5 * self.lambda_ * (self.w @ self.w) + hinge)

    @property
    def weight_norm_(self):
        """Euclidean norm of the fitted weights (in RFF space when enabled)."""
        self._check_fitted()
        return float(np.linalg.norm(self.w))

    @property
    def margin_width_(self):
        """Distance between scores -1 and +1; infinity if w=0 (degenerate)."""
        norm = self.weight_norm_
        return 2.0 / norm if norm > 0.0 else float("inf")
