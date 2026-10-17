"""SVM-only checks on synthetic data; run: python -m pytest tests/test_svm.py.

scikit-learn appears only in independent baseline tests. No real dataset,
preprocessing module, network access, or experiment runner is needed.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from src.svm import PegasosSVM, RandomFourierFeatures


@pytest.fixture
def toy():
    return np.array([[-2.0], [-1.0], [1.0], [2.0]]), np.array([0, 0, 1, 1])


@pytest.fixture
def fixed_draws(monkeypatch):
    """Choose examples explicitly so individual updates can be checked by hand."""
    def select(indices):
        draws = iter(indices)
        monkeypatch.setattr(
            np.random, "default_rng",
            lambda seed: SimpleNamespace(integers=lambda n: next(draws)),
        )
    return select


@pytest.mark.parametrize("dtype", [int, float, bool, np.uint8])
def test_binary_label_conversion(dtype):
    y = np.array([0, 1, 1, 0], dtype=dtype)
    assert_array_equal(PegasosSVM._to_pm1(y), [-1.0, 1.0, 1.0, -1.0])
    assert_array_equal(y, [0, 1, 1, 0])


def test_manual_hinge_objective_and_decision_scores(toy):
    model = PegasosSVM(lambda_=0.25, n_iters=1).fit(*toy)
    model.w = np.array([2.0])
    model.b = 0.5
    X, y = np.array([[-1.0], [0.0], [1.0]]), np.array([0, 1, 0])
    assert_allclose(model.decision_function(X), [-1.5, 0.5, 2.5])
    # Margins [1.5, .5, -2.5] -> hinge losses [0, .5, 3.5].
    assert model.hinge_loss(X, y) == pytest.approx(4.0 / 3.0)
    assert model.objective(X, y) == pytest.approx(4.0 / 3.0 + 0.5)
    # Changing b affects the scores but contributes NO quadratic penalty.
    model.b = 3.0
    assert model.objective(X, y) - model.hinge_loss(X, y) == pytest.approx(0.5)


def test_predictions_shapes_mapping_and_zero_tie(toy):
    model = PegasosSVM(n_iters=1).fit(*toy)
    model.w, model.b = np.array([1.0]), 0.0
    X = np.array([[-2.0], [0.0], [2.0]])
    assert_array_equal(model.predict(X), [0, 1, 1])
    assert_array_equal(model.classes_, [0, 1])
    assert model.predict(X).shape == model.decision_function(X).shape == (3,)
    assert model.predict(X[:1]).shape == (1,)
    assert model.predict(np.empty((0, 1))).shape == (0,)


@pytest.mark.parametrize("steps, expected_w, expected_b", [
    (1, 1.0, 1.0), (2, 0.5, 1.0), (3, 1.0, 2.0 / 3.0),
])
def test_hand_calculated_pegasos_updates(fixed_draws, steps, expected_w, expected_b):
    fixed_draws([0, 0, 1])
    model = PegasosSVM(lambda_=1.0, n_iters=steps).fit([[2.0], [-2.0]], [1, 0])
    # t=1: active, raw w=2 projects to 1, b=1.
    # t=2: inactive, w=.5, b unchanged.
    # t=3: active, w=(2/3)*.5+(1/3)*2=1, b=1-1/3.
    assert_allclose(model.w, [expected_w])
    assert model.b == pytest.approx(expected_b)


def test_margin_test_precedes_shrinkage_and_boundary_is_inactive(fixed_draws):
    fixed_draws([0, 1])
    model = PegasosSVM(lambda_=1.0, n_iters=2).fit([[2.0], [-2.0]], [1, 0])
    # Before step 2, the second margin is exactly 1. Shrinking first would
    # incorrectly make it active. The chosen subgradient at the kink is zero.
    assert_allclose(model.w, [0.5])
    assert model.b == pytest.approx(1.0)


def test_bias_is_not_projected(fixed_draws):
    fixed_draws([0])
    model = PegasosSVM(lambda_=0.25, n_iters=1).fit([[2.0], [-2.0]], [1, 0])
    assert model.weight_norm_ == pytest.approx(2.0)
    assert model.b == pytest.approx(4.0)


@pytest.mark.parametrize("lambda_", [2.0, 8.0])
def test_regularization_matches_analytic_soft_margin_optimum(lambda_):
    # For balanced x=-1,+1 and lambda>1, both hinge terms are active.
    # J(w,0)=lambda*w^2/2+1-w, so w*=1/lambda, J*=1-1/(2*lambda).
    X, y = np.array([[-1.0], [1.0]]), np.array([0, 1])
    model = PegasosSVM(lambda_=lambda_, n_iters=4000).fit(X, y)
    assert_allclose(model.w, [1.0 / lambda_], atol=1e-10)
    assert model.objective(X, y) == pytest.approx(1.0 - 0.5 / lambda_)
    assert model.margin_width_ == pytest.approx(2 * lambda_)
    assert model.n_margin_active_ == 2


def test_constant_learning_rate_updates(fixed_draws):
    fixed_draws([0, 0])
    model = PegasosSVM(
        lambda_=1.0, n_iters=2, learning_rate="constant", eta0=0.1,
    ).fit([[2.0], [-2.0]], [1, 0])
    # Both steps are active: w=.2, then .9*.2+.2=.38; b=.1+.1.
    assert_allclose(model.w, [0.38])
    assert model.b == pytest.approx(0.2)


def test_epochs_override_iterations(toy):
    model = PegasosSVM(n_iters=1, n_epochs=3, loss_interval=5).fit(*toy)
    assert model.n_iter_ == 12
    assert model.loss_iterations_ == [0, 5, 10, 12]


def test_learns_separable_data_with_original_defaults(toy):
    model = PegasosSVM()
    assert model.fit(*toy) is model
    assert_array_equal(model.predict(toy[0]), toy[1])
    assert model.loss_history_[-1] < model.loss_history_[0]
    assert model.weight_norm_ <= 1.0 / np.sqrt(model.lambda_) + 1e-12


def test_learns_boundary_requiring_nonzero_bias():
    X = np.array([[1.0], [1.5], [2.0], [4.0], [4.5], [5.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = PegasosSVM(lambda_=0.01, n_iters=12000).fit(X, y)
    assert_array_equal(model.predict(X), y)
    assert model.b < -1.0


@pytest.mark.parametrize("n_components", [None, 32])
def test_reproducibility_refitting_and_unchanged_inputs(toy, n_components):
    X, y = toy
    original_X, original_y = X.copy(), y.copy()
    kwargs = dict(n_iters=200, random_state=7, n_components=n_components)
    first, second = PegasosSVM(**kwargs).fit(X, y), PegasosSVM(**kwargs).fit(X, y)
    assert_array_equal(first.w, second.w)
    assert first.b == second.b
    assert first.loss_history_ == second.loss_history_
    first.fit(X, y)  # fit restarts instead of silently performing a warm start.
    assert_array_equal(first.w, second.w)
    assert first.loss_iterations_ == second.loss_iterations_
    assert_array_equal(X, original_X)
    assert_array_equal(y, original_y)


def test_local_rng_does_not_change_global_state(toy):
    before = np.random.get_state()
    PegasosSVM(n_iters=10, n_components=8).fit(*toy)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_weight_norm_margin_and_zero_norm(toy):
    model = PegasosSVM(n_iters=1).fit(*toy)
    model.w = np.array([3.0, 4.0])
    assert model.weight_norm_ == pytest.approx(5.0)
    assert model.margin_width_ == pytest.approx(0.4)
    model.w[:] = 0.0
    assert model.weight_norm_ == 0.0
    assert np.isinf(model.margin_width_)


def test_margin_active_points_include_boundary_and_errors(fixed_draws):
    fixed_draws([0])
    X = np.array([[2.0], [0.0], [-1.0], [1.0]])
    y = np.array([1, 1, 0, 0])
    model = PegasosSVM(lambda_=1.0, n_iters=1).fit(X, y)
    # w=1,b=1: margins [3,1,0,-2]. Include the boundary and misclassified rows.
    assert_array_equal(model.margin_active_indices_, [1, 2, 3])
    assert model.n_margin_active_ == 3
    assert model.margin_active_fraction_ == pytest.approx(0.75)


@pytest.mark.parametrize("n_components", [None, 16])
def test_history_records_exact_objective_at_correct_iterations(toy, n_components):
    kwargs = dict(lambda_=0.1, random_state=3, n_components=n_components)
    model = PegasosSVM(**kwargs, n_iters=13, loss_interval=5).fit(*toy)
    assert model.loss_iterations_ == [0, 5, 10, 13]
    assert model.loss_history_[0] == 1.0
    assert len(model.loss_history_) == len(model.loss_iterations_)
    assert np.all(np.isfinite(model.loss_history_))
    # Independent shorter runs end at precisely the recorded checkpoints.
    for t, value in zip(model.loss_iterations_[1:], model.loss_history_[1:]):
        checkpoint = PegasosSVM(**kwargs, n_iters=t).fit(*toy)
        assert value == pytest.approx(checkpoint.objective(*toy))
    assert model.loss_history_[-1] == pytest.approx(model.objective(*toy))


def test_history_default_cost_is_bounded_and_final_not_duplicated(toy):
    model = PegasosSVM(n_iters=503).fit(*toy)
    assert len(model.loss_history_) <= 51
    assert model.loss_iterations_[-1] == 503
    assert np.all(np.diff(model.loss_iterations_) > 0)
    model = PegasosSVM(n_iters=10, loss_interval=5).fit(*toy)
    assert model.loss_iterations_ == [0, 5, 10]
    model = PegasosSVM(n_iters=10, loss_interval=100).fit(*toy)
    assert model.loss_iterations_ == [0, 10]


@pytest.mark.parametrize("X, y, message", [
    ([1, 2], [0, 1], "shape"),
    (np.ones((2, 1, 1)), [0, 1], "shape"),
    (np.empty((0, 2)), [], "shape"),
    (np.empty((2, 0)), [0, 1], "shape"),
    ([[0], [1]], [[0], [1]], "1-D"),
    ([[0], [1]], [], "1-D"),
    ([[0], [1]], [0], "same number"),
    ([[0], [1]], [1, 1], "both classes"),
    ([[0], [1]], [-1, 1], "binary labels"),
    ([[0], [1]], [0, 2], "binary labels"),
    ([[0], [1]], [0, 0.5], "binary labels"),
    ([[0], [1]], [0, np.nan], "binary labels"),
    ([[0], [1]], [0, np.inf], "binary labels"),
    ([[0], [1]], ["0", "1"], "binary labels"),
    ([[0], [np.nan]], [0, 1], "finite"),
    ([[0], [np.inf]], [0, 1], "finite"),
    ([[0j], [1j]], [0, 1], "real numeric"),
    ([["a"], ["b"]], [0, 1], "real numeric"),
])
def test_invalid_training_inputs(X, y, message):
    with pytest.raises(ValueError, match=message):
        PegasosSVM(n_iters=1).fit(X, y)


@pytest.mark.parametrize("kwargs, message", [
    ({"lambda_": 0}, "lambda_"), ({"lambda_": -1}, "lambda_"),
    ({"lambda_": np.nan}, "lambda_"), ({"lambda_": np.inf}, "lambda_"),
    ({"lambda_": True}, "lambda_"), ({"lambda_": "0.1"}, "lambda_"),
    ({"n_iters": 0}, "n_iters"), ({"n_iters": 2.5}, "n_iters"),
    ({"n_iters": True}, "n_iters"), ({"n_epochs": 0}, "n_epochs"),
    ({"n_epochs": 1.5}, "n_epochs"), ({"loss_interval": 0}, "loss_interval"),
    ({"loss_interval": 1.2}, "loss_interval"),
    ({"random_state": -1}, "random_state"), ({"random_state": 1.5}, "random_state"),
    ({"random_state": True}, "random_state"),
    ({"learning_rate": "unknown"}, "learning_rate"),
    ({"eta0": 0}, "eta0"), ({"eta0": np.inf}, "eta0"),
    ({"lambda_": 2, "learning_rate": "constant", "eta0": 1}, "eta0"),
    ({"n_components": 0}, "n_components"), ({"n_components": 2.5}, "n_components"),
    ({"gamma": 0}, "gamma"), ({"gamma": np.nan}, "gamma"),
])
def test_invalid_parameters(toy, kwargs, message):
    with pytest.raises(ValueError, match=message):
        PegasosSVM(**kwargs).fit(*toy)


def test_unfitted_models_and_invalid_inference(toy):
    model = PegasosSVM()
    for method in (model.predict, model.decision_function):
        with pytest.raises(RuntimeError, match="fit"):
            method(toy[0])
    with pytest.raises(RuntimeError, match="fit"):
        _ = model.margin_width_
    model = PegasosSVM(n_iters=1).fit(*toy)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        model.predict([[np.inf]])
    with pytest.raises(ValueError, match="same number"):
        model.hinge_loss(toy[0], [0, 1])
    with pytest.raises(ValueError, match="binary labels"):
        model.objective(toy[0], [0, 1, 1, 2])


def test_zero_features_are_finite_and_have_degenerate_margin():
    model = PegasosSVM(n_iters=100).fit(np.zeros((4, 2)), [0, 1, 0, 1])
    assert model.weight_norm_ == 0.0
    assert np.isinf(model.margin_width_)
    assert np.all(np.isfinite(model.loss_history_))


@pytest.mark.parametrize("lambda_, magnitude", [(1e-300, 1e200), (1e-100, 1e80)])
def test_numerical_overflow_raises_instead_of_returning_nan(lambda_, magnitude):
    with pytest.raises(FloatingPointError):
        PegasosSVM(lambda_=lambda_, n_iters=1).fit([[magnitude], [-magnitude]], [0, 1])


def test_rff_dimensions_formula_and_reproducibility():
    X = np.arange(12, dtype=float).reshape(4, 3) / 10.0
    first = RandomFourierFeatures(n_components=20, gamma=0.4, random_state=8)
    Z = first.fit_transform(X)
    second = RandomFourierFeatures(n_components=20, gamma=0.4, random_state=8).fit(X)
    assert Z.shape == (4, 20)
    assert first.weights_.shape == (3, 20)
    assert first.offsets_.shape == (20,)
    assert_allclose(Z, np.sqrt(2.0 / 20) * np.cos(X @ first.weights_ + first.offsets_))
    assert_array_equal(Z, second.transform(X))
    assert np.all((first.offsets_ >= 0) & (first.offsets_ < 2 * np.pi))
    assert first.transform(np.empty((0, 3))).shape == (0, 20)
    # Changing gamma from .4 to 1.6 doubles frequency standard deviation.
    wider = RandomFourierFeatures(20, gamma=1.6, random_state=8).fit(X)
    assert_allclose(wider.weights_, 2 * first.weights_)
    different = RandomFourierFeatures(20, gamma=0.4, random_state=9).fit(X)
    assert not np.array_equal(first.weights_, different.weights_)


def test_rff_approximates_the_stated_rbf_kernel():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.2, -0.4], [-1.0, 1.0]])
    gamma = 0.7
    rff = RandomFourierFeatures(20000, gamma=gamma, random_state=5)
    Z = rff.fit_transform(X)
    squared_distances = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    assert_allclose(Z @ Z.T, np.exp(-gamma * squared_distances), atol=0.03, rtol=0)


@pytest.mark.parametrize("kwargs", [
    {"n_components": 0}, {"n_components": 1.5}, {"gamma": -1},
    {"gamma": np.inf}, {"random_state": -1},
])
def test_rff_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        RandomFourierFeatures(**kwargs).fit([[0.0]])


def test_rff_unfitted_and_invalid_inputs():
    rff = RandomFourierFeatures(4)
    with pytest.raises(RuntimeError, match="fit"):
        rff.transform([[0.0]])
    with pytest.raises(ValueError, match="shape"):
        rff.fit([0.0, 1.0])
    rff.fit([[0.0]])
    with pytest.raises(ValueError, match="features"):
        rff.transform([[0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        rff.transform([[np.nan]])


def test_inference_reuses_fitted_rff_and_never_changes_model(toy):
    model = PegasosSVM(n_components=32, gamma=0.5, n_iters=500).fit(*toy)
    weights, offsets = model.rff_.weights_.copy(), model.rff_.offsets_.copy()
    w, b, losses = model.w.copy(), model.b, model.loss_history_.copy()
    active = model.margin_active_indices_.copy()
    X_test = np.array([[-100.0], [0.2], [100.0]])
    scores = model.decision_function(X_test)
    expected = model.rff_.transform(X_test) @ model.w + model.b
    assert_allclose(scores, expected)
    assert_allclose(scores, np.concatenate([model.decision_function(row[None]) for row in X_test]))
    model.predict(X_test[::-1])
    assert_array_equal(weights, model.rff_.weights_)
    assert_array_equal(offsets, model.rff_.offsets_)
    assert_array_equal(w, model.w)
    assert model.b == b
    assert model.loss_history_ == losses
    assert_array_equal(active, model.margin_active_indices_)
    margins = (2 * toy[1] - 1) * model.decision_function(toy[0])
    assert_array_equal(active, np.flatnonzero(margins <= 1))


def _circles(seed, n_per_class):
    """Two concentric rings, with separate seeded training and test samples."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n_per_class)
    angles = rng.uniform(0, 2 * np.pi, size=len(y))
    radii = np.where(y == 0, 0.5, 1.5) + rng.normal(0, 0.03, size=len(y))
    return radii[:, None] * np.column_stack([np.cos(angles), np.sin(angles)]), y


def test_nonlinear_learning_on_independent_concentric_rings():
    X_train, y_train = _circles(10, 100)
    X_test, y_test = _circles(11, 150)
    kwargs = dict(lambda_=0.01, n_iters=8000, random_state=42)
    linear = PegasosSVM(**kwargs).fit(X_train, y_train)
    nonlinear = PegasosSVM(**kwargs, n_components=256, gamma=1.0).fit(X_train, y_train)
    linear_accuracy = np.mean(linear.predict(X_test) == y_test)
    nonlinear_accuracy = np.mean(nonlinear.predict(X_test) == y_test)
    assert nonlinear_accuracy >= 0.97
    assert nonlinear_accuracy - linear_accuracy >= 0.25
    assert nonlinear.loss_history_[-1] < 0.3 * nonlinear.loss_history_[0]


@pytest.mark.parametrize("noise", [0.0, 0.35])
def test_linear_agreement_with_independent_sklearn_baseline(noise):
    from sklearn.svm import SVC

    rng = np.random.default_rng(2026)
    X = rng.normal(size=(360, 2))
    y = (X[:, 0] + 0.7 * X[:, 1] + noise * rng.normal(size=len(X)) > 0.2).astype(int)
    X_train, X_test, y_train, y_test = X[:240], X[240:], y[:240], y[240:]
    lambda_ = 0.02
    ours = PegasosSVM(lambda_=lambda_, n_iters=12000).fit(X_train, y_train)
    # SVC minimises .5*||w||^2 + C*sum(hinge): C=1/(n*lambda).
    baseline = SVC(kernel="linear", C=1.0 / (len(y_train) * lambda_)).fit(X_train, y_train)
    reference, actual = baseline.predict(X_test), ours.predict(X_test)
    assert np.mean(actual == reference) >= 0.95
    assert np.mean(actual == y_test) >= 0.9
    reference_w = baseline.coef_[0]
    reference_scores = X_train @ reference_w + baseline.intercept_[0]
    reference_loss = np.maximum(0, 1 - (2 * y_train - 1) * reference_scores).mean()
    reference_objective = lambda_ / 2 * (reference_w @ reference_w) + reference_loss
    assert abs(ours.objective(X_train, y_train) - reference_objective) < 0.025


def test_nonlinear_agreement_with_independent_sklearn_baseline():
    from sklearn.svm import SVC

    X_train, y_train = _circles(31, 60)
    X_test, y_test = _circles(32, 80)
    lambda_, gamma = 0.01, 1.0
    ours = PegasosSVM(
        lambda_=lambda_, n_iters=6000, n_components=256, gamma=gamma,
    ).fit(X_train, y_train)
    baseline = SVC(kernel="rbf", gamma=gamma, C=1.0 / (len(y_train) * lambda_))
    baseline.fit(X_train, y_train)
    assert np.mean(ours.predict(X_test) == baseline.predict(X_test)) >= 0.95
    assert np.mean(ours.predict(X_test) == y_test) >= 0.97


def test_tail_average_matches_manual_updates(fixed_draws):
    fixed_draws([0,0,1])
    model=PegasosSVM(lambda_=1.,n_iters=3,average=True).fit([[2.],[-2.]],[1,0])
    assert_allclose(model.w,[.75])
    assert model.b==pytest.approx(5/6)
    assert model.loss_history_[-1]==pytest.approx(model.objective([[2.],[-2.]],[1,0]))
    assert model.n_averaged_==2


def test_tail_average_is_seeded_and_reports_final_margin_state(toy):
    params=dict(lambda_=.01,n_epochs=10,average=True,random_state=7)
    first=PegasosSVM(**params).fit(*toy)
    second=PegasosSVM(**params).fit(*toy)
    assert_allclose(first.w,second.w)
    assert first.b==second.b
    margins=(2*toy[1]-1)*first.decision_function(toy[0])
    assert_array_equal(first.margin_active_indices_,np.flatnonzero(margins<=1))
    with pytest.raises(ValueError):
        PegasosSVM(average='yes').fit(*toy)
