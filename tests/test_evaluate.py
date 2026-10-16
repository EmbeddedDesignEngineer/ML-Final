import numpy as np
import pytest
from sklearn import metrics

from src.evaluate import rmse, mae, r2, confusion_matrix, precision_recall_f1, roc_auc, compare


def test_hand_calculated_regression():
    truth, pred = [100, 200, 300], [110, 190, 300]
    assert mae(truth, pred) == pytest.approx(20 / 3)
    assert rmse(truth, pred) == pytest.approx(np.sqrt(200 / 3))
    assert r2(truth, pred) == pytest.approx(0.99)


def test_hand_calculated_classification():
    truth, pred = [0, 0, 1, 1, 1], [0, 1, 0, 1, 1]
    np.testing.assert_array_equal(confusion_matrix(truth, pred), [[1, 1], [1, 2]])
    assert precision_recall_f1(truth, pred) == pytest.approx((2 / 3, 2 / 3, 2 / 3))


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_regression_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    truth = rng.normal(200000, 50000, 100)
    pred = truth + rng.normal(0, 30000, 100)
    assert mae(truth, pred) == pytest.approx(metrics.mean_absolute_error(truth, pred))
    assert rmse(truth, pred) == pytest.approx(np.sqrt(metrics.mean_squared_error(truth, pred)))
    assert r2(truth, pred) == pytest.approx(metrics.r2_score(truth, pred))


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_classification_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, 200)
    scores = np.round(rng.normal(size=200), 1)  # Many tied scores.
    pred = (scores >= 0).astype(int)
    np.testing.assert_array_equal(confusion_matrix(truth, pred), metrics.confusion_matrix(truth, pred, labels=[0, 1]))
    expected = metrics.precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)[:3]
    assert precision_recall_f1(truth, pred) == pytest.approx(expected)
    assert roc_auc(truth, scores) == pytest.approx(metrics.roc_auc_score(truth, scores))


@pytest.mark.parametrize("scores, expected", [([0, 0, 1, 1], 1), ([1, 1, 0, 0], 0), ([2, 2, 2, 2], 0.5), ([0, 1, 1, 2], 0.875)])
def test_auc_known_cases(scores, expected):
    assert roc_auc([0, 0, 1, 1], scores) == expected


def test_auc_matches_pairwise_definition():
    truth = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([0.5, 0.5, -1, 2, 3, -1])
    wins = 0
    for positive in scores[truth == 1]:
        for negative in scores[truth == 0]:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    assert roc_auc(truth, scores) == pytest.approx(wins / 9)


def test_perfect_and_constant_regression():
    assert mae([1, 2], [1, 2]) == 0
    assert rmse([1, 2], [1, 2]) == 0
    assert r2([1, 2], [1, 2]) == 1
    assert r2([5, 5], [5, 5]) == 1
    assert r2([5, 5], [4, 4]) == 0
    assert r2([1, 2], [10, 20]) < 0
    with pytest.raises(ValueError):
        r2([1], [1])


@pytest.mark.parametrize("truth,pred", [([0, 0], [0, 0]), ([1, 1], [0, 0]), ([0, 0], [1, 1])])
def test_zero_denominators(truth, pred):
    assert precision_recall_f1(truth, pred) == (0, 0, 0)
    assert confusion_matrix(truth, pred).shape == (2, 2)


@pytest.mark.parametrize("function", [rmse, mae, r2, confusion_matrix, precision_recall_f1, roc_auc])
@pytest.mark.parametrize("truth,pred", [([], []), ([0, 1], [1]), ([[0], [1]], [0, 1]), ([0, np.nan], [0, 1]), ([0, 1], [0, np.inf]), (["0", "1"], [0, 1]), ([0, 1j], [0, 1])])
def test_invalid_inputs(function, truth, pred):
    with pytest.raises(ValueError):
        function(truth, pred)


@pytest.mark.parametrize("function", [confusion_matrix, precision_recall_f1])
def test_invalid_predictions(function):
    with pytest.raises(ValueError):
        function([0, 1], [0, 2])


@pytest.mark.parametrize("truth", [[0, 0], [1, 1], [-1, 1]])
def test_invalid_auc_labels(truth):
    with pytest.raises(ValueError):
        roc_auc(truth, [0.2, 0.8])


def test_comparison_table(capsys):
    assert compare({"Tree": {"MAE": 10}, "SVM": {"F1": 0.8}}) is None
    text = capsys.readouterr().out
    assert all(word in text for word in ["Model", "Tree", "SVM", "MAE", "F1", "10.0000", "0.8000", "-"])
    compare({})
    assert "No model results" in capsys.readouterr().out


def test_existing_svm_outputs():
    """Integration with Farid's actual API, using development data only."""
    from src.svm import PegasosSVM
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = PegasosSVM(lambda_=0.1, n_iters=200, random_state=42).fit(X, y)
    scores, pred = model.decision_function(X), model.predict(X)
    assert roc_auc(y, scores) == pytest.approx(metrics.roc_auc_score(y, scores))
    assert precision_recall_f1(y, pred)[2] == pytest.approx(metrics.f1_score(y, pred))
