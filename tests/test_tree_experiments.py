"""Tree-only validation sweeps for Bailar's runner; never accepts test data."""
import numpy as np
from numpy.testing import assert_array_equal

from experiments.tree_experiments import compare_tree_configs


def test_validation_sweep_compares_matched_models_and_keeps_inputs():
    X = np.arange(8.).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    before = X.copy()
    rows, best = compare_tree_configs(X, y, X, y, task="classification",
                                    depths=(1, 2), leaf_sizes=(1,))
    assert len(rows) == 8  # two criteria, two depths, two implementations
    assert set(best) == {"custom", "sklearn"}
    assert all(row["validation_score"] == 1 for row in rows)
    assert all(row["n_leaves"] == 2 for row in rows)
    assert best["custom"]["max_depth"] == 1  # first configuration wins a tie
    assert_array_equal(X, before)


def test_regression_sweep_minimizes_rmse():
    X = np.arange(6.).reshape(-1, 1)
    y = np.array([0., 0., 0., 9., 9., 9.])
    rows, best = compare_tree_configs(X, y, X, y, task="regression",
                                    depths=(1, 2), leaf_sizes=(1,))
    assert len(rows) == 4
    assert best["custom"]["validation_score"] == 0
    assert all(row["metric"] == "rmse" for row in rows)
