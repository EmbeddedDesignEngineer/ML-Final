import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.evaluation_plots import (
    plot_confusion_matrix, plot_actual_vs_predicted, plot_feature_values,
    summarize_errors_by_group, summarize_price_errors,
)


def test_saved_plots(tmp_path):
    paths = [
        plot_confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1], tmp_path / "cm.png"),
        plot_actual_vs_predicted([100, 200, 300], [110, 190, 300], tmp_path / "price.png"),
        plot_actual_vs_predicted([5, 5], [5, 5], tmp_path / "constant.png"),
        plot_feature_values(["area", "rooms"], [0.5, -0.3], tmp_path / "weights.png"),
    ]
    for path in paths:
        pixels = plt.imread(path)
        assert pixels.shape[0] > 100 and pixels.shape[1] > 100
        assert pixels.std() > 0
    assert plt.get_fignums() == []


def test_group_errors():
    result = summarize_errors_by_group([100, 200, 300], [110, 180, 330], ["A", "A", "B"])
    assert result[0]["group"] == "A"
    assert result[0]["count"] == 2
    assert result[0]["mae"] == 15
    assert result[0]["rmse"] == pytest.approx(np.sqrt(250))
    assert result[0]["bias"] == -5
    assert result[1]["count"] == 1
    assert result[1]["mae"] == 30


def test_price_threshold():
    result = summarize_price_errors([100, 200, 300], [110, 180, 330], 200)
    groups = {row["group"]: row for row in result}
    assert groups["Expensive"]["count"] == 1
    assert groups["Other"]["count"] == 2
    assert groups["Expensive"]["mae"] == 30


@pytest.mark.parametrize("groups", [["A"], [["A"], ["B"]], [None, "B"], [np.nan, "B"]])
def test_invalid_groups(groups):
    with pytest.raises(ValueError):
        summarize_errors_by_group([1, 2], [1, 2], groups)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), "200"])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        summarize_price_errors([1, 2], [1, 2], threshold)


def test_invalid_feature_names(tmp_path):
    with pytest.raises(ValueError):
        plot_feature_values(["area"], [1, 2], tmp_path / "bad.png")
