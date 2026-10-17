import numpy as np
import pytest
from experiments.price_quality import price_metrics, error_report, compare


def test_price_error_metrics_have_known_values():
    result = price_metrics([100, 200, 400], [100, 240, 200])
    assert result['mae'] == 80
    assert result['median_ae'] == 40
    assert result['within10'] == 1/3
    assert result['within20'] == 2/3
    assert result['median_ape'] == .2
    with pytest.raises(ValueError):
        price_metrics([0], [10])


def test_bands_cover_every_row_and_perfect_predictions():
    actual = np.array([50000, 100000, 200000, 500000])
    report = error_report(actual, actual)
    assert sum(r['count'] for r in report['price_bands'].values()) == 4
    assert report['top_one_percent_squared_error_share'] == 0
    assert report['overall']['within10'] == 1


def test_integer_prices_do_not_overflow_squared_error_share():
    report = error_report(np.array([100000, 200000, 300000], dtype=np.int32),
                          np.array([200000, 400000, 300000], dtype=np.int32))
    assert report['top_one_percent_squared_error_share'] == .8


def test_price_comparison_does_not_require_test_data():
    X = np.arange(1., 41.).reshape(-1, 1)
    splits = dict(X_tr=X[:30], y_tr=X[:30, 0]*1000,
                  X_val=X[30:], y_val=X[30:, 0]*1000, feature_names=['area_m2'])
    configs = [dict(name='ppm', kind='bagging', target='log_price_per_area', params=dict(n_estimators=2))]
    result = compare(splits, configs)
    assert result[-1]['report']['overall']['mae'] < 1e-8
    assert result[0]['report']['overall']['mae'] > 0
