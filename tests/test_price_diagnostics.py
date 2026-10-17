import json
import numpy as np
import pandas as pd
import pytest
from experiments.price_diagnostics.study import (_drop_extreme_ppm, diagnostic_tasks, inner_splits,
                                                run_price_diagnostics, run_tasks)
from experiments.price_diagnostics.neighbour_feature import NeighbourPriceFeature


def _frame(rows=600):
    rng = np.random.default_rng(0)
    category = rng.choice(['new_build', 'old_build', 'house'], rows)
    area = rng.uniform(40, 200, rows).round()
    lat, lng = 40.4 + rng.normal(0, .02, rows), 49.85 + rng.normal(0, .02, rows)
    frame = pd.DataFrame(dict(
        price=(area * (2000 + 20000 * (lat - 40.4))).round(), category=category, area_m2=area,
        rooms=rng.integers(1, 5, rows), floor=rng.integers(1, 9, rows), total_floors=9,
        land_sot=np.where(category == 'house', 2., 0.), is_repaired=1, has_bill_of_sale=1, has_mortgage=0,
        is_agency=0, is_complex=0, is_vip=0, is_featured=0, location='Yasamal r.', lat=lat, lng=lng))
    return pd.concat([frame, frame.iloc[:20]], ignore_index=True)


def test_inner_splits_drop_duplicates_and_do_not_overlap():
    frame = _frame()
    for _, train, val in inner_splits(frame, (1, 2)):
        assert not train.duplicated().any()
        assert not set(map(tuple, train.to_numpy())) & set(map(tuple, val.to_numpy()))
        assert len(train) + len(val) == len(frame) - 20


def test_extreme_price_per_m2_rows_are_dropped_from_training_only():
    frame = _frame().drop_duplicates()
    frame.loc[0, 'price'] = 1.
    kept = _drop_extreme_ppm(frame)
    assert 0 not in kept.index and len(kept) >= len(frame) - 5


def test_neighbour_feature_uses_out_of_fold_prices():
    X = np.column_stack([np.full(40, 50.), np.r_[np.full(20, 40.), np.full(20, 41.)], np.full(40, 49.)])
    y = np.r_[np.full(20, 50000.), np.full(20, 250000.)]
    y[0] = 5e7
    values = NeighbourPriceFeature(0, 1, 2, k=5).fit_transform(X, y)
    np.testing.assert_allclose(values[:2], np.log(1000))
    np.testing.assert_allclose(NeighbourPriceFeature(0, 1, 2, k=5).fit_transform(X, y)[20], np.log(5000))


def test_fast_diagnostics_write_complete_outputs_and_refuse_reuse(tmp_path):
    clean = tmp_path / 'clean.csv'
    _frame().to_csv(clean, index=False)
    run_price_diagnostics(clean, tmp_path / 'out', fast=True, workers=2)
    manifest = json.loads((tmp_path / 'out' / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete'
    assert manifest['duplicates']['all_rows'] == 20
    audit = json.loads((tmp_path / 'out' / 'validation_audit.json').read_text(encoding='utf-8'))
    assert len(audit['worst_30']) == 30
    curves = json.loads((tmp_path / 'out' / 'overfitting.json').read_text(encoding='utf-8'))
    assert [p['trees'] for p in curves['trees'][0]['curve']] == [1, 2, 3]
    with pytest.raises(ValueError):
        run_price_diagnostics(clean, tmp_path / 'out', fast=True, workers=2)


def test_parallel_tasks_match_serial_results():
    frame = _frame()
    tasks = [t for t in diagnostic_tasks(fast=True) if t[0] in ('complexity', 'single_tree')][:6]
    serial, parallel = run_tasks(frame, tasks, workers=1), run_tasks(frame, tasks, workers=3)
    for key in serial:
        for a, b in zip(serial[key], parallel[key]):
            assert a['validation'] == b['validation'] and a['train'] == b['train']
