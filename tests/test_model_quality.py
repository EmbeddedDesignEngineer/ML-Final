import numpy as np
import pytest

from experiments.model_quality import compare, configurations


def test_comparison_needs_only_train_and_validation():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 3))
    splits = dict(X_tr=X[:60], X_val=X[60:], y_tr=100 + X[:60, 0],
                  y_val=100 + X[60:, 0], t_tr=(X[:60, 0] > 0).astype(int),
                  t_val=(X[60:, 0] > 0).astype(int))
    configs = [dict(name='tree', kind='tree', params=dict(max_depth=3)),
               dict(name='bagging', kind='bagging', params=dict(n_estimators=3)),
               dict(name='svm', kind='svm', params=dict(n_epochs=5, average=True))]
    first = compare(splits, configs)
    splits.update(X_te=np.full((20, 3), np.nan), y_te=np.full(20, np.inf))
    second = compare(splits, configs)
    assert [r['validation'] for r in first] == [r['validation'] for r in second]
    assert first[-1]['validation']['updates'] == 300
    assert first[0]['validation']['rmse'] < 1


def test_grid_preserves_baselines_and_has_unique_names():
    configs = configurations()
    assert len({c['name'] for c in configs}) == len(configs)
    old = [c for c in configs if c['name'].endswith('_previous')]
    assert len(old) == 3
    assert all(not c['params'].get('average', False) for c in old)


def test_cli_routes_quality_study_without_full_pipeline(monkeypatch, tmp_path):
    from src import run_all
    from experiments import model_quality
    calls = []
    monkeypatch.setattr('sys.argv', ['run_all', '--quality-study', '--output', str(tmp_path)])
    monkeypatch.setattr(model_quality, 'run_quality_study', lambda raw, output: calls.append((raw, output)))
    monkeypatch.setattr(run_all, 'run_pipeline', lambda *a, **kw: (_ for _ in ()).throw(AssertionError()))
    run_all.main()
    assert calls == [(run_all.Path('data/house_sale.csv'), tmp_path)]


def test_study_preserves_existing_evidence(tmp_path):
    from experiments.model_quality import run_quality_study
    evidence = tmp_path / 'validation.json'
    evidence.write_text('previous result')
    with pytest.raises(ValueError, match='empty output'):
        run_quality_study('missing.csv', tmp_path)
    assert evidence.read_text() == 'previous result'
