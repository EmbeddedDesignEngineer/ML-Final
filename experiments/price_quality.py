"""Compare price models and expose errors by fixed price bands."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from src import data_prep, evaluate
from src.ensemble import BaggedTreeRegressor
from src.price_models import BoostedTreeRegressor, PriceRegressor


def price_metrics(actual, predicted):
    actual, predicted = evaluate._check_arrays(actual, predicted)
    if np.any(actual <= 0):
        raise ValueError('Actual prices must be positive')
    error = np.abs(actual - predicted)
    relative = error / actual
    return dict(count=len(actual), rmse=evaluate.rmse(actual, predicted),
                mae=evaluate.mae(actual, predicted), median_ae=float(np.median(error)),
                r2=evaluate.r2(actual, predicted) if len(actual) > 1 else None,
                p90_ae=float(np.quantile(error, .9)),
                median_ape=float(np.median(relative)),
                within10=float(np.mean(relative <= .1)), within20=float(np.mean(relative <= .2)))


def error_report(actual, predicted):
    actual, predicted = evaluate._check_arrays(actual, predicted)
    result = dict(overall=price_metrics(actual, predicted), price_bands={})
    for low, high in ((0, 100000), (100000, 200000), (200000, 500000), (500000, np.inf)):
        mask = (actual >= low) & (actual < high)
        if mask.any():
            result['price_bands'][f'{low}-{high}'] = price_metrics(actual[mask], predicted[mask])
    squared = (actual - predicted) ** 2
    result['top_one_percent_squared_error_share'] = float(
        np.sort(squared)[-max(1, int(np.ceil(len(actual) * .01))):].sum() / squared.sum()
    ) if squared.sum() else 0.
    return result


def configurations():
    return [
        dict(name='previous_bagging', kind='bagging', target='price', params=dict(max_features=.8)),
        dict(name='all_features_bagging', kind='bagging', target='price', params=dict(max_features=1.)),
        dict(name='log_area_bagging', kind='bagging', target='log_price_per_area', params=dict(max_features=1.)),
        dict(name='residual_boosting', kind='boosting', target='price', params=dict(n_estimators=100, max_depth=5)),
        dict(name='log_area_boosting', kind='boosting', target='log_price_per_area', params=dict(n_estimators=100, max_depth=5)),
        dict(name='deep_bagging', kind='bagging', target='price', params=dict(max_features=1., max_depth=24, min_samples_leaf=1)),
        dict(name='deep_log_area_bagging', kind='bagging', target='log_price_per_area', params=dict(max_features=1., max_depth=24, min_samples_leaf=1)),
    ]


def compare(splits, configs, on_result=None):
    X, y, V, actual = (splits[k] for k in ('X_tr', 'y_tr', 'X_val', 'y_val'))
    area = splits['feature_names'].index('area_m2')
    rows = []
    for name, value in [('train_median', np.median(y)), ('train_mean', np.mean(y))]:
        rows.append(dict(name=name, report=error_report(actual, np.full(len(actual), value))))
    for config in configs:
        start = time.perf_counter()
        constructors = dict(bagging=BaggedTreeRegressor, boosting=BoostedTreeRegressor)
        estimator = constructors[config['kind']](**config['params'])
        model = PriceRegressor(estimator, target=config['target'], area_index=area).fit(X, y)
        prediction = model.predict(V)
        rows.append(dict(**config, seconds=time.perf_counter() - start,
                         report=error_report(actual, prediction)))
        if on_result:
            on_result(rows)
    return rows


def run_price_study(clean_path, output):
    clean_path, output = Path(clean_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Price study requires a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    with clean_path.open('rb') as handle:
        data_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    configs = configurations()
    root = Path(__file__).resolve().parents[1]
    sources = list((root / 'src').glob('*.py')) + [Path(__file__)]
    manifest = dict(status='running', clean_sha256=data_hash, configurations=configs,
                    protocol='Seed 42; fit train only, evaluate validation only; historic test not scored.',
                    source_sha256={p.name: hashlib.sha256(p.read_text().encode()).hexdigest() for p in sources})
    def save_manifest():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    def save(rows):
        (output / 'errors.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(rows[-1]['name'], rows[-1]['report']['overall'], flush=True)
    save_manifest()
    try:
        splits = data_prep.get_splits(path=str(clean_path))
        rows = compare(splits, configs, save)
        selected = {metric: min(rows[2:], key=lambda r: r['report']['overall'][metric])['name']
                    for metric in ('rmse', 'mae', 'median_ape')}
        (output / 'selected.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
        manifest['status'] = 'complete'
        return rows
    except Exception:
        manifest['status'] = 'failed'
        raise
    finally:
        save_manifest()
