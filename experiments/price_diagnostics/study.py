"""Outlier, overfitting and feature diagnostics for the price models.

Model choices are made on inner splits of the deduplicated official train part;
the official validation split is used only for the error audit of the selected
model. The test split is never scored. Every fit is independent and seeded, so
the parallel run gives the same numbers as a serial one.
"""
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from src import data_prep
from src.decision_tree import DecisionTree
from src.ensemble import BaggedTreeRegressor
from src.price_models import PriceRegressor
from experiments.price_quality import price_metrics, error_report
from .neighbour_feature import NeighbourPriceFeature, NeighbourPriceRegressor

INNER_SEEDS = (11, 23, 37)
CURVE_SEEDS = (11, 23)
REGULARIZATION = (dict(name='deep', depth=24, leaf=1, drop_extreme=False),
                  dict(name='leaf3', depth=24, leaf=3, drop_extreme=False),
                  dict(name='moderate', depth=16, leaf=5, drop_extreme=False),
                  dict(name='leaf3_drop_extreme', depth=24, leaf=3, drop_extreme=True))
NEIGHBOUR = (dict(name='nb_k10_category', k=10, by_category=True),
             dict(name='nb_k25_category', k=25, by_category=True),
             dict(name='nb_k10_all', k=10, by_category=False))
COMPLEXITY = ((6, 1), (10, 1), (14, 1), (18, 1), (24, 1), (32, 1), (24, 3), (24, 10), (24, 30))
SINGLE_DEPTHS = (6, 8, 10, 12, 14, 16, 20, 24, None)
SINGLE_LEAVES = (1, 5, 10, 20, 50, 100)
_STATE = {}


def _bagging(depth, leaf, n_estimators):
    return BaggedTreeRegressor(n_estimators=n_estimators, max_depth=depth, min_samples_leaf=leaf,
                               max_features=1., random_state=42)


def _log_area(estimator, names):
    return PriceRegressor(estimator, target='log_price_per_area', area_index=names.index('area_m2'))


def _ppm_bounds(train):
    """Category log-ppm limits Q1 - 3 IQR .. Q3 + 3 IQR from training rows only."""
    z = np.log(train.price / train.area_m2)
    bounds = {}
    for value, group in z.groupby(train.category):
        q1, q3 = np.quantile(group, [.25, .75])
        bounds[value] = (q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1))
    return bounds


def _is_extreme(rows, bounds):
    z = np.log(rows.price / rows.area_m2).to_numpy()
    low = rows.category.map(lambda c: bounds[c][0]).to_numpy()
    high = rows.category.map(lambda c: bounds[c][1]).to_numpy()
    return (z < low) | (z > high)


def _drop_extreme_ppm(train):
    return train[~_is_extreme(train, _ppm_bounds(train))]


def inner_splits(development, seeds):
    """Deduplicate the development frame, then yield (seed, inner-train, inner-validation)."""
    development = development.drop_duplicates().reset_index(drop=True)
    for seed in seeds:
        a, b, _ = data_prep._split_indices(development.price.to_numpy(), val_size=.2, test_size=0, seed=seed)
        yield seed, development.iloc[a], development.iloc[b]


def _fit_and_score(model, train, val, encoder_rows=None):
    encoder = data_prep.FeatureEncoder().fit(train if encoder_rows is None else encoder_rows)
    X, y, names = data_prep.make_features(train, encoder)
    V, v, _ = data_prep.make_features(val, encoder)
    model = model(names)
    start = time.perf_counter()
    model.fit(X, y)
    return dict(seconds=time.perf_counter() - start, train=price_metrics(y, model.predict(X)),
                validation=price_metrics(v, model.predict(V)))


def _init(development):
    """Worker initializer: share the official-train frame once per process."""
    _STATE['development'] = development


def _split(seed):
    return next(inner_splits(_STATE['development'], (seed,)))[1:]


def _regularization_task(seed, config, n_estimators):
    train, val = _split(seed)
    fit_rows = _drop_extreme_ppm(train) if config['drop_extreme'] else train
    model = lambda names: _log_area(_bagging(config['depth'], config['leaf'], n_estimators), names)
    return dict(seed=seed, **config, dropped=len(train) - len(fit_rows), **_fit_and_score(model, fit_rows, val))


def _neighbour_task(seed, config, n_estimators):
    train, val = _split(seed)

    def model(names):
        groups = [i for i, n in enumerate(names) if n.startswith('cat_')] if config['by_category'] else ()
        feature = NeighbourPriceFeature(names.index('area_m2'), names.index('lat'), names.index('lng'),
                                        k=config['k'], group_indices=groups)
        return NeighbourPriceRegressor(_log_area(_bagging(24, 1, n_estimators), names), feature)
    return dict(seed=seed, **config, **_fit_and_score(model, train, val))


def _complexity_task(seed, depth, leaf, n_estimators):
    train, val = _split(seed)
    model = lambda names: _log_area(_bagging(depth, leaf, n_estimators), names)
    return dict(seed=seed, depth=depth, leaf=leaf, **_fit_and_score(model, train, val))


def _learning_task(seed, fraction, n_estimators):
    train, val = _split(seed)
    rows = train.iloc[np.random.default_rng(seed).permutation(len(train))[:int(len(train) * fraction)]]
    model = lambda names: _log_area(_bagging(24, 1, n_estimators), names)
    # Encoder stays fitted on the full inner-train fold so the feature set is fixed.
    return dict(seed=seed, fraction=fraction, rows=len(rows), **_fit_and_score(model, rows, val, encoder_rows=train))


def _trees_task(seed, curve_trees):
    train, val = _split(seed)
    encoder = data_prep.FeatureEncoder().fit(train)
    X, y, names = data_prep.make_features(train, encoder)
    V, v, _ = data_prep.make_features(val, encoder)
    bag = _bagging(24, 1, curve_trees)
    _log_area(bag, names).fit(X, y)
    area = V[:, names.index('area_m2')]
    return dict(seed=seed, curve=[dict(trees=i, **price_metrics(v, np.exp(p) * area))
                                  for i, p in enumerate(bag.staged_predict(V), start=1)])


def _single_tree_task(seed, depth, leaf):
    """Task A single tree on the raw price target."""
    train, val = _split(seed)
    encoder = data_prep.FeatureEncoder().fit(train)
    X, y, _ = data_prep.make_features(train, encoder)
    V, v, _ = data_prep.make_features(val, encoder)
    tree = DecisionTree(task='regression', criterion='mse', max_depth=depth, min_samples_leaf=leaf)
    start = time.perf_counter()
    tree.fit(X, y)
    return dict(seed=seed, max_depth=depth, min_samples_leaf=leaf, seconds=time.perf_counter() - start,
                fitted_depth=tree.get_depth(), n_leaves=tree.get_n_leaves(),
                train=price_metrics(y, tree.predict(X)), validation=price_metrics(v, tree.predict(V)))


def diagnostic_tasks(fast=False):
    """(output key, function, kwargs) for every independent fit, longest first."""
    n = 2 if fast else 8
    tasks = [('trees', _trees_task, dict(seed=s, curve_trees=3 if fast else 24)) for s in CURVE_SEEDS]
    tasks += [('regularization', _regularization_task, dict(seed=s, config=c, n_estimators=n))
              for s in INNER_SEEDS for c in REGULARIZATION]
    tasks += [('neighbour', _neighbour_task, dict(seed=s, config=c, n_estimators=n))
              for s in INNER_SEEDS for c in NEIGHBOUR]
    tasks += [('complexity', _complexity_task, dict(seed=s, depth=d, leaf=l, n_estimators=n))
              for s in CURVE_SEEDS for d, l in COMPLEXITY]
    tasks += [('learning', _learning_task, dict(seed=s, fraction=f, n_estimators=n))
              for s in CURVE_SEEDS for f in (.25, .5, .75, 1.)]
    depths, leaves = ((4, None), (1, 20)) if fast else (SINGLE_DEPTHS, SINGLE_LEAVES)
    tasks += [('single_tree', _single_tree_task, dict(seed=s, depth=d, leaf=l))
              for s in INNER_SEEDS for d in depths for l in leaves]
    return tasks


def _call(task):
    _, function, kwargs = task
    return function(**kwargs)


def run_tasks(development, tasks, workers=1):
    """Run independent seeded fits; results keep task order, so they match a serial run."""
    if workers <= 1:
        _init(development)
        results = [_call(task) for task in tasks]
    else:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(development,)) as pool:
            results = list(pool.map(_call, tasks))
    grouped = {}
    for (key, _, _), result in zip(tasks, results):
        grouped.setdefault(key, []).append(result)
    return grouped


def validation_audit(splits, frame, validation_rows, train_rows, n_estimators=48, depth=32):
    """Error audit of the selected model on the official validation split."""
    names = splits['feature_names']
    model = _log_area(_bagging(depth, 1, n_estimators), names).fit(splits['X_tr'], splits['y_tr'])
    prediction = model.predict(splits['X_val'])
    val = frame.iloc[validation_rows].reset_index(drop=True).assign(prediction=prediction)
    features = [c for c in frame.columns if c != 'price']
    seen = set(map(tuple, frame.iloc[train_rows][features].to_numpy()))
    twin = np.array([tuple(r) in seen for r in val[features].to_numpy()])
    ratio = val.price / val.prediction
    bins = pd.qcut(val.prediction, [0, .25, .5, .75, .9, .97, 1])
    calibration = [dict(bin=str(b), count=int(len(g)), median_ratio=float(g.median()), mean_ratio=float(g.mean()))
                   for b, g in ratio.groupby(bins, observed=True)]
    extreme = _is_extreme(val, _ppm_bounds(frame.iloc[train_rows]))
    worst = val.assign(absolute_error=(val.price - prediction).abs(), price_per_m2=val.price / val.area_m2,
                       extreme_ppm=extreme).nlargest(30, 'absolute_error')
    return dict(report=error_report(val.price.to_numpy(), prediction),
                duplicate_twins=dict(count=int(twin.sum()),
                                     with_twin=price_metrics(val.price[twin], prediction[twin]) if twin.any() else None,
                                     without_twin=price_metrics(val.price[~twin], prediction[~twin])),
                calibration=calibration,
                extreme_ppm=dict(validation_count=int(extreme.sum()), in_worst_30=int(worst.extreme_ppm.sum())),
                worst_30=json.loads(worst.to_json(orient='records', force_ascii=False)))


def _audit(clean_path, validation_rows, train_rows, n_estimators):
    frame = pd.read_csv(clean_path)
    splits = data_prep.get_splits(path=str(clean_path))
    return validation_audit(splits, frame, validation_rows, train_rows, n_estimators=n_estimators)


def run_price_diagnostics(clean_path, output, fast=False, workers=None):
    clean_path, output = Path(clean_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Price diagnostics require a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / 'src').glob('*.py')) + sorted(Path(__file__).parent.glob('*.py')) \
        + [root / 'experiments' / 'price_quality.py']
    workers = workers or max(2, min(12, (os.cpu_count() or 2) - 2))
    manifest = dict(status='running', protocol=__doc__.strip(), workers=workers,
                    clean_sha256=hashlib.sha256(clean_path.read_bytes()).hexdigest(),
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
        print('saved', name, flush=True)

    save('manifest.json', manifest)
    try:
        frame = pd.read_csv(clean_path)
        train_rows, validation_rows, _ = data_prep._split_indices(frame.price.to_numpy())
        development = frame.iloc[train_rows]
        manifest['duplicates'] = dict(all_rows=int(frame.duplicated().sum()),
                                      official_train=int(development.duplicated().sum()))
        with ProcessPoolExecutor(1) as audit_pool:
            # The 48-tree audit is the longest single fit, so it runs beside the CV pool.
            audit = audit_pool.submit(_audit, clean_path, validation_rows, train_rows, 2 if fast else 48)
            grouped = run_tasks(development, diagnostic_tasks(fast), max(1, workers - 1))
            save('regularization_cv.json', grouped['regularization'])
            save('neighbour_cv.json', grouped['neighbour'])
            save('overfitting.json', dict(complexity=grouped['complexity'], learning=grouped['learning'],
                                          trees=grouped['trees']))
            save('single_tree_grid.json', grouped['single_tree'])
            save('validation_audit.json', audit.result())
        manifest['status'] = 'complete'
    except Exception:
        manifest['status'] = 'failed'
        raise
    finally:
        save('manifest.json', manifest)
