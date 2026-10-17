"""Development-only comparison; never scores the held-out test partition."""
import hashlib
import json
import time
from pathlib import Path
import pandas as pd

from src import data_prep, evaluate
from src.decision_tree import DecisionTree
from src.ensemble import BaggedTreeRegressor
from src.svm import PegasosSVM


def configurations():
    configs = [
        dict(name='tree_previous', kind='tree', params=dict(max_depth=12, min_samples_leaf=5)),
        dict(name='tree_deeper', kind='tree', params=dict(max_depth=16, min_samples_leaf=10)),
        dict(name='bagged_tree', kind='bagging', params=dict(n_estimators=24, max_depth=16,
             min_samples_leaf=5, max_samples=.8, max_features=.8, random_state=42)),
    ]
    for mode, extra in [('linear', dict(lambda_=1e-4)),
                        ('rff', dict(lambda_=.001, n_components=512, gamma=.01))]:
        schedule = 'constant' if mode == 'linear' else 'pegasos'
        configs.append(dict(name=f'{mode}_previous', kind='svm', params=dict(
            **extra, learning_rate=schedule, n_iters=20000)))
        for epochs in (5, 20):
            for rate in ('constant', 'pegasos'):
                configs.append(dict(name=f'{mode}_{rate}_{epochs}_averaged', kind='svm',
                    params=dict(**extra, learning_rate=rate, n_epochs=epochs, average=True)))
    return configs


def compare(splits, configs, on_result=None):
    train, validation = data_prep.standardize(splits['X_tr'], splits['X_val'])
    rows = []
    for config in configs:
        kind, params = config['kind'], config['params']
        started = time.perf_counter()
        if kind == 'svm':
            steps = params.get('n_epochs', 0) * len(train) or params.get('n_iters', 20000)
            model = PegasosSVM(**params, loss_interval=max(1, steps // 10)).fit(train, splits['t_tr'])
            scores = model.decision_function(validation)
            metrics = dict(f1=evaluate.precision_recall_f1(splits['t_val'], scores >= 0)[2],
                           auc=evaluate.roc_auc(splits['t_val'], scores),
                           train_objective=model.objective(train, splits['t_tr']),
                           updates=model.n_iter_)
        else:
            if kind == 'tree':
                model = DecisionTree(task='regression', criterion='mse', **params)
            elif kind == 'bagging':
                model = BaggedTreeRegressor(**params)
            else:
                raise ValueError(f'Unknown model kind: {kind}')
            model.fit(splits['X_tr'], splits['y_tr'])
            prediction = model.predict(splits['X_val'])
            metrics = dict(rmse=evaluate.rmse(splits['y_val'], prediction),
                           mae=evaluate.mae(splits['y_val'], prediction),
                           r2=evaluate.r2(splits['y_val'], prediction))
        row = dict(**config, validation=metrics, seconds=time.perf_counter() - started)
        rows.append(row)
        if on_result:
            on_result(rows)
    return rows


def run_quality_study(raw_path, output):
    raw_path, output = Path(raw_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configs = configurations()
    with raw_path.open('rb') as handle:
        raw_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    manifest = dict(protocol='Train/validation only; historic test already observed.',
                    seed=42, configurations=configs,
                    raw_sha256=raw_hash)
    sources = list(Path('src').glob('*.py')) + [Path(__file__)]
    manifest['source_sha256'] = {p.name: hashlib.sha256(p.read_text().encode()).hexdigest()
                                 for p in sources}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    clean = output / 'clean.csv'
    data_prep.clean(pd.read_csv(raw_path)).to_csv(clean, index=False)
    splits = data_prep.get_splits(path=str(clean), seed=42)

    def save(rows):
        (output / 'validation.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(rows[-1]), flush=True)

    return compare(splits, configs, on_result=save)
