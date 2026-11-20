"""Aligned, nested cohort-spectrum comparison. All per-person outputs are private."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from research.common import ROOT, PRIVATE, sha256, write_json
from research.cohort_spectrum_data import LOCAL, PUBLIC
from research.core import candidate_models, pipeline, splits, oof, select_thresholds
from research.legacy_supplement import columns

COHORTS = ['broad', 'restricted', 'matched']
CONFIGS = ['age_only', 'full', 'no_age', 'sensors']


def signature(cohort, cols):
    return dict(data=sha256(LOCAL/f'{cohort}.parquet'), columns=cols,
                protocol=sha256(PUBLIC/'PROTOCOL.md'),
                code={n: sha256(ROOT/'research'/n) for n in
                      ['cohort_spectrum_run.py', 'core.py', 'legacy_supplement.py']})


def job(spec):
    warnings.filterwarnings('ignore', message='The `probability` parameter was deprecated', category=FutureWarning)
    cohort, config = spec
    d = pd.read_parquet(LOCAL/f'{cohort}.parquet')
    cols = columns(d, config); sig = signature(cohort, cols)
    dest = LOCAL/'runs'/f'{cohort}_{config}'; dest.mkdir(parents=True, exist_ok=True)
    complete = dest/'complete.json'
    if complete.exists():
        old = json.loads(complete.read_text()); assert old['signature'] == sig
        assert old['prediction_sha256'] == sha256(dest/'predictions.parquet')
        return dest.name
    X = d[cols].to_numpy(float); y = d.y.to_numpy(int); groups = d.pair_id.to_numpy()
    rows, choices = [], []
    for fold in range(5):
        tr = np.flatnonzero(d.fold.to_numpy() != fold); te = np.flatnonzero(d.fold.to_numpy() == fold)
        assert not set(groups[tr]) & set(groups[te])
        inner = splits(y[tr], groups[tr], n=3, seed=143)
        models = candidate_models(42)
        P = np.column_stack([oof(X[tr], y[tr], inner, m) for m in models.values()])
        scores = [roc_auc_score(y[tr], p) for p in P.T]
        winner = int(np.argmax(scores)); name = list(models)[winner]
        fitted = pipeline(models[name]).fit(X[tr], y[tr])
        q = fitted.predict_proba(X[te])[:, 1]
        thresholds = select_thresholds(y[tr], P[:, winner])
        modelpath = dest/f'fold{fold}.joblib'; joblib.dump(fitted, modelpath)
        np.savez_compressed(dest/f'fold{fold}_inner.npz', probabilities=P, train_indices=tr,
                            **{f'inner_train_{i}': a for i, (a, b) in enumerate(inner)},
                            **{f'inner_valid_{i}': b for i, (a, b) in enumerate(inner)})
        choices.append(dict(fold=fold, selected=name, candidate_auc=dict(zip(models, scores)),
                            thresholds=thresholds, train_indices=tr.tolist(), test_indices=te.tolist(),
                            model_sha256=sha256(modelpath), inner_sha256=sha256(dest/f'fold{fold}_inner.npz')))
        for policy, tau in thresholds.items():
            for i, prob in zip(te, q):
                rows.append(dict(participant_id=d.iloc[i].participant_id, pair_id=groups[i], fold=fold,
                                 y=int(y[i]), probability=float(prob), threshold=tau,
                                 prediction=int(prob >= tau), policy=policy))
        print(f'{cohort}/{config}: fold {fold+1}/5; selected {name}', flush=True)
    pd.DataFrame(rows).to_parquet(dest/'predictions.parquet', index=False)
    write_json(dest/'choices.json', choices)
    write_json(complete, dict(signature=sig, prediction_sha256=sha256(dest/'predictions.parquet'),
                             choices_sha256=sha256(dest/'choices.json')))
    return dest.name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    data = json.loads((PUBLIC/'data_verification.json').read_text())
    assert data['status'] == 'passed' and data['protocol_sha256'] == sha256(PUBLIC/'PROTOCOL.md')
    for c, h in data['cohort_hashes'].items(): assert sha256(LOCAL/f'{c}.parquet') == h
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(job, (c, k)) for c in COHORTS for k in CONFIGS]):
            print('Complete:', f.result(), flush=True)
