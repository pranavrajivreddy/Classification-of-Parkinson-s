"""Post-audit exploratory replacements; never imports historical fitted models."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from research.common import ROOT, OUT, PRIVATE, sha256, write_json, manifest
from research.core import feature_sets, splits, candidate_models, pipeline, oof, select_thresholds, metrics

PUBLIC = OUT/'legacy_review'
LOCAL = PRIVATE/'legacy_review'
SEED = 42


def columns(d, config):
    f = feature_sets(d)
    return {'full': f['full'], 'no_age': [c for c in f['full'] if c != 'demo_age'],
            'sensors': f['sensors'], 'age_only': ['demo_age']}[config]


def select_fit(X, y, groups, test):
    inner = splits(y, groups, n=3, seed=143)
    models = candidate_models(SEED)
    probabilities = [oof(X, y, inner, m) for m in models.values()]
    scores = [roc_auc_score(y, p) for p in probabilities]
    winner = int(np.argmax(scores))
    name = list(models)[winner]
    fitted = pipeline(models[name]).fit(X, y)
    return fitted.predict_proba(test)[:, 1], select_thresholds(y, probabilities[winner]), name, fitted


def job(spec):
    warnings.filterwarnings('ignore', message='The `probability` parameter was deprecated', category=FutureWarning)
    cohort, config = spec
    source = PRIVATE/('cohort.parquet' if cohort == 'matched' else 'eligible.parquet')
    d = pd.read_parquet(source).reset_index(drop=True)
    if cohort != 'matched': d['pair_id'] = d.participant_id
    cols = columns(d, config)
    dest = LOCAL/f'{cohort}_{config}'
    dest.mkdir(parents=True, exist_ok=True)
    signature = dict(data=sha256(source), code={n: sha256(ROOT/'research'/n) for n in
        ['legacy_supplement.py', 'core.py']}, plan=sha256(OUT/'LEGACY_CORRECTION_PLAN.md'), columns=cols)
    done = dest/'complete.json'
    if done.exists():
        old = json.loads(done.read_text())
        assert old['signature'] == signature, f'Archive stale job before rerun: {dest}'
        assert sha256(dest/'predictions.parquet') == old['prediction_sha256']
        return dest.name
    X = d[cols].to_numpy(float); y = d.y.to_numpy(int); groups = d.pair_id.to_numpy()
    rows = []; choices = []; importance = []
    for fold, (tr, te) in enumerate(splits(y, groups, n=5, seed=SEED)):
        q, thresholds, selected, fitted = select_fit(X[tr], y[tr], groups[tr], X[te])
        choices.append(dict(fold=fold, selected=selected, thresholds=thresholds,
                            train_indices=tr.tolist(), test_indices=te.tolist()))
        for policy, tau in thresholds.items():
            for i, prob in zip(te, q):
                rows.append(dict(participant_id=d.iloc[i].participant_id, pair_id=groups[i],
                    fold=fold, y=int(y[i]), probability=float(prob), threshold=tau,
                    prediction=int(prob >= tau), policy=policy))
        if cohort == 'matched' and config == 'full':
            base_auc = roc_auc_score(y[te], q)
            f = feature_sets(d)
            for block in ['demographics', 'tapping', 'walking', 'voice']:
                indices = [cols.index(c) for c in f[block]]
                for repeat in range(30):
                    rng = np.random.default_rng(18000 + 100*fold + repeat)
                    z = X[te].copy(); z[:, indices] = X[te][rng.permutation(len(te))][:, indices]
                    perm_auc = roc_auc_score(y[te], fitted.predict_proba(z)[:, 1])
                    importance.append(dict(fold=fold, block=block, repeat=repeat, n=len(te),
                        baseline_auc=base_auc, permuted_auc=perm_auc, auc_drop=base_auc-perm_auc))
        print(f'{cohort}/{config} fold {fold+1}/5', flush=True)
    pd.DataFrame(rows).to_parquet(dest/'predictions.parquet', index=False)
    write_json(dest/'choices.json', choices)
    if importance: pd.DataFrame(importance).to_csv(dest/'importance.csv', index=False)
    write_json(done, dict(signature=signature, prediction_sha256=sha256(dest/'predictions.parquet')))
    return dest.name


def paired_associations(d):
    markers = ['tap_rate', 'tap_flight_iqr', 'walk_speed_mean', 'voice_jitter_local']
    rows = []
    for feature in markers:
        wide = d.pivot(index='pair_id', columns='y', values=[feature, 'demo_age'])
        v = (wide[feature][1]-wide[feature][0]).to_numpy(float)
        age = (wide.demo_age[1]-wide.demo_age[0]).to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(age); v = v[ok]; age = age[ok]
        design = np.column_stack([np.ones(len(v)), age])
        assert np.linalg.matrix_rank(design) == 2
        adjusted = np.linalg.lstsq(design, v, rcond=None)[0][0]
        rng = np.random.default_rng(20261118); boots = []
        for _ in range(2000):
            idx = rng.integers(len(v), size=len(v))
            assert np.linalg.matrix_rank(design[idx]) == 2
            boots.append([v[idx].mean(), np.linalg.lstsq(design[idx], v[idx], rcond=None)[0][0]])
        ci = np.percentile(boots, [2.5, 97.5], axis=0)
        for j, (name, value) in enumerate([('paired_difference', v.mean()), ('age_adjusted_pair_difference', adjusted)]):
            rows.append(dict(feature=feature, contrast='case minus control', estimator=name,
                complete_pairs=len(v), excluded_pairs=len(wide)-len(v), estimate=float(value),
                ci_low=ci[0,j], ci_high=ci[1,j], interpretation='descriptive association; not causal'))
    return pd.DataFrame(rows)


def report():
    PUBLIC.mkdir(exist_ok=True); d = pd.read_parquet(PRIVATE/'cohort.parquet')
    summary = []; prediction_sets = {}; counts = 0
    for cohort in ['matched', 'eligible']:
        source = pd.read_parquet(PRIVATE/('cohort.parquet' if cohort == 'matched' else 'eligible.parquet'))
        if cohort != 'matched': source['pair_id'] = source.participant_id
        truth = source.set_index('participant_id'); reference_folds = None
        for config in ['full', 'no_age', 'sensors', 'age_only']:
            dest = LOCAL/f'{cohort}_{config}'
            p = pd.read_parquet(dest/'predictions.parquet')
            completion = json.loads((dest/'complete.json').read_text())
            assert completion['prediction_sha256'] == sha256(dest/'predictions.parquet')
            assert all(sha256(ROOT/'research'/n) == h for n,h in completion['signature']['code'].items())
            assert completion['signature']['data'] == sha256(PRIVATE/('cohort.parquet' if cohort == 'matched' else 'eligible.parquet'))
            assert completion['signature']['plan'] == sha256(OUT/'LEGACY_CORRECTION_PLAN.md')
            for choice in json.loads((dest/'choices.json').read_text()):
                tr, te = choice['train_indices'], choice['test_indices']
                assert not set(source.iloc[tr].pair_id) & set(source.iloc[te].pair_id)
                assert len(tr)+len(te) == len(source)
            for policy, g in p.groupby('policy'):
                assert len(g) == len(source) and g.participant_id.is_unique
                assert np.array_equal(g.y, truth.loc[g.participant_id,'y'])
                assert np.array_equal(g.prediction, (g.probability >= g.threshold).astype(int))
                assert g.probability.between(0,1).all()
                summary.append(dict(cohort=cohort, configuration=config, policy=policy, **metrics(g.y,g.probability,g.prediction)))
            base = p[p.policy=='default'].sort_values('participant_id').reset_index(drop=True)
            if reference_folds is None: reference_folds = base[['participant_id','fold']]
            else: assert base[['participant_id','fold']].equals(reference_folds)
            prediction_sets[cohort,config] = base; counts += len(p)
    # Numerical replay of selected predictions against earlier corrected results.
    for config in ['full','sensors']:
        old = pd.read_parquet(PRIVATE/f'runs/{config}_seed42/predictions.parquet')
        old = old[(old.method=='selected') & (old.policy=='default')].sort_values('participant_id')
        new = prediction_sets['matched', config]
        assert list(old.participant_id) == list(new.participant_id)
        np.testing.assert_allclose(old.probability, new.probability, atol=1e-12, rtol=0)
    old = pd.read_parquet(PRIVATE/'unmatched_eligible_predictions.parquet').sort_values('participant_id')
    new = prediction_sets['eligible','full']
    assert list(old.participant_id) == list(new.participant_id)
    np.testing.assert_allclose(old.probability, new.probability, atol=1e-12, rtol=0)
    m = pd.DataFrame(summary); m.to_csv(PUBLIC/'age_ablation_metrics.csv',index=False)
    paired_associations(d).to_csv(PUBLIC/'paired_associations.csv', index=False)
    effects = []
    full = prediction_sets['matched','full']; pairids = full.pair_id.unique()
    positions = [np.flatnonzero(full.pair_id.to_numpy()==x) for x in pairids]
    for config in ['no_age','sensors']:
        alternative = prediction_sets['matched',config]; rng=np.random.default_rng(18042)
        delta = roc_auc_score(full.y,alternative.probability)-roc_auc_score(full.y,full.probability)
        bs=[]
        for _ in range(2000):
            ix=np.concatenate([positions[j] for j in rng.integers(len(positions),size=len(positions))])
            bs.append(roc_auc_score(full.y.iloc[ix],alternative.probability.iloc[ix])-roc_auc_score(full.y.iloc[ix],full.probability.iloc[ix]))
        lo,hi=np.percentile(bs,[2.5,97.5]);effects.append(dict(comparison=config+'_minus_full',auc_difference=delta,ci_low=lo,ci_high=hi))
    pd.DataFrame(effects).to_csv(PUBLIC/'matched_age_ablation_effects.csv', index=False)
    imp=pd.read_csv(LOCAL/'matched_full/importance.csv');imp.to_csv(PUBLIC/'heldout_block_importance.csv',index=False)
    # Descriptive Monte Carlo variability; these are not confidence intervals.
    agg=imp.groupby(['block','repeat']).apply(lambda g: np.average(g.auc_drop,weights=g.n),include_groups=False).rename('auc_drop').reset_index()
    agg.groupby('block').auc_drop.agg(['mean','std','min','max']).to_csv(PUBLIC/'heldout_block_importance_summary.csv')
    write_json(PUBLIC/'verification.json',manifest(dict(status='passed',completed_jobs=8,
        prediction_rows=counts,metric_groups=len(m),existing_prediction_replays=3,
        note='Exploratory; associations and feature importance are not causal effects.')))
    print(m[m.policy=='default'][['cohort','configuration','n','auc']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report-only',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(exist_ok=True)
    if not a.report_only:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for f in as_completed([pool.submit(job,(c,k)) for c in ['matched','eligible'] for k in ['full','no_age','sensors','age_only']]):print('Complete:',f.result(),flush=True)
    report()
