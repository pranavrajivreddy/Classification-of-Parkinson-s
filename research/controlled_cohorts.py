"""Fixed cases, fold/sex quotas, equal-N control sampling; all model outputs private."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from research.common import ROOT, PRIVATE, OUT, sha256, write_json
from research.core import candidate_models, pipeline, splits, oof, select_thresholds
from research.legacy_supplement import columns

PUBLIC = OUT/'controlled_followup'
LOCAL = PRIVATE/'controlled_followup'
SOURCE = PRIVATE/'cohort_comparison'
CONFIGS = ['full', 'no_age', 'sensors', 'age_only']
SEEDS = list(range(2026111900, 2026111920))


def sources():
    paths = [SOURCE/f'{c}.parquet' for c in ['broad', 'restricted', 'matched']]
    paths += [PRIVATE/'runs/full_seed42'/n for n in ['predictions.parquet', 'selections.json', 'complete.json']]
    paths += [PRIVATE/'runs/full_seed42'/f'fold{f}_meta.npz' for f in range(5)]
    paths += [PRIVATE/'cohort.parquet', PRIVATE/'eligible.parquet', PUBLIC/'PROTOCOL.md']
    return {str(p): sha256(p) for p in paths}


def prepare():
    LOCAL.mkdir(parents=True, exist_ok=True)
    dest = LOCAL/'inputs.json'
    if dest.exists(): assert json.loads(dest.read_text()) == sources()
    else: write_json(dest, sources())
    matched = pd.read_parquet(SOURCE/'matched.parquet')
    cases = matched[matched.y == 1]
    quota = cases.groupby(['fold', 'demo_gender_male']).size()
    desc = []
    for arm in ['broad', 'restricted']:
        pool = pd.read_parquet(SOURCE/f'{arm}.parquet'); ctrls = pool[pool.y == 0]
        for seed in SEEDS:
            rng = np.random.default_rng(seed); controls = []
            for (fold, sex), n in quota.items():
                available = ctrls[(ctrls.fold == fold) & (ctrls.demo_gender_male == sex)]
                assert len(available) >= n
                controls.append(available.iloc[rng.choice(len(available), size=n, replace=False)])
            d = pd.concat([cases, *controls]).sort_values('participant_id').reset_index(drop=True)
            assert len(d) == 702 and d.participant_id.is_unique
            assert set(d.loc[d.y == 1, 'participant_id']) == set(cases.participant_id)
            assert d.groupby('pair_id').fold.nunique().max() == 1
            pd.testing.assert_series_equal(d[d.y == 0].groupby(['fold', 'demo_gender_male']).size(), quota)
            p = LOCAL/'cohorts'/f'{arm}_{seed}.parquet'; p.parent.mkdir(exist_ok=True)
            if p.exists(): pd.testing.assert_frame_equal(pd.read_parquet(p), d, check_exact=True)
            else: d.to_parquet(p, index=False)
            for y, g in d.groupby('y'):
                desc.append(dict(arm=arm, seed=seed, y=int(y), n=len(g), age_mean=g.demo_age.mean(),
                                 age_sd=g.demo_age.std(), male_n=int(g.demo_gender_male.sum())))
    pd.DataFrame(desc).to_csv(PUBLIC/'sampling_description.csv', index=False)
    write_json(PUBLIC/'sampling_gate.json', dict(status='passed', cohorts=40, n_each=702,
        same_case_identities=True, exact_fold_sex_class_quotas=True, protocol_sha256=sha256(PUBLIC/'PROTOCOL.md')))


def job(spec):
    warnings.filterwarnings('ignore', message='The `probability` parameter was deprecated', category=FutureWarning)
    arm, seed, config = spec
    source = LOCAL/'cohorts'/f'{arm}_{seed}.parquet'
    d = pd.read_parquet(source); matched = pd.read_parquet(SOURCE/'matched.parquet')
    dest = LOCAL/'cohort_runs'/f'{arm}_{seed}_{config}'; dest.mkdir(parents=True, exist_ok=True)
    cols = columns(d, config)
    sig = dict(cohort=sha256(source), test=sha256(SOURCE/'matched.parquet'),
               protocol=sha256(PUBLIC/'PROTOCOL.md'), code=sha256(__file__),
               core=sha256(ROOT/'research/core.py'), columns=cols)
    done = dest/'complete.json'
    if done.exists():
        old = json.loads(done.read_text()); assert old['signature'] == sig
        for n, h in old['files'].items(): assert sha256(dest/n) == h
        return dest.name
    X = d[cols].to_numpy(float); y = d.y.to_numpy(int); groups = d.pair_id.to_numpy()
    rows, choices = [], []
    for fold in range(5):
        tr = np.flatnonzero(d.fold.to_numpy() != fold)
        own = d[d.fold == fold]
        common = matched[matched.fold == fold]
        test = pd.concat([own, common]).drop_duplicates('participant_id').sort_values('participant_id')
        assert not set(d.iloc[tr].participant_id) & set(test.participant_id)
        assert not set(d.iloc[tr].pair_id) & set(test.pair_id)
        inner = splits(y[tr], groups[tr], n=3, seed=143)
        models = candidate_models(42)
        P = np.column_stack([oof(X[tr], y[tr], inner, m) for m in models.values()])
        scores = [roc_auc_score(y[tr], p) for p in P.T]
        winner = int(np.argmax(scores)); name = list(models)[winner]
        fit = pipeline(models[name]).fit(X[tr], y[tr])
        q = fit.predict_proba(test[cols].to_numpy(float))[:, 1]
        tau = select_thresholds(y[tr], P[:, winner])
        mp = dest/f'fold{fold}.joblib'; joblib.dump(fit, mp)
        np.testing.assert_allclose(joblib.load(mp).predict_proba(test[cols].to_numpy(float))[:, 1], q, atol=1e-12, rtol=0)
        ip = dest/f'fold{fold}_inner.npz'
        np.savez_compressed(ip, probabilities=P, train_indices=tr,
            **{f'tr{i}': a for i, (a, b) in enumerate(inner)},
            **{f'va{i}': b for i, (a, b) in enumerate(inner)})
        choices.append(dict(fold=fold, train_indices=tr.tolist(), selected=name,
            aucs=scores, thresholds=tau, model_sha256=sha256(mp), inner_sha256=sha256(ip)))
        test = test.assign(probability=q)
        for evaluation, ids in [('own', set(own.participant_id)), ('common', set(common.participant_id))]:
            for policy, threshold in tau.items():
                g = test[test.participant_id.isin(ids)]
                for r in g.itertuples():
                    rows.append(dict(evaluation=evaluation, policy=policy, participant_id=r.participant_id,
                        pair_id=r.pair_id, y=int(r.y), fold=fold, probability=float(r.probability),
                        threshold=threshold, prediction=int(r.probability >= threshold)))
    pd.DataFrame(rows).to_parquet(dest/'predictions.parquet', index=False)
    write_json(dest/'choices.json', choices)
    write_json(done, dict(signature=sig, files={p.name:sha256(p) for p in dest.iterdir() if p.name != 'complete.json'}))
    return dest.name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    prepare()
    print('Sampling gate passed: 40 cohorts, identical case/fold/sex/class counts.', flush=True)
    if not args.prepare_only:
        jobs = [(a, s, c) for s in SEEDS for a in ['broad', 'restricted'] for c in CONFIGS]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, f in enumerate(as_completed([pool.submit(job, j) for j in jobs]), 1):
                print(f'Completed {i}/{len(jobs)}: {f.result()}', flush=True)
