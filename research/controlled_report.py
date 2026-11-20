"""Replay selected models, audit fold boundaries, summarize fixed follow-up."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from research.common import ROOT, PRIVATE, sha256, write_json, manifest
from research.core import metrics, candidate_models, select_thresholds, feature_sets, splits
from research.controlled_cohorts import LOCAL, PUBLIC, SOURCE, CONFIGS, SEEDS, sources
from research.cohort_spectrum_report import bootstrap_aucs
from research.legacy_supplement import columns


def prep_check(fit, X):
    med = np.nan_to_num(np.nanmedian(X, axis=0), nan=0.)
    np.testing.assert_allclose(fit.steps[0][1].statistics_, med, atol=1e-14)
    Z = np.where(np.isnan(X), med, X)
    np.testing.assert_allclose(fit.steps[1][1].center_, np.median(Z, axis=0), atol=1e-14)
    scale = np.diff(np.percentile(Z, [25, 75], axis=0), axis=0)[0]
    scale[scale < 10*np.finfo(float).eps] = 1
    np.testing.assert_allclose(fit.steps[1][1].scale_, scale, atol=1e-14)


def check_preds(p, truth):
    assert p.participant_id.is_unique and len(p) == 702
    t = truth.set_index('participant_id').loc[p.participant_id]
    np.testing.assert_array_equal(p.y, t.y)
    np.testing.assert_array_equal(p.pair_id, t.pair_id)
    np.testing.assert_array_equal(p.fold, t.fold)
    np.testing.assert_array_equal(p.prediction, (p.probability >= p.threshold).astype(int))
    assert p.probability.between(0, 1).all()
    m = metrics(p.y, p.probability, p.prediction)
    tn, fp, fn, tp = confusion_matrix(p.y, p.prediction, labels=[0, 1]).ravel()
    assert (m['tn'], m['fp'], m['fn'], m['tp']) == (tn, fp, fn, tp)
    return m


def report_cohorts():
    assert sources() == json.loads((LOCAL/'inputs.json').read_text())
    original = json.loads((SOURCE/'input_hashes.json').read_text())
    assert all(sha256(p) == h for p, h in original.items())
    matched = pd.read_parquet(SOURCE/'matched.parquet')
    results, aucrows, boots, counts = [], [], {}, dict(models_replayed=0, inner_boundaries=0, cohorts=40)
    references = {}
    for config in CONFIGS:
        dest = SOURCE/'runs'/f'matched_{config}'
        done = json.loads((dest/'complete.json').read_text())
        assert done['prediction_sha256'] == sha256(dest/'predictions.parquet')
        p = pd.read_parquet(dest/'predictions.parquet')
        g = p[p.policy == 'default'].sort_values('participant_id').reset_index(drop=True)
        references[config] = g
        boots['matched', 0, config] = bootstrap_aucs(g)
        for policy, group in p.groupby('policy'):
            results.append(dict(arm='matched', seed=0, configuration=config, evaluation='common', policy=policy,
                                **check_preds(group, matched)))
    for seed in SEEDS:
        for arm in ['broad', 'restricted']:
            d = pd.read_parquet(LOCAL/'cohorts'/f'{arm}_{seed}.parquet')
            cases = d[d.y == 1]; mcase = matched[matched.y == 1]
            assert set(cases.participant_id) == set(mcase.participant_id)
            pd.testing.assert_series_equal(d.groupby(['fold','y','demo_gender_male']).size(),
                                           matched.groupby(['fold','y','demo_gender_male']).size())
            for config in CONFIGS:
                dest = LOCAL/'cohort_runs'/f'{arm}_{seed}_{config}'
                done = json.loads((dest/'complete.json').read_text())
                assert done['signature']['code'] == sha256(ROOT/'research/controlled_cohorts.py')
                assert done['signature']['protocol'] == sha256(PUBLIC/'PROTOCOL.md')
                assert done['signature']['cohort'] == sha256(LOCAL/'cohorts'/f'{arm}_{seed}.parquet')
                assert done['signature']['test'] == sha256(SOURCE/'matched.parquet')
                assert done['signature']['core'] == sha256(ROOT/'research/core.py')
                for n, h in done['files'].items(): assert sha256(dest/n) == h
                cols = columns(d, config); assert cols == done['signature']['columns']
                X = d[cols].to_numpy(float); y = d.y.to_numpy(int); groups = d.pair_id.to_numpy()
                p = pd.read_parquet(dest/'predictions.parquet')
                for choice in json.loads((dest/'choices.json').read_text()):
                    fold = choice['fold']; tr = np.asarray(choice['train_indices'])
                    np.testing.assert_array_equal(tr, np.flatnonzero(d.fold.to_numpy() != fold))
                    test = pd.concat([d[d.fold == fold], matched[matched.fold == fold]]).drop_duplicates('participant_id').sort_values('participant_id')
                    assert not set(d.iloc[tr].participant_id) & set(test.participant_id)
                    assert not set(groups[tr]) & set(test.pair_id)
                    z = np.load(dest/f'fold{fold}_inner.npz'); np.testing.assert_array_equal(z['train_indices'], tr)
                    seen = np.zeros(len(tr), int)
                    for i in range(3):
                        a, b = z[f'tr{i}'], z[f'va{i}']; seen[b] += 1
                        assert not set(groups[tr[a]]) & set(groups[tr[b]])
                        assert not set(a) & set(b) and set(a) | set(b) == set(range(len(tr)))
                        counts['inner_boundaries'] += 1
                    assert (seen == 1).all()
                    P = z['probabilities']; assert P.shape == (len(tr), 7) and np.isfinite(P).all()
                    aucs = [roc_auc_score(y[tr], v) for v in P.T]
                    np.testing.assert_allclose(aucs, choice['aucs'], atol=1e-14)
                    win = int(np.argmax(aucs)); assert list(candidate_models(42))[win] == choice['selected']
                    assert select_thresholds(y[tr], P[:, win]) == choice['thresholds']
                    fit = joblib.load(dest/f'fold{fold}.joblib'); prep_check(fit, X[tr])
                    q = fit.predict_proba(test[cols].to_numpy(float))[:, 1]
                    for policy, tau in choice['thresholds'].items():
                        g = p[(p.fold == fold) & (p.policy == policy)].drop_duplicates('participant_id').set_index('participant_id').loc[test.participant_id]
                        np.testing.assert_allclose(q, g.probability, atol=1e-12, rtol=0)
                        assert (g.threshold == tau).all()
                    counts['models_replayed'] += 1
                for (evaluation, policy), g in p.groupby(['evaluation', 'policy']):
                    results.append(dict(arm=arm, seed=seed, configuration=config, evaluation=evaluation,
                                        policy=policy, **check_preds(g, d if evaluation == 'own' else matched)))
                g = p[(p.evaluation == 'common') & (p.policy == 'default')].sort_values('participant_id').reset_index(drop=True)
                boots[arm, seed, config] = bootstrap_aucs(g)
        print('Cohort verification seed', seed, flush=True)
    r = pd.DataFrame(results); r.to_csv(PUBLIC/'cohort_metrics_all.csv', index=False)
    summaries = []
    for (arm, config, evaluation), g in r[r.policy == 'default'].groupby(['arm', 'configuration', 'evaluation']):
        summaries.append(dict(arm=arm, configuration=config, evaluation=evaluation, draws=len(g),
            auc_mean=g.auc.mean(), auc_sd=g.auc.std() if len(g)>1 else 0., auc_min=g.auc.min(), auc_max=g.auc.max()))
    pd.DataFrame(summaries).to_csv(PUBLIC/'cohort_auc_summary.csv', index=False)
    effects = []
    for arm in ['broad', 'restricted']:
        for config in CONFIGS:
            g = r[(r.arm == arm)&(r.configuration == config)&(r.evaluation == 'common')&(r.policy == 'default')]
            ref = references[config]; delta = roc_auc_score(ref.y, ref.probability)-g.auc.mean()
            draws = boots['matched', 0, config]-np.mean([boots[arm, s, config] for s in SEEDS], axis=0)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            effects.append(dict(contrast='matched_minus_mean_'+arm, configuration=config,
                                auc_difference=delta, ci_low=lo, ci_high=hi))
    pd.DataFrame(effects).to_csv(PUBLIC/'cohort_common_effects.csv', index=False)
    write_json(PUBLIC/'cohort_verification.json', manifest(dict(status='passed', **counts,
        source_hashes_unchanged=True, joint_pair_bootstrap=2000, metric_rows=len(r),
        uncertainty='conditional on fitted predictions and fixed draw set; draws are not independent studies')))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['cohorts']); parser.parse_args()
    report_cohorts()
