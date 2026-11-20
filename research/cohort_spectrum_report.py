"""Independent artifact verification and aggregate cohort-spectrum reporting."""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from research.common import ROOT, PRIVATE, sha256, write_json, manifest
from research.cohort_spectrum_data import LOCAL, PUBLIC
from research.cohort_spectrum_run import COHORTS, CONFIGS, signature
from research.core import metrics, candidate_models, select_thresholds
from research.legacy_supplement import columns


def bootstrap_aucs(p, seed=19042, repetitions=2000):
    """Pair/singleton cluster bootstrap conditional on fixed held-out predictions.

    Weighted Mann–Whitney calculation preserves ties and avoids sorting every draw.
    Verified below against sklearn's weighted ROC-AUC on a resample.
    """
    y = p.y.to_numpy(int); score = p.probability.to_numpy(float)
    _, unit = np.unique(p.pair_id.to_numpy(), return_inverse=True)
    groups = unit.max()+1
    _, ties = np.unique(score, return_inverse=True)
    def auc(weights):
        pos = np.bincount(ties, weights=weights*y)
        neg = np.bincount(ties, weights=weights*(1-y))
        return np.sum(pos*(np.cumsum(neg)-neg/2))/(pos.sum()*neg.sum())
    np.testing.assert_allclose(auc(np.ones(len(y))), roc_auc_score(y, score), atol=1e-14)
    rng = np.random.default_rng(seed); boot = []
    for i in range(repetitions):
        counts = np.bincount(rng.integers(groups, size=groups), minlength=groups)
        weights = counts[unit]
        value = auc(weights)
        if i == 0:
            np.testing.assert_allclose(value, roc_auc_score(y, score, sample_weight=weights), atol=1e-14)
        boot.append(value)
    return np.asarray(boot)


def verify():
    warnings.filterwarnings('ignore', message='The `probability` parameter was deprecated', category=FutureWarning)
    data = json.loads((PUBLIC/'data_verification.json').read_text())
    assert data['protocol_sha256'] == sha256(PUBLIC/'PROTOCOL.md')
    sources = json.loads((LOCAL/'input_hashes.json').read_text())
    assert all(sha256(p) == h for p, h in sources.items())
    raw = pd.read_csv(LOCAL/'raw_file_manifest.csv')
    assert all(sha256(ROOT/r.path) == r.sha256 for r in raw.itertuples())
    predictions = {}; models_checked = 0; inner_folds_checked = 0; matched_replays = 0
    for cohort in COHORTS:
        assert data['cohort_hashes'][cohort] == sha256(LOCAL/f'{cohort}.parquet')
        d = pd.read_parquet(LOCAL/f'{cohort}.parquet'); truth = d.set_index('participant_id')
        assert d.participant_id.is_unique
        for config in CONFIGS:
            dest = LOCAL/'runs'/f'{cohort}_{config}'
            done = json.loads((dest/'complete.json').read_text())
            cols = columns(d, config)
            assert done['signature'] == signature(cohort, cols)
            assert done['prediction_sha256'] == sha256(dest/'predictions.parquet')
            assert done['choices_sha256'] == sha256(dest/'choices.json')
            p = pd.read_parquet(dest/'predictions.parquet')
            X = d[cols].to_numpy(float); y = d.y.to_numpy(int); groups = d.pair_id.to_numpy()
            for choice in json.loads((dest/'choices.json').read_text()):
                fold = choice['fold']; tr = np.asarray(choice['train_indices']); te = np.asarray(choice['test_indices'])
                assert len(tr)+len(te) == len(d) and not set(tr) & set(te)
                assert set(tr) | set(te) == set(range(len(d)))
                assert not set(groups[tr]) & set(groups[te])
                assert (d.iloc[te].fold == fold).all() and (d.iloc[tr].fold != fold).all()
                mp = dest/f'fold{fold}.joblib'; ip = dest/f'fold{fold}_inner.npz'
                assert sha256(mp) == choice['model_sha256'] and sha256(ip) == choice['inner_sha256']
                fit = joblib.load(mp); z = np.load(ip)
                np.testing.assert_array_equal(z['train_indices'], tr)
                seen = np.zeros(len(tr), int)
                for i in range(3):
                    a, b = z[f'inner_train_{i}'], z[f'inner_valid_{i}']
                    assert not set(groups[tr[a]]) & set(groups[tr[b]])
                    assert not set(a) & set(b) and set(a) | set(b) == set(range(len(tr)))
                    seen[b] += 1; inner_folds_checked += 1
                assert (seen == 1).all()
                P = z['probabilities']; assert P.shape == (len(tr), 7) and np.isfinite(P).all()
                models = candidate_models(42)
                aucs = [roc_auc_score(y[tr], v) for v in P.T]
                for name, value in zip(models, aucs):
                    np.testing.assert_allclose(value, choice['candidate_auc'][name], atol=1e-14)
                winner = int(np.argmax(aucs)); assert list(models)[winner] == choice['selected']
                assert select_thresholds(y[tr], P[:, winner]) == choice['thresholds']
                imputer, scaler = fit.steps[0][1], fit.steps[1][1]
                medians = np.nanmedian(X[tr], axis=0); medians = np.nan_to_num(medians, nan=0)
                np.testing.assert_allclose(imputer.statistics_, medians, atol=1e-14)
                imputed = np.where(np.isnan(X[tr]), medians, X[tr])
                np.testing.assert_allclose(scaler.center_, np.median(imputed, axis=0), atol=1e-14)
                spread = np.diff(np.percentile(imputed, [25, 75], axis=0), axis=0)[0]
                spread[spread < 10*np.finfo(spread.dtype).eps] = 1
                np.testing.assert_allclose(scaler.scale_, spread, atol=1e-14)
                target = p[(p.fold == fold) & (p.policy == 'default')].set_index('participant_id').loc[d.iloc[te].participant_id]
                np.testing.assert_allclose(fit.predict_proba(X[te])[:, 1], target.probability, atol=1e-12, rtol=0)
                for policy, tau in choice['thresholds'].items():
                    assert (p.loc[(p.fold == fold) & (p.policy == policy), 'threshold'] == tau).all()
                models_checked += 1
            for policy, g in p.groupby('policy'):
                assert len(g) == len(d) and g.participant_id.is_unique
                np.testing.assert_array_equal(g.y, truth.loc[g.participant_id, 'y'])
                np.testing.assert_array_equal(g.fold, truth.loc[g.participant_id, 'fold'])
                np.testing.assert_array_equal(g.pair_id, truth.loc[g.participant_id, 'pair_id'])
                np.testing.assert_array_equal(g.prediction, (g.probability >= g.threshold).astype(int))
                assert g.probability.between(0, 1).all()
            if cohort == 'matched':
                old = pd.read_parquet(PRIVATE/'legacy_review'/f'matched_{config}'/'predictions.parquet')
                keys = ['policy', 'participant_id']
                a = p.sort_values(keys).reset_index(drop=True); b = old.sort_values(keys).reset_index(drop=True)
                pd.testing.assert_frame_equal(a, b, check_exact=True)
                matched_replays += 1
            predictions[cohort, config] = p
        print('Verified:', cohort, flush=True)
    matched_ids = set(pd.read_parquet(LOCAL/'matched.parquet').participant_id)
    common_folds = None
    for p in predictions.values():
        common = p[(p.policy == 'default') & p.participant_id.isin(matched_ids)].sort_values('participant_id')
        got = common[['participant_id', 'fold', 'pair_id', 'y']].reset_index(drop=True)
        if common_folds is None: common_folds = got
        else: pd.testing.assert_frame_equal(common_folds, got, check_exact=True)
    return predictions, dict(status='passed', selected_models_replayed=models_checked,
        inner_fold_boundaries_verified=inner_folds_checked, previous_matched_jobs_exactly_reproduced=matched_replays,
        raw_file_hashes_verified=len(raw), original_and_existing_sources_unchanged=True,
        common_participant_fold_alignment_verified=True)


def report():
    predictions, verification = verify()
    matched_ids = set(pd.read_parquet(LOCAL/'matched.parquet').participant_id)
    rows, intervals, bootstraps, common_frames = [], [], {}, {}
    for (cohort, config), p in predictions.items():
        for evaluation in ['own_cohort', 'common_matched']:
            frame = p if evaluation == 'own_cohort' else p[p.participant_id.isin(matched_ids)]
            for policy, g in frame.groupby('policy'):
                rows.append(dict(training_cohort=cohort, evaluation=evaluation, configuration=config,
                                 policy=policy, **metrics(g.y, g.probability, g.prediction)))
            g = frame[frame.policy == 'default'].sort_values('participant_id').reset_index(drop=True)
            boot = bootstrap_aucs(g); lo, hi = np.percentile(boot, [2.5, 97.5])
            intervals.append(dict(training_cohort=cohort, evaluation=evaluation, configuration=config,
                                  n=len(g), auc=roc_auc_score(g.y, g.probability), ci_low=lo, ci_high=hi))
            bootstraps[cohort, config, evaluation] = boot
            if evaluation == 'common_matched': common_frames[cohort, config] = g
    results = pd.DataFrame(rows); results.to_csv(PUBLIC/'metrics_all.csv', index=False)
    cis = pd.DataFrame(intervals); cis.to_csv(PUBLIC/'auc_intervals.csv', index=False)
    effects = []
    for config in CONFIGS:
        for cohort in ['broad', 'restricted']:
            a = common_frames[cohort, config]; b = common_frames['matched', config]
            assert list(a.participant_id) == list(b.participant_id) and list(a.pair_id) == list(b.pair_id)
            delta = roc_auc_score(a.y, a.probability)-roc_auc_score(b.y, b.probability)
            boot = bootstraps[cohort, config, 'common_matched']-bootstraps['matched', config, 'common_matched']
            lo, hi = np.percentile(boot, [2.5, 97.5])
            effects.append(dict(configuration=config, contrast=cohort+'_training_minus_matched_training',
                                evaluation='common_matched_702', auc_difference=delta, ci_low=lo, ci_high=hi))
    pd.DataFrame(effects).to_csv(PUBLIC/'common_test_differences.csv', index=False)
    desc = pd.read_csv(PUBLIC/'cohort_description.csv'); balance = []
    for cohort, d in desc.groupby('cohort'):
        d = d.set_index('label'); gap = d.loc[1, 'age_mean']-d.loc[0, 'age_mean']
        sd = np.sqrt((d.loc[1, 'age_sd']**2+d.loc[0, 'age_sd']**2)/2)
        balance.append(dict(cohort=cohort, age_gap=gap, age_smd=gap/sd,
                            male_fraction_gap=d.loc[1, 'male_fraction']-d.loc[0, 'male_fraction']))
    pd.DataFrame(balance).to_csv(PUBLIC/'balance.csv', index=False)
    figures(cis)
    write_json(PUBLIC/'verification.json', manifest(dict(**verification,
        metric_rows=len(results), auc_interval_rows=len(cis), bootstrap_replicates=2000,
        uncertainty='conditional on fixed OOF predictions; no refitting; exploratory, no multiplicity adjustment',
        result_hashes={n: sha256(PUBLIC/n) for n in ['metrics_all.csv', 'auc_intervals.csv', 'common_test_differences.csv']})))
    print(cis.to_string(index=False), flush=True)


def figures(cis):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    labels = {'broad': 'Broad adult\nN=2,431', 'restricted': 'Age 35–75\nN=1,243', 'matched': 'Age/sex matched\nN=702'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    colors = ['#777777', '#1565c0', '#ee8c20', '#16836c']
    names = ['Age only', 'Full multimodal', 'Full without age', 'Sensors only']
    for ax, evaluation, title in zip(axes, ['own_cohort', 'common_matched'],
        ['A. Each cohort’s held-out participants', 'B. Same 702 held-out matched participants']):
        for k, color, name, offset in zip(CONFIGS, colors, names, [-.21, -.07, .07, .21]):
            d = cis[(cis.evaluation == evaluation) & (cis.configuration == k)].set_index('training_cohort').loc[COHORTS]
            ax.errorbar(np.arange(3)+offset, d.auc, yerr=[d.auc-d.ci_low, d.ci_high-d.auc],
                        fmt='o', color=color, capsize=3, ms=5, label=name)
        ax.set_xticks(range(3), [labels[c] for c in COHORTS]); ax.set_xlabel('Training cohort')
        ax.set_title(title, fontsize=10); ax.axhline(.5, linestyle=':', color='gray'); ax.grid(axis='y', alpha=.2)
        ax.set_ylim(.45, 1.0)
    axes[0].set_ylabel('Held-out ROC-AUC'); axes[1].legend(loc='upper right', fontsize=8)
    fig.text(.5, .025, '95% cluster-bootstrap intervals conditional on predictions. Nested cohorts; exploratory comparison.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .06, 1, 1])
    for ext in ['png', 'pdf']: fig.savefig(PUBLIC/f'cohort_auc_comparison.{ext}', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharex=True, sharey=True)
    for ax, cohort in zip(axes, COHORTS):
        d = pd.read_parquet(LOCAL/f'{cohort}.parquet')
        for label, color, name in [(0, '#1565c0', 'Control'), (1, '#cf493c', 'PD')]:
            ax.hist(d.loc[d.y == label, 'demo_age'], bins=np.arange(15, 96, 5), density=True,
                    histtype='step', linewidth=1.8, color=color, label=name)
        ax.set_title(labels[cohort].replace('\n', ' — '), fontsize=10); ax.set_xlabel('Age (years)')
    axes[0].set_ylabel('Within-group density'); axes[-1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(PUBLIC/f'age_distributions.{ext}', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    report()
