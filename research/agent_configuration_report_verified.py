"""Conditional paired uncertainty and independent reconciliation of the bounded run."""
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from research.agent_configuration_search_verified import (
    P, L, load, signature, defaults, score, gate, choose, columns,
    check_fit, measures, operating_threshold, SEEDS, dump, canonical)
from research.common import ROOT, sha256
from research.core import splits


def fast(y, p, tau):
    pred = p >= tau; pos = y == 1; neg = ~pos
    tp, tn = np.sum(pred & pos), np.sum(~pred & neg)
    fp, fn = np.sum(pred & neg), np.sum(~pred & pos)
    n1, n0 = pos.sum(), neg.sum()
    auc = (rankdata(p)[pos].sum()-n1*(n1+1)/2)/(n1*n0)
    return np.array([auc, tp/n1, tn/n0, np.mean((p-y)**2), (tp+tn)/len(y),
                     2*tp/max(2*tp+fp+fn, 1), 5*tp/max(5*tp+4*fn+fp, 1),
                     .5*(tp/n1+tn/n0), 1-(4*fn+2*fp)/len(y)])


NAMES = ['auc', 'sensitivity', 'specificity', 'brier', 'accuracy', 'f1', 'f2',
         'balanced_accuracy', 'utility']


def verify():
    frozen = json.loads((P/'frozen.json').read_text())
    assert frozen['signature'] == signature()
    assert all(sha256(P/f) == h for f, h in frozen['provenance'].items())
    assert json.loads((P/'evaluation_verification.json').read_text())['status'] == 'passed'
    d, e = load('discovery'), load('evaluation')
    baseline = json.loads((P/'baseline_scores.json').read_text())
    best_name = sorted(baseline, key=lambda k: (-baseline[k]['mean_auc'], k))[0]
    base = baseline[best_name]
    arms = json.loads((P/'all_candidates.json').read_text())
    for arm, candidates in arms.items():
        assert len(candidates) == 12
        assert choose(candidates, base) == frozen['decisions'][arm]
        for c in candidates:
            if 'summary' in c: assert c['gate'] == gate(c['summary'], base)
    assert all(len({canonical(c['summary']['configuration']) for c in records if 'summary' in c}) <= 12
               for records in arms.values())
    replays = 0
    for marker in (L/'discovery').glob('*/complete.json'):
        info = json.loads(marker.read_text()); c = info['configuration']; path = marker.parent
        assert info['signature'] == signature()
        assert all(sha256(path/f) == h for f, h in info['outputs'].items())
        saved = pd.read_csv(path/'fold_metrics.csv'); yy = []; pp = []; rows = []
        x = d[columns(c, d)].to_numpy(float); y = d.y.to_numpy(); g = d.pair_id.to_numpy()
        for repeat in SEEDS:
            for fold, (tr, va) in enumerate(splits(y, g, n=3, seed=repeat)):
                a = np.load(path/f'{repeat}_{fold}.npz')
                np.testing.assert_array_equal(a['tr'], tr); np.testing.assert_array_equal(a['va'], va)
                fit = joblib.load(path/f'{repeat}_{fold}.joblib'); check_fit(fit, c, x[tr], y[tr])
                prob = fit.predict_proba(x[va])[:, 1]
                np.testing.assert_allclose(prob, a['p'], atol=1e-12, rtol=0)
                row = measures(y[va], prob, c['threshold'])
                row['train_auc'] = roc_auc_score(y[tr], fit.predict_proba(x[tr])[:, 1])
                target = saved[(saved.repeat == repeat) & (saved.fold == fold)].iloc[0]
                for key, value in row.items():
                    if value is None: assert pd.isna(target[key])
                    else: np.testing.assert_allclose(value, target[key], atol=1e-12, rtol=0)
                yy.extend(y[va]); pp.extend(prob); rows.append(dict(repeat=repeat, **row)); replays += 1
        frame = pd.DataFrame(rows); summary = info['summary']
        np.testing.assert_allclose(summary['mean_auc'], frame.auc.mean(), atol=1e-12, rtol=0)
        np.testing.assert_allclose(summary['mean_brier'], frame.brier.mean(), atol=1e-12, rtol=0)
        for repeat in SEEDS:
            np.testing.assert_allclose(summary['repeat_auc'][str(repeat)], frame[frame.repeat == repeat].auc.mean(), atol=1e-12, rtol=0)
        assert summary['exact_sensitivity85_threshold'] == operating_threshold(yy, pp)
        assert summary['recommended_threshold_metrics'] == measures(yy, pp, c['threshold'])
        assert summary['exact_sensitivity85_metrics'] == measures(yy, pp, summary['exact_sensitivity85_threshold'])
    metrics = pd.read_csv(P/'evaluation_metrics.csv'); probs = {}; thresholds = {}
    for arm, dec in frozen['decisions'].items():
        c = dec['selected']['configuration']; path = L/'evaluation'/arm
        done = json.loads((path/'complete.json').read_text())
        assert done['frozen_sha256'] == sha256(P/'frozen.json')
        assert all(sha256(path/f) == h for f, h in done['outputs'].items())
        assert (path/'model.joblib').stat().st_mtime >= frozen['created_unix']
        fit = joblib.load(path/'model.joblib'); cols = columns(c, d)
        check_fit(fit, c, d[cols].to_numpy(float), d.y.to_numpy())
        p = fit.predict_proba(e[cols].to_numpy(float))[:, 1]
        np.testing.assert_allclose(p, pd.read_parquet(path/'predictions.parquet').probability, atol=1e-12, rtol=0)
        probs[arm] = p
        thresholds[arm] = dict(default=.5, recommended=c['threshold'],
                               sensitivity85=dec['selected']['exact_sensitivity85_threshold'])
        for policy, tau in thresholds[arm].items():
            row = metrics[(metrics.arm == arm) & (metrics.policy == policy)].iloc[0]
            np.testing.assert_allclose(row.threshold, tau, atol=1e-12, rtol=0)
            exact = measures(e.y, p, tau)
            for key, value in exact.items():
                if value is None: assert pd.isna(row[key])
                else: np.testing.assert_allclose(value, row[key], atol=1e-12, rtol=0)
            np.testing.assert_allclose(fast(e.y.to_numpy(), p, tau), [exact[k] for k in NAMES], atol=1e-12, rtol=0)
    for request in (P/'api').glob('*_request.json'):
        txt = request.read_text()
        assert not any(s in txt for s in pd.concat([d, e]).participant_id.astype(str))
        assert 'sk-or-' not in txt
    ledger = json.loads((P/'api_ledger.json').read_text())
    assert len(ledger) == 3 and all(r['status'] == 'completed' for r in ledger)
    cost = sum(r['usage']['cost'] for r in ledger)
    assert cost <= sum(r['reserved_usd'] for r in ledger) <= 2
    return e, frozen, arms, probs, thresholds, replays, cost


def report():
    e, frozen, candidates, probs, thresholds, replays, cost = verify()
    y = e.y.to_numpy(); groups = e.pair_id.to_numpy(); unique = np.unique(groups)
    pairs = np.array([np.flatnonzero(groups == g) for g in unique]); assert pairs.shape == (105, 2)
    rng = np.random.default_rng(20261120); boot = {}
    for arm in probs:
        for policy in thresholds[arm]: boot[(arm, policy)] = np.empty((2000, len(NAMES)))
    for b in range(2000):
        ix = pairs[rng.integers(len(pairs), size=len(pairs))].ravel()
        for (arm, policy), values in boot.items():
            values[b] = fast(y[ix], probs[arm][ix], thresholds[arm][policy])
    intervals = []; comparisons = []
    for (arm, policy), values in boot.items():
        point = fast(y, probs[arm], thresholds[arm][policy]); limits = np.quantile(values, [.025, .975], axis=0)
        for j, name in enumerate(NAMES):
            intervals.append(dict(arm=arm, policy=policy, metric=name, estimate=point[j], lower=limits[0, j], upper=limits[1, j]))
    for comparator in ['baseline', 'random', 'optuna']:
        for policy in ['default', 'recommended', 'sensitivity85']:
            delta = boot[('agent', policy)]-boot[(comparator, policy)]
            limits = np.quantile(delta, [.025, .975], axis=0)
            point = fast(y, probs['agent'], thresholds['agent'][policy])-fast(y, probs[comparator], thresholds[comparator][policy])
            for j, name in enumerate(NAMES):
                comparisons.append(dict(comparison=f'agent_minus_{comparator}', policy=policy, metric=name,
                                        estimate=point[j], lower=limits[0, j], upper=limits[1, j]))
    ci = pd.DataFrame(intervals); diff = pd.DataFrame(comparisons)
    ci.to_csv(P/'evaluation_intervals.csv', index=False); diff.to_csv(P/'paired_differences.csv', index=False)
    metrics = pd.read_csv(P/'evaluation_metrics.csv')
    baseline_row = metrics[(metrics.arm == 'baseline') & (metrics.policy == 'sensitivity85')].iloc[0]
    lead = (f"**Main finding:** the agent and random-search procedures retained the shared default model. "
            f"That model achieved evaluation AUC {baseline_row.auc:.4f}, sensitivity {baseline_row.sensitivity:.2%} "
            f"and specificity {baseline_row.specificity:.2%} at the discovery-selected operating threshold. "
            "These are reference-model results, not gains caused by agent tuning. The held-out sensitivity "
            "did not reach the 85% discovery target. No clear agent advantage was established by this run.")
    text = ['# Agent-directed configuration search: results', '',
            lead, '',
            '**Technical history:** an initial three-call attempt allowed family-incompatible parameter dictionaries through its API schema. Local validation blocked seven of twelve proposals. That attempt and all its discovery results are preserved, and no evaluation performance was inspected. One documented technical replication used closed family-specific schemas, with unchanged scientific rules. The results below belong to that replication; it must not be described as the first and only attempt.', '',
            'This exploratory experiment tested whether an agent could choose input combinations, variable-selection rules, model families, hyperparameters and a recommended threshold. It did not generate new features or change the cohort.', '',
            '**Design:** 492 participants for discovery and final fitting; 210 separate participants for evaluation. Both sets came from the existing matched 702-person cohort and had been investigated previously. Results below are direct hold-out results, not the earlier within-subset cross-validation results.', '',
            '**Search:** Claude Sonnet 4.6, three feedback rounds with four candidate slots each. Random search and Optuna received the same six shared baseline scores, twelve additional slots, search space and six validation fits per valid configuration. Same trial counts do not mean identical compute time or identical proposal-information processing.', '',
            '**Selection:** mean of six discovery fold AUCs. To replace the best default family, a candidate needed mean gain >=0.002 and positive gain in both CV repetitions. Otherwise the default remained. This configuration-level rule differs from the earlier cross-family feature rule and is not a significance test.', '',
            '## Discovery and chosen procedures', '',
            '| Arm | Best searched discovery AUC | Selected discovery AUC | Replaced default | Selected family | Inputs | Selection size |',
            '|---|---:|---:|---|---|---|---|']
    for arm, dec in frozen['decisions'].items():
        s = dec['selected']; c = s['configuration']; best = dec.get('best_searched')
        best_text = f"{best['summary']['mean_auc']:.4f}" if best else 'Not searched'
        text.append(f"| {arm} | {best_text} | {s['mean_auc']:.4f} | {dec['replaced_reference']} | {c['family']} | {c['inputs']} | {'all' if c['keep']==0 else c['keep']} |")
    text += ['', 'Candidate accounting:', '']
    for arm, records in candidates.items():
        valid = [r for r in records if 'summary' in r]
        distinct = len({canonical(r['summary']['configuration']) for r in valid})
        text.append(f'- {arm}: {len(records)} slots, {len(valid)} valid, {distinct} distinct configurations. Duplicates reuse saved predictions and consume slots.')
    text += ['', 'Discovery scores are used for selection and are optimistic estimates of the selected procedure; evaluation below is the relevant comparison.', '',
             '## Evaluation at the default 0.5 threshold', '',
             '| Arm | ROC-AUC (conditional 95% interval) | Sensitivity | Specificity | AP | Brier | F2 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for _, row in metrics[metrics.policy == 'default'].iterrows():
        interval = ci[(ci.arm == row.arm) & (ci.policy == 'default') & (ci.metric == 'auc')].iloc[0]
        text.append(f"| {row.arm} | {row.auc:.4f} [{interval.lower:.4f}, {interval.upper:.4f}] | {row.sensitivity:.2%} | {row.specificity:.2%} | {row.average_precision:.4f} | {row.brier:.4f} | {row.f2:.4f} |")
    text += ['', '## Operating thresholds', '',
             'Recommended means the numeric threshold in the selected configuration. Sensitivity85 is calculated exactly from discovery out-of-fold predictions: maximize specificity while attaining at least 85% discovery sensitivity. Neither is selected using evaluation outcomes. AUC is identical across thresholds for a given fitted model.', '',
             '| Arm | Policy | Threshold | Evaluation sensitivity | Evaluation specificity | TP / FN / TN / FP |',
             '|---|---|---:|---:|---:|---|']
    for _, r in metrics[metrics.policy != 'default'].iterrows():
        text.append(f"| {r.arm} | {r.policy} | {r.threshold:.4f} | {r.sensitivity:.2%} | {r.specificity:.2%} | {r.tp} / {r.fn} / {r.tn} / {r.fp} |")
    text += ['', '## Paired AUC comparisons', '', '| Comparison | AUC difference | Conditional 95% interval |', '|---|---:|---|']
    for _, r in diff[(diff.policy == 'default') & (diff.metric == 'auc')].iterrows():
        text.append(f'| {r.comparison} | {r.estimate:+.4f} | [{r.lower:+.4f}, {r.upper:+.4f}] |')
    text += ['', 'Intervals resample the same 105 evaluation pairs 2,000 times across all arms. They hold fitted models and the selected agent trajectory fixed, are not adjusted for multiple comparisons, and do not capture variability from repeating discovery. An interval including zero does not establish superiority or equivalence.', '',
             'The exact zero differences for agent versus baseline/random follow from retaining identical configurations and predictions. They are an identity check, not evidence that different search methods are generally equivalent. AUC 0.8103 here must not replace the main 702-person cross-validation AUC as though it came from the same experiment.', '',
             '## Exact selected settings', '']
    for arm, dec in frozen['decisions'].items():
        text += [f'### {arm}', '', '```json', json.dumps(dec['selected']['configuration'], indent=2), '```', '']
    text += ['## Interpretation limits and audit', '',
             '- The agent made configuration proposals and responded to discovery feedback. Local code fitted models, calculated metrics and applied the same numerical gate to all search methods.',
             '- This is bounded agent-directed model tuning, not proof that an agent is better than a human expert or conventional optimization. There is no human-search arm, and only one trajectory per search method.',
             '- This does not test every useful parameter combination. Twelve trials per arm is a small search budget, and the input-selection space itself was specified by researchers.',
             '- A shared default already selected the best of six model families. It is stronger than a single arbitrary untuned model; discovery gains must be judged against it.',
             '- Agent explanations of why a setting might work are hypotheses. Predictive improvements do not establish causal mechanisms or clinical readiness.',
             '- The existing cohort has been repeatedly examined. This run is exploratory even though its feedback and evaluation boundaries are separated.',
             f'- Verified {replays} cached discovery-model predictions and their training-only imputers, scalers and feature selectors; replayed four final models and reconciled all twelve evaluation metric rows.',
             f'- Standalone release run: {len(json.loads((P/"api_ledger.json").read_text()))} API calls; provider-reported cost ${cost:.6f}. This ledger excludes earlier research attempts.',
             '- API payloads contain study descriptions, feature definitions and aggregate discovery metrics; no participant records, identifiers, raw sensor signals or evaluation results. Credentials were held in memory, not written to artifacts.', '',
             'Files: `PROTOCOL.md`, `baseline_scores.json`, `round*_feedback.json`, `selection.json`, `evaluation_metrics.csv`, `evaluation_intervals.csv`, `paired_differences.csv`, `final_verification.json`. Participant-level predictions and model files stay in the private directory. The manuscript has not been changed by this experiment.', '']
    (P/'RESULTS.md').write_text('\n'.join(text))
    counts = {arm: dict(slots=len(records), valid=sum('summary' in r for r in records),
                       unique=len({canonical(r['summary']['configuration']) for r in records if 'summary' in r}))
              for arm, records in candidates.items()}
    dump(P/'final_verification.json', dict(status='passed', discovery_fit_replays=replays,
         evaluation_fit_replays=4, evaluation_metric_rows=12, bootstrap_pairs=105, bootstrap_resamples=2000,
         trials=counts, api_cost_usd=cost,
         total_api_cost_usd=cost, frozen_sha256=sha256(P/'frozen.json'), report_sha256=sha256(P/'RESULTS.md'),
         report_code_sha256=sha256(__file__), finished_unix=time.time()))
    print(metrics[['arm', 'policy', 'auc', 'sensitivity', 'specificity', 'brier']].to_string(index=False))
    print(diff[(diff.policy == 'default') & (diff.metric == 'auc')].to_string(index=False))
    print(f'Independent replay and report complete; API cost ${cost:.6f}', flush=True)


if __name__ == '__main__': report()
