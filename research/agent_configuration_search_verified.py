"""Bounded configuration search. Provider output is data, never executed code."""
import argparse
import getpass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
import requests
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss)

from research.common import ROOT, PRIVATE, sha256, write_json
from research.core import feature_sets, splits, operating_table
from research.agent_auc_followup import load as original_load
from research.claude_single_core import models

P = ROOT / 'paper_ready/agent_configuration_search_verified'
L = PRIVATE / 'agent_configuration_search_verified'
SEEDS = [4242, 731]
MODEL = 'anthropic/claude-sonnet-4.6'
TAG = 'amazon-bedrock/global'
BASE = 'https://openrouter.ai/api/v1'
CAP = 2.0
INPUTS = ['demographics', 'tapping', 'walking', 'voice', 'tap_walk', 'tap_voice',
          'walk_voice', 'sensors', 'full']
THRESHOLDS = [round(.2 + .05*i, 2) for i in range(10)]
SPACE = {
    'lr': {'C': [.01, .1, 1, 10], 'class_weight': ['none', 'balanced']},
    'rf': {'n_estimators': [120, 240], 'max_depth': [3, 5, 8],
           'min_samples_leaf': [3, 8, 15], 'max_features': ['sqrt', .5, 1.]},
    'xgboost': {'n_estimators': [80, 160, 240], 'max_depth': [1, 2, 3],
                'learning_rate': [.02, .05, .1], 'min_child_weight': [3, 8],
                'reg_lambda': [1, 5, 10]},
    'lightgbm': {'n_estimators': [80, 160, 240], 'max_depth': [2, 3, 4],
                 'num_leaves': [3, 7], 'learning_rate': [.02, .05, .1],
                 'min_child_samples': [15, 30], 'reg_lambda': [1, 5, 10]},
    'catboost': {'iterations': [80, 160, 240], 'depth': [2, 3, 4],
                 'learning_rate': [.02, .05, .1], 'l2_leaf_reg': [3, 10]},
    'mlp': {'hidden_units': [8, 16, 32], 'alpha': [.1, 1, 10]},
}


def safe(value):
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    return value


def dump(path, value): write_json(path, safe(value))


def load(which):
    assert which in ['discovery', 'evaluation']
    return original_load('evaluation' if which == 'discovery' else 'discovery')


def signature():
    paths = [Path(__file__), P/'PROTOCOL.md', ROOT/'research/core.py',
             ROOT/'research/common.py', ROOT/'research/agent_auc_followup.py',
             ROOT/'research/claude_single_core.py', ROOT/'paper_ready/FEATURE_DICTIONARY.md',
             PRIVATE/'agent_extension/discovery.parquet',
             PRIVATE/'agent_extension/evaluation.parquet',
             PRIVATE/'agent_extension/split_manifest.json']
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def validate(c):
    assert isinstance(c, dict) and set(c) == {'family', 'inputs', 'keep', 'params', 'threshold'}
    assert c['family'] in SPACE and c['inputs'] in INPUTS
    assert type(c['keep']) is int and c['keep'] in [0, 10, 20]
    assert type(c['threshold']) in [int, float] and c['threshold'] in THRESHOLDS
    s = SPACE[c['family']]
    assert isinstance(c['params'], dict) and set(c['params']) == set(s)
    for k, v in c['params'].items():
        assert type(v) in [int, float, str] and v in s[k], f'Invalid {k}'
    return c


def defaults():
    out = {}
    for name, m in models().items():
        params = m.get_params()
        if name == 'mlp': params['hidden_units'] = params['hidden_layer_sizes'][0]
        if name == 'lr': params['class_weight'] = 'none'
        # CatBoost's unspecified learning rate is fixed explicitly in this run.
        if name == 'catboost': params['learning_rate'] = .05
        out[name] = validate(dict(family=name, inputs='full', keep=0,
                                 params={k: params[k] for k in SPACE[name]}, threshold=.5))
    return out


def columns(c, d): return feature_sets(d)[c['inputs']]


def estimator(c, n_columns):
    validate(c)
    model = models()[c['family']]
    ps = dict(c['params'])
    if c['family'] == 'lr' and ps['class_weight'] == 'none': ps['class_weight'] = None
    if c['family'] == 'mlp': ps['hidden_layer_sizes'] = (ps.pop('hidden_units'),)
    model.set_params(**ps)
    k = 'all' if c['keep'] == 0 else min(c['keep'], n_columns)
    return Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scaler', RobustScaler()), ('selector', SelectKBest(f_classif, k=k)),
                     ('model', model)])


def measures(y, p, tau):
    y = np.asarray(y, int); p = np.asarray(p, float); pred = p >= tau
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    div = lambda a, b: a/b if b else None
    se, sp = tp/(tp+fn), tn/(tn+fp)
    return dict(n=len(y), tp=tp, fp=fp, tn=tn, fn=fn, auc=roc_auc_score(y, p),
                average_precision=average_precision_score(y, p), brier=brier_score_loss(y, p),
                log_loss=log_loss(y, p, labels=[0, 1]), sensitivity=se, specificity=sp,
                accuracy=(tp+tn)/len(y), balanced_accuracy=(se+sp)/2,
                f1=div(2*tp, 2*tp+fp+fn), f2=div(5*tp, 5*tp+4*fn+fp),
                ppv=div(tp, tp+fp), npv=div(tn, tn+fn), utility=1-(4*fn+2*fp)/len(y))


def operating_threshold(y, p):
    table = operating_table(y, p)
    row = table[table.sensitivity >= .85].sort_values(
        ['specificity', 'sensitivity', 'threshold'], ascending=False).iloc[0]
    return float(row.threshold)


def canonical(c): return json.dumps(c, sort_keys=True)


def fit_and_log(c, x, y, path):
    est = estimator(c, x.shape[1])
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        est.fit(x, y)
    joblib.dump(est, path)
    return est, [dict(category=w.category.__name__, message=str(w.message)) for w in ws]


def check_fit(est, c, x, y):
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
    np.testing.assert_allclose(est['imputer'].statistics_, imp.statistics_, rtol=0, atol=0)
    z = imp.transform(x); scale = RobustScaler().fit(z)
    np.testing.assert_allclose(est['scaler'].center_, scale.center_, rtol=0, atol=0)
    np.testing.assert_allclose(est['scaler'].scale_, scale.scale_, rtol=0, atol=0)
    k = 'all' if c['keep'] == 0 else min(c['keep'], x.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); sel = SelectKBest(f_classif, k=k).fit(scale.transform(z), y)
    np.testing.assert_array_equal(est['selector'].get_support(), sel.get_support())
    np.testing.assert_allclose(est['selector'].scores_, sel.scores_, equal_nan=True)


def score(c):
    validate(c)
    code = hashlib.sha256(canonical(c).encode()).hexdigest()[:20]
    path = L/'discovery'/code; marker = path/'complete.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        assert saved['configuration'] == c and saved['signature'] == signature()
        assert all(sha256(path/f) == h for f, h in saved['outputs'].items())
        return saved['summary']
    path.mkdir(parents=True, exist_ok=True)
    d = load('discovery'); cols = columns(c, d); x = d[cols].to_numpy(float)
    y, groups = d.y.to_numpy(), d.pair_id.to_numpy(); rows = []; all_y = []; all_p = []
    for repeat in SEEDS:
        for fold, (tr, va) in enumerate(splits(y, groups, n=3, seed=repeat)):
            stem = f'{repeat}_{fold}'
            est, ws = fit_and_log(c, x[tr], y[tr], path/f'{stem}.joblib')
            p = est.predict_proba(x[va])[:, 1]
            train_auc = roc_auc_score(y[tr], est.predict_proba(x[tr])[:, 1])
            np.savez_compressed(path/f'{stem}.npz', tr=tr, va=va, p=p)
            dump(path/f'{stem}.json', dict(warnings=ws, columns=cols))
            rows.append(dict(repeat=repeat, fold=fold, train_auc=train_auc,
                             **measures(y[va], p, c['threshold'])))
            all_y.extend(y[va]); all_p.extend(p)
    df = pd.DataFrame(rows); df.to_csv(path/'fold_metrics.csv', index=False)
    tau = operating_threshold(all_y, all_p)
    summary = dict(configuration=c, cache_id=code, mean_auc=float(df.auc.mean()),
                   repeat_auc={str(s): float(df[df.repeat == s].auc.mean()) for s in SEEDS},
                   mean_train_auc=float(df.train_auc.mean()), mean_brier=float(df.brier.mean()),
                   mean_ap=float(df.average_precision.mean()),
                   recommended_threshold_metrics=measures(all_y, all_p, c['threshold']),
                   exact_sensitivity85_threshold=tau,
                   exact_sensitivity85_metrics=measures(all_y, all_p, tau))
    dump(marker, dict(configuration=c, signature=signature(), summary=summary,
                     outputs={p.name: sha256(p) for p in path.iterdir() if p != marker}))
    print(f"Scored {c['family']}/{c['inputs']}/keep={c['keep']}: discovery AUC {summary['mean_auc']:.4f}", flush=True)
    return summary


def gate(s, base):
    gains = {str(r): s['repeat_auc'][str(r)]-base['repeat_auc'][str(r)] for r in SEEDS}
    gain = s['mean_auc']-base['mean_auc']
    return dict(gain=gain, repeat_gains=gains,
                eligible=bool(gain >= .002 and all(g > 0 for g in gains.values())))


def choose(candidates, base):
    valid = [c for c in candidates if 'summary' in c]
    ordered = sorted(valid, key=lambda r: (-r['summary']['mean_auc'],
                                         r['summary']['mean_brier'], r['slot']))
    eligible = [r for r in ordered if gate(r['summary'], base)['eligible']]
    return dict(best_searched=ordered[0] if ordered else None,
                selected=eligible[0]['summary'] if eligible else base,
                replaced_reference=bool(eligible),
                gate=gate(eligible[0]['summary'], base) if eligible else None)


def suggest(trial):
    fam = trial.suggest_categorical('family', list(SPACE))
    return validate(dict(family=fam, inputs=trial.suggest_categorical('inputs', INPUTS),
                         keep=trial.suggest_categorical('keep', [0, 10, 20]),
                         params={k: trial.suggest_categorical(f'{fam}.{k}', v) for k, v in SPACE[fam].items()},
                         threshold=trial.suggest_categorical('threshold', THRESHOLDS)))


def flat(c):
    return dict(family=c['family'], inputs=c['inputs'], keep=c['keep'], threshold=c['threshold'],
                **{f"{c['family']}.{k}": v for k, v in c['params'].items()})


SYSTEM = '''You are one scientific agent selecting bounded model configurations for
an exploratory Parkinson's smartphone classification experiment. You may choose only
the listed input combination, variable-selection size, model family, hyperparameters
and recommended threshold. Study the discovery-only validation feedback and revise
your choices. Return JSON, never executable code. No records or evaluation outcomes
are available to you. Do not claim causality or verified gains. Do not change cohort,
labels, folds, preprocessing, objectives, budgets or selection rules. Threshold changes
cannot improve ROC-AUC. Explain your reasoning briefly and propose exactly four
configurations per round, using complete parameter dictionaries from the allowed
space. Consider validation AUC, training-validation gaps, calibration and operating
metrics. Do not invent measurements. There is no required AUC target.'''


def context():
    d = load('discovery'); fs = feature_sets(d)
    dictionary = (ROOT/'paper_ready/FEATURE_DICTIONARY.md').read_text()
    definitions = [line for line in dictionary.splitlines()
                   if any(line.startswith('| '+c+' |') for c in fs['full'])]
    return dict(study='Self-administered mPower tasks; self-reported professional PD diagnosis versus controls. Same-person modalities, not synchronized. Matched by age and sex. Retrospective classification, not early disease diagnosis.',
                discovery_n=len(d), discovery_pd=int(d.y.sum()), discovery_pairs=d.pair_id.nunique(),
                data_limits='Tapping timing/button events; walking/pedometer summaries, not raw tremor or accelerometry; whole-record acoustic summaries without validated isolated phonation segmentation; age, sex, smoking. No medication features. Missing recordings remain imputed, not replaced.',
                feature_definitions=definitions, input_combinations=fs, parameter_space=SPACE,
                fixed_settings='Seed42, single-thread learners; median imputation and RobustScaler then ANOVA top-k, all fit only on training folds. LR max_iter3000; XGB histogram subsample1 colsample1; RF baseline; MLP L-BFGS max_iter2000 max_fun50000; other settings in recorded defaults.',
                keep_options=[0, 10, 20], keep_meaning='0 means all; otherwise choose min(k,available) highest ANOVA F scores within each training fold. Agent chooses selection rule, not data-derived feature rankings.',
                threshold_options=THRESHOLDS, threshold_meaning='Secondary recommended numeric cutoff; not used in primary AUC. All finalists also receive exact discovery-OOF sensitivity>=85% max-specificity threshold and default0.5.',
                selection='Primary mean-fold AUC, 2 repetitions of 3-fold pair-grouped CV. Must improve best default-family mean AUC by>=.002 and be positive in BOTH repetitions; otherwise default retained. Pick highest eligible AUC, ties lower Brier then slot. Scores are noisy, not clinical evidence.',
                budget='3 rounds,4 slots each. Invalid/duplicate configurations consume slots. Same space and trial count for random/TPE comparators. No evaluation feedback. Aggregate information only.',
                instructions='Return analysis:string and configurations:[{rationale:string,configuration:{family,inputs,keep,params,threshold}}]. All parameters for chosen family required; no extra parameters. Use provided numeric values exactly.')


def configuration_schema():
    variants = []
    for family, params in SPACE.items():
        properties = {}
        for key, values in params.items():
            types = sorted(set('string' if isinstance(v, str) else
                               'integer' if isinstance(v, int) else 'number' for v in values))
            properties[key] = dict(enum=values)
            if set(types) <= {'integer', 'number'}:
                properties[key]['type'] = 'number'
            elif len(types) == 1:
                properties[key]['type'] = types[0]
        variants.append(dict(type='object', additionalProperties=False,
            required=['family', 'inputs', 'keep', 'params', 'threshold'], properties={
            'family': {'type': 'string', 'enum': [family]},
            'inputs': {'type': 'string', 'enum': INPUTS},
            'keep': {'type': 'integer', 'enum': [0, 10, 20]},
            'threshold': {'type': 'number', 'enum': THRESHOLDS},
            'params': dict(type='object', additionalProperties=False,
                           required=list(params), properties=properties)}))
    return {'anyOf': variants}


class Client:
    def __init__(self, key):
        self.headers = {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}
        self.ledger_path = P/'api_ledger.json'
        # Standalone release: this ledger accounts for this run only.
        self.prior_cost = 0.0
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else []
        r = requests.get(BASE+'/key', headers=self.headers, timeout=30)
        if r.status_code != 200: raise RuntimeError(f'Credential check HTTP {r.status_code}; no inference')
        r = requests.get(BASE+'/endpoints/zdr', timeout=30); r.raise_for_status()
        es = [e for e in r.json()['data'] if e.get('model_id') == MODEL and
              e.get('tag') == TAG and e.get('safety_retention') is False]
        assert len(es) == 1, 'Pinned endpoint unavailable'
        e = es[0]
        assert float(e['pricing']['prompt']) <= 3e-6 and float(e['pricing']['completion']) <= 15e-6
        assert 'response_format' in e['supported_parameters']
        dump(P/'provider_preflight.json', dict(endpoint=e, checked_unix=time.time()))

    def call(self, number, user):
        cs = configuration_schema()
        schema = dict(type='object', additionalProperties=False, required=['analysis', 'configurations'],
                      properties={'analysis': {'type': 'string'}, 'configurations': dict(type='array',
                      minItems=4, maxItems=4, items=dict(type='object', additionalProperties=False,
                      required=['rationale', 'configuration'], properties={
                      'rationale': {'type': 'string'}, 'configuration': cs}))})
        body = dict(model=MODEL, messages=[dict(role='system', content=SYSTEM),
                    dict(role='user', content=json.dumps(safe(user), ensure_ascii=True))],
                    temperature=.2, max_tokens=6000, reasoning={'enabled': False},
                    response_format=dict(type='json_schema', json_schema=dict(name='configuration_proposals', strict=True, schema=schema)),
                    provider=dict(only=[TAG], allow_fallbacks=False, require_parameters=True,
                                  data_collection='deny', zdr=True, max_price={'prompt': 3, 'completion': 15}), stream=False)
        request = P/f'api/round{number}_request.json'; output = P/f'api/round{number}_response.json'
        if output.exists():
            assert json.loads(request.read_text()) == body
            assert next(e for e in self.ledger if e['round'] == number)['status'] == 'completed'
            return json.loads(json.loads(output.read_text())['choices'][0]['message']['content'])
        assert not request.exists(), 'Unresolved request; no retry'
        size = len(json.dumps(body, ensure_ascii=True).encode())
        reserve = (size+2048)*3e-6 + 6000*15e-6
        assert len(self.ledger) < 3 and self.prior_cost + sum(e['reserved_usd'] for e in self.ledger)+reserve <= CAP
        dump(request, body)
        entry = dict(round=number, reserved_usd=reserve, status='dispatched', started_unix=time.time())
        self.ledger.append(entry); dump(self.ledger_path, self.ledger)
        try:
            r = requests.post(BASE+'/chat/completions', headers=self.headers, json=body, timeout=(20, 120))
        except requests.RequestException as exc:
            entry.update(status='network_failure_no_retry', error_type=type(exc).__name__)
            dump(self.ledger_path, self.ledger); raise RuntimeError('Network failure; no retry') from None
        entry.update(http_status=r.status_code)
        if r.status_code != 200:
            entry['status'] = 'http_failure_no_retry'; dump(self.ledger_path, self.ledger)
            dump(P/f'api/round{number}_error.json', r.json())
            raise RuntimeError(f'Inference HTTP {r.status_code}; no retry')
        response = r.json(); dump(output, response)
        entry.update(status='received', usage=response.get('usage'), finished_unix=time.time())
        dump(self.ledger_path, self.ledger)
        assert response.get('provider') == 'Amazon Bedrock' and response.get('model') == MODEL
        cost = response['usage']['cost']; assert isinstance(cost, (int, float)) and 0 <= cost <= reserve
        assert response['choices'][0]['finish_reason'] == 'stop'
        answer = json.loads(response['choices'][0]['message']['content'])
        assert isinstance(answer['analysis'], str) and len(answer['configurations']) == 4
        entry['status'] = 'completed'; dump(self.ledger_path, self.ledger)
        return answer


def preflight():
    P.mkdir(parents=True, exist_ok=True); L.mkdir(parents=True, exist_ok=True)
    d, e = load('discovery'), load('evaluation')
    assert len(d) == 492 and len(e) == 210 and d.y.sum() == 246 and e.y.sum() == 105
    assert not set(d.participant_id) & set(e.participant_id)
    assert not set(d.pair_id) & set(e.pair_id)
    assert d.participant_id.is_unique and e.participant_id.is_unique
    for v in [d, e]:
        assert v.groupby('pair_id').size().eq(2).all()
        assert v.groupby('pair_id').y.sum().eq(1).all()
        assert v.groupby('pair_id').demo_gender_male.nunique().eq(1).all()
        assert v.groupby('pair_id').demo_age.agg(lambda x: x.max()-x.min()).le(2).all()
    assert len(feature_sets(d)['full']) == 52
    start = P/'start_manifest.json'
    if start.exists(): assert json.loads(start.read_text())['signature'] == signature()
    else: dump(start, dict(signature=signature(), created_unix=time.time(), discovery_n=492, evaluation_n=210,
                          versions={k: importlib.metadata.version(k) for k in
                                    ['numpy', 'pandas', 'scikit-learn', 'optuna', 'xgboost', 'lightgbm', 'catboost']}))
    print('Preflight passed: 702 unique people; disjoint 492/210 paired split; matching and hashes verified.', flush=True)


def discover(key):
    preflight()
    if (P/'frozen.json').exists():
        assert json.loads((P/'frozen.json').read_text())['signature'] == signature(); return
    client = Client(key)
    baseline = {k: score(c) for k, c in defaults().items()}
    best_name = sorted(baseline, key=lambda k: (-baseline[k]['mean_auc'], k))[0]
    base = baseline[best_name]; dump(P/'baseline_scores.json', baseline)
    history = []; agent = []
    ctx = context(); dump(P/'api_context.json', ctx)
    for r in range(1, 4):
        # Only discovery results and configurations enter the payload.
        user = dict(context=ctx, round=r, baseline_scores=baseline, reference=base,
                    previous_agent_rounds=history)
        answer = client.call(r, user); records = []
        for i, proposal in enumerate(answer['configurations']):
            record = dict(slot=(r-1)*4+i, rationale=proposal.get('rationale', ''), proposal=proposal.get('configuration'))
            try: c = validate(proposal['configuration'])
            except (AssertionError, KeyError, TypeError): record['invalid'] = 'schema_or_range_violation'
            else:
                record['summary'] = score(c); record['gate'] = gate(record['summary'], base)
            records.append(record)
        history.append(dict(round=r, analysis=answer['analysis'], candidates=records)); agent.extend(records)
        dump(P/f'round{r}_feedback.json', history[-1]); print(f'Agent feedback round {r}/3 complete.', flush=True)
    client.headers.clear()
    arms = {'agent': agent}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for name, sampler in [('random', optuna.samplers.RandomSampler(seed=20261120)),
                          ('optuna', optuna.samplers.TPESampler(seed=20261120, n_startup_trials=6))]:
        study = optuna.create_study(direction='maximize', sampler=sampler)
        for c in defaults().values(): study.enqueue_trial(flat(c))
        for _ in range(6):
            t = study.ask(); c = suggest(t); s = score(c); study.tell(t, s['mean_auc'])
        records = []
        for slot in range(12):
            t = study.ask(); c = suggest(t); s = score(c); study.tell(t, s['mean_auc'])
            records.append(dict(slot=slot, summary=s, gate=gate(s, base)))
            dump(P/f'{name}_trials.json', records)
        dump(P/f'{name}_sampler.json', dict(name=type(sampler).__name__, seed=20261120,
                                           warm_start_trials=6, additional_trials=12))
        arms[name] = records
    decisions = {name: choose(rows, base) for name, rows in arms.items()}
    decisions['baseline'] = dict(selected=base, replaced_reference=False)
    dump(P/'all_candidates.json', arms); dump(P/'selection.json', decisions)
    verify_discovery()
    dump(P/'frozen.json', dict(signature=signature(), decisions=decisions, created_unix=time.time(),
         provenance={str(p.relative_to(P)): sha256(p) for p in P.rglob('*') if p.is_file() and p.name != 'frozen.json'}))
    print('Search and discovery audit complete. All decisions frozen before evaluation.', flush=True)


def verify_discovery():
    d = load('discovery'); y = d.y.to_numpy(); g = d.pair_id.to_numpy(); count = 0; warnings_n = 0
    for marker in sorted((L/'discovery').glob('*/complete.json')):
        m = json.loads(marker.read_text()); path = marker.parent; c = m['configuration']
        assert m['signature'] == signature()
        assert all(sha256(path/f) == h for f, h in m['outputs'].items())
        x = d[columns(c, d)].to_numpy(float); aucs = []
        for repeat in SEEDS:
            for fold, (tr, va) in enumerate(splits(y, g, n=3, seed=repeat)):
                stem = f'{repeat}_{fold}'; a = np.load(path/f'{stem}.npz')
                np.testing.assert_array_equal(a['tr'], tr); np.testing.assert_array_equal(a['va'], va)
                est = joblib.load(path/f'{stem}.joblib'); check_fit(est, c, x[tr], y[tr])
                p = est.predict_proba(x[va])[:, 1]; np.testing.assert_allclose(a['p'], p, atol=1e-12, rtol=0)
                aucs.append(roc_auc_score(y[va], p)); count += 1
                warnings_n += len(json.loads((path/f'{stem}.json').read_text())['warnings'])
        np.testing.assert_allclose(np.mean(aucs), m['summary']['mean_auc'], atol=1e-12, rtol=0)
    # Payload checks against exact participant identifiers and forbidden data fields.
    identifiers = list(d.participant_id.astype(str))
    for request in (P/'api').glob('*_request.json'):
        text = request.read_text()
        assert not any(identifier in text for identifier in identifiers)
        assert 'sk-or-' not in text
        body = json.loads(text); user = json.loads(body['messages'][1]['content'])
        assert set(user) == {'context', 'round', 'baseline_scores', 'reference', 'previous_agent_rounds'}
    dump(P/'discovery_verification.json', dict(status='passed', replayed_fits=count,
                                              warning_count=warnings_n, train_only_preprocessing=True,
                                              payloads='Feature definitions and aggregate discovery feedback only'))
    print(f'Discovery audit passed: {count} model replays and training-only preprocessing checks.', flush=True)


def evaluate():
    preflight(); frozen = json.loads((P/'frozen.json').read_text())
    assert frozen['signature'] == signature()
    assert all(sha256(P/f) == h for f, h in frozen['provenance'].items())
    d, e = load('discovery'), load('evaluation'); rows = []
    for arm, decision in frozen['decisions'].items():
        summary = decision['selected']; c = summary['configuration']; cols = columns(c, d)
        x, y = d[cols].to_numpy(float), d.y.to_numpy(); xt = e[cols].to_numpy(float)
        path = L/'evaluation'/arm; path.mkdir(parents=True, exist_ok=True)
        marker = path/'complete.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            assert saved['frozen_sha256'] == sha256(P/'frozen.json')
            assert all(sha256(path/f) == h for f, h in saved['outputs'].items())
        else:
            est, ws = fit_and_log(c, x, y, path/'model.joblib')
            p = est.predict_proba(xt)[:, 1]
            pd.DataFrame(dict(row_index=np.arange(len(e)), probability=p)).to_parquet(path/'predictions.parquet', index=False)
            dump(path/'details.json', dict(warnings=ws, columns=cols, configuration=c,
                                           selected_columns=np.array(cols)[est['selector'].get_support()].tolist()))
            dump(marker, dict(frozen_sha256=sha256(P/'frozen.json'),
                              outputs={p.name: sha256(p) for p in path.iterdir() if p != marker}))
        est = joblib.load(path/'model.joblib'); check_fit(est, c, x, y)
        p = est.predict_proba(xt)[:, 1]
        np.testing.assert_allclose(p, pd.read_parquet(path/'predictions.parquet').probability, atol=1e-12, rtol=0)
        for policy, tau in [('default', .5), ('recommended', c['threshold']),
                            ('sensitivity85', summary['exact_sensitivity85_threshold'])]:
            rows.append(dict(arm=arm, policy=policy, threshold=tau, family=c['family'], inputs=c['inputs'],
                             kept_variables=int(est['selector'].get_support().sum()), **measures(e.y, p, tau)))
    pd.DataFrame(rows).to_csv(P/'evaluation_metrics.csv', index=False)
    dump(P/'evaluation_verification.json', dict(status='passed', n=210, fitted_on=492,
                                               model_replays=4, frozen_sha256=sha256(P/'frozen.json'),
                                               discovery_evaluation_disjoint=True))
    print('Evaluation complete and replayed: four frozen procedures on 210 held-out participants.', flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['preflight', 'discover', 'evaluate', 'verify'])
    stage = parser.parse_args().stage
    if stage == 'preflight': preflight()
    elif stage == 'discover':
        key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter key (hidden, memory only): ')
        try: discover(key)
        finally: key = None
    elif stage == 'evaluate': evaluate()
    else: verify_discovery()


if __name__ == '__main__': main()
