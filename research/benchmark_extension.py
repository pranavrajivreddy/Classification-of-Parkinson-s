"""Additional nested benchmarks; never modify the frozen earlier experiments.

All fit APIs take training labels only. Held-out outcomes enter reporting only.
Participant-level artifacts stay under paper_ready/private.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from research.agent_features import augment
from research.common import ROOT, PRIVATE, sha256, write_json
from research.core import feature_sets, metrics, operating_table, pipeline, select_thresholds, splits

PUBLIC = ROOT / 'paper_ready/benchmark_extension'
LOCAL = PRIVATE / 'benchmark_extension'
FAMILIES = ['lr', 'rf', 'xgboost', 'lightgbm', 'catboost', 'mlp']
TRIALS = 8
MODES = ['uncalibrated', 'sigmoid', 'isotonic', 'sigmoid_isotonic']
POLICIES = ['default', 'utility', 'sensitivity90', 'specificity60', 'f2_max']


def thresholds(y, p):
    out = select_thresholds(y, p)
    table = operating_table(y, p)
    table['f2'] = 5*table.tp/(5*table.tp+4*table.fn+table.fp)
    best = table.sort_values(['f2', 'specificity', 'threshold'], ascending=False).iloc[0]
    out['f2_max'] = float(best.threshold)
    return out


def load_data(cohort, config, arm):
    if cohort == 'n702':
        assert arm == 'raw'
        path = PRIVATE / 'cohort.parquet'
        folds = PRIVATE / f'runs/{config}_seed42/fold_assignments.csv'
    else:
        assert cohort == 'n492'
        path = PRIVATE / 'agent_extension/evaluation.parquet'
        folds = PRIVATE / f'agent_extension/runs/{config}_{arm}/fold_assignments.csv'
    df = pd.read_parquet(path)
    assignment = pd.read_csv(folds).set_index('participant_id').loc[df.participant_id]
    assert df.participant_id.is_unique and len(assignment) == len(df)
    np.testing.assert_array_equal(assignment.pair_id, df.pair_id)
    columns = feature_sets(df)[config]
    spec = json.loads((ROOT/'paper_ready/agent_extension/frozen_features.json').read_text())
    additions = [] if arm == 'raw' else [f for f in spec['features'] if f['name'] in spec['configuration_features'][config]]
    frame, added = augment(df, columns, additions)
    X = frame.replace([np.inf, -np.inf], np.nan).to_numpy(float)
    return df, X, columns+added, assignment.fold.to_numpy(int)


def parameters(trial, family):
    if family == 'lr':
        return dict(C=trial.suggest_float('C', .01, 100, log=True))
    if family == 'rf':
        return dict(n_estimators=160, max_depth=trial.suggest_categorical('max_depth', [3, 5, 8, None]),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 2, 12),
                    max_features=trial.suggest_categorical('max_features', ['sqrt', .7, 1.]))
    if family in ('xgboost', 'lightgbm'):
        p = dict(n_estimators=trial.suggest_categorical('n_estimators', [80, 160, 240]),
                 max_depth=trial.suggest_int('max_depth', 2, 5),
                 learning_rate=trial.suggest_float('learning_rate', .02, .15, log=True),
                 reg_lambda=trial.suggest_float('reg_lambda', .1, 20, log=True))
        if family == 'xgboost':
            p['min_child_weight'] = trial.suggest_float('min_child_weight', 1, 10, log=True)
        else:
            p['num_leaves'] = trial.suggest_categorical('num_leaves', [7, 15, 31])
            p['min_child_samples'] = trial.suggest_int('min_child_samples', 10, 40)
        return p
    if family == 'catboost':
        return dict(iterations=trial.suggest_categorical('iterations', [80, 160, 240]),
                    depth=trial.suggest_int('depth', 2, 5),
                    learning_rate=trial.suggest_float('learning_rate', .02, .15, log=True),
                    l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1, 20, log=True))
    if family == 'mlp':
        return dict(hidden_layer_sizes=trial.suggest_categorical('hidden_layer_sizes', ['16', '32', '64', '32,16']),
                    alpha=trial.suggest_float('alpha', .0001, .1, log=True),
                    learning_rate_init=trial.suggest_float('learning_rate_init', .0005, .005, log=True))
    raise ValueError(family)


def estimator(family, params, seed):
    if family == 'lr':
        model = LogisticRegression(**params, max_iter=3000, random_state=seed)
    elif family == 'rf':
        model = RandomForestClassifier(**params, random_state=seed, n_jobs=1)
    elif family == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(**params, random_state=seed, n_jobs=1, tree_method='hist', eval_metric='logloss')
    elif family == 'lightgbm':
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**params, random_state=seed, n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True)
    elif family == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**params, random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False)
    elif family == 'mlp':
        p = dict(params)
        p['hidden_layer_sizes'] = tuple(map(int, p['hidden_layer_sizes'].split(',')))
        model = MLPClassifier(**p, random_state=seed, max_iter=400, early_stopping=False, solver='adam')
    else:
        raise ValueError(family)
    return pipeline(model)


def capture_fit(est, X, y, warning_counts):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        est.fit(X, y)
    for item in caught:
        key = item.category.__name__ + ': ' + str(item.message)
        warning_counts[key] = warning_counts.get(key, 0)+1
    return est


def tune_fit(X, y, groups, Xtest, family, seed, n_trials=TRIALS):
    """Outer-test labels are deliberately absent from this API."""
    inner = splits(y, groups, n=3, seed=143)
    candidates, records, warning_counts = [], [], {}
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=4))
    anchors = {
        'lr': dict(C=1.),
        'rf': dict(max_depth=5, min_samples_leaf=3, max_features='sqrt'),
        'xgboost': dict(n_estimators=160, max_depth=3, learning_rate=.05, reg_lambda=1., min_child_weight=3.),
        'lightgbm': dict(n_estimators=160, max_depth=3, learning_rate=.05, reg_lambda=1., num_leaves=7, min_child_samples=20),
        'catboost': dict(iterations=160, depth=3, learning_rate=.05, l2_leaf_reg=3.),
        'mlp': dict(hidden_layer_sizes='32', alpha=.001, learning_rate_init=.001),
    }
    study.enqueue_trial(anchors[family])

    def objective(trial):
        p = parameters(trial, family)
        pred = np.full(len(y), np.nan)
        scores = []
        for tr, va in inner:
            model = capture_fit(estimator(family, p, seed), X[tr], y[tr], warning_counts)
            pred[va] = model.predict_proba(X[va])[:, 1]
            scores.append(float(roc_auc_score(y[va], pred[va])))
        assert np.isfinite(pred).all()
        candidates.append(pred)
        records.append(dict(number=trial.number, params=p, fold_auc=scores, mean_auc=float(np.mean(scores))))
        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    chosen = study.best_trial.number
    est = capture_fit(estimator(family, records[chosen]['params'], seed), X, y, warning_counts)
    out = est.predict_proba(Xtest)[:, 1]
    inner_fold = np.full(len(y), -1)
    for i, (_, va) in enumerate(inner):
        inner_fold[va] = i
    return candidates[chosen], out, est, dict(trials=records, chosen=chosen,
        warning_counts=warning_counts, seed=seed), np.array(candidates), inner_fold


class Sigmoid:
    """Platt's smoothed binary targets, fitted on held-out log-odds scores."""
    def fit(self, scores, y):
        pos, neg = int(np.sum(y)), int(len(y)-np.sum(y))
        target = np.where(y == 1, (pos+1)/(pos+2), 1/(neg+2))
        scores = np.asarray(scores, float)
        def loss(ab):
            z = ab[0]*scores+ab[1]
            residual = expit(z)-target
            return float(np.mean(np.logaddexp(0, z)-target*z)), np.array([np.mean(residual*scores), np.mean(residual)])
        fit = minimize(loss, np.array([0., np.log((pos+1)/(neg+1))]), jac=True, method='BFGS', options={'gtol':1e-7, 'maxiter':1000})
        if not np.isfinite(fit.x).all() or np.max(np.abs(fit.jac)) > 1e-5:
            raise RuntimeError('Sigmoid calibration failed to converge')
        self.ab = fit.x
        return self

    def predict(self, scores):
        return expit(self.ab[0]*scores+self.ab[1])


def calibrate_fit(X, y, groups, Xtest, family, seed, boundary_records):
    """Fixed base hyperparameters; all four modes share identical base fits.

    Each calibration member is trained on 2/3 of its input. A separate 1/3
    calibrates single-stage methods. The two-stage chain splits that calibration
    partition by matched pair into disjoint sigmoid and isotonic subsets.
    """
    params = dict(C=1.) if family == 'lr' else dict(n_estimators=160, max_depth=3,
        learning_rate=.05, reg_lambda=1., min_child_weight=3.)
    outputs = {mode: [] for mode in MODES}
    warning_counts = {}
    for member, (tr, ca) in enumerate(splits(y, groups, n=3, seed=seed)):
        a, b = splits(y[ca], groups[ca], n=2, seed=seed+7)[0]
        stage_a, stage_b = ca[a], ca[b]
        sets = [set(groups[idx]) for idx in [tr, stage_a, stage_b]]
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
        boundary_records.append(dict(train_groups=sorted(sets[0]), sigmoid_groups=sorted(sets[1]),
                                     isotonic_groups=sorted(sets[2]), member=member))
        model = capture_fit(estimator(family, params, seed+member), X[tr], y[tr], warning_counts)
        pc = model.predict_proba(X[ca])[:, 1]
        pt = model.predict_proba(Xtest)[:, 1]
        sc, st = logit(np.clip(pc, 1e-7, 1-1e-7)), logit(np.clip(pt, 1e-7, 1-1e-7))
        sigmoid = Sigmoid().fit(sc, y[ca])
        isotonic = IsotonicRegression(out_of_bounds='clip').fit(pc, y[ca])
        chain_sigmoid = Sigmoid().fit(sc[a], y[stage_a])
        chain_iso = IsotonicRegression(out_of_bounds='clip').fit(chain_sigmoid.predict(sc[b]), y[stage_b])
        outputs['uncalibrated'].append(pt)
        outputs['sigmoid'].append(sigmoid.predict(st))
        outputs['isotonic'].append(isotonic.predict(pt))
        outputs['sigmoid_isotonic'].append(chain_iso.predict(chain_sigmoid.predict(st)))
    return {mode: np.mean(p, axis=0) for mode, p in outputs.items()}, warning_counts


def calibrated_crossfit(X, y, groups, Xtest, family, seed):
    inner = {mode: np.full(len(y), np.nan) for mode in MODES}
    audit, warning_counts = [], {}
    for tr, va in splits(y, groups, n=3, seed=143):
        boundaries = []
        values, wc = calibrate_fit(X[tr], y[tr], groups[tr], X[va], family, seed, boundaries)
        held = set(groups[va])
        for b in boundaries:
            assert not held & set(b['train_groups']+b['sigmoid_groups']+b['isotonic_groups'])
        audit.append(dict(threshold_holdout_groups=sorted(held), members=boundaries))
        for mode in MODES:
            inner[mode][va] = values[mode]
        for k, v in wc.items(): warning_counts[k] = warning_counts.get(k, 0)+v
    boundaries = []
    outer, wc = calibrate_fit(X, y, groups, Xtest, family, seed, boundaries)
    for k, v in wc.items(): warning_counts[k] = warning_counts.get(k, 0)+v
    return inner, outer, dict(crossfit=audit, final=boundaries, warning_counts=warning_counts)


def run_one(task):
    kind, cohort, config, arm, family, fold = task
    folder = LOCAL/kind/f'{cohort}_{config}_{arm}_{family}'/f'fold{fold}'
    if (folder/'complete.json').exists():
        saved = json.loads((folder/'complete.json').read_text())
        assert saved['protocol_sha256'] == sha256(PUBLIC/'protocol.json')
        for name, digest in saved['artifacts'].items(): assert sha256(folder/name) == digest
        return dict(task=task, cached=True)
    start = time.time()
    df, X, columns, outer_fold = load_data(cohort, config, arm)
    y, groups = df.y.to_numpy(int), df.pair_id.to_numpy()
    tr, te = np.flatnonzero(outer_fold != fold), np.flatnonzero(outer_fold == fold)
    assert not set(groups[tr]) & set(groups[te])
    folder.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        if kind == 'benchmark':
            ip, op, est, detail, trials, inner_fold = tune_fit(X[tr], y[tr], groups[tr], X[te], family, 42+fold)
            inner, outer = {'tuned':ip}, {'tuned':op}
            np.savez_compressed(folder/'trial_predictions.npz', probabilities=trials, inner_fold=inner_fold)
            joblib.dump(est, folder/'model.joblib', compress=3)
        else:
            inner, outer, detail = calibrated_crossfit(X[tr], y[tr], groups[tr], X[te], family, 42+fold)
    details = dict(kind=kind, cohort=cohort, configuration=config, arm=arm, family=family,
                   fold=fold, features=columns, n_train=len(tr), n_test=len(te), details=detail, thresholds={})
    rows, inner_rows = [], []
    for mode in inner:
        assert np.isfinite(inner[mode]).all() and np.isfinite(outer[mode]).all()
        assert ((outer[mode]>=0)&(outer[mode]<=1)).all()
        cuts = thresholds(y[tr], inner[mode])
        details['thresholds'][mode] = cuts
        it = df.iloc[tr][['participant_id','pair_id','y']].copy()
        it['mode'], it['probability'] = mode, inner[mode]
        inner_rows.append(it)
        for policy, tau in cuts.items():
            row = df.iloc[te][['participant_id','pair_id','y']].copy()
            row['mode'], row['policy'], row['threshold'] = mode, policy, tau
            row['probability'], row['prediction'], row['fold'] = outer[mode], (outer[mode]>=tau).astype(int), fold
            rows.append(row)
    pd.concat(rows).to_parquet(folder/'predictions.parquet', index=False)
    pd.concat(inner_rows).to_parquet(folder/'training_predictions.parquet', index=False)
    write_json(folder/'details.json', details)
    write_json(folder/'complete.json', dict(task=task, elapsed_seconds=time.time()-start,
        protocol_sha256=sha256(PUBLIC/'protocol.json'), artifacts={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='complete.json'}))
    return dict(task=task, seconds=round(time.time()-start, 1))


def tasks(kind):
    configs = list(feature_sets(pd.read_parquet(PRIVATE/'cohort.parquet')))
    cohorts = [('n702','raw'), ('n492','raw'), ('n492','agent')]
    return [(kind, cohort, config, arm, family, fold)
            for cohort, arm in cohorts
            for config in (configs if kind=='benchmark' else ['tapping','full'])
            for family in (FAMILIES if kind=='benchmark' else ['lr','xgboost'])
            for fold in range(5)]


def preflight():
    expected = {'n702':702, 'n492':492}
    dev = pd.read_parquet(PRIVATE/'agent_extension/discovery.parquet')
    totals = []
    canonical = {}
    for cohort, arm in [('n702','raw'), ('n492','raw'), ('n492','agent')]:
        for config in feature_sets(pd.read_parquet(PRIVATE/'cohort.parquet')):
            df, X, columns, fold = load_data(cohort, config, arm)
            assert len(df)==expected[cohort] and set(fold)==set(range(5))
            assert not any(s in c.lower() for c in columns for s in ['medication','device','createdon','sessions','handle','participant','pair_id'])
            if cohort=='n492':
                assert not set(df.participant_id)&set(dev.participant_id)
                assert not set(df.pair_id)&set(dev.pair_id)
            check = pd.DataFrame({'participant_id':df.participant_id,'fold':fold}).sort_values('participant_id').reset_index(drop=True)
            if cohort in canonical: pd.testing.assert_frame_equal(check, canonical[cohort])
            else: canonical[cohort] = check
            assert pd.DataFrame({'pair':df.pair_id,'fold':fold}).groupby('pair').fold.nunique().max()==1
            for f in range(5):
                tr, te = np.flatnonzero(fold!=f), np.flatnonzero(fold==f)
                assert not set(df.pair_id.iloc[tr])&set(df.pair_id.iloc[te])
                splits(df.y.to_numpy()[tr], df.pair_id.to_numpy()[tr], seed=143)
            totals.append(dict(cohort=cohort, arm=arm, configuration=config, n=len(df), features=len(columns)))
    write_json(PUBLIC/'preflight.json', dict(checked_at=datetime.now(timezone.utc).isoformat(), combinations=totals,
        benchmark_tasks=len(tasks('benchmark')), calibration_tasks=len(tasks('calibration')), status='passed'))
    print('Preflight passed: 27 data/configuration/arm combinations; matched pairs and discovery boundaries intact.', flush=True)


def freeze():
    assert not (PUBLIC/'protocol.json').exists(), 'Protocol already frozen'
    source = ROOT/'research/benchmark_extension.py'
    datafiles = [PRIVATE/'cohort.parquet', PRIVATE/'agent_extension/evaluation.parquet',
                 PRIVATE/'agent_extension/discovery.parquet', ROOT/'paper_ready/agent_extension/frozen_features.json']
    protocol = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), interpretation='Exploratory additional internal evaluation after prior results; not preregistered or independent validation',
        cohorts={'n702':'all 351 matched pairs, raw features', 'n492':'246 evaluation pairs, raw and frozen agent features; discovery excluded'},
        outer_folds='Existing saved participant assignments; five matched-pair grouped folds; no regeneration',
        configurations=list(feature_sets(pd.read_parquet(PRIVATE/'cohort.parquet'))),
        families=FAMILIES, trials_per_family_outer_fold=TRIALS,
        search='Optuna TPE, fixed anchor first then three random startup trials then four TPE trials, eight total; mean three-inner-fold AUC; fixed seed 42+outer_fold; no pruning or outer-test early stopping',
        preprocessing='Median imputation and RobustScaler fitted separately in every model training partition; all families get identical feature inputs',
        selection='Report each family and inner-AUC-selected family; no selection using pooled outer metrics. Inner OOF tuning scores are adaptively reused for thresholds and are not validation performance estimates; outer test remains untouched.',
        policies=POLICIES, f2_tie_rule='Highest F2, then specificity, then threshold; exact unique scores and all-negative decision',
        calibration=dict(configurations=['tapping','full'], families=['lr','xgboost'], modes=MODES,
            base='Fixed LR C=1 or XGBoost 160 trees, depth3, lr.05, lambda1, child_weight3; no tuning or family selection in calibration study',
            nesting='Threefold crossfit of entire calibration procedure for threshold fitting; inside each fit, threefold base/calibration ensemble. Identical base members across four modes.',
            chain='Separate matched-pair halves of each calibration partition for smoothed Platt sigmoid and isotonic; single-stage methods use whole calibration partition. Exploratory unequal calibration allocation by design; not claimed superior.',
            scores='Sigmoid fits clipped probability log-odds. Isotonic fits probability; chain maps sigmoid probability through isotonic.',
            limitation='Calibration on balanced case-control sampling does not estimate population PD risks; small calibration samples may overfit, especially isotonic.'),
        uncertainty='2000 matched-pair bootstrap draws of fixed outer predictions; conditional intervals, not full training uncertainty; no five-fold Wilcoxon or post-selection significance claims',
        fit_warnings='Record every training warning; retain nonconverged bounded MLP fits with explicit counts and limitation; no result-driven budget changes',
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [source,ROOT/'research/core.py',ROOT/'research/agent_features.py']},
        input_hashes={str(p.relative_to(ROOT)):sha256(p) for p in datafiles},
        versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','scipy','xgboost','lightgbm','catboost','optuna']})
    PUBLIC.mkdir(parents=True,exist_ok=True)
    write_json(PUBLIC/'protocol.json',protocol)
    print('Protocol frozen',sha256(PUBLIC/'protocol.json'),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['preflight','freeze','benchmark','calibration','smoke'])
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.action=='preflight': return preflight()
    if args.action=='freeze': return freeze()
    protocol=json.loads((PUBLIC/'protocol.json').read_text())
    for path,digest in {**protocol['source_hashes'],**protocol['input_hashes']}.items():
        assert sha256(ROOT/path)==digest, 'Frozen source/input changed: '+path
    work=tasks(args.action) if args.action!='smoke' else [
        ('benchmark','n702','full','raw',family,0) for family in FAMILIES]+[
        ('calibration','n492','full','agent',family,0) for family in ['lr','xgboost']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_one,t) for t in work]
        for i,f in enumerate(as_completed(futures),1):
            print(json.dumps(dict(completed=i,total=len(work),**f.result())),flush=True)


if __name__=='__main__': main()
