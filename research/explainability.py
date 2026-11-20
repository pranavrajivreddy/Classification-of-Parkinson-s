"""Exploratory model explanations; deliberately NOT a clinical causal estimator."""
import json
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
import shap
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from research.common import ROOT, OUT, PRIVATE, sha256, write_json
from research.core import feature_sets
from research.cohort_spectrum_data import LOCAL as SPECTRUM

PUBLIC = OUT/'explainability'
LOCAL = PRIVATE/'explainability'
SOURCE = SPECTRUM/'matched.parquet'
MODELS = SPECTRUM/'runs'/'matched_full'
BLOCKS = ['tapping', 'walking', 'voice']
LEVELS = [-.2, -.1, 0., .1, .2]


def transform(fit, x):
    return fit[:-1].transform(x)


def walking_scenario(x, columns, fraction):
    z = x.copy()
    if fraction == 0:
        return z
    ix = {name: columns.index(name) for name in feature_sets(pd.DataFrame(columns=columns))['walking']}
    steps = np.maximum(1, np.rint(x[:, ix['walk_steps_total']]*(1+fraction)))
    duration = x[:, ix['walk_duration_sec']]
    distance = steps*x[:, ix['walk_step_length']]
    z[:, ix['walk_steps_total']] = steps
    z[:, ix['walk_distance_total']] = distance
    z[:, ix['walk_speed_mean']] = distance/duration
    z[:, ix['walk_cadence']] = 60*steps/duration
    return z


def check_algebra(x, columns):
    a = {c: x[:, j] for j, c in enumerate(columns)}
    for observed, expected in [
        (a['walk_cadence'], 60*a['walk_steps_total']/a['walk_duration_sec']),
        (a['walk_speed_mean'], a['walk_distance_total']/a['walk_duration_sec']),
        (a['walk_step_length'], a['walk_distance_total']/a['walk_steps_total']),
        (a['tap_count_total'], a['tap_count_valid']+a['tap_count_miss']),
        (a['tap_rate'], a['tap_count_valid']/a['tap_duration_sec']),
        (a['tap_miss_ratio'], a['tap_count_miss']/a['tap_count_total']),
        (a['tap_flight_cv'], a['tap_flight_std']/a['tap_flight_mean'])]:
        np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=1e-10)


def permutation_shap(predict, background, x, seed):
    """Marginal Shapley Monte Carlo: a permutation and its reverse per donor.

    Each path telescopes exactly. Background records receive equal weight;
    permutation error remains and is assessed with an independent second run.
    """
    rng = np.random.default_rng(seed); phi = np.zeros_like(x,dtype=float)
    p = x.shape[1]; n = len(x)
    for reference in background:
        order = rng.permutation(p)
        for sequence in [order,order[::-1]]:
            path = np.tile(reference,(p+1,n,1))
            for k,j in enumerate(sequence):
                path[k+1:,:,j] = x[:,j]
            scores = predict(path.reshape(-1,p)).reshape(p+1,n)
            phi[:,sequence] += np.diff(scores,axis=0).T
    phi /= (2*len(background))
    return phi,float(predict(background).mean())


def explanations(fit, background, x, seed):
    # Installed TreeSHAP failed random-forest additivity. Do not repair its base
    # value post hoc or relax tolerance; evaluate the actual predictor instead.
    if fit[-1].__class__.__name__ == 'RandomForestClassifier':
        return permutation_shap(lambda z:fit[-1].predict_proba(z)[:,1],background,x,seed)
    explainer = shap.TreeExplainer(fit[-1], data=background,
        feature_perturbation='interventional', model_output='probability')
    values = np.asarray(explainer.shap_values(x))
    base = np.asarray(explainer.expected_value)
    if values.ndim == 3:
        values, base = values[:, :, 1], base[1]
    return values, float(base)


def signature():
    return dict(protocol=sha256(PUBLIC/'PROTOCOL.md'), code=sha256(__file__),
                cohort=sha256(SOURCE), choices=sha256(MODELS/'choices.json'),
                predictions=sha256(MODELS/'predictions.parquet'),
                versions={p: importlib.metadata.version(p) for p in
                          ['shap', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib']})


def run():
    LOCAL.mkdir(parents=True, exist_ok=True)
    sig = signature()
    if (LOCAL/'complete.json').exists():
        old = json.loads((LOCAL/'complete.json').read_text())
        assert old['signature'] == sig, 'Archive stale extension before rerun'
        assert all(sha256(LOCAL/n) == h for n, h in old['outputs'].items())
        print('Existing extension verified by hashes; no search repeated.', flush=True)
        return
    write_json(LOCAL/'run_start.json', sig)
    d = pd.read_parquet(SOURCE).reset_index(drop=True)
    reference = pd.read_parquet(PRIVATE/'cohort.parquet').set_index('participant_id')
    sets = feature_sets(d); cols = sets['full']; x = d[cols].to_numpy(float)
    pd.testing.assert_frame_equal(d.set_index('participant_id')[cols+['y','pair_id']],
        reference.loc[d.participant_id, cols+['y','pair_id']], check_dtype=False)
    assert len(d) == 702 and d.participant_id.is_unique and d.y.sum() == 351
    assert len(cols) == 52 and all(c.startswith(('demo_', 'tap_', 'walk_', 'voice_')) for c in cols)
    check_algebra(x, cols)
    predictions = pd.read_parquet(MODELS/'predictions.parquet')
    predictions = predictions[predictions.policy == 'default'].set_index('participant_id')
    attribution = np.zeros_like(x); second = np.zeros_like(x); baseline = np.zeros(len(d))
    donor_rows, cadence_rows, fold_records = [], [], []
    for choice in json.loads((MODELS/'choices.json').read_text()):
        fold = choice['fold']; tr = np.asarray(choice['train_indices']); te = np.asarray(choice['test_indices'])
        assert not set(d.iloc[tr].pair_id) & set(d.iloc[te].pair_id)
        assert set(tr) | set(te) == set(range(len(d))) and not set(tr) & set(te)
        path = MODELS/f'fold{fold}.joblib'; assert sha256(path) == choice['model_sha256']
        fit = joblib.load(path); q = fit.predict_proba(x[te])[:, 1]; baseline[te] = q
        np.testing.assert_allclose(q, predictions.loc[d.iloc[te].participant_id, 'probability'], atol=1e-12, rtol=0)
        ztr, zte = transform(fit, x[tr]), transform(fit, x[te])
        rng = np.random.default_rng(20261119+fold); bg = rng.choice(len(tr), 100, replace=False)
        bg2 = np.random.default_rng(20261219+fold).choice(len(tr), 100, replace=False)
        combined,base=explanations(fit,ztr[bg],np.vstack([zte,ztr]),20261119+fold)
        sv,train_sv=combined[:len(te)],combined[len(te):]; train_base=base
        sv2, base2 = explanations(fit, ztr[bg2], zte,20261219+fold)
        err = float(np.max(np.abs(base+sv.sum(1)-q)))
        err2 = float(np.max(np.abs(base2+sv2.sum(1)-q)))
        train_err = float(np.max(np.abs(train_base+train_sv.sum(1)-fit.predict_proba(x[tr])[:,1])))
        assert max(err, err2, train_err) < 1e-6
        attribution[te], second[te] = sv, sv2
        mean_train = np.abs(train_sv).mean(0)
        sensor_indices = [cols.index(c) for c in sets['sensors']]
        top_index = max(sensor_indices, key=lambda j: mean_train[j])
        top = cols[top_index]; guided = next(b for b in BLOCKS if top in sets[b])
        distances = cdist(ztr, ztr)
        samepair = d.iloc[tr].pair_id.to_numpy()[:, None] == d.iloc[tr].pair_id.to_numpy()[None, :]
        distances[samepair] = np.inf
        cutoff = float(np.quantile(distances.min(1), .95))
        wi = [cols.index(c) for c in sets['walking']]
        low, high = np.quantile(x[tr][:, wi], [.01, .99], axis=0)
        folds_cf = []
        for block in BLOCKS:
            bi = [cols.index(c) for c in sets[block]]
            unchanged = [j for j in range(len(cols)) if j not in bi]
            for pos, i in enumerate(te):
                eligible = ((np.abs(d.iloc[tr].demo_age.to_numpy()-d.iloc[i].demo_age) <= 5)
                    & (d.iloc[tr].demo_gender_male.to_numpy() == d.iloc[i].demo_gender_male)
                    & np.all(np.isnan(x[tr][:,bi]) == np.isnan(x[i,bi]), axis=1))
                candidates = np.flatnonzero(eligible)
                near = np.argsort(np.sum((ztr[candidates]-zte[pos])**2, axis=1), kind='stable')[:30]
                donors = tr[candidates[near]]
                cf = np.tile(x[i], (len(donors), 1)); cf[:,bi] = x[donors][:,bi]
                np.testing.assert_equal(cf[:,unchanged], np.tile(x[i,unchanged], (len(cf),1)))
                assert np.array_equal(np.isnan(cf), np.tile(np.isnan(x[i]), (len(cf),1)))
                check_algebra(cf, cols)
                zz = transform(fit, cf) if len(cf) else np.empty((0,len(cols)))
                support = cdist(zz, ztr).min(1) <= cutoff if len(cf) else np.zeros(0, bool)
                accepted = np.flatnonzero(support)
                selected_donor = -1; chosen = x[i].copy(); newscore = float(q[pos]); distance = 0.
                crossings = 0
                if len(accepted):
                    probs = fit.predict_proba(cf[accepted])[:,1]
                    change_distance = np.linalg.norm(zz[accepted]-zte[pos], axis=1)
                    cross = np.flatnonzero((probs < .5) & (q[pos] >= .5)); crossings = len(cross)
                    j = int(cross[np.argmin(change_distance[cross])]) if len(cross) else int(np.argmin(probs))
                    if probs[j] < q[pos]:
                        selected_donor = int(donors[accepted[j]])
                        chosen = cf[accepted[j]].copy(); newscore = float(probs[j]); distance = float(change_distance[j])
                row = dict(row_index=int(i), fold=fold, block=block, guided=(block == guided),
                    donor_index=selected_donor, eligible_donors=int(eligible.sum()), attempted=len(donors),
                    supported=len(accepted), crossing_candidates=crossings, before=float(q[pos]),
                    after=newscore, delta=newscore-float(q[pos]), change_distance=distance,
                    baseline_positive=bool(q[pos]>=.5), crossed=bool(q[pos]>=.5 and newscore<.5))
                donor_rows.append(row); folds_cf.append(chosen)
        np.savez_compressed(LOCAL/f'fold{fold}_donor_inputs.npz', inputs=np.asarray(folds_cf),
                            train_indices=tr, test_indices=te, background_indices=tr[bg],
                            second_background_indices=tr[bg2])
        scenarios = []
        for level in LEVELS:
            cf = walking_scenario(x[te], cols, level); check_algebra(cf, cols)
            unchanged = [j for j in range(len(cols)) if j not in wi]
            np.testing.assert_equal(cf[:,unchanged], x[te][:,unchanged])
            assert np.array_equal(np.isnan(cf), np.isnan(x[te]))
            bounded = ((cf[:,wi]>=low) & (cf[:,wi]<=high)).all(1)
            supported = cdist(transform(fit, cf), ztr).min(1) <= cutoff
            accepted = (bounded & supported) if level else np.ones(len(te), bool)
            probs = fit.predict_proba(cf)[:,1]
            if level == 0: np.testing.assert_allclose(probs, q, atol=1e-14)
            cadence = cols.index('walk_cadence')
            for pos, i in enumerate(te):
                cadence_rows.append(dict(row_index=int(i), fold=fold, requested_fraction=level,
                    accepted=bool(accepted[pos]), within_bounds=bool(bounded[pos]), supported=bool(supported[pos]),
                    actual_cadence_change=float(cf[pos,cadence]-x[i,cadence]),
                    before=float(q[pos]), after=float(probs[pos]), delta=float(probs[pos]-q[pos])))
            scenarios.append(cf)
        np.savez_compressed(LOCAL/f'fold{fold}_cadence_inputs.npz', inputs=np.asarray(scenarios))
        fold_records.append(dict(fold=fold, model=choice['selected'], model_sha256=sha256(path),
            shap_method=('antithetic_permutation_200_paths' if choice['selected']=='rf' else 'TreeSHAP'),
            n_train=len(tr), n_test=len(te), support_cutoff=cutoff,
            top_training_sensor=top, guided_block=guided, background_expected_score=base,
            shap_max_additivity_error=err, second_max_additivity_error=err2,
            training_max_additivity_error=train_err,
            training_importance=dict(zip(cols, map(float,mean_train))),
            test_importance=dict(zip(cols, map(float,np.abs(sv).mean(0))))))
        print(f'Explanation fold {fold+1}/5: {choice["selected"]}, SHAP error {err:.2g}; all scenarios retained', flush=True)
    pd.DataFrame(donor_rows).to_parquet(LOCAL/'donor_attempts.parquet', index=False)
    pd.DataFrame(cadence_rows).to_parquet(LOCAL/'cadence_scenarios.parquet', index=False)
    np.savez_compressed(LOCAL/'shap.npz', values=attribution, second_values=second, probabilities=baseline)
    write_json(LOCAL/'folds.json', fold_records)
    outputs = {p.name: sha256(p) for p in sorted(LOCAL.iterdir()) if p.is_file() and p.name != 'complete.json'}
    write_json(LOCAL/'complete.json', dict(signature=sig, outputs=outputs,
        n=702, auc=float(roc_auc_score(d.y, baseline)), clinical_causal_effect_identified=False))


if __name__ == '__main__':
    run()
