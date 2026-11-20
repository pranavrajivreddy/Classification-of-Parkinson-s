"""Replay every saved scenario and produce aggregate-only publication artifacts."""
import json
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from research.common import PRIVATE, sha256, write_json
from research.core import feature_sets
from research.explainability import PUBLIC, LOCAL, SOURCE, MODELS, BLOCKS, LEVELS, signature, check_algebra


def verify():
    done = json.loads((LOCAL/'complete.json').read_text())
    assert done['signature'] == signature()
    assert all(sha256(LOCAL/n) == h for n,h in done['outputs'].items())
    d = pd.read_parquet(SOURCE).reset_index(drop=True); sets = feature_sets(d); cols = sets['full']
    x = d[cols].to_numpy(float); a = np.load(LOCAL/'shap.npz')
    donors = pd.read_parquet(LOCAL/'donor_attempts.parquet')
    cadence = pd.read_parquet(LOCAL/'cadence_scenarios.parquet')
    folds = json.loads((LOCAL/'folds.json').read_text())
    choices = json.loads((MODELS/'choices.json').read_text())
    assert len(donors) == 702*3 and len(cadence) == 702*5
    assert donors.groupby(['row_index','block']).size().eq(1).all()
    assert donors.groupby('row_index').guided.sum().eq(1).all()
    assert cadence.groupby(['row_index','requested_fraction']).size().eq(1).all()
    max_replay = 0.; candidates_checked = 0
    for meta, choice in zip(folds, choices):
        fold = choice['fold']; tr = np.array(choice['train_indices']); te = np.array(choice['test_indices'])
        fitpath = MODELS/f'fold{fold}.joblib'; assert sha256(fitpath) == meta['model_sha256']
        fit = joblib.load(fitpath); q = fit.predict_proba(x[te])[:,1]
        np.testing.assert_allclose(q, a['probabilities'][te], atol=1e-12, rtol=0)
        assert not set(d.iloc[tr].pair_id) & set(d.iloc[te].pair_id)
        z = fit[:-1].transform(x[tr]); zte = fit[:-1].transform(x[te])
        dist = cdist(z,z); dist[d.iloc[tr].pair_id.to_numpy()[:,None] == d.iloc[tr].pair_id.to_numpy()[None,:]] = np.inf
        cutoff = np.percentile(dist.min(1),95); np.testing.assert_allclose(cutoff, meta['support_cutoff'])
        saved = np.load(LOCAL/f'fold{fold}_donor_inputs.npz')
        np.testing.assert_array_equal(saved['train_indices'],tr); np.testing.assert_array_equal(saved['test_indices'],te)
        for name in ['background_indices','second_background_indices']:
            assert len(saved[name]) == len(set(saved[name])) == 100 and set(saved[name]) <= set(tr)
        for background_key,values_key in [('background_indices','values'),('second_background_indices','second_values')]:
            expected=fit.predict_proba(x[saved[background_key]])[:,1].mean()
            np.testing.assert_allclose(expected+a[values_key][te].sum(1),q,atol=1e-6,rtol=0)
        training_imp = meta['training_importance']
        top = max(sets['sensors'],key=lambda c:training_imp[c]); assert top == meta['top_training_sensor']
        rows = donors[donors.fold == fold].reset_index(drop=True); cf = saved['inputs']
        assert len(rows) == len(cf) == 3*len(te); check_algebra(cf,cols)
        reproduced = fit.predict_proba(cf)[:,1]
        max_replay = max(max_replay,float(np.max(np.abs(reproduced-rows.after))))
        np.testing.assert_allclose(reproduced,rows.after,atol=1e-12,rtol=0)
        for j,r in enumerate(rows.itertuples()):
            i = r.row_index; bi = [cols.index(c) for c in sets[r.block]]
            keep = [k for k in range(len(cols)) if k not in bi]
            np.testing.assert_equal(cf[j,keep], x[i,keep]); np.testing.assert_array_equal(np.isnan(cf[j]),np.isnan(x[i]))
            assert r.guided == (r.block == meta['guided_block'])
            eligible = ((abs(d.iloc[tr].demo_age.to_numpy()-d.iloc[i].demo_age)<=5)
                & (d.iloc[tr].demo_gender_male.to_numpy()==d.iloc[i].demo_gender_male)
                & np.all(np.isnan(x[tr][:,bi])==np.isnan(x[i,bi]),axis=1))
            pool = np.flatnonzero(eligible); zi = fit[:-1].transform(x[i:i+1])[0]
            pool = pool[np.argsort(np.sum((z[pool]-zi)**2,axis=1),kind='stable')[:30]]
            candidates = np.tile(x[i],(len(pool),1)); candidates[:,bi]=x[tr[pool]][:,bi]
            assert r.attempted == len(pool) and r.eligible_donors == eligible.sum()
            assert r.baseline_positive == (r.before>=.5)
            accepted = cdist(fit[:-1].transform(candidates),z).min(1)<=cutoff if len(pool) else np.zeros(0,bool)
            assert r.supported == accepted.sum(); candidates_checked += len(pool)
            pp = fit.predict_proba(candidates[accepted])[:,1] if accepted.any() else np.array([])
            crosses = (pp<.5) & (r.before>=.5); assert crosses.sum() == r.crossing_candidates
            expected_donor = -1
            if len(pp):
                delta_distance = np.linalg.norm(fit[:-1].transform(candidates[accepted])-zi,axis=1)
                pick = np.flatnonzero(crosses)[np.argmin(delta_distance[crosses])] if crosses.any() else np.argmin(pp)
                if pp[pick]<r.before: expected_donor=int(tr[pool[accepted][pick]])
            assert expected_donor == r.donor_index
            if r.donor_index<0:
                np.testing.assert_equal(cf[j],x[i]); assert r.after == r.before
            else:
                assert r.donor_index in tr and r.donor_index not in te
                np.testing.assert_equal(cf[j,bi],x[r.donor_index,bi])
                assert r.after<r.before
            assert r.crossed == (r.before>=.5 and r.after<.5)
            np.testing.assert_allclose(r.delta,r.after-r.before,atol=1e-14)
        wc = np.load(LOCAL/f'fold{fold}_cadence_inputs.npz')['inputs']
        wi = [cols.index(c) for c in sets['walking']]; keep = [k for k in range(len(cols)) if k not in wi]
        low,high = np.percentile(x[tr][:,wi],[1,99],axis=0)
        for level,zcf in zip(LEVELS,wc):
            check_algebra(zcf,cols); np.testing.assert_equal(zcf[:,keep],x[te][:,keep])
            np.testing.assert_array_equal(np.isnan(zcf),np.isnan(x[te]))
            for c in ['walk_duration_sec','walk_step_length']:
                np.testing.assert_equal(zcf[:,cols.index(c)],x[te,cols.index(c)])
            if level==0: np.testing.assert_equal(zcf,x[te])
            else:
                expected=np.maximum(1,np.rint(x[te,cols.index('walk_steps_total')]*(1+level)))
                np.testing.assert_array_equal(zcf[:,cols.index('walk_steps_total')],expected)
            rows = cadence[(cadence.fold==fold)&(cadence.requested_fraction==level)].set_index('row_index').loc[te]
            prob=fit.predict_proba(zcf)[:,1]
            np.testing.assert_allclose(prob,rows.after,atol=1e-12,rtol=0)
            np.testing.assert_allclose(prob-q,rows.delta,atol=1e-12,rtol=0)
            within=((zcf[:,wi]>=low)&(zcf[:,wi]<=high)).all(1)
            support=cdist(fit[:-1].transform(zcf),z).min(1)<=cutoff
            np.testing.assert_array_equal(within,rows.within_bounds); np.testing.assert_array_equal(support,rows.supported)
            np.testing.assert_array_equal((within&support) if level else np.ones(len(te),bool),rows.accepted)
    np.testing.assert_allclose(done['auc'],roc_auc_score(d.y,a['probabilities']),atol=1e-14)
    record=dict(status='passed',participants=702,matched_pairs=351,outer_models=5,
        donor_attempts=len(donors),donor_candidates_replayed=candidates_checked,
        cadence_scenarios=len(cadence),prediction_replay_max_error=max_replay,
        max_shap_additivity_error=max(max(r['shap_max_additivity_error'],r['second_max_additivity_error'],
            r['training_max_additivity_error']) for r in folds),
        clinical_causal_effect_identified=False,run_manifest_sha256=sha256(LOCAL/'complete.json'))
    record['verifier_source_sha256']=sha256(__file__)
    write_json(PUBLIC/'verification.json',record)
    return d,cols,a,donors,cadence,folds


def interval(values,pairs):
    values=np.asarray(values,float); pairs=np.asarray(pairs)
    if not len(values): return np.nan,np.nan,np.nan
    _,inverse=np.unique(pairs,return_inverse=True); n=inverse.max()+1
    sums=np.bincount(inverse,weights=values); counts=np.bincount(inverse)
    rng=np.random.default_rng(20261119); draws=rng.integers(n,size=(2000,n))
    b=sums[draws].sum(1)/counts[draws].sum(1)
    return float(values.mean()),*map(float,np.percentile(b,[2.5,97.5]))


def report():
    d,cols,a,donors,cadence,folds=verify()
    importance=pd.DataFrame(dict(feature=cols,mean_absolute_shap=np.abs(a['values']).mean(0),
        second_background_mean_absolute_shap=np.abs(a['second_values']).mean(0)))
    importance=importance.sort_values('mean_absolute_shap',ascending=False)
    importance.to_csv(PUBLIC/'heldout_shap_importance.csv',index=False)
    pd.DataFrame([dict(fold=f['fold'],model=f['model'],top_training_sensor=f['top_training_sensor'],
        guided_block=f['guided_block'],support_cutoff=f['support_cutoff']) for f in folds]).to_csv(PUBLIC/'fold_selection.csv',index=False)
    rank_agreement=float(spearmanr(np.abs(a['values']).mean(0),np.abs(a['second_values']).mean(0)).statistic)
    arms=pd.concat([donors.assign(arm=donors.block),donors[donors.guided].assign(arm='shap_guided')])
    summaries=[]
    for arm,base in arms.groupby('arm'):
        for group in ['all','PD','control']:
            mask=np.ones(len(base),bool) if group=='all' else d.y.to_numpy()[base.row_index]==(1 if group=='PD' else 0)
            g=base.loc[mask]; ids=g.row_index.to_numpy(); pairs=d.iloc[ids].pair_id
            delta,lo,hi=interval(g.delta,pairs)
            pos=g[g.baseline_positive]; rate,rlo,rhi=interval(pos.crossed.astype(float),d.iloc[pos.row_index].pair_id)
            summaries.append(dict(arm=arm,group=group,n=len(g),baseline_positive=len(pos),
                supported_attempts=int((g.supported>0).sum()),changed_inputs=int((g.donor_index>=0).sum()),
                crossed=int(pos.crossed.sum()),crossing_fraction=rate,crossing_ci_low=rlo,crossing_ci_high=rhi,
                mean_score_change=delta,change_ci_low=lo,change_ci_high=hi,
                mean_attempted_candidates=float(g.attempted.mean())))
    summary=pd.DataFrame(summaries);summary.to_csv(PUBLIC/'block_counterfactual_summary.csv',index=False)
    accepted=cadence[cadence.requested_fraction!=0].groupby('row_index').accepted.all()
    common=accepted.index[accepted].to_numpy()
    scenarios=[]
    for mode in ['scenario_specific_support','common_support']:
        for group in ['all','PD','control']:
            for level,g0 in cadence.groupby('requested_fraction'):
                g=g0[g0.accepted] if mode=='scenario_specific_support' else g0[g0.row_index.isin(common)]
                if group!='all':g=g[d.y.to_numpy()[g.row_index]==(1 if group=='PD' else 0)]
                value,lo,hi=interval(g.delta,d.iloc[g.row_index].pair_id)
                eligible=702 if group=='all' else 351
                scenarios.append(dict(support_set=mode,group=group,requested_fraction=level,eligible=eligible,
                    accepted=len(g),excluded=eligible-len(g),mean_score_change=value,ci_low=lo,ci_high=hi,
                    mean_achieved_cadence_change=float(g.actual_cadence_change.mean()) if len(g) else np.nan))
    scenario=pd.DataFrame(scenarios);scenario.to_csv(PUBLIC/'cadence_summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    top=importance.head(12).iloc[::-1]
    axs[0].barh(top.feature.str.replace('_',' '),top.mean_absolute_shap,color='#246778')
    axs[0].set_xlabel('Mean absolute held-out SHAP (score units)');axs[0].set_title('Classifier attribution; not causal importance')
    for group,color in [('PD','#bd4929'),('control','#246778')]:
        g=scenario[(scenario.support_set=='common_support')&(scenario.group==group)]
        axs[1].errorbar(g.requested_fraction*100,g.mean_score_change*100,
            yerr=np.array([g.mean_score_change-g.ci_low,g.ci_high-g.mean_score_change])*100,
            label=f'{group} (n={int(g.accepted.iloc[0])})',color=color,marker='o',capsize=3)
    axs[1].axhline(0,color='gray',lw=.8);axs[1].set_xlabel('Requested cadence change (%)')
    axs[1].set_ylabel('Mean classifier score change (percentage points)')
    axs[1].set_title('Coherent walking-input scenarios; common support');axs[1].legend()
    fig.text(.5,.015,'Exploratory model-input analysis. Score changes do not estimate disease improvement or treatment effects.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1])
    for ext in ['png','pdf']:fig.savefig(PUBLIC/f'explanation_summary.{ext}',dpi=200)
    plt.close(fig)
    result=dict(auc=float(roc_auc_score(d.y,a['probabilities'])),background_importance_spearman=rank_agreement,
        common_support_n=len(common),common_support_PD=int(d.iloc[common].y.sum()),
        shap_top_features=importance.head(10).feature.tolist())
    write_json(PUBLIC/'summary.json',result)
    print(json.dumps(result,indent=2));print(summary.to_string(index=False));print(scenario[scenario.support_set=='common_support'].to_string(index=False))


if __name__=='__main__':
    report()
