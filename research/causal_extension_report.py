"""Independent replay, aggregate uncertainty, and multi-model interpretation."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist,pdist
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
from research.common import ROOT,PRIVATE,sha256,write_json
from research.core import feature_sets
from research.explainability import SOURCE,check_algebra
from research.explainability_report import interval
from research.causal_association import PUBLIC,LOCAL,MARKERS,contrast,bh
from research.causal_model_panel import METHODS,POLICIES,BLOCKS,LEVELS,SCENARIOS,panel,structural_inputs,diverse_choices


def verify():
    complete=json.loads((LOCAL/'panel_complete.json').read_text())
    assert complete['signature']==dict(code=sha256(ROOT/'research/causal_model_panel.py'),
        protocol=sha256(PUBLIC/'PROTOCOL.md'),cohort=sha256(SOURCE))
    assert all(sha256(LOCAL/n)==h for n,h in complete['outputs'].items())
    av=json.loads((PUBLIC/'association_verification.json').read_text())
    assert sha256(ROOT/'research/causal_association.py')==av['source_sha256']
    assert sha256(PUBLIC/'association_summary.csv')==av['result_sha256']
    assert sha256(PRIVATE/'cohort.parquet')==av['cohort_sha256']
    assert sha256(PUBLIC/'PROTOCOL.md')==av['protocol_sha256']
    association=pd.read_csv(PUBLIC/'association_summary.csv')
    np.testing.assert_allclose(bh(association.label_swap_p),association.label_swap_BH_q)
    d=pd.read_parquet(SOURCE).reset_index(drop=True);cols=feature_sets(d)['full'];x=d[cols].to_numpy(float)
    attempts=pd.read_parquet(LOCAL/'panel_attempts.parquet');scenarios=pd.read_parquet(LOCAL/'panel_scenarios.parquet')
    assert len(attempts)==702*3*8*2 and len(scenarios)==702*3*5*8
    assert attempts.groupby(['row_index','block','method','policy']).size().eq(1).all()
    assert scenarios.groupby(['row_index','scenario','method','level']).size().eq(1).all()
    metadata=json.loads((LOCAL/'panel_models.json').read_text());raw=np.load(LOCAL/'tapping_transform_valid.npy')
    for path,h in json.loads((LOCAL/'tapping_raw_hashes.json').read_text()).items():assert sha256(ROOT/path)==h
    candidates=0;returns=0;maxerror=0.;baselines=[]
    for info in metadata:
        fold=info['fold'];saved=np.load(LOCAL/f'panel_fold{fold}.npz');tr,te=saved['tr'],saved['te']
        models,cuts,repeat=panel(d,cols,fold,tr,te)
        assert repeat['expanded_winner']==info['expanded_winner'] and cuts==info['thresholds']
        assert repeat['model_hashes']==info['model_hashes']
        fit=models['earlier_selector'];zt=fit[:-1].transform(x[tr]);zq=fit[:-1].transform(x[te]);cf=saved['inputs']
        check_algebra(cf,cols);zc=fit[:-1].transform(cf);qix=saved['queries'];donors=saved['donors']
        assert set(donors)<=set(tr) and not set(donors)&set(te)
        for bi,block in enumerate(BLOCKS):
            rows=saved['blocks']==bi;selected=[cols.index(c) for c in feature_sets(d)[block]]
            other=[j for j in range(len(cols)) if j not in selected]
            np.testing.assert_equal(cf[rows][:,selected],x[donors[rows]][:,selected])
            np.testing.assert_equal(cf[rows][:,other],x[qix[rows]][:,other])
            assert np.array_equal(np.isnan(cf[rows]),np.isnan(x[qix[rows]]))
        assert np.all(abs(d.iloc[donors].demo_age.to_numpy()-d.iloc[qix].demo_age.to_numpy())<=5)
        np.testing.assert_array_equal(d.iloc[donors].demo_gender_male,d.iloc[qix].demo_gender_male)
        dist=cdist(zt,zt);pair=d.iloc[tr].pair_id.to_numpy();dist[pair[:,None]==pair[None,:]]=np.inf
        cutoff=float(np.percentile(dist.min(1),95));np.testing.assert_allclose(cutoff,info['cutoff'])
        constant=np.ptp(fit[0].transform(x[tr]),axis=0)<1e-12
        immutable=(abs(fit[0].transform(cf)[:,constant]-fit[0].transform(x[qix])[:,constant])<=1e-10).all(1)
        support=(cdist(zc,zt).min(1)<=cutoff)&immutable
        np.testing.assert_array_equal(support,saved['support'])
        lof=LocalOutlierFactor(n_neighbors=20,novelty=True).fit(zt)
        np.testing.assert_allclose(lof.score_samples(zc),saved['lof'],atol=1e-12,rtol=0)
        p=np.column_stack([models[m].predict_proba(cf)[:,1] for m in METHODS])
        b=np.column_stack([models[m].predict_proba(x[te])[:,1] for m in METHODS])
        maxerror=max(maxerror,float(abs(p-saved['probabilities']).max()),float(abs(b-saved['baseline']).max()))
        np.testing.assert_allclose(p,saved['probabilities'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(b,saved['baseline'],atol=1e-12,rtol=0)
        for mi,m in enumerate(METHODS):
            baselines.extend([dict(row_index=int(i),fold=fold,method=m,probability=float(value)) for i,value in zip(te,b[:,mi])])
        positions={int(i):j for j,i in enumerate(te)}
        for (i,block),g in attempts[attempts.fold==fold].groupby(['row_index','block']):
            index=np.flatnonzero((qix==i)&(saved['blocks']==BLOCKS.index(block)))
            bi=[cols.index(c) for c in feature_sets(d)[block]];pos=positions[i]
            eligible=((abs(d.iloc[tr].demo_age.to_numpy()-d.iloc[i].demo_age)<=5)
                &(d.iloc[tr].demo_gender_male.to_numpy()==d.iloc[i].demo_gender_male)
                &np.all(np.isnan(x[tr][:,bi])==np.isnan(x[i,bi]),axis=1))
            pool=np.flatnonzero(eligible);pool=pool[np.argsort(((zt[pool]-zq[pos])**2).sum(1),kind='stable')[:30]]
            np.testing.assert_array_equal(donors[index],tr[pool]);candidates+=len(index)
            for r in g.itertuples():
                mi=METHODS.index(r.method);tau=cuts[r.method][r.policy]
                assert r.threshold==tau and r.baseline_positive==(b[pos,mi]>=tau)
                assert r.attempted==len(index) and r.supported==support[index].sum()
                selected=np.array([getattr(r,f'candidate_{j}') for j in range(3) if getattr(r,f'candidate_{j}')>=0],int)
                expected=diverse_choices(zc[index],zq[pos],p[index,mi],tau,support[index]) if r.baseline_positive else []
                np.testing.assert_array_equal(selected,index[expected]);assert len(selected)==r.returned;returns+=r.returned
                if len(selected):
                    assert (p[selected,mi]<tau).all() and support[selected].all() and r.validity==1
                    assert len(np.unique(zc[selected],axis=0))==len(selected)
                    feasible=index[support[index]&(p[index,mi]<tau)]
                    np.testing.assert_allclose(np.mean(abs(zc[selected[0]]-zq[pos])),np.mean(abs(zc[feasible]-zq[pos]),axis=1).min())
                    np.testing.assert_allclose(np.mean(abs(zc[selected]-zq[pos])),r.proximity)
                    np.testing.assert_allclose(np.nanmean(abs(cf[selected]-x[i])/saved['madscale']),r.mad_proximity)
                    changed=~np.isclose(cf[selected],x[i],atol=1e-8,rtol=0,equal_nan=True)
                    np.testing.assert_allclose(1-changed.mean(),r.sparsity)
                    if len(selected)>1:np.testing.assert_allclose(pdist(zc[selected],metric='cityblock').mean()/len(cols),r.diversity)
                    else:assert pd.isna(r.diversity)
                    np.testing.assert_allclose(saved['lof'][selected].mean(),r.lof)
                    np.testing.assert_allclose(p[selected[0],mi]-b[pos,mi],r.first_delta)
        for family in SCENARIOS:
            block='tapping' if family=='tapping_time' else 'walking';bi=[cols.index(c) for c in feature_sets(d)[block]]
            other=[i for i in range(len(cols)) if i not in bi];low,high=np.nanpercentile(x[tr][:,bi],[1,99],axis=0)
            for li,level in enumerate(LEVELS):
                transformed=structural_inputs(x[te],cols,family,level);check_algebra(transformed,cols)
                np.testing.assert_equal(transformed[:,other],x[te][:,other])
                bounds=((transformed[:,bi]>=low)&(transformed[:,bi]<=high)).all(1)
                supported=cdist(fit[:-1].transform(transformed),zt).min(1)<=cutoff
                rawok=raw[li,te] if family=='tapping_time' else np.ones(len(te),bool)
                accept=(bounds&supported&rawok) if level else np.ones(len(te),bool)
                for mi,m in enumerate(METHODS):
                    g=scenarios[(scenarios.fold==fold)&(scenarios.scenario==family)&(scenarios.level==level)&(scenarios.method==m)].set_index('row_index').loc[te]
                    prediction=models[m].predict_proba(transformed)[:,1]
                    np.testing.assert_allclose(prediction,g.after,atol=1e-12,rtol=0)
                    np.testing.assert_allclose(prediction-b[:,mi],g.delta,atol=1e-12,rtol=0)
                    np.testing.assert_array_equal(accept,g.accepted)
                    np.testing.assert_array_equal(rawok,g.raw_valid);np.testing.assert_array_equal(bounds,g.bounds)
                    np.testing.assert_array_equal(supported,g.support)
        print(f'Independent panel replay {fold+1}/5 passed',flush=True)
    baseline=pd.DataFrame(baselines)
    write_json(PUBLIC/'panel_verification.json',dict(status='passed',participants=702,models_replayed=35,
        methods=8,unique_donor_candidates=candidates,attempt_rows=len(attempts),returned_examples_replayed=returns,
        structural_prediction_rows=len(scenarios),prediction_replay_max_error=maxerror,
        original_tapping_files_verified=702,clinical_causal_effect_identified=False,
        run_manifest_sha256=sha256(LOCAL/'panel_complete.json'),verifier_sha256=sha256(__file__)))
    return d,cols,attempts,scenarios,baseline,metadata


def report():
    d,cols,attempts,scenarios,baseline,metadata=verify()
    metrics=[]
    for method,g in baseline.groupby('method'):
        truth=d.y.to_numpy()[g.row_index];metrics.append(dict(method=method,auc=roc_auc_score(truth,g.probability)))
    pd.DataFrame(metrics).to_csv(PUBLIC/'model_panel_auc.csv',index=False)
    pd.DataFrame([dict(fold=r['fold'],expanded_selected_family=r['expanded_winner']) for r in metadata]).to_csv(PUBLIC/'model_selection.csv',index=False)
    records=[]
    for keys,g0 in attempts.groupby(['method','policy','block']):
        for group in ['all','PD','control']:
            g=g0 if group=='all' else g0[d.y.to_numpy()[g0.row_index]==(1 if group=='PD' else 0)]
            positive=g[g.baseline_positive];success=positive[positive.returned>0]
            value,lo,hi=interval((positive.returned>0).astype(float),d.iloc[positive.row_index].pair_id)
            rec=dict(zip(['method','policy','block'],keys),group=group,total_queries=len(g),positive_queries=len(positive),
                successful_queries=len(success),failed_positive_queries=len(positive)-len(success),
                yield_fraction=value,yield_ci_low=lo,yield_ci_high=hi,
                returned_examples=int(success.returned.sum()),mean_examples_per_positive=float(positive.returned.mean()))
            for column in ['validity','proximity','mad_proximity','sparsity','diversity','lof','first_delta']:
                rec['successful_query_mean_'+column]=float(success[column].mean())
            rec['queries_with_diversity']=int(success.diversity.notna().sum());records.append(rec)
    pd.DataFrame(records).to_csv(PUBLIC/'counterfactual_quality.csv',index=False)
    # Transfers of the PRIMARY selector's first counterfactual; identical inputs.
    transfers=[];feature_rows=[]
    for info in metadata:
        fold=info['fold'];saved=np.load(LOCAL/f'panel_fold{fold}.npz');te=saved['te'];pos={int(i):j for j,i in enumerate(te)}
        g=attempts[(attempts.fold==fold)&(attempts.method=='expanded_selector')&(attempts.returned>0)]
        for r in g.itertuples():
            candidate=r.first_candidate;i=r.row_index;j=pos[i]
            for mi,method in enumerate(METHODS):
                before=saved['baseline'][j,mi];after=saved['probabilities'][candidate,mi];tau=info['thresholds'][method][r.policy]
                transfers.append(dict(row_index=i,method=method,policy=r.policy,block=r.block,
                    positive=bool(before>=tau),crossed=bool(before>=tau and after<tau),delta=float(after-before)))
            for col in feature_sets(d)[r.block]:
                value=saved['inputs'][candidate,cols.index(col)]-float(d.iloc[i][col])
                feature_rows.append(dict(row_index=i,policy=r.policy,block=r.block,feature=col,delta=float(value)))
    transfer=pd.DataFrame(transfers);summary=[]
    for keys,g in transfer.groupby(['method','policy','block']):
        both=g[g.positive];rate,lo,hi=interval(both.crossed.astype(float),d.iloc[both.row_index].pair_id)
        summary.append(dict(zip(['method','policy','block'],keys),primary_successes=len(g),both_positive=len(both),
            crossed_under_comparator=int(both.crossed.sum()),transfer_fraction=rate,ci_low=lo,ci_high=hi,
            mean_score_change=float(g.delta.mean()),score_decreased_fraction=float((g.delta<0).mean())))
    pd.DataFrame(summary).to_csv(PUBLIC/'cross_model_transfer.csv',index=False)
    f=pd.DataFrame(feature_rows);f.groupby(['policy','block','feature']).delta.agg(['count','mean','median','std','min','max']).reset_index().to_csv(PUBLIC/'primary_counterfactual_feature_changes.csv',index=False)
    # Common support is determined by inputs and training support only, same for every model.
    records=[]
    for family in SCENARIOS:
        ref=scenarios[(scenarios.scenario==family)&(scenarios.method=='expanded_selector')&(scenarios.level!=0)]
        common=ref.groupby('row_index').accepted.all();common_ids=common.index[common]
        for (method,level),g0 in scenarios[scenarios.scenario==family].groupby(['method','level']):
            for support_set in ['scenario_support','common_support']:
                g1=g0[g0.accepted] if support_set=='scenario_support' else g0[g0.row_index.isin(common_ids)]
                for group in ['all','PD','control']:
                    g=g1 if group=='all' else g1[d.y.to_numpy()[g1.row_index]==(1 if group=='PD' else 0)]
                    value,lo,hi=interval(g.delta,d.iloc[g.row_index].pair_id)
                    records.append(dict(method=method,scenario=family,level=level,support_set=support_set,group=group,
                        n=len(g),excluded=(702 if group=='all' else 351)-len(g),mean_score_change=value,ci_low=lo,ci_high=hi))
    pd.DataFrame(records).to_csv(PUBLIC/'structural_scenario_results.csv',index=False)
    # Audit exclusions without multiplying counts by the eight repeated models.
    q=scenarios[scenarios.method=='expanded_selector']
    q.groupby(['scenario','level'])[['accepted','raw_valid','bounds','support']].sum().reset_index().to_csv(PUBLIC/'scenario_support_counts.csv',index=False)
    print('All aggregate panel results saved.',flush=True)


if __name__=='__main__':
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    report()
