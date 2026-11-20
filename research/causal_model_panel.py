"""Training-selected model panel, constrained multi-example counterfactuals.

This module estimates model responses, never clinical treatment effects.
"""
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import cdist,pdist
from sklearn.neighbors import LocalOutlierFactor
from research.common import ROOT,OUT,PRIVATE,sha256,write_json
from research.core import feature_sets,select_thresholds
from research.explainability import SOURCE,MODELS,check_algebra,walking_scenario
from research.causal_association import PUBLIC,LOCAL

FAMILIES=['lr','rf','xgboost','lightgbm','catboost','mlp']
METHODS=FAMILIES+['expanded_selector','earlier_selector']
POLICIES=['default','sensitivity90']
BLOCKS=['tapping','walking','voice']
LEVELS=[-.2,-.1,0.,.1,.2]
SCENARIOS=['tapping_time','walking_cadence','walking_step_length']


def panel(d,cols,fold,tr,te):
    models={};thresholds={};scores={};hashes={}
    for family in FAMILIES:
        path=PRIVATE/'benchmark_extension/benchmark'/f'n702_full_raw_{family}'/f'fold{fold}'
        done=json.loads((path/'complete.json').read_text())
        assert all(sha256(path/n)==h for n,h in done['artifacts'].items())
        details=json.loads((path/'details.json').read_text());assert details['features']==cols
        model=joblib.load(path/'model.joblib');models[family]=model
        inner=pd.read_parquet(path/'training_predictions.parquet').set_index('participant_id').loc[d.iloc[tr].participant_id]
        assert set(inner.index)==set(d.iloc[tr].participant_id)
        assert not set(inner.pair_id)&set(d.iloc[te].pair_id)
        cuts=select_thresholds(inner.y,inner.probability)
        assert all(cuts[p]==details['thresholds']['tuned'][p] for p in POLICIES)
        thresholds[family]={p:cuts[p] for p in POLICIES}
        c=details['details']['chosen'];scores[family]=details['details']['trials'][c]['mean_auc']
        reference=pd.read_parquet(path/'predictions.parquet')
        reference=reference[reference.policy=='default'].set_index('participant_id').loc[d.iloc[te].participant_id]
        np.testing.assert_allclose(model.predict_proba(d.iloc[te][cols].to_numpy(float))[:,1],reference.probability,atol=1e-12,rtol=0)
        hashes[family]=sha256(path/'model.joblib')
    winner=sorted(FAMILIES,key=lambda f:(-scores[f],f))[0]
    models['expanded_selector']=models[winner];thresholds['expanded_selector']=thresholds[winner]
    choice=json.loads((MODELS/'choices.json').read_text())[fold]
    assert np.array_equal(choice['train_indices'],tr) and np.array_equal(choice['test_indices'],te)
    path=MODELS/f'fold{fold}.joblib';assert sha256(path)==choice['model_sha256']
    models['earlier_selector']=joblib.load(path);thresholds['earlier_selector']={p:choice['thresholds'][p] for p in POLICIES}
    hashes['earlier_selector']=sha256(path)
    # Reference to the prior explanation prediction replay.
    prev=pd.read_parquet(MODELS/'predictions.parquet')
    prev=prev[prev.policy=='default'].set_index('participant_id').loc[d.iloc[te].participant_id]
    np.testing.assert_allclose(models['earlier_selector'].predict_proba(d.iloc[te][cols].to_numpy(float))[:,1],prev.probability,atol=1e-12,rtol=0)
    for model in models.values():
        np.testing.assert_equal(model[0].statistics_,models['earlier_selector'][0].statistics_)
        np.testing.assert_allclose(model[1].scale_,models['earlier_selector'][1].scale_,atol=1e-12)
    return models,thresholds,dict(fold=fold,expanded_winner=winner,inner_auc=scores,model_hashes=hashes)


def diverse_choices(z,query,probabilities,threshold,support,k=3):
    eligible=np.flatnonzero(support & (probabilities<threshold))
    if not len(eligible):return []
    # Duplicate generated feature vectors are not diverse alternatives.
    _,unique=np.unique(z[eligible],axis=0,return_index=True)
    eligible=eligible[np.sort(unique)]
    distance=np.mean(np.abs(z[eligible]-query),axis=1)
    first=int(np.argmin(distance));selected=[int(eligible[first])]
    while len(selected)<min(k,len(eligible)):
        remaining=np.array([i for i in eligible if i not in selected])
        sep=cdist(z[remaining],z[selected],metric='cityblock').min(1)/z.shape[1]
        rank=sep/(1+np.mean(np.abs(z[remaining]-query),axis=1))
        selected.append(int(remaining[np.argmax(rank)]))
    return selected


def structural_inputs(x,cols,family,level):
    z=x.copy();index={c:i for i,c in enumerate(cols)}
    if not level:return z
    if family=='walking_cadence':return walking_scenario(x,cols,level)
    if family=='walking_step_length':
        for c in ['walk_step_length','walk_distance_total','walk_speed_mean']:z[:,index[c]]*=1+level
    elif family=='tapping_time':
        for c in ['tap_duration_sec']+['tap_flight_'+s for s in ['mean','median','std','iqr','min','max']]:z[:,index[c]]/=1+level
        z[:,index['tap_rate']]*=1+level
    else:raise ValueError(family)
    return z


def raw_tapping_checks(d,cols):
    from research.data import tapping,DATA
    flags=np.ones((len(LEVELS),len(d)),bool);hashes={}
    raw_manifest=pd.read_csv(PRIVATE/'raw_file_manifest.csv')
    expected=raw_manifest[raw_manifest.modality=='tapping'].set_index('participant_id')
    x=d[cols].to_numpy(float);tap=[c for c in cols if c.startswith('tap_')]
    for i,r in enumerate(d.itertuples()):
        path=DATA/'raw/tapping'/f'{r.tapping_handle}.json'
        assert sha256(path)==expected.loc[r.participant_id,'sha256'];hashes[str(path.relative_to(ROOT))]=sha256(path)
        samples=json.loads(path.read_text());baseline=tapping(samples,r.tap_duration_sec)
        np.testing.assert_allclose([baseline[c] for c in tap],d.loc[i,tap].to_numpy(float),atol=1e-10)
        times=np.array([s['TapTimeStamp'] for s in samples if s.get('TappedButtonId') in ['TappedButtonLeft','TappedButtonRight']],float)
        intervals=np.diff(times);keep=(intervals>=.01)&(intervals<=5)
        for li,level in enumerate(LEVELS):
            if level==0:continue
            scale=1/(1+level); duration=r.tap_duration_sec*scale
            kept=(intervals*scale>=.01)&(intervals*scale<=5)
            valid=5<=duration<=60 and np.array_equal(keep,kept)
            if valid:
                changed=[dict(s,TapTimeStamp=float(s['TapTimeStamp'])*scale) for s in samples]
                try:actual=tapping(changed,duration)
                except ValueError:valid=False
                if valid:
                    proposed=structural_inputs(x[i:i+1],cols,'tapping_time',level)[0]
                    valid=bool(np.allclose([actual[c] for c in tap],[proposed[cols.index(c)] for c in tap],rtol=1e-9,atol=1e-10))
            flags[li,i]=valid
    write_json(LOCAL/'tapping_raw_hashes.json',hashes)
    np.save(LOCAL/'tapping_transform_valid.npy',flags)
    return flags


def run():
    LOCAL.mkdir(exist_ok=True)
    signature=dict(code=sha256(__file__),protocol=sha256(PUBLIC/'PROTOCOL.md'),cohort=sha256(SOURCE))
    if (LOCAL/'panel_complete.json').exists():
        old=json.loads((LOCAL/'panel_complete.json').read_text());assert old['signature']==signature
        assert all(sha256(LOCAL/n)==h for n,h in old['outputs'].items())
        print('Panel hashes verified; existing run reused.');return
    write_json(LOCAL/'panel_start.json',signature)
    d=pd.read_parquet(SOURCE).reset_index(drop=True);cols=feature_sets(d)['full'];x=d[cols].to_numpy(float)
    assert len(d)==702 and d.participant_id.is_unique
    tapvalid=raw_tapping_checks(d,cols)
    meta=[];attempts=[];scenario_rows=[]
    choices=json.loads((MODELS/'choices.json').read_text())
    for choice in choices:
        fold=choice['fold'];tr=np.array(choice['train_indices']);te=np.array(choice['test_indices'])
        assert not set(d.iloc[tr].pair_id)&set(d.iloc[te].pair_id)
        models,cuts,info=panel(d,cols,fold,tr,te)
        fit=models['earlier_selector'];ztr=fit[:-1].transform(x[tr]);zte=fit[:-1].transform(x[te])
        zdist=cdist(ztr,ztr);groups=d.iloc[tr].pair_id.to_numpy();zdist[groups[:,None]==groups[None,:]]=np.inf
        cutoff=float(np.percentile(zdist.min(1),95))
        med=np.nanmedian(x[tr],axis=0);mad=np.nanmedian(np.abs(x[tr]-med),axis=0)
        iqr=np.nanpercentile(x[tr],75,axis=0)-np.nanpercentile(x[tr],25,axis=0)
        madscale=np.maximum.reduce([mad,.1*iqr,np.full(len(cols),1e-8)])
        constant=np.ptp(fit[0].transform(x[tr]),axis=0)<1e-12
        lof=LocalOutlierFactor(n_neighbors=20,novelty=True).fit(ztr)
        all_cf=[];donors=[];queries=[];blocks=[];ranges=[]
        for block in BLOCKS:
            bi=[cols.index(c) for c in feature_sets(d)[block]]
            for pos,i in enumerate(te):
                eligible=((abs(d.iloc[tr].demo_age.to_numpy()-d.iloc[i].demo_age)<=5)
                    &(d.iloc[tr].demo_gender_male.to_numpy()==d.iloc[i].demo_gender_male)
                    &np.all(np.isnan(x[tr][:,bi])==np.isnan(x[i,bi]),axis=1))
                pool=np.flatnonzero(eligible);pool=pool[np.argsort(((ztr[pool]-zte[pos])**2).sum(1),kind='stable')[:30]]
                cf=np.tile(x[i],(len(pool),1));cf[:,bi]=x[tr[pool]][:,bi]
                start=len(donors);all_cf.extend(cf);donors.extend(tr[pool]);queries.extend([i]*len(pool));blocks.extend([BLOCKS.index(block)]*len(pool))
                ranges.append((int(i),pos,block,start,len(donors)))
        cf=np.asarray(all_cf);check_algebra(cf,cols);zcf=fit[:-1].transform(cf)
        qix=np.array(queries);immutable=(np.abs(fit[0].transform(cf)[:,constant]-fit[0].transform(x[qix])[:,constant])<=1e-10).all(1)
        support=(cdist(zcf,ztr).min(1)<=cutoff)&immutable
        lofscore=lof.score_samples(zcf)
        probs=np.column_stack([models[m].predict_proba(cf)[:,1] for m in METHODS])
        base=np.column_stack([models[m].predict_proba(x[te])[:,1] for m in METHODS])
        for i,pos,block,start,end in ranges:
            for mi,method in enumerate(METHODS):
                for policy in POLICIES:
                    tau=cuts[method][policy];positive=bool(base[pos,mi]>=tau)
                    selection=diverse_choices(zcf[start:end],zte[pos],probs[start:end,mi],tau,support[start:end]) if positive else []
                    selected=np.array([start+s for s in selection],int)
                    row=dict(fold=fold,row_index=i,block=block,method=method,policy=policy,threshold=tau,
                        before=float(base[pos,mi]),baseline_positive=positive,attempted=end-start,
                        supported=int(support[start:end].sum()),returned=len(selected),
                        first_candidate=int(selected[0]) if len(selected) else -1)
                    for k in range(3):row[f'candidate_{k}']=int(selected[k]) if k<len(selected) else -1
                    if len(selected):
                        difference=cf[selected]-x[i];changed=~np.isclose(cf[selected],x[i],atol=1e-8,rtol=0,equal_nan=True)
                        row.update(validity=float(np.mean(probs[selected,mi]<tau)),
                            proximity=float(np.mean(np.abs(zcf[selected]-zte[pos]))),
                            mad_proximity=float(np.nanmean(np.abs(difference)/madscale)),
                            sparsity=float(1-changed.mean()),
                            diversity=float(pdist(zcf[selected],metric='cityblock').mean()/len(cols)) if len(selected)>1 else np.nan,
                            lof=float(lofscore[selected].mean()),
                            first_delta=float(probs[selected[0],mi]-base[pos,mi]))
                    attempts.append(row)
        np.savez_compressed(LOCAL/f'panel_fold{fold}.npz',inputs=cf,donors=donors,queries=qix,blocks=blocks,
            support=support,lof=lofscore,probabilities=probs,baseline=base,tr=tr,te=te,madscale=madscale)
        # Structural propagation is evaluated without optimizing direction or outcomes.
        for family in SCENARIOS:
            block='tapping' if family=='tapping_time' else 'walking';bi=[cols.index(c) for c in feature_sets(d)[block]]
            low,high=np.nanpercentile(x[tr][:,bi],[1,99],axis=0)
            for li,level in enumerate(LEVELS):
                transformed=structural_inputs(x[te],cols,family,level);check_algebra(transformed,cols)
                rawok=tapvalid[li,te] if family=='tapping_time' else np.ones(len(te),bool)
                bounds=((transformed[:,bi]>=low)&(transformed[:,bi]<=high)).all(1)
                supported=cdist(fit[:-1].transform(transformed),ztr).min(1)<=cutoff
                accepted=(rawok&bounds&supported) if level else np.ones(len(te),bool)
                predicted=np.column_stack([models[m].predict_proba(transformed)[:,1] for m in METHODS])
                if not level:np.testing.assert_allclose(predicted,base,atol=1e-12,rtol=0)
                for pos,i in enumerate(te):
                    for mi,method in enumerate(METHODS):
                        scenario_rows.append(dict(fold=fold,row_index=int(i),scenario=family,level=level,method=method,
                            accepted=bool(accepted[pos]),raw_valid=bool(rawok[pos]),bounds=bool(bounds[pos]),support=bool(supported[pos]),
                            before=float(base[pos,mi]),after=float(predicted[pos,mi]),delta=float(predicted[pos,mi]-base[pos,mi])))
        info.update(thresholds=cuts,cutoff=cutoff);meta.append(info)
        print(f'Panel fold {fold+1}/5: {info["expanded_winner"]}; {len(cf)} candidates; all structural scenarios evaluated',flush=True)
    pd.DataFrame(attempts).to_parquet(LOCAL/'panel_attempts.parquet',index=False)
    pd.DataFrame(scenario_rows).to_parquet(LOCAL/'panel_scenarios.parquet',index=False)
    write_json(LOCAL/'panel_models.json',meta)
    names=['panel_attempts.parquet','panel_scenarios.parquet','panel_models.json','tapping_raw_hashes.json','tapping_transform_valid.npy']+[f'panel_fold{i}.npz' for i in range(5)]
    write_json(LOCAL/'panel_complete.json',dict(signature=signature,outputs={n:sha256(LOCAL/n) for n in names},
        methods=METHODS,attempt_rows=len(attempts),scenario_rows=len(scenario_rows),clinical_causal_effect_identified=False))


if __name__=='__main__':
    warnings.filterwarnings('ignore',message='X does not have valid feature names')
    run()
