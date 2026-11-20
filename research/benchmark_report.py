"""Independent artifact reconciliation and aggregate-only extension reporting."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from research.benchmark_extension import (PUBLIC, LOCAL, ROOT, PRIVATE, POLICIES, MODES,
    TRIALS, load_data, tasks, thresholds)
from research.common import sha256, write_json
from research.core import metrics


def expanded_metrics(frame):
    y,p,z=frame.y.to_numpy(),frame.probability.to_numpy(),frame.prediction.to_numpy()
    m=metrics(y,p,z)
    m.update(accuracy=float(accuracy_score(y,z)),f1=float(f1_score(y,z)),
             mcc=float(matthews_corrcoef(y,z)) if np.unique(z).size>1 else None)
    return {k: None if isinstance(v,float) and not np.isfinite(v) else v for k,v in m.items()}


def reconcile():
    collected=[]; selections=[];warnings=Counter();bounds=0;count=0
    for task in tasks('benchmark')+tasks('calibration'):
        kind,cohort,config,arm,family,fold=task
        folder=LOCAL/kind/f'{cohort}_{config}_{arm}_{family}'/f'fold{fold}'
        complete=json.loads((folder/'complete.json').read_text())
        assert complete['protocol_sha256']==sha256(PUBLIC/'protocol.json')
        for name,digest in complete['artifacts'].items(): assert sha256(folder/name)==digest
        d=json.loads((folder/'details.json').read_text())
        p=pd.read_parquet(folder/'predictions.parquet')
        ip=pd.read_parquet(folder/'training_predictions.parquet')
        df,X,columns,folds=load_data(cohort,config,arm)
        truth=df.set_index('participant_id')
        tr,te=df.loc[folds!=fold],df.loc[folds==fold]
        assert d['features']==columns
        assert set(p.participant_id)==set(te.participant_id)
        assert set(ip.participant_id)==set(tr.participant_id)
        assert not set(p.pair_id)&set(ip.pair_id)
        for frame in [p,ip]:
            np.testing.assert_array_equal(frame.y,truth.loc[frame.participant_id,'y'])
            np.testing.assert_array_equal(frame.pair_id,truth.loc[frame.participant_id,'pair_id'])
            assert np.isfinite(frame.probability).all()
            assert frame.probability.between(0,1).all()
        assert set(p.policy)==set(POLICIES)
        assert set(p['mode'])==({'tuned'} if kind=='benchmark' else set(MODES))
        for mode,it in ip.groupby('mode'):
            assert it.participant_id.is_unique
            cuts=thresholds(it.y.to_numpy(),it.probability.to_numpy())
            for policy,tau in cuts.items():
                z=p[(p['mode']==mode)&(p.policy==policy)]
                assert z.participant_id.is_unique and len(z)==len(te)
                np.testing.assert_allclose(z.threshold,tau,atol=1e-14,rtol=0)
                np.testing.assert_array_equal(z.prediction,z.probability>=tau)
                assert np.isclose(d['thresholds'][mode][policy],tau,atol=1e-14)
        if kind=='benchmark':
            data=np.load(folder/'trial_predictions.npz')
            trial=data['probabilities']; inner=data['inner_fold']
            assert trial.shape==(TRIALS,len(tr))
            assert pd.DataFrame({'pair':tr.pair_id.to_numpy(),'inner':inner}).groupby('pair').inner.nunique().max()==1
            scores=[]
            for i,record in enumerate(d['details']['trials']):
                auc=[roc_auc_score(tr.y.to_numpy()[inner==j],trial[i,inner==j]) for j in range(3)]
                np.testing.assert_allclose(auc,record['fold_auc'],atol=1e-12)
                np.testing.assert_allclose(np.mean(auc),record['mean_auc'],atol=1e-12)
                scores.append(np.mean(auc))
            chosen=int(np.argmax(scores));assert chosen==d['details']['chosen']
            np.testing.assert_array_equal(trial[chosen],ip.probability)
            selections.append(dict(cohort=cohort,configuration=config,arm=arm,family=family,fold=fold,
                best_inner_auc=scores[chosen],chosen_trial=chosen,params=json.dumps(d['details']['trials'][chosen]['params'])))
        else:
            train_groups=set(map(str,tr.pair_id));test_groups=set(map(str,te.pair_id))
            allhold=[]
            for record in d['details']['crossfit']:
                held=set(map(str,record['threshold_holdout_groups']));allhold.extend(held)
                assert held.issubset(train_groups)
                for b in record['members']:
                    parts=[set(map(str,b[k])) for k in ['train_groups','sigmoid_groups','isotonic_groups']]
                    assert set.union(*parts)==train_groups-held
                    assert all(not part & (held|test_groups) for part in parts)
                    assert all(not parts[i]&parts[j] for i in range(3) for j in range(i))
                    bounds+=1
            assert len(allhold)==len(train_groups) and set(allhold)==train_groups
            for b in d['details']['final']:
                parts=[set(map(str,b[k])) for k in ['train_groups','sigmoid_groups','isotonic_groups']]
                assert set.union(*parts)==train_groups
                assert all(not part&test_groups for part in parts)
                assert all(not parts[i]&parts[j] for i in range(3) for j in range(i))
                bounds+=1
        warnings.update(d['details']['warning_counts'])
        for k,v in dict(kind=kind,cohort=cohort,configuration=config,arm=arm,family=family).items():p[k]=v
        collected.append(p);count+=1
    pred=pd.concat(collected,ignore_index=True)
    sel=pd.DataFrame(selections)
    # This selector uses only the inner-fold AUC, never an outer outcome.
    chosen=sel.sort_values(['best_inner_auc','family'],ascending=[False,True]).drop_duplicates(['cohort','configuration','arm','fold'])
    selected=[]
    for r in chosen.to_dict('records'):
        q=pred[(pred.kind=='benchmark')&(pred.cohort==r['cohort'])&(pred.configuration==r['configuration'])&
            (pred.arm==r['arm'])&(pred.family==r['family'])&(pred.fold==r['fold'])].copy()
        q['family']='inner_selected';selected.append(q)
    pred=pd.concat([pred,*selected],ignore_index=True)
    pred.to_parquet(LOCAL/'predictions_all.parquet',index=False)
    sel.to_csv(PUBLIC/'tuning_choices.csv',index=False)
    chosen.to_csv(PUBLIC/'family_choices.csv',index=False)
    cols=['kind','cohort','configuration','arm','family','mode','policy']
    rows=[]
    for key,g in pred.groupby(cols,sort=True):
        assert g.participant_id.is_unique
        assert len(g)==(702 if key[1]=='n702' else 492)
        rows.append(dict(zip(cols,key))|expanded_metrics(g))
    result=pd.DataFrame(rows)
    assert len(result)==1185
    result.to_csv(PUBLIC/'metrics_all.csv',index=False)
    references=[]
    for cohort,path in [('n702',PRIVATE/'cohort.parquet'),('n492',PRIVATE/'agent_extension/evaluation.parquet')]:
        ref=pd.read_parquet(path)[['y']].copy()
        ref['probability']=.5
        for decision in [0,1]:
            ref['prediction']=decision
            references.append(dict(cohort=cohort,reference='all_positive' if decision else 'all_negative',
                probability_reference='constant 0.5 balanced-sample prior',**expanded_metrics(ref)))
    pd.DataFrame(references).to_csv(PUBLIC/'reference_decisions.csv',index=False)
    # No-feature agent arms must reproduce their raw counterpart bit-for-bit.
    for config in ['demographics','walking','sensors']:
        q=pred[(pred.kind=='benchmark')&(pred.cohort=='n492')&(pred.configuration==config)]
        key=['family','mode','policy','participant_id']
        a=q[q.arm=='raw'].sort_values(key);b=q[q.arm=='agent'].sort_values(key)
        np.testing.assert_array_equal(a.probability,b.probability)
        np.testing.assert_array_equal(a.prediction,b.prediction)
    protocol=json.loads((PUBLIC/'protocol.json').read_text())
    for path,digest in {**protocol['source_hashes'],**protocol['input_hashes']}.items():assert sha256(ROOT/path)==digest
    write_json(PUBLIC/'verification.json',dict(status='passed',completed_outer_tasks=count,
        metric_groups=len(result),private_prediction_rows=len(pred),calibration_member_boundaries=bounds,
        tuning_trials_verified=len(sel)*TRIALS,all_thresholds_recomputed=True,unchanged_no_feature_agent_controls=True,
        warnings=dict(warnings),limitations='Outer folds protect evaluation, but repeated research use remains exploratory. Audit does not certify clinical validity.'))
    print(json.dumps(dict(tasks=count,metrics=len(result),predictions=len(pred),calibration_boundaries=bounds)),flush=True)
    return pred,result


def paired_bootstrap(a,b,seed=20261118):
    a=a.sort_values('participant_id').reset_index(drop=True)
    b=b.set_index('participant_id').loc[a.participant_id].reset_index()
    np.testing.assert_array_equal(a.y,b.y)
    np.testing.assert_array_equal(a.pair_id,b.pair_id)
    blocks=[np.flatnonzero(a.pair_id.to_numpy()==g) for g in a.pair_id.unique()]
    assert all(len(x)==2 for x in blocks)
    blocks=np.stack(blocks);rng=np.random.default_rng(seed)
    ya=a.y.to_numpy();pa=a.probability.to_numpy();pb=b.probability.to_numpy()
    za=a.prediction.to_numpy();zb=b.prediction.to_numpy()
    def values(idx):
        y=ya[idx];pos=y==1;neg=~pos
        def score(p,z):
            fn=((z[idx]==0)&pos).sum();fp=((z[idx]==1)&neg).sum();tp=((z[idx]==1)&pos).sum()
            return np.array([roc_auc_score(y,p[idx]),1-(4*fn+2*fp)/len(y),5*tp/(5*tp+4*fn+fp),
                np.mean((p[idx]-y)**2),tp/pos.sum(),1-fp/neg.sum()])
        aa,bb=score(pa,za),score(pb,zb)
        return np.stack([aa-bb,aa,bb])
    point=values(np.arange(len(a)))
    draws=np.array([values(blocks[rng.integers(len(blocks),size=len(blocks))].ravel()) for _ in range(2000)])
    lo,hi=np.quantile(draws,[.025,.975],axis=0)
    return {name:dict(difference=float(point[0,i]),low=float(lo[0,i]),high=float(hi[0,i]),
        estimate_a=float(point[1,i]),a_low=float(lo[1,i]),a_high=float(hi[1,i]),
        estimate_b=float(point[2,i]),b_low=float(lo[2,i]),b_high=float(hi[2,i]))
        for i,name in enumerate(['auc','utility','f2','brier','sensitivity','specificity'])}


def comparisons(pred):
    records=[]
    def get(kind,cohort,config,arm,family,mode,policy):
        return pred[(pred.kind==kind)&(pred.cohort==cohort)&(pred.configuration==config)&(pred.arm==arm)&
                    (pred.family==family)&(pred['mode']==mode)&(pred.policy==policy)]
    for cohort,arm in [('n702','raw'),('n492','raw'),('n492','agent')]:
        path=PRIVATE/'runs/full_seed42/predictions.parquet' if cohort=='n702' else PRIVATE/f'agent_extension/runs/full_{arm}/predictions.parquet'
        old=pd.read_parquet(path)
        old=old[(old.method==('selected' if cohort=='n702' else 'selected_auc'))&(old.policy=='utility')]
        new=get('benchmark',cohort,'full',arm,'inner_selected','tuned','utility')
        records.append(dict(comparison='expanded_inner_selected_minus_previous_selected',cohort=cohort,arm=arm,
                            conditional_bootstrap=paired_bootstrap(new,old)))
    for family in ['lr','rf','xgboost','lightgbm','catboost','mlp','inner_selected']:
        a=get('benchmark','n492','full','agent',family,'tuned','utility')
        b=get('benchmark','n492','full','raw',family,'tuned','utility')
        records.append(dict(comparison='agent_minus_raw_full',family=family,conditional_bootstrap=paired_bootstrap(a,b)))
    for cohort,arm in [('n702','raw'),('n492','raw'),('n492','agent')]:
        for family in ['lr','xgboost']:
            base=get('calibration',cohort,'full',arm,family,'uncalibrated','f2_max')
            for mode in ['sigmoid','isotonic','sigmoid_isotonic']:
                a=get('calibration',cohort,'full',arm,family,mode,'f2_max')
                records.append(dict(comparison='calibration_minus_uncalibrated_full',cohort=cohort,arm=arm,family=family,
                    mode=mode,conditional_bootstrap=paired_bootstrap(a,base)))
    write_json(PUBLIC/'paired_comparisons.json',dict(bootstrap_replicates=2000,unit='matched pair',
        note='Conditional on fixed OOF predictions; exploratory, unadjusted descriptive intervals; not selection-adjusted or full-pipeline uncertainty.',comparisons=records))


def figures(pred,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    q=result[(result.kind=='benchmark')&(result.cohort=='n702')&(result.policy=='default')]
    table=q.pivot(index='configuration',columns='family',values='auc')
    fig,ax=plt.subplots(figsize=(9,5))
    im=ax.imshow(table.to_numpy(),vmin=.45,vmax=.85,cmap='viridis')
    ax.set_xticks(range(len(table.columns)),table.columns,rotation=30,ha='right')
    ax.set_yticks(range(len(table)),table.index)
    for i in range(len(table)):
        for j in range(len(table.columns)):ax.text(j,i,f'{table.iloc[i,j]:.3f}',ha='center',va='center',color='white' if table.iloc[i,j]<.68 else 'black')
    ax.set_title('Expanded nested benchmark: N=702, pooled outer-fold AUC')
    fig.colorbar(im,ax=ax,label='AUC');fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(PUBLIC/f'benchmark_auc.{ext}',dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharex=True,sharey=True)
    bins=[]
    for ax,family in zip(axes,['lr','xgboost']):
        for mode in MODES:
            p=pred[(pred.kind=='calibration')&(pred.cohort=='n702')&(pred.configuration=='full')&
                (pred.family==family)&(pred['mode']==mode)&(pred.policy=='default')]
            obs,mean=calibration_curve(p.y,p.probability,n_bins=8,strategy='uniform')
            ax.plot(mean,obs,'o-',label=mode)
            codes=np.minimum((p.probability.to_numpy()*8).astype(int),7)
            for b in range(8):
                z=p.iloc[np.flatnonzero(codes==b)]
                if len(z):bins.append(dict(family=family,mode=mode,bin=b,n=len(z),mean_probability=float(z.probability.mean()),fraction_pd=float(z.y.mean())))
        ax.plot([0,1],[0,1],'k--',lw=1);ax.set_title(family);ax.set_xlabel('Predicted probability')
    axes[0].set_ylabel('Observed case fraction');axes[1].legend(fontsize=7)
    fig.suptitle('Full features, N=702 balanced sample; not population risk calibration')
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(PUBLIC/f'calibration_curves.{ext}',dpi=200)
    plt.close(fig)
    pd.DataFrame(bins).to_csv(PUBLIC/'calibration_bins.csv',index=False)


def main():
    pred,result=reconcile()
    comparisons(pred)
    figures(pred,result)
    files=[p for p in PUBLIC.iterdir() if p.is_file() and p.name!='manifest.json']
    files += [ROOT/'research'/name for name in ['benchmark_report.py','test_benchmark_extension.py',
        'benchmark_checkpoints.py','benchmark_mlp_convergence.py']]
    write_json(PUBLIC/'manifest.json',{str(p.relative_to(ROOT)):sha256(p) for p in sorted(files)})
    print('Reporting, paired intervals and figures complete.',flush=True)


if __name__=='__main__':main()
