"""Replay every outer model, verify saved decisions, report all AUC follow-up arms."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score,roc_curve
from research.agent_auc_followup import P,L,OLD,load,models,check_frozen,signature,rank_auc
from research.agent_features import augment
from research.core import feature_sets,splits,select_thresholds,metrics
from research.common import ROOT,sha256,write_json
LABELS={'raw':'Conventional inputs','prior_utility_agent':'Previous utility-selected feature','auc_agent':'New AUC-selected agent features','automatic_control':'Automated log-transform control'}
def verify():
    spec=check_frozen();done=json.loads((L/'complete.json').read_text());assert done['signature']==signature() and done['frozen_sha256']==sha256(P/'frozen.json')
    assert all(sha256(L/f)==h for f,h in done['outputs'].items())
    d=load('evaluation');cols=feature_sets(d)['full'];y=d.y.to_numpy();g=d.pair_id.to_numpy();pred=pd.read_parquet(L/'predictions.parquet')
    assert len(pred)==492*4*5*4
    assert pred.groupby(['arm','method','policy','row_index']).size().eq(1).all()
    errors=[];fits=0;foldscores=[]
    for fold,(tr,te) in enumerate(splits(y,g,n=5,seed=42)):
        assert not set(g[tr])&set(g[te])
        for arm,features in spec['arms'].items():
            x,_=augment(d,cols,features);xx=x.to_numpy(float);path=L/f'fold{fold}_{arm}'
            saved=np.load(path/'inputs_predictions.npz');details=json.loads((path/'details.json').read_text())
            np.testing.assert_array_equal(saved['train_indices'],tr);np.testing.assert_array_equal(saved['test_indices'],te);assert details['columns']==list(x)
            candidate_scores={};probabilities={}
            for method in models():
                fit=joblib.load(path/f'{method}.joblib');fits+=1
                np.testing.assert_allclose(fit[0].statistics_,np.nanmedian(xx[tr],axis=0),equal_nan=True)
                train_imputed=fit[0].transform(xx[tr]);np.testing.assert_allclose(fit[1].center_,np.median(train_imputed,axis=0))
                probability=fit.predict_proba(xx[te])[:,1];probabilities[method]=probability
                errors.append(float(abs(probability-saved['outer_'+method]).max()));np.testing.assert_allclose(probability,saved['outer_'+method],atol=1e-12,rtol=0)
                inner=saved['inner_'+method];assert len(inner)==len(tr) and np.isfinite(inner).all()
                aucs=[]
                for it,iv in splits(y[tr],g[tr],n=3,seed=143):
                    assert not set(g[tr][it])&set(g[tr][iv]);aucs.append(roc_auc_score(y[tr][iv],inner[iv]))
                np.testing.assert_allclose(aucs,details['inner_auc'][method],atol=1e-12,rtol=0);candidate_scores[method]=np.mean(aucs)
                assert select_thresholds(y[tr],inner)==details['thresholds'][method]
            winner=sorted(candidate_scores,key=lambda k:(-candidate_scores[k],k))[0];assert winner==details['winner']
            for method in list(models())+['inner_selected']:
                actual=winner if method=='inner_selected' else method;p=probabilities[actual]
                for policy,tau in details['thresholds'][actual].items():
                    a=pred[(pred.fold==fold)&(pred.arm==arm)&(pred.method==method)&(pred.policy==policy)].set_index('row_index').loc[te]
                    np.testing.assert_array_equal(a.y,y[te]);np.testing.assert_allclose(a.probability,p,atol=1e-12,rtol=0);np.testing.assert_allclose(a.threshold,tau,atol=0,rtol=0);np.testing.assert_array_equal(a.prediction,p>=tau)
                    foldscores.append(dict(arm=arm,method=method,policy=policy,fold=fold,**metrics(y[te],p,p>=tau)))
        print(f'All outer model replays passed: fold {fold+1}/5',flush=True)
    write_json(P/'verification.json',dict(status='passed',participants=492,discovery_participants=210,outer_models_replayed=fits,maximum_probability_error=max(errors),prediction_records=len(pred),frozen_sha256=sha256(P/'frozen.json'),source_hashes={str(f.relative_to(ROOT)):sha256(f) for f in [Path(__file__),ROOT/'research/agent_auc_followup.py',ROOT/'research/test_agent_auc_followup.py']},scope='All outer fits replayed; inner scores/thresholds recomputed from saved inner predictions; unit tests cover isolation. Not independent rediscovery.'))
    return d,pred,pd.DataFrame(foldscores)

def report():
    d,pred,foldscores=verify();foldscores.to_csv(P/'metrics_by_fold.csv',index=False);rows=[]
    for (arm,method,policy),g in pred.groupby(['arm','method','policy']):
        m=metrics(g.y,g.probability,g.prediction)
        m['mean_outer_fold_auc']=float(foldscores[(foldscores.arm==arm)&(foldscores.method==method)&(foldscores.policy==policy)].auc.mean())
        rows.append(dict(arm=arm,method=method,policy=policy,**m))
    summary=pd.DataFrame(rows);summary.to_csv(P/'metrics_all.csv',index=False)
    comparisons=[('new_agent_minus_raw','auc_agent','raw','inner_selected'),('prior_agent_minus_raw','prior_utility_agent','raw','inner_selected'),('control_minus_raw','automatic_control','raw','inner_selected'),('new_agent_minus_control','auc_agent','automatic_control','inner_selected')]+[(f'new_agent_minus_raw_{m}','auc_agent','raw',m) for m in models()]
    wanted=sorted({(a,m) for _,a,b,m in comparisons}|{(b,m) for _,a,b,m in comparisons})
    probs={}
    for arm,method in wanted:
        a=pred[(pred.arm==arm)&(pred.method==method)&(pred.policy=='default')].sort_values('row_index');np.testing.assert_array_equal(a.row_index,np.arange(492));probs[(arm,method)]=a.probability.to_numpy()
    y=d.y.to_numpy();groups=d.pair_id.to_numpy();outer=splits(y,groups,n=5,seed=42);foldrows=[te for tr,te in outer]
    within=[np.array([np.flatnonzero(groups==pair) for pair in np.unique(groups[te])]) for te in foldrows]
    assert all(arr.shape[1]==2 for arr in within)
    rng=np.random.default_rng(20261119);boot={k:np.empty((2000,2)) for k in wanted}
    for rep in range(2000):
        drawn=[arr[rng.integers(0,len(arr),len(arr))].ravel() for arr in within];pool=np.concatenate(drawn)
        for key,p in probs.items():
            boot[key][rep]=[np.mean([rank_auc(y[ix],p[ix]) for ix in drawn]),rank_auc(y[pool],p[pool])]
    records=[]
    for name,a,b,method in comparisons:
        delta=boot[(a,method)]-boot[(b,method)]
        for j,metric in enumerate(['mean_outer_fold_auc','pooled_auc']):
            pa,pb=probs[(a,method)],probs[(b,method)]
            effect=np.mean([rank_auc(y[ix],pa[ix])-rank_auc(y[ix],pb[ix]) for ix in foldrows]) if j==0 else rank_auc(y,pa)-rank_auc(y,pb)
            lo,hi=np.percentile(delta[:,j],[2.5,97.5]);records.append(dict(comparison=name,method=method,metric=metric,estimate=effect,ci_low=lo,ci_high=hi))
    effects=pd.DataFrame(records);effects.to_csv(P/'paired_auc_effects.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for arm in LABELS:
        p=probs[(arm,'inner_selected')];fpr,tpr,_=roc_curve(y,p);axes[0].plot(fpr,tpr,label=f'{LABELS[arm]} ({rank_auc(y,p):.3f})',lw=1.5)
    axes[0].plot([0,1],[0,1],ls='--',color='0.6');axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='Pooled held-out ROC; same 492 people');axes[0].legend(fontsize=7,loc='lower right')
    e=effects[(effects.method=='inner_selected')&(effects.metric=='mean_outer_fold_auc')]
    labels={'new_agent_minus_raw':'New agent − conventional','prior_agent_minus_raw':'Previous feature − conventional','control_minus_raw':'Control − conventional','new_agent_minus_control':'New agent − control'}
    for i,r in enumerate(e.itertuples()):axes[1].errorbar(r.estimate,i,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='o',capsize=3,color='#246c8c')
    axes[1].set_yticks(range(len(e)),[labels[c] for c in e.comparison],fontsize=9);axes[1].invert_yaxis();axes[1].axvline(0,ls='--',color='0.6');axes[1].set(xlabel='Difference in mean outer-fold ROC-AUC',title='Conditional paired 95% intervals')
    fig.suptitle('AUC-focused follow-up: exploratory, previously inspected cohort');fig.tight_layout();fig.savefig(P/'auc_followup.png',dpi=180);fig.savefig(P/'auc_followup.pdf');plt.close(fig)
    print(summary[(summary.method=='inner_selected')&(summary.policy=='utility')][['arm','mean_outer_fold_auc','auc','sensitivity','specificity','utility']].to_string(index=False))
    print(effects.to_string(index=False))

if __name__=='__main__':
    report()
