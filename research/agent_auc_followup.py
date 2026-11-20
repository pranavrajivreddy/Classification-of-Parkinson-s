"""Bounded AUC discovery and separate exploratory nested evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from research.common import ROOT,PRIVATE,sha256,write_json
from research.core import feature_sets,splits,pipeline,select_thresholds,metrics
from research.agent_features import augment,parsed
P=ROOT/'paper_ready/agent_auc_followup';L=PRIVATE/'agent_auc_followup';OLD=PRIVATE/'agent_extension'
REPEATS=[4242,731]
def models():
    return {'lr_c01':LogisticRegression(C=.1,max_iter=3000,random_state=42),
            'lr_c1':LogisticRegression(C=1,max_iter=3000,random_state=42),
            **{f'xgb_d{depth}':XGBClassifier(n_estimators=160,max_depth=depth,learning_rate=.05,min_child_weight=3,reg_lambda=1,subsample=1,colsample_bytree=1,tree_method='hist',n_jobs=1,random_state=42,eval_metric='logloss') for depth in [2,3]}}
def load(which):
    assert which in ['discovery','evaluation']
    m=json.loads((OLD/'split_manifest.json').read_text());f=OLD/f'{which}.parquet'
    assert sha256(f)==m[f'{which}_sha256'];return pd.read_parquet(f).reset_index(drop=True)
def signature():
    return {str(f.relative_to(ROOT)):sha256(f) for f in [Path(__file__),ROOT/'research/core.py',ROOT/'research/agent_features.py',P/'PROTOCOL.md',OLD/'discovery.parquet',OLD/'evaluation.parquet']}
def score_features(features):
    d=load('discovery');cols=feature_sets(d)['full'];y=d.y.to_numpy();g=d.pair_id.to_numpy();rows=[]
    for name,fs in [('baseline',[])]+[(f['name'],[f]) for f in features]:
        x,_=augment(d,cols,fs)
        for repeat in REPEATS:
            for fold,(tr,va) in enumerate(splits(y,g,n=3,seed=repeat)):
                for method,model in models().items():
                    est=pipeline(model);est.fit(x.iloc[tr].to_numpy(float),y[tr]);p=est.predict_proba(x.iloc[va].to_numpy(float))[:,1]
                    rows.append(dict(candidate=name,repeat=repeat,fold=fold,method=method,auc=roc_auc_score(y[va],p)))
        print('Discovery scored',name,flush=True)
    return pd.DataFrame(rows)
def summary(scores):
    base=scores[scores.candidate=='baseline'].drop(columns='candidate').rename(columns={'auc':'baseline_auc'})
    z=scores.merge(base,on=['repeat','fold','method'],validate='many_to_one');z['gain']=z.auc-z.baseline_auc
    return z.groupby('candidate').agg(mean_auc=('auc','mean'),mean_gain=('gain','mean')).reset_index(),z

def score_round(number):
    assert not (P/'frozen.json').exists(),'Discovery frozen'
    output=P/f'round{number}_scores.csv';assert not output.exists(),'Round already executed'
    proposal=json.loads((P/f'round{number}_proposal.json').read_text());fs=proposal['features'];assert 0<len(fs)<=6
    for f in fs:parsed(f['formula'],set(feature_sets(load('discovery'))['sensors']))
    if number==2:
        assert (P/'round1_scores.csv').exists()
        old=json.loads((P/'round1_proposal.json').read_text())['features']
        assert not {f['name'] for f in fs}&{f['name'] for f in old}
        assert not {f['formula'] for f in fs}&{f['formula'] for f in old}
    scores=score_features(fs);scores.to_csv(output,index=False)
    s,_=summary(scores);s.to_csv(P/f'round{number}_summary.csv',index=False)
    write_json(P/f'round{number}_manifest.json',dict(signature=signature(),proposal_sha256=sha256(P/f'round{number}_proposal.json'),scores_sha256=sha256(output)))
    print(s.to_string(index=False))
def eligible_bundle(scores,features):
    s,z=summary(scores);per=z.groupby(['candidate','repeat']).gain.mean().unstack()
    positive=per.index[(per>0).all(axis=1)]
    eligible=s[(s.candidate!='baseline')&(s.mean_gain>=.002)&s.candidate.isin(positive)].sort_values(['mean_gain','candidate'],ascending=[False,True])
    names=eligible.candidate.head(2).tolist();lookup={f['name']:f for f in features}
    return [lookup[n] for n in names]
def score_bundle(features):
    d=load('discovery');cols=feature_sets(d)['full'];y=d.y.to_numpy();g=d.pair_id.to_numpy();rows=[]
    x,_=augment(d,cols,features)
    for repeat in REPEATS:
        for fold,(tr,va) in enumerate(splits(y,g,n=3,seed=repeat)):
            for name,model in models().items():
                est=pipeline(model);est.fit(x.iloc[tr].to_numpy(float),y[tr]);p=est.predict_proba(x.iloc[va].to_numpy(float))[:,1]
                rows.append(dict(candidate='bundle',repeat=repeat,fold=fold,method=name,auc=roc_auc_score(y[va],p)))
    return pd.DataFrame(rows)
def freeze():
    assert not (P/'frozen.json').exists()
    pieces=[];features=[]
    for r in [1,2]:
        m=json.loads((P/f'round{r}_manifest.json').read_text());assert m['signature']==signature()
        assert m['scores_sha256']==sha256(P/f'round{r}_scores.csv') and m['proposal_sha256']==sha256(P/f'round{r}_proposal.json')
        t=pd.read_csv(P/f'round{r}_scores.csv');pieces.append(t if r==1 else t[t.candidate!='baseline']);features+=json.loads((P/f'round{r}_proposal.json').read_text())['features']
    agent_scores=pd.concat(pieces,ignore_index=True);d=load('discovery');cols=sorted(feature_sets(d)['sensors'])
    chosen=np.random.default_rng(20261119).choice(cols,12,replace=False)
    control=[dict(name='agent_control_'+str(i),formula=f'logabs({c})',rationale='Outcome-blind seeded log-transform control.') for i,c in enumerate(chosen)]
    write_json(P/'control_proposal.json',dict(seed=20261119,features=control));control_scores=score_features(control);control_scores.to_csv(P/'control_scores.csv',index=False)
    frozen={};decisions=[]
    for arm,fs,scores in [('auc_agent',features,agent_scores),('automatic_control',control,control_scores)]:
        bundle=eligible_bundle(scores,fs);b=score_bundle(bundle);b.to_csv(P/f'{arm}_bundle_scores.csv',index=False)
        base=scores[scores.candidate=='baseline'];s,z=summary(pd.concat([base,b],ignore_index=True));row=s[s.candidate=='bundle'].iloc[0]
        gains=z[z.candidate=='bundle'].groupby('repeat').gain.mean();accept=len(bundle)>0 and row.mean_gain>=.002 and (gains>0).all()
        frozen[arm]=bundle if accept else []
        decisions.append(dict(arm=arm,proposed=[f['name'] for f in bundle],accepted=bool(accept),mean_gain=float(row.mean_gain),repetition_gains=gains.to_dict()))
        print(arm,decisions[-1],flush=True)
    old=json.loads((ROOT/'paper_ready/agent_extension/frozen_features.json').read_text());names=old['configuration_features']['full']
    frozen['prior_utility_agent']=[f for f in old['features'] if f['name'] in names];frozen['raw']=[]
    provenance={f.name:sha256(f) for f in P.glob('*') if f.is_file()}
    write_json(P/'frozen.json',dict(signature=signature(),arms=frozen,decisions=decisions,discovery_provenance=provenance,prior_specification_sha256=sha256(ROOT/'paper_ready/agent_extension/frozen_features.json'),note='Frozen before this new evaluation; previous evaluation results were already seen. Exploratory only.'))

def tune_predict(x,y,groups,xtest):
    """No test labels or evaluation outcomes accepted by this API."""
    inner=splits(y,groups,n=3,seed=143);ps={};fits={};qs={};scores={};cuts={}
    for name,model in models().items():
        p=np.full(len(y),np.nan);auc=[]
        for tr,va in inner:
            est=pipeline(model);est.fit(x[tr],y[tr]);p[va]=est.predict_proba(x[va])[:,1];auc.append(roc_auc_score(y[va],p[va]))
        fit=pipeline(model);fit.fit(x,y);ps[name]=p;fits[name]=fit;qs[name]=fit.predict_proba(xtest)[:,1];scores[name]=auc;cuts[name]=select_thresholds(y,p)
    winner=sorted(scores,key=lambda n:(-np.mean(scores[n]),n))[0]
    return ps,fits,qs,scores,cuts,winner

def check_frozen():
    frozen=json.loads((P/'frozen.json').read_text());assert frozen['signature']==signature()
    assert all(sha256(P/n)==h for n,h in frozen['discovery_provenance'].items())
    assert sha256(ROOT/'paper_ready/agent_extension/frozen_features.json')==frozen['prior_specification_sha256']
    return frozen

def evaluate():
    spec=check_frozen();done=L/'complete.json'
    if done.exists():
        v=json.loads(done.read_text());assert v['frozen_sha256']==sha256(P/'frozen.json') and v['signature']==signature()
        assert all(sha256(L/f)==h for f,h in v['outputs'].items());print('Existing evaluation hashes verified.');return
    assert not (L/'predictions.parquet').exists(),'Incomplete evaluation; inspect before overwriting'
    d=load('evaluation');discovery=load('discovery');assert not set(d.participant_id)&set(discovery.participant_id)
    assert not set(d.pair_id)&set(discovery.pair_id)
    source=pd.read_parquet(PRIVATE/'cohort.parquet');pd.testing.assert_frame_equal(pd.concat([d,discovery]).sort_values('participant_id').reset_index(drop=True),source.sort_values('participant_id').reset_index(drop=True))
    cols=feature_sets(d)['full'];y=d.y.to_numpy();g=d.pair_id.to_numpy();rows=[];choices=[]
    for fold,(tr,te) in enumerate(splits(y,g,n=5,seed=42)):
        for arm,fs in spec['arms'].items():
            x,_=augment(d,cols,fs);xx=x.to_numpy(float);ps,fits,qs,scores,cuts,winner=tune_predict(xx[tr],y[tr],g[tr],xx[te])
            path=L/f'fold{fold}_{arm}';path.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path/'inputs_predictions.npz',train_indices=tr,test_indices=te,**{'inner_'+k:v for k,v in ps.items()},**{'outer_'+k:v for k,v in qs.items()})
            for method,fit in fits.items():joblib.dump(fit,path/f'{method}.joblib')
            write_json(path/'details.json',dict(columns=list(x),inner_auc=scores,thresholds=cuts,winner=winner))
            choices.append(dict(fold=fold,arm=arm,winner=winner,inner_mean_auc=float(np.mean(scores[winner]))))
            for method in list(models())+['inner_selected']:
                actual=winner if method=='inner_selected' else method;p=qs[actual]
                for policy,tau in cuts[actual].items():
                    rows.extend([dict(row_index=int(i),pair_id=str(g[i]),fold=fold,arm=arm,method=method,policy=policy,y=int(y[i]),probability=float(prob),threshold=tau,prediction=int(prob>=tau)) for i,prob in zip(te,p)])
        print(f'Evaluation fold {fold+1}/5 complete',flush=True)
    pd.DataFrame(rows).to_parquet(L/'predictions.parquet',index=False);pd.DataFrame(choices).to_csv(P/'model_choices.csv',index=False)
    write_json(done,dict(signature=signature(),frozen_sha256=sha256(P/'frozen.json'),outputs={str(f.relative_to(L)):sha256(f) for f in L.rglob('*') if f.is_file() and f!=done}))

def rank_auc(y,p):
    y=np.asarray(y);p=np.asarray(p);n1=y.sum();n0=len(y)-n1
    return float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['round1','round2','freeze','evaluate']);arg=parser.parse_args().stage
    if arg.startswith('round'):score_round(int(arg[-1]))
    else:globals()[arg]()
