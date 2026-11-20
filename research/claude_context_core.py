"""Local computation for the bounded single-Claude feedback experiment."""
import json,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss,average_precision_score
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from research.common import ROOT,PRIVATE,sha256,write_json
from research.core import feature_sets,splits,pipeline,select_thresholds
from research.agent_auc_followup import load
from research.agent_features import augment,parsed
from research.reporting_metrics import allmetrics

P=ROOT/'paper_ready/claude_context_replication';L=PRIVATE/'claude_context_replication'
REPEATS=[4242,731]
def models():
 return dict(lr=LogisticRegression(C=1,max_iter=3000,random_state=42),
 rf=RandomForestClassifier(n_estimators=120,max_depth=5,min_samples_leaf=3,random_state=42,n_jobs=1),
 xgboost=XGBClassifier(n_estimators=160,max_depth=2,learning_rate=.05,min_child_weight=3,reg_lambda=1,subsample=1,colsample_bytree=1,tree_method='hist',n_jobs=1,random_state=42,eval_metric='logloss'),
 lightgbm=LGBMClassifier(n_estimators=160,num_leaves=7,max_depth=3,min_child_samples=15,learning_rate=.05,reg_lambda=1,n_jobs=1,random_state=42,verbosity=-1),
 catboost=CatBoostClassifier(iterations=160,depth=3,learning_rate=.05,l2_leaf_reg=3,random_seed=42,thread_count=1,verbose=False,allow_writing_files=False),
 mlp=MLPClassifier(hidden_layer_sizes=(16,),alpha=1,solver='lbfgs',max_iter=2000,max_fun=50000,random_state=42))
def signature():
 paths=[Path(__file__),ROOT/'research/claude_context_api.py',ROOT/'research/claude_context_run.py',ROOT/'research/core.py',ROOT/'research/agent_features.py',ROOT/'research/agent_auc_followup.py',ROOT/'research/common.py',ROOT/'research/reporting_metrics.py',P/'PROTOCOL.md',P/'STUDY_CONTEXT.json',PRIVATE/'agent_extension/discovery.parquet',PRIVATE/'agent_extension/evaluation.parquet',PRIVATE/'agent_extension/split_manifest.json',ROOT/'paper_ready/FEATURE_DICTIONARY.md',ROOT/'paper_ready/agent_extension/frozen_features.json',ROOT/'paper_ready/agent_auc_followup/round1_proposal.json',ROOT/'paper_ready/agent_auc_followup/round2_proposal.json',ROOT/'configs/prior_multiagent_round1.json',ROOT/'configs/prior_multiagent_round2.json']
 return {str(p.relative_to(ROOT)):sha256(p) for p in paths}
def fit(model,x,y):
 with warnings.catch_warnings(record=True) as found:
  warnings.simplefilter('always');est=pipeline(clone(model));est.fit(x,y)
 return est,[dict(category=w.category.__name__,message=str(w.message)) for w in found]
def validate(features,allowed,seen=None):
 seen=set() if seen is None else set(seen);valid=[];bad=[]
 for f in features:
  try:
   assert isinstance(f,dict) and isinstance(f.get('formula'),str)
   tree,deps=parsed(f['formula'],set(allowed));import ast
   canonical=ast.dump(tree,include_attributes=False)
   assert canonical not in seen,'duplicate formula'
   assert f['name'].startswith('agent_') and len(f['name'])<90
   seen.add(canonical);valid.append(dict(name=f['name'],formula=f['formula'],inputs=sorted(deps),rationale=str(f.get('rationale',''))[:1600]))
  except (ValueError,AssertionError,KeyError,TypeError,SyntaxError) as exc:bad.append(dict(candidate=str(f.get('name','unknown')) if isinstance(f,dict) else 'nonobject',reason=type(exc).__name__+': '+str(exc)))
 return valid,bad,seen
def score(name,features):
 path=L/'discovery'/f'{name}.csv';path.parent.mkdir(parents=True,exist_ok=True)
 stamp=dict(signature=signature(),features=features)
 if path.exists():
  assert json.loads(path.with_suffix('.json').read_text())['spec']==stamp
  assert sha256(path)==json.loads(path.with_suffix('.json').read_text())['output_sha256'];return pd.read_csv(path)
 d=load('discovery');x,_=augment(d,feature_sets(d)['full'],features);xx=x.to_numpy(float);y=d.y.to_numpy();g=d.pair_id.to_numpy();rows=[];alerts=[]
 for seed in REPEATS:
  for fold,(tr,va) in enumerate(splits(y,g,n=3,seed=seed)):
   predictions=[]
   for method,model in models().items():
    est,ws=fit(model,xx[tr],y[tr]);p=est.predict_proba(xx[va])[:,1];assert np.isfinite(p).all();predictions.append(p)
    alerts += [dict(repeat=seed,fold=fold,method=method,**w) for w in ws]
    rows.append(dict(candidate=name,repeat=seed,fold=fold,method=method,**allmetrics(y[va],p,(p>=.5).astype(int))))
   p=np.mean(predictions,axis=0);rows.append(dict(candidate=name,repeat=seed,fold=fold,method='equal_ensemble',**allmetrics(y[va],p,(p>=.5).astype(int))))
 out=pd.DataFrame(rows);out.to_csv(path,index=False);write_json(path.with_suffix('.json'),dict(spec=stamp,output_sha256=sha256(path),warnings=alerts));print('Discovery complete:',name,flush=True);return out
def feedback(scores,baseline):
 b=baseline.drop(columns='candidate').rename(columns={k:'base_'+k for k in ['auc','brier','average_precision']})
 z=scores.merge(b,on=['repeat','fold','method'],validate='many_to_one');z['gain']=z.auc-z.base_auc
 rows=[]
 for name,g in z.groupby('candidate'):
  a=g[g.method=='equal_ensemble'];reps=a.groupby('repeat').gain.mean();fam=g[g.method!='equal_ensemble'].groupby('method').gain.mean()
  rows.append(dict(candidate=name,ensemble_auc=float(a.auc.mean()),gain=float(a.gain.mean()),repeat_gains={str(k):float(v) for k,v in reps.items()},family_gains={str(k):float(v) for k,v in fam.items()},eligible=bool(a.gain.mean()>=.002 and (reps>0).all() and (fam>=0).sum()>=4),brier_change=float((a.brier-a.base_brier).mean())))
 return rows
def choose(features,scores,baseline,arm):
 if not features:return [],dict(arm=arm,proposed=[],accepted=False,reason='no eligible candidates')
 infos=feedback(scores,baseline);eligible=sorted([r for r in infos if r['eligible']],key=lambda r:(-r['gain'],r['candidate']))[:2]
 lookup={f['name']:f for f in features};bundle=[lookup[r['candidate']] for r in eligible]
 if not bundle:return [],dict(arm=arm,proposed=[],accepted=False,reason='no eligible candidates')
 bs=score(arm+'_bundle',bundle);r=feedback(bs,baseline)[0];accepted=r['eligible']
 return bundle if accepted else [],dict(arm=arm,proposed=bundle,accepted=accepted,score=r)
def automatic_candidates():
 allowed=sorted(feature_sets(load('discovery'))['sensors']);rng=np.random.default_rng(20261120);out=[]
 for i,c in enumerate(rng.choice(allowed,8,replace=False)):out.append(dict(name=f'agent_control_{i}',formula=f'{"logabs" if i%2==0 else "sqrtabs"}({c})',rationale='Seeded outcome-blind transform'))
 for i in range(8):
  prefix=['tap_','walk_','voice_'][i%3];a,b=rng.choice([c for c in allowed if c.startswith(prefix)],2,replace=False)
  formula=f'pdiv({a}, {b})' if i%2 else f'logabs({a}) * logabs({b})'
  out.append(dict(name=f'agent_control_{i+8}',formula=formula,rationale='Seeded outcome-blind within-modality interaction'))
 valid,bad,_=validate(out,allowed);assert not bad and len(valid)==16;return valid
def traincell(x,y,groups,xtest):
 ps={};qs={};fits={};aucs={};notes={};boundaries=[]
 inner=splits(y,groups,n=3,seed=143)
 for method,model in models().items():
  p=np.full(len(y),np.nan);a=[];alerts=[]
  for fold,(tr,va) in enumerate(inner):
   est,ws=fit(model,x[tr],y[tr]);p[va]=est.predict_proba(x[va])[:,1];a.append(roc_auc_score(y[va],p[va]));alerts+=ws
   boundaries.append(dict(method=method,fold=fold,train=tr.tolist(),validation=va.tolist(),medians=est[0].statistics_.tolist(),center=est[1].center_.tolist(),scale=est[1].scale_.tolist()))
  est,ws=fit(model,x,y);alerts+=ws;q=est.predict_proba(xtest)[:,1];assert np.isfinite(p).all() and np.isfinite(q).all()
  ps[method]=p;qs[method]=q;fits[method]=est;aucs[method]=a;notes[method]=alerts
 winner=sorted(aucs,key=lambda k:(-np.mean(aucs[k]),k))[0]
 ps['equal_ensemble']=np.mean(list(ps.values()),axis=0);qs['equal_ensemble']=np.mean(list(qs.values()),axis=0)
 ps['inner_selected']=ps[winner];qs['inner_selected']=qs[winner]
 cuts={k:select_thresholds(y,v) for k,v in ps.items()}
 return ps,qs,fits,dict(inner_auc=aucs,winner=winner,thresholds=cuts,warnings=notes,inner_boundaries=boundaries)
def evaluate():
 from research.claude_context_run import evaluation_required
 assert evaluation_required(), 'No selected Claude bundle: evaluation is not authorized by this protocol'
 frozen=json.loads((P/'frozen.json').read_text());assert frozen['signature']==signature()
 for name,h in frozen['provenance'].items():assert sha256(P/name)==h
 complete=L/'complete.json'
 if complete.exists():
  done=json.loads(complete.read_text());assert done['signature']==signature() and done['frozen_sha256']==sha256(P/'frozen.json');assert all(sha256(L/p)==h for p,h in done['outputs'].items());print('Completed evaluation verified.',flush=True);return
 d=load('evaluation');disc=load('discovery');assert not set(d.participant_id)&set(disc.participant_id) and not set(d.pair_id)&set(disc.pair_id)
 y=d.y.to_numpy();g=d.pair_id.to_numpy();rows=[]
 for fold,(tr,te) in enumerate(splits(y,g,n=5,seed=42)):
  for arm,fs in frozen['arms'].items():
   x,_=augment(d,feature_sets(d)['full'],fs);xx=x.to_numpy(float);path=L/f'fold{fold}_{arm}';path.mkdir(parents=True,exist_ok=True)
   marker=path/'complete.json'
   if marker.exists():
    old=json.loads(marker.read_text());assert old['frozen_sha256']==sha256(P/'frozen.json');assert all(sha256(path/p)==h for p,h in old['outputs'].items())
    rows+=pd.read_parquet(path/'predictions.parquet').to_dict('records');continue
   ps,qs,fits,info=traincell(xx[tr],y[tr],g[tr],xx[te]);part=[]
   for method,est in fits.items():joblib.dump(est,path/f'{method}.joblib')
   np.savez_compressed(path/'inner.npz',tr=tr,te=te,**ps);write_json(path/'details.json',dict(columns=list(x),**info))
   for method,p in qs.items():
    for policy,tau in info['thresholds'][method].items():
     part += [dict(row_index=int(i),fold=fold,arm=arm,method=method,policy=policy,y=int(y[i]),probability=float(prob),threshold=tau,prediction=int(prob>=tau)) for i,prob in zip(te,p)]
   pd.DataFrame(part).to_parquet(path/'predictions.parquet',index=False);rows+=part
   write_json(marker,dict(frozen_sha256=sha256(P/'frozen.json'),outputs={p.name:sha256(p) for p in path.iterdir() if p.is_file() and p!=marker}));print(f'Evaluation {fold+1}/5 {arm} complete',flush=True)
 pd.DataFrame(rows).to_parquet(L/'predictions.parquet',index=False)
 write_json(complete,dict(signature=signature(),frozen_sha256=sha256(P/'frozen.json'),outputs={str(p.relative_to(L)):sha256(p) for p in L.rglob('*') if p.is_file() and p!=complete}))
def learning_curves():
 path=P/'discovery_learning_curves.csv'
 if path.exists():return
 d=load('discovery');x=d[feature_sets(d)['full']].to_numpy(float);y=d.y.to_numpy();g=d.pair_id.to_numpy();rows=[]
 for seed in REPEATS:
  for fold,(tr,va) in enumerate(splits(y,g,n=3,seed=seed)):
   pairs=np.random.default_rng(20261120+fold+seed).permutation(np.unique(g[tr]))
   for frac in [.4,.7,1.]:
    chosen=pairs[:max(2,int(np.ceil(len(pairs)*frac)))];ix=tr[np.isin(g[tr],chosen)]
    for name in ['lr','xgboost']:
     est,ws=fit(models()[name],x[ix],y[ix]);rows.append(dict(repeat=seed,fold=fold,fraction=frac,n_train=len(ix),method=name,train_auc=roc_auc_score(y[ix],est.predict_proba(x[ix])[:,1]),validation_auc=roc_auc_score(y[va],est.predict_proba(x[va])[:,1]),warning_count=len(ws)))
 pd.DataFrame(rows).to_csv(path,index=False)
