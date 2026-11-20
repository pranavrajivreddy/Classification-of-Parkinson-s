"""Independent pre-evaluation audit, prediction replay and complete reporting."""
import argparse,json,ast,re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from research.common import ROOT,sha256,write_json
from research.final_multiagent_core import P,L,signature,load,models,feature_sets,feedback,automatic_candidates
from research.final_multiagent_run import schema_context
from research.final_multiagent_api import payload,MODEL,CAP
from research.agent_features import augment
from research.core import splits,select_thresholds
from research.reporting_metrics import allmetrics
from research.agent_auc_followup import rank_auc

def audit():
 frozen=json.loads((P/'frozen.json').read_text());assert frozen['signature']==signature()
 assert json.loads((P/'start_manifest.json').read_text())['signature']==signature()
 assert all(sha256(P/name)==h for name,h in frozen['provenance'].items())
 context=json.loads((P/'api_context.json').read_text());assert schema_context()==context
 d=load('discovery');e=load('evaluation');assert len(d)==210 and len(e)==492
 assert not set(d.participant_id)&set(e.participant_id) and not set(d.pair_id)&set(e.pair_id)
 source=pd.read_parquet(ROOT/'paper_ready/private/cohort.parquet')
 pd.testing.assert_frame_equal(pd.concat([d,e]).sort_values('participant_id').reset_index(drop=True),source.sort_values('participant_id').reset_index(drop=True))
 ledger=json.loads((P/'api_ledger.json').read_text());assert len(ledger)==6 and len({r['name'] for r in ledger})==6
 assert sum(r['reserved_usd'] for r in ledger)<=CAP
 baseline=pd.read_csv(L/'discovery/baseline.csv');prior=[];candidates=[]
 for r in [1,2]:
  record=json.loads((P/f'round{r}_proposals.json').read_text());approved={a['name'] for a in record['critic'] if a['approve']}
  assert record['approved']==[f for f in record['proposed'] if f['name'] in approved]
  for role in ['measurement','integration','critic']:
   name=f'round{r}_{role}';entry=next(v for v in ledger if v['name']==name)
   assert entry['status']=='completed' and entry['http_status']==200 and entry['provider']=='DeepInfra'
   request=json.loads((P/'api'/f'{name}_request.json').read_text());response=json.loads((P/'api'/f'{name}_response.json').read_text());user=json.loads(request['messages'][1]['content'])
   expected,reserve=payload(request['messages'][0]['content'],user);assert request==expected and reserve==entry['reserved_usd']
   assert response['provider']=='DeepInfra' and response['model']==MODEL and response['usage']==entry['usage']
   assert response['choices'][0]['finish_reason']=='stop';assert user['context']==context and user['history']==prior
   if role!='critic':assert user['baseline']==baseline.groupby('method').auc.mean().round(5).to_dict()
   else:assert user['candidates']==record['proposed']
   serialized=json.dumps(request)
   assert 'sk-or-' not in serialized and 'Authorization' not in serialized and '/Users/' not in serialized
   assert not any(str(pid) in serialized for pid in source.participant_id)
  candidates+=record['approved'];prior.append(json.loads((P/f'round{r}_feedback.json').read_text()))
  scores=pd.concat([pd.read_csv(L/'discovery'/f"{f['name']}.csv") for f in record['approved']],ignore_index=True)
  assert feedback(scores,baseline)==prior[-1]['feedback']
 assert frozen['agent_candidates_scored']==len(candidates)==frozen['automatic_candidates_scored']
 controls=automatic_candidates()[:len(candidates)]
 for arm,fs in [('multiagent',candidates),('automatic',controls)]:
  saved=pd.read_csv(P/f'{arm}_discovery_scores.csv');pieces=pd.concat([pd.read_csv(L/'discovery'/f"{f['name']}.csv") for f in fs],ignore_index=True)
  pd.testing.assert_frame_equal(saved,pieces)
  info=feedback(saved,baseline);eligible=sorted([r for r in info if r['eligible']],key=lambda r:(-r['gain'],r['candidate']))[:2];names=[r['candidate'] for r in eligible]
  decision=next(v for v in frozen['decisions'] if v['arm']==arm)
  assert [f['name'] for f in decision['proposed']]==names
  if names:
   bs=pd.read_csv(L/'discovery'/f'{arm}_bundle.csv');r=feedback(bs,baseline)[0]
   assert r==decision['score'] and bool(r['eligible'])==decision['accepted']
   assert frozen['arms'][arm]==(decision['proposed'] if r['eligible'] else [])
  else:assert frozen['arms'][arm]==[] and not decision['accepted']
 for f in (L/'discovery').glob('*.json'):
  s=json.loads(f.read_text());assert s['spec']['signature']==signature() and s['output_sha256']==sha256(f.with_suffix('.csv'))
 extra=[ROOT/'research/agent_auc_followup.py',ROOT/'research/common.py',ROOT/'paper_ready/private/agent_extension/split_manifest.json',ROOT/'paper_ready/FEATURE_DICTIONARY.md',ROOT/'paper_ready/agent_extension/frozen_features.json',ROOT/'paper_ready/agent_auc_followup/round1_proposal.json',ROOT/'paper_ready/agent_auc_followup/round2_proposal.json']
 result=dict(status='passed',n_discovery=210,n_evaluation=492,api_calls=len(ledger),reported_cost_usd=sum(float(r['usage'].get('cost',0)) for r in ledger),reserved_usd=sum(r['reserved_usd'] for r in ledger),candidate_count=len(candidates),frozen_sha256=sha256(P/'frozen.json'),discovery_hashes={str(p.relative_to(L)):sha256(p) for p in (L/'discovery').glob('*') if p.is_file()},additional_dependency_hashes={str(f.relative_to(ROOT)):sha256(f) for f in extra},execution_note='Uninterrupted first trajectory; cached API resume path was not used. Resume-hardening issues found by independent review are recorded; this audit verifies actual requests/providers/context, frozen choices and private discovery hashes before evaluation.')
 write_json(P/'pre_evaluation_audit.json',result);print('Pre-evaluation audit passed; reported API cost',result['reported_cost_usd'],flush=True)

def replay():
 a=json.loads((P/'pre_evaluation_audit.json').read_text());assert a['frozen_sha256']==sha256(P/'frozen.json')
 assert all(sha256(L/f)==h for f,h in a['discovery_hashes'].items()) and all(sha256(ROOT/f)==h for f,h in a['additional_dependency_hashes'].items())
 done=json.loads((L/'complete.json').read_text());assert done['signature']==signature() and done['frozen_sha256']==sha256(P/'frozen.json')
 assert all(sha256(L/f)==h for f,h in done['outputs'].items())
 frozen=json.loads((P/'frozen.json').read_text());d=load('evaluation');y=d.y.to_numpy();g=d.pair_id.to_numpy();pred=pd.read_parquet(L/'predictions.parquet');assert len(pred)==492*3*8*4
 assert pred.groupby(['arm','method','policy','row_index']).size().eq(1).all();nmodels=0;nboundaries=0;maxerror=0.
 for fold,(tr,te) in enumerate(splits(y,g,n=5,seed=42)):
  for arm,fs in frozen['arms'].items():
   x,_=augment(d,feature_sets(d)['full'],fs);xx=x.to_numpy(float);path=L/f'fold{fold}_{arm}';info=json.loads((path/'details.json').read_text());inner=np.load(path/'inner.npz')
   np.testing.assert_array_equal(inner['tr'],tr);np.testing.assert_array_equal(inner['te'],te);assert info['columns']==list(x)
   innerfolds=splits(y[tr],g[tr],n=3,seed=143)
   for b in info['inner_boundaries']:
    it,iv=innerfolds[b['fold']];assert b['train']==it.tolist() and b['validation']==iv.tolist();assert not set(g[tr][it])&set(g[tr][iv]);nboundaries+=1
    med=np.nanmedian(xx[tr][it],axis=0);np.testing.assert_allclose(med,b['medians']);z=np.where(np.isnan(xx[tr][it]),med,xx[tr][it]);np.testing.assert_allclose(np.median(z,axis=0),b['center'])
    scale=np.diff(np.percentile(z,[25,75],axis=0),axis=0).ravel();scale[scale<10*np.finfo(scale.dtype).eps]=1;np.testing.assert_allclose(scale,b['scale'])
   q={};aucs={}
   for method in models():
    est=joblib.load(path/f'{method}.joblib');nmodels+=1;np.testing.assert_allclose(est[0].statistics_,np.nanmedian(xx[tr],axis=0));z=est[0].transform(xx[tr]);np.testing.assert_allclose(est[1].center_,np.median(z,axis=0))
    scale=np.diff(np.percentile(z,[25,75],axis=0),axis=0).ravel();scale[scale<10*np.finfo(scale.dtype).eps]=1;np.testing.assert_allclose(scale,est[1].scale_)
    q[method]=est.predict_proba(xx[te])[:,1];aucs[method]=[roc_auc_score(y[tr][iv],inner[method][iv]) for it,iv in innerfolds];np.testing.assert_allclose(aucs[method],info['inner_auc'][method])
   winner=sorted(aucs,key=lambda k:(-np.mean(aucs[k]),k))[0];assert winner==info['winner'];q['equal_ensemble']=np.mean(list(q.values()),axis=0);q['inner_selected']=q[winner]
   np.testing.assert_allclose(inner['equal_ensemble'],np.mean([inner[k] for k in models()],axis=0));np.testing.assert_allclose(inner['inner_selected'],inner[winner])
   for method,p in q.items():
    cuts=select_thresholds(y[tr],inner[method]);assert cuts==info['thresholds'][method]
    for policy,tau in cuts.items():
     v=pred[(pred.arm==arm)&(pred.method==method)&(pred.policy==policy)&(pred.fold==fold)].set_index('row_index').loc[te]
     np.testing.assert_allclose(p,v.probability,atol=1e-12,rtol=0);np.testing.assert_array_equal(v.y,y[te]);np.testing.assert_allclose(v.threshold,tau);np.testing.assert_array_equal(v.prediction,p>=tau);maxerror=max(maxerror,float(np.max(abs(p-v.probability.to_numpy()))))
  print('Final multi-agent replay fold',fold+1,'passed',flush=True)
 # No-feature fallbacks must be exact controls, including all thresholds.
 for arm,fs in frozen['arms'].items():
  if arm!='raw' and not fs:
   a=pred[pred.arm==arm].drop(columns='arm').sort_values(['fold','method','policy','row_index']).reset_index(drop=True);b=pred[pred.arm=='raw'].drop(columns='arm').sort_values(['fold','method','policy','row_index']).reset_index(drop=True);pd.testing.assert_frame_equal(a,b)
 write_json(P/'replay_verification.json',dict(status='passed',outer_model_replays=nmodels,inner_boundary_checks=nboundaries,prediction_rows=len(pred),probability_max_error=maxerror,report_source_sha256=sha256(__file__)))
 return d,pred

def report():
 d,pred=replay();rows=[];foldrows=[]
 for (arm,method,policy),v in pred.groupby(['arm','method','policy']):
  auc=[]
  for fold,t in v.groupby('fold'):
   m=allmetrics(t.y,t.probability,t.prediction);auc.append(m['auc']);foldrows.append(dict(arm=arm,method=method,policy=policy,fold=fold,**m))
  rows.append(dict(arm=arm,method=method,policy=policy,mean_fold_auc=np.mean(auc),**allmetrics(v.y,v.probability,v.prediction)))
 out=pd.DataFrame(rows);out.to_csv(P/'metrics_all.csv',index=False);pd.DataFrame(foldrows).to_csv(P/'fold_metrics.csv',index=False)
 y=d.y.to_numpy();groups=d.pair_id.to_numpy();folds=[te for tr,te in splits(y,groups,n=5,seed=42)];units=[np.array([np.flatnonzero(groups==pair) for pair in np.unique(groups[ix])]) for ix in folds];rng=np.random.default_rng(20261120)
 probs={}
 for (arm,method),v in pred[pred.policy=='default'].groupby(['arm','method']):
  v=v.sort_values('row_index');np.testing.assert_array_equal(v.row_index,np.arange(len(y)));probs[(arm,method)]=v.probability.to_numpy()
 boot={key:np.empty((2000,2)) for key in probs}
 for r in range(2000):
  ix=[u[rng.integers(0,len(u),len(u))].ravel() for u in units];pool=np.concatenate(ix)
  for k,p in probs.items():boot[k][r]=[np.mean([rank_auc(y[i],p[i]) for i in ix]),rank_auc(y[pool],p[pool])]
 effects=[]
 for method in list(models())+['equal_ensemble','inner_selected']:
  for a,b in [('multiagent','raw'),('multiagent','automatic'),('automatic','raw')]:
   p,q=probs[(a,method)],probs[(b,method)]
   for j,metric in enumerate(['mean_fold_auc','pooled_auc']):
    value=np.mean([rank_auc(y[ix],p[ix])-rank_auc(y[ix],q[ix]) for ix in folds]) if j==0 else rank_auc(y,p)-rank_auc(y,q)
    lo,hi=np.percentile(boot[(a,method)][:,j]-boot[(b,method)][:,j],[2.5,97.5]);effects.append(dict(method=method,comparison=a+'_minus_'+b,metric=metric,estimate=value,ci_low=lo,ci_high=hi))
 pd.DataFrame(effects).to_csv(P/'paired_auc_effects.csv',index=False)
 alerts=[]
 for f in (L/'discovery').glob('*.json'):
  for a in json.loads(f.read_text())['warnings']:alerts.append(dict(stage='discovery',job=f.stem,**a))
 for f in L.glob('fold*/details.json'):
  for method,ws in json.loads(f.read_text())['warnings'].items():
   for w in ws:alerts.append(dict(stage='evaluation',job=f.parent.name,method=method,**w))
 write_json(P/'fit_warnings.json',alerts)
 print(out[(out.method=='equal_ensemble')&(out.policy=='default')][['arm','mean_fold_auc','auc','sensitivity','specificity','brier']].to_string(index=False),flush=True)
 print(pd.DataFrame(effects).query("method=='equal_ensemble'").to_string(index=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','report']);a=p.parse_args();globals()[a.stage]()
