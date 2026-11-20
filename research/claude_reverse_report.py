"""Audit a single-agent trajectory and report every fixed-panel comparison."""
import argparse,ast,json,re
import numpy as np
import pandas as pd
from research.common import ROOT,sha256,write_json
from research.claude_reverse_core import P,L,signature,load,models,feature_sets,feedback,automatic_candidates,validate
from research.claude_reverse_run import schema_context,metric_summary,canonical,SYSTEM
from research.claude_reverse_api import payload,MODEL,PROVIDER,CAP
from research.claude_reverse_replay import replay
from research.reporting_metrics import allmetrics
from research.agent_auc_followup import rank_auc
from research.core import splits

def equivalent(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equivalent(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):equivalent(x,y)
 elif type(a) is bool or isinstance(a,str) or a is None:assert a==b
 else:np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12,equal_nan=True)

def audit():
 frozen=json.loads((P/'frozen.json').read_text());assert frozen['signature']==signature()
 assert json.loads((P/'start_manifest.json').read_text())['signature']==signature()
 assert all(sha256(P/name)==h for name,h in frozen['provenance'].items())
 d,e=load('discovery'),load('evaluation');assert len(d)==492 and len(e)==210
 assert not set(d.participant_id)&set(e.participant_id) and not set(d.pair_id)&set(e.pair_id)
 source=pd.read_parquet(ROOT/'paper_ready/private/cohort.parquet')
 pd.testing.assert_frame_equal(pd.concat([d,e]).sort_values('participant_id').reset_index(drop=True),source.sort_values('participant_id').reset_index(drop=True))
 context=json.loads((P/'api_context.json').read_text());assert context==schema_context()
 ledger=json.loads((P/'api_ledger.json').read_text());assert len(ledger)==3 and len({v['name'] for v in ledger})==3
 assert sum(v['reserved_usd'] for v in ledger)<=CAP
 baseline=pd.read_csv(L/'discovery/baseline.csv');prior=[];features=[]
 seen={canonical(f) for f in context['prior_formulas']}
 for r in [1,2,3]:
  name=f'round{r}_single';entry=next(v for v in ledger if v['name']==name)
  request=json.loads((P/'api'/f'{name}_request.json').read_text());response=json.loads((P/'api'/f'{name}_response.json').read_text())
  user=json.loads(request['messages'][1]['content']);assert request['messages'][0]['content']==SYSTEM
  expected,reserve=payload(SYSTEM,user);assert expected==request and reserve==entry['reserved_usd']
  assert entry['status']=='completed' and entry['http_status']==200 and entry['provider']==PROVIDER
  assert response['model']==MODEL and response['provider']==PROVIDER and response['usage']==entry['usage']
  assert response['choices'][0]['finish_reason']=='stop' and 0<=entry['usage']['cost']<=reserve
  assert user['history']==prior and user['context']==context and user['round']==r and user['total_rounds']==3
  equivalent(user['baseline'],metric_summary(baseline))
  serialized=json.dumps(request);assert '/Users/' not in serialized and 'sk-or-' not in serialized and 'Authorization' not in serialized
  assert not any(str(pid) in serialized for pid in source.participant_id)
  answer=json.loads(response['choices'][0]['message']['content']);proposed=[dict(f,name=f'agent_claude_r{r}_{i}') for i,f in enumerate(answer['features'])]
  good,bad,seen=validate(proposed,context['allowed_inputs'],seen)
  record=json.loads((P/f'round{r}_proposals.json').read_text());assert record==dict(round=r,analysis=answer['analysis'],proposed=proposed,valid=good,invalid=bad)
  pieces=[pd.read_csv(L/'discovery'/f"{f['name']}.csv") for f in good]
  expected=dict(round=r,analysis=answer['analysis'],proposed=proposed,invalid=bad,
   feedback=feedback(pd.concat(pieces,ignore_index=True),baseline) if pieces else [],
   metrics={f['name']:metric_summary(s) for f,s in zip(good,pieces)})
  saved=json.loads((P/f'round{r}_feedback.json').read_text());equivalent(expected,saved);prior.append(saved);features+=good
 assert frozen['agent_candidates_scored']==len(features)==frozen['automatic_candidates_scored']<=12
 controls=automatic_candidates()[:len(features)]
 for arm,fs in [('claude',features),('automatic',controls)]:
  decision=next(v for v in frozen['decisions'] if v['arm']==arm)
  if not fs:
   assert frozen['arms'][arm]==[] and not decision['accepted'];continue
  saved=pd.read_csv(P/f'{arm}_discovery_scores.csv');pieces=pd.concat([pd.read_csv(L/'discovery'/f"{f['name']}.csv") for f in fs],ignore_index=True)
  pd.testing.assert_frame_equal(saved,pieces)
  info=feedback(saved,baseline);equivalent(info,json.loads((P/f'{arm}_candidate_summary.json').read_text()))
  eligible=sorted([v for v in info if v['eligible']],key=lambda v:(-v['gain'],v['candidate']))[:2]
  assert [v['candidate'] for v in eligible]==[f['name'] for f in decision['proposed']]
  if eligible:
   expected=feedback(pd.read_csv(L/'discovery'/f'{arm}_bundle.csv'),baseline)[0];equivalent(expected,decision['score'])
   assert expected['eligible']==decision['accepted']
   assert frozen['arms'][arm]==(decision['proposed'] if expected['eligible'] else [])
  else:assert not frozen['arms'][arm] and not decision['accepted']
 for p in (L/'discovery').glob('*.json'):
  stamp=json.loads(p.read_text());assert stamp['spec']['signature']==signature() and stamp['output_sha256']==sha256(p.with_suffix('.csv'))
 extra=[ROOT/'research/claude_reverse_report.py',ROOT/'research/claude_reverse_replay.py',ROOT/'research/test_claude_reverse.py']
 result=dict(status='passed',n_discovery=len(d),n_evaluation=len(e),cohort_union_n=len(source),
  api_calls=len(ledger),reported_cost_usd=sum(v['usage']['cost'] for v in ledger),reserved_usd=sum(v['reserved_usd'] for v in ledger),
  candidates_scored=len(features),frozen_sha256=sha256(P/'frozen.json'),
  discovery_hashes={str(p.relative_to(L)):sha256(p) for p in (L/'discovery').glob('*') if p.is_file()},
  additional_dependency_hashes={str(p.relative_to(ROOT)):sha256(p) for p in extra},
  privacy='Inspected payloads contain feature definitions and aggregate discovery metrics; no participant identifiers, records, raw signals, evaluation outcomes or credentials')
 write_json(P/'pre_evaluation_audit.json',result)
 print('Pre-evaluation audit passed:',{k:result[k] for k in ['n_discovery','n_evaluation','api_calls','reported_cost_usd','candidates_scored']},flush=True)

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
  for a,b in [('claude','raw'),('claude','automatic'),('automatic','raw')]:
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


def finish():
 pre=json.loads((P/'pre_evaluation_audit.json').read_text());assert pre['status']=='passed'
 frozen=json.loads((P/'frozen.json').read_text());assert frozen['signature']==signature()
 assert all(sha256(ROOT/p)==h for p,h in pre['additional_dependency_hashes'].items())
 pred=pd.read_parquet(L/'predictions.parquet');metrics=pd.read_csv(P/'metrics_all.csv').set_index(['arm','method','policy'])
 for key,v in pred.groupby(['arm','method','policy']):
  for name,value in allmetrics(v.y,v.probability,v.prediction).items():
   np.testing.assert_allclose(metrics.loc[key,name],value,atol=1e-12,equal_nan=True)
 assert len(metrics)==96 and len(pd.read_csv(P/'fold_metrics.csv'))==480
 warnings=json.loads((P/'fit_warnings.json').read_text())
 replay=json.loads((P/'replay_verification.json').read_text());assert replay['status']=='passed'
 assert replay['report_source_sha256']==sha256(ROOT/'research/claude_reverse_replay.py')
 # This partition is different: equality to the prior raw predictions is not expected.
 for path in list(P.rglob('*'))+list((ROOT/'research').glob('*claude_reverse*.py')):
  if path.is_file() and path.suffix in ['.json','.md','.csv','.py','.log']:
   assert not re.search(rb'sk-or-v1-[A-Za-z0-9]{20,}',path.read_bytes())
 write_json(P/'final_integrity.json',dict(status='passed',metric_groups=96,fold_groups=480,
  reversed_split_verified=True,historical_script_audit_included=False,
  outer_model_replays=replay['outer_model_replays'],inner_boundary_checks=replay['inner_boundary_checks'],
  prediction_rows=len(pred),max_replay_error=replay['probability_max_error'],fit_warning_count=len(warnings),
  reported_cost_usd=pre['reported_cost_usd'],reserved_usd=pre['reserved_usd'],
  scope='Completed exploratory trajectory verification; not external/clinical validation'))
 print('Final verification passed:96 metric groups,480 fold groups; reversed split and model replays verified.',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','report','finish']);args=p.parse_args();globals()[args.stage]()
