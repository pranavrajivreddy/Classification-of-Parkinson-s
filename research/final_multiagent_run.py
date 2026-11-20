"""Two proposing API roles + critic; all fitting remains local."""
import argparse,json,getpass,ast,os
import numpy as np
import pandas as pd
from research.common import ROOT,sha256,write_json
from research.final_multiagent_core import P,L,signature,load,feature_sets,validate,score,feedback,choose,automatic_candidates,evaluate,learning_curves
from research.final_multiagent_api import Client

SYSTEM='''You are one role in a bounded scientific feature-engineering experiment.
Return valid JSON only. Never claim clinical causality, diagnosis or verified improvement.
No external tools/code execution. Inputs are feature definitions and aggregate development
metrics; you do not have patient records. Propose formulas only in the permitted language.
Do not change outcome labels, cohorts, folds, models, selection criteria or stopping rule.
All thresholds/model evaluation happen locally. Favor measurement-grounded parsimonious
features; mathematical transformations add no new measured information.'''

def schema_context():
 text=(ROOT/'paper_ready/FEATURE_DICTIONARY.md').read_text();allowed=feature_sets(load('discovery'))['sensors']
 definitions=[line for line in text.splitlines() if any(line.startswith('| '+c+' |') for c in allowed)]
 old=[]
 for file in [ROOT/'paper_ready/agent_extension/frozen_features.json',ROOT/'paper_ready/agent_auc_followup/round1_proposal.json',ROOT/'paper_ready/agent_auc_followup/round2_proposal.json']:
  j=json.loads(file.read_text());old.extend([f['formula'] for f in j.get('features',[])])
 return dict(definitions=definitions,allowed_inputs=allowed,prior_formulas=old,functions='pdiv(a,b): divide by b with signed absolute denominator floor1e-4; logabs(a)=log(1+abs(a)); sqrtabs(a)=sqrt(abs(a)); abs(a). Operators +,-,*. No /,**,indexing,attribute access or other function calls. At most80 AST nodes; constants absolute value<=100.',constraints='Only sensor inputs listed. No demographics, medication, identity, disease labels, quality/engagement totals. walking speed=cadence*step_length/60, so do not propose equivalent rearrangements. Tapping flight means inter-event interval, not airborne finger time. Jitter/shimmer are percent. Sensor sessions are co-registered by person, not synchronous.',evaluation='Six fixed families LR/RF/XGBoost/LightGBM/CatBoost/MLP. Discovery only210 participants,2x3 pair-grouped CV. Objective equal-probability ensemble mean fold AUC; gain>=.002 positive both repeats and >=4/6family gains nonnegative; max2selected then bundle check.',note='Prior project outcomes were exposed to investigators. This is exploratory. Avoid repeating prior formulas; propose plausible new representations rather than unsupported biomarkers.')

def discover(key):
 P.mkdir(parents=True,exist_ok=True);L.mkdir(parents=True,exist_ok=True)
 frozen=P/'frozen.json'
 if frozen.exists():assert json.loads(frozen.read_text())['signature']==signature();return
 start=P/'start_manifest.json'
 if start.exists():assert json.loads(start.read_text())['signature']==signature()
 else:write_json(start,dict(signature=signature(),note='Before API proposals and new fitting.'))
 control=automatic_candidates();write_json(P/'automatic_sequence.json',control)
 context=schema_context();write_json(P/'api_context.json',context)
 client=Client(key);allfeatures=[];allscores=[];seen=set();history=[];baseline=score('baseline',[])
 write_json(P/'baseline_discovery_summary.json',baseline.groupby('method')[['auc','brier','average_precision']].mean().round(5).to_dict('index'))
 try:
  for r in [1,2]:
   roundpath=P/f'round{r}_proposals.json'
   if roundpath.exists():
    record=json.loads(roundpath.read_text());candidates=record['approved'];seen.update(ast.dump(ast.parse(f['formula'],mode='eval'),include_attributes=False) for f in candidates)
   else:
    candidates=[];rejected=[]
    for role,description in [('measurement','Within-modality measurement scientist: robust dimensionless or interpretable transforms grounded in tapping, walking or phonation measurement. Avoid identities and fragile ratios.'),('integration','Cross-modal statistical scientist: complement correlated summaries using stable interactions or asymmetry descriptors. Units are descriptive, not biological mechanism claims.')]:
     request=dict(context=context,round=r,role=description,baseline=baseline.groupby('method').auc.mean().round(5).to_dict(),history=history,instruction='Propose at most4 NEW distinct features. JSON object with features array, each {formula: string, rationale: short string}. No executable code. Empty array is acceptable if no defensible proposal.')
     answer=client.call(f'round{r}_{role}',SYSTEM,request);fs=answer.get('features',[]);assert isinstance(fs,list) and len(fs)<=4
     for i,f in enumerate(fs):f['name']=f'agent_r{r}_{role}_{i}'
     valid,bad,seen=validate(fs,context['allowed_inputs'],seen);candidates+=valid;rejected+=bad
    answer=client.call(f'round{r}_critic',SYSTEM,dict(context=context,role='Independent skeptical reviewer. Veto exact algebraic redundancy with existing inputs, indefensible measurement meaning, unstable denominator reasoning or unsupported physiological claims. Do not veto purely because an empirical benefit is uncertain. You cannot modify/add formulas.',candidates=candidates,history=history,instruction='JSON object with decisions array, exactly one per supplied candidate: {name:string, approve:boolean, reason:short string}.'))
    decisions=answer.get('decisions',[]);assert len(decisions)==len(candidates) and {v['name'] for v in decisions}=={f['name'] for f in candidates} and all(type(v['approve']) is bool for v in decisions)
    keep={v['name'] for v in decisions if v['approve']};approved=[f for f in candidates if f['name'] in keep]
    record=dict(proposed=candidates,invalid=rejected,critic=decisions,approved=approved);write_json(roundpath,record);candidates=approved
   scores=[score(f['name'],[f]) for f in candidates];allfeatures+=candidates;allscores+=scores
   summary=feedback(pd.concat(scores,ignore_index=True),baseline) if scores else []
   history.append(dict(round=r,proposals=record['proposed'],feedback=summary,critic=record['critic']));write_json(P/f'round{r}_feedback.json',history[-1]);print('API round',r,'approved',len(candidates),flush=True)
 finally:client.close()
 controls=control[:len(allfeatures)];control_scores=[score(f['name'],[f]) for f in controls]
 decisions=[];arms={'raw':[]}
 for arm,fs,rows in [('multiagent',allfeatures,allscores),('automatic',controls,control_scores)]:
  scores=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame();scores.to_csv(P/f'{arm}_discovery_scores.csv',index=False)
  arms[arm],decision=choose(fs,scores,baseline,arm);decisions.append(decision)
  write_json(P/f'{arm}_candidate_summary.json',feedback(scores,baseline) if rows else [])
 provenance={str(p.relative_to(P)):sha256(p) for p in P.rglob('*') if p.is_file() and p!=frozen}
 write_json(frozen,dict(signature=signature(),arms=arms,decisions=decisions,agent_candidates_scored=len(allfeatures),automatic_candidates_scored=len(controls),provenance=provenance,scope='Exploratory prior-exposed evaluation; one API trajectory; no independent multiagent superiority claim.'))
 print('Discovery frozen. Selected counts:',{k:len(v) for k,v in arms.items()},flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['discover','evaluate','all','curves']);arg=parser.parse_args().stage
 if arg in ['discover','all']:
  if not (P/'frozen.json').exists():
   key=os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key (hidden, memory only): ')
   try:discover(key)
   finally:key=None
 if arg in ['evaluate','all']:evaluate()
 if arg in ['curves','all']:learning_curves()
if __name__=='__main__':main()
