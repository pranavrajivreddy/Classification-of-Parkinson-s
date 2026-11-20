"""Prediction and preprocessing replay adapted from the prior verified panel."""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from research.common import ROOT,sha256,write_json
from research.claude_prompt_core import P,L,signature,load,models,feature_sets
from research.agent_features import augment
from research.core import splits,select_thresholds

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
  print('Single-Claude replay fold',fold+1,'passed',flush=True)
 # No-feature fallbacks must be exact controls, including all thresholds.
 for arm,fs in frozen['arms'].items():
  if arm!='raw' and not fs:
   a=pred[pred.arm==arm].drop(columns='arm').sort_values(['fold','method','policy','row_index']).reset_index(drop=True);b=pred[pred.arm=='raw'].drop(columns='arm').sort_values(['fold','method','policy','row_index']).reset_index(drop=True);pd.testing.assert_frame_equal(a,b)
 write_json(P/'replay_verification.json',dict(status='passed',outer_model_replays=nmodels,inner_boundary_checks=nboundaries,prediction_rows=len(pred),probability_max_error=maxerror,report_source_sha256=sha256(__file__)))
 return d,pred
