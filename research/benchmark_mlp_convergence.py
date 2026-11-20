"""Warning-triggered sensitivity analysis, separate from the frozen benchmark.

Keep selected hyperparameters; refit with a longer fixed iteration ceiling.
No new search or choice based on outer results. Not an independently tuned MLP.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research.benchmark_extension import PUBLIC,LOCAL,load_data,estimator,capture_fit,thresholds,tasks
from research.benchmark_report import expanded_metrics
from research.common import ROOT,sha256,write_json
from research.core import splits

CAP=4000


def work():
    return [t for t in tasks('benchmark') if t[4]=='mlp']


def run(task):
    _,cohort,config,arm,_,fold=task
    source=LOCAL/'benchmark'/f'{cohort}_{config}_{arm}_mlp'/f'fold{fold}'
    destination=LOCAL/'mlp_convergence'/f'{cohort}_{config}_{arm}'/f'fold{fold}'
    if (destination/'complete.json').exists():
        c=json.loads((destination/'complete.json').read_text())
        assert c['protocol_sha256']==sha256(PUBLIC/'mlp_convergence_protocol.json')
        for name,h in c['files'].items():assert sha256(destination/name)==h
        return {'task':task,'cached':True}
    assert (source/'complete.json').exists()
    start=time.time()
    d=json.loads((source/'details.json').read_text())
    params=d['details']['trials'][d['details']['chosen']]['params']
    df,X,columns,folds=load_data(cohort,config,arm)
    tr,te=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
    y=df.y.to_numpy(int);g=df.pair_id.to_numpy();ip=np.full(len(tr),np.nan)
    records=[];wc={}
    def fit(a,b):
        est=estimator('mlp',params,42+fold)
        est.set_params(mlpclassifier__max_iter=CAP)
        capture_fit(est,X[a],y[a],wc)
        m=est.named_steps['mlpclassifier']
        records.append(dict(n_train=len(a),iterations=int(m.n_iter_),at_cap=bool(m.n_iter_>=CAP)))
        return est.predict_proba(X[b])[:,1]
    with threadpool_limits(limits=1):
        for a,b in splits(y[tr],g[tr],n=3,seed=143):
            assert not set(g[tr[a]])&set(g[tr[b]])
            ip[b]=fit(tr[a],tr[b])
        op=fit(tr,te)
    assert not set(g[tr])&set(g[te]) and np.isfinite(ip).all() and np.isfinite(op).all()
    cuts=thresholds(y[tr],ip);rows=[]
    for policy,tau in cuts.items():
        q=df.iloc[te][['participant_id','pair_id','y']].copy()
        q['probability']=op;q['prediction']=(op>=tau).astype(int);q['threshold']=tau;q['policy']=policy;q['fold']=fold
        rows.append(q)
    it=df.iloc[tr][['participant_id','pair_id','y']].copy();it['probability']=ip
    destination.mkdir(parents=True,exist_ok=True)
    pd.concat(rows).to_parquet(destination/'predictions.parquet',index=False)
    it.to_parquet(destination/'training_predictions.parquet',index=False)
    write_json(destination/'details.json',dict(task=task,parameters=params,fit_records=records,warning_counts=wc,
        source_details_sha256=sha256(source/'details.json'),thresholds=cuts))
    write_json(destination/'complete.json',dict(protocol_sha256=sha256(PUBLIC/'mlp_convergence_protocol.json'),
        files={p.name:sha256(p) for p in destination.iterdir() if p.name!='complete.json'}))
    return dict(task=task,seconds=round(time.time()-start,1))


def report():
    allrows=[];fits=0;atcap=0
    for task in work():
        _,cohort,config,arm,_,fold=task
        folder=LOCAL/'mlp_convergence'/f'{cohort}_{config}_{arm}'/f'fold{fold}'
        done=json.loads((folder/'complete.json').read_text())
        assert done['protocol_sha256']==sha256(PUBLIC/'mlp_convergence_protocol.json')
        for name,h in done['files'].items():assert sha256(folder/name)==h
        d=json.loads((folder/'details.json').read_text());fits+=len(d['fit_records']);atcap+=sum(r['at_cap'] for r in d['fit_records'])
        source=LOCAL/'benchmark'/f'{cohort}_{config}_{arm}_mlp'/f'fold{fold}'/'details.json'
        assert sha256(source)==d['source_details_sha256']
        sd=json.loads(source.read_text());assert d['parameters']==sd['details']['trials'][sd['details']['chosen']]['params']
        p=pd.read_parquet(folder/'predictions.parquet');it=pd.read_parquet(folder/'training_predictions.parquet')
        df,_,_,folds=load_data(cohort,config,arm)
        assert set(p.participant_id)==set(df.loc[folds==fold,'participant_id'])
        assert set(it.participant_id)==set(df.loc[folds!=fold,'participant_id'])
        assert not set(p.pair_id)&set(it.pair_id)
        truth=df.set_index('participant_id')
        for frame in [p,it]:
            np.testing.assert_array_equal(frame.y,truth.loc[frame.participant_id,'y'])
            np.testing.assert_array_equal(frame.pair_id,truth.loc[frame.participant_id,'pair_id'])
            assert frame.probability.between(0,1).all()
        for policy,tau in thresholds(it.y.to_numpy(),it.probability.to_numpy()).items():
            q=p[p.policy==policy];np.testing.assert_allclose(q.threshold,tau,atol=1e-14,rtol=0)
            np.testing.assert_array_equal(q.prediction,q.probability>=tau)
        p['cohort']=cohort;p['configuration']=config;p['arm']=arm;allrows.append(p)
    pred=pd.concat(allrows,ignore_index=True);pred.to_parquet(LOCAL/'mlp_convergence_predictions.parquet',index=False)
    rows=[]
    for key,p in pred.groupby(['cohort','configuration','arm','policy']):
        assert p.participant_id.is_unique and len(p)==(702 if key[0]=='n702' else 492)
        rows.append(dict(zip(['cohort','configuration','arm','policy'],key))|expanded_metrics(p))
    assert len(rows)==135
    pd.DataFrame(rows).to_csv(PUBLIC/'mlp_convergence_metrics.csv',index=False)
    for config in ['demographics','walking','sensors']:
        p=pred[(pred.cohort=='n492')&(pred.configuration==config)]
        order=['policy','participant_id']
        np.testing.assert_array_equal(p[p.arm=='raw'].sort_values(order).probability,p[p.arm=='agent'].sort_values(order).probability)
    write_json(PUBLIC/'mlp_convergence_verification.json',dict(status='passed',outer_tasks=135,metric_groups=135,
        fits=fits,fits_at_iteration_ceiling=atcap,limit=CAP,interpretation='Same selected parameters, longer fit; no repeated hyperparameter search.'))
    print('Convergence sensitivity verified:',fits,'fits;',atcap,'at ceiling',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','report']);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    path=PUBLIC/'mlp_convergence_protocol.json'
    if a.action=='freeze':
        assert not path.exists()
        write_json(path,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),reason='Warning-triggered diagnostic: 30/45 selected N702 MLP final fits reached the original 400-iteration ceiling.',
            source_sha256=sha256(Path(__file__)),parent_protocol_sha256=sha256(PUBLIC/'protocol.json'),
            design='All 27 cohort/configuration/arm combinations, same five outer folds; retain selected eight-trial hyperparameters; repeat inner OOF and outer fit with max_iter=4000, no early stopping or new tuning; all five training-selected threshold policies.',
            limitation='Added after original MLP warnings and aggregate results were visible; sensitivity analysis, not an independent or fully retuned comparison. Keep original bounded results and do not select between budgets on outer outcomes.'))
        return
    d=json.loads(path.read_text());assert sha256(Path(__file__))==d['source_sha256'];assert sha256(PUBLIC/'protocol.json')==d['parent_protocol_sha256']
    if a.action=='report':return report()
    pending=[t for t in work() if (LOCAL/'benchmark'/f'{t[1]}_{t[2]}_{t[3]}_mlp'/f'fold{t[5]}'/'complete.json').exists()]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,t) for t in pending]
        for i,f in enumerate(as_completed(futures),1):print(json.dumps(dict(completed=i,available=len(pending),**f.result())),flush=True)


if __name__=='__main__':main()
