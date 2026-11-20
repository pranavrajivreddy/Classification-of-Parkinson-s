"""Resume-safe nested evaluation. Run only after freezing ANALYSIS_PLAN.md."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from research.common import OUT,PRIVATE,sha256,write_json,manifest,fingerprint
from research.core import engineered,feature_sets,splits,fit_fold,select_thresholds,metrics,POLICIES,oof


def save_predictions(records,df,test,y,p,train_y,train_p,method,seed,fold,config,opt_seed=None,override=None):
    policies=select_thresholds(train_y,train_p)
    if override is not None:policies=override
    for policy,tau in policies.items():
        for j,i in enumerate(test):
            records.append(dict(participant_id=df.iloc[i].participant_id,pair_id=df.iloc[i].pair_id,
                y=int(y[i]),probability=float(p[j]),threshold=float(tau),prediction=int(p[j]>=tau),
                outer_seed=seed,fold=fold,configuration=config,method=method,optimizer_seed=opt_seed,policy=policy))


def run_job(job):
    # Version-locked legacy SVC calibration remains training-only. Keep numerical
    # convergence warnings visible; suppress only the known API deprecation.
    warnings.filterwarnings('ignore',message='The `probability` parameter was deprecated',category=FutureWarning)
    config,seed,evo=job
    if evo: raise ValueError('Evolutionary experiments are excluded from this paper release')
    name=f'{config}_seed{seed}';out=PRIVATE/'runs'/name
    out.mkdir(parents=True,exist_ok=True)
    cohort_path=PRIVATE/'cohort.parquet';data_hash=sha256(cohort_path)
    # Source-specific hash avoids invalidating completed tasks for unrelated report edits.
    code={k:v for k,v in fingerprint().items() if Path(k).name in ['run.py','core.py']}
    signature=dict(cohort_sha256=data_hash,code=code,config=config,seed=seed,evolution=evo)
    done=out/'complete.json'
    if done.exists() and json.loads(done.read_text()).get('signature')==signature:
        return name+' cached'
    df=engineered(pd.read_parquet(cohort_path));sets=feature_sets(df)
    if config=='engineered':cols=sets['full']+[c for c in df if c.startswith('eng_')]
    elif config=='engagement_control':cols=['tapping_sessions_total','walking_sessions_total','voice_sessions_total']
    elif config=='missingness_control':
        cols=sets['full'];df=df.assign(**{c:df[c].isna().astype(float) for c in cols})
    elif config=='device_control':cols=['tapping_device','walking_device','voice_device']
    else:cols=sets[config]
    X=df[cols].to_numpy(dtype=str if config=='device_control' else float)
    y=df.y.to_numpy(int);groups=df.pair_id.to_numpy();records=[];selections=[];partitions=[];front_records=[]
    outer=splits(y,groups,n=5,seed=seed)
    start=time.time()
    for fold,(tr,te) in enumerate(outer):
        print(f'{name} fold {fold+1}/5',flush=True)
        for i in te:partitions.append(dict(participant_id=df.iloc[i].participant_id,pair_id=df.iloc[i].pair_id,fold=fold,outer_seed=seed))
        if config=='device_control':
            inner=splits(y[tr],groups[tr],seed=seed+101);p=np.zeros(len(tr))
            def make():return make_pipeline(OneHotEncoder(handle_unknown='ignore'),LogisticRegression(C=1,max_iter=3000))
            for a,b in inner:p[b]=make().fit(X[tr][a],y[tr][a]).predict_proba(X[tr][b])[:,1]
            q=make().fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
            preds={'selected':(p,q)};sel={'selected_model':'onehot_lr'}
        else:
            preds,sel,P,Q,inner=fit_fold(X[tr],y[tr],groups[tr],X[te],seed)
            np.savez_compressed(out/f'fold{fold}_meta.npz',train_indices=tr,test_indices=te,
                                inner_probabilities=P,outer_probabilities=Q,train_labels=y[tr])
        sel.update(fold=fold,features=cols,train_n=len(tr),test_n=len(te),outer_seed=seed)
        selections.append(sel)
        for method,(p,q) in preds.items():save_predictions(records,df,te,y,q,y[tr],p,method,seed,fold,config)
    pd.DataFrame(records).to_parquet(out/'predictions.parquet',index=False)
    pd.DataFrame(partitions).to_csv(out/'fold_assignments.csv',index=False)
    if front_records:pd.DataFrame(front_records).to_csv(out/'frontier_outer.csv',index=False)
    write_json(out/'selections.json',selections)
    write_json(done,manifest(dict(signature=signature,elapsed_sec=time.time()-start,
                                  predictions_sha256=sha256(out/'predictions.parquet'))))
    return name+f' completed in {time.time()-start:.1f}s'


def main(workers=4,stage='main'):
    if stage=='main':
        jobs=[(c,42,False) for c in ['demographics','tapping','walking','voice','tap_walk','tap_voice','walk_voice','sensors','full','engineered','engagement_control','device_control','missingness_control']]
    elif stage=='repeats':jobs=[(c,s,False) for s in [137,2027] for c in ['full','engineered','tapping','sensors']]
    else:jobs=[('demographics',42,False)]
    write_json(OUT/f'run_{stage}_manifest.json',manifest(dict(jobs=jobs,analysis_plan_sha256=sha256(OUT/'ANALYSIS_PLAN.md'))))
    failures=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        tasks={ex.submit(run_job,j):j for j in jobs}
        for future in as_completed(tasks):
            try:print(future.result(),flush=True)
            except Exception as exc:
                import traceback
                traceback.print_exc();failures.append(dict(job=tasks[future],error=repr(exc)))
    write_json(OUT/f'run_{stage}_failures.json',failures)
    if failures:raise RuntimeError(f'{len(failures)} failed jobs; results are incomplete')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--stage',choices=['main','repeats','smoke'],default='main')
    args=parser.parse_args();main(args.workers,args.stage)
