"""Inspect completed fits without selecting models or changing their protocol."""
import argparse
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from research.benchmark_extension import PUBLIC, LOCAL, load_data, thresholds
from research.common import sha256,write_json


def check(name, replay_limit):
    checked=0;replayed=0;preprocessing=0
    for done in sorted(q for kind in ['benchmark','calibration'] for q in (LOCAL/kind).glob('*/fold*/complete.json')):
        folder=done.parent
        c=json.loads(done.read_text())
        assert c['protocol_sha256']==sha256(PUBLIC/'protocol.json')
        for filename,digest in c['artifacts'].items():assert sha256(folder/filename)==digest
        kind,cohort,config,arm,family,fold=c['task']
        df,X,columns,folds=load_data(cohort,config,arm)
        tr,te=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        p=pd.read_parquet(folder/'predictions.parquet')
        it=pd.read_parquet(folder/'training_predictions.parquet')
        assert not set(p.pair_id)&set(it.pair_id)
        truth=df.set_index('participant_id')
        np.testing.assert_array_equal(p.y,truth.loc[p.participant_id,'y'])
        for mode,group in it.groupby('mode'):
            cuts=thresholds(group.y.to_numpy(),group.probability.to_numpy())
            for policy,tau in cuts.items():
                g=p[(p['mode']==mode)&(p.policy==policy)]
                np.testing.assert_allclose(g.threshold,tau,rtol=0,atol=1e-14)
                np.testing.assert_array_equal(g.prediction,g.probability>=tau)
        if kind=='benchmark' and replayed<replay_limit:
            est=joblib.load(folder/'model.joblib')
            imputer=est.named_steps['simpleimputer'];scaler=est.named_steps['robustscaler']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                medians=np.nanmedian(X[tr],axis=0)
            medians=np.nan_to_num(medians,nan=0.)
            np.testing.assert_allclose(imputer.statistics_,medians,rtol=0,atol=1e-12)
            np.testing.assert_allclose(scaler.center_,np.median(imputer.transform(X[tr]),axis=0),rtol=0,atol=1e-12)
            preprocessing+=1
            g=p[p.policy=='default'].set_index('participant_id').loc[df.iloc[te].participant_id]
            np.testing.assert_allclose(est.predict_proba(X[te])[:,1],g.probability,rtol=0,atol=1e-12)
            replayed+=1
        checked+=1
    result=dict(status='passed',completed_outer_tasks_checked=checked,models_replayed=replayed,
                train_only_preprocessors_verified=preprocessing)
    write_json(PUBLIC/f'{name}.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='checkpoint');p.add_argument('--replay-limit',type=int,default=6)
    a=p.parse_args();check(a.name,a.replay_limit)
