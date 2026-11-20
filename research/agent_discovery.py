"""Discovery-only controller I/O. No evaluation data loaded after prepare()."""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from research.common import OUT,PRIVATE,sha256,write_json,manifest
from research.core import feature_sets,splits,oof,select_thresholds,metrics
from research.agent_features import augment,parsed

PUBLIC=OUT/'agent_extension';LOCAL=PRIVATE/'agent_extension'


def prepare():
    LOCAL.mkdir(exist_ok=True,parents=True);PUBLIC.mkdir(exist_ok=True)
    if (LOCAL/'split_manifest.json').exists():raise RuntimeError('Split already exists; do not silently redefine it')
    d=pd.read_parquet(PRIVATE/'cohort.parquet');pairs=np.sort(d.pair_id.unique())
    chosen=np.random.default_rng(20261118).choice(pairs,105,replace=False)
    discovery=d[d.pair_id.isin(chosen)].copy();evaluation=d[~d.pair_id.isin(chosen)].copy()
    discovery.to_parquet(LOCAL/'discovery.parquet',index=False);evaluation.to_parquet(LOCAL/'evaluation.parquet',index=False)
    assert not set(discovery.participant_id)&set(evaluation.participant_id)
    write_json(LOCAL/'split_manifest.json',manifest(dict(seed=20261118,discovery_n=len(discovery),evaluation_n=len(evaluation),
        discovery_sha256=sha256(LOCAL/'discovery.parquet'),evaluation_sha256=sha256(LOCAL/'evaluation.parquet'),
        source_cohort_sha256=sha256(PRIVATE/'cohort.parquet'),protocol_sha256=sha256(PUBLIC/'PROTOCOL.md'))))
    print('Discovery=210, evaluation=492; matched pairs disjoint.')


def load():
    p=LOCAL/'discovery.parquet';m=json.loads((LOCAL/'split_manifest.json').read_text())
    assert sha256(p)==m['discovery_sha256']
    return pd.read_parquet(p)


def diagnostics():
    d=load();rows=[]
    for c in feature_sets(d)['full']:
        a=d.loc[d.y==1,c];b=d.loc[d.y==0,c]
        sd=np.sqrt((a.var()+b.var())/2)
        rows.append(dict(feature=c,case_mean=float(a.mean()),control_mean=float(b.mean()),
            smd=float((a.mean()-b.mean())/sd) if sd>0 else 0.,
            rho=float(spearmanr(d[c],d.y,nan_policy='omit').statistic)))
    t=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan).fillna(0).sort_values('rho',key=abs,ascending=False)
    t.to_csv(LOCAL/'discovery_diagnostics.csv',index=False)
    prompt=dict(controller='Current ChatGPT/Codex conversation; backend model ID not independently exported',
        task='Propose at most 12 transparent formulas over these measured inputs. Use no biological validation claims. Two rounds only; see PROTOCOL.md.',
        n=210,case_n=105,control_n=105,allowed_functions=['pdiv','logabs','sqrtabs','abs'],
        feature_diagnostics=t.round(5).to_dict('records'))
    write_json(LOCAL/'initial_prompt.json',prompt)
    print(json.dumps(prompt,indent=2))


def score(round_number):
    if (LOCAL/'frozen_features.json').exists():raise RuntimeError('Discovery already frozen')
    d=load();proposal=json.loads((LOCAL/f'round{round_number}_proposal.json').read_text())
    features=proposal['features'];assert 0<len(features)<=12
    allowed=feature_sets(d)['full']
    for item in features:
        _,deps=parsed(item['formula'],set(allowed));item['inputs']=sorted(deps)
    y=d.y.to_numpy(int);folds=splits(y,d.pair_id.to_numpy(),n=3,seed=4242)
    rows=[]
    for config,cols in feature_sets(d).items():
        choices=[('baseline',[])]
        usable=[f for f in features if set(f['inputs']).issubset(cols)]
        choices.extend([(f['name'],[f]) for f in usable])
        if len(usable)>1:choices.append(('bundle',usable))
        for name,fs in choices:
            x,_=augment(d,cols,fs);p=oof(x.to_numpy(float),y,folds,LogisticRegression(C=1,max_iter=3000))
            tau=select_thresholds(y,p)['utility'];m=metrics(y,p,p>=tau)
            rows.append(dict(configuration=config,candidate=name,n_added=len(fs),threshold=tau,**m))
    result=pd.DataFrame(rows);result.to_csv(LOCAL/f'round{round_number}_feedback.csv',index=False)
    write_json(LOCAL/f'round{round_number}_evaluation_manifest.json',manifest(dict(
        discovery_sha256=sha256(LOCAL/'discovery.parquet'),proposal_sha256=sha256(LOCAL/f'round{round_number}_proposal.json'),
        feedback_sha256=sha256(LOCAL/f'round{round_number}_feedback.csv'),interpretation='Adaptive development feedback, not validation.')))
    print(result[['configuration','candidate','auc','sensitivity','specificity','utility']].round(4).to_string(index=False))


def freeze():
    path=LOCAL/'frozen_features.json'
    if path.exists():raise RuntimeError('Already frozen')
    spec=json.loads((LOCAL/'final_decision.json').read_text())
    assert spec['controller']=='ChatGPT/Codex current conversation'
    allowed=feature_sets(load())['full'];seen=set()
    for item in spec['features']:
        assert item['name'] not in seen;seen.add(item['name'])
        _,deps=parsed(item['formula'],set(allowed));item['inputs']=sorted(deps)
    for config,cols in feature_sets(load()).items():
        names=spec['configuration_features'][config]
        assert len(names)<=1 and set(names).issubset(seen)
        assert all(set(f['inputs']).issubset(cols) for f in spec['features'] if f['name'] in names)
    spec['provenance']=manifest(dict(discovery_sha256=sha256(LOCAL/'discovery.parquet'),
        protocol_sha256=sha256(PUBLIC/'PROTOCOL.md'),history_hashes={p.name:sha256(p) for p in sorted(LOCAL.glob('round*'))}))
    write_json(path,spec);write_json(PUBLIC/'frozen_features.json',spec)
    print('Features frozen before new evaluation. SHA256:',sha256(path))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','diagnostics','score','freeze']);p.add_argument('--round',type=int,choices=[1,2],default=1)
    a=p.parse_args();score(a.round) if a.action=='score' else globals()[a.action]()
