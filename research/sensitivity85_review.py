"""Post-hoc research question; thresholds selected only on saved training scores.

No model refitting, no selection by outer performance, no test threshold search.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.benchmark_extension import LOCAL,PUBLIC,FAMILIES,MODES
from research.benchmark_report import expanded_metrics
from research.common import ROOT,PRIVATE,sha256,write_json
from research.core import operating_table

OUT=ROOT/'paper_ready/sensitivity85_review'
PVT=PRIVATE/'sensitivity85_review'


def main():
    OUT.mkdir(exist_ok=True);PVT.mkdir(exist_ok=True)
    protocol=OUT/'protocol.json'
    if not protocol.exists():
        write_json(protocol,dict(created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=sha256(Path(__file__)),parent_manifest_sha256=sha256(PUBLIC/'manifest.json'),
            scope='Full configuration only: N702 raw, N492 raw/agent; all six tuned families, previously inner-selected family, all four calibration modes for fixed LR/XGBoost.',
            rule='Maximize specificity subject to training OOF sensitivity >=0.85; ties prefer sensitivity then higher threshold. No model refits or selection by held-out results.',
            interpretation='Exploratory policy analysis requested after previous aggregate results; not prospective preregistration or independent validation.',
            uncertainty='2000 matched-pair bootstrap draws of fixed predictions for sensitivity and specificity; conditional, descriptive, unadjusted.'))
    fixed=json.loads(protocol.read_text())
    assert fixed['source_sha256']==sha256(Path(__file__))
    assert fixed['parent_manifest_sha256']==sha256(PUBLIC/'manifest.json')
    rows=[];threshold_rows=[];files={}
    for cohort,arm in [('n702','raw'),('n492','raw'),('n492','agent')]:
        for kind,families in [('benchmark',FAMILIES),('calibration',['lr','xgboost'])]:
            for family in families:
                for fold in range(5):
                    folder=LOCAL/kind/f'{cohort}_full_{arm}_{family}'/f'fold{fold}'
                    complete=json.loads((folder/'complete.json').read_text())
                    for name in ['training_predictions.parquet','predictions.parquet']:
                        digest=sha256(folder/name);assert digest==complete['artifacts'][name]
                        files[str((folder/name).relative_to(ROOT))]=digest
                    train=pd.read_parquet(folder/'training_predictions.parquet')
                    test=pd.read_parquet(folder/'predictions.parquet')
                    for mode in (['tuned'] if kind=='benchmark' else MODES):
                        a=train[train['mode']==mode];b=test[(test['mode']==mode)&(test.policy=='default')].copy()
                        assert a.participant_id.is_unique and b.participant_id.is_unique
                        assert not set(a.pair_id)&set(b.pair_id)
                        t=operating_table(a.y.to_numpy(),a.probability.to_numpy())
                        feasible=t[t.sensitivity>=.85]
                        choice=feasible.sort_values(['specificity','sensitivity','threshold'],ascending=False).iloc[0]
                        tau=float(choice.threshold)
                        # Independent count check of the training constraint and optimum.
                        z=a.probability.to_numpy()>=tau;y=a.y.to_numpy()
                        assert np.mean(z[y==1])>=.85
                        assert np.isclose(np.mean(~z[y==0]),feasible.specificity.max())
                        b['threshold']=tau;b['prediction']=(b.probability>=tau).astype(int);b['policy']='sensitivity85'
                        for key,value in dict(kind=kind,cohort=cohort,arm=arm,family=family).items():b[key]=value
                        rows.append(b)
                        threshold_rows.append(dict(kind=kind,cohort=cohort,arm=arm,family=family,mode=mode,fold=fold,
                            threshold=tau,training_sensitivity=float(choice.sensitivity),training_specificity=float(choice.specificity)))
    pred=pd.concat(rows,ignore_index=True)
    selection=pd.read_csv(PUBLIC/'family_choices.csv')
    selection=selection[selection.configuration=='full']
    chosen=[]
    for r in selection.to_dict('records'):
        p=pred[(pred.kind=='benchmark')&(pred.cohort==r['cohort'])&(pred.arm==r['arm'])&
            (pred.family==r['family'])&(pred.fold==r['fold'])].copy()
        assert len(p)>0;p['family']='inner_selected';chosen.append(p)
    pred=pd.concat([pred,*chosen],ignore_index=True)
    pred.to_parquet(PVT/'predictions.parquet',index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT/'training_thresholds.csv',index=False)
    results=[]
    for keys,g in pred.groupby(['kind','cohort','arm','family','mode']):
        assert len(g)==(702 if keys[1]=='n702' else 492) and g.participant_id.is_unique
        assert np.isfinite(g.probability).all()
        y=g.y.to_numpy();z=g.prediction.to_numpy()
        blocks=np.stack([np.flatnonzero(g.pair_id.to_numpy()==pair) for pair in g.pair_id.unique()])
        assert blocks.shape[1]==2
        rng=np.random.default_rng(20261118)
        sampled=blocks[rng.integers(len(blocks),size=(2000,len(blocks)))].reshape(2000,-1)
        yy=y[sampled];zz=z[sampled]
        se=((yy==1)&(zz==1)).sum(1)/(yy==1).sum(1)
        sp=((yy==0)&(zz==0)).sum(1)/(yy==0).sum(1)
        sl,sh=np.quantile(se,[.025,.975]);pl,ph=np.quantile(sp,[.025,.975])
        result=dict(zip(['kind','cohort','arm','family','mode'],keys))|expanded_metrics(g)
        result.update(sensitivity_low=float(sl),sensitivity_high=float(sh),specificity_low=float(pl),specificity_high=float(ph),
            meets_85_55=bool(result['sensitivity']>=.85 and result['specificity']>=.55),
            meets_85_60=bool(result['sensitivity']>=.85 and result['specificity']>=.60))
        results.append(result)
    d=pd.DataFrame(results);assert len(d)==45
    d.to_csv(OUT/'metrics.csv',index=False)
    write_json(OUT/'verification.json',dict(status='passed',training_thresholds_checked=len(threshold_rows),
        metric_groups=len(d),prediction_rows=len(pred),meets_85_55=int(d.meets_85_55.sum()),meets_85_60=int(d.meets_85_60.sum()),
        parent_inputs_unchanged=all(sha256(ROOT/path)==h for path,h in files.items())))
    selected=d[d.family=='inner_selected']
    lines=['# Exploratory sensitivity-85 threshold review','',
        'All thresholds maximize specificity subject to at least 85% sensitivity on saved training predictions. The outer participants do not set thresholds. The existing training-selected family choices are retained. No model was retrained. This policy was considered after prior results; it is exploratory.','',
        '| Cohort | Arm | Sensitivity (%) | Specificity (%) | Conditional sensitivity interval (%) | Conditional specificity interval (%) |',
        '|---|---|---:|---:|---|---|']
    for r in selected.itertuples():
        lines.append(f'| {r.cohort} | {r.arm} | {100*r.sensitivity:.2f} | {100*r.specificity:.2f} | {100*r.sensitivity_low:.2f}–{100*r.sensitivity_high:.2f} | {100*r.specificity_low:.2f}–{100*r.specificity_high:.2f} |')
    lines += ['',f'Across all 45 full-feature family/mode/cohort combinations, {int(d.meets_85_55.sum())} meet 85% sensitivity and 55% specificity as held-out point estimates; {int(d.meets_85_60.sum())} meet 85% and 60%. These counts must not be used to select a new winner from held-out performance.',
        '', 'Intervals resample matched pairs of fixed predictions and do not include full training/search uncertainty. A training sensitivity constraint does not guarantee held-out sensitivity. The earlier sensor-only random-blend seed-11 utility result (87.40% sensitivity, 55.28% specificity on N=492) is a separate exploratory observation, duplicated across identical raw/agent feature arms, not two confirmations or an agent improvement.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write_json(OUT/'manifest.json',{str(p.relative_to(ROOT)):sha256(p) for p in [*sorted(OUT.glob('*.csv')),*sorted(OUT.glob('*.md')),OUT/'protocol.json',OUT/'verification.json',Path(__file__)]})
    print(d[['kind','cohort','arm','family','mode','sensitivity','specificity','meets_85_55']].round(4).to_string(index=False))


if __name__=='__main__':main()
