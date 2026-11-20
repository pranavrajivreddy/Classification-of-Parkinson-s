"""Pair-aware descriptive biomarker contrasts and genuine refitting checks."""
import json
import numpy as np
import pandas as pd
from research.common import OUT, PRIVATE, sha256, write_json

PUBLIC=OUT/'causal_extension'
LOCAL=PRIVATE/'causal_extension'
MARKERS=['tap_rate','tap_flight_iqr','walk_speed_mean','voice_jitter_local']


def contrast(y,age,extra=None):
    design=np.column_stack([np.ones(len(y)),age] + ([] if extra is None else [extra]))
    if np.linalg.matrix_rank(design)<design.shape[1]:raise ValueError('Rank-deficient contrast design')
    return float(np.linalg.lstsq(design,y,rcond=None)[0][0])


def swapped_intercepts(y,age,signs):
    # Within-pair label swaps must reverse both contrasts.
    yy=signs*y; aa=signs*age
    ym=yy.mean(1);am=aa.mean(1)
    variance=(aa*aa).mean(1)-am*am
    if np.any(variance<=1e-15):raise ValueError('Degenerate age contrasts')
    slope=((yy*aa).mean(1)-ym*am)/variance
    return ym-slope*am


def bh(p):
    p=np.asarray(p);order=np.argsort(p);ranked=p[order]
    adjusted=np.minimum.accumulate((ranked*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result=np.empty_like(p);result[order]=np.minimum(1,adjusted);return result


def run():
    PUBLIC.mkdir(exist_ok=True);LOCAL.mkdir(exist_ok=True)
    d=pd.read_parquet(PRIVATE/'cohort.parquet'); assert len(d)==702
    old=pd.read_csv(OUT/'legacy_review/paired_associations.csv')
    results=[];repeats=[];private={}
    for number,feature in enumerate(MARKERS):
        w=d.pivot(index='pair_id',columns='y',values=[feature,'demo_age'])
        y=(w[feature][1]-w[feature][0]).to_numpy();age=(w.demo_age[1]-w.demo_age[0]).to_numpy()
        ok=np.isfinite(y)&np.isfinite(age);y=y[ok];age=age[ok];n=len(y)
        estimate=contrast(y,age)
        target=old[(old.feature==feature)&(old.estimator=='age_adjusted_pair_difference')].estimate.item()
        np.testing.assert_allclose(estimate,target,atol=1e-12,rtol=0)
        rng=np.random.default_rng(20261120+number);draws=rng.integers(n,size=(2000,n))
        boots=np.array([contrast(y[ix],age[ix]) for ix in draws])
        signs=np.random.default_rng(20262120+number).choice([-1.,1.],size=(5000,n))
        null=swapped_intercepts(y,age,signs)
        # Independent least-squares replay of initial randomization draws.
        for s,v in zip(signs[:10],null[:10]):np.testing.assert_allclose(contrast(y*s,age*s),v,atol=1e-12)
        p=(1+np.sum(np.abs(null)>=abs(estimate)))/(len(null)+1)
        noise=[];subsets=[];rng=np.random.default_rng(20263120+number)
        for rep in range(200):
            z=rng.normal(size=(n,2));extra=z[:,1]-z[:,0]
            noise.append(contrast(y,age,extra))
            ix=rng.choice(n,int(np.floor(.8*n)),replace=False)
            subsets.append(contrast(y[ix],age[ix]))
            repeats.append(dict(feature=feature,repetition=rep,noise_adjusted_intercept=noise[-1],
                subset_intercept=subsets[-1],subset_pairs=len(ix)))
        lo,hi=np.percentile(boots,[2.5,97.5]);slo,shi=np.percentile(subsets,[2.5,97.5])
        results.append(dict(feature=feature,pairs=n,excluded_pairs=len(w)-n,
            unadjusted_difference=float(y.mean()),age_adjusted_difference=estimate,ci_low=lo,ci_high=hi,
            label_swap_p=p,noise_mean_shift=float(np.mean(noise)-estimate),
            noise_max_absolute_shift=float(np.max(np.abs(np.array(noise)-estimate))),
            subsample_median=float(np.median(subsets)),subsample_p025=slo,subsample_p975=shi,
            subsample_same_sign_fraction=float(np.mean(np.sign(subsets)==np.sign(estimate)))))
        private[feature+'_bootstrap']=boots;private[feature+'_null']=null
    summary=pd.DataFrame(results);summary['label_swap_BH_q']=bh(summary.label_swap_p)
    summary.to_csv(PUBLIC/'association_summary.csv',index=False)
    pd.DataFrame(repeats).to_csv(PUBLIC/'association_refit_distributions.csv',index=False)
    np.savez_compressed(LOCAL/'association_resamples.npz',**private)
    write_json(PUBLIC/'association_verification.json',dict(status='passed',pairs=351,markers=4,
        bootstrap_refits=8000,label_swap_replicates=20000,noise_refits=800,subsample_refits=800,
        independent_swap_regression_checks=40,previous_point_estimates_reproduced=4,
        cohort_sha256=sha256(PRIVATE/'cohort.parquet'),source_sha256=sha256(__file__),
        protocol_sha256=sha256(PUBLIC/'PROTOCOL.md'),clinical_causal_effect_identified=False,
        result_sha256=sha256(PUBLIC/'association_summary.csv')))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':run()
