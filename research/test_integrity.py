import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from research.data import walking,tapping,first_rows,match_pairs
from research.core import operating_table,select_thresholds,metrics,splits,pipeline,engineered


def test_cumulative_pedometer_is_not_summed():
    items=[dict(startDate='2020-01-01T00:00:00Z',endDate=f'2020-01-01T00:00:{s:02d}Z',
                numberOfSteps=n,distance=d) for s,n,d in [(10,10,5),(20,20,10),(30,30,15)]]
    r=walking(items)
    assert r['walk_steps_total']==30
    assert r['walk_distance_total']==15
    assert r['walk_cadence']==60
    assert r['walk_step_length']==.5
    items[-1]['numberOfSteps']=5
    with pytest.raises(ValueError,match='reset'):walking(items)


def test_first_rows_never_splices_sessions():
    df=pd.DataFrame(dict(healthCode=['a','a'],createdOn=[1,2],recordId=['1','2'],handle=[5,6],value=[np.nan,8]))
    assert np.isnan(first_rows(df,'handle').loc['a','value'])


def test_matching_exact_sex_age_no_reuse():
    df=pd.DataFrame(dict(participant_id=['a','b','c','d','e'],y=[1,1,0,0,0],
                        demo_gender_male=[1,0,1,0,1],demo_age=[60,55,61,54,20]))
    m=match_pairs(df)
    assert len(m)==4 and m.participant_id.is_unique
    for _,g in m.groupby('pair_id'):
        assert len(g)==2 and set(g.y)=={0,1} and g.demo_gender_male.nunique()==1
        assert g.demo_age.max()-g.demo_age.min()<=2


def test_threshold_enumeration_and_utility_identity():
    y=np.array([0,1,0,1,1,0]);p=np.array([.1,.2,.2,.5,.9,.9])
    for row in operating_table(y,p).itertuples():
        m=metrics(y,p,p>=row.threshold)
        assert m['tp']==row.tp and m['fp']==row.fp
        assert np.isclose(m['utility'],1-(4*m['fn']+2*m['fp'])/len(y))
    thresholds=select_thresholds(y,p)
    assert ((p>=thresholds['sensitivity90'])[y==1]).mean()>=.9
    assert ((p<thresholds['specificity60'])[y==0]).mean()>=.6


def test_train_only_preprocessing():
    X=np.array([[0,np.nan],[1,10],[2,12],[3,14]],float); y=np.array([0,0,1,1])
    model=pipeline(LogisticRegression()).fit(X,y)
    before=model.named_steps['simpleimputer'].statistics_.copy()
    model.predict_proba(np.array([[1e12,-1e12]]))
    np.testing.assert_equal(before,model.named_steps['simpleimputer'].statistics_)
    assert before[1]==12


def test_group_folds_exclusive():
    y=np.tile([0,1],30);g=np.repeat(np.arange(30),2)
    for a,b in splits(y,g,n=5):
        assert not set(g[a])&set(g[b])




def test_feature_engineering_applies_without_configuration_names():
    df=pd.DataFrame({k:[1.] for k in ['tap_flight_iqr','tap_flight_mean','tap_flight_std','tap_miss_ratio',
        'voice_jitter_local','voice_shimmer_local','voice_f0_std','voice_f0_mean','voice_spectral_rolloff','voice_spectral_centroid']})
    assert len([c for c in engineered(df) if c.startswith('eng_')])==6
