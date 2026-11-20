"""Boundary and decision-rule tests for the additional experiment."""
import inspect
import numpy as np
from sklearn.metrics import fbeta_score
from research.benchmark_extension import thresholds, tune_fit, calibrated_crossfit, Sigmoid, MODES


def toy():
    rng=np.random.default_rng(731)
    groups=np.repeat(np.arange(60),2)
    y=np.tile([0,1],60)
    X=rng.normal(size=(120,4)); X[:,0]+=y
    X[::7,2]=np.nan
    return X,y,groups


def test_f2_exact_and_tied_scores():
    y=np.array([0,1,1,0,1,0,1,0])
    p=np.array([.1,.1,.3,.4,.5,.6,.6,.9])
    tau=thresholds(y,p)['f2_max']
    expected=max(fbeta_score(y,p>=t,beta=2) for t in np.r_[np.unique(p),np.nextafter(p.max(),np.inf)])
    assert np.isclose(fbeta_score(y,p>=tau,beta=2),expected)


def test_tuning_training_only_and_deterministic():
    X,y,g=toy()
    assert 'ytest' not in inspect.signature(tune_fit).parameters
    a=tune_fit(X[:100],y[:100],g[:100],X[100:],'lr',42,n_trials=2)
    b=tune_fit(X[:100],y[:100],g[:100],X[100:]+1000,'lr',42,n_trials=2)
    np.testing.assert_array_equal(a[0],b[0])
    np.testing.assert_array_equal(a[4],b[4])
    assert a[3]['chosen']==b[3]['chosen']
    assert a[3]['trials']==b[3]['trials']
    # The extreme held-out shift must not change fitted preprocessing.
    np.testing.assert_array_equal(a[2].named_steps['robustscaler'].center_,b[2].named_steps['robustscaler'].center_)
    assert set(a[5])=={0,1,2}
    for pair in np.unique(g[:100]): assert len(set(a[5][g[:100]==pair]))==1


def test_calibration_crossfit_and_chain_disjointness():
    X,y,g=toy()
    inner,outer,audit=calibrated_crossfit(X[:100],y[:100],g[:100],X[100:],'lr',42)
    heldouts=[]
    for record in audit['crossfit']:
        held=set(record['threshold_holdout_groups']);heldouts.extend(held)
        for member in record['members']:
            parts=[set(member[k]) for k in ['train_groups','sigmoid_groups','isotonic_groups']]
            assert all(not p&held for p in parts)
            assert set.union(*parts)==set(g[:100])-held
            assert all(not parts[i]&parts[j] for i in range(3) for j in range(i))
    assert sorted(heldouts)==list(range(50))
    for mode in MODES:
        assert len(inner[mode])==100 and len(outer[mode])==20
        assert np.isfinite(inner[mode]).all() and np.isfinite(outer[mode]).all()
        assert ((outer[mode]>=0)&(outer[mode]<=1)).all()


def test_sigmoid_handles_constant_scores():
    y=np.tile([0,1],20)
    p=Sigmoid().fit(np.ones(40),y).predict(np.ones(10))
    np.testing.assert_allclose(p,.5,atol=1e-6)
