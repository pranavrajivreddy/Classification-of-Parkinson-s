import numpy as np
from research.core import fit_fold


def test_test_features_cannot_change_training_selection_or_threshold_data():
    rng=np.random.default_rng(4);X=rng.normal(size=(60,4));y=np.tile([0,1],30);groups=np.repeat(np.arange(30),2)
    X[:,0]+=y
    p,a,P,_,folds=fit_fold(X,y,groups,rng.normal(size=(4,4)))
    q,b,Q,_,_=fit_fold(X,y,groups,np.full((4,4),1e9))
    assert a==b
    np.testing.assert_allclose(P,Q)
    for key in p:np.testing.assert_allclose(p[key][0],q[key][0])
    for tr,va in folds:assert not set(groups[tr]) & set(groups[va])
