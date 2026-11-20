import inspect
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from research.agent_auc_followup import rank_auc,eligible_bundle,summary,tune_predict,signature
from research.agent_features import parsed
from research.core import splits

def test_rank_auc_matches_reference_with_ties():
    rng=np.random.default_rng(10)
    for n in [20,100]:
        y=np.tile([0,1],n//2);p=np.round(rng.random(n),1)
        assert abs(rank_auc(y,p)-roc_auc_score(y,p))<1e-12

def test_selection_requires_both_repeat_gains_and_fixed_margin():
    rows=[]
    gains={'z_tie':[.01,.01],'a_tie':[.01,.01],'unstable':[.04,-.01],'weak':[.001,.001]}
    fs=[dict(name=k,formula='x') for k in gains]
    for repeat in [4242,731]:
        rows.append(dict(candidate='baseline',repeat=repeat,fold=0,method='lr',auc=.7))
        for name,g in gains.items():rows.append(dict(candidate=name,repeat=repeat,fold=0,method='lr',auc=.7+g[[4242,731].index(repeat)]))
    chosen=eligible_bundle(pd.DataFrame(rows),fs)
    assert [f['name'] for f in chosen]==['a_tie','z_tie']
    assert eligible_bundle(pd.DataFrame(rows[:1]),[])==[]

def test_no_test_labels_and_training_preprocessing():
    assert list(inspect.signature(tune_predict).parameters)==['x','y','groups','xtest']
    rng=np.random.default_rng(50);x=rng.normal(size=(36,3));x[0,1]=np.nan;y=np.tile([0,1],18);g=np.repeat(np.arange(18),2)
    a=tune_predict(x,y,g,np.zeros((2,3)));b=tune_predict(x,y,g,np.ones((2,3))*100)
    assert a[-1]==b[-1] and a[3]==b[3] and a[4]==b[4]
    for k in a[0]:
        np.testing.assert_allclose(a[0][k],b[0][k]);np.testing.assert_allclose(a[1][k][0].statistics_,np.nanmedian(x,axis=0))
    for tr,te in splits(y,g,n=3):assert not set(g[tr])&set(g[te])

@pytest.mark.private_data
def test_signature_uses_private_inputs():
    assert len(signature())==6

def test_formula_boundaries():
    for expression in ['y + tap_rate','participant_id','__import__("os")']:
        with pytest.raises(ValueError):parsed(expression,{'tap_rate'})
