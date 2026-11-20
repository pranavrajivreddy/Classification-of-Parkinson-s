import numpy as np
from research.causal_association import contrast,swapped_intercepts,bh
from research.causal_model_panel import structural_inputs,diverse_choices
from research.data import tapping
from research.test_explainability import example
from research.explainability import check_algebra


def test_pair_regression_known_intercept():
    age=np.array([-2.,-1.,0.,1.,2.]);y=3+2*age
    np.testing.assert_allclose(contrast(y,age),3,atol=1e-12)


def test_label_swaps_flip_both_contrasts():
    age=np.array([-2.,-1.,0.,1.,2.]);y=3+2*age
    signs=np.array([[1.,-1.,1.,-1.,1.],[-1.,-1.,-1.,-1.,-1.]])
    values=swapped_intercepts(y,age,signs)
    for v,s in zip(values,signs):np.testing.assert_allclose(v,contrast(y*s,age*s),atol=1e-12)
    np.testing.assert_allclose(values[1],-3,atol=1e-12)


def test_noise_control_really_adjusts_regression():
    age=np.arange(10.);z=np.sin(age);y=4+2*age+3*z
    np.testing.assert_allclose(contrast(y,age,z),4,atol=1e-12)
    assert abs(contrast(y,age)-4)>.1


def test_bh_multiplicity():
    np.testing.assert_allclose(bh([.01,.04,.03,.2]),[.04,.0533333333333,.0533333333333,.2])


def test_no_feasible_counterfactual_retained_as_failure():
    z=np.array([[0.,0.],[1.,1.]])
    assert diverse_choices(z,np.zeros(2),np.array([.8,.9]),.5,np.ones(2,bool))==[]
    assert diverse_choices(z,np.zeros(2),np.array([.1,.2]),.5,np.zeros(2,bool))==[]


def test_counterfactuals_distinct_and_first_nearest():
    z=np.array([[1.,0.],[1.,0.],[2.,0.],[3.,1.]])
    selected=diverse_choices(z,np.zeros(2),np.full(4,.1),.5,np.ones(4,bool))
    assert len(selected)==3 and selected[0]==0 and len(np.unique(z[selected],axis=0))==3


def test_walking_step_length_propagates_exactly():
    cols,x=example();z=structural_inputs(x,cols,'walking_step_length',.2)
    check_algebra(z,cols)
    assert z[0,cols.index('walk_step_length')]==.6
    assert z[0,cols.index('walk_distance_total')]==24
    for c in ['walk_steps_total','walk_duration_sec','walk_cadence','demo_age','tap_rate']:
        assert z[0,cols.index(c)]==x[0,cols.index(c)]


def test_tapping_summary_matches_raw_time_rescaling():
    samples=[dict(TapTimeStamp=float(i*.3),TappedButtonId='TappedButtonLeft' if i%2 else 'TappedButtonRight') for i in range(50)]
    base=tapping(samples,20.);cols=list(base);x=np.array([list(base.values())])
    for r in [-.2,-.1,0.,.1,.2]:
        raw=tapping([dict(s,TapTimeStamp=s['TapTimeStamp']/(1+r)) for s in samples],20/(1+r))
        z=structural_inputs(x,cols,'tapping_time',r)
        np.testing.assert_allclose(z[0],[raw[c] for c in cols],atol=1e-12)


def test_raw_filter_membership_can_change():
    # This demonstrates why summary-only time scaling needs the actual raw check.
    dt=np.array([.0095,.5,4.8]);old=(dt>=.01)&(dt<=5)
    slowed=dt/.9;new=(slowed>=.01)&(slowed<=5)
    assert not np.array_equal(old,new)
