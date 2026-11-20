"""Tests for scientific failure modes, using synthetic task summaries only."""
import numpy as np
import pytest
from research.explainability import check_algebra, walking_scenario, permutation_shap


def example():
    values = dict(walk_duration_sec=30., walk_steps_total=40., walk_distance_total=20.,
        walk_speed_mean=2/3, walk_cadence=80., walk_step_length=.5,
        tap_duration_sec=20., tap_count_total=42., tap_count_valid=40., tap_count_miss=2.,
        tap_rate=2., tap_miss_ratio=2/42, tap_flight_mean=.4, tap_flight_std=.1,
        tap_flight_cv=.25, demo_age=60., demo_gender_male=1., demo_smoked=np.nan,
        voice_jitter_local=1.)
    return list(values),np.array([list(values.values())])


def test_coherent_cadence_change_preserves_other_inputs():
    columns,x=example(); z=walking_scenario(x,columns,.1)
    check_algebra(z,columns)
    for key,expected in dict(walk_steps_total=44.,walk_distance_total=22.,
                             walk_cadence=88.,walk_speed_mean=22/30).items():
        assert z[0,columns.index(key)]==expected
    for key in ['demo_age','demo_gender_male','voice_jitter_local','walk_duration_sec','walk_step_length']:
        assert z[0,columns.index(key)]==x[0,columns.index(key)]
    assert np.array_equal(np.isnan(x),np.isnan(z))


def test_zero_change_is_exact_identity():
    columns,x=example(); np.testing.assert_equal(walking_scenario(x,columns,0),x)


def test_independent_cadence_edit_is_rejected():
    columns,x=example(); x[0,columns.index('walk_cadence')]+=10
    with pytest.raises(AssertionError): check_algebra(x,columns)


def test_inconsistent_tapping_counts_are_rejected():
    columns,x=example(); x[0,columns.index('tap_count_valid')]+=1
    with pytest.raises(AssertionError): check_algebra(x,columns)


def test_counts_are_discrete_and_achieved_change_can_differ():
    columns,x=example(); z=walking_scenario(x,columns,.013)
    assert z[0,columns.index('walk_steps_total')]==41
    assert z[0,columns.index('walk_cadence')]==82
    check_algebra(z,columns)


def test_permutation_shap_matches_analytic_linear_attributions():
    background=np.array([[0.,1.,2.],[2.,3.,4.]])
    x=np.array([[3.,2.,1.],[1.,4.,5.]])
    coef=np.array([.1,.2,-.05]); f=lambda z:.2+z@coef
    values,base=permutation_shap(f,background,x,17)
    np.testing.assert_allclose(values,(x-background.mean(0))*coef,atol=1e-14)
    np.testing.assert_allclose(base+values.sum(1),f(x),atol=1e-14)


def test_permutation_shap_nonlinear_efficiency_and_dummy_feature():
    background=np.array([[0.,1.,2.],[2.,3.,4.]])
    x=np.array([[3.,2.,1.],[1.,4.,5.]])
    f=lambda z:np.tanh(z[:,0]*z[:,1])
    values,base=permutation_shap(f,background,x,17)
    np.testing.assert_allclose(base+values.sum(1),f(x),atol=1e-14)
    np.testing.assert_array_equal(values[:,2],0)
