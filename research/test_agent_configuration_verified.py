import copy
import numpy as np
import pytest
from research.agent_configuration_search_verified import (
    defaults, validate, estimator, check_fit, measures, operating_threshold, gate, choose,
    SPACE, suggest, flat)
from research.core import splits
import optuna


def test_defaults_and_forbidden_choices():
    for c in defaults().values(): validate(c)
    c = defaults()['lr']; c['params']['C'] = 100000
    with pytest.raises(AssertionError): validate(c)
    c = defaults()['lr']; c['inputs'] = 'participant_id'
    with pytest.raises(AssertionError): validate(c)
    c = defaults()['lr']; c['params']['exec'] = 'anything'
    with pytest.raises(AssertionError): validate(c)
    c = defaults()['lr']; c['keep'] = True
    with pytest.raises(AssertionError): validate(c)


def test_train_only_preprocessing_and_selection():
    rng = np.random.default_rng(47)
    x = rng.normal(size=(48, 25)); x[::3, 2] = np.nan; y = np.tile([0, 1], 24)
    c = defaults()['lr']; c['keep'] = 10
    est = estimator(c, 25).fit(x[:32], y[:32])
    check_fit(est, c, x[:32], y[:32])
    before = est['selector'].get_support().copy()
    test = np.full((16, 25), 1e9); est.predict_proba(test)
    check_fit(est, c, x[:32], y[:32]); np.testing.assert_array_equal(before, est['selector'].get_support())
    with pytest.raises(AssertionError): check_fit(est, c, np.vstack([x[:32], test]), y)


def test_group_boundaries():
    y = np.tile([0, 1], 24); g = np.repeat(np.arange(24), 2)
    for tr, va in splits(y, g, n=3): assert not set(g[tr]) & set(g[va])


def test_threshold_and_auc_invariance():
    y = np.array([1, 1, 0, 0, 1, 0]); p = np.array([.9, .7, .6, .4, .3, .1])
    tau = operating_threshold(y, p)
    assert measures(y, p, tau)['sensitivity'] >= .85
    assert measures(y, p, .2)['auc'] == measures(y, p, .6)['auc']
    assert measures(y, p, 1.1)['ppv'] is None


def test_selection_rule():
    base = dict(mean_auc=.75, repeat_auc={'4242': .75, '731': .75}, mean_brier=.2)
    unstable = dict(mean_auc=.76, repeat_auc={'4242': .78, '731': .74}, mean_brier=.19)
    good = dict(mean_auc=.76, repeat_auc={'4242': .76, '731': .76}, mean_brier=.19)
    assert not gate(unstable, base)['eligible']
    assert gate(good, base)['eligible']
    assert choose([dict(slot=0, summary=unstable)], base)['selected'] == base
    assert choose([dict(slot=0, summary=good)], base)['selected'] == good


def test_comparator_space_and_warm_start():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=3, n_startup_trials=6))
    for c in defaults().values(): study.enqueue_trial(flat(c))
    for expected in defaults().values():
        t = study.ask(); assert suggest(t) == expected; study.tell(t, .7)
    for _ in range(12):
        t = study.ask(); validate(suggest(t)); study.tell(t, .71)
