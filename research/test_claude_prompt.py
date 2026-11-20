import inspect
import json
import numpy as np
import pytest
from research.claude_prompt_api import payload, budget_ok, Client, CAP, MODEL
from research.claude_prompt_core import traincell, load, signature, validate, feature_sets
from research.claude_prompt_run import canonical, schema_context, metric_summary
from research.common import ROOT
from research.test_final_multiagent import (
    test_safe_formula_and_duplicate_gate,
    test_eligibility_uses_repeats_and_multiple_families,
    test_automatic_control_is_deterministic_and_sensor_only)


def test_hard_budget_and_route():
    p, cost = payload('system', {'description': 'synthetic'})
    assert p['model'] == MODEL and p['provider']['only'] == ['amazon-bedrock/global']
    assert p['provider']['zdr'] and p['provider']['data_collection'] == 'deny'
    assert not p['provider']['allow_fallbacks'] and 'tools' not in p
    assert 3*((80000+2048)*3e-6+6000*15e-6) < CAP
    assert budget_ok([], cost)
    assert not budget_ok([{'reserved_usd': 1.9}], .2)
    assert not budget_ok([{'reserved_usd': .01}]*3, .01)
    with pytest.raises(AssertionError):
        payload('system', {'text': 'x'*80000})


def test_no_silent_truncation_or_code_response():
    r = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(
        {'analysis': 'synthetic', 'features': [{'formula': 'logabs(tap_rate)', 'rationale': 'synthetic'}]})}}]}
    assert len(Client.content(r)['features']) == 1
    r['choices'][0]['finish_reason'] = 'length'
    with pytest.raises(AssertionError):
        Client.content(r)


@pytest.mark.private_data
def test_cohort_and_discovery_payload_boundary():
    d, e = load('discovery'), load('evaluation')
    assert len(d) == 210 and len(e) == 492
    assert not set(d.participant_id)&set(e.participant_id)
    assert not set(d.pair_id)&set(e.pair_id)
    context = schema_context()
    text = json.dumps(payload('system', {'context': context})[0])
    assert '/Users/' not in text and 'sk-or-' not in text
    assert not any(str(x) in text for x in list(d.participant_id)+list(e.participant_id))
    assert set(context['allowed_inputs']) == set(feature_sets(d)['sensors'])
    seen = {canonical(f) for f in context['prior_formulas']}
    old = context['prior_formulas'][0]
    good, bad, _ = validate([dict(name='agent_repeat', formula=old)], context['allowed_inputs'], seen)
    assert not good and bad


def test_no_evaluation_labels_in_training_interface():
    assert list(inspect.signature(traincell).parameters) == ['x','y','groups','xtest']


@pytest.mark.private_data
def test_signature_includes_split_and_protocol():
    keys = signature()
    assert 'paper_ready/claude_prompt_replication/PROTOCOL.md' in keys
    assert 'paper_ready/private/agent_extension/split_manifest.json' in keys
    assert 'research/claude_prompt_api.py' in keys


@pytest.mark.private_data
def test_prompt_only_adds_truthful_baseline_clarification():
    from research.claude_single_run import schema_context as original_context
    d = load('discovery')
    assert len(feature_sets(d)['full']) == 52
    assert len(feature_sets(d)['sensors']) == 49
    now = schema_context()
    statement = now.pop('baseline_composition')
    assert now == original_context()
    assert 'evaluated separately' in statement and 'equal probability weights' in statement


def test_gate_implementation_and_model_settings_unchanged():
    from research import claude_prompt_core as new, claude_single_core as old
    for name in ['models', 'feedback', 'choose', 'score', 'automatic_candidates', 'traincell']:
        assert inspect.getsource(getattr(new, name)) == inspect.getsource(getattr(old, name))


def test_evaluation_skipped_without_selected_claude(tmp_path, monkeypatch):
    from research import claude_prompt_run as run
    monkeypatch.setattr(run, 'P', tmp_path)
    monkeypatch.setattr(run, 'signature', lambda: {'synthetic': 'test'})
    path = tmp_path / 'frozen.json'
    path.write_text(json.dumps({'signature': {'synthetic': 'test'}, 'arms': {'claude': [], 'automatic': ['synthetic']}}))
    assert not run.evaluation_required()
    path.write_text(json.dumps({'signature': {'synthetic': 'test'}, 'arms': {'claude': ['synthetic']}}))
    assert run.evaluation_required()
