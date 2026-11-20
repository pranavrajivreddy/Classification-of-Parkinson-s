import inspect
import numpy as np
import pandas as pd
import pytest
from research.final_multiagent_core import validate,feedback,traincell,automatic_candidates
from research.final_multiagent_api import payload,CAP

def test_safe_formula_and_duplicate_gate():
 fs=[dict(name='agent_a',formula='pdiv(tap_a, tap_b)'),dict(name='agent_b',formula='pdiv(tap_a,tap_b)'),dict(name='agent_c',formula='demo_age * tap_a'),dict(name='agent_d',formula="__import__('os').system('echo unsafe')")]
 good,bad,_=validate(fs,['tap_a','tap_b']);assert len(good)==1 and len(bad)==3

def test_payload_budget_privacy():
 p,reserve=payload('system',{'definitions':['synthetic inputs only']})
 assert p['provider']['only']==['deepinfra'] and p['provider']['zdr'] and p['provider']['data_collection']=='deny'
 assert not p['provider']['allow_fallbacks'] and p['max_tokens']==4096 and 'tools' not in p
 assert reserve*8<CAP
 with pytest.raises(AssertionError):payload('x',{'x':'x'*40000})

def test_eligibility_uses_repeats_and_multiple_families():
 rows=[]
 for repeat in [4242,731]:
  for fold in range(3):
   for method in ['a','b','c','d','e','f','equal_ensemble']:
    rows.append(dict(candidate='baseline',repeat=repeat,fold=fold,method=method,auc=.7,brier=.2,average_precision=.7))
 base=pd.DataFrame(rows);good=base.assign(candidate='agent_good',auc=.71);r=feedback(good,base)[0];assert r['eligible']
 bad=good.copy();bad.loc[bad.repeat==731,'auc']=.69;assert not feedback(bad,base)[0]['eligible']
 bad=good.copy();bad.loc[bad.method.isin(['a','b','c','d']),'auc']=.69;assert not feedback(bad,base)[0]['eligible']

def test_training_api_has_no_test_labels():
 assert list(inspect.signature(traincell).parameters)==['x','y','groups','xtest']

def test_automatic_control_is_deterministic_and_sensor_only(monkeypatch):
 from research import final_multiagent_core as core
 from research.common import ROOT
 columns=[line.split('|')[1].strip() for line in (ROOT/'paper_ready/FEATURE_DICTIONARY.md').read_text().splitlines() if line.startswith(('| tap_', '| walk_', '| voice_', '| demo_'))]
 synthetic=pd.DataFrame(columns=columns)
 monkeypatch.setattr(core,'load',lambda _:synthetic)
 a=automatic_candidates();b=automatic_candidates();assert a==b and len(a)==16
 assert all(not any(n.startswith('demo_') for n in f['inputs']) for f in a)
