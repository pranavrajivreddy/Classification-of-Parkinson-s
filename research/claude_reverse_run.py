"""One Claude agent, three feedback rounds; local numerical gate and frozen evaluation."""
import argparse
import ast
import getpass
import json
import os
import pandas as pd
from research.common import ROOT, sha256, write_json
from research.claude_reverse_core import (
    P, L, signature, load, feature_sets, validate, score, feedback, choose,
    automatic_candidates, evaluate)
from research.claude_reverse_api import Client

SYSTEM = '''You are the single scientific agent in a bounded biomedical feature-engineering
feedback loop. Inspect measurements, critique your earlier proposals using the supplied
aggregate development feedback, and propose parsimonious numerical transformations.
You have no participant records, external tools or executable-code permissions.
Return the specified JSON only. Do not claim causal mechanisms, diagnosis, clinical
validity or verified improvement. Do not change labels, cohort, folds, models, objective,
selection criteria or stopping rule. Formulas cannot introduce new measured information.
Favor ideas that can improve ranking across model families, while considering AP, Brier
and fixed-threshold metrics; optimizing a threshold alone cannot improve ROC-AUC.
There is no metric target you must reach. Empty proposals are acceptable.'''


def canonical(formula):
    return ast.dump(ast.parse(formula, mode='eval'), include_attributes=False)


def schema_context():
    allowed = feature_sets(load('discovery'))['sensors']
    text = (ROOT / 'paper_ready/FEATURE_DICTIONARY.md').read_text()
    definitions = [line for line in text.splitlines()
                   if any(line.startswith('| ' + c + ' |') for c in allowed)]
    old = []
    for path in ['agent_extension/frozen_features.json', 'agent_auc_followup/round1_proposal.json',
                 'agent_auc_followup/round2_proposal.json', 'final_multiagent/round1_proposals.json',
                 'final_multiagent/round2_proposals.json']:
        source = ROOT / 'configs' / ('prior_multiagent_' + path.split('/')[-1].replace('_proposals', '')) if path.startswith('final_multiagent/') else ROOT / 'paper_ready' / path
        j = json.loads(source.read_text())
        old.extend(f['formula'] for f in j.get('features', j.get('proposed', [])))
    return dict(allowed_inputs=allowed, definitions=definitions,
        prior_formulas=sorted(set(old)),
        baseline_definition='The baseline has exactly52 conventional inputs:49 sensor summaries and3 demographic/history variables. NONE of prior_formulas is included in the baseline. Each candidate adds one column. Six ensemble weights are equal; no model dominates by assigned weight.',
        functions='pdiv(a,b): a/b with signed absolute denominator floor1e-4; logabs(a)=log(1+abs(a)); sqrtabs(a)=sqrt(abs(a)); abs(a). Operators +,-,*. No /,**,indexing,attribute access or other calls. At most80 AST nodes; finite constants abs<=100.',
        constraints='Only listed sensor inputs. No demographics, medication, identity, labels or study engagement totals. Task records are same-person but not synchronous. Walking speed=cadence*step_length/60: avoid redundant rearrangements. Tapping flight is inter-event time, not airborne finger time. Jitter/shimmer in percent. Entire voice record acoustics, not validated isolated phonation. Do not relabel arbitrary ratios as validated neurological biomarkers.',
        objective='492 discovery participants,2x3 pair-grouped CV. Six fixed families LR/RF/XGBoost/LightGBM/CatBoost/MLP. Primary mean-fold AUC of equal-probability ensemble; eligibility gain>=.002, positive both repetitions, >=4/6family mean gains nonnegative. At most2selected then one bundle check. Raw fallback if gate fails.',
        historical_lessons='Original Claude-labelled scripts hard-coded features/selection/calibration/weights. Some selected features globally or used outer labels for stacking and thresholds. None of those choices may use evaluation data here. Corrected preprocessing is fold-local; calibration/threshold routines cannot rescue ranking by inspecting evaluation labels.',
        status='Exploratory follow-up: prior project results exposed to investigators. This context contains no evaluation results. Prior syntax-equivalent formulas are ineligible as new proposals.')


def metric_summary(scores):
    cols = ['auc', 'average_precision', 'brier', 'sensitivity', 'specificity', 'f1', 'f2', 'balanced_accuracy']
    return scores.groupby('method')[cols].mean().round(6).to_dict('index')


def discover(key):
    P.mkdir(exist_ok=True)
    L.mkdir(parents=True, exist_ok=True)
    frozen = P / 'frozen.json'
    if frozen.exists():
        assert json.loads(frozen.read_text())['signature'] == signature()
        return
    start = P / 'start_manifest.json'
    if start.exists():
        assert json.loads(start.read_text())['signature'] == signature()
    else:
        write_json(start, dict(signature=signature(), scope='Additional exploratory single-agent trajectory'))
    controls = automatic_candidates()
    write_json(P / 'automatic_sequence.json', controls)
    context = schema_context()
    write_json(P / 'api_context.json', context)
    client = Client(key)
    history, allfeatures, allscores = [], [], []
    seen = {canonical(f) for f in context['prior_formulas']}
    try:
        baseline = score('baseline', [])
        summary = metric_summary(baseline)
        write_json(P / 'baseline_discovery_summary.json', summary)
        for r in [1, 2, 3]:
            user = dict(context=context, round=r, total_rounds=3, baseline=summary, history=history,
                instruction='Analyze the available aggregate evidence, identify representation limitations, and propose up to4 NEW formulas. Return {analysis:string,features:[{formula:string,rationale:string}]}. Later rounds must respond to prior feedback; no claims of improvement before scoring. An empty array is allowed.')
            answer = client.call(f'round{r}_single', SYSTEM, user)
            proposals = [dict(f, name=f'agent_claude_r{r}_{i}') for i, f in enumerate(answer['features'])]
            valid, invalid, seen = validate(proposals, context['allowed_inputs'], seen)
            record = dict(round=r, analysis=answer['analysis'], proposed=proposals, valid=valid, invalid=invalid)
            write_json(P / f'round{r}_proposals.json', record)
            pieces = [score(f['name'], [f]) for f in valid]
            result = dict(round=r, analysis=answer['analysis'], proposed=proposals, invalid=invalid,
                feedback=feedback(pd.concat(pieces, ignore_index=True), baseline) if pieces else [],
                metrics={f['name']: metric_summary(s) for f, s in zip(valid, pieces)})
            history.append(result)
            write_json(P / f'round{r}_feedback.json', result)
            allfeatures += valid
            allscores += pieces
            print(f'Claude round {r}/3 complete: {len(valid)} new valid, {len(invalid)} rejected', flush=True)
    finally:
        client.close()
    controls = controls[:len(allfeatures)]
    control_scores = [score(f['name'], [f]) for f in controls]
    decisions, arms = [], {'raw': []}
    for arm, fs, pieces in [('claude', allfeatures, allscores), ('automatic', controls, control_scores)]:
        scores = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        scores.to_csv(P / f'{arm}_discovery_scores.csv', index=False)
        arms[arm], decision = choose(fs, scores, baseline, arm)
        decisions.append(decision)
        write_json(P / f'{arm}_candidate_summary.json', feedback(scores, baseline) if pieces else [])
    provenance = {str(p.relative_to(P)): sha256(p) for p in P.rglob('*') if p.is_file() and p != frozen}
    write_json(frozen, dict(signature=signature(), arms=arms, decisions=decisions,
        agent_candidates_scored=len(allfeatures), automatic_candidates_scored=len(controls),
        provenance=provenance, scope='Previously exposed exploratory evaluation; no provider/agent-count causal comparison'))
    print('Discovery frozen; selected feature counts:', {k: len(v) for k, v in arms.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['discover', 'evaluate'])
    stage = parser.parse_args().stage
    if stage == 'discover':
        key = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter key (hidden): ')
        try:
            discover(key)
        finally:
            key = None
    else:
        check = json.loads((P / 'pre_evaluation_audit.json').read_text())
        assert check['status'] == 'passed' and check['frozen_sha256'] == sha256(P / 'frozen.json')
        evaluate()


if __name__ == '__main__':
    main()
