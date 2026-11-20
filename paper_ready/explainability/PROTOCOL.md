# Held-out SHAP and constrained model-input counterfactuals

Fixed 19 November 2026 before executing this extension. Exploratory analysis of
previously evaluated participants; not preregistration or external validation.

## Question and causal boundary

Which measurements explain the existing classifier, and how does its output change
under specified, algebraically coherent input substitutions? The outcome is the
classifier score, not future disease, disease severity, or a treatment response.
No identified biomarker-to-disease causal effect is claimed. No medication analysis.

The original score-as-treatment regression is circular: its treatment was learned
from the same diagnosis labels used as outcomes. Its refutation flags and p-values
are not verified evidence. Setting Disease_Status=0 in an assumed structural model
does not specify an available clinical intervention. Neither analysis is repeated.

## Frozen population and models

Use all 702 corrected participants (351 matched pairs), the existing five outer
folds, and persisted full-feature models from the cohort-spectrum comparison.
Three-inner-fold AUC selection was already completed using seven candidates.
No model, feature, threshold, or optimizer is retuned using held-out outcomes.
Replay predictions against saved results before explaining them. Keep participant
records, attributions, backgrounds and counterfactual inputs in private storage.

## SHAP

TreeSHAP, probability output, with 100 deterministically sampled outer-training
background records per fold (seed 20261119+fold). Explain every held-out person.
Also explain all outer-training records to select the top sensor feature and its
modality for a SHAP-guided counterfactual arm. This training ranking is a model
description, not a generalization estimate. Report held-out mean absolute SHAP,
fold rankings and attribution additivity error. Repeat held-out explanations with
a second independent training-background sample to assess background sensitivity.
Interventional TreeSHAP is a software definition of missing-feature integration;
it does not identify clinical intervention effects, and can break feature
dependencies. Correlated and derived-feature attributions are not uniquely causal.

## Constrained block substitution

For every held-out participant and each sensor modality, take up to 30 nearest
outer-training donors with the same survey sex, age within five years, and an
identical missingness pattern for that modality. Rank donors by Euclidean distance
on the train-imputed, robust-scaled full feature vector; never by diagnosis.
Replace the complete modality block, preserving all other input values, including
age, sex and smoking. Whole-block substitution preserves within-task observed
relationships but does not establish cross-task physiological feasibility.

Retain candidates only when their distance to the nearest outer-training record
does not exceed the 95th percentile of training nearest-neighbour distances,
excluding the same matched pair in that training reference calculation. This is
an empirical support heuristic, not proof of actionability. Include the unchanged
input as a no-change option. Select the nearest accepted candidate crossing below
the fixed illustrative score threshold 0.5; if none crosses, select the candidate
with the smallest score, provided it reduces the score. Report all attempts,
candidate counts, failures, score shifts, and crossing rates among baseline
score-positive participants. Report both diagnosis groups and all three blocks;
the SHAP-guided arm uses the training-selected modality, not held-out SHAP ranking.
The 0.5 boundary is illustrative, not a validated screening policy.

## Cadence scenario, fixed in advance

Evaluate requested cadence changes -20%, -10%, 0%, +10%, +20% for every held-out
record, retaining duration and step length. Round resulting step counts to whole
steps; recalculate distance, speed, and cadence together using the extractor's
identities: cadence=60*steps/duration; distance=step_length*steps;
speed=distance/duration. At zero change preserve the original record exactly.
Report achieved, not just requested, cadence change. Accept a nonzero scenario
only if all six walking features lie within the training 1st–99th percentile
ranges and it passes the same full-vector distance check. Report rejected scenarios
and results on the common-support participants passing all four nonzero scenarios
to avoid silently comparing different subsets. These scenarios change several
related inputs and cannot isolate cadence's clinical effect.

## Inference and verification

Use 2,000 matched-pair bootstrap resamples for mean score changes and crossing
fractions, conditional on saved models, backgrounds and searches. These intervals
do not include model-fitting uncertainty, are exploratory, and are not causal
effect intervals. No significance claims or treatment recommendations.
Verify cohort identity, pair-disjoint folds, predictor exclusion of identifiers
and diagnosis, model and input hashes, prediction replay, SHAP additivity, frozen
demographics, missingness preservation, walking/tapping algebra, donor provenance,
support filtering, retained failures and counterfactual prediction replay.

## Age-column ablation

Verify the existing corrected broad-cohort full/no-age/sensors/age-only experiment
and independently recompute its headline metrics. Do not rerun an identical search
or replace the original cohort with the corrected cohort without stating the
difference. Age omission does not remove age-correlated information from sensors.

## Stop rule

Run these fixed arms once; repair implementation failures only. No widening donor
calipers, changing support cutoffs, or trying new perturbations for favorable
results. Any infeasibility or null/mixed result is retained.

## Implementation repair recorded before accepting results

The first run stopped after the three gradient-boosting folds: TreeSHAP 0.48.0
with this installed random-forest/runtime combination failed the 1e-6 additivity
check (maximum observed training error approximately 0.0055). No attribution
results from that run were accepted. No thresholds or scientific arms changed.
For the two random-forest folds only, use model-agnostic marginal Shapley Monte
Carlo: one random feature ordering and its reverse for each of the same 100
background records (200 paths), calling the actual predictor. Each path telescopes
and the global base is the mean prediction over those backgrounds. Test analytic
linear attribution, dummy features and nonlinear efficiency. Repeat with the
independent second background and ordering seed. These are approximate Shapley
values; additivity alone does not certify attribution convergence. Gradient
boosting retains TreeSHAP. Backgrounds and rankings remain training-only.
