# Exploratory cohort-spectrum comparison — 19 November 2026

This plan is frozen before fitting the new broad-cohort models. It is an exploratory
follow-up informed by historical and previously corrected results, not preregistration.

## Question and cohorts
Does diagnostic discrimination change across the broad adult local multimodal pool,
the age-restricted eligible cohort, and the existing age/sex-matched cohort?

Start from the original 2,660-person processed participant pool. Reconstruct features
from local raw recordings with `research/data.py` functions and its original earliest
whole-record selection, label-conflict exclusions, and modality quality criteria.
The only demographic eligibility change is replacing ages 35–75 with valid adult
ages 18–100. Known binary survey sex is required consistently. The resulting count
is determined by QC, not forced to 2,660. No new download or later-record rescue.
The ages-35–75 subset must reproduce all existing eligible participants and model
features exactly. Preserve the existing 702-person matched cohort and pair identities.
These cohorts are nested, not independent replications. Modalities are linked by
participant, not necessarily synchronized in time.

## Evaluation
Four configurations: age only; demographics plus all sensors; all features except
age; sensors only. Reuse the existing seven-candidate library in `research/core.py`
with random seed 42. Five outer folds; three group-stratified inner folds (seed 143).
All imputation and scaling occur inside each fit. Select the candidate by inner OOF
AUC; select thresholds using its inner OOF predictions only. Report default 0.5,
utility, target sensitivity 0.90, and target specificity 0.60 policies. Targets are
training-selection targets, not guaranteed held-out performance.

Anchor the common matched participants to their existing outer fold assignments.
Assign remaining age-restricted participants and remaining broad-age participants
separately by deterministic stratified five-fold assignment. Each participant thus
has the same outer fold in every cohort containing them. Keep matched pairs in the
same outer and inner fold; unmatched participants are singleton groups.

Report (1) each cohort's own held-out performance and (2) the predictions restricted
to the same 702 held-out matched participants under each training-cohort strategy.
Save all seven candidates' inner OOF scores, selected models, thresholds and indices.
No agent or evolutionary feature discovery is repeated for this focused analysis.

## Uncertainty and interpretation
Primary endpoint: pooled outer OOF ROC-AUC. Report AP, sensitivity, specificity, PPV,
NPV, F2, balanced accuracy, Brier score, log loss, confusion counts and utility.
Use 2,000 resamples of held-out prediction units (matched pairs and unmatched singleton
participants) for conditional AUC intervals; use paired matched-pair resampling for
common-test differences. These intervals do not include model-refitting variability.
Do not use unpaired-cohort DeLong comparisons or infer a causal age effect.

Broad/matched comparisons change training size, prevalence, age/sex distributions
and case spectrum. Common-test comparison controls the evaluation population but
still changes training size and composition. Matching does not remove every
confounder; removing age from predictors does not remove age information in sensors.
No assertion of superiority over prior studies from cross-study AUCs.

## Verification and preservation
Assert exact reproduction of existing eligible model features and labels; retain
original source hashes. Verify cohort nesting, unique participants, complete outer
predictions, train/test group exclusion, identical common-person folds, saved-model
prediction replay and training-only preprocessing statistics. Recompute metrics from
private predictions. Keep participant-level data private. Store new results separately;
do not overwrite historical or previous corrected cohorts and predictions.
