# Fixed AUC-focused conversational discovery follow-up — 19 November 2026

Exploratory follow-up requested after seeing prior agent results. Existing N=492
results are already known to the investigator and controller. This is not a new
untouched holdout or independent confirmation. Preserve the earlier experiment.

Question: can a bounded AUC-directed feature-discovery loop improve ranking
relative to the same learners without additions, and a simple automated-feature
control? No guaranteed improvement, target AUC, or continuation until success.

## Fixed data and objective
Reuse original 210 discovery / 492 evaluation partition, verifying source hashes,
participant/pair disjointness and exact coverage of corrected N=702. Full 52-input
configuration only. No medication, participant IDs, future data or new cohort.
Discovery reader loads only discovery data. Controller sees aggregate feedback.

Primary target: mean within-fold ROC-AUC, no threshold/utility-based selection.
Four fixed candidates: logistic regression C=0.1 or 1 (max_iter=3000) and XGBoost
max_depth=2 or 3 (160 trees, learning_rate=.05, min_child_weight=3, reg_lambda=1,
subsample=1, colsample_bytree=1, hist, seed42, one thread). Balanced class weights
unchanged. All imputation/scaling refit inside each training split. No calibration.
These are controlled learner comparisons, not the full previous Optuna benchmark.

## Bounded conversational loop and automatic control
Two rounds of at most six new transparent formulas each, over measured inputs,
using the existing restricted arithmetic grammar. Current ChatGPT/Codex
conversation proposes, reads first-round feedback, revises once. No verified
backend model ID is asserted. Formula rationales are hypotheses, not mechanisms.
Evaluate single additions with all four candidates on two fixed repetitions of
3 pair-grouped discovery folds (seeds4242,731). Preserve all results and failures.

Automatic-feature control: 12 sensor inputs drawn uniformly without replacement
from sorted feature names with seed20261119, each transformed log(1+abs(x)). No
outcome-informed choice of input names. Same candidate evaluation and bundle
selection budget. This is a limited log-transform control, not an exhaustive
non-LLM optimizer or proof of LLM necessity.

For each feature, average AUC difference versus raw baseline across the four
learners and six validation folds. Eligible if average gain >=0.002 and gain is
positive in both repetition averages. Sort descending mean gain, then name.
Propose a bundle of the first two eligible features (one/zero if fewer eligible).
Evaluate that bundle using the same discovery folds. Retain only if bundle gain
>=0.002 and positive in both repetitions; otherwise retain no addition. No further
bundle search, dropping features after seeing bundle results, or changing cutoffs.
Run this same procedure independently for agent and automatic-control features.
Freeze formulas and full provenance before new evaluation runs. Adaptive discovery
scores are development feedback, not unbiased performance estimates.

## Evaluation
Four arms: raw52; prior utility-selected full-model feature; new AUC-selected
agent bundle; automatic-control bundle. Same 492 participants and five original
outer pair-grouped folds (seed42). Within each outer train, fit the same four
candidates on three inner pair-grouped folds (seed143). Choose the largest mean
inner-fold AUC, deterministic name tie-break. Also retain each fixed candidate's
outer predictions for matched-learner contrasts. Outer test labels are not passed
to fitting/selection. Save fitted models, inner scores/predictions, choices and
preprocessor parameters privately. Do not change proposals after evaluation.

Primary comparison: new agent vs raw for the inner-selected procedure, using
mean five outer-fold AUC. Secondary: pooled OOF AUC, prior feature, control bundle,
matched-candidate differences, and standard classification/calibration metrics.
Threshold policies (.5, utility, sensitivity90, specificity60) are selected solely
from outer-training inner-OOF predictions after AUC selection. They cannot establish
independent clinical utility or rescue a failed AUC endpoint.

Report 2,000 conditional paired-bootstrap intervals, sampling matched pairs within
each outer fold, for mean-fold and pooled AUC differences. Model/discovery fitting
is not repeated; intervals omit training/discovery uncertainty. No confirmatory
p-value or multiple-comparison superiority claim. A threshold-target label does
not guarantee the target in held-out data.

## Verification and stop
Test rank-AUC handling, selection/fallback/ties, formula restrictions, training
preprocessing and group separation. Replay every saved outer prediction, recompute
all aggregate metrics and thresholds, verify model/data/code/specification hashes
and test/training disjointness. Report optimization and computational limitations.
Stop after two rounds, one frozen evaluation and verification regardless of result.
No original run or endpoint is replaced. Further independent claims require new
appropriately untouched participants and repeated independently initialized agent
trajectories. Do not attribute gains from a different learner to the agent.
