# SMRITI-AI corrected analysis plan — 2026-11-17

This is a timestamped plan for a corrective analysis AFTER extensive exploratory
work, not prospective preregistration and not an independent confirmation study.
The historical reports/results are retained; none are approved publication results.

## Population and measurements (fixed before corrected model evaluation)

Reconstruct from local mPower tables and raw files. One record per participant;
earliest non-null recording reference per task, preserving entire rows (never
groupby.first across columns). Use earliest valid diagnosis survey, exclude
participants with contradictory diagnosis answers. Outcome is self-reported
professional diagnosis, not adjudicated PD. Primary age range 35–75 inclusive,
known male/female survey category, all three tasks with technically usable local
data. Retain an exclusion ledger and broader unmatched eligible population.
Match 1:1 without replacement within exact survey gender, age caliper 2 years,
maximum cardinality then minimum total age distance. Do not force N=842.

Tapping: event timestamps and recorded task duration; missing files are missing,
not zero performance. Dynamic half-task rate ratio. Pedometer: final cumulative
steps and distance over its time interval; exclude reset/nonmonotonic sequences;
distance/step is named step length. Do not claim accelerometer gait variability
or tremor. Audio: deterministic 16 kHz mono conversion; Praat acoustic measures
and MFCCs; QC duration >=2 seconds, finite F0, voiced fraction >=0.30. These are
technical choices, not clinically validated cutoffs. Archive QC metadata.
No future session counts, medication, diagnosis dates, device, or missingness
flags in primary predictors. Device, medication, task lag, and QC are audit fields.

## Evaluation

Primary: 5 outer folds, 3 inner folds, outer seed 42. Matched pairs are grouped
to preserve the sampling design. Repeat sensitivity analyses at seeds 137 and
2027. No test labels may enter preprocessing, feature selection, model choice,
calibration, ensemble weights, or thresholds. All candidate comparisons use the
same inner splits. Save immutable fold assignments, selections, outer predictions,
candidate scores, runtime, hashes, and environment. Final full-data fits are
deployment/reproducibility artifacts, not new validation results.

Primary comparison: frozen engineered features versus conventional sensor+
demographic features with identical model candidates, inner AUC selection and
threshold policy. This tests a fixed feature hypothesis, NOT autonomous LLM
discovery. Primary endpoint: paired delta ROC-AUC. Secondary: specificity under
an inner-selected >=90% sensitivity policy, calibration, and referral burden.
Three primary-family contrasts: engineered vs conventional, sensor fusion vs
tap-only, and DE vs equal-weight ensemble; Holm adjustment where p-values are
reported. Report effect intervals first. No significance claims from five-fold
Wilcoxon or repeated-seed pseudo-replication.

Model candidates: L2 logistic C=0.1,1,10; elastic-net logistic C=1/l1_ratio=.5;
RF 120 trees/depth 5; GBDT 100 trees/depth 2/learning rate .05; RBF SVM C=1.
No resampling or class balancing is needed in the matched cohort. Imputer median
and RobustScaler fit within each inner train. All-missing columns retained as 0.
Calibrated sigmoid is an explicitly separate train-only analysis.

Operating policies: default .5; maximize the historical utility
U=1-(4FN+2FP)/N; maximize specificity subject to inner sensitivity >=.90;
maximize sensitivity subject to inner specificity >=.60. All-positive and
all-negative options included. Threshold selection never uses outer outcomes.
Utility has an effective FN:FP loss ratio of 2:1, not 3:1. Also report pure 3:1
error cost and hypothetical prevalence scenarios .01,.05,.10,.20,.50. Decision
curves on matched prevalence are internal only; transported curves assume prior
shift and are scenarios, not clinical validation.

## Evolutionary analyses

Use all conventional primary features and fixed outer partitions. Optimize only
inner predictions. Compare NSGA-II joint weights/thresholds against equal-weight
and single-model exact threshold sweeps and equal-budget Dirichlet random search.
Report an empirical nondominated set, not a true/continuous frontier. DE uses
normalized nonnegative weights on the full simplex; compare equal and logistic
stacking. BGA uses a fixed logistic learner, compares identical-budget random
subsets, and a no-selection learner; selection criterion inner AUC minus .005
times fraction selected. GP supervised transforms are fit only on training
partitions, with protected arithmetic, parsimony, expression/duplicate reporting.
Primary optimizer seed 11, stability seeds 29 and 47; all seeds reported, none
selected by outer performance. No guaranteed noninferiority claim; exploratory
margin .02 AUC, explicitly not retrospectively confirmatory.

## Uncertainty, robustness, and publication gates

Paired matched-pair bootstrap of fixed seed-42 outer predictions (2000 draws)
gives conditional intervals, not full training/search uncertainty. Report outer
fold dispersion and repeat sensitivity separately. Subgroup intervals are
exploratory; suppress interpretations in small cells. Include demographics-only,
device-only, engagement-only, missingness-only diagnostic controls; these controls
are not deployable primary models. Sensitivity: age-restricted historical cohort,
short task lag, and full eligible unmatched data when counts permit. Permutation
negative control reruns the selection procedure; preserve exchangeability by
swapping labels within matched pairs. No causal claims from predictive scores.

Independent clinical validation, adjudication, ethics determination, data-use
permission, and original LLM prompts cannot be created by code. Record these as
unresolved evidence gates unless documentation is supplied. A two-paper plan is
conditional on a distinct, supported contribution for each; null findings stay.
