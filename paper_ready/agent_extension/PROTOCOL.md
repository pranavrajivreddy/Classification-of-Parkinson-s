# Codex-controlled discovery and hybrid extension — 18 November 2026

This is a new exploratory internal study after earlier whole-cohort results were
seen by the investigator and this assistant. It is not preregistered independent
confirmation. No Claude or Gemini calls will be used. A connectivity-only Claude
probe failed before this protocol and sent no research data.

## Controller and evidence
The controller is the current ChatGPT/Codex conversation. Do not invent an API
model identifier or claim a separate stateless model instance. Store the exact
aggregate discovery inputs, executable feature proposals, rationales, evaluation
feedback, revisions and final frozen specification. The local runner performs
training/evaluation; the conversational controller proposes and revises features.
This is an interactive, tool-mediated LLM loop, not a standalone unattended API
agent. Replaying stored decisions reproduces computations, not fresh LLM responses.

## Separation fixed before new discovery
Use the corrected 702-person cohort. Randomly reserve 105 of 351 matched pairs
(seed 20261118) for discovery: 210 people. The remaining 246 pairs / 492 people
form a separate evaluation pool. Never expose new evaluation outcomes or metrics
to the controller until proposals and selection rules are frozen. Both groups
retain the original exact-sex/age matching. Earlier aggregate exposure remains
a limitation and cannot be undone by this split.

The discovery routine accepts discovery data only. Allow two proposal rounds,
at most 12 formulas each. Score single additions and per-configuration bundles
with the same 3 grouped folds, fixed logistic C=1, and training-selected utility
thresholds (development metrics can be optimistic due to adaptive reuse).
The controller may revise/reject proposals from this feedback; all proposals
and failures remain in the record. Features use a restricted arithmetic grammar
over measured predictors; no labels, IDs, medication, future engagement or device
metadata. No arbitrary generated Python is executed.

## Evaluation after freeze
All nine modality configurations on the 492-person pool, five outer grouped
folds, three inner grouped folds, seed42. Discovery participants are excluded
from ALL evaluation model fits to make the sample accounting simple.
No feature or hyperparameter changes after evaluation results are examined.

Stages on common folds:
1. Plain conventional baseline: fixed LR C=1 and the original seven-candidate
   inner-AUC selection.
2. Programmatic utility search: the seven candidates times positive-class
   weights 1,2,3, select by inner maximum historical utility; exact thresholding.
3. Agent-guided features: original plus frozen Codex-proposed features with
   fixed LR C=1. This isolates feature engineering; it is not direct LLM diagnosis.
4. Agent/programmatic hybrid: the same weighted candidate search on augmented
   inputs, alongside seven-candidate inner-AUC selection.
5. Separate NSGA-II, BGA, GP, DE searches on both conventional and augmented
   inputs, with the existing fixed budgets, optimizer seeds 11,29,47 and matched
   equal-weight/logistic/random/no-selection comparators.
6. Full-modality integrated BGA -> GP -> ensemble -> DE/NSGA hybrid, with and
   without agent features. Refit supervised transforms inside each inner fold;
   fit BGA using an additional training-only grouped layer. Do not call parallel
   separate algorithms a sequential integrated hybrid.

Preserve .5, utility, sensitivity90 and specificity60 policies; NSGA uses its
training-front operating point. Utility is U=1-(4FN+2FP)/N, effective2:1 error loss.
A separate pure3:1 error cost is reported. These are assumed preferences, not
validated clinical cost ratios.

## Reporting
Store every fold, model/weight choice, mask, GP expression, optimizer seed,
threshold and participant prediction privately. Summaries show all seeds and
paired conditional intervals (2000 pair-bootstrap draws) on the same 492 people.
No comparison of these AUCs to the earlier 702-person AUC as if only the algorithm
changed. Primary contrasts: augmented vs raw fixedLR; augmented vs raw programmatic;
augmented vs raw DE; integrated vs standaloneDE. No superiority/equivalence from
point estimates or unadjusted multiple testing. No live-Claude/Gemini comparison.

