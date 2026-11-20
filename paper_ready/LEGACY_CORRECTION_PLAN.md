# Historical dependency audit and correction plan

18 November 2026. Scope: the single mPower Parkinson's classification manuscript,
including conventional prediction, demographics, conversational discovery,
evolutionary methods and explanations. Earlier dementia/smart-home experiments
are inventory items, not evidence for this endpoint. No original BMP modifications.

1. Inventory historical source/results and current downstream uses. Inspect
   source rather than accepting report labels. Preserve input hashes and originals.
2. Trace corrected data provenance: historical processed pool is used for source
   participant eligibility, historical N=842 membership only for descriptive
   overlap; numerical historical predictions/metrics must not feed current fits.
3. Classify every relevant experiment family: retained/reverified, superseded,
   replaced, excluded from this manuscript, or awaiting new evidence. Document
   specific failures and required work. Do not imply all old scripts were rerun.
4. Replace demographic claims with exploratory nested age-only/full/no-age/sensor
   comparisons on corrected matched and unmatched eligible cohorts. Fixed 5 outer
   / 3 inner grouped folds, seed 42, existing seven-model selection, existing
   thresholds, fresh inner preprocessing; no search based on outer results.
5. Replace causal-effect claims with four descriptive paired biomarker differences
   (tap rate, interval IQR, walking speed, voice jitter). Report raw pair differences
   and within-pair age-adjusted intercepts with 2,000 pair-bootstrap intervals.
   These are post-audit exploratory associations, not treatment effects. No causal
   estimator or counterfactual cure simulation is promoted as validated evidence.
6. Replace in-sample SHAP-based mechanism claims with held-out full-model grouped
   permutation importance: shuffle each entire feature block jointly within outer
   folds, 30 repeats, report AUC loss. This measures predictive reliance; permutations
   may break cross-block dependence and do not identify causation or acquisition value.
7. Audit saved historical DiCE outputs for arithmetic consistency; retain only
   failure counts/limitations in the audit, not clinical recourse claims. External
   voice and older specialist data require separate reconstruction before use.
8. Add clear legacy notices and execution guards in BMP_CODEX, update the paper
   plan to one manuscript and reconcile evidence-status documents. Preserve old
   reports as history, not editable replacements for canonical evidence.
9. Reconcile new metrics with individual predictions, pair bootstrap calculations,
   hashes and overlapping previously corrected results; rerun existing integrity
   checks where new concerns warrant it. Freeze a dated evidence inventory.

The plan is fixed before these additional analyses, after extensive prior results
have been seen. It is not preregistration or independent confirmation. New results
are exploratory supplements and must not be used to change frozen main/agent models.
Independent clinical validation, causal identification and ethics authorization
cannot be supplied by code changes.
