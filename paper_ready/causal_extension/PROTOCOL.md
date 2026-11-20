# Sequential corrected causal-related analysis

Fixed 19 November 2026 after the historical audit and earlier explanation results.
Exploratory extension on previously inspected participants; not preregistration.
Original BMP and earlier corrected runs are immutable inputs.

## Estimands and model choice

A. Age-adjusted within-matched-pair differences in four measured biomarkers are
associations. They do not depend on a classifier and are not selected using SHAP.
B. Counterfactual outputs are changes in a known classifier function f(T(x))-f(x).
C. Structural propagation here concerns known measurement equations, not an
identified effect of modifying a biomarker on Parkinson pathology or future disease.

Primary prediction procedure: the existing expanded full-input family selector,
selecting by mean inner-fold AUC separately in every outer fold. Comparators: all
six already tuned families (logistic regression, random forest, XGBoost, LightGBM,
CatBoost, multilayer perceptron) plus the earlier seven-candidate full-input
selector. Do not choose the largest pooled outer AUC or best explanation result.
The family-specific procedures and selectors share participants and are not
independent replications. All use the same 52 predictors and five pair-disjoint
folds on N=702. This extension does not explain the separate N=492 agent experiment
or all evolutionary variants. Its scope is explicitly the full-input benchmark.

Full input is chosen for the project's multimodal explanatory question and
continuity, not a claim that it is universally best. Existing held-out SHAP remains
the earlier selector's explanation; do not attribute it to the expanded selector.

## Stage 1: biomarker association and refitting checks

Fixed markers: tapping rate, inter-tap interval IQR, walking speed, local vocal
jitter. For each pair form biomarker difference case minus control and age
difference case minus control. Fit delta_biomarker = intercept + beta*delta_age.
The intercept is the equal-age pair-contrast association. Reproduce saved estimates.
Use 2,000 pair bootstrap refits for intervals; also report unadjusted differences.

Compute 5,000 within-pair label-swap permutations, flipping BOTH biomarker and age
contrasts. Report two-sided empirical p-values with +1 correction, and BH-adjusted
q-values across the four markers. This is an exchangeability-based association
check, not randomized treatment inference; residual confounding can violate its
clinical interpretation. Fit 200 independent-noise augmented regressions and 200
80%-pair subsamples, reporting coefficient shifts/distributions, not invented
PASSED flags. Adding random noise is not a test eliminating unmeasured confounding.
Use actual regression refits, not correlations between the outcome and noise.

These transparent pair-aware checks replace the original generic DoWhy refutations.
Do not label them outputs of the DoWhy package or causal ATE estimates.

## Stage 2: full counterfactual quality panel and model sensitivity

Rebuild exactly the earlier label-blind candidate pool: per participant and each
sensor modality, up to 30 nearest training donors, same sex, age within five years,
same block missingness. Whole observed blocks preserve internal identities;
demographics and other modalities cannot change. Candidate ordering and support
are identical for all models. Support cutoff is training pair-excluded nearest-
neighbour distance 95th percentile in median-imputed RobustScaler space. Reject
any changes to a training-constant feature. Record all attempts and support failures.

For each model, evaluate both score threshold 0.5 and its previously selected
inner-training sensitivity90 threshold. Evaluate counterfactual success only among
baseline score-positive queries, reporting PD and control denominators separately.
Return at most three distinct, supported examples below the threshold. First is
minimum mean absolute robust-scaled distance. Remaining examples maximize minimum
distance to selected examples divided by (1+distance to query); deterministic ties.
No change is a failure for positive-query crossing, not an invented counterfactual.

Report attempt-level yield, valid returned examples, number returned, normalized
proximity, unchanged-feature sparsity (absolute tolerance 1e-8), mean pairwise
diversity (undefined for one example), and Local Outlier Factor score from a
20-neighbour training-only fit in scaled space. Report MAD-normalized proximity
using max(MAD,0.1*IQR,1e-8) per feature as a secondary scale. Neither distance nor
LOF establishes clinical feasibility. Validity among returned examples is
conditional by construction and must accompany attempt-level yield.

Report per-feature shifts of the FIRST returned example as descriptions of
optimized examples, never treatment prescriptions. Test the expanded selector's
first example on every comparator using that comparator's own threshold, among
queries positive under both models; retain direction of score change too. No
claim of independent validation across these correlated models.

## Stage 3: specified measurement-equation counterfactuals

Three families, each at requested -20%, -10%, 0%, +10%, +20%:
1. Tapping temporal dilation: multiply tapping rate by 1+r, divide duration and
   all six dimensional interval summaries (mean, median, std, IQR, min, max) by
   1+r; keep counts, alternation, miss ratios, half-rate ratio and interval CV fixed.
   This represents rescaling a recording's time axis, not training a patient.
   Require duration 5–60 seconds and no discarded raw inter-tap interval to enter
   or retained interval to exit the extractor's [0.01,5] second filter. Verify
   this against the original selected tapping samples before accepting the
   summary-level transformation. If that raw check cannot be established, omit
   the tapping arm with a stated reason rather than assume validity.
2. Walking cadence: scale and round steps, retain duration and step length,
   recalculate distance, cadence and speed (same earlier scenario).
3. Walking step length: scale step length and distance, retain steps and duration,
   recalculate speed; cadence unchanged.

Require modified modality variables to lie within outer-training 1st–99th
percentile bounds and pass the common support-distance rule; report all exclusions.
Use each scenario's own support and, primarily, participants passing all four
nonzero levels of that family. All models use the same supported records. Report
score changes by diagnosis with 2,000 pair-bootstrap intervals conditional on
saved models. No result-driven direction selection or widening of support rules.
The third stage is a deterministic structural measurement model, not a learned
biological causal graph. Voice remains covered by joint observed-block substitutions;
independent changes to acoustic summaries are not assumed waveform-consistent.

## Verification and stopping

Check hashes, original cohort identity, inner selection and threshold arithmetic,
pair-disjoint folds, probability replay, failed-attempt denominators, immutable
features, candidate provenance, all task identities, support, diversity/proximity
calculations and cross-model replay. Unit-test label-swap handling, regression
refits, graph propagation, exact raw tapping time rescaling, unchanged variables,
and search failures. Freeze outputs and stop after these arms. Repair code defects
only; preserve prior protocol/results. No new clinical causal-effect claims.

## Reporting boundary

DoWhy and DiCE were historical tools; these corrected implementations retain the
valid research questions, not their package labels. We will name actual algorithms
used. Assumed disease-reversal SCMs, fabricated refuter flags, and causal claims
about agentic/evolutionary gains remain excluded. Refutation checks cannot prove
causal identification: https://www.pywhy.org/dowhy/v0.13/user_guide/refuting_causal_estimates/index.html
