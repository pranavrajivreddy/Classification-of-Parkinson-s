# Frozen exploratory follow-up — 19 November 2026

Fixed before new fits. Earlier project results informed this design; it is not
preregistered or independent confirmation. Execute the entire specified set once,
without adding seeds/models/thresholds after viewing results. Private outputs stay
under paper_ready/private/controlled_followup. Preserve original and prior outputs.

## A. Equal-size fixed-case sampling
Use the existing 351 matched cases in every arm. Controls: existing 351 matched,
or 351 randomly sampled without replacement from broad/restricted pools. Within
each existing outer fold and sex stratum, sample the exact case count. All arms
therefore share case identities, N=702, class/sex composition and outer training
sizes. Twenty sampling seeds 2026111900–2026111919; both broad and ages-35–75 controls.
Use full, no-age, sensors and age-only configurations and the existing seven-model
library, inner3 seed143 and outer5 aligned folds. Inner groups preserve any existing
matched pairs; imputation/scaling fitted within training. Reuse matched reference
fits only after hash/feature/fold verification. All candidate OOF probabilities and
selected models will be retained. Evaluate both own-cohort OOF and the same original
702 matched participants, excluding the entire global test fold from each model's
training set, including matched test people absent from a sampled arm.
Primary: matched-reference full AUC minus mean common-test full AUC over 20 broad
draws. Secondary: restricted, other configurations, own-cohort AUC. Sampling-draw
SD/range are stability summaries, not independent-replicate inference. Conditional
paired intervals resample the same 351 pairs jointly across all fixed predictions
(2000 draws). Do not average probabilities then substitute ensemble AUC for mean
per-draw AUC. No causal age-effect, external-validation or population-risk claim.

## B. Compression on N=702
Consolidate all existing BGA, equal-evaluation random-subset and full-LR predictions
for seeds11/29/47. Add two conventional selectors: top absolute L1-logistic(C=1,
liblinear) coefficients after training-only scaling; and ANOVA F ranking. Select
exactly the number of features retained by BGA in that outer fold/seed (20–28 from
prior runs); count is a training-selected hyperparameter reused adaptively, not an
independent inner-validation decision. Ranking is refit separately inside each inner
training fold; downstream LR C=1 max_iter3000 is identical. No extra hyperparameter
search. Four original threshold policies plus exact inner sensitivity85 for new
selectors. Existing BGA comparisons use the original four policies only (its inner
scores were not saved). Evaluate all seeds and conditional paired AUC differences;
record feature count, all-modalities retention and selector stability. Do not infer
non-inferiority from non-significance or choose a margin after seeing results.
Selected-column count is not a measured runtime or acquisition-burden reduction.

## C. Threshold/search controls on N=702
Reuse the existing seven base learners' inner/outer OOF score matrices, checking
labels/folds/hashes and reproducing saved equal-vote scores. No outer labels enter
search. Optimize weight vectors with exact inner threshold enumeration for maximum
specificity subject to sensitivity>=0.85 (ties sensitivity, then threshold).
Compare equal weights, existing inner-AUC best single model, logistic stacking(C=1),
DE weights and random weights. DE uses a fixed 40-vector initial population including
equal weights, max19 generations, max800 scored vectors, no polishing. Random search
uses the identical initial population plus independent simplex draws to exactly
match the actual DE evaluations, with seeds11/29/47. Both use normalized nonnegative
weights, identical exact objective and identical tie rules. Actual evaluations and
runtime are recorded, not assumed equivalent runtime. NSGA-II uses weights+threshold,
population40 and20 generations (800 evaluations), optimizing sensitivity/specificity.
Compare native-front sensitivity85 point against selecting from the same front's
weights with exact threshold enumeration; also compare simple controls. Report
fallbacks, front size and actual evaluations. NSGA's native and exact threshold
postprocessing differ in computational opportunity; do not call these equal-runtime.

Use default, utility, sensitivity90, specificity60 and sensitivity85 threshold policies
for all exact-policy methods. For native NSGA retain its training-selected threshold.
Primary: DE_exact minus random_exact specificity at training sensitivity85; always
report sensitivity difference and outer target attainment alongside it. Secondary:
DE vs equal, NSGA exact vs native/equal, AUC and balanced accuracy. A threshold-only
change cannot improve AUC. No posthoc outer threshold adjustment to meet targets.
All-seed mean metric differences use joint participant-pair bootstrap conditional on
fitted predictions; optimizer seeds are not independent study samples. No claim of
clinical benefit, universal superiority or multiplicity-adjusted confirmation.

## Verification gates
Before fits: hashes, unique subjects, quotas, exact case identities, pair/fold
exclusion, feature dimensions. During fits: inner coverage, group boundaries, finite
probabilities, saved-model replay. Before reporting: source signatures, every selected
outer model replay, recompute training-only preprocessing and selection where saved,
threshold counts, independent confusion-matrix reconstruction, equal-score AUC under
different thresholds, equal DE/random budgets, and prior/source file preservation.
