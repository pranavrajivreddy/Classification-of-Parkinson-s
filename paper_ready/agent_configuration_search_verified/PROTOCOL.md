# Transport compatibility correction within the technical replication

The first request of the technical replication received HTTP 400 before a model
completion: Amazon Bedrock rejected `type: [integer, number]` for a numeric enum.
One identical diagnostic HTTP request reproduced the same error and saved the
provider message; both calls are recorded under agent_configuration_search_v2.
This was an explicit diagnostic exception to the earlier no-HTTP-retry statement.
Neither request produced a model completion or discovery proposal. No evaluation
outcomes were inspected. No new scientific search trajectory was completed.

The numeric schema now uses `type: number` for numeric enums and no redundant
union type for the mixed string/numeric max_features enum. All family dictionaries
remain closed, with every applicable parameter required. Local validation still
requires exact family-specific keys and allowed values. Science protocol unchanged.
The three-round technical replication begins here. No further scientific attempts
or budget extensions will be made. Conservatively reserve BOTH rejected requests'
original maximum charges, in addition to the initial actual cost, before accepting
any new inference call. Combined reservations must remain within USD 2.

---

# Technical replication amendment (before evaluation)

The initial attempt is preserved in `paper_ready/agent_configuration_search`.
Its permissive parameter schema allowed family-incompatible dictionaries: 7/12
agent proposals were invalid. Automatic searches completed 12 valid proposals
apiece. No evaluation model was fitted or evaluation performance inspected.
The provider reported USD 0.163263 for that attempt.

This ONE technical replication changes only the API schema to a union of six
closed family-specific objects, with every applicable parameter required and
every field explicitly typed. Same scientific search space, data, defaults,
folds, seeds, selection rule, 3 rounds x 4 slots, comparators and stopping rule.
No historical evaluation results or initial-attempt feedback are sent in the
replication. Keep both attempts; do not describe their combined spending or
selection history as a single clean first attempt. No further replication after
this one. Combined actual initial cost plus this run's conservative reservations
must stay <= USD 2. No uncertain failed request is retried.

The initial protocol's no-extension rule is amended for this documented technical
repair, not in response to held-out performance. The original text follows and
applies within this replication except for the combined cost accounting above.

---

# Bounded agent-directed configuration search

Fixed before running the search, 20 November 2026. This is an exploratory follow-up,
not a replacement for earlier experiments or a fresh external validation.

## Question and design
Can a recorded Claude agent choose model settings and input-variable rules more
effectively than the same-budget random search or Optuna search?

Reuse the existing matched 702-person split, reversed: 492 people (246 matched
pairs) for discovery and final model fitting; 210 people (105 pairs) for evaluation.
No person or pair crosses this boundary. These people have been investigated in
earlier research. No results from the 210 evaluation people enter API requests,
candidate selection, thresholds or stopping decisions for this experiment.
Unlike earlier evaluation-within-subset CV, this experiment trains on the 492 and
tests the frozen fitted procedures directly on the 210. Their results are not
directly interchangeable with earlier CV metrics.

## Allowed choices
Six families: logistic regression, random forest, XGBoost, LightGBM, CatBoost and
a one-hidden-layer neural network. Nine existing input configurations: demographics,
tapping, walking, voice, tapping+walking, tapping+voice, walking+voice, sensors,
full sensors+demographics. No new measurements, engineered formulas, participants,
labels, exclusions, preprocessing methods or calibration methods may be proposed.

Variable selection: retain all inputs, or the top 10 or 20 by ANOVA F score, capped
at the available input count. Imputation, scaling and supervised variable selection
are fitted afresh using only each training partition. ANOVA scores are selection
heuristics, not causal evidence or inferential p-values.

Discrete parameter choices (all other settings remain at their documented baseline):

| Family | Allowed settings |
|---|---|
| Logistic regression | C: 0.01, 0.1, 1, 10; class_weight: none, balanced |
| Random forest | trees: 120, 240; depth: 3, 5, 8; minimum leaf: 3, 8, 15; max_features: sqrt, 0.5, 1.0 |
| XGBoost | trees: 80, 160, 240; depth: 1, 2, 3; rate: 0.02, 0.05, 0.1; minimum child weight: 3, 8; lambda: 1, 5, 10 |
| LightGBM | trees: 80, 160, 240; depth: 2, 3, 4; leaves: 3, 7; rate: 0.02, 0.05, 0.1; minimum child samples: 15, 30; lambda: 1, 5, 10 |
| CatBoost | iterations: 80, 160, 240; depth: 2, 3, 4; rate: 0.02, 0.05, 0.1; L2: 3, 10 |
| Neural network | hidden units: 8, 16, 32; alpha: 0.1, 1, 10; L-BFGS solver; 2,000 iterations, 50,000 function calls |

The agent may recommend a threshold from 0.20 through 0.65 in increments of 0.05.
This is secondary and does not change AUC. Every selected procedure is also assessed
at 0.5 and at the exact discovery-OOF threshold maximizing specificity subject to
discovery sensitivity >=85%. The latter target is a research operating preference,
not an established clinical requirement or a guarantee on evaluation performance.

## Budget, scoring and selection
One Claude Sonnet 4.6 agent, three calls/feedback rounds, four candidate slots per
round. It receives feature definitions, cohort/protocol descriptions and aggregate
discovery scores only. Random search and Optuna TPE receive the same search space,
six shared default-family scores, twelve additional candidate slots and six fits per
valid candidate. Search is over single-model procedures, not a six-model ensemble.
TPE receives the six defaults as initial observations, with six startup trials.
This matches candidate counts, not wall-clock cost or the information-processing
abilities of the methods. Duplicates consume slots and reuse identical cached fits;
invalid agent candidates consume slots and are reported. No paid retries or fallback
providers. Maximum inference reservation is USD 2, including uncertain requests.

All scores use the same two repetitions (seeds 4242 and 731) of three-fold matched-
pair grouped discovery CV. Primary score: mean of the six validation ROC-AUCs.
Other feedback: training AUC, AP, Brier, sensitivity, specificity, F1, F2 and balanced
accuracy. Thresholds use discovery predictions only.

Shared reference: best mean-AUC default family on full inputs, ties by family name.
A candidate may replace it only if mean AUC gain >=0.002 and mean gain is positive
in each repetition. Highest eligible mean AUC wins, ties by lower mean Brier then
candidate slot. Otherwise retain the reference. This is a new configuration-level
consistency rule, not the earlier feature rule requiring improvement across four
families, and not a calibrated significance test. Both the best searched candidate
and deployed selection are recorded. No target AUC, adaptive budget extension,
evaluation-driven revision or searching until a positive finding.

Freeze all choices and artifact hashes before evaluation. Fit each selected model
on all 492 discovery participants and predict all 210 evaluation participants once.
Report pooled AUC, AP, Brier, log loss, sensitivity, specificity, accuracy, F1, F2,
balanced accuracy, PPV/NPV, confusion counts and utility at all three threshold
policies. Conditional 95% intervals: 2,000 paired bootstrap samples of the 105
evaluation pairs, holding the learned procedures fixed. Primary comparisons are
agent-minus-default, agent-minus-random and agent-minus-Optuna. They are exploratory,
unadjusted for multiple comparisons and do not represent discovery-run variability.

## Verification and provenance
Before inference: schema/range/forbidden-variable tests; train-only preprocessing
test; split/hash/cohort checks; provider and budget checks. During search: save
every request, response, selected configuration, fit warning, fold boundary,
prediction and score; no participant-level records leave the computer. Before
evaluation: replay every discovery model's predictions and verify train-only
imputer/scaler/selector values, selection decisions and API payload privacy. After
evaluation: replay fitted models, reconcile confusion counts/thresholds, bootstrap
paired differences, preserve all results and costs. API model responses are
proposals only; arbitrary code from the provider is never executed.

Sources: OpenRouter ZDR documentation; scikit-learn SelectKBest and nested-CV
documentation; Optuna TPESampler documentation. ZDR requests and provider pinning
are privacy controls, not a claim that no aggregate data were sent to an AI service.
