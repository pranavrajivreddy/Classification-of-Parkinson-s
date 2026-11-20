# Single-Claude closed-loop follow-up

20 November 2026. Written before new inference and scores. The user explicitly
authorized this additional run after the earlier final multi-agent experiment.
This is another exploratory follow-up on previously exposed participants, not a
preregistered replication or independent test. All prior outcomes remain retained.

## Scope and cohort

Use the corrected age/sex-matched N=702, preserving the existing 105-pair discovery
(210 participants) / 246-pair evaluation (492 participants) split. Pair and person
overlap across discovery/evaluation is prohibited. This tests full-input feature
augmentation across six fixed classifier families, not all nine modality sets.
No new cohort, labels, recordings, feature-extraction procedure or medications.
All participant records and model fitting remain local. API inputs consist only
of feature definitions, formulas and aggregate discovery results.

## One agent and its decisions

One Claude Sonnet 4.6 agent (`anthropic/claude-sonnet-4.6`) receives the same role
through three rounds, at most four new formulas per round. It reasons about the
measurement definitions, critiques its own earlier proposals and chooses the next
formulas from aggregate local feedback. There is no second proposer or critic.
Full round history is supplied; JSON requests/responses and decisions are saved.
Claude cannot change cases, labels, split assignments, learners, primary objective,
acceptance rule or evaluation thresholds. The local controller validates formulas,
scores proposals, applies the selection rule and freezes evaluation inputs.
This is actual API-mediated feedback, unlike the fixed decisions in the inspected
historical scripts. It does not isolate provider from number of roles or search budget.

Three intended billable requests only, no automatic retries or additional search
rounds after failure. Empty proposals are permitted. Maximum 12 formulas. Claude
sees prior formulas from earlier agent experiments but no evaluation metrics.
Exact syntax-equivalent prior/current formulas are excluded. No arbitrary Python:
use the existing bounded AST (+,-,*,pdiv,logabs,sqrtabs,abs), sensor inputs only,
maximum80 nodes and constant magnitude100. No unobserved neurological constructs.

## Budget and routing

Maximum total reserved spend US$2, including any dispatched request whose charge
is ambiguous. Pin Amazon Bedrock global, no provider fallback, `zdr=true`,
`data_collection=deny`, supported parameters required, no external tools. Preflight
must find the endpoint in OpenRouter's current ZDR list with safety_retention=false.
Provider price caps: input$3/M and output$15/M. Maximum6000 completion tokens and
80000 UTF8 request bytes. Each request reserves (bytes+2048)*3e-6 +6000*15e-6
dollars, conservatively treating every request byte as an input token plus overhead.
Three maximal reservations total below$2. No explicit prompt-cache writes. Stop
on missing usage/cost, unexpected provider/model, incomplete JSON or network error;
never silently switch models or relax privacy. Credentials stay in process memory.
Save actual returned usage; the ceiling is not a spending target.

## Local discovery and controls

Same frozen six-family panel as the earlier multi-agent experiment: LR C1;
RF120 trees/depth5/minleaf3; XGBoost160/depth2/lr.05/minchild3/lambda1;
LightGBM160/leaves7/depth3/minchild15/lr.05/lambda1;
CatBoost160/depth3/lr.05/l2=3; MLP16 units/alpha1/lbfgs/max_iter2000/max_fun50000.
Seeds42, CPU threads1; retain all convergence warnings. Median imputation and
RobustScaler fitted inside every training partition. No feature screening using
evaluation labels. Hard fit or nonfinite-prediction errors stop for code review.

Each formula is scored as one addition to 52 raw inputs with 2x3 pair-grouped CV
(seeds4242,731). Primary objective: equal-probability ensemble mean fold AUC.
Feedback also contains per-family AUC/AP/Brier and fixed0.5 decision metrics;
no validation-optimized threshold metrics enter the feedback.
Eligibility: mean ensemble gain>=.002, positive both repetitions and nonnegative
mean gain for >=4/6 families. At most two candidates ranked by gain then name
form one bundle; evaluate it once with the same rule, otherwise use raw features.
Do not search alternative bundles after a failure or relax the rule to achieve.81.

Automatic control is the pre-existing outcome-blind seed20261120 sequence; take
first K valid transforms, K=number of new Claude candidates scored. Same discovery
folds, learners, gate, two-feature bundle and fallback. This matches numerical
candidate counts, not semantic complexity or adaptivity. Keep rejected proposals.

## Evaluation and inference

After discovery freeze and an independent code audit, evaluate raw, Claude and
automatic arms using identical five outer pair folds(seed42), three inner folds
(seed143), six learners, equal ensemble(PRIMARY) and inner-selected family.
Threshold policies: fixed.5; inner utility; inner sensitivity90; inner specificity60.
No outer label enters model weighting, selection, preprocessing or threshold choice.
Report all existing metrics/confusion counts; threshold changes do not improve AUC.

Primary contrast: Claude-minus-raw mean outer-fold ensemble AUC. Secondary pooled
AUC, all family comparisons and Claude-minus-automatic. 2000 within-fold matched-
pair bootstrap resamples of fixed predictions, conditional/unadjusted exploratory
intervals; not complete training/discovery uncertainty. No superiority assertion
from a positive point estimate alone. Do not compare N492 agent AUC against N702
baseline as a paired gain. Not an independent Claude-versus-DeepSeek comparison.

## Integrity

Before inference test safe AST, budget reservations, privacy route and selection
logic. Freeze hashes of code/protocol/cohort/split and original-source audit.
Before evaluation reconstruct feedback and selection; prove N210/N492 partition
equals the corrected N702. Replay every saved outer model, imputation/scaling,
inner family choice and threshold; independently recompute aggregate metrics.
Preserve original BMP and all earlier result packages; no manuscript claims change
until the run is complete, even if the outcome is unchanged or worse.
