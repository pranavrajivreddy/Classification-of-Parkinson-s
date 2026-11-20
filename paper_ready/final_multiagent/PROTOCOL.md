# Final bounded multi-agent feature study — 20 November 2026

Written before new API proposals, discovery scores or evaluation fits. This is
the final authorized exploratory agent experiment. Prior outcomes on these
participants were already inspected. Not preregistration or independent testing.
No target AUC is a stopping criterion; no extra rounds after the specified run.

## Estimand and boundaries

Can a two-round, role-separated API discovery procedure select sensor feature
transformations that improve full-multimodal discrimination across a fixed
learner panel? Same existing N=210 discovery / N=492 evaluation split. Discovery
participants excluded from all evaluation model fits. Matched pairs kept intact
in every split. Six fixed families, raw versus selected agent features versus
budget-matched automatic features. This does not isolate number of agents from
provider choice, proposal budget or role instructions; no multi-agent-superiority
or DeepSeek-versus-ChatGPT claim.

## API procedure, privacy and budget

deepseek/deepseek-v3.2 through OpenRouter, pinned to DeepInfra with no provider
fallbacks; require supported parameters, data_collection=deny, zdr=true. Request
JSON, temperature .2, reasoning disabled, maximum4096 completion tokens. Actual
returned model/provider/usage saved. DeepInfra's published inference policy says
no storage/training without consent; OpenRouter privacy controls are routing
policies, not a guarantee against every legal exception or account opt-in.

Two independent roles (within-modality measurement and cross-modal statistics)
each propose at most4 sensor-only numerical formulas per round. A third critic
reviews proposals before local scoring, can approve/veto, cannot add or repair
formulas. Second-round proposers/critic receive aggregate discovery feedback and
prior proposals only. At most16 candidates in total. No evaluation feedback is
sent. Raw signals, participant records, identifiers, per-person predictions,
timestamps, medication records, secrets and local paths never enter prompts.
External tools/code execution are disabled. Formulas use the existing safe AST
language (no eval/exec), only sensor inputs, bounded complexity and finite constants.
All exact prompts/completions and rejected proposals are recorded locally.

At most8 billable request attempts: six intended plus two optional JSON-format
repairs (no new scientific rounds). No automatic retries after network/HTTP
failure or ambiguous charge. Input body limit40000 UTF8 bytes; per-request
reservation (bytes+4096+1024)/1e6 dollars under provider input/output caps$1/M.
Total conservative reservations <=$0.50, even if actual reported cost is lower.
Credentials read without echo into process memory, never persisted. Public model
and authenticated key metadata checks precede inference; no account-setting changes.

## Discovery and selection

Three pair-grouped folds, repeated with seeds4242,731. Fixed models:
logistic regression C=1/max_iter3000; random forest120trees/depth5/minleaf3;
XGBoost160trees/depth2/lr.05/minchild3/lambda1; LightGBM160trees/leaves7/
maxdepth3/minchild15/lr.05/lambda1; CatBoost160iterations/depth3/lr.05/l2=3;
MLP one16-unit layer/alpha1/lbfgs/max_iter2000/max_fun50000. Seed42, CPU threads1.
All applicable preprocessing fits separately inside training folds. No tuning
of model parameters based on API proposals or evaluation scores. Convergence
warnings are recorded, never used to discard an unfavorable result. Any hard
fit or nonfinite-prediction error stops the run for implementation review.

Each valid, critic-approved, unique candidate is evaluated as a single addition
to the same52 inputs. Exact formula duplicates across this experiment are rejected;
past formulas are disclosed in the prompts to discourage repetition, not treated
as new discoveries if repeated. Primary discovery score is mean fold AUC of the
equal-probability ensemble. Eligible: gain>=.002, positive gain in both repetitions,
and nonnegative mean gain in at least4/6 individual families. At most2 candidates,
sorted by gain then name, form a bundle. Bundle is scored once with the same
criteria; if it fails, the arm falls back to raw (no alternate bundles searched).
No thresholds, labels or data records are changed to meet desired performance.

Automatic control: seed20261120 defines an outcome-blind16-candidate sequence
of8 log/square-root sensor transforms and8 within-modality pair products/ratios.
Take the firstK unique valid controls, where K is the number of valid unique
agent candidates actually scored. Identical folds, learners, selection criteria,
bundle size and fallback. This matches numerical candidate budgets, not semantic
complexity or adaptive feedback. Zero accepted agent candidates impliesK=0.

Optional descriptive learning curves are fixed here: discovery-only LR/XGBoost,
40%,70%,100% of each training fold's pairs, nested subsets seed20261120+fold+
repeat; same validation partitions, train and validation AUC. These cannot
identify label noise or the irreducible performance ceiling and do not guide
additional candidates or evaluation choices.

## Frozen evaluation and outcomes

After both rounds, control and bundle decisions, freeze all formulas and hashes.
Evaluation uses same five outer pair folds seed42; three inner pair folds seed143.
Every arm fits six candidates in inner training and full outer training. Report
every family, equal-probability ensemble (PRIMARY), and inner-mean-fold-AUC family
selector. Four threshold policies: fixed.5, training-selected utility,
training-selected sensitivity90, training-selected specificity60. Thresholds use
inner OOF only. Utility=1−(4FN+2FP)/N; separate3:1 error cost also reported.

Primary contrast: mean outer-fold ensemble AUC, agent minus raw. Secondary:
agent minus automatic and automatic minus raw, plus all per-family contrasts.
Pooled AUC is secondary. Report all metrics, including average precision, confusion
counts, sensitivity, specificity, PPV,NPV,F1,F2,accuracy,balanced accuracy,MCC,
Brier,log loss,utility,error cost. No guarantee of simultaneous improvement.
2,000 matched-pair bootstraps within outer folds, fixed predictions; intervals
are conditional, exploratory, unadjusted, and omit refitting/discovery uncertainty.
Report no superiority if evidence is inconclusive; don't select a headline family
after seeing its outer score. No clinical deployment or early/prodromal claim.

## Verification and preservation

Test safe formula parsing, training/test separation, deterministic selection and
budget/privacy payload controls before live calls. Check input/split/protocol/code
hashes at freeze/evaluation; replay all saved outer models, training medians/scales,
inner selection and thresholds; reconcile every aggregate metric with predictions.
Verify equal arms reproduce identical predictions. Save requests/response hashes,
request attempt ledger/cost and convergence warnings. Preserve original BMP and
all prior frozen result packages. Stop after the predefined report, irrespective
of effect size or whether AUC reaches0.81. Writing proceeds using the actual result.
