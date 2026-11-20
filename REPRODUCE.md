# Running the released analysis

Run commands from the release folder, with its environment activated. This is a dependency guide, not evidence that a fresh full run has completed. The release was checked with synthetic tests; no participant data or paid API runs were used during packaging.

## 1. Authorized inputs and deterministic preparation

Follow `DATA_SETUP.md`. The exact study also requires the original local participant-pool manifest, which is restricted and is not distributed. Do not substitute a different pool and describe its results as an exact reproduction.

```bash
python -m research.data --workers 4
python -m research.release_prepare initialize
python -m research.run --workers 4 --stage main
```

The preparation helper verifies the 702-person/351-pair cohort, creates the 210-discovery/492-evaluation split and fold assignments, and installs the released fixed formulas for offline replay. It does not recreate the conversational discovery history. The baseline runner no longer runs evolutionary algorithms.

## 2. Demographic comparisons

```bash
python -m research.cohort_spectrum_data --workers 4
python -m research.cohort_spectrum_run --workers 4
python -m research.cohort_spectrum_report
python -m research.controlled_cohorts --workers 4
python -m research.controlled_report cohorts
```

Keep all specified control draws. The sampled-population and common-population results answer different questions; do not select a favorable draw or label their difference as an identified causal effect of age.

## 3. Expanded nested benchmark

```bash
python -m research.benchmark_extension preflight
python -m research.benchmark_extension freeze
python -m research.benchmark_extension benchmark --workers 4
python -m research.benchmark_extension calibration --workers 4
python -m research.benchmark_report
```

This is substantial computation across input sets, model families and folds. Run once into clean outputs; completion manifests check whether cached results match the current inputs and code. Do not copy completion markers from the private research workspace into this release.

The separate convergence and operating-point checks are available after the benchmark:

```bash
python -m research.benchmark_mlp_convergence freeze
python -m research.benchmark_mlp_convergence run --workers 2
python -m research.benchmark_mlp_convergence report
python -m research.sensitivity85_review
```

## 4. Interpretation and constrained measurement scenarios

```bash
python -m research.explainability
python -m research.explainability_report
python -m research.release_prepare association-reference
python -m research.causal_association
python -m research.causal_model_panel
python -m research.causal_extension_report
```

These require the earlier fitted demographic selector and expanded benchmark. The association reference is generated locally, not downloaded. Module names containing `causal` retain their research names: the outputs are adjusted associations and model-response analyses, **not identified medication effects, disease-risk reductions or clinical treatment recommendations**. Support restrictions and different denominators must remain in reports.

## 5. Offline replay of frozen feature proposals

```bash
python -m research.agent_auc_followup round1
python -m research.agent_auc_followup round2
python -m research.agent_auc_followup freeze
python -m research.agent_auc_followup evaluate
python -m research.agent_auc_report
```

The shipped formulas are fixed inputs to this replay, not new agent discoveries. Follow each protocol's separation of discovery feedback and evaluation. Later exploratory experiments reused this study cohort; these are not independent external confirmations.

## 6. Optional paid scripted discovery

Do not run this section merely to install or test the code. Read the corresponding `PROTOCOL.md` and API module first. Model endpoints, prices and availability may change. Clients require the specified endpoint and privacy settings and fail closed rather than silently changing providers. Supply `OPENROUTER_API_KEY` in your local environment or use the script's hidden prompt; never commit the key. Each experiment has its own cost accounting; running multiple experiments incurs multiple budgets.

For the scripted multi-agent experiment:

```bash
python -m research.final_multiagent_run discover
python -m research.final_multiagent_report audit
python -m research.final_multiagent_run evaluate
python -m research.final_multiagent_report report
```

The single-agent experiments use the same sequence, replacing `final_multiagent` with `claude_single`, `claude_reverse`, `claude_prompt` or `claude_context`. Check the frozen decision and each protocol before evaluation: prompt/context variants can stop at discovery. The reversed design uses 492 discovery and 210 evaluation participants. The other standard design uses 210/492. Absolute AUCs across different evaluation groups do not rank agent quality.

The bounded input/model/parameter experiment is separate:

```bash
python -m research.agent_configuration_search_verified preflight
python -m research.agent_configuration_search_verified discover
python -m research.agent_configuration_search_verified evaluate
python -m research.agent_configuration_search_verified verify
python -m research.agent_configuration_report_verified
```

This release starts a standalone ledger and does not charge prior historical attempts against it. Do not alter the candidate budget, selection gate or evaluation split after seeing evaluation results. The historical protocols describe past decisions; newly generated API responses need not reproduce them.

## 7. Figures and validation

After the corresponding reports exist:

```bash
python -m research.plot_demographic_paper
python -m research.plot_demographic_bars
python -m research.plot_nine_configurations_paper
python -m research.plot_interpretation_paper
python -m research.plot_agent_comparison_paper
```

Plot scripts retain checks specific to the reported study. If another dataset produces different counts or selections, investigate and document the difference; do not disable assertions just to obtain the published figure. Agent plots need the completed comparison reports.

```bash
python -m pytest -q
# Only after the appropriate authorized private artifacts exist:
python -m pytest -q --private-data
```

Generated records, probabilities, models and API logs remain local. Run `python tools/check_release.py` before publishing. It intentionally reports unreviewed files in addition to common credential patterns. Do not update its manifest to approve private outputs.
