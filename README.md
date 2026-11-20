# Smartphone-based Parkinson's classification research code

Code accompanying the study of demographic controls, model interpretation and agent-assisted model development using mPower smartphone measurements.

This is a **code-only release**, with method definitions and formula configurations. It contains no participant records, recordings, saved models, prediction files, credentials or private agent transcripts. Obtain mPower authorization independently before using study data. This is retrospective research on self-reported professional diagnosis, not a validated diagnostic or treatment system.

## Scope

- Deterministic recording selection, feature extraction and age/sex matching.
- Nine input configurations; nested tuning of six model families, calibration and training-selected thresholds.
- Broad/restricted/matched cohort comparisons, including 20 equal-size control draws and common-population evaluation.
- Held-out SHAP, descriptive matched biomarker comparisons, supported donor substitutions and measurement scenarios. These are not identified clinical causal effects.
- Frozen conversational feature formulas; scripted DeepSeek/Claude feedback loops; reversed discovery/evaluation split; bounded model/input/parameter search with random-search and Optuna comparators.
- Synthetic tests and separate opt-in checks requiring authorized study artifacts.

Evolutionary algorithms and medication experiments are excluded from this release's executable workflow. Historical protocol documents retain their original context where necessary; inclusion does not make those other experiments part of this paper.

## Install and test

Python 3.12 was used for release validation. Create your own environment **inside this folder**, then install the pinned requirements:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python tools/check_release.py
```

Raw audio extraction additionally requires `ffmpeg` on your PATH. LightGBM may require an OpenMP runtime on macOS. No test in the default test run intentionally contacts a model provider or downloads data. Tests marked `private_data` are skipped unless `--private-data` is supplied.

## Reproduction

Read [DATA_SETUP.md](DATA_SETUP.md), then [REPRODUCE.md](REPRODUCE.md). Commands depend on separately authorized local inputs. **A code-only download cannot recreate the exact 702 participants without the original local participant-pool manifest and source recordings.** We do not distribute that manifest. Reconstructing another pool is a new analysis, not an exact reproduction.

The release has packaging adaptations, documented in [PORTABILITY.md](PORTABILITY.md). Model mathematics and selection rules were retained where exported, but full study fitting was not rerun during packaging. Do not treat the release validation report as a new confirmation of the paper's numerical results. Source hashes in the original research manifests will differ from adapted release files.

## Credentials and API runs

No Synapse login is used by the released analysis scripts; downloads are separate. Optional agent scripts accept `OPENROUTER_API_KEY` from the environment or a hidden interactive prompt. Do not place a key in source files, notebooks, command examples or commits. Runtime request/response logs and outputs are ignored by Git.

Agent discovery is a new paid run, not a deterministic replay of the historical language-model responses. Historical model names, provider routes, pricing caps and stopping rules are retained; availability can change and clients stop rather than silently switch. Review the cap in the corresponding API module before explicitly starting discovery. `REPRODUCE.md` separates offline commands from paid API commands.

## Publishing

Push **this folder only**, not its parent research workspace. Run `python tools/check_release.py` immediately before the first push. The allowlist scan detects added files and common credential patterns, but cannot replace review of later changes. Keep generated data and models outside Git even if a filename is not covered by `.gitignore`.

No repository has been created or uploaded by this packaging step. Add the final manuscript citation, author information and an author-approved code license when publishing; no third-party data license is granted by this folder.
