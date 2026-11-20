# Release adaptations and limits

The original research workspace is preserved. This folder is a curated derivative, not a copy of private experiment archives.

- Removed the dependency on the historical 842-person file and an absolute path to the original workstation. These were used for historical auditing, not current model inputs. The audit-only membership flag is retained as false for schema compatibility.
- Removed evolutionary calls from the baseline runner and evolutionary report branches. Core predictive calculations are retained.
- Extracted the unchanged `allmetrics` helper into `research/reporting_metrics.py`, avoiding imports of an unrelated historical transfer experiment.
- Removed historical-script hash checks from Claude reports and signatures; current input, split, model, discovery and prediction checks remain. Release reports no longer claim those historical scripts were verified.
- Git metadata gracefully falls back to `unversioned-release` before the user initializes a repository.
- Added a preparation helper for deterministic fold assignments and split initialization. It explicitly identifies frozen-formula replay; it does not claim to regenerate the original interactive conversation.
- Kept historical formula definitions in `configs/` separately from fresh scripted output folders. No original API conversations or private discovery feedback are shipped.
- Removed redundant plotting checks against manuscript-export CSV copies; plots still read regenerated numerical result tables. Some plot assertions are specific to the published study and should not be silently relaxed for a new cohort.
- Default tests exclude checks requiring restricted participant artifacts; synthetic checks remain active. One automatic-candidate test uses synthetic column definitions instead of loading real participants.

`SOURCE_PROVENANCE.json` records original source hashes. `RELEASE_MANIFEST.json` records this package's file hashes. Packaging adaptations invalidate old source signatures; run in a clean output directory and do not mix old completion markers with release runs.

Paid API runs are nondeterministic. Aggregate-feedback prompts, explicit selection rules and bounded clients are included, but reproducing a past language-model trajectory exactly is not promised. Host-provider availability and installed library support may change.

Tests run during release preparation cover numerical components, isolation interfaces, syntax/formula restrictions and bounded client construction. They do not rerun full raw extraction, all nested fits or paid discovery. A completed full reproduction still needs authorized data and considerable compute.

The standalone configuration-search client starts its own cost ledger. Its report verifies that ledger rather than requiring private logs of historical failed transport attempts. The per-run cap and three-round search remain unchanged.
