# Authorized local data setup

The release deliberately contains no data downloader and no Synapse token. Obtain access under your own approved mPower data-use terms. Raw and participant-level data must not be added to a public repository.

The extraction code expects the following local layout beneath the repository root:

```text
RealDementiaData/
  processed/mpower_multimodal.parquet
  mPower/
    tables/mpower_demographics.csv
    tables/mpower_tapping.csv
    tables/mpower_walking.csv
    tables/mpower_voice.csv
    raw/tapping/<fileHandleId>.json
    raw/walking/<fileHandleId>.json
    raw/voice/<fileHandleId>.m4a
```

`processed/mpower_multimodal.parquet` supplies the pre-existing study pool through its `healthCode` column (one unique participant identifier per person). This private pool was assembled before the corrected analysis. Do not substitute the entire mPower population and call it the same study. The code reconstructs the predictors from raw recordings, rather than trusting precomputed feature values in this input.

All four tables require `healthCode`, `recordId` and `createdOn`. The demographics table also requires `professional-diagnosis`, `age`, `gender`, and `smoked`; `diagnosis-year` is optional descriptive metadata. Sensor file-handle columns are:

- Tapping: `tapping_results.json.TappingSamples`, plus `tapping_results.json.startDate` and `tapping_results.json.endDate` in milliseconds.
- Walking: `pedometer_walking_outbound.json.items`.
- Voice: `audio_audio.m4a`.

The optional `phoneInfo` and `medTimepoint` fields are descriptive and are not model predictors. Tapping JSON contains events with `TapTimeStamp` and `TappedButtonId`; cumulative pedometer JSON contains `startDate`, `endDate`, `numberOfSteps` and `distance`. See `research/data.py` for the exact quality checks and feature definitions.

The earliest referenced row is selected per modality. Missing/invalid selected records are not replaced using later sessions. Participant-level reconstruction, cached audio features, pairs, folds, fitted models and predictions are generated under `paper_ready/private/` or `paper_ready/cache/` and must remain local.

The fixed study expects 702 matched participants. Exact replication additionally requires the same source tables, selected recordings, original pool and software versions. Participant identifiers, selected file handles and exact pool membership are intentionally not included in this public package. The compatibility field `in_historical_842` is set to false in the release because that audit-only membership list is excluded; it is never a predictor.
