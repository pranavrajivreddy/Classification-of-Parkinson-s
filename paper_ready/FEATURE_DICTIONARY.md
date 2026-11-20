# Executed feature dictionary

Generated from the actual feature order. There are 52 conventional and six frozen engineered predictors.
GP X0–X51 refer to median-imputed, RobustScaler-transformed conventional inputs, not raw units.
Each fold has its own fitted medians and scales; an expression alone is not a deployable formula.
Feature names are computational descriptors, not validated disease mechanisms. See METHODS.md for QC.
Acoustic defaults are fixed by requirements.lock.txt and research/data.py; MFCC numbering is one-based.

| Predictor | GP index | Units | Definition |
|---|---|---|---|
| demo_age | X0 | years | Age from selected diagnosis survey |
| demo_gender_male | X1 | indicator | Survey male=1, female=0; other/unknown excluded for matching |
| demo_smoked | X2 | indicator | Binary survey smoked response; unparseable values missing |
| tap_duration_sec | X3 | seconds | Task end minus start metadata |
| tap_count_total | X4 | events | All recorded tapping events |
| tap_count_valid | X5 | events | Events assigned to left or right button |
| tap_count_miss | X6 | events | Events assigned to neither valid button |
| tap_miss_ratio | X7 | fraction | Invalid-button events / all events |
| tap_rate | X8 | events/second | Valid event count / metadata duration |
| tap_half_rate_ratio | X9 | ratio | Second-half valid count / max(first-half valid count,1) |
| tap_alternation_rate | X10 | fraction | Consecutive valid-button events that alternate buttons |
| tap_left_button_ratio | X11 | fraction | Left-button events / valid events |
| tap_flight_mean | X12 | seconds | mean of valid-event intervals retained within [0.01,5] seconds; flight is a historical name, not measured finger flight time |
| tap_flight_median | X13 | seconds | median of valid-event intervals retained within [0.01,5] seconds; flight is a historical name, not measured finger flight time |
| tap_flight_std | X14 | seconds | population SD of valid-event intervals retained within [0.01,5] seconds; flight is a historical name, not measured finger flight time |
| tap_flight_iqr | X15 | seconds | 75th minus 25th percentile of valid-event intervals retained within [0.01,5] seconds; flight is a historical name, not measured finger flight time |
| tap_flight_cv | X16 | ratio | Population SD / mean of retained inter-tap intervals |
| tap_flight_min | X17 | seconds | minimum of valid-event intervals retained within [0.01,5] seconds; flight is a historical name, not measured finger flight time |
| tap_flight_max | X18 | seconds | maximum of valid-event intervals retained within [0.01,5] seconds; flight is a historical name, not measured finger flight time |
| walk_duration_sec | X19 | seconds | Final cumulative end minus common start |
| walk_steps_total | X20 | steps | Final cumulative step count, not sum of packets |
| walk_distance_total | X21 | meters | Final device-reported cumulative distance |
| walk_speed_mean | X22 | meters/second | Final distance / duration |
| walk_cadence | X23 | steps/minute | 60 times final steps / duration |
| walk_step_length | X24 | meters/step | Final distance / final steps; device-derived, not stride length |
| voice_duration_sec | X25 | seconds | Full decoded recording length |
| voice_f0_mean | X26 | Hz | Praat mean pitch over full recording; pitch range 75–500 Hz |
| voice_f0_std | X27 | Hz | Praat pitch standard deviation over full recording |
| voice_jitter_local | X28 | percent | Praat jitter local times 100; period bounds .0001–.02 s, maximum period factor 1.3 |
| voice_jitter_rap | X29 | percent | Praat jitter rap times 100; period bounds .0001–.02 s, maximum period factor 1.3 |
| voice_jitter_ppq5 | X30 | percent | Praat jitter ppq5 times 100; period bounds .0001–.02 s, maximum period factor 1.3 |
| voice_shimmer_local | X31 | percent | Praat shimmer local times 100; period bounds .0001–.02 s, period factor 1.3, amplitude factor 1.6 |
| voice_shimmer_apq3 | X32 | percent | Praat shimmer apq3 times 100; period bounds .0001–.02 s, period factor 1.3, amplitude factor 1.6 |
| voice_shimmer_apq5 | X33 | percent | Praat shimmer apq5 times 100; period bounds .0001–.02 s, period factor 1.3, amplitude factor 1.6 |
| voice_shimmer_apq11 | X34 | percent | Praat shimmer apq11 times 100; period bounds .0001–.02 s, period factor 1.3, amplitude factor 1.6 |
| voice_hnr_mean | X35 | dB | Mean Praat cross-correlation harmonicity |
| voice_spectral_centroid | X36 | Hz | Mean librosa framewise spectral centroid |
| voice_spectral_rolloff | X37 | Hz | Mean librosa framewise spectral rolloff (default 85% energy) |
| voice_zero_crossing_rate | X38 | fraction | Mean librosa framewise zero-crossing fraction |
| voice_mfcc_1 | X39 | implementation-defined coefficient units | Mean librosa MFCC output row 0; identifier 1 is one-based, includes the zeroth coefficient |
| voice_mfcc_2 | X40 | implementation-defined coefficient units | Mean librosa MFCC output row 1; identifier 2 is one-based, includes the zeroth coefficient |
| voice_mfcc_3 | X41 | implementation-defined coefficient units | Mean librosa MFCC output row 2; identifier 3 is one-based, includes the zeroth coefficient |
| voice_mfcc_4 | X42 | implementation-defined coefficient units | Mean librosa MFCC output row 3; identifier 4 is one-based, includes the zeroth coefficient |
| voice_mfcc_5 | X43 | implementation-defined coefficient units | Mean librosa MFCC output row 4; identifier 5 is one-based, includes the zeroth coefficient |
| voice_mfcc_6 | X44 | implementation-defined coefficient units | Mean librosa MFCC output row 5; identifier 6 is one-based, includes the zeroth coefficient |
| voice_mfcc_7 | X45 | implementation-defined coefficient units | Mean librosa MFCC output row 6; identifier 7 is one-based, includes the zeroth coefficient |
| voice_mfcc_8 | X46 | implementation-defined coefficient units | Mean librosa MFCC output row 7; identifier 8 is one-based, includes the zeroth coefficient |
| voice_mfcc_9 | X47 | implementation-defined coefficient units | Mean librosa MFCC output row 8; identifier 9 is one-based, includes the zeroth coefficient |
| voice_mfcc_10 | X48 | implementation-defined coefficient units | Mean librosa MFCC output row 9; identifier 10 is one-based, includes the zeroth coefficient |
| voice_mfcc_11 | X49 | implementation-defined coefficient units | Mean librosa MFCC output row 10; identifier 11 is one-based, includes the zeroth coefficient |
| voice_mfcc_12 | X50 | implementation-defined coefficient units | Mean librosa MFCC output row 11; identifier 12 is one-based, includes the zeroth coefficient |
| voice_mfcc_13 | X51 | implementation-defined coefficient units | Mean librosa MFCC output row 12; identifier 13 is one-based, includes the zeroth coefficient |
| eng_tap_iqr_ratio |  | ratio | tap_flight_iqr / (tap_flight_mean + 0.0001 seconds) |
| eng_tap_variability_miss |  | seconds | tap_flight_std times tap_miss_ratio |
| eng_voice_perturbation_mean |  | percent | (voice_jitter_local + voice_shimmer_local)/2; heuristic composite |
| eng_voice_f0_cv |  | ratio | voice_f0_std / (voice_f0_mean + 0.0001 Hz) |
| eng_voice_spectral_ratio |  | ratio | voice_spectral_rolloff / (voice_spectral_centroid + 0.0001 Hz) |
| eng_tap_voice_interaction |  | seconds times percent | tap_flight_iqr times voice_jitter_local |
