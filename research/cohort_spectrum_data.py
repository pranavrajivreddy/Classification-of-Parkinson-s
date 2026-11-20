"""Broaden age eligibility only; reuse frozen corrected extraction/QC functions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from research.common import ROOT, PRIVATE, OUT, sha256, write_json, manifest
from research.data import DATA, HANDLES, binary, first_rows, tapping, walking, acoustic
from research.core import feature_sets

LOCAL = PRIVATE / 'cohort_comparison'
PUBLIC = OUT / 'cohort_comparison'


def build(workers=4):
    LOCAL.mkdir(parents=True, exist_ok=True)
    sources = [ROOT/'RealDementiaData/processed/mpower_multimodal.parquet',
               PRIVATE/'eligible.parquet', PRIVATE/'cohort.parquet',
               PRIVATE/'runs/full_seed42/fold_assignments.csv']
    sources += [DATA/'tables'/f'mpower_{m}.csv' for m in ['demographics', *HANDLES]]
    before = {str(p): sha256(p) for p in sources}
    write_json(LOCAL/'input_hashes.json', before)
    base = pd.read_parquet(sources[0])
    historical_ids = set()  # Historical membership is not a predictor.
    tables = {m: pd.read_csv(DATA/'tables'/f'mpower_{m}.csv', low_memory=False)
              for m in ['demographics', *HANDLES]}
    demo = tables['demographics'].copy()
    demo['label'] = demo['professional-diagnosis'].map(binary)
    conflicts = set(demo.groupby('healthCode').label.nunique().loc[lambda x: x > 1].index)
    demos = first_rows(demo, 'label')
    selected = {m: first_rows(tables[m], h) for m, h in HANDLES.items()}
    counts = {m: tables[m].healthCode.value_counts() for m in HANDLES}
    rows, ledger, files, jobs = [], [], [], {}
    for hc in sorted(set(base.healthCode)):
        pid = hashlib.sha256(('smriti-research-v1:'+hc).encode()).hexdigest()[:20]
        reason = None
        if hc not in demos.index: reason = 'no_valid_diagnosis_survey'
        elif hc in conflicts: reason = 'conflicting_diagnosis_answers'
        if reason:
            ledger.append(dict(participant_id=pid, exclusion=reason)); continue
        dr = demos.loc[hc]
        age = pd.to_numeric(dr.get('age'), errors='coerce')
        sex = {'male': 1., 'female': 0.}.get(str(dr.get('gender')).lower(), np.nan)
        if not np.isfinite(age) or not 18 <= age <= 100: reason = 'age_outside_18_100_or_missing'
        elif not np.isfinite(sex): reason = 'unknown_gender_for_matching'
        if reason:
            ledger.append(dict(participant_id=pid, exclusion=reason)); continue
        row = dict(participant_id=pid, healthCode=hc, y=int(dr.label), demo_age=float(age),
                   demo_gender_male=sex, demo_smoked=binary(dr.get('smoked')),
                   diagnosis_year=pd.to_numeric(dr.get('diagnosis-year'), errors='coerce'),
                   in_historical_842=hc in historical_ids)
        dates = []
        for mod, col in HANDLES.items():
            if hc not in selected[mod].index: reason = mod+'_no_record'; break
            rec = selected[mod].loc[hc]
            handle = str(int(rec[col])); suffix = 'm4a' if mod == 'voice' else 'json'
            path = DATA/'raw'/mod/f'{handle}.{suffix}'
            row.update({f'{mod}_handle': handle, f'{mod}_createdOn': rec.createdOn,
                        f'{mod}_device': str(rec.get('phoneInfo', 'unknown')),
                        f'{mod}_medication': str(rec.get('medTimepoint', 'unknown')),
                        f'{mod}_sessions_total': int(counts[mod][hc])})
            dates.append(float(rec.createdOn))
            if not path.exists(): reason = mod+'_file_missing'; break
            files.append(dict(participant_id=pid, modality=mod, path=str(path.relative_to(ROOT)), sha256=sha256(path)))
            try:
                if mod == 'tapping':
                    duration = (float(rec['tapping_results.json.endDate'])-float(rec['tapping_results.json.startDate']))/1000
                    row.update(tapping(json.loads(path.read_text()), duration))
                elif mod == 'walking': row.update(walking(json.loads(path.read_text())))
                else: jobs[handle] = str(path)
            except Exception as exc: reason = f'{mod}_{exc}'; break
        if reason:
            ledger.append(dict(participant_id=pid, y=row['y'], exclusion=reason)); continue
        row['task_lag_days'] = (max(dates)-min(dates))/86400000
        rows.append(row)
    print(f'Before voice QC: {len(rows)} participants; {len(jobs)} audio jobs', flush=True)
    voice = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (h, result) in enumerate(pool.map(acoustic, jobs.items()), 1):
            voice[h] = result
            if i % 100 == 0: print(f'Voice QC {i}/{len(jobs)}', flush=True)
    usable = []
    for row in rows:
        res = voice[row['voice_handle']]
        if not res['ok']:
            ledger.append(dict(participant_id=row['participant_id'], y=row['y'], exclusion='voice_'+res.get('error', 'qc')))
            continue
        row.update(res['features'])
        row.update(voice_voiced_fraction=res['voiced_fraction'], voice_clipping_fraction=res['clipping_fraction'])
        usable.append(row)
    broad = pd.DataFrame(usable).sort_values('participant_id').reset_index(drop=True)
    restricted = broad[broad.demo_age.between(35, 75)].copy()
    eligible = pd.read_parquet(PRIVATE/'eligible.parquet').sort_values('participant_id').reset_index(drop=True)
    # Stronger than a feature-only check: all original reconstructed metadata too.
    pd.testing.assert_frame_equal(restricted[eligible.columns].reset_index(drop=True), eligible, check_exact=True)
    matched = pd.read_parquet(PRIVATE/'cohort.parquet').sort_values('participant_id').reset_index(drop=True)
    assert broad.participant_id.is_unique and set(matched.participant_id) <= set(restricted.participant_id)
    anchor = pd.read_csv(PRIVATE/'runs/full_seed42/fold_assignments.csv').set_index('participant_id')
    broad['pair_id'] = broad.participant_id.map(anchor.pair_id).fillna('single_'+broad.participant_id)
    broad['fold'] = broad.participant_id.map(anchor.fold)
    for within in [True, False]:
        ix = broad.index[broad.fold.isna() & (broad.demo_age.between(35, 75) == within)].to_numpy()
        split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (_, test) in enumerate(split.split(np.zeros(len(ix)), broad.loc[ix, 'y'])):
            broad.loc[ix[test], 'fold'] = fold
    broad['fold'] = broad.fold.astype(int)
    assert broad.groupby('pair_id').fold.nunique().max() == 1
    restricted = broad[broad.demo_age.between(35, 75)].reset_index(drop=True)
    matched_new = broad[broad.participant_id.isin(matched.participant_id)].reset_index(drop=True)
    pd.testing.assert_frame_equal(matched_new[matched.columns], matched, check_exact=True)
    description = []
    for name, d in [('broad', broad), ('restricted', restricted), ('matched', matched_new)]:
        d.to_parquet(LOCAL/f'{name}.parquet', index=False)
        for label, g in d.groupby('y'):
            description.append(dict(cohort=name, label=int(label), n=len(g), age_mean=g.demo_age.mean(),
                age_sd=g.demo_age.std(), age_min=g.demo_age.min(), age_max=g.demo_age.max(),
                male_n=int(g.demo_gender_male.sum()), male_fraction=g.demo_gender_male.mean()))
    pd.DataFrame(files).to_csv(LOCAL/'raw_file_manifest.csv', index=False)
    pd.DataFrame(ledger).to_csv(LOCAL/'exclusion_ledger.csv', index=False)
    pd.DataFrame(ledger).exclusion.value_counts().rename_axis('reason').reset_index(name='n').to_csv(PUBLIC/'exclusions.csv', index=False)
    desc = pd.DataFrame(description); desc.to_csv(PUBLIC/'cohort_description.csv', index=False)
    assert len(broad)+len(ledger) == base.healthCode.nunique()
    assert before == {str(p): sha256(p) for p in sources}
    write_json(PUBLIC/'data_verification.json', manifest(dict(status='passed', original_pool_n=len(base),
        broad_n=len(broad), restricted_n=len(restricted), matched_n=len(matched_new),
        restricted_exact_reproduction=True, matched_exact_reproduction=True,
        model_feature_count=len(feature_sets(broad)['full']), sources_unchanged=True,
        protocol_sha256=sha256(PUBLIC/'PROTOCOL.md'),
        cohort_hashes={n: sha256(LOCAL/f'{n}.parquet') for n in ['broad', 'restricted', 'matched']})))
    print(desc.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=4)
    build(parser.parse_args().workers)
