"""Deterministic raw-data reconstruction; identifiers stay in private outputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from research.common import ROOT, OUT, PRIVATE, CACHE, sha256, write_json, manifest

DATA = ROOT / 'RealDementiaData/mPower'
HANDLES = {'tapping':'tapping_results.json.TappingSamples',
           'walking':'pedometer_walking_outbound.json.items', 'voice':'audio_audio.m4a'}


def binary(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    return 1.0 if s in ('true','1','1.0') else 0.0 if s in ('false','0','0.0') else np.nan


def first_rows(table, field):
    return (table.dropna(subset=[field]).sort_values(['createdOn','recordId'], kind='stable')
            .drop_duplicates('healthCode', keep='first').set_index('healthCode'))


def tapping(samples, duration):
    if not np.isfinite(duration) or not 5 <= duration <= 60:
        raise ValueError('invalid_metadata_duration')
    if not samples: raise ValueError('empty_tapping')
    times = np.array([float(s['TapTimeStamp']) for s in samples])
    if not np.isfinite(times).all() or np.any(np.diff(times)<0) or times.min()<0 or times.max()>duration+1:
        raise ValueError('invalid_tap_timestamps')
    good = np.array([s.get('TappedButtonId') in ('TappedButtonLeft','TappedButtonRight') for s in samples])
    ts=times[good]; buttons=np.array([s.get('TappedButtonId') for s in samples])[good]
    if len(ts)<3: raise ValueError('insufficient_valid_taps')
    dt=np.diff(ts); dt=dt[(dt>=.01)&(dt<=5)]
    if len(dt)<2: raise ValueError('insufficient_intervals')
    first=(ts<duration/2).sum();second=(ts>=duration/2).sum()
    x={'tap_duration_sec':duration,'tap_count_total':len(samples),'tap_count_valid':int(good.sum()),
       'tap_count_miss':int((~good).sum()),'tap_miss_ratio':float((~good).mean()),
       'tap_rate':len(ts)/duration,'tap_half_rate_ratio':second/max(first,1),
       'tap_alternation_rate':float(np.mean(buttons[1:]!=buttons[:-1])),
       'tap_left_button_ratio':float(np.mean(buttons=='TappedButtonLeft')),
       'tap_flight_mean':float(dt.mean()),'tap_flight_median':float(np.median(dt)),
       'tap_flight_std':float(dt.std()),'tap_flight_iqr':float(np.ptp(np.percentile(dt,[25,75]))),
       'tap_flight_cv':float(dt.std()/dt.mean()),'tap_flight_min':float(dt.min()),
       'tap_flight_max':float(dt.max())}
    # Raw screen coordinates excluded: screen geometry/device dependence.
    return x


def walking(items):
    if not items: raise ValueError('empty_pedometer')
    starts=pd.to_datetime([v['startDate'] for v in items], utc=True)
    ends=pd.to_datetime([v['endDate'] for v in items], utc=True)
    order=np.argsort(ends); starts=starts[order];ends=ends[order]
    steps=np.array([float(items[i]['numberOfSteps']) for i in order])
    distance=np.array([float(items[i].get('distance',np.nan)) for i in order])
    if len(set(starts))!=1: raise ValueError('inconsistent_cumulative_start')
    if not np.isfinite(steps).all() or np.any(np.diff(steps)<0): raise ValueError('step_counter_reset')
    duration=(ends[-1]-starts[-1]).total_seconds()
    if not 5<=duration<=120 or steps[-1]<=0: raise ValueError('invalid_walk_duration_or_steps')
    dist=distance[-1]
    if not np.isfinite(dist) or dist<=0 or np.any(np.diff(distance)<-1e-6):
        raise ValueError('invalid_walk_distance')
    return {'walk_duration_sec':duration,'walk_steps_total':float(steps[-1]),
            'walk_distance_total':float(dist),'walk_speed_mean':float(dist/duration),
            'walk_cadence':float(60*steps[-1]/duration),'walk_step_length':float(dist/steps[-1])}


def acoustic(job):
    """Pure extraction; no labels are passed to this worker."""
    handle, path = job
    cache=CACHE/'audio'/f'{handle}.json'
    source_hash=sha256(path)
    extractor_hash=sha256(__file__)
    if cache.exists():
        old=json.loads(cache.read_text())
        if old.get('source_sha256')==source_hash and old.get('extractor_sha256')==extractor_hash:
            return handle,old
    result={'source_sha256':source_hash,'extractor_sha256':extractor_hash,'ok':False}
    try:
        import parselmouth
        from parselmouth.praat import call
        import librosa
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            wav=Path(tmp)/'voice.wav'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(path),'-ar','16000','-ac','1',str(wav)],
                           check=True, capture_output=True, timeout=30)
            y,sr=sf.read(wav); sound=parselmouth.Sound(str(wav))
            pitch=call(sound,'To Pitch',0.,75,500)
            f=pitch.selected_array['frequency']; voiced=float(np.mean(f>0))
            result.update(duration=len(y)/sr,voiced_fraction=voiced,
                          clipping_fraction=float(np.mean(np.abs(y)>=.999)),rms=float(np.sqrt(np.mean(y*y))))
            if len(y)/sr<2 or voiced<.30: raise ValueError('duration_or_voicing_qc')
            point=call(sound,'To PointProcess (periodic, cc)',75,500)
            x={'voice_duration_sec':len(y)/sr,'voice_f0_mean':float(call(pitch,'Get mean',0,0,'Hertz')),
               'voice_f0_std':float(call(pitch,'Get standard deviation',0,0,'Hertz'))}
            for short,praat in [('local','local'),('rap','rap'),('ppq5','ppq5')]:
                x['voice_jitter_'+short]=float(call(point,f'Get jitter ({praat})',0,0,.0001,.02,1.3))*100
            for short in ['local','apq3','apq5','apq11']:
                x['voice_shimmer_'+short]=float(call([sound,point],f'Get shimmer ({short})',0,0,.0001,.02,1.3,1.6))*100
            h=call(sound,'To Harmonicity (cc)',.01,75,.1,1)
            x['voice_hnr_mean']=float(call(h,'Get mean',0,0))
            x['voice_spectral_centroid']=float(librosa.feature.spectral_centroid(y=y,sr=sr).mean())
            x['voice_spectral_rolloff']=float(librosa.feature.spectral_rolloff(y=y,sr=sr).mean())
            x['voice_zero_crossing_rate']=float(librosa.feature.zero_crossing_rate(y).mean())
            for i,v in enumerate(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13).mean(axis=1),1):x[f'voice_mfcc_{i}']=float(v)
            if not np.isfinite(x['voice_f0_mean']): raise ValueError('invalid_f0')
            result.update(ok=True,features={k:(v if np.isfinite(v) else None) for k,v in x.items()})
    except Exception as exc:
        result['error']=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
    write_json(cache,result)
    return handle,result


def match_pairs(frame):
    """Max cardinality then min age difference, deterministic exact-sex matching."""
    pairs=[]
    for sex in [0.,1.]:
        cases=frame[(frame.y==1)&(frame.demo_gender_male==sex)].sort_values('participant_id')
        ctrls=frame[(frame.y==0)&(frame.demo_gender_male==sex)].sort_values('participant_id')
        if not len(cases) or not len(ctrls): continue
        diff=np.abs(cases.demo_age.to_numpy()[:,None]-ctrls.demo_age.to_numpy()[None,:])
        penalty=10*(len(cases)+len(ctrls)+1)
        cost=np.where(diff<=2,diff,penalty*3)
        cost=np.hstack([cost,np.full((len(cases),len(cases)),penalty)])
        rr,cc=linear_sum_assignment(cost)
        for r,c in zip(rr,cc):
            if c<len(ctrls) and diff[r,c]<=2:
                pairs.append((cases.index[r],ctrls.index[c]))
    out=[]
    for i,(a,b) in enumerate(pairs):
        for idx in [a,b]:
            row=frame.loc[idx].copy();row['pair_id']=f'pair_{i:04d}';out.append(row)
    if not out: raise ValueError('No eligible matched pairs')
    return pd.DataFrame(out).sort_values('participant_id').reset_index(drop=True)


def build(workers=4):
    PRIVATE.mkdir(parents=True,exist_ok=True); OUT.mkdir(exist_ok=True)
    base=pd.read_parquet(ROOT/'RealDementiaData/processed/mpower_multimodal.parquet')
    tables={name:pd.read_csv(DATA/'tables'/f'mpower_{name}.csv',low_memory=False) for name in ['demographics',*HANDLES]}
    demo=tables['demographics'].copy();demo['label']=demo['professional-diagnosis'].map(binary)
    conflicts=set(demo.groupby('healthCode').label.nunique().loc[lambda x:x>1].index)
    demos=first_rows(demo,'label')
    selected={m:first_rows(tables[m],h) for m,h in HANDLES.items()}
    # Restrict to the pre-existing local multimodal participant pool: no hidden new test set.
    pool=sorted(set(base.healthCode));rows=[];ledger=[];file_records=[];jobs={}
    for hc in pool:
        pid=hashlib.sha256(('smriti-research-v1:'+hc).encode()).hexdigest()[:20]
        reason=None
        if hc not in demos.index: reason='no_valid_diagnosis_survey'
        elif hc in conflicts: reason='conflicting_diagnosis_answers'
        if reason:ledger.append(dict(participant_id=pid,exclusion=reason));continue
        dr=demos.loc[hc]; age=pd.to_numeric(dr.get('age'),errors='coerce')
        sex={'male':1.,'female':0.}.get(str(dr.get('gender')).lower(),np.nan)
        if not np.isfinite(age) or not 35<=age<=75:reason='age_outside_35_75_or_missing'
        elif not np.isfinite(sex):reason='unknown_gender_for_matching'
        if reason:ledger.append(dict(participant_id=pid,exclusion=reason));continue
        row=dict(participant_id=pid,healthCode=hc,y=int(dr.label),demo_age=float(age),demo_gender_male=sex,
                 demo_smoked=binary(dr.get('smoked')),diagnosis_year=pd.to_numeric(dr.get('diagnosis-year'),errors='coerce'),
                 in_historical_842=False)
        dates=[]
        for mod,handle_col in HANDLES.items():
            if hc not in selected[mod].index:reason=f'{mod}_no_record';break
            rec=selected[mod].loc[hc];handle=str(int(rec[handle_col]));suffix='m4a' if mod=='voice' else 'json'
            path=DATA/'raw'/mod/f'{handle}.{suffix}'
            row[f'{mod}_handle']=handle;row[f'{mod}_createdOn']=rec.createdOn
            row[f'{mod}_device']=str(rec.get('phoneInfo','unknown'))
            row[f'{mod}_medication']=str(rec.get('medTimepoint','unknown'))
            dates.append(float(rec.createdOn))
            row[f'{mod}_sessions_total']=int((tables[mod].healthCode==hc).sum())
            if not path.exists():reason=f'{mod}_file_missing';break
            file_records.append(dict(participant_id=pid,modality=mod,path=str(path.relative_to(ROOT)),sha256=sha256(path)))
            try:
                if mod=='tapping':
                    duration=(float(rec['tapping_results.json.endDate'])-float(rec['tapping_results.json.startDate']))/1000
                    row.update(tapping(json.loads(path.read_text()),duration))
                elif mod=='walking':row.update(walking(json.loads(path.read_text())))
                else:jobs[handle]=path
            except Exception as exc:reason=f'{mod}_{exc}';break
        if reason:ledger.append(dict(participant_id=pid,y=row['y'],exclusion=reason));continue
        row['task_lag_days']=(max(dates)-min(dates))/86400000
        rows.append(row)
    print(f'Raw eligible before audio QC: {len(rows)}; audio jobs: {len(jobs)}',flush=True)
    voice={}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i,(h,res) in enumerate(ex.map(acoustic,[(h,str(p)) for h,p in jobs.items()]),1):
            voice[h]=res
            if i%50==0: print(f'Audio extraction {i}/{len(jobs)}',flush=True)
    usable=[]
    for row in rows:
        res=voice[row['voice_handle']]
        if not res['ok']:
            ledger.append(dict(participant_id=row['participant_id'],y=row['y'],exclusion='voice_'+res.get('error','qc')));continue
        row.update(res['features']);row['voice_voiced_fraction']=res['voiced_fraction'];row['voice_clipping_fraction']=res['clipping_fraction']
        usable.append(row)
    eligible=pd.DataFrame(usable);matched=match_pairs(eligible)
    matched.to_parquet(PRIVATE/'cohort.parquet',index=False);eligible.to_parquet(PRIVATE/'eligible.parquet',index=False)
    pd.DataFrame(ledger).to_csv(PRIVATE/'exclusion_ledger.csv',index=False)
    pd.DataFrame(file_records).to_csv(PRIVATE/'raw_file_manifest.csv',index=False)
    ledger_counts=pd.DataFrame(ledger).exclusion.value_counts().rename_axis('reason').reset_index(name='n')
    ledger_counts.to_csv(OUT/'exclusions.csv',index=False)
    desc=[]
    for name,d in [('eligible',eligible),('matched',matched)]:
        for label,g in d.groupby('y'):
            desc.append(dict(cohort=name,label=int(label),n=len(g),age_mean=g.demo_age.mean(),age_sd=g.demo_age.std(),
                             age_min=g.demo_age.min(),age_max=g.demo_age.max(),male_n=int(g.demo_gender_male.sum()),
                             task_lag_median_days=g.task_lag_days.median(),task_lag_max_days=g.task_lag_days.max()))
    pd.DataFrame(desc).to_csv(OUT/'cohort_description.csv',index=False)
    sources=[DATA/'tables'/f'mpower_{m}.csv' for m in tables]
    sources += [ROOT/'RealDementiaData/processed/mpower_multimodal.parquet']
    write_json(OUT/'data_manifest.json',manifest(dict(pool_n=len(pool),eligible_n=len(eligible),matched_n=len(matched),
        cohort_sha256=sha256(PRIVATE/'cohort.parquet'),sources={str(p.relative_to(ROOT)):sha256(p) for p in sources},
        note='Participant IDs, recording paths, and per-person data are restricted local artifacts.')))
    print(pd.DataFrame(desc).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args();build(a.workers)
