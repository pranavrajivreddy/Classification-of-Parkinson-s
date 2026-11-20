"""Prepare release-run splits/configurations without publishing participant records.

This initializes artifacts, not model fits or historical result claims.
"""
import argparse
import json
import shutil
import pandas as pd
from research.common import ROOT, OUT, PRIVATE, sha256, write_json
from research.core import feature_sets, splits

def initialize():
    source = PRIVATE/'cohort.parquet'
    if not source.is_file():
        raise FileNotFoundError('Run python -m research.data using authorized local inputs first; see DATA_SETUP.md')
    d = pd.read_parquet(source)
    if len(d) != 702 or d.pair_id.nunique() != 351:
        raise ValueError('This is the fixed 702-person study protocol; do not relabel a different cohort as its reproduction')
    from research.agent_discovery import prepare
    split = PRIVATE/'agent_extension/split_manifest.json'
    if not split.exists():
        prepare()
    else:
        manifest = json.loads(split.read_text())
        assert manifest['source_cohort_sha256'] == sha256(source)
        for group in ['discovery', 'evaluation']:
            assert manifest[f'{group}_sha256'] == sha256(PRIVATE/f'agent_extension/{group}.parquet')
    for frame, root, arms in [(d, PRIVATE/'runs', ['']),
            (pd.read_parquet(PRIVATE/'agent_extension/evaluation.parquet'), PRIVATE/'agent_extension/runs', ['raw','agent'])]:
        rows=[]
        for fold, (_, te) in enumerate(splits(frame.y.to_numpy(), frame.pair_id.to_numpy(), n=5, seed=42)):
            for i in te:
                rows.append(dict(participant_id=frame.iloc[i].participant_id,pair_id=frame.iloc[i].pair_id,fold=fold))
        assignments = pd.DataFrame(rows)
        for config in feature_sets(frame):
            for arm in arms:
                folder = root/(f'{config}_{arm}' if arm else f'{config}_seed42')
                folder.mkdir(parents=True,exist_ok=True)
                path=folder/'fold_assignments.csv'
                if path.exists():
                    old=pd.read_csv(path)
                    pd.testing.assert_frame_equal(old[assignments.columns].sort_values('participant_id').reset_index(drop=True),
                                                  assignments.sort_values('participant_id').reset_index(drop=True))
                else: assignments.to_csv(path,index=False)
    src=OUT/'agent_extension/frozen_features.json';dest=PRIVATE/'agent_extension/frozen_features.json'
    if dest.exists(): assert sha256(src)==sha256(dest)
    else: shutil.copyfile(src,dest)
    for number in [1,2]:
        src=ROOT/f'configs/interactive_round{number}.json';dest=PRIVATE/f'agent_extension/round{number}_proposal.json'
        if not dest.exists():shutil.copyfile(src,dest)
    print('Prepared pair-preserving splits and frozen-formula replay inputs. No model fitting or API calls performed.')

def association_reference():
    from research.legacy_supplement import paired_associations
    dest=OUT/'legacy_review';dest.mkdir(exist_ok=True)
    paired_associations(pd.read_parquet(PRIVATE/'cohort.parquet')).to_csv(dest/'paired_associations.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','association-reference']);a=p.parse_args()
    initialize() if a.stage=='initialize' else association_reference()
