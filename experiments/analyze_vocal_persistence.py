"""v0.6.3 offline same-individual/state/episode persistence. Never runs a simulation."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.vocal_persistence import Options, Outputs, RunAnalysis, SCHEMAS, SUMMARY_COLUMNS
from experiments.vocal_persistence_data import catalog, discover
from experiments.vocal_learning_data import read_chunks
from simulation.vocal_learning_observer import hash_file


SUMMARY_KEYS={
 'persistence_run_summary':['condition'],
 'persistence_by_window':['condition','window_start','window_end','actual_window_end'],
 'persistence_by_visit_index':['condition','visit_kind','visit_index'],
 'persistence_by_age':['condition','age_start'],
 'persistence_by_experience':['condition','update_bin'],
 'paired_episode_development':['condition','first_age_start'],
 'retained_vs_recreated':['condition','window_start','window_end','actual_window_end','encounter_class'],
 'agent_turnover':['condition','window_start','window_end','actual_window_end'],
 'persistence_by_cohort':['condition','window_start','window_end','actual_window_end','cohort'],
 'silence_preference_transitions':['condition','window_start','window_end','actual_window_end','from_category','to_category'],
}


def equal_seed(tables):
    pieces=[]
    for name,keys in SUMMARY_KEYS.items():
        source=tables[name]
        if source.empty:continue
        fields=[c for c in source.select_dtypes(include='number') if c not in {*keys,'seed'}]
        data=source.melt(id_vars=[*keys,'seed'],value_vars=fields,var_name='metric',value_name='value')
        if data.duplicated([*keys,'metric','seed']).any():raise ValueError('Duplicate seed contribution')
        frame=data.groupby([*keys,'metric'],dropna=False,sort=True).value.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'})
        frame.insert(0,'table',name);pieces.append(frame)
    fields=list(dict.fromkeys(['table',*(k for keys in SUMMARY_KEYS.values() for k in keys),'metric','mean','std','min','max','n']))
    return pd.concat(pieces,ignore_index=True).reindex(columns=fields) if pieces else pd.DataFrame(columns=fields)


def verified_communication(runs,communication_dir,vocal_dir,communication_verification,vocal_verification):
    """Bind all four diagnostic files, Config and simultaneous communication hashes.

    Reuse only previously verified output bytes; no permutation/MI calculation.
    """
    cdir,vdir=Path(communication_dir),Path(vocal_dir)
    cm=json.loads((cdir/'analysis_metadata.json').read_text())
    vm=json.loads((vdir/'analysis_metadata.json').read_text())
    cv=json.loads(Path(communication_verification).read_text())['output_hashes']
    vv=json.loads(Path(vocal_verification).read_text())['analysis_output_sha256']
    cfiles=['analysis_metadata.json','sender_information.csv','vocal_preferences_by_agent.csv','vocal_preference_changes.csv']
    vfiles=['analysis_metadata.json','windows.csv']
    receipts={}
    for directory,files,saved in [(cdir,cfiles,cv),(vdir,vfiles,vv)]:
        for name in files:
            path=directory/name
            if hash_file(path)!=saved[name]:raise ValueError(f'{path}: historical analysis hash mismatch')
            receipts[str(path)]=saved[name]
    mapping={}
    for run in runs:
        matches=[r for r in vm['inputs'] if r['metadata_sha256']==run['metadata_sha256'] and
                 r['config']==run['metadata']['config'] and r['files']==run['metadata']['files']]
        if len(matches)!=1:raise ValueError('No exact v0.6.2 diagnostic identity; seed alone is insufficient')
        r=matches[0]
        certificate=run['metadata'].get('communication')
        if not certificate:raise ValueError('No simultaneous communication certificate for MI join')
        path=Path(certificate['file']);side=Path(str(path)+'.metadata.json')
        if hash_file(path)!=certificate['sha256'] or hash_file(side)!=certificate['metadata_sha256']:
            raise ValueError('Simultaneous communication recording/metadata changed')
        archived=json.loads(side.read_text())
        if any(archived[k]!=run['metadata'][k] for k in ('config','final_tick','statistics')):
            raise ValueError('Full communication configuration/trajectory differs')
        old=[x for x in cm['inputs'] if x['sha256']==certificate['sha256'] and x['metadata_sha256']==certificate['metadata_sha256']
             and all(x['simulation_metadata'][k]==run['metadata'][k] for k in ('config','final_tick','statistics'))]
        if len(old)!=1:raise ValueError('No exact archived analysis input for MI join')
        mapping[Path(old[0]['file']).stem]=run['run']
        receipts[str(path)]=certificate['sha256'];receipts[str(side)]=certificate['metadata_sha256']
    if cm['options']['window_size']!=vm['window_size']:
        raise ValueError('Historical communication and diagnostic window sizes differ')
    return mapping,receipts


def join_communication(tables,runs,cdir,vdir,cverification,vverification):
    mapping,receipts=verified_communication(runs,cdir,vdir,cverification,vverification)
    context=['run','condition','seed','window_start','window_end','actual_window_end']
    mi=pd.read_csv(Path(cdir)/'sender_information.csv',float_precision='round_trip')
    mi=mi.loc[mi.run.isin(mapping)&mi.scope.eq('vocal_including_silence')&mi.conditioning.eq('within_agent_eligible')].copy()
    mi['run']=mi.run.map(mapping)
    mi=mi.rename(columns={'mi_bits':'raw_within_agent_mi_bits','corrected_mi_bits':'corrected_within_agent_mi_bits','n':'mi_decisions'})
    prefs=pd.read_csv(Path(cdir)/'vocal_preferences_by_agent.csv',float_precision='round_trip')
    prefs=prefs.loc[prefs.run.isin(mapping)&prefs.eligible_decisions].copy();prefs['run']=prefs.run.map(mapping)
    entropy=prefs.groupby(context,sort=True).agg(mean_individual_vocal_entropy=('vocal_entropy_bits','mean'),
        mean_individual_emitted_entropy=('emitted_entropy_bits','mean'),entropy_agents=('agent_id','size'),
        entropy_decisions=('n_decisions','sum')).reset_index()
    old=pd.read_csv(Path(vdir)/'windows.csv',float_precision='round_trip')
    old=old.rename(columns={'hit_rate':'v062_state_hit_rate','eviction_rate':'v062_eviction_rate'})
    joined=tables['persistence_by_window'].merge(mi[[*context,'feature','mi_decisions','eligible_agents',
        'raw_within_agent_mi_bits','corrected_within_agent_mi_bits']],on=context,validate='one_to_many')
    joined=joined.merge(entropy,on=context,how='left',validate='many_to_one')
    joined=joined.merge(old[[*context,'v062_state_hit_rate','v062_eviction_rate']],on=context,validate='many_to_one')
    shifts=pd.read_csv(Path(cdir)/'vocal_preference_changes.csv',float_precision='round_trip')
    shifts=shifts.loc[shifts.run.isin(mapping)].copy();shifts['run']=shifts.run.map(mapping)
    shifts=shifts.rename(columns={'early_late_jsd_bits':'observed_emitted_first_last_jsd_bits'})
    individual=tables['persistence_by_agent'].merge(shifts,on=['run','condition','seed','agent_id'],how='left',validate='one_to_one')
    if len(joined)!=len(tables['persistence_by_window'])*5:
        raise ValueError('Historical window endpoints do not match; no partial or seed-only joins')
    return joined,individual,receipts


def trajectories(runs,results):
    selected={}
    for run,result in zip(runs,results):
        c=run['condition']
        if c not in selected and result.trajectory_candidates:
            selected[c]={'run':run['run'],'seed':run['seed'],'episodes':result.trajectory_candidates}
    records=[]
    for run in runs:
        selection=selected.get(run['condition'])
        if selection is None or selection['run']!=run['run']:continue
        candidates=selection['episodes']
        keys={(r['agent_id'],r['state_id']) for r in candidates}
        for frame in read_chunks(run,'decisions'):
            # Keep only the deterministic chosen individual/state episodes.
            pairs=list(zip(frame.agent_id,frame.state_id))
            block=frame.loc[[pair in keys for pair in pairs]]
            for d in block.itertuples(index=False):
                for candidate in candidates:
                    if (d.agent_id,d.state_id)==(candidate['agent_id'],candidate['state_id']) and candidate['first_encounter_tick']<=d.tick<=candidate['last_encounter_tick']:
                        from experiments.vocal_persistence_metrics import infer_maxima
                        preferred,_=infer_maxima(d)
                        records.append({'run':run['run'],'condition':run['condition'],'seed':run['seed'],
                            'agent_id':int(d.agent_id),'state_id':int(d.state_id),'episode':candidate['episode'],'tick':int(d.tick),
                            'age':int(d.age),'gap':d.q_gap,'p_silence':d.p_silence,'selected_action':int(d.selected_action),
                            'exploratory':bool(d.exploratory),'preferred_action':preferred})
    columns=['run','condition','seed','agent_id','state_id','episode','tick','age','gap','p_silence','selected_action','exploratory','preferred_action']
    return pd.DataFrame(records,columns=columns),selected


def analyze(data_dir=Path('data/vocal_learning'),output_dir=Path('analysis/vocal_persistence'),options=None,
            communication_dir=Path('analysis/communication'),vocal_dir=Path('analysis/vocal_learning'),
            communication_verification=Path('data/v061_verification.json'),vocal_verification=Path('data/v062_verification.json'),
            no_mi=False):
    options=options or Options();runs=discover(data_dir)
    output_dir=Path(output_dir).resolve();data_dir=Path(data_dir).resolve()
    historical=[Path(communication_dir).resolve(),Path(vocal_dir).resolve()]
    if output_dir==data_dir or output_dir.is_relative_to(data_dir) or data_dir.is_relative_to(output_dir):
        raise ValueError('Output directory must be separate from original diagnostic data')
    if any(output_dir==p or output_dir.is_relative_to(p) or p.is_relative_to(output_dir) for p in historical):
        raise ValueError('Output must not overwrite historical analysis directories')
    if output_dir.exists() and any(output_dir.iterdir()):
        side=output_dir/'analysis_metadata.json'
        if not side.exists() or json.loads(side.read_text()).get('version')!='0.6.3':
            raise ValueError('Output directory contains non-v0.6.3 results; choose a new directory')
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.vocal-persistence-',dir=output_dir.parent) as scratch:
        temp=Path(scratch);results=[];dictionaries=[]
        with ExitStack() as stack:
            output=Outputs(temp,stack)
            for run in runs:
                print(f"Analyzing {run['condition']} seed={run['seed']}",flush=True)
                keys=catalog(run)
                dictionaries.extend({'run':run['run'],'condition':run['condition'],'seed':run['seed'],'state_id':sid,'key_json':key} for sid,key in sorted(keys.items()))
                results.append(RunAnalysis(run,options,output).process())
        tables={name:pd.read_csv(path,float_precision='round_trip') for name,path in output.files.items() if name!='persistence_by_episode'}
        summary=equal_seed(tables);summary.to_csv(temp/'persistence_seed_summary.csv',index=False,float_format='%.12g')
        dictionary=pd.DataFrame(dictionaries,columns=['run','condition','seed','state_id','key_json'])
        dictionary.to_csv(temp/'state_dictionary.csv',index=False)
        joined=pd.DataFrame(columns=['run','condition','seed','window_start','actual_window_end','feature',
            'preferred_action_match_rate','identity_coverage','raw_within_agent_mi_bits','corrected_within_agent_mi_bits'])
        individual=pd.DataFrame(columns=['run','condition','seed','agent_id','observed_emitted_first_last_jsd_bits'])
        receipts={}
        if not no_mi:
            joined,individual,receipts=join_communication(tables,runs,communication_dir,vocal_dir,communication_verification,vocal_verification)
        joined.to_csv(temp/'persistence_mi_comparison.csv',index=False,float_format='%.12g')
        individual.to_csv(temp/'persistence_individual_comparison.csv',index=False,float_format='%.12g')
        trace,selection=trajectories(runs,results)
        trace.to_csv(temp/'individual_episode_trajectories.csv',index=False,float_format='%.12g')
        from experiments.vocal_persistence_plots import plots
        images=plots(tables,summary,joined,trace,selection,options,temp)
        import matplotlib
        files=sorted(temp.iterdir())
        metadata={'version':'0.6.3','options':asdict(options),'inputs':[{'run':r['run'],'condition':r['condition'],'seed':r['seed'],
            'metadata_sha256':r['metadata_sha256'],'config':r['metadata']['config'],'config_id':r['metadata']['config_id'],
            'final_tick':r['metadata']['final_tick'],'files':r['metadata']['files'],
            'diagnostic_version':r['metadata']['version'],'schema_version':r['metadata']['schema_version'],
            'completed':r['metadata']['completed']} for r in runs],
            'pilot_manifest_sha256':hash_file(data_dir/'pilot_manifest.json') if (data_dir/'pilot_manifest.json').exists() else None,
            'software':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__},
            'source_sha256':{p.name:hash_file(p) for p in (Path(__file__),*(Path(__file__).with_name(n) for n in
                ('vocal_persistence.py','vocal_persistence_data.py','vocal_persistence_metrics.py','vocal_persistence_plots.py',
                 'vocal_learning_data.py')),Path(__file__).resolve().parents[1]/'simulation/vocal_learning_observer.py',
                 Path(__file__).resolve().parents[1]/'config.py')},
            'historical_reuse_sha256':receipts,'selection':selection,'schemas':{**{name:list(cols) for name,cols in SCHEMAS.items()},
                'persistence_seed_summary':list(summary),'state_dictionary':list(dictionary),'persistence_mi_comparison':list(joined),
                'persistence_individual_comparison':list(individual),'individual_episode_trajectories':list(trace)},
            'output_files':{p.name:{'sha256':hash_file(p),'bytes':p.stat().st_size} for p in files},
            'streamed_records':dict(sorted(output.counts.items())),
            'peak_active_rows_by_run':{r.run['run']:r.peak_active_rows for r in results},
            'definitions':{'unit':'Run-local exact key, individual, actual allocation epoch. No cross-run state-ID comparison.',
                'ordering':'Recorded actor order within tick; prior nonterminal update before choice, terminal update/death after choice; run end after all actors.',
                'identity':'Unique maximum identified only by recorded selected-Q membership/cardinality or SILENCE membership; full max sets only when these constraints exhaust their members or complement. No Q reconstruction or temporal imputation.',
                'policy':'Exact TV/Jaccard only when complete endpoint max sets are identified (epsilon=1 also proves TV=0). Other identity/set estimates remain undefined.',
                'aggregation':'Each run first, equal-seed mean/sample SD ddof=1/finite n. Every rate retains its own denominator; no decision pooling across seeds.',
                'windows':'Observed windows only; no filling or interpolation. Missing/incomplete encounters are rejected. Partial terminal windows retain actual endpoints.',
                'turnover':'First ever observed learner; actual death tick. Absence in one state/window is not death. Age/update distributions weight decisions within run.',
                'development':'Nonoverlapping first/last 3 encounters of same episode (>=6). Age profiles are cross-sectional; paired development is distinct.',
                'limits':'Stable choices, SILENCE probability or gap do not establish policy identity, meaning, communication, cooperation or language.'}}
        (temp/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        output_dir.mkdir(parents=True,exist_ok=True)
        for path in sorted(temp.iterdir()):path.replace(output_dir/path.name)
    print(tables['persistence_run_summary'][['condition','seed','decisions','pairs','id_pairs','identity_coverage','preferred_action_match_rate','choice_match_rate']].to_string(index=False))
    print(f'Outputs: {output_dir}')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/vocal_learning'))
    parser.add_argument('--output-dir',type=Path,default=Path('analysis/vocal_persistence'))
    parser.add_argument('--window-size',type=int,default=100)
    parser.add_argument('--age-bin',type=int,default=25)
    parser.add_argument('--min-identity-pairs',type=int,default=20)
    parser.add_argument('--min-episode-encounters',type=int,default=6)
    parser.add_argument('--communication-dir',type=Path,default=Path('analysis/communication'))
    parser.add_argument('--vocal-dir',type=Path,default=Path('analysis/vocal_learning'))
    parser.add_argument('--communication-verification',type=Path,default=Path('data/v061_verification.json'))
    parser.add_argument('--vocal-verification',type=Path,default=Path('data/v062_verification.json'))
    parser.add_argument('--no-mi',action='store_true',help='Explicitly omit historical MI joins (e.g. synthetic data)')
    args=vars(parser.parse_args())
    try:
        options=Options(**{k:args.pop(k) for k in ('window_size','age_bin','min_identity_pairs','min_episode_encounters')})
        analyze(**args,options=options)
    except (ValueError,OSError,KeyError) as error:parser.error(str(error))

if __name__=='__main__':main()
