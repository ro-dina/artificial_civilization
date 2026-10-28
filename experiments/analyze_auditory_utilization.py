"""v0.6.4: historical observations and a separate predefined small receiver test.

--part retrospective starts no simulations. --part controlled/all explicitly
trains isolated one-agent experiments, never the historical ecological pilots.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
import random

import numpy as np
import pandas as pd
if not __package__:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.auditory_retrospective import SCHEMAS as RETRO_SCHEMAS,Options,run_analysis
from experiments.auditory_controlled import Design,CONDITIONS,definition,train,evaluate_run
from experiments.auditory_policy import evaluate,identified,simulation_fingerprint
from agents.learning import Physiology
from agents.observation import Observation,TileObservation
from simulation.vocal_learning_observer import hash_file

CF_COLUMNS=('experiment','evidence','condition','seed','probe','physical_context','intervention','evaluation_tick',
 'original_current','counterfactual_current','original_memory','counterfactual_memory','original_state','counterfactual_state',
 'original_probabilities','counterfactual_probabilities','delta_probabilities','actions','original_maxima','counterfactual_maxima',
 'original_training_visits','counterfactual_training_visits','original_training_updates','counterfactual_training_updates',
 'encoded_state_changed','both_updated_present','two_distinct_updated_states','tv','jsd_bits','max_set_changed',
 'unique_greedy_changed','original_unique_action','counterfactual_unique_action','original_present','counterfactual_present',
 'row_pattern','coverage','original_uniform','counterfactual_uniform','both_uniform','original_uniform_reason','counterfactual_uniform_reason')
CURVE_COLUMNS=('experiment','evidence','condition','seed','trials','ticks','physical_updates','mean_correct_action_probability',
 'training_first_action_correct','training_water_per_trial','training_reward_per_action','block_trials','q_rows')
RESULT_COLUMNS=('experiment','evidence','condition','seed','training_trials','training_ticks','cue_7_expected_action',
 'cue_12_expected_action','cue_7_unique_action','cue_12_unique_action','cue_7_correct_probability','cue_12_correct_probability',
 'mean_correct_probability','root_identity_tv','root_identity_jsd_bits','both_root_states_present','distinct_root_states',
 'both_root_states_updated','correct_cues','learning_result','training_water_consumed','physical_updates','physical_q_rows','alive')
SCHEMAS={**RETRO_SCHEMAS,'counterfactual_probes':CF_COLUMNS,'training_learning_curves':CURVE_COLUMNS,
         'controlled_experiment_results':RESULT_COLUMNS}


class Outputs:
    def __init__(self,directory,stack):
        self.writers={};self.paths={};self.counts={}
        for name,columns in SCHEMAS.items():
            zipped=name in ('retrospective_matched_contexts','retrospective_action_distances')
            path=directory/(name+('.csv.gz' if zipped else '.csv'));self.paths[name]=path;self.counts[name]=0
            if zipped:
                raw=stack.enter_context(path.open('wb'))
                compressed=stack.enter_context(gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=1))
                stream=stack.enter_context(io.TextIOWrapper(compressed,encoding='utf-8',newline=''))
            else:stream=stack.enter_context(path.open('w',encoding='utf-8',newline=''))
            self.writers[name]=csv.writer(stream);self.writers[name].writerow(columns)
    def add(self,name,row):
        extra=set(row)-set(SCHEMAS[name])
        if extra:raise ValueError(f'Undocumented output fields in {name}: {extra}')
        def cell(value):
            if value is None:return ''
            if isinstance(value,(float,np.floating)):return format(value,'.12g') if np.isfinite(value) else ''
            return value
        self.writers[name].writerow(cell(row.get(c)) for c in SCHEMAS[name]);self.counts[name]+=1


def source_hashes():
    """Bind the evaluator and complete current simulation implementation."""
    root=Path(__file__).resolve().parents[1]
    paths={Path(__file__).resolve().relative_to(root),Path('config.py'),Path('main.py')}
    paths.update(Path('experiments')/n for n in ('auditory_policy.py','auditory_controlled.py',
        'auditory_retrospective.py','auditory_utilization_plots.py','communication_data.py'))
    for directory in ('agents','simulation','systems','world'):
        paths.update(Path(directory).glob('*.py'))
    return {str(p):hash_file(p) for p in sorted(paths)}


def bool_values(series):return series.map({True:1.,False:0.,'True':1.,'False':0.})


def counterfactual_summaries(frame):
    keys=['condition','seed','physical_context','intervention','coverage'];rows=[]
    for key,b in frame.groupby(keys,sort=True):
        unique=bool_values(b.unique_greedy_changed);distinct=bool_values(b.two_distinct_updated_states)
        rows.append({**dict(zip(keys,key)),'evidence':'frozen_policy_intervention','probes':len(b),'mean_tv':b.tv.mean(),
            'mean_jsd_bits':b.jsd_bits.mean(),'policy_changed_fraction':b.tv.gt(0).mean(),
            'max_set_changed_fraction':bool_values(b.max_set_changed).mean(),'unique_comparisons':int(unique.count()),
            'unique_greedy_changed_fraction':unique.mean(),'distinct_updated_comparisons':int(distinct.sum()),
            'nonzero_distinct_updated_comparisons':int((distinct.eq(1)&b.tv.gt(0)).sum()),
            'both_uniform_fraction':bool_values(b.both_uniform).mean()})
    return pd.DataFrame(rows,columns=[*keys,'evidence','probes','mean_tv','mean_jsd_bits','policy_changed_fraction',
        'max_set_changed_fraction','unique_comparisons','unique_greedy_changed_fraction','distinct_updated_comparisons',
        'nonzero_distinct_updated_comparisons','both_uniform_fraction'])


def seed_summary(frame,keys,exclude=()):
    fields=[c for c in frame.select_dtypes(include='number') if c not in {*keys,'seed',*exclude}]
    columns=[*keys,'metric','mean','std','min','max','n']
    if frame.empty:return pd.DataFrame(columns=columns)
    data=frame.melt(id_vars=[*keys,'seed'],value_vars=fields,var_name='metric',value_name='value')
    if data.duplicated([*keys,'seed','metric']).any():raise ValueError('A seed must contribute once per summary cell')
    return data.groupby([*keys,'metric'],sort=True,dropna=False).value.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'})


def benchmark(sim,repetitions=5,calls=5000):
    body=Physiology(0,60,100,True);obs=Observation(sim.tick+1,(TileObservation(0,0,0,0),),identified(7))
    before=simulation_fingerprint(sim);global_before=random.getstate();results={}
    for label,memory in [('live',None),('explicit_empty',())]:
        times=[]
        for _ in range(repetitions):
            start=perf_counter()
            for _ in range(calls):evaluate(sim.agents[0].brain,body,obs,memory)
            times.append((perf_counter()-start)/calls)
        results[label]={'calls_per_repeat':calls,'repeats':repetitions,'seconds_per_call':times,'median_seconds':float(np.median(times))}
    if before!=simulation_fingerprint(sim) or global_before!=random.getstate():raise RuntimeError('Benchmark changed receiver state')
    return {'measurements':results,'state_unchanged':True,'normal_hot_path':'No calls or branches added; only a diagnostic method definition.'}


def analyze(data_dir=Path('data/communication'),output_dir=Path('analysis/auditory_utilization'),part='all',options=None,design=None,
            archive_metadata=Path('analysis/communication/analysis_metadata.json'),verification=Path('data/v061_verification.json'),
            performance_path=None):
    if part not in ('all','retrospective','controlled'):raise ValueError('Unknown analysis component')
    options=options or Options();design=design or Design();output_dir=Path(output_dir).resolve()
    data_dir=Path(data_dir).resolve()
    historic=[Path('analysis')/x for x in ('communication','vocal_learning','vocal_persistence','perception_transition','fixed_perception')]
    if output_dir==data_dir or output_dir.is_relative_to(data_dir) or data_dir.is_relative_to(output_dir):raise ValueError('Output must be separate from original recordings')
    if any(output_dir==p.resolve() or output_dir.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output_dir) for p in historic):
        raise ValueError('Cannot overwrite historical analysis directories')
    previous_outputs=set()
    if performance_path is not None:
        performance_path=Path(performance_path).resolve()
        if performance_path.is_relative_to(data_dir) or performance_path.is_relative_to(output_dir) or any(performance_path.is_relative_to(p.resolve()) for p in historic):
            raise ValueError('Performance receipt must be separate from raw recordings and analysis outputs')
    if output_dir.exists() and any(output_dir.iterdir()):
        side=output_dir/'analysis_metadata.json'
        if not side.exists() or json.loads(side.read_text()).get('version')!='0.6.4':raise ValueError('Output contains historical/unknown files')
        previous_outputs=set(json.loads(side.read_text())['output_files'])|{'analysis_metadata.json'}
        if any(p.name not in previous_outputs for p in output_dir.iterdir()):
            raise ValueError('Output contains files outside the previous v0.6.4 manifest')
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    protocol=definition(design);protocol_text=json.dumps(protocol,sort_keys=True,separators=(',',':'))
    with tempfile.TemporaryDirectory(prefix='.auditory-utilization-',dir=output_dir.parent) as folder:
        temporary=Path(folder);(temporary/'controlled_design.json').write_text(json.dumps(protocol,indent=2)+'\n')
        runs=[];receipt={};proofs=[];configs=[];perf=None
        with ExitStack() as stack:
            output=Outputs(temporary,stack)
            if part in ('all','retrospective'):runs,receipt=run_analysis(data_dir,output,options,archive_metadata,verification)
            if part in ('all','controlled'):
                for seed in sorted(design.seeds):
                    for condition in CONDITIONS:
                        print(f'Controlled {condition} seed={seed}',flush=True)
                        sim,exposure,updates,curves,checkpoints=train(design,seed,condition,
                            checkpoint_sink=lambda row:output.add('training_learning_curves',row))
                        probes,result,proof=evaluate_run(sim,exposure,updates,design,seed,condition)
                        for probe in probes:output.add('counterfactual_probes',probe)
                        output.add('controlled_experiment_results',result)
                        proof['checkpoint_hashes']=checkpoints;proofs.append(proof)
                        configs.append({'condition':condition,'seed':seed,'config':asdict(sim.config)})
                        if performance_path is not None and seed==min(design.seeds) and condition=='predictive_on':perf=benchmark(sim)
        # Large observational context/pair tables are not loaded a second time.
        tables={name:pd.read_csv(path,float_precision='round_trip') for name,path in output.paths.items()
                if name not in ('retrospective_matched_contexts','retrospective_action_distances','retrospective_feature_frequencies')}
        cf=counterfactual_summaries(tables['counterfactual_probes'])
        cf.to_csv(temporary/'counterfactual_summary_by_run.csv',index=False,float_format='%.12g')
        cfs=seed_summary(cf,['condition','physical_context','intervention','coverage'])
        cfs.to_csv(temporary/'counterfactual_summary_by_intervention.csv',index=False,float_format='%.12g')
        coverage=[]
        for (condition,seed),frame in tables['counterfactual_probes'].groupby(['condition','seed'],sort=True):
            for kind in ('both_present','one_present','neither_present'):
                n=int(frame.coverage.eq(kind).sum())
                coverage.append({'condition':condition,'seed':int(seed),'evidence':'frozen_policy_intervention','coverage':kind,
                    'probes':n,'total_probes':len(frame),'fraction':n/len(frame)})
        cv=pd.DataFrame(coverage,columns=['condition','seed','evidence','coverage','probes','total_probes','fraction'])
        cv.to_csv(temporary/'counterfactual_state_coverage.csv',index=False,float_format='%.12g')
        css=seed_summary(tables['controlled_experiment_results'],['condition'])
        css.to_csv(temporary/'controlled_experiment_seed_summary.csv',index=False,float_format='%.12g')
        rss=seed_summary(tables['retrospective_matched_by_run'],['condition','scope','threshold','contrast'])
        rss.to_csv(temporary/'retrospective_seed_summary.csv',index=False,float_format='%.12g')
        tables.update(counterfactual_summary_by_run=cf,counterfactual_summary_by_intervention=cfs,counterfactual_state_coverage=cv,
                      controlled_experiment_seed_summary=css,retrospective_seed_summary=rss)
        from experiments.auditory_utilization_plots import plots
        images=plots(tables,temporary)
        import matplotlib
        files=sorted(temporary.iterdir())
        metadata={'version':'0.6.4','schema_version':1,'completed':True,'part':part,
            'observational_options':asdict(options),'controlled_protocol_sha256':hashlib.sha256(protocol_text.encode()).hexdigest(),
            'controlled_definition':protocol,'controlled_configs':configs,'frozen_evaluation_verification':proofs,
            'historical_registry':receipt,'inputs':[{'run':r['run'],'condition':r['condition'],'seed':r['seed'],
                'sha256':r['sha256'],'metadata_sha256':r['metadata_sha256'],'config':r['metadata']['config'],
                'final_tick':r['metadata']['final_tick']} for r in runs],
            'schemas':{**{k:list(v) for k,v in SCHEMAS.items()},**{k:list(tables[k]) for k in ('counterfactual_summary_by_run',
                'counterfactual_summary_by_intervention','counterfactual_state_coverage','controlled_experiment_seed_summary','retrospective_seed_summary')}},
            'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
            'source_sha256':source_hashes(),
            'output_files':{p.name:{'sha256':hash_file(p),'bytes':p.stat().st_size} for p in files},'streamed_records':output.counts,
            'definitions':{'retrospective':'Observed empirical action frequencies only; no historical physical Q or counterfactual policy reconstruction.',
                'matching':'Within run, individual, window and all seven body/visual integers. No cross-individual/context pooling. Threshold applies separately to each exact audio cell.',
                'weights':'Equal audio-pair means inside each context, then equal contexts inside each run; seeds receive equal weight, SD ddof=1, finite n. Coverage uses unique supported decisions, not summed pair exposures.',
                'scopes':'current/encoded separately; current_given_encoded fixes exact encoded feature; encoded_given_current fixes exact current percept. IDs/directions compared holding the other field and encoded age fixed.',
                'memory':'Current removal preserves the same explicit pre-current past events. Memory-content intervention fixes current SILENCE. Empty auditory feature differs from a listening-disabled seven-integer key.',
                'frozen':'Existing Physical Q table and epsilon; no decisions, updates, allocations, recency, memory or RNG changes. Pending transitions are held as-is, not completed during evaluation.',
                'coverage':'Both present / exactly one / neither, plus directional absence patterns. Present all-zero rows are distinct from missing priors; training visit/update counters additionally identify familiar updated states.',
                'evidence':'Level 1 observational; Level 2 frozen-policy causal input dependence; Level 3 requires predictive learning beyond controls; Level 4 ecological benefit not tested. None implies shared semantics or language.',
                'uncertainty':'Time, unrecorded Q, sub-bin physiology, learning, survival selection and memory remain confounders of retrospective contrasts. Sparse unavailable distances are blank, not zero.'}}
        (temporary/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        output_dir.mkdir(parents=True,exist_ok=True)
        new_outputs={p.name for p in temporary.iterdir()}
        for p in sorted(temporary.iterdir()):p.replace(output_dir/p.name)
        for name in sorted(previous_outputs-new_outputs):(output_dir/name).unlink()
    if performance_path is not None and perf is not None:
        performance_path.parent.mkdir(parents=True,exist_ok=True)
        performance_path.write_text(json.dumps(perf,indent=2)+'\n')
    print(f'Output: {output_dir}')
    if not tables['controlled_experiment_results'].empty:
        print(tables['controlled_experiment_results'][['condition','seed','learning_result','root_identity_tv','mean_correct_probability']].to_string(index=False))
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=('all','retrospective','controlled'),default='all')
    parser.add_argument('--data-dir',type=Path,default=Path('data/communication'))
    parser.add_argument('--output-dir',type=Path,default=Path('analysis/auditory_utilization'))
    parser.add_argument('--window-size',type=int,default=100)
    parser.add_argument('--thresholds',type=int,nargs='+',default=[2,5,10])
    parser.add_argument('--trials',type=int,default=256)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--archive-metadata',type=Path,default=Path('analysis/communication/analysis_metadata.json'))
    parser.add_argument('--verification',type=Path,default=Path('data/v061_verification.json'))
    parser.add_argument('--performance-path',type=Path)
    args=vars(parser.parse_args())
    try:
        options=Options(args.pop('window_size'),tuple(args.pop('thresholds')))
        design=Design(trials=args.pop('trials'),seeds=tuple(args.pop('seeds')))
        analyze(**args,options=options,design=design)
    except (ValueError,OSError,KeyError) as error:parser.error(str(error))

if __name__=='__main__':main()
