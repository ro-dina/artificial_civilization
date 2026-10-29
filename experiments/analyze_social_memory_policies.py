"""v0.7.1 memory-policy replay and equal-seed statistics; no ecological steps."""
from contextlib import ExitStack
from dataclasses import asdict
import argparse,csv,json,platform,resource,sys,tempfile
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
if not __package__:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.social_history import load_metadata,iter_history
from experiments.social_policy_metrics import Treatment,primary,Evaluation,METRICS,ratio
from simulation.vocal_learning_observer import hash_file

FIELDS=('treatment','condition','formation_threshold','eviction_policy','capacity','candidate_capacity','tau')
BASE=('seed',*FIELDS)
RUN=(*BASE,'final_tick','unique_visual_targets',*METRICS)
WINDOW=(*BASE,'window_start','window_end','actual_window_end','partial_window',*METRICS)
AGE=(*BASE,'age_start','age_end',*[k for k in METRICS if not k.startswith('cumulative_') and k not in ('population','living_agent_ticks','full_agent_ticks','living_occupancy_sum','candidate_living_sum','fraction_agents_full','mean_living_occupancy')])
DIST=(*BASE,'metric','reason','value','count')
SCHEMAS={'run_summary':RUN,'window_summary':WINDOW,'age_summary':AGE,'distributions':DIST}


def treatments(conditions=('A','B','C','D'),capacities=(32,),taus=(100.,),candidate_capacity=None):
    if not conditions or len(set(conditions))!=len(conditions) or any(c not in 'ABCD' or len(c)!=1 for c in conditions):raise ValueError('Distinct A/B/C/D conditions required')
    if not capacities or len(set(capacities))!=len(capacities) or not taus or len(set(taus))!=len(taus):raise ValueError('Distinct capacities/taus required')
    result=[]
    for k in sorted(capacities):
        for c in sorted(conditions):
            for tau in (sorted(taus) if c in 'CD' else (100.,)):
                t=Treatment(c,k,5 if c in 'BD' else 1,'contact_decay' if c in 'CD' else 'lru',tau,candidate_capacity)
                t.validate();result.append(t)
    return tuple(result)


def summarize(frame,keys,metrics):
    columns=[*keys,'metric','mean','std','min','max','n']
    if frame.empty:return pd.DataFrame(columns=columns)
    b=frame.melt(id_vars=[*keys,'seed'],value_vars=list(metrics),var_name='metric',value_name='value')
    if b.duplicated([*keys,'seed','metric']).any():raise ValueError('Seed contributes more than once to a summary cell')
    return b.groupby([*keys,'metric'],sort=True,dropna=False).value.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'})


def paired(frame):
    result=[]
    for seed,b in frame.groupby('seed',sort=True):
        for _,alternative in b.iterrows():
            for control_condition in (('A',) if alternative.condition!='A' else ()) + (('C',) if alternative.condition=='D' else ()) + (('B',) if alternative.condition=='D' else ()):
                controls=b[b.condition.eq(control_condition)&b.capacity.eq(alternative.capacity)]
                if control_condition=='C':controls=controls[controls.tau.eq(alternative.tau)]
                if len(controls)!=1:continue
                control=controls.iloc[0]
                for metric in ('recognition_eligible','recognition_repeated','recognition_previously_committed','recognition_qualifying_reunion','evictions','allocations','mean_occupancy'):
                    if alternative.eligible_contacts!=control.eligible_contacts:raise ValueError('Paired eligibility exposure mismatch')
                    result.append({'seed':seed,'alternative':alternative.treatment,'control':control.treatment,
                        'contrast':alternative.condition+'-'+control_condition,'capacity':alternative.capacity,'tau':alternative.tau,
                        'metric':metric,'alternative_value':alternative[metric],'control_value':control[metric],
                        'difference':alternative[metric]-control[metric],'eligible_contacts':alternative.eligible_contacts})
    return pd.DataFrame(result,columns=['seed','alternative','control','contrast','capacity','tau','metric','alternative_value','control_value','difference','eligible_contacts'])


def analyze(histories,output_dir=Path('analysis/social_memory_v071/primary'),horizon=500,policies=None,window_size=100,age_bin=25):
    policies=tuple(policies or primary());folder=Path(output_dir)
    if type(horizon) is not int or horizon<1:raise ValueError('Positive horizon required')
    if folder.exists() and any(folder.iterdir()):raise ValueError('Refusing nonempty v0.7.1 analysis destination')
    forbidden=[Path('analysis/social_memory').resolve(),Path('data/social_memory').resolve()]
    if any(folder.resolve()==p or folder.resolve().is_relative_to(p) or p.is_relative_to(folder.resolve()) for p in forbidden):raise ValueError('Historical destinations protected')
    runs=[]
    for path in histories:
        p=Path(path);meta=load_metadata(p)
        if folder.resolve()==p.resolve() or p.resolve().is_relative_to(folder.resolve()):raise ValueError('Output/input overlap')
        runs.append((meta['seed'],p,meta))
    if not runs or len({r[0] for r in runs})!=len(runs):raise ValueError('Distinct seeded histories required')
    configurations=[{**m['config']['simulation'],'random_seed':0,'max_ticks':horizon} for _,_,m in runs]
    if any(c!=configurations[0] for c in configurations):raise ValueError('Ecological configurations differ; do not pool')
    for t in policies:
        if any(t.reunion!=m['config']['social']['social_reunion_min_absence_ticks'] for _,_,m in runs):raise ValueError('Reunion thresholds differ')
    folder.parent.mkdir(parents=True,exist_ok=True);timing=[];inputs=[]
    with tempfile.TemporaryDirectory(prefix='.memory-policy-',dir=folder.parent) as temp:
        staging=Path(temp)
        with ExitStack() as stack:
            writers={}
            for name,columns in SCHEMAS.items():
                f=stack.enter_context((staging/(name+'.csv')).open('w',encoding='utf-8',newline=''))
                w=csv.writer(f);w.writerow(columns);writers[name]=w
            def write(name,row):
                def cell(v):return '' if v is None else format(v,'.12g') if isinstance(v,float) and np.isfinite(v) else v
                writers[name].writerow(cell(row.get(k)) for k in SCHEMAS[name])
            for seed,path,meta in sorted(runs):
                print(f'Replay seed={seed} horizon={horizon} policies={len(policies)}',flush=True)
                start=perf_counter();evaluation=Evaluation(policies,window_size,age_bin);opportunities=0
                for row in iter_history(path,meta,horizon):
                    evaluation.accept(row);opportunities+=row.kind=='observation'
                    if row.kind=='tick_end' and row.tick%100==0:print(f'  tick={row.tick} population={row.population}',flush=True)
                evaluation.finish()
                for i,t in enumerate(policies):
                    total=evaluation.total[i];base={'seed':seed,**t.fields()}
                    write('run_summary',{**base,'final_tick':evaluation.tick,'unique_visual_targets':len(evaluation.unique_targets),**total.values()})
                    for (index,w),g in sorted(evaluation.windows.items()):
                        if index!=i:continue
                        start_tick=w*window_size;actual=min(start_tick+window_size,evaluation.tick)
                        cumulative=evaluation.window_cumulative[(i,w)]
                        rates={'cumulative_ever_committed_fraction':ratio(cumulative[0],cumulative[1]),'cumulative_candidate_conversion':ratio(cumulative[2],cumulative[3])}
                        write('window_summary',{**base,'window_start':start_tick,'window_end':start_tick+window_size,'actual_window_end':actual,
                            'partial_window':actual<start_tick+window_size,**g.values(),**rates})
                    for (index,age),g in sorted(evaluation.ages.items()):
                        if index==i:write('age_summary',{**base,'age_start':age*age_bin,'age_end':(age+1)*age_bin,**g.values()})
                    for (metric,reason),hist in sorted(total.hist.items()):
                        for value,count in sorted(hist.items()):write('distributions',{**base,'metric':metric,'reason':reason,'value':value,'count':count})
                for (metric,reason,value),count in sorted(evaluation.source_hist.items()):
                    write('distributions',{'seed':seed,'treatment':'SOURCE','condition':'SOURCE','metric':metric,'reason':reason,'value':value,'count':count})
                timing.append({'seed':seed,'seconds':perf_counter()-start,'opportunities':opportunities,'peak_live_observer_target_pairs':evaluation.peak_live_pairs})
                inputs.append({'file':path.name,'sha256':meta['file']['sha256'],'metadata_sha256':hash_file(str(path)+'.metadata.json'),
                               'config':meta['config'],'source_config':meta['source_config'],'provenance':meta['provenance'],
                               'source_final_tick':meta['final_tick'],'evaluated_horizon':evaluation.tick,
                               'private_state_sha256':{t.key:evaluation.private_state_hashes[i] for i,t in enumerate(policies)}})
        frames={name:pd.read_csv(staging/(name+'.csv')) for name in ('run_summary','window_summary','age_summary')}
        summary=summarize(frames['run_summary'],list(FIELDS),METRICS)
        summary.to_csv(staging/'seed_summary.csv',index=False,float_format='%.12g')
        window_keys=[*FIELDS,'window_start','window_end','actual_window_end','partial_window']
        summarize(frames['window_summary'],window_keys,METRICS).to_csv(staging/'window_seed_summary.csv',index=False,float_format='%.12g')
        age_metrics=[k for k in METRICS if k in AGE]
        summarize(frames['age_summary'],[*FIELDS,'age_start','age_end'],age_metrics).to_csv(staging/'age_seed_summary.csv',index=False,float_format='%.12g')
        pairs=paired(frames['run_summary']);pairs.to_csv(staging/'paired_by_seed.csv',index=False,float_format='%.12g')
        # Retain original metric identity rather than colliding with melt's name.
        if pairs.empty:pairs_summary=pd.DataFrame(columns=['alternative','control','contrast','capacity','tau','metric','mean','std','min','max','n'])
        else:pairs_summary=pairs.groupby(['alternative','control','contrast','capacity','tau','metric'],sort=True).difference.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'})
        pairs_summary.to_csv(staging/'paired_seed_summary.csv',index=False,float_format='%.12g')
        from experiments.social_policy_plots import plots
        images=plots(staging)
        import matplotlib
        meta={'version':'0.7.1','schema_version':1,'completed':True,'horizon':horizon,'window_size':window_size,'age_bin':age_bin,
            'inputs':inputs,'treatments':[{'key':t.key,'social':asdict(t.social),'policy':asdict(t.policy),'effective_candidate_capacity':t.policy.resolved_candidate_capacity(t.capacity)} for t in policies],
            'definitions':{'eligibility':'At least five consecutive visual opportunities strictly before current contact, regardless of the tested memory algorithm/candidate pressure.',
                'recognition':'Resident membership before update; current commitment never counts as recognition.',
                'rates':'Count numerator/denominator per seed, then equal-seed mean/sample SD ddof1/finite n; missing denominator stays blank.',
                'source':'Complete ordered opportunity stream shared by all policies. Birth/death exact, empty observations retained, no future or auditory identity.',
                'strength':'Observed sightings in current memory residence including the forming N; no lifetime agent registry.',
                'episodes':'Lengths are actual consecutive observation opportunities. Absence closes an episode; observer death/run end are censored separately.',
                'retention':'Eviction durations separate from censored retained rows. First observed is not first commitment.',
                'candidate_budget':'Separate finite working capacity. Sensitivity can hold it fixed while varying long-term K.',
                'causality':'Controlled memory-mechanics effect on a fixed encounter history only; no behavior/survival/communication effect tested.'},
            'schemas':{**{k:list(v) for k,v in SCHEMAS.items()},'seed_summary':list(summary),'paired_by_seed':list(pairs),'paired_seed_summary':list(pairs_summary)},
            'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
            'source_sha256':{p:hash_file(p) for p in ('agents/social_memory.py','agents/social_memory_policy.py','config.py','experiments/social_history.py','experiments/social_policy_metrics.py','experiments/analyze_social_memory_policies.py','experiments/social_policy_plots.py')},
            'outputs':{p.name:{'sha256':hash_file(p),'bytes':p.stat().st_size} for p in sorted(staging.iterdir())}}
        (staging/'analysis_metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
        folder.mkdir(parents=True,exist_ok=True)
        for p in sorted(staging.iterdir()):p.replace(folder/p.name)
    print(frames['run_summary'][['seed','condition','capacity','tau','recognition_eligible','eligible_contacts','evictions','allocations']].to_string(index=False))
    return meta,{'runs':timing,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--histories',type=Path,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,default=Path('analysis/social_memory_v071/primary'))
    p.add_argument('--horizon',type=int,default=500);p.add_argument('--window-size',type=int,default=100)
    p.add_argument('--conditions',nargs='+',default=['A','B','C','D']);p.add_argument('--capacities',type=int,nargs='+',default=[32])
    p.add_argument('--decay-taus',type=float,nargs='+',default=[100.]);p.add_argument('--candidate-capacity',type=int)
    args=vars(p.parse_args());cs=args.pop('conditions');ks=args.pop('capacities');ts=args.pop('decay_taus');cc=args.pop('candidate_capacity')
    try:analyze(**args,policies=treatments(cs,ks,ts,cc))
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
