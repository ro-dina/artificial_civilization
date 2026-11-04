"""Read-only v0.7.2 aggregates, equal-seed summaries and matched-state diagnostics."""
import argparse,gzip,json,platform,sys,tempfile
from itertools import combinations
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd

if not __package__:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import Config,SocialConfig,SocialPolicyConfig,SocialLearningConfig
from simulation.social_learning_observer import SCHEMAS
from simulation.social_observer import config_hash
from simulation.vocal_learning_observer import hash_file

RATES={
 'visible_fraction':('visible_decisions','decisions'),'remembered_fraction':('remembered_decisions','decisions'),
 'mean_visible':('visible_contacts','decisions'),'mean_remembered':('remembered_contacts','decisions'),'mean_unknown':('unknown_contacts','decisions'),
 'repeated_contact_fraction':('repeated_contacts','visible_contacts'),'reunion_recognition':('remembered_reunions','long_gap_contacts'),
 'toward_fraction':('toward','target_moves'),'away_fraction':('away','target_moves'),'equal_fraction':('equal','target_moves'),
 'remembered_toward_fraction':('remembered_toward','remembered_target_moves'),'remembered_away_fraction':('remembered_away','remembered_target_moves'),
 'mean_physical_occupancy':('occupancy_sum','decisions'),'q_full_fraction':('at_capacity','decisions'),
 'new_state_fraction':('first_agent_states','decisions'),'q_evictions_per_decision':('evictions','decisions'),
 'exploratory_fraction':('exploratory','decisions'),'reproduction_choice_fraction':('reproduction_choices','decisions'),
 'signals_per_decision':('signals_emitted','decisions'),'identified_fraction':('identified','decisions'),'masked_fraction':('masked','decisions')}

def ratio(a,b):return a/b if b else None


def load(path):
    p=Path(path);m=json.loads(p.read_text())
    try:
        payload=m['config'];cfg=Config(**payload['simulation']);SocialConfig(**payload['social']);SocialLearningConfig(**payload['social_learning'])
        if payload['policy'] is not None:SocialPolicyConfig(**payload['policy'])
        if (m['version']!='0.7.2' or m['format']!='social_learning_aggregates' or m['completed'] is not True
                or m['schema_version']!=1 or m['config_id']!=config_hash(payload) or m['seed']!=cfg.random_seed
                or m['schemas']!={k:list(v) for k,v in SCHEMAS.items()}):raise ValueError('Configuration/schema mismatch')
        for kind,file in m['files'].items():
            if Path(file['file']).name!=file['file']:raise ValueError('Unsafe file name')
            f=p.parent/file['file']
            if hash_file(f)!=file['sha256'] or f.stat().st_size!=file['bytes']:raise ValueError('File hash mismatch: '+str(f))
        if set(m['files'])!=set(SCHEMAS):raise ValueError('Incomplete streams')
    except (KeyError,TypeError) as error:raise ValueError(f'{p}: invalid metadata') from error
    return m


def frame(path,meta,kind):
    f=pd.read_csv(Path(path).parent/meta['files'][kind]['file'])
    if tuple(f)!=SCHEMAS[kind] or len(f)!=meta['files'][kind]['records']:raise ValueError('Row count/header mismatch')
    return f


def summary(frame,keys,metrics):
    if frame.empty:return pd.DataFrame(columns=[*keys,'metric','mean','std','min','max','n'])
    t=frame.melt(id_vars=[*keys,'seed'],value_vars=metrics,var_name='metric',value_name='value')
    if t.duplicated([*keys,'seed','metric']).any():raise ValueError('Repeated seed contribution')
    return t.groupby([*keys,'metric'],dropna=False,sort=True).value.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'})


def utilization(actions,minimum=5):
    """Observed action associations within agent/window/exact legacy key only."""
    result=[]
    for selection in ('all','exploitative'):
        data=actions if selection=='all' else actions[actions.exploratory.eq(0)]
        if data.empty:
            for kind in ('all_social','memory_only'):result.append(dict(selection=selection,kind=kind,min_visits=minimum,cells=0,qualified_cells=0,matched_groups=0,pairs=0,decisions=0,tv_mean=None))
            continue
        keys=['window','agent_id','base_id','visible','remembered_direction']
        cells=data.groupby([*keys,'action'],sort=True).n.sum().unstack('action',fill_value=0).reindex(columns=range(8),fill_value=0)
        totals=cells.sum(axis=1);eligible=cells[totals.ge(minimum)]
        for kind in ('all_social','memory_only'):
            groups=pairs=0;tv=0.;support=0
            levels=[0,1,2] if kind=='all_social' else [0,1,2,3]
            for _,g in eligible.groupby(level=levels,sort=True):
                if len(g)<2:continue
                groups+=1;support+=int(g.to_numpy().sum());probs=g.div(g.sum(axis=1),axis=0).to_numpy()
                for a,b in combinations(probs,2):pairs+=1;tv+=np.abs(a-b).sum()/2
            result.append(dict(selection=selection,kind=kind,min_visits=minimum,cells=len(cells),qualified_cells=len(eligible),matched_groups=groups,pairs=pairs,decisions=support,tv_mean=ratio(tv,pairs)))
    return result


def analyze(metadata_paths,output_dir):
    start=perf_counter();folder=Path(output_dir)
    if folder.exists() and any(folder.iterdir()):raise ValueError('Refusing nonempty analysis destination')
    for old in ('data/social_memory','data/social_memory_v071','analysis/social_memory','analysis/social_memory_v071'):
        if folder.resolve().is_relative_to(Path(old).resolve()) or Path(old).resolve().is_relative_to(folder.resolve()):raise ValueError('Historical paths protected')
    inputs=[(Path(p),load(p)) for p in metadata_paths]
    inputs.sort(key=lambda x:(x[1]['condition'],x[1]['seed']))
    if not inputs or len({(m['condition'],m['seed']) for _,m in inputs})!=len(inputs):raise ValueError('Distinct condition/seed inputs required')
    for p,m in inputs:
        if p.resolve().is_relative_to(folder.resolve()):raise ValueError('Input/output overlap')
    baselines=[{k:v for k,v in m['config']['simulation'].items() if k not in ('random_seed','learning_memory_capacity')} for _,m in inputs]
    if any(c!=baselines[0] for c in baselines):raise ValueError('Ecologies/horizons differ; do not pool')
    runs=[];time_series=[];windows=[];distributions=[];probes=[];uses=[];actions_out=[];strata=[];vocals=[]
    for path,m in inputs:
        tag={'condition':m['condition'],'seed':m['seed']};counts={k:0 for k in SCHEMAS['windows'] if k!='tick'};counts.update(m['totals'])
        s=m['statistics'];row={**tag,**counts,**{k:s[k] for k in ('population','births','deaths','food_consumed','water_consumed','highest_generation','starvation_deaths','dehydration_deaths','energy_deaths','old_age_deaths','successful_reproductions')},
            'final_tick':m['tick'],'initial_population':m['config']['simulation']['initial_population'],'physical_capacity':m['config']['simulation']['learning_memory_capacity'],'distinct_global_states':m['distinct_global_states'],
            **{k:ratio(counts[a],counts[b]) for k,(a,b) in RATES.items()}}
        if s['population']!=m['config']['simulation']['initial_population']+s['births']-s['deaths']:raise ValueError('Population accounting mismatch')
        if counts['updates']!=s['total_learning_updates'] or counts['decisions']!=s['exploratory_actions']+s['exploitative_actions']:raise ValueError('Physical diagnostics disagree with statistics')
        if counts['signals_emitted']!=s['signals_emitted']:raise ValueError('Vocal count mismatch')
        t=frame(path,m,'ticks');time_series.append(t.assign(**tag,initial_population=m['config']['simulation']['initial_population']))
        w=frame(path,m,'windows')
        for k,(a,b) in RATES.items():w[k]=w[a]/w[b].replace(0,np.nan)
        windows.append(w.assign(**tag))
        d=frame(path,m,'distributions');distributions.append(d.assign(**tag))
        visits=d[d.metric.eq('physical_state_visits')]
        if int((visits.value*visits['count']).sum())!=counts['decisions'] or int(visits['count'].sum())!=counts['first_agent_states']:raise ValueError('State visit histogram accounting mismatch')
        for key,reason in (('mean_death_age',None),('mean_survivor_age','run_end_censored')):
            v=d[d.metric.eq('survival_age')];v=v[v.reason.ne('run_end_censored')] if reason is None else v[v.reason.eq(reason)]
            row[key]=ratio(float((v.value*v['count']).sum()),int(v['count'].sum()))
        runs.append(row)
        q=frame(path,m,'probes');q['tv_mean']=q.tv_sum/q.pairs.replace(0,np.nan);q['greedy_change_fraction']=q.changed_greedy_sets/q.pairs.replace(0,np.nan)
        probes.append(q.assign(**tag))
        a=frame(path,m,'actions');uses.extend({**tag,**r} for r in utilization(a))
        # Unmatched action distributions are descriptive exposure associations.
        category=np.where(a.visible.eq(0),'none',np.where(a.remembered_direction.eq(0),'unremembered','remembered'))
        a=a.assign(social_category=category)
        for category,g in a.groupby('social_category',sort=True):
            den=int(g.n.sum());c=g.groupby('action').n.sum()
            for action in range(8):actions_out.append({**tag,'social_category':category,'action':action,'n':int(c.get(action,0)),'decisions':den,'probability':ratio(int(c.get(action,0)),den)})
        b=frame(path,m,'strata');strata.append(b.assign(**tag))
        vocals.append(frame(path,m,'vocal').assign(**tag,vocal_action_count=m['config']['simulation']['signal_vocab_size']+1))
        frame(path,m,'states')
    run=pd.DataFrame(runs);ticks=pd.concat(time_series,ignore_index=True);win=pd.concat(windows,ignore_index=True)
    dist=pd.concat(distributions,ignore_index=True);q=pd.concat(probes,ignore_index=True);use=pd.DataFrame(uses)
    paired=[]
    for seed,g in run.groupby('seed',sort=True):
        for alt,control in (('B','A'),('C','A'),('C','B'),('C_decay','C'),('B_large','B'),('C_large','C')):
            x=g[g.condition.eq(alt)];y=g[g.condition.eq(control)]
            if len(x)!=1 or len(y)!=1:continue
            for metric in ('population','births','deaths','new_state_fraction','q_evictions_per_decision','toward_fraction','remembered_toward_fraction','reunion_recognition','signals_per_decision','identified_fraction'):
                a,b=x[metric].iloc[0],y[metric].iloc[0]
                paired.append({'seed':seed,'contrast':alt+'-'+control,'metric':metric,'difference':np.nan if pd.isna(a) or pd.isna(b) else a-b})
    pair=pd.DataFrame(paired,columns=['seed','contrast','metric','difference'])
    metrics=[k for k in run if k not in ('condition','seed')]
    tables={'run_summary':run,'seed_summary':summary(run,['condition'],metrics),'timeseries':ticks,'windows':win,
            'window_seed_summary':summary(win,['condition','tick'],list(RATES)),
            'distributions':dist,'probes':q,'state_utilization':use,'action_probabilities':pd.DataFrame(actions_out,columns=['condition','seed','social_category','action','n','decisions','probability']),
            'vocal_actions':pd.concat(vocals,ignore_index=True),'movement_strata':pd.concat(strata,ignore_index=True),'paired_by_seed':pair}
    tables['paired_summary']=pair.groupby(['contrast','metric'],sort=True).difference.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'}) if not pair.empty else pd.DataFrame(columns=['contrast','metric','mean','std','min','max','n'])
    folder.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.social-learning-',dir=folder.parent) as tmp:
        staging=Path(tmp)
        for name,f in tables.items():f.to_csv(staging/(name+'.csv'),index=False,float_format='%.12g')
        from experiments.social_learning_plots import plots
        plots(staging)
        import matplotlib
        meta={'version':'0.7.2','completed':True,'inputs':[{'file':p.name,'metadata_sha256':hash_file(p),'condition':m['condition'],'seed':m['seed'],'config':m['config'],'fingerprints':m['fingerprints'],'action_sha256':m['action_sha256']} for p,m in inputs],
              'definitions':{'n':'Independent seeds. Ratios calculated per run before equal-seed mean/sample SD ddof=1; NA for zero exposure.',
                'trajectories':'Same initial ecology/seeds, potentially different actions and ecology. No common-history causal control across new modes.',
                'utilization':'Observed action associations within agent, window and exact original key, cells n>=5; exploration separated. Distinct from read-only trained retained Q-row probes.',
                'survival':'Death age conditions on having died; survivors are right-censored and separate. No lifespan superiority inferred from mean death age.',
                'probes':'Pairs within current living brain, fixed exact original key, >=2/5 current-row visits and >=1 current-row update. memory_only additionally holds visible token fixed. Pair count is support, not inferential sample size.',
                'confounds':'Resource/need/age/count strata retained, observational movement is not social preference; no significance or language claims.'},
              'source_sha256':{p:hash_file(p) for p in ('experiments/analyze_social_learning.py','experiments/social_learning_plots.py')},
              'software':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__},
              'outputs':{p.name:{'sha256':hash_file(p),'bytes':p.stat().st_size} for p in sorted(staging.iterdir())}}
        (staging/'analysis_metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
        folder.mkdir(parents=True,exist_ok=True)
        for p in sorted(staging.iterdir()):p.replace(folder/p.name)
    print(run[['condition','seed','population','births','deaths','new_state_fraction','q_evictions_per_decision','toward_fraction']].to_string(index=False),flush=True)
    return meta,perf_counter()-start


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/social_learning_v072/primary'))
    p.add_argument('--output-dir',type=Path,default=Path('analysis/social_learning_v072/reanalysis'))
    a=p.parse_args()
    try:analyze(sorted(a.data_dir.glob('*.metadata.json')),a.output_dir)
    except (OSError,ValueError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
