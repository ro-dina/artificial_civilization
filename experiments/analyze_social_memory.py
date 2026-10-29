"""Offline v0.7.0 social measurements. Never constructs or steps a simulation."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
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
from experiments.social_data import discover, records, Validator
from simulation.social_observer import CONTACT_KINDS, DEFINITIONS
from simulation.vocal_learning_observer import hash_file

COUNTS = ('observations','contacts','first_encounters','lifetime_first_pairs','lifetime_recontacts',
          'memory_reappearances','raw_identified_reappearances','continuous_contacts','return_contacts','reunions',
          'memory_hits','memory_misses','evictions','births','deaths','starvation_deaths','dehydration_deaths','old_age_deaths','energy_deaths')
METRICS = (*COUNTS,'active_observing_agents','observed_individuals','first_observed_individuals','died_individuals',
    'population_end','mean_population','mean_memory_occupancy','median_memory_occupancy','memory_utilization',
    'full_memory_fraction','memory_hit_rate','known_visible_fraction','unknown_visible_fraction',
    'evictions_per_observation','mean_visible_count','mean_perception_radius','continuous_given_remembered',
    'mean_recontact_interval','median_recontact_interval','recontact_intervals',
    'mean_reunion_interval','median_reunion_interval','reunion_intervals')
PERSON_METRICS = tuple(k for k in METRICS if k not in {
    'births','deaths','starvation_deaths','dehydration_deaths','old_age_deaths','energy_deaths',
    'active_observing_agents','first_observed_individuals','died_individuals','population_end','mean_population'})
COMMON = ('run','condition','seed')
WINDOW = ('window_start','window_end','actual_window_end','observed_ticks','partial_window')
PERSON = ('individual_id','birth_tick','generation','actual_death_tick','last_observation_tick','end_reason')
SCHEMAS = {
    'social_by_run':(*COMMON,'final_tick','initial_population',*METRICS),
    'social_by_window':(*COMMON,*WINDOW,*METRICS),
    'social_by_individual':(*COMMON,*PERSON,*PERSON_METRICS),
    'social_individual_windows':(*COMMON,*WINDOW,'individual_id',*PERSON_METRICS),
    'social_interval_distribution':(*COMMON,'scope','window_start','window_end','interval_kind','elapsed_ticks','count'),
}


def divide(a,b):
    return a/b if b else None


def histogram_stats(hist):
    n=sum(hist.values())
    if not n:return None,None,0
    ranks=((n-1)//2,n//2);found=[];cumulative=0
    for value,count in sorted(hist.items()):
        found.extend(value for rank in ranks if cumulative<=rank<cumulative+count)
        cumulative+=count
    return sum(v*n for v,n in hist.items())/n,sum(found)/2,n


class Aggregate:
    def __init__(self,capacity):
        self.capacity=capacity;self.c=Counter();self.active=set();self.observed=set();self.first_observed=set();self.died=set()
        self.occupancy=Counter();self.intervals={'recontact':Counter(),'reunion':Counter(),'reappearance':Counter()}
        self.population_end=None;self.population_sum=self.population_ticks=self.radius_sum=0
    def observation(self,row):
        self.c['observations']+=1;self.active.add(row.observer_id)
        self.c['memory_hits']+=row.known_visible;self.c['memory_misses']+=row.unknown_visible
        self.occupancy[row.memory_occupancy]+=1;self.radius_sum+=row.perception_radius
    def contact(self,row,previous_tick,first_global):
        self.c['contacts']+=1;self.observed.add(row.observed_id)
        if first_global:self.first_observed.add(row.observed_id)
        if previous_tick is None:self.c['lifetime_first_pairs']+=1
        else:
            self.c['lifetime_recontacts']+=1
            self.intervals['recontact'][row.tick-previous_tick]+=1
        if row.event_kind in ('first_encounter','memory_miss_after_eviction'):
            self.c['first_encounters']+=1
            if previous_tick is not None:
                self.c['memory_reappearances']+=1
                self.intervals['reappearance'][row.tick-previous_tick]+=1
            if row.event_kind=='memory_miss_after_eviction':self.c['raw_identified_reappearances']+=1
        elif row.event_kind=='continuous_contact':self.c['continuous_contacts']+=1
        elif row.event_kind=='return_contact':self.c['return_contacts']+=1
        elif row.event_kind=='reunion':
            self.c['reunions']+=1;self.intervals['reunion'][row.elapsed_since_last_seen]+=1
    def lifecycle(self,row):
        if row.event_kind=='birth':self.c['births']+=1
        elif row.event_kind=='death':
            self.c['deaths']+=1;self.died.add(row.observer_id);self.c[row.death_cause+'_deaths']+=1
    def population(self,row):
        self.population_end=row.population
        if row.tick>0:self.population_ticks+=1;self.population_sum+=row.population
    def values(self):
        mean_occ,median_occ,_=histogram_stats(self.occupancy)
        mean_gap,median_gap,n_gap=histogram_stats(self.intervals['recontact'])
        mean_reunion,median_reunion,n_reunion=histogram_stats(self.intervals['reunion'])
        n=self.c['observations'];contacts=self.c['contacts'];hits=self.c['memory_hits']
        return {**{k:self.c[k] for k in COUNTS},'active_observing_agents':len(self.active),
            'observed_individuals':len(self.observed),'first_observed_individuals':len(self.first_observed),
            'died_individuals':len(self.died),'population_end':self.population_end,
            'mean_population':divide(self.population_sum,self.population_ticks),'mean_memory_occupancy':mean_occ,
            'median_memory_occupancy':median_occ,'memory_utilization':divide(mean_occ,self.capacity) if mean_occ is not None else None,
            'full_memory_fraction':divide(self.occupancy[self.capacity],n),'memory_hit_rate':divide(hits,contacts),
            'known_visible_fraction':divide(hits,contacts),'unknown_visible_fraction':divide(self.c['memory_misses'],contacts),
            'evictions_per_observation':divide(self.c['evictions'],n),'mean_visible_count':divide(contacts,n),
            'mean_perception_radius':divide(self.radius_sum,n),
            'continuous_given_remembered':divide(self.c['continuous_contacts'],hits),
            'mean_recontact_interval':mean_gap,'median_recontact_interval':median_gap,'recontact_intervals':n_gap,
            'mean_reunion_interval':mean_reunion,'median_reunion_interval':median_reunion,'reunion_intervals':n_reunion}


class Output:
    def __init__(self,folder,stack):
        self.writers={};self.counts=Counter()
        for name,columns in SCHEMAS.items():
            stream=stack.enter_context((folder/(name+'.csv')).open('w',encoding='utf-8',newline=''))
            self.writers[name]=csv.writer(stream);self.writers[name].writerow(columns)
    def add(self,name,row):
        if set(row)-set(SCHEMAS[name]):raise ValueError(f'Unknown output fields: {name}')
        def cell(value):
            if value is None:return ''
            if isinstance(value,float):return format(value,'.12g') if np.isfinite(value) else ''
            return value
        self.writers[name].writerow(cell(row.get(c)) for c in SCHEMAS[name]);self.counts[name]+=1


class RunAnalysis:
    def __init__(self,run,window_size,output):
        self.run=run;self.window_size=window_size;self.output=output
        self.base={k:run[k] for k in COMMON};self.capacity=run['social'].social_memory_capacity
        self.total=Aggregate(self.capacity);self.window=None;self.window_index=None
        self.people={};self.individual_windows={};self.first_targets=set();self.last_tick=0
    def context(self):
        start=self.window_index*self.window_size
        end=min(start+self.window_size,self.run['metadata']['final_tick'])
        return {**self.base,'window_start':start,'window_end':start+self.window_size,'actual_window_end':end,
                'observed_ticks':end-start,'partial_window':end<start+self.window_size}
    def intervals(self,aggregate,scope,start,end):
        for kind,hist in aggregate.intervals.items():
            for gap,n in sorted(hist.items()):
                self.output.add('social_interval_distribution',{**self.base,'scope':scope,'window_start':start,
                    'window_end':end,'interval_kind':kind,'elapsed_ticks':gap,'count':n})
    def flush_window(self):
        if self.window is None:return
        context=self.context()
        self.output.add('social_by_window',{**context,**self.window.values()})
        self.intervals(self.window,'window',context['window_start'],context['actual_window_end'])
        for identity,aggregate in sorted(self.individual_windows.items()):
            self.output.add('social_individual_windows',{**context,'individual_id':identity,**{k:v for k,v in aggregate.values().items() if k in PERSON_METRICS}})
        self.window=None;self.individual_windows={}
    def finish_person(self,identity,death_tick=None):
        person=self.people.pop(identity)
        self.output.add('social_by_individual',{**self.base,'individual_id':identity,'birth_tick':person['birth'],
            'generation':person['generation'],'actual_death_tick':death_tick,'last_observation_tick':person['last_observation'],
            'end_reason':'actual_death' if death_tick is not None else 'run_end',**{k:v for k,v in person['aggregate'].values().items() if k in PERSON_METRICS}})
    def accept(self,row):
        if row.tick>0:
            index=(row.tick-1)//self.window_size
            if self.window_index!=index:
                self.flush_window();self.window_index=index;self.window=Aggregate(self.capacity)
        if row.event_kind in ('founder','birth'):
            self.people[row.observer_id]={'birth':row.tick,'generation':row.observer_generation,'last_observation':None,
                'last_contacts':{},'aggregate':Aggregate(self.capacity)}
        aggregates=[self.total]
        if row.tick>0:aggregates.append(self.window)
        if row.event_kind=='observation':
            person=self.people[row.observer_id];person['last_observation']=row.tick
            individual=self.individual_windows.setdefault(row.observer_id,Aggregate(self.capacity))
            for a in (*aggregates,person['aggregate'],individual):a.observation(row)
        elif row.event_kind in CONTACT_KINDS:
            person=self.people[row.observer_id];previous=person['last_contacts'].get(row.observed_id)
            if row.event_kind=='memory_miss_after_eviction' and previous is None:
                raise ValueError('Claimed reappearance without a previous recorded encounter')
            first_global=row.observed_id not in self.first_targets
            self.first_targets.add(row.observed_id)
            for a in (*aggregates,person['aggregate'],self.individual_windows[row.observer_id]):a.contact(row,previous,first_global)
            person['last_contacts'][row.observed_id]=row.tick
        elif row.event_kind=='eviction':
            for a in (*aggregates,self.people[row.observer_id]['aggregate'],self.individual_windows[row.observer_id]):a.c['evictions']+=1
        elif row.event_kind in ('birth','death'):
            for a in aggregates:a.lifecycle(row)
            if row.event_kind=='death':self.finish_person(row.observer_id,row.tick)
        elif row.event_kind=='tick_end':
            for a in aggregates:a.population(row)
        self.last_tick=row.tick
    def finish(self):
        self.flush_window()
        for identity in sorted(self.people):self.finish_person(identity)
        self.output.add('social_by_run',{**self.base,'final_tick':self.last_tick,
            'initial_population':self.run['config'].initial_population,**self.total.values()})
        self.intervals(self.total,'run',None,None)


def seed_summary(frame,windowed=False):
    keys=['condition',*WINDOW] if windowed else ['condition']
    data=frame.melt(id_vars=[*keys,'seed'],value_vars=list(METRICS),var_name='metric',value_name='value')
    if data.duplicated([*keys,'seed','metric']).any():raise ValueError('Repeated seed in aggregate cell')
    if frame.empty:return pd.DataFrame(columns=[*keys,'metric','mean','std','min','max','n'])
    return data.groupby([*keys,'metric'],sort=True,dropna=False).value.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'})


def analyze(data_dir=Path('data/social_memory'),output_dir=Path('analysis/social_memory'),window_size=100,pattern='*.csv*'):
    if type(window_size) is not int or window_size<1:raise ValueError('window_size must be a positive integer')
    data_dir,output_dir=Path(data_dir).resolve(),Path(output_dir).resolve()
    if output_dir.is_relative_to(data_dir) or data_dir.is_relative_to(output_dir):raise ValueError('Raw inputs and analysis outputs must be separate')
    if output_dir.exists() and any(output_dir.iterdir()):raise ValueError('Refusing nonempty analysis destination')
    runs=discover(data_dir,pattern)
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.social-analysis-',dir=output_dir.parent) as temp:
        folder=Path(temp)
        with ExitStack() as stack:
            output=Output(folder,stack)
            for run in runs:
                print(f"Social {run['condition']} seed={run['seed']}: {run['path'].name}",flush=True)
                validator=Validator(run);summary=RunAnalysis(run,window_size,output)
                for row in records(run):validator.accept(row);summary.accept(row)
                validator.finish();summary.finish()
        run_frame=pd.read_csv(folder/'social_by_run.csv')
        windows=pd.read_csv(folder/'social_by_window.csv')
        seed_summary(run_frame).to_csv(folder/'social_seed_summary.csv',index=False,float_format='%.12g')
        seed_summary(windows,True).to_csv(folder/'social_window_seed_summary.csv',index=False,float_format='%.12g')
        from experiments.social_plots import plots
        selections=plots(folder)
        import matplotlib
        inputs=[{'run':r['run'],'condition':r['condition'],'seed':r['seed'],'sha256':r['sha256'],
                 'metadata_sha256':r['metadata_sha256'],'config':r['metadata']['config'],'final_tick':r['metadata']['final_tick']} for r in runs]
        meta={'version':'0.7.0','schema_version':1,'completed':True,'window_size':window_size,'inputs':inputs,
            'definitions':{**DEFINITIONS,'windows':'(start,end], explicit actual end/partial flag. No interpolation/post-extinction filling. Tick0 lifecycle/initial population only; a zero-tick run has no windows.',
                'weighting':'Each run summarized first; equal-seed means/sample SD ddof1/finite n. No pooled decision-level inference. Occupancy is post-observation, opportunity weighted within a run.',
                'firsts':'first_encounters counts bounded-memory misses; lifetime_first_pairs and exact memory_reappearances reconstructed offline for each observer from complete contacts, independent of bounded recorder cache.',
                'denominators':'known/unknown/hit fractions per directed visible contact; evictions per observation opportunity; full fraction/occupancy per opportunity. Zero denominator is blank.',
                'demography':'Actual death events only. Active observers, visible targets, first-ever globally visible targets, and explicitly dead IDs are distinct. Population comes only from completed tick rows.',
                'persistence':'continuous_given_remembered=continuous contacts / memory hits. Recontact intervals include forgotten pairs; reunion intervals use remembered qualifying returns only.',
                'offline_memory':'Bounded mechanical replay per living observer; separate lifetime contact ID/tick maps released at explicit observer death/end. Global observed-ID set and scalar window/individual summaries are offline diagnostics, never agent knowledge.'},
            'schemas':{**{k:list(v) for k,v in SCHEMAS.items()},'social_seed_summary':list(seed_summary(run_frame)),
                       'social_window_seed_summary':list(seed_summary(windows,True))},
            'records':dict(output.counts),'example_selection':selections,
            'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
            'source_sha256':{name:hash_file(name) for name in ('experiments/analyze_social_memory.py','experiments/social_data.py',
                'experiments/social_plots.py','agents/social_memory.py','simulation/social_observer.py','config.py')},
            'outputs':{p.name:{'sha256':hash_file(p),'bytes':p.stat().st_size} for p in sorted(folder.iterdir())}}
        (folder/'analysis_metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
        output_dir.mkdir(parents=True,exist_ok=True)
        for p in sorted(folder.iterdir()):p.replace(output_dir/p.name)
    print(run_frame[['run','seed','observations','contacts','first_encounters','reunions','mean_memory_occupancy','evictions','population_end','deaths']].to_string(index=False))
    print(f'Output: {output_dir}')
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/social_memory'))
    parser.add_argument('--output-dir',type=Path,default=Path('analysis/social_memory'))
    parser.add_argument('--window-size',type=int,default=100)
    parser.add_argument('--pattern',default='*.csv*')
    try:analyze(**vars(parser.parse_args()))
    except (ValueError,OSError,KeyError) as error:parser.error(str(error))

if __name__=='__main__':main()
