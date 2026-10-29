"""Compact complete ordered visual opportunities, never an ecological replay.

v0.7.0 contacts supply ALL IDs (not just hits). Empty observation rows and exact
lifecycle/tick-end order make alternative memory policies identifiable.
"""
from contextlib import ExitStack
from dataclasses import dataclass,asdict,replace
import csv,gzip,io,json
from pathlib import Path
from time import perf_counter

from config import Config,SocialConfig
from simulation.social_observer import CONTACT_KINDS,config_hash
from simulation.vocal_learning_observer import hash_file

COLUMNS=('sequence','tick','kind','agent_id','age','generation','visible_ids','population','births','deaths','death_cause')


@dataclass(frozen=True,slots=True)
class Opportunity:
    sequence:int
    tick:int
    kind:str
    agent_id:int|None=None
    age:int|None=None
    generation:int|None=None
    visible_ids:tuple=()
    population:int|None=None
    births:int|None=None
    deaths:int|None=None
    death_cause:str|None=None


def open_stream(stack,path,mode='x'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if mode=='x' and (path.exists() or Path(str(path)+'.metadata.json').exists()):raise ValueError('Refusing existing history output')
    if path.suffix=='.gz':
        raw=stack.enter_context(path.open(mode+'b'))
        zipped=stack.enter_context(gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=1))
        return stack.enter_context(io.TextIOWrapper(zipped,encoding='utf-8',newline=''))
    return stack.enter_context(path.open(mode,encoding='utf-8',newline=''))


class HistoryCollector:
    """Fold the legacy stream into one opportunity row plus lifecycle/checkpoints."""
    def __init__(self,stream):
        self.writer=csv.writer(stream);self.writer.writerow(COLUMNS)
        self.records=0;self.received=0;self.pending=None;self.ids=[];self.last_end=None;self.finished=False
    def _write(self,tick,kind,**values):
        if self.finished:raise ValueError('History already finished')
        row={'sequence':self.records,'tick':tick,'kind':kind,**values}
        self.writer.writerow(json.dumps(row.get(k,()),separators=(',',':')) if k=='visible_ids' else row.get(k) for k in COLUMNS)
        self.records+=1
    def _complete(self):
        h=self.pending
        if len(self.ids)!=h.visible_count or len(set(self.ids))!=len(self.ids):raise ValueError('Missing/duplicate visual contacts')
        self._write(h.tick,'observation',agent_id=h.observer_id,age=h.observer_age,generation=h.observer_generation,visible_ids=sorted(self.ids))
        self.pending=None;self.ids=[]
    def __call__(self,row):
        if row.sequence!=self.received:raise ValueError('Unordered original social stream')
        self.received+=1
        if row.event_kind=='observation':
            if self.pending is not None:raise ValueError('Incomplete prior observation')
            self.pending=row;self.ids=[]
            if row.visible_count==0:self._complete()
        elif row.event_kind in CONTACT_KINDS:
            if self.pending is None or (row.tick,row.observer_id)!=(self.pending.tick,self.pending.observer_id):raise ValueError('Contact outside opportunity')
            self.ids.append(row.observed_id)
            if len(self.ids)==self.pending.visible_count:self._complete()
        elif row.event_kind=='eviction':
            if self.pending is None:raise ValueError('Eviction outside opportunity')
        elif row.event_kind in ('founder','birth','death'):
            if self.pending is not None:raise ValueError('Incomplete opportunity before lifecycle')
            self._write(row.tick,row.event_kind,agent_id=row.observer_id,age=row.observer_age,generation=row.observer_generation,death_cause=row.death_cause)
        elif row.event_kind in ('tick_end','run_end'):
            if self.pending is not None:raise ValueError('Incomplete opportunity before checkpoint')
            self._write(row.tick,row.event_kind,population=row.population,births=row.births,deaths=row.deaths)
            if row.event_kind=='tick_end':self.last_end=row
            else:self.finished=True
        else:raise ValueError('Unknown social event')
    def prefix_end(self):
        if self.last_end is None or self.pending is not None:raise ValueError('Prefix must end at a complete tick')
        row=self.last_end
        self._write(row.tick,'run_end',population=row.population,births=row.births,deaths=row.deaths);self.finished=True


def write_history_metadata(path,collector,config,social,requested_horizon,provenance):
    if not collector.finished:raise ValueError('History incomplete')
    effective=asdict(replace(config,max_ticks=requested_horizon))
    payload={'simulation':effective,'social':asdict(social)}
    meta={'version':'0.7.1','format':'visual_opportunities','schema_version':1,'completed':True,
        'seed':config.random_seed,'config':payload,'config_id':config_hash(payload),'source_config':asdict(config),
        'requested_horizon':requested_horizon,'final_tick':collector.last_end.tick,'columns':COLUMNS,'records':collector.records,
        'final_population':collector.last_end.population,'births':collector.last_end.births,'deaths':collector.last_end.deaths,
        'provenance':provenance,'received_legacy_records':collector.received,
        'file':{'sha256':hash_file(path),'bytes':Path(path).stat().st_size},
        'definitions':'Complete simultaneous sorted visible IDs, including misses/candidates; empty opportunities retained. Original actor/lifecycle order; no missing observations guessed. Requested prefix is explicit censoring, not a fabricated continuation.',
        'source_sha256':{p:hash_file(p) for p in ('experiments/social_history.py','agents/social_memory.py','simulation/social_observer.py','simulation/simulation.py','config.py')}}
    with Path(str(path)+'.metadata.json').open('x') as stream:stream.write(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    return meta


def convert_legacy(path,destination,horizon=None):
    from experiments.social_data import discover,records,Validator
    path=Path(path);run=discover(path.parent,path.name)[0]
    requested=run['metadata']['final_tick'] if horizon is None else horizon
    if type(requested) is not int or requested<0:raise ValueError('Nonnegative horizon required')
    end=min(requested,run['metadata']['final_tick'])
    if end<requested and run['metadata']['statistics']['population']!=0:raise ValueError('Cannot invent missing future observations')
    start=perf_counter();validator=Validator(run)
    with ExitStack() as stack:
        collector=HistoryCollector(open_stream(stack,destination))
        for row in records(run):
            validator.accept(row);collector(row)
            if row.event_kind=='tick_end' and row.tick==end and end<run['metadata']['final_tick']:
                if validator.turn is not None:raise ValueError('Incomplete prefix')
                collector.prefix_end();break
            if row.event_kind=='tick_end' and row.tick%100==0:print(f'Validated historical seed={run["seed"]} tick={row.tick}',flush=True)
        else:validator.finish()
    provenance={'kind':'validated_legacy_prefix' if end<run['metadata']['final_tick'] else 'validated_legacy_full',
        'file':path.name,'sha256':run['sha256'],'metadata_sha256':run['metadata_sha256'],
        'original_final_tick':run['metadata']['final_tick'],'evaluated_through_tick':end,
        'legacy_memory_replay_validated':True}
    meta=write_history_metadata(destination,collector,run['config'],run['social'],requested,provenance)
    return meta,{'seconds':perf_counter()-start,'rows':collector.records,'bytes':Path(destination).stat().st_size}


def generate_history(destination,seed,ticks):
    """One unchanged legacy ecology, independent of experimental memory policies."""
    from simulation.simulation import Simulation
    cfg=Config.evolution(brain='learning',random_seed=seed,max_ticks=ticks);social=SocialConfig(True)
    start=perf_counter();step_seconds=0.
    with ExitStack() as stack:
        collector=HistoryCollector(open_stream(stack,destination));sim=Simulation(cfg,social_config=social,social_sink=collector)
        for _ in range(ticks):
            if sim.population==0:break
            before=perf_counter();sim.step();step_seconds+=perf_counter()-before
            if sim.tick%100==0:print(f'Baseline history seed={seed} tick={sim.tick} population={sim.population}',flush=True)
        sim.social_observer.finish(sim.statistics)
    meta=write_history_metadata(destination,collector,cfg,social,ticks,{'kind':'online_unchanged_legacy','new_ecological_run':True})
    return meta,{'seconds':perf_counter()-start,'step_seconds':step_seconds,'rows':collector.records,'bytes':Path(destination).stat().st_size,'statistics':asdict(sim.statistics)}


def load_metadata(path):
    path=Path(path)
    try:
        m=json.loads(Path(str(path)+'.metadata.json').read_text())
        cfg=Config(**m['config']['simulation']);social=SocialConfig(**m['config']['social'])
        if (m['version']!='0.7.1' or m['format']!='visual_opportunities' or m['schema_version']!=1 or m['completed'] is not True
                or tuple(m['columns'])!=COLUMNS or m['config_id']!=config_hash(m['config']) or m['seed']!=cfg.random_seed
                or m['file']['sha256']!=hash_file(path) or m['file']['bytes']!=path.stat().st_size or not social.social_memory_enabled):
            raise ValueError('History integrity/schema mismatch')
        return m
    except (OSError,KeyError,TypeError,ValueError) as error:raise ValueError(f'{path}: invalid history metadata: {error}') from error


def iter_history(path,meta=None,horizon=None):
    """Validate the complete selected prefix; release all live replay state on end."""
    path=Path(path);meta=meta or load_metadata(path)
    end=meta['final_tick'] if horizon is None else min(horizon,meta['final_tick'])
    if horizon is not None and horizon>meta['final_tick'] and meta['final_population']!=0:raise ValueError('History too short')
    alive={};started=set();acted=set();next_id=0;last_end=-1;last_tick=0;births=deaths=0;finished=False;seen_rows=0
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        if tuple(reader.fieldnames or ())!=COLUMNS:raise ValueError('History columns mismatch')
        for index,raw in enumerate(reader):
            if finished:raise ValueError('Rows after end')
            data={k:(None if not v else v if k in ('kind','death_cause','visible_ids') else int(v)) for k,v in raw.items()}
            ids=tuple(json.loads(data.pop('visible_ids') or '[]'))
            row=Opportunity(**data,visible_ids=ids);seen_rows+=1
            if row.sequence!=index or row.tick<last_tick or row.tick>last_end+1:raise ValueError('Missing/unordered history rows')
            last_tick=row.tick
            if row.kind!='observation' and ids:raise ValueError('Visible IDs outside observation')
            if row.kind in ('founder','birth'):
                if (type(row.agent_id) is not int or row.agent_id!=next_id or row.age!=0 or row.generation is None
                        or row.kind=='founder' and (row.tick!=0 or row.generation!=0)
                        or row.kind=='birth' and (row.tick==0 or acted!=started or row.generation<1)):
                    raise ValueError('Invalid history lifecycle')
                next_id+=1;alive[row.agent_id]=(row.tick,row.generation)
                if row.kind=='birth':births+=1
            elif row.kind=='death':
                if row.agent_id not in alive or row.agent_id not in acted or row.death_cause not in ('starvation','dehydration','old_age','energy'):raise ValueError('Invalid death')
                birth,generation=alive[row.agent_id]
                if row.tick!=last_end+1 or (row.age,row.generation)!=(row.tick-birth,generation):raise ValueError('Incorrect death timing/age')
                del alive[row.agent_id];deaths+=1
            elif row.kind=='observation':
                if row.tick<=0 or row.tick!=last_end+1 or row.agent_id not in alive or row.agent_id not in started or row.agent_id in acted:raise ValueError('Missing/duplicate observer')
                b,g=alive[row.agent_id]
                if row.age!=row.tick-b-1 or row.generation!=g or any(type(i) is not int or i==row.agent_id or i not in alive or alive[i][0]>=row.tick for i in ids) or any(a>=b for a,b in zip(ids,ids[1:])):
                    raise ValueError('Invalid visual observation')
                acted.add(row.agent_id)
            elif row.kind=='tick_end':
                if (row.tick!=last_end+1 or row.tick>0 and acted!=started or row.population!=len(alive)
                        or row.births!=births or row.deaths!=deaths or len(alive)!=meta['config']['simulation']['initial_population']+births-deaths):raise ValueError('Incomplete history tick')
                last_end=row.tick;started=set(alive);acted.clear()
            elif row.kind=='run_end':
                if row.tick!=last_end or (row.population,row.births,row.deaths)!=(len(alive),births,deaths):raise ValueError('Incorrect end')
                finished=True
            else:raise ValueError('Unknown history row')
            yield row
            if row.kind=='tick_end' and row.tick==end and end<meta['final_tick']:return
        if (not finished or seen_rows!=meta['records'] or last_end!=meta['final_tick'] or
                (len(alive),births,deaths)!=(meta['final_population'],meta['births'],meta['deaths'])):raise ValueError('Incomplete history file')
