"""Optional v0.7.2 streaming aggregates. No rewards, table writes or RNG.

Only the external observer keeps lifetime state/contact histories for living
observers. Physical Q hooks measure actual allocation (including pending-row
recreation); no per-update payload or whole-Q dump is written.
"""
from collections import Counter,defaultdict
from contextlib import ExitStack
from dataclasses import asdict,fields
from itertools import combinations
import csv,gzip,io,json,hashlib
from pathlib import Path

from agents.learning import LearningBrain
from agents.social_learning import nearest,extract
from simulation.vocal_learning_observer import hash_file
from simulation.statistics import Statistics

SCHEMAS={
 'ticks':(*(f.name for f in fields(Statistics)),'physical_rows','physical_agents','distinct_global_states'),
 'windows':('tick','decisions','visible_decisions','remembered_decisions','visible_contacts','remembered_contacts','unknown_contacts','repeated_contacts','long_gap_contacts','remembered_reunions','moves','move_attempts','blocked_moves','target_moves','toward','away','equal','remembered_target_moves','remembered_toward','remembered_away','remembered_equal','occupancy_sum','at_capacity','first_agent_states','first_global_states','recreated_decisions','allocations','evictions','updates','exploratory','exploitative','reproduction_choices','identified','masked','silence','signals_emitted','physical_action_0','physical_action_1','physical_action_2','physical_action_3','physical_action_4','physical_action_5','physical_action_6','physical_action_7'),
 'actions':('window','agent_id','base_id','visible','remembered_direction','action','exploratory','n'),
 'states':('base_id','key_json'),
 'strata':('window','food_here','water_here','food_visible','water_visible','hunger_bin','thirst_bin','age_bin','visible_count_bin','decisions','target_moves','toward','away','equal','remembered_target_moves','remembered_toward','remembered_away','remembered_equal'),
 'distributions':('metric','reason','value','count'),
 'probes':('tick','kind','min_visits','retained_rows','qualified_rows','matched_groups','supported_agents','pairs','nonflat_pairs','changed_greedy_sets','tv_sum','q_linf_sum'),
 'vocal':('tick','vocal_action','n')}


def paths(path):
    p=Path(path);stem=p.name[:-7] if p.name.endswith('.csv.gz') else p.stem
    suffix='.csv.gz' if p.suffix=='.gz' else '.csv'
    return {'ticks':p,**{k:p.with_name(stem+'.'+k+suffix) for k in SCHEMAS if k!='ticks'}}


class SocialLearningWriter:
    def __init__(self,streams):
        self.writers={k:csv.DictWriter(s,fieldnames=SCHEMAS[k],extrasaction='ignore') for k,s in streams.items()}
        self.counts=Counter()
        for writer in self.writers.values():writer.writeheader()
    @classmethod
    def open(cls,stack,path):
        outputs=paths(path)
        if any(p.exists() for p in (*outputs.values(),Path(str(path)+'.metadata.json'))):
            raise ValueError('Refusing existing v0.7.2 output/sidecar')
        streams={}
        for kind,p in outputs.items():
            p.parent.mkdir(parents=True,exist_ok=True)
            raw=stack.enter_context(p.open('xb'))
            source=stack.enter_context(gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=1)) if p.suffix=='.gz' else raw
            streams[kind]=stack.enter_context(io.TextIOWrapper(source,encoding='utf8',newline=''))
        return cls(streams)
    def __call__(self,kind,row):
        self.writers[kind].writerow(row);self.counts[kind]+=1


def policy(values,epsilon):
    best=max(values);ties=sum(v==best for v in values)
    return tuple(epsilon/len(values)+(1-epsilon)*(v==best)/ties for v in values)


def probe(agents,mode,tick,minimum=5,kind='all_social'):
    """Observed retained same-agent/base rows only; no synthetic/unvisited priors."""
    result=dict.fromkeys(SCHEMAS['probes'],0);result.update(tick=tick,kind=kind,min_visits=minimum)
    suffix=0 if mode=='none' else 1 if mode=='perception' else 2
    for human,history in agents:
        brain=human.brain;groups=defaultdict(list);supported=False
        for state,values in brain._values.items():
            result['retained_rows']+=1
            visits,row_visits,row_updates=history.get(state,(0,0,0))
            if row_visits<minimum or row_updates<1:continue
            result['qualified_rows']+=1
            if not suffix:continue
            if kind=='memory_only' and mode!='memory':continue
            base=state[:-suffix] if kind=='all_social' else state[:-1]
            groups[base].append(tuple(values))
        for rows in groups.values():
            if len(rows)<2:continue
            result['matched_groups']+=1;supported=True
            for a,b in combinations(rows,2):
                pa,pb=policy(a,brain.epsilon),policy(b,brain.epsilon)
                result['pairs']+=1
                result['nonflat_pairs']+=max(a)!=min(a) or max(b)!=min(b)
                result['changed_greedy_sets']+=tuple(i for i,v in enumerate(a) if v==max(a))!=tuple(i for i,v in enumerate(b) if v==max(b))
                result['tv_sum']+=sum(abs(x-y) for x,y in zip(pa,pb))/2
                result['q_linf_sum']+=max(abs(x-y) for x,y in zip(a,b))
        result['supported_agents']+=supported
    return result


class SocialLearningObserver:
    def __init__(self,sink,window=100):
        if type(window) is not int or window<1:raise ValueError('Positive observation window required')
        self.sink=sink;self.window=window;self.agents={};self.partners={};self.previous={}
        self.last_visible={}
        self.total=Counter();self.current=Counter();self.actions=Counter();self.strata=defaultdict(Counter)
        self.vocal=Counter();self.keys={};self.global_states=set();self.dist=Counter();self.finished=False;self.last_flush=0
        self.peak_live_state_histories=0;self.peak_live_contact_histories=0
        self.action_digest=hashlib.sha256()
        self.visual_targets=set()

    def add(self,**values):
        self.current.update(values);self.total.update(values)

    def begin(self,human,observation,tick):
        if self.finished:raise RuntimeError('Diagnostic already finished')
        if not isinstance(human.brain,LearningBrain):raise ValueError('Physical learning diagnostics require LearningBrain')
        self.tick=tick;self.agent_id=human.id;self.human_age=human.age
        self.history=self.agents.setdefault(human.id,{})
        self.brain=human.brain;self.brain._physical_observer=self
        ids=getattr(observation,'individuals',())
        memory=getattr(human,'social_memory',None)
        known=tuple(i for i in ids if memory is not None and memory.get(i.individual_id) is not None)
        self.features=extract(ids,memory,include_memory=True)
        self.target=nearest(ids);self.known_target=nearest(known);self.start=(human.x,human.y)
        visible={i.individual_id for i in ids};known_ids={i.individual_id for i in known}
        self.visual_targets.update(visible)
        partners=self.partners.setdefault(human.id,{})
        previous=self.previous.get(human.id)
        for identity in self.last_visible.get(human.id,set())-visible:
            self.dist['contact_length','observed_absence',partners[identity][1]]+=1
        repeated=long_gap=reunions=0
        for identity in sorted(visible):
            old=partners.get(identity)
            if old is not None:
                repeated+=1
                if old[0]!=previous and tick-old[0]>=20:
                    long_gap+=1;reunions+=identity in known_ids
            partners[identity]=(tick,old[1]+1 if old is not None and old[0]==previous else 1)
        self.previous[human.id]=tick
        self.last_visible[human.id]=visible
        self.add(decisions=1,visible_decisions=bool(ids),remembered_decisions=bool(known),
                 visible_contacts=len(ids),remembered_contacts=len(known),unknown_contacts=len(ids)-len(known),
                 repeated_contacts=repeated,long_gap_contacts=long_gap,remembered_reunions=reunions)
        here=next((t for t in observation.tiles if t.dx==0 and t.dy==0),None)
        self.context=(int(here is not None and here.food>0),int(here is not None and here.water>0),
                      int(any(t.food>0 for t in observation.tiles)),int(any(t.water>0 for t in observation.tiles)),
                      self.brain._bin(human.hunger,self.brain.max_hunger),self.brain._bin(human.thirst,self.brain.max_thirst),
                      human.age//25,self.features.count_bin)
        for name,value,limit in (('hunger',human.hunger,self.brain.max_hunger),('thirst',human.thirst,self.brain.max_thirst),('energy',human.energy,self.brain.max_energy)):
            self.dist[name,'pre_action_decile',max(0,min(9,int(10*value/limit)))]+=1
        self.add(**{observation.auditory.kind.value.lower():1})

    def record(self,state):
        return self.history.setdefault(state,[0,0,0])
    def start_state(self,state,present):
        self.novel=state not in self.history;self.recreated=not self.novel and not present
        self.new_global=state not in self.global_states
        self.global_states.add(state)
    def allocated(self,state):
        h=self.record(state);h[1]=h[2]=0;self.add(allocations=1)
    def evicted(self,state):self.add(evictions=1)
    def updated(self,state,index,old,new,terminal):
        self.record(state)[2]+=1;self.add(updates=1)
    def selected(self,brain,state,values,index):
        h=self.record(state);h[0]+=1;h[1]+=1
        suffix=0 if not hasattr(brain,'social_mode') else 1 if brain.social_mode=='perception' else 2
        base=state[:-suffix] if suffix else state
        if base not in self.keys:
            self.keys[base]=len(self.keys);self.sink('states',{'base_id':self.keys[base],'key_json':json.dumps(base,separators=(',',':'))})
        key=((self.tick-1)//self.window,self.agent_id,self.keys[base],self.features.visible,self.features.remembered_direction,index,int(brain.last_exploratory))
        self.actions[key]+=1
        self.add(first_agent_states=self.novel,first_global_states=self.new_global,recreated_decisions=self.recreated,
                 occupancy_sum=brain.memory_size,at_capacity=brain.memory_size==brain.capacity,
                 exploratory=int(brain.last_exploratory),exploitative=int(not brain.last_exploratory),**{f'physical_action_{index}':1})

    @staticmethod
    def movement(target,start,human):
        if target is None:return None
        dx,dy=human.x-start[0],human.y-start[1]
        before=max(abs(target.dx),abs(target.dy));after=max(abs(target.dx-dx),abs(target.dy-dy))
        return 'toward' if after<before else 'away' if after>before else 'equal'

    def outcome(self,human,decision,cause):
        self.action_digest.update(json.dumps((self.tick,human.id,*self.start,decision.action.value,decision.signal_id,
            human.x,human.y,human.age,human.hunger,human.thirst,human.energy,human.alive),separators=(',',':')).encode())
        moved=(human.x,human.y)!=self.start
        attempted=decision.action.value.startswith('move_')
        changes=Counter(decisions=1)
        for target,prefix in ((self.target,''),(self.known_target,'remembered_')):
            if moved and target is not None:
                changes[prefix+'target_moves']+=1;changes[prefix+self.movement(target,self.start,human)]+=1
        self.strata[self.context].update(changes)
        self.add(**{k:v for k,v in changes.items() if k!='decisions'},moves=moved,move_attempts=attempted,
                 blocked_moves=attempted and not moved,reproduction_choices=decision.action.value=='reproduce',signals_emitted=decision.signal_id is not None)
        self.vocal[0 if decision.signal_id is None else decision.signal_id+1]+=1
        del human.brain._physical_observer
        self.brain=None
        if cause is not None:
            self.dist['survival_age',cause,human.age]+=1;self.close_agent(human.id,'observer_death')

    def close_agent(self,identity,reason):
        prior=self.previous.pop(identity,None)
        partners=self.partners.pop(identity,{})
        self.dist['distinct_partners',reason,len(partners)]+=1
        for target in self.last_visible.pop(identity,set()):
            self.dist['contact_length',reason,partners[target][1]]+=1
        for visits,_,_ in self.agents.pop(identity,{}).values():self.dist['physical_state_visits',reason,visits]+=1

    def flush(self,tick):
        self.sink('windows',{'tick':tick,**{k:self.current[k] for k in SCHEMAS['windows'] if k!='tick'}})
        self.current.clear()
        for key,n in sorted(self.actions.items()):self.sink('actions',{**dict(zip(SCHEMAS['actions'][:-1],key)),'n':n})
        self.actions.clear()
        for context,c in sorted(self.strata.items()):
            self.sink('strata',{'window':(tick-1)//self.window,**dict(zip(SCHEMAS['strata'][1:9],context)),**{k:c[k] for k in SCHEMAS['strata'][9:]}})
        self.strata.clear()
        for action,n in sorted(self.vocal.items()):self.sink('vocal',{'tick':tick,'vocal_action':action,'n':n})
        self.vocal.clear();self.last_flush=tick

    def tick_end(self,sim):
        row=asdict(sim.statistics)
        row.update(physical_rows=sum(h.brain.memory_size for h in sim.agents if isinstance(h.brain,LearningBrain)),
                   physical_agents=sum(isinstance(h.brain,LearningBrain) for h in sim.agents),distinct_global_states=len(self.global_states))
        self.sink('ticks',row)
        self.peak_live_state_histories=max(self.peak_live_state_histories,sum(len(h) for h in self.agents.values()))
        self.peak_live_contact_histories=max(self.peak_live_contact_histories,sum(len(h) for h in self.partners.values()))
        if sim.tick and sim.tick%self.window==0:
            self.flush(sim.tick);self.probes(sim)

    def probes(self,sim):
        agents=[(h,self.agents.get(h.id,{})) for h in sorted(sim.agents,key=lambda h:h.id)]
        for minimum in (2,5):
            for kind in ('all_social','memory_only'):
                self.sink('probes',probe(agents,sim.social_learning.mode,sim.tick,minimum,kind))

    def finish(self,sim):
        if self.finished:raise ValueError('Diagnostic already finished')
        if sim.tick!=self.last_flush:
            self.flush(sim.tick);self.probes(sim)
        for h in sim.agents:
            self.dist['survival_age','run_end_censored',h.age]+=1;self.close_agent(h.id,'run_end_censored')
        for (metric,reason,value),n in sorted(self.dist.items()):self.sink('distributions',{'metric':metric,'reason':reason,'value':value,'count':n})
        self.finished=True


def write_metadata(path,sim,observer,writer,condition=None,fingerprints=None):
    if not observer.finished:raise ValueError('Recording must complete before metadata')
    payload={'simulation':asdict(sim.config),'social':asdict(sim.social_config),'policy':asdict(sim.social_policy) if sim.social_policy else None,'social_learning':asdict(sim.social_learning)}
    from simulation.social_observer import config_hash
    source=['config.py','main.py']
    for folder in ('agents','simulation','systems','world'):source.extend(str(p) for p in sorted(Path(folder).glob('*.py')))
    meta={'version':'0.7.2','format':'social_learning_aggregates','schema_version':1,'completed':True,'config':payload,'config_id':config_hash(payload),'seed':sim.config.random_seed,
          'condition':condition or sim.social_learning.mode,'tick':sim.tick,'statistics':asdict(sim.statistics),'window':observer.window,
          'totals':dict(sorted(observer.total.items())),'distinct_global_states':len(observer.global_states),'unique_visual_targets':len(observer.visual_targets),'peak_live_state_histories':observer.peak_live_state_histories,
          'peak_live_contact_histories':observer.peak_live_contact_histories,'fingerprints':fingerprints,'action_sha256':observer.action_digest.hexdigest(),'schemas':SCHEMAS,
          'files':{k:{'file':p.name,'sha256':hash_file(p),'bytes':p.stat().st_size,'records':writer.counts[k]} for k,p in paths(path).items()},
          'source_sha256':{p:hash_file(p) for p in source},
          'definitions':{'snapshot':'One actual sequential visual batch, remembered membership BEFORE this batch updates LTM. Fifth commitment not a pre-update memory.',
            'physical_state':'Legacy key + visible token (B/C) + remembered direction (C only). Valid social combinations 11 (B),37 (C); no ID/offset in suffix.',
            'vocal_state':'Exactly legacy physiological/visual/auditory state; no social feature.',
            'q_novelty':'first_agent_states counts first lifetime decision keys per living-agent history; first_global_states counts distinct keys across all agents. Actual allocation/eviction hooks include previous-transition recreation and terminal updates.',
            'movement':'Actual moved actors only; Chebyshev before/after against nearest visual or pre-remembered target frozen at its pre-action position. Lowest ID distance ties. Stationary/blocked actions outside movement rate denominator; equal distance measured separately.',
            'contact':'Actual pre-action observation opportunities; repeated means seen previously. Long-gap means missed intervening opportunity and elapsed>=20. Retained-before-update is remembered reunion; no auditory identity or remote death knowledge.',
            'strata':'Pre-action local/visible resources, physiological need bins, age//25, actual visible-count bin. Visual count is a sensory exposure, not a fixed-radius physical-density estimate.',
            'probe':'Read-only retained observed same-agent original-key contexts, current-residence visits>=2/5 and TD updates>=1; all_social varies suffix, memory_only holds B token fixed. No absent/synthetic rows or RNG. Pair counts are diagnostic support, not independent seeds.',
            'auditory':'IDENTIFIED/MASKED/SILENCE decisions, not omniscient emitted-within-radius counts. Vocal actions in separate stream support arbitrary vocabulary.',
            'replication':'One simulation seed is one replication; no contact pooling for inferential n. No social rewards or social-preference claim.'}}
    with Path(str(path)+'.metadata.json').open('x') as f:f.write(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    return meta
