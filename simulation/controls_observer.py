"""v0.7.3 optional ordered outcomes and passive state-fragmentation diagnostics."""
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import csv,gzip,hashlib,io,json
from pathlib import Path

from agents.social_learning import SocialFeatures,extract
from simulation.social_learning_observer import SocialLearningObserver,SCHEMAS as OLD_SCHEMAS,probe
from simulation.statistics import Statistics
from simulation.vocal_learning_observer import hash_file


def _pairs():
    result=[]
    for visible in range(11):
        for remembered in range(6):
            try:SocialFeatures(visible,remembered)
            except ValueError:continue
            result.append((visible,remembered))
    return tuple(result)

C_PAIRS=_pairs()
C_INDEX={pair:i for i,pair in enumerate(C_PAIRS)}
SCHEMAS={**OLD_SCHEMAS,
 'outcomes':('sequence','tick','agent_id','base_id','b_token','c_token','used_token','action','exploratory','reward','terminal'),
 'retained':('tick','visits','updates','rows'),
 'lifetime_updates':('updates','states')}


def paths(path,light=False):
    p=Path(path);stem=p.name[:-7] if p.name.endswith('.csv.gz') else p.stem
    suffix='.csv.gz' if p.suffix=='.gz' else '.csv'
    roles=('ticks','states','outcomes','retained','lifetime_updates','distributions','probes') if light else tuple(SCHEMAS)
    return {k:p if k=='ticks' else p.with_name(stem+'.'+k+suffix) for k in roles}


class ControlsWriter:
    def __init__(self,streams):
        self.writers={k:csv.DictWriter(s,fieldnames=SCHEMAS[k],extrasaction='ignore') for k,s in streams.items()};self.counts=Counter()
        for writer in self.writers.values():writer.writeheader()
    @classmethod
    def open(cls,stack,path,light=False):
        outputs=paths(path,light)
        if any(p.exists() for p in (*outputs.values(),Path(str(path)+'.metadata.json'))):raise ValueError('Refusing existing controls outputs')
        streams={}
        for kind,p in outputs.items():
            p.parent.mkdir(parents=True,exist_ok=True);raw=stack.enter_context(p.open('xb'))
            stream=stack.enter_context(gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=1)) if p.suffix=='.gz' else raw
            streams[kind]=stack.enter_context(io.TextIOWrapper(stream,encoding='utf8',newline=''))
        return cls(streams)
    def __call__(self,kind,row):
        if kind not in self.writers:return
        self.writers[kind].writerow(row);self.counts[kind]+=1


class ControlsObserver(SocialLearningObserver):
    supports_nuisance=True
    def __init__(self,sink,condition,light=False,window=100):
        if condition not in ('A','B','C','D','E'):raise ValueError('Expected A..E')
        super().__init__(sink,window);self.condition=condition;self.light=light;self.sequence=0
        self.lifetime_updates={};self.update_hist=Counter()

    @property
    def suffix_size(self):return 0 if self.condition=='A' else 2 if self.condition=='C' else 1

    def begin(self,human,observation,tick):
        if self.light:
            if self.finished:raise RuntimeError('Observer finished')
            self.tick=tick;self.agent_id=human.id;self.brain=human.brain;self.brain._physical_observer=self
            self.history=self.agents.setdefault(human.id,{})
            self.features=extract(getattr(observation,'individuals',()),getattr(human,'social_memory',None),include_memory=True)
            self.start=(human.x,human.y)
        else:super().begin(human,observation,tick)

    def start_state(self,state,present):
        super().start_state(state,present)
    def allocated(self,state):
        super().allocated(state)
    def evicted(self,state):
        super().evicted(state)
    def updated(self,state,index,old,new,terminal):
        super().updated(state,index,old,new,terminal)
        h=self.lifetime_updates.setdefault(self.agent_id,Counter());h[state]+=1

    def selected(self,brain,state,values,index):
        suffix=self.suffix_size;base=state[:-suffix] if suffix else state
        if base not in self.keys:
            self.keys[base]=len(self.keys);self.sink('states',{'base_id':self.keys[base],'key_json':json.dumps(base,separators=(',',':'))})
        b=self.features.visible;c=C_INDEX[(b,self.features.remembered_direction)]
        used=-1 if self.condition=='A' else b if self.condition=='B' else c if self.condition=='C' else state[-1]
        self.turn={'sequence':self.sequence,'tick':self.tick,'agent_id':self.agent_id,'base_id':self.keys[base],
                   'b_token':b,'c_token':c,'used_token':used,'action':index,'exploratory':int(brain.last_exploratory)}
        self.sequence+=1
        h=self.record(state);h[0]+=1;h[1]+=1
        if self.light:self.add(decisions=1)
        else:
            key=((self.tick-1)//self.window,self.agent_id,self.keys[base],b,self.features.remembered_direction,index,int(brain.last_exploratory))
            self.actions[key]+=1
        self.add(first_agent_states=self.novel,first_global_states=self.new_global,recreated_decisions=self.recreated,
                 occupancy_sum=brain.memory_size,at_capacity=brain.memory_size==brain.capacity,
                 exploratory=int(brain.last_exploratory),exploitative=int(not brain.last_exploratory),**{f'physical_action_{index}':1})

    def outcome(self,human,decision,cause):
        self.sink('outcomes',{**self.turn,'reward':human.brain.last_reward,'terminal':int(cause is not None)})
        if self.light:
            self.action_digest.update(json.dumps((self.tick,human.id,*self.start,decision.action.value,decision.signal_id,
                human.x,human.y,human.age,human.hunger,human.thirst,human.energy,human.alive),separators=(',',':')).encode())
            del human.brain._physical_observer;self.brain=None
            if cause is not None:self.close_agent(human.id,'observer_death')
        else:super().outcome(human,decision,cause)

    def close_agent(self,identity,reason):
        counts=self.lifetime_updates.pop(identity,Counter())
        # Include observed states with zero updates (e.g. pending cutoff rows).
        for state in self.agents.get(identity,{}):self.update_hist[counts[state]]+=1
        if self.light:
            for visits,_,_ in self.agents.pop(identity,{}).values():self.dist['physical_state_visits',reason,visits]+=1
        else:super().close_agent(identity,reason)

    def probes(self,sim):
        agents=[(h,self.agents.get(h.id,{})) for h in sorted(sim.agents,key=lambda h:h.id)]
        mode='none' if self.condition=='A' else 'memory' if self.condition=='C' else 'perception'
        for minimum in (2,5):
            self.sink('probes',probe(agents,mode,sim.tick,minimum,'all_social'))
        hist=Counter()
        for h,history in agents:
            for state in h.brain._values:
                values=history.get(state,(0,0,0));hist[values[1],values[2]]+=1
        for (visits,updates),rows in sorted(hist.items()):self.sink('retained',{'tick':sim.tick,'visits':visits,'updates':updates,'rows':rows})

    def tick_end(self,sim):
        if self.light:
            self.sink('ticks',{**asdict(sim.statistics),'physical_rows':sum(h.brain.memory_size for h in sim.agents),
                              'physical_agents':sim.population,'distinct_global_states':len(self.global_states)})
            if sim.tick and sim.tick%self.window==0:self.probes(sim);self.last_flush=sim.tick
        else:super().tick_end(sim)

    def finish(self,sim):
        if self.light:
            if self.finished:raise ValueError('Already finished')
            if sim.tick!=self.last_flush:self.probes(sim)
            for h in sim.agents:self.close_agent(h.id,'run_end_censored')
            for (metric,reason,value),count in sorted(self.dist.items()):self.sink('distributions',{'metric':metric,'reason':reason,'value':value,'count':count})
            self.finished=True
        else:
            super().finish(sim)
        for updates,states in sorted(self.update_hist.items()):self.sink('lifetime_updates',{'updates':updates,'states':states})


def physical_fingerprint(sim):
    """Actual physical tables/pending state, including untouched newborns."""
    rows=[]
    for h in sim.agents:
        b=h.brain;p=b._pending
        rows.append((h.id,tuple((k,tuple(v)) for k,v in b._values.items()),None if p is None else (p.state,p.action_index,p.reward)))
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()


def write_metadata(path,sim,observer,writer,*,realization=0,calibration=None,fingerprints=None,version='0.7.3'):
    if not observer.finished:raise ValueError('Complete recording before metadata')
    from agents.nuisance import NonstationaryNuisanceModel
    scheduled=sim.nuisance is not None and isinstance(sim.nuisance.model,NonstationaryNuisanceModel)
    if scheduled:version='0.7.4'
    from simulation.social_observer import config_hash
    payload={'simulation':asdict(sim.config),'social':asdict(sim.social_config),'policy':asdict(sim.social_policy) if sim.social_policy else None,
             'social_learning':asdict(sim.social_learning),'nuisance':asdict(sim.nuisance) if sim.nuisance is not None else None}
    sources=['config.py','main.py']
    for d in ('agents','simulation','systems','world'):sources.extend(str(p) for p in sorted(Path(d).glob('*.py')))
    files=paths(path,observer.light)
    meta={'version':version,'format':'social_information_controls','completed':True,'schema_version':1,'config':payload,'config_id':config_hash(payload),
          'condition':observer.condition,'seed':sim.config.random_seed,'nuisance_seed':None if sim.nuisance is None else sim.nuisance.seed,'realization':realization,
          'light':observer.light,'tick':sim.tick,'statistics':asdict(sim.statistics),'totals':dict(sorted(observer.total.items())),
          'action_sha256':observer.action_digest.hexdigest(),'physical_q_sha256':physical_fingerprint(sim),'living_ids':[h.id for h in sim.agents],
          'fingerprints':fingerprints,'calibration':calibration,'c_pairs':C_PAIRS,
          'schemas':{k:SCHEMAS[k] for k in files},'files':{k:{'file':p.name,'sha256':hash_file(p),'bytes':p.stat().st_size,'records':writer.counts[k]} for k,p in files.items()},
          'source_sha256':{p:hash_file(p) for p in sources},
          'definitions':{'outcomes':'One actual ordered decision with existing immediate physiological reward, after outcome. Original key encoded at actual decision. No inferred reward or next observation.',
            'used_token':'A=-1(no suffix); B=visible11; C=canonical valid(visible,remembered) index37; D/E independent token11/37. Actual C Q suffix remains the original pair.',
            'next':'Next row of same surviving individual is its next real observation; terminal rows have no next state. Last alive row is right-censored and its pending TD update is not fabricated.',
            'nuisance':'Stationary categorical initialization, then hold with alpha else fresh categorical draw; independent dedicated substreams allocated by construction order. No ID, age, position, state or outcome is passed to the generator.',
            'policy_probe':'Same v072 retained-row >=2/5 visits and >=1 current-row update, original state held fixed. D/E vary nuisance only. Observed rows only.',
            'retained':'Histogram of actual current-residence Q visits and TD updates in living tables, not lifetime visits. Lifetime update histograms separate.',
            'interpretation':'Exogenous innovations do not guarantee zero empirical correlation after policy/selection feedback. Predictive analysis uses fixed B/C experience and independent attached nuisance; no ecological counterfactual claims.'}}
    if scheduled:
        meta['definitions']['nuisance']='Frozen calendar schedule chooses pi(t),alpha(t), not a tick hash. Independent private streams and stationary initial uniform mapped at first actual turn. No reset at boundaries, no live social/X/resource/reward/ID feedback. First-turn read-only preview maps that same uniform without draws.'
    with Path(str(path)+'.metadata.json').open('x') as f:f.write(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    return meta
