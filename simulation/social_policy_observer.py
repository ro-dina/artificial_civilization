"""v0.7.1 compact optional diagnostics, separate from untouched v0.7.0 CSV."""
from collections import Counter
from dataclasses import asdict
import csv,gzip,io,json
from pathlib import Path

from config import SocialPolicyConfig
from agents.social_memory_policy import PolicyUpdate,log_score,contact_log
from simulation.social_observer import config_hash
from simulation.vocal_learning_observer import hash_file

SUMMARY=('tick','event','agent_id','age','generation','visible','known_before','long_term_occupancy','candidate_occupancy',
 'candidate_created','candidate_progress','candidate_resets','candidate_evictions','candidate_commits','long_term_commits',
 'long_term_evictions','reunions','population','births','deaths','death_cause')
TRACE=('tick','agent_id','event','individual_id','count','first_seen_tick','last_seen_tick','committed_tick','streak',
       'elapsed_since_last_seen','log_strength','log_score')


def contacts(update):
    """Return (id, recognized-before, committed-now, entry, evicted, kind)."""
    if isinstance(update,PolicyUpdate):
        for c in update.contacts:yield c.individual_id,c.recognized,c.kind=='commitment',c.entry,c.evicted,c.kind
    else:
        for c in update.contacts:
            committed=c.kind=='first_encounter'
            yield c.entry.individual_id,not committed,committed,c.entry,c.evicted,c.kind


def paths(path,trace=False):
    p=Path(path);base=p.name[:-7] if p.name.endswith('.csv.gz') else p.stem
    return {'summary':p,**({'trace':p.with_name(base+'.trace'+('.csv.gz' if p.suffix=='.gz' else '.csv'))} if trace else {})}


class PolicyWriter:
    def __init__(self,streams):
        self.writers={k:csv.writer(v) for k,v in streams.items()};self.counts=Counter()
        for k,w in self.writers.items():w.writerow(SUMMARY if k=='summary' else TRACE)
    @classmethod
    def open(cls,stack,path,trace=False):
        destinations=paths(path,trace)
        if any(p.exists() for p in (*destinations.values(),Path(str(path)+'.metadata.json'))):
            raise ValueError('Refusing existing policy diagnostics')
        streams={}
        for k,p in destinations.items():
            p.parent.mkdir(parents=True,exist_ok=True)
            if p.suffix=='.gz':
                raw=stack.enter_context(p.open('xb'));z=stack.enter_context(gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=1))
                streams[k]=stack.enter_context(io.TextIOWrapper(z,encoding='utf-8',newline=''))
            else:streams[k]=stack.enter_context(p.open('x',encoding='utf-8',newline=''))
        return cls(streams)
    def __call__(self,kind,row):
        if kind not in self.writers:return
        self.writers[kind].writerow(row.get(k) for k in (SUMMARY if kind=='summary' else TRACE));self.counts[kind]+=1


class PolicyObserver:
    def __init__(self,social,policy,sink):
        self.config=social;self.policy=policy or SocialPolicyConfig();self.sink=sink;self.counts=Counter();self.finished=False
        self.trace_enabled='trace' in sink.writers if hasattr(sink,'writers') else True
    def _write(self,tick,event,**values):
        if self.finished:raise RuntimeError('Policy diagnostics already finished')
        self.sink('summary',{'tick':tick,'event':event,**values});self.counts[event]+=1
    def life(self,tick,human,event_kind,cause=None):
        self._write(tick,event_kind,agent_id=human.id,age=human.age,generation=human.generation,
                    long_term_occupancy=len(human.social_memory),candidate_occupancy=len(getattr(human.social_memory,'_candidates',())),death_cause=cause)
    def observation(self,tick,human,update,observation):
        counts=Counter(e.kind for e in getattr(update,'candidate_events',()))
        committed=evicted=reunions=0
        for identity,known,commit,entry,old,kind in contacts(update):
            committed+=commit;evicted+=old is not None;reunions+=kind=='reunion'
            if commit:self._trace(tick,human.id,'commitment',entry)
            if old is not None:self._trace(tick,human.id,'long_term_eviction',old)
        for event in (getattr(update,'candidate_events',()) if self.trace_enabled else ()):
            c=event.candidate
            self.sink('trace',{'tick':tick,'agent_id':human.id,'event':event.kind,'individual_id':c.individual_id,
                'count':c.progress,'first_seen_tick':c.first_seen_tick,'last_seen_tick':c.last_seen_tick,'streak':c.progress})
        self._write(tick,'observation',agent_id=human.id,age=human.age,generation=human.generation,
            visible=len(observation.individuals),known_before=update.known_visible,long_term_occupancy=update.occupancy,
            candidate_occupancy=getattr(update,'candidate_occupancy',0),candidate_created=counts['candidate_created'],
            candidate_progress=counts['candidate_progress'],candidate_resets=counts['candidate_reset'],candidate_evictions=counts['candidate_eviction'],
            candidate_commits=counts['candidate_committed'],long_term_commits=committed,long_term_evictions=evicted,reunions=reunions)
    def _trace(self,tick,agent_id,event,entry):
        if not self.trace_enabled:return
        value=getattr(entry,'log_strength',None)
        if value is None:value=contact_log(entry.encounter_count)
        self.sink('trace',{'tick':tick,'agent_id':agent_id,'event':event,'individual_id':entry.individual_id,
            'count':entry.encounter_count,'first_seen_tick':entry.first_seen_tick,'last_seen_tick':entry.last_seen_tick,
            'committed_tick':getattr(entry,'committed_tick',entry.first_seen_tick),
            'streak':getattr(entry,'consecutive_observations',None),'elapsed_since_last_seen':tick-entry.last_seen_tick,
            'log_strength':value,'log_score':log_score(entry,tick,self.policy.decay_tau) if hasattr(entry,'log_strength') else None})
    def tick_end(self,statistics):
        self._write(statistics.tick,'tick_end',population=statistics.population,births=statistics.births,deaths=statistics.deaths)
    def finish(self,statistics):
        self._write(statistics.tick,'run_end',population=statistics.population,births=statistics.births,deaths=statistics.deaths);self.finished=True


def write_metadata(path,sim,writer,trace=False):
    if not sim.social_observer.finished:raise ValueError('Policy recording not finished')
    policy=sim.social_policy or SocialPolicyConfig()
    payload={'simulation':asdict(sim.config),'social':asdict(sim.social_config),'policy':asdict(policy)}
    files=paths(path,trace)
    meta={'version':'0.7.1','format':'social_memory_policy_diagnostics','schema_version':1,'completed':True,
          'config':payload,'config_id':config_hash(payload),'seed':sim.config.random_seed,'final_tick':sim.tick,
          'statistics':asdict(sim.statistics),'candidate_capacity':policy.resolved_candidate_capacity(sim.social_config.social_memory_capacity),
          'schemas':{'summary':SUMMARY,**({'trace':TRACE} if trace else {})},
          'files':{k:{'sha256':hash_file(p),'bytes':p.stat().st_size,'records':writer.counts[k]} for k,p in files.items()},
          'source_sha256':{p:hash_file(p) for p in ('config.py','agents/social_memory.py','agents/social_memory_policy.py','simulation/social_policy_observer.py','simulation/simulation.py','main.py')},
          'definitions':{'recognition':'Resident membership before this observation; same-tick commitment is not recognition.',
          'strength':'Observed opportunities in this retained episode, including forming N sightings; not friendship.',
          'trace':'Optional transition records only; no repeated recognized-contact rows. Legacy A streak is unavailable in legacy entries.',
          'score':'log(log1p(C))-(t-last)/tau. Missing if unrepresentable; count/time/log-strength plus tau define stable components.',
          'ordering':'Committed hits, continuing candidates, new candidates, each ascending ID; legacy A unchanged.',
          'no_reward':'No social reward or Q input.'}}
    with Path(str(path)+'.metadata.json').open('x') as stream:stream.write(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    return meta
