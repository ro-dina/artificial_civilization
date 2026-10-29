"""Memory mechanics on a shared visual history; seeds, not contacts, replicate."""
from collections import Counter
import hashlib,json
from experiments.auditory_policy import frozen
from dataclasses import dataclass,field,asdict

from agents.social_memory_policy import make_memory
from config import SocialConfig,SocialPolicyConfig
from simulation.social_policy_observer import contacts

COUNTS=('observations','contacts','new_observed_pairs','candidate_created','candidate_progress','candidate_reset','candidate_eviction',
 'candidate_committed','allocations','evictions','first_committed_pairs','relearning_allocations','relearned_pairs','repeatedly_forgotten_pairs',
 'hits','reunions','eligible_contacts','eligible_hits','repeated_contacts','repeated_hits','previously_committed_contacts',
 'previously_committed_hits','forgotten_contacts','never_committed_recontacts','eligible_forgotten_misses','eligible_never_committed_misses',
 'qualifying_reunion_contacts','qualifying_reunion_hits','full_observations','occupancy_sum','candidate_occupancy_sum',
 'living_agent_ticks','full_agent_ticks','living_occupancy_sum','candidate_living_sum','births','deaths')
RATES={'recognition_eligible':('eligible_hits','eligible_contacts'),'recognition_repeated':('repeated_hits','repeated_contacts'),
 'recognition_previously_committed':('previously_committed_hits','previously_committed_contacts'),
 'recognition_qualifying_reunion':('qualifying_reunion_hits','qualifying_reunion_contacts'),
 'mean_occupancy':('occupancy_sum','observations'),'mean_candidate_occupancy':('candidate_occupancy_sum','observations'),
 'fraction_observations_full':('full_observations','observations'),'fraction_agents_full':('full_agent_ticks','living_agent_ticks'),
 'mean_living_occupancy':('living_occupancy_sum','living_agent_ticks'),'candidate_evictions_per_observation':('candidate_eviction','observations'),
 'evictions_per_allocation':('evictions','allocations'),'evictions_per_observation':('evictions','observations')}
METRICS=(*COUNTS,*RATES,'peak_occupancy','peak_candidate_occupancy','active_observers','population',
         'cumulative_ever_committed_fraction','cumulative_candidate_conversion','mean_age',
         'mean_strength_at_eviction','mean_elapsed_at_eviction','mean_retention_at_eviction')


def ratio(a,b):return a/b if b else None


@dataclass(frozen=True)
class Treatment:
    condition:str
    capacity:int=32
    threshold:int=1
    eviction:str='lru'
    tau:float=100.
    candidate_capacity:int|None=None
    reunion:int=20
    @property
    def policy(self):return SocialPolicyConfig(self.threshold,self.eviction,self.tau,candidate_capacity=self.candidate_capacity)
    @property
    def social(self):return SocialConfig(True,self.capacity,self.reunion)
    @property
    def key(self):return f'{self.condition}_N{self.threshold}_{self.eviction}_K{self.capacity}_C{self.policy.resolved_candidate_capacity(self.capacity)}_T{self.tau:g}'
    def fields(self):return {'treatment':self.key,'condition':self.condition,'formation_threshold':self.threshold,'eviction_policy':self.eviction,
        'capacity':self.capacity,'candidate_capacity':self.policy.resolved_candidate_capacity(self.capacity),'tau':self.tau}
    def validate(self):self.social;self.policy


def primary():
    return (Treatment('A'),Treatment('B',threshold=5),Treatment('C',eviction='contact_decay'),Treatment('D',threshold=5,eviction='contact_decay'))


class Tally:
    def __init__(self):
        self.c=Counter();self.hist={};self.active=set();self.peak=0;self.candidate_peak=0;self.population=None;self.age_sum=0
    def add_hist(self,kind,value,reason='observed'):
        self.hist.setdefault((kind,reason),Counter())[value]+=1
    def values(self,cumulative=None):
        cum=cumulative or self
        means={}
        for name,kind in (('mean_strength_at_eviction','strength_at_eviction'),('mean_elapsed_at_eviction','elapsed_at_eviction'),('mean_retention_at_eviction','retention_duration')):
            h=self.hist.get((kind,'eviction'),{});means[name]=ratio(sum(v*n for v,n in h.items()),sum(h.values()))
        return {**{k:self.c[k] for k in COUNTS},**{k:ratio(self.c[a],self.c[b]) for k,(a,b) in RATES.items()},
            'peak_occupancy':self.peak,'peak_candidate_occupancy':self.candidate_peak,'active_observers':len(self.active),
            'population':self.population,'cumulative_ever_committed_fraction':ratio(cum.c['first_committed_pairs'],cum.c['new_observed_pairs']),
            'cumulative_candidate_conversion':ratio(cum.c['candidate_committed'],cum.c['candidate_created']),
            'mean_age':ratio(self.age_sum,self.c['observations']),**means}


@dataclass(slots=True)
class Partner:
    total:int=0
    streak:int=0
    last_tick:int|None=None
    eligible:bool=False
    first_opportunity:int=0


class Agent:
    def __init__(self,treatments,birth,generation):
        self.birth=birth;self.generation=generation;self.opportunity=0;self.previous_tick=None
        self.partners={};self.visible=set()
        self.memories=[make_memory(t.social,t.policy) for t in treatments]
        self.committed=[set() for _ in treatments];self.forgotten=[Counter() for _ in treatments];self.relearned=[set() for _ in treatments]


class Evaluation:
    """Only offline histories are unbounded; private memories/candidates stay bounded."""
    def __init__(self,treatments,window_size=100,age_bin=25):
        self.treatments=tuple(treatments)
        if not self.treatments or len({t.key for t in treatments})!=len(treatments):raise ValueError('Distinct treatments required')
        for t in treatments:t.validate()
        if type(window_size) is not int or window_size<1 or type(age_bin) is not int or age_bin<1:raise ValueError('Positive bins required')
        self.window_size=window_size;self.age_bin=age_bin;self.agents={};self.total=[Tally() for _ in treatments]
        self.windows={};self.window_cumulative={};self.ages={};self.source_hist=Counter();self.tick=0;self.population=0
        self.unique_targets=set();self.peak_live_pairs=0;self.private_state_hashes=[]
    def groups(self,index,tick,age=None):
        window=(tick-1)//self.window_size if tick else 0
        result=[self.total[index],self.windows.setdefault((index,window),Tally())] if tick else [self.total[index]]
        if age is not None:result.append(self.ages.setdefault((index,age//self.age_bin),Tally()))
        return result
    def _close(self,agent,tick,reason):
        for identity in agent.visible:self.source_hist[('episode_length',reason,agent.partners[identity].streak)]+=1
        for i,memory in enumerate(agent.memories):
            for entry in memory.entries():
                duration=tick-getattr(entry,'committed_tick',entry.first_seen_tick)
                for group in self.groups(i,tick):group.add_hist('retention_duration',duration,reason)
    def accept(self,row):
        self.tick=row.tick
        if row.kind in ('founder','birth'):
            self.agents[row.agent_id]=Agent(self.treatments,row.tick,row.generation)
            if row.kind=='birth':
                for i in range(len(self.treatments)):
                    for g in self.groups(i,row.tick):g.c['births']+=1
            return
        if row.kind=='death':
            agent=self.agents.pop(row.agent_id);self._close(agent,row.tick,'observer_death')
            for i in range(len(self.treatments)):
                for g in self.groups(i,row.tick):g.c['deaths']+=1
            return
        if row.kind=='tick_end':
            self.peak_live_pairs=max(self.peak_live_pairs,sum(len(a.partners) for a in self.agents.values()))
            self.population=row.population
            for i,t in enumerate(self.treatments):
                memories=[a.memories[i] for a in self.agents.values()]
                full=sum(len(m)==t.capacity for m in memories);occupancy=sum(len(m) for m in memories)
                candidates=sum(len(getattr(m,'_candidates',())) for m in memories)
                for g in self.groups(i,row.tick):
                    g.population=row.population
                    if row.tick:
                        g.c.update(living_agent_ticks=row.population,full_agent_ticks=full,living_occupancy_sum=occupancy,candidate_living_sum=candidates)
                if row.tick:
                    c=self.total[i].c
                    self.window_cumulative[(i,(row.tick-1)//self.window_size)]=(c['first_committed_pairs'],c['new_observed_pairs'],c['candidate_committed'],c['candidate_created'])
            return
        if row.kind=='run_end':return
        if row.kind!='observation':raise ValueError('Unknown evaluation row')
        agent=self.agents[row.agent_id];agent.opportunity+=1;visible=set(row.visible_ids)
        for identity in agent.visible-visible:self.source_hist[('episode_length','observed_absence',agent.partners[identity].streak)]+=1
        prior={identity:agent.partners.get(identity) for identity in row.visible_ids}
        for i,t in enumerate(self.treatments):
            memory=agent.memories[i];update=memory.observe(row.visible_ids,row.tick,capture=True)
            groups=self.groups(i,row.tick,row.age)
            for g in groups:
                g.active.add(row.agent_id);g.age_sum+=row.age;g.c['observations']+=1
                g.c['occupancy_sum']+=len(memory);g.c['candidate_occupancy_sum']+=len(getattr(memory,'_candidates',()))
                g.c['full_observations']+=len(memory)==t.capacity;g.peak=max(g.peak,len(memory));g.candidate_peak=max(g.candidate_peak,len(getattr(memory,'_candidates',())))
            for event in getattr(update,'candidate_events',()):
                for g in groups:g.c[event.kind]+=1
            for identity,recognized,committed,entry,evicted,kind in contacts(update):
                h=prior[identity];ever=identity in agent.committed[i]
                eligible=h is not None and h.eligible
                repeat=h is not None
                qualifying=eligible and h.last_tick!=agent.previous_tick and row.tick-h.last_tick>=t.reunion
                for g in groups:
                    c=g.c;c['contacts']+=1;c['hits']+=recognized;c['new_observed_pairs']+=not repeat
                    c['eligible_contacts']+=eligible;c['eligible_hits']+=eligible and recognized
                    c['repeated_contacts']+=repeat;c['repeated_hits']+=repeat and recognized
                    c['previously_committed_contacts']+=ever;c['previously_committed_hits']+=ever and recognized
                    c['forgotten_contacts']+=ever and not recognized;c['never_committed_recontacts']+=repeat and not ever
                    c['eligible_forgotten_misses']+=eligible and ever and not recognized
                    c['eligible_never_committed_misses']+=eligible and not ever and not recognized
                    c['qualifying_reunion_contacts']+=qualifying;c['qualifying_reunion_hits']+=qualifying and recognized
                    c['reunions']+=kind=='reunion'
                if committed:
                    for g in groups:
                        g.c['allocations']+=1
                        if not ever:
                            g.c['first_committed_pairs']+=1
                            g.add_hist('contact_opportunities_until_first_commit',(h.total if h else 0)+1)
                            g.add_hist('observer_opportunities_until_first_commit',agent.opportunity-(h.first_opportunity if h else agent.opportunity)+1)
                        else:g.c['relearning_allocations']+=1
                    if ever and identity not in agent.relearned[i]:
                        for g in groups:g.c['relearned_pairs']+=1
                        agent.relearned[i].add(identity)
                    agent.committed[i].add(identity)
                if evicted is not None:
                    old_id=evicted.individual_id;agent.forgotten[i][old_id]+=1
                    for g in groups:
                        g.c['evictions']+=1;g.c['repeatedly_forgotten_pairs']+=agent.forgotten[i][old_id]==2
                        g.add_hist('strength_at_eviction',evicted.encounter_count,'eviction')
                        g.add_hist('elapsed_at_eviction',row.tick-evicted.last_seen_tick,'eviction')
                        g.add_hist('retention_duration',row.tick-getattr(evicted,'committed_tick',evicted.first_seen_tick),'eviction')
        for identity in row.visible_ids:
            h=agent.partners.get(identity)
            if h is None:h=Partner(first_opportunity=agent.opportunity);agent.partners[identity]=h
            h.streak=h.streak+1 if h.last_tick==agent.previous_tick else 1
            h.total+=1;h.last_tick=row.tick;h.eligible=h.eligible or h.streak>=5
            self.unique_targets.add(identity)
        agent.visible=visible;agent.previous_tick=row.tick
    def finish(self):
        self.private_state_hashes=[hashlib.sha256(json.dumps(tuple((identity,frozen(vars(agent.memories[i])))
            for identity,agent in sorted(self.agents.items())),separators=(',',':')).encode()).hexdigest() for i in range(len(self.treatments))]
        for agent in self.agents.values():self._close(agent,self.tick,'run_end_censored')
        self.agents.clear()
        reference=self.total[0].c
        for other in self.total[1:]:
            for key in ('observations','contacts','eligible_contacts','repeated_contacts','qualifying_reunion_contacts','new_observed_pairs'):
                if other.c[key]!=reference[key]:raise RuntimeError(f'Treatments did not receive identical exposures: {key}')
