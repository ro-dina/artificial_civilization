"""Streaming allocation episodes and consecutive same-individual/state encounters.

Only actual events/counters and recorded summaries are used. No Q tables exist
in this analyzer. Episode detail streams to gzip; memory follows active rows.
"""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import csv
import gzip
import io
import math

import numpy as np

from experiments.vocal_persistence_data import TickCursor, tick_records, integer
from experiments.vocal_persistence_metrics import (Bag, I, SUMMARY_COLUMNS, SILENCE_CATEGORIES,
                                                   encounter, decision_vector, pair_vector)

CONTEXT=('run','condition','seed')
WINDOW=('window_start','window_end','actual_window_end','observed_ticks','partial_window','complete_window')
EPISODE_COLUMNS=('agent_id','state_id','episode','allocated_tick','end_tick','end_reason','storage_duration',
    'first_encounter_tick','last_encounter_tick','encounter_span','first_age','encounters','row_updates',
    'retained_pairs','mean_interval','max_interval','unique_max_fraction','mean_gap','mean_p_silence',
    'observed_silence_rate','choice_match_rate','id_pairs','id_same','identity_coverage','preferred_action_match_rate',
    'set_pairs','mean_max_set_jaccard','mean_exact_policy_tv','eligible_early_late','early_encounters','late_encounters',
    'early_mean_gap','late_mean_gap','delta_gap','early_mean_p_silence','late_mean_p_silence','delta_p_silence',
    'first_observed_unique_tick','first_observed_unique_visit')
SCHEMAS={
 'persistence_run_summary':(*CONTEXT,*SUMMARY_COLUMNS,'agents','episodes','actual_evictions','deaths','run_ends',
                            'eligible_paired_episodes','mean_paired_gap_change','mean_paired_p_silence_change'),
 'persistence_by_window':(*CONTEXT,*WINDOW,*SUMMARY_COLUMNS),
 'persistence_by_episode':(*CONTEXT,*EPISODE_COLUMNS),
 'persistence_by_agent':(*CONTEXT,'agent_id','generation','first_tick','last_tick','first_age','last_age','death_tick',
                         'end_reason','episodes','actual_evictions',*SUMMARY_COLUMNS),
 'persistence_by_agent_window':(*CONTEXT,*WINDOW,'agent_id',*SUMMARY_COLUMNS),
 'persistence_by_visit_index':(*CONTEXT,'visit_kind','visit_index',*SUMMARY_COLUMNS),
 'persistence_by_age':(*CONTEXT,'age_start',*SUMMARY_COLUMNS),
 'persistence_by_experience':(*CONTEXT,'update_bin',*SUMMARY_COLUMNS),
 'paired_episode_development':(*CONTEXT,'first_age_start','episodes','mean_gap_change','mean_p_silence_change'),
 'preference_transitions':(*CONTEXT,*WINDOW,'branch','from_action','to_action','count','branch_pairs','window_id_pairs','probability'),
 'silence_preference_transitions':(*CONTEXT,*WINDOW,'from_category','to_category','count','origin_pairs','probability','retained_pairs'),
 'retained_vs_recreated':(*CONTEXT,*WINDOW,'encounter_class',*SUMMARY_COLUMNS),
 'state_revisit_intervals':(*CONTEXT,'elapsed_ticks','count','retained_pairs','probability'),
 'agent_turnover':(*CONTEXT,*WINDOW,'active_agents','first_time_learners','first_time_founders','newly_appearing',
    'shared_with_previous','previous_agents_absent','absent_dead_by_window_end','absent_alive_at_window_end',
    'actual_deaths','previous_cohort_deaths','mean_age','median_age','p10_age','p90_age','mean_updates','median_updates',
    'p10_updates','p90_updates','low_experience_decision_share','high_experience_decision_share',
    'newly_allocated_decision_share','retained_row_decision_share'),
 'persistence_by_cohort':(*CONTEXT,*WINDOW,'cohort','agents',*SUMMARY_COLUMNS),
}


@dataclass(frozen=True)
class Options:
    window_size:int=100
    age_bin:int=25
    min_identity_pairs:int=20
    min_episode_encounters:int=6
    paired_block:int=3
    plot_visits:int=20
    plot_episodes:int=2
    low_updates:int=10
    high_updates:int=100

    def __post_init__(self):
        if any(type(v) is not int or v<1 for v in self.__dict__.values()):
            raise ValueError('Analysis options must be positive integers')
        if self.min_episode_encounters<2*self.paired_block or self.low_updates>=self.high_updates:
            raise ValueError('Early/late blocks must not overlap; experience thresholds must increase')


class Outputs:
    def __init__(self,directory,stack):
        self.writers={};self.counts=Counter();self.files={}
        for name,columns in SCHEMAS.items():
            suffix='.csv.gz' if name=='persistence_by_episode' else '.csv'
            path=directory/(name+suffix);self.files[name]=path
            if suffix.endswith('.gz'):
                raw=stack.enter_context(path.open('wb'))
                zipped=stack.enter_context(gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=1))
                stream=stack.enter_context(io.TextIOWrapper(zipped,encoding='utf-8',newline=''))
            else:
                stream=stack.enter_context(path.open('w',encoding='utf-8',newline=''))
            writer=csv.writer(stream);writer.writerow(columns);self.writers[name]=writer

    def add(self,name,row):
        def value(x):
            if isinstance(x,(float,np.floating)):
                return format(x,'.12g') if math.isfinite(x) else ''
            return '' if x is None else x
        self.writers[name].writerow(value(row.get(k)) for k in SCHEMAS[name])
        self.counts[name]+=1


@dataclass(slots=True)
class History:
    visits:int=0
    updates:int=0
    epoch:int=0
    last:object=None


@dataclass(slots=True)
class Episode:
    aid:int
    sid:int
    epoch:int
    allocated:int
    bag:Bag=field(default_factory=Bag)
    updates:int=0
    first:object=None
    last:object=None
    maximum_interval:int=0
    early:list=field(default_factory=list)
    late:deque=field(default_factory=deque)
    first_unique:object=None


@dataclass(slots=True)
class Agent:
    aid:int
    generation:int
    first_tick:int
    first_age:int
    last_tick:int=0
    last_age:int=0
    end_reason:str|None=None
    death_tick:int|None=None
    bag:Bag=field(default_factory=Bag)
    active:dict=field(default_factory=dict)
    histories:dict=field(default_factory=dict)
    updates:int=0
    allocations:int=0
    evictions:int=0
    last_decision:object=None


def hist_quantile(counts,q):
    """Exact linear quantile of integer observations without expanding them."""
    n=sum(counts.values())
    if not n:return None
    p=(n-1)*q;low=int(p);high=math.ceil(p);cum=0;a=b=None
    for x,count in sorted(counts.items()):
        if cum<=low<cum+count:a=x
        if cum<=high<cum+count:b=x;break
        cum+=count
    return a+(b-a)*(p-low)


class RunAnalysis:
    def __init__(self,run,options,output):
        self.run,self.options,self.output=run,options,output
        self.context={k:run[k] for k in CONTEXT}
        self.total=Bag();self.agents={};self.windows=defaultdict(Bag)
        self.age=defaultdict(Bag);self.visits=defaultdict(Bag);self.experience=defaultdict(Bag)
        self.kinds=defaultdict(Bag);self.agent_windows=defaultdict(Bag)
        self.active_windows=defaultdict(set);self.observed_ticks=defaultdict(set)
        self.age_hist=defaultdict(Counter);self.update_hist=defaultdict(Counter)
        self.row_presence=defaultdict(Counter);self.intervals=Counter()
        self.transitions=Counter();self.silence_transitions=Counter();self.development=defaultdict(lambda:[0,0.,0.])
        self.episode_count=self.eviction_count=0;self.trajectory_candidates=[]
        self.peak_active_rows=0;self.active_rows=0

    def emit(self,name,row):self.output.add(name,{**self.context,**row})

    def window(self,start):
        end=start+self.options.window_size;actual=min(end,self.run['metadata']['final_tick'])
        observed=len(self.observed_ticks[start])
        return dict(window_start=start,window_end=end,actual_window_end=actual,observed_ticks=observed,
                    partial_window=actual<end,complete_window=actual==end and observed==actual-start)

    def start(self,tick):return ((tick-1)//self.options.window_size)*self.options.window_size

    def row_event(self,event,agent):
        if event.event=='allocate':
            sid=integer(event.state_id,'allocation state');epoch=integer(event.epoch,'allocation epoch',1)
            history=agent.histories.setdefault(sid,History())
            if sid in agent.active or epoch!=history.epoch+1 or event.visits!=history.visits or event.updates!=history.updates:
                raise ValueError('Inconsistent actual row allocation/epoch/history')
            history.epoch=epoch;agent.allocations+=1;self.episode_count+=1
            agent.active[sid]=Episode(agent.aid,sid,epoch,int(event.tick))
            self.active_rows+=1;self.peak_active_rows=max(self.peak_active_rows,self.active_rows)
        elif event.event=='evict':
            sid=integer(event.state_id,'evicted state')
            e=agent.active.get(sid)
            if e is None or event.epoch!=e.epoch or event.visits!=e.bag.values[I['decisions']] or event.updates!=e.updates:
                raise ValueError('Eviction must match an actual allocated row and recorded counters')
            self.close(agent,sid,int(event.tick),'evict');agent.evictions+=1;self.eviction_count+=1
        else:
            raise ValueError('Unexpected pre-decision row event')

    def update(self,u,agent):
        sid=integer(u.state_id,'updated state');e=agent.active.get(sid)
        if e is None or u.epoch!=e.epoch:
            raise ValueError('Update references a missing/wrong allocation episode')
        history=agent.histories[sid]
        if u.visits!=history.visits or u.updates!=history.updates+1 or u.row_updates!=e.updates+1:
            raise ValueError('Update visit/count history inconsistent')
        if u.recreated!=int(e.epoch>1) or u.age!=agent.last_age:
            raise ValueError('Update epoch flag or age inconsistent')
        if u.td_error!=u.target-u.old_q or u.abs_td_error!=abs(u.td_error) or u.q_change!=abs(u.new_q-u.old_q):
            raise ValueError('TD diagnostic arithmetic inconsistent')
        e.updates+=1;history.updates+=1;agent.updates+=1

    def decision(self,d,agent,initial_present):
        sid=integer(d.state_id,'decision state');e=agent.active.get(sid)
        if e is None:raise ValueError('Encounter without actual allocated episode')
        h=agent.histories[sid];previous=h.last
        if d.state_visits!=h.visits+1 or d.updates!=agent.updates or d.allocations!=agent.allocations or d.evictions!=agent.evictions:
            raise ValueError('Decision lifetime counters inconsistent')
        expected='first' if not h.visits else 'stored' if initial_present else 'recreated'
        if d.novelty!=expected or bool(d.present_at_start)!=initial_present:
            raise ValueError('Recorded first/hit/recreated history inconsistent')
        if d.rows!=len(agent.active):raise ValueError('Decision Q occupancy inconsistent')
        kind='first' if previous is None else 'retained' if previous.epoch==e.epoch else 'recreated'
        if bool(d.newly_allocated)==bool(d.present_before_selection):
            raise ValueError('Selection presence/allocation flags inconsistent')
        ep_visit=int(e.bag.values[I['decisions']])+1
        now=encounter(d,e.epoch,ep_visit,e.updates)
        v=decision_vector(d,kind,now.preferred is not None)
        start=self.start(d.tick);age_start=(d.age//self.options.age_bin)*self.options.age_bin
        update_bin='low' if d.updates<self.options.low_updates else 'high' if d.updates>=self.options.high_updates else 'middle'
        bags=[self.total,self.windows[start],agent.bag,e.bag,self.age[age_start],
              self.visits[('episode',ep_visit)],self.visits[('lifetime',int(d.state_visits))],
              self.experience[update_bin],self.agent_windows[(start,agent.aid)],self.kinds[(start,kind)]]
        for bag in bags:bag.add(v)
        if previous is not None and kind=='retained':
            p=pair_vector(previous,now,previous.max_ties==1 and now.max_ties==1)
            for bag in bags:bag.add(p)
            delta=now.tick-previous.tick;self.intervals[delta]+=1;e.maximum_interval=max(e.maximum_interval,delta)
            self.silence_transitions[(start,previous.silence_category,now.silence_category)]+=1
            if previous.preferred is not None and now.preferred is not None:
                branch='both_exploit' if not previous.exploratory and not now.exploratory else 'any_explore'
                self.transitions[(start,branch,previous.preferred,now.preferred)]+=1
        if e.first is None:e.first=now
        e.last=now
        if now.max_ties==1 and e.first_unique is None:e.first_unique=now
        sample=(now.gap,now.p_silence)
        if len(e.early)<self.options.paired_block:e.early.append(sample)
        e.late.append(sample)
        if len(e.late)>self.options.paired_block:e.late.popleft()
        h.visits+=1;h.last=now
        agent.last_decision=d
        self.active_windows[start].add(agent.aid);self.observed_ticks[start].add(int(d.tick))
        self.age_hist[start][int(d.age)]+=1;self.update_hist[start][int(d.updates)]+=1
        self.row_presence[start]['new']+=int(d.newly_allocated)
        self.row_presence[start]['retained']+=int(d.present_before_selection)

    def close(self,agent,sid,tick,reason):
        e=agent.active.pop(sid);self.active_rows-=1;s=e.bag.summary();n=s['decisions']
        eligible=n>=self.options.min_episode_encounters
        early_gap=late_gap=early_p=late_p=None
        if eligible:
            early_gap=sum(x[0] for x in e.early)/len(e.early);late_gap=sum(x[0] for x in e.late)/len(e.late)
            early_p=sum(x[1] for x in e.early)/len(e.early);late_p=sum(x[1] for x in e.late)/len(e.late)
            age_start=(e.first.age//self.options.age_bin)*self.options.age_bin
            dev=self.development[age_start];dev[0]+=1;dev[1]+=late_gap-early_gap;dev[2]+=late_p-early_p
        row={'agent_id':agent.aid,'state_id':sid,'episode':e.epoch,'allocated_tick':e.allocated,'end_tick':tick,
             'end_reason':reason,'storage_duration':tick-e.allocated,'first_encounter_tick':e.first.tick if n else None,
             'last_encounter_tick':e.last.tick if n else None,'encounter_span':e.last.tick-e.first.tick if n else None,
             'first_age':e.first.age if n else None,'encounters':n,'row_updates':e.updates,'retained_pairs':s['pairs'],
             'mean_interval':s['mean_revisit_interval'],'max_interval':e.maximum_interval if n>1 else None,
             **{k:s[k] for k in ('unique_max_fraction','mean_gap','mean_p_silence','observed_silence_rate',
                'choice_match_rate','id_pairs','id_same','identity_coverage','preferred_action_match_rate',
                'set_pairs','mean_max_set_jaccard','mean_exact_policy_tv')},'eligible_early_late':int(eligible),
             'early_encounters':len(e.early) if eligible else None,'late_encounters':len(e.late) if eligible else None,
             'early_mean_gap':early_gap,'late_mean_gap':late_gap,'delta_gap':late_gap-early_gap if eligible else None,
             'early_mean_p_silence':early_p,'late_mean_p_silence':late_p,'delta_p_silence':late_p-early_p if eligible else None,
             'first_observed_unique_tick':e.first_unique.tick if e.first_unique else None,
             'first_observed_unique_visit':e.first_unique.episode_visit if e.first_unique else None}
        self.emit('persistence_by_episode',row)
        if eligible:
            self.trajectory_candidates.append(row)
            self.trajectory_candidates.sort(key=lambda r:(-r['encounters'],r['agent_id'],r['state_id'],r['episode']))
            del self.trajectory_candidates[self.options.plot_episodes:]

    def ending(self,event,agent):
        if agent.end_reason is not None or event.visits!=agent.bag.values[I['decisions']] or event.updates!=agent.updates or event.rows!=len(agent.active):
            raise ValueError('Death/run-end lifecycle or counters inconsistent')
        if event.event not in ('death','run_end'):raise ValueError('Unknown episode termination')
        for sid in list(agent.active):self.close(agent,sid,int(event.tick),event.event)
        agent.end_reason=event.event
        agent.death_tick=int(event.tick) if event.event=='death' else None
        agent.histories.clear();agent.last_decision=None

    def process(self):
        rows=TickCursor(tick_records(self.run,'rows'));updates=TickCursor(tick_records(self.run,'updates'))
        last_tick=0
        for tick,decisions in tick_records(self.run,'decisions'):
            last_tick=tick
            events=defaultdict(list);feedback=defaultdict(list);end=[]
            for event in rows.pop(tick):
                if event.event=='run_end':end.append(event)
                else:events[int(event.agent_id)].append(event)
            for u in updates.pop(tick):feedback[int(u.agent_id)].append(u)
            for d in decisions:
                aid=int(d.agent_id)
                if aid not in self.agents:
                    self.agents[aid]=Agent(aid,int(d.generation),tick,int(d.age))
                a=self.agents[aid]
                if a.end_reason is not None:raise ValueError('Agent acts after actual death/end')
                if a.last_tick and (tick!=a.last_tick+1 or d.age!=a.last_age+1):
                    raise ValueError('Missing actor encounters or inconsistent age; incomplete recordings rejected')
                if d.generation!=a.generation:raise ValueError('Agent generation changed')
                previous=a.last_decision;a.last_tick=tick;a.last_age=int(d.age)
                initial_present=int(d.state_id) in a.active
                alloc_before,evict_before=a.allocations,a.evictions
                es=deque(events.pop(aid,[]));us=feedback.pop(aid,[])
                before=[u for u in us if not u.terminal];terminal=[u for u in us if u.terminal]
                if len(before)!=int(previous is not None) or len(terminal)>1:
                    raise ValueError('Missing/extra transition update in completed recording')
                for u in before:
                    if u.state_id!=previous.state_id or u.action!=previous.selected_action:
                        raise ValueError('Previous transition identity inconsistent')
                    while (int(u.state_id) not in a.active or a.active[int(u.state_id)].epoch!=u.epoch):
                        if not es or es[0].event not in ('allocate','evict'):
                            raise ValueError('Missing actual allocation before update')
                        self.row_event(es.popleft(),a)
                    self.update(u,a)
                selection_allocated=False
                while es and es[0].event in ('allocate','evict'):
                    event=es.popleft()
                    if event.event=='allocate' and event.state_id==d.state_id:selection_allocated=True
                    self.row_event(event,a)
                if d.newly_allocated!=int(selection_allocated) or d.decision_allocations!=a.allocations-alloc_before or d.decision_evictions!=a.evictions-evict_before:
                    raise ValueError('Actual decision allocations/evictions disagree with diagnostics')
                self.decision(d,a,initial_present)
                for u in terminal:
                    if u.state_id!=d.state_id or u.action!=d.selected_action:raise ValueError('Terminal transition identity inconsistent')
                    self.update(u,a)
                if es:
                    if len(es)!=1 or es[0].event!='death' or len(terminal)!=1:raise ValueError('Invalid death boundary')
                    self.ending(es.popleft(),a)
                elif terminal:raise ValueError('Terminal update without actual death')
            if events or feedback:raise ValueError('Events reference an unrecorded actor')
            for event in end:
                if tick!=self.run['metadata']['final_tick'] or int(event.agent_id) not in self.agents:
                    raise ValueError('Invalid final run-end event')
                self.ending(event,self.agents[int(event.agent_id)])
        rows.exhausted();updates.exhausted()
        if last_tick!=self.run['metadata']['final_tick'] or self.active_rows or any(a.end_reason is None for a in self.agents.values()):
            raise ValueError('Incomplete encounters or unclosed allocation episodes')
        meta=self.run['metadata'];s=self.total.summary()
        if s['decisions']!=meta['files']['decisions']['records'] or sum(a.updates for a in self.agents.values())!=meta['files']['updates']['records']:
            raise ValueError('Encounter/update exposure does not reconcile')
        if sum(a.end_reason=='death' for a in self.agents.values())!=meta['statistics']['deaths']:
            raise ValueError('Actual deaths differ from final metadata')
        self.write_summaries()
        return self

    def write_summaries(self):
        cfg=self.run['metadata']['config'];previous=set()
        for start in sorted(self.windows):
            context=self.window(start);bag=self.windows[start];summary=bag.summary()
            self.emit('persistence_by_window',{**context,**summary})
            active=self.active_windows[start];shared=active&previous;new=active-previous;absent=previous-active
            first={aid for aid in active if self.start(self.agents[aid].first_tick)==start}
            deaths={aid for aid,a in self.agents.items() if a.death_tick is not None and self.start(a.death_tick)==start}
            end=context['actual_window_end']
            absent_dead={aid for aid in absent if self.agents[aid].death_tick is not None and self.agents[aid].death_tick<=end}
            ah,uh=self.age_hist[start],self.update_hist[start];n=summary['decisions']
            turnover={**context,'active_agents':len(active),'first_time_learners':len(first),
                'first_time_founders':sum(self.agents[aid].generation==0 for aid in first),'newly_appearing':len(new),
                'shared_with_previous':len(shared),'previous_agents_absent':len(absent),'absent_dead_by_window_end':len(absent_dead),
                'absent_alive_at_window_end':len(absent-absent_dead),'actual_deaths':len(deaths),'previous_cohort_deaths':len(deaths&previous),
                'mean_age':sum(k*v for k,v in ah.items())/n,'mean_updates':sum(k*v for k,v in uh.items())/n,
                'low_experience_decision_share':sum(v for k,v in uh.items() if k<self.options.low_updates)/n,
                'high_experience_decision_share':sum(v for k,v in uh.items() if k>=self.options.high_updates)/n,
                'newly_allocated_decision_share':self.row_presence[start]['new']/n,
                'retained_row_decision_share':self.row_presence[start]['retained']/n}
            for name,hist in [('age',ah),('updates',uh)]:
                for tag,q in [('median',.5),('p10',.1),('p90',.9)]:turnover[tag+'_'+name]=hist_quantile(hist,q)
            self.emit('agent_turnover',turnover)
            for cohort,ids in [('all',active),('shared_with_previous',shared),('newly_appearing',new)]:
                b=Bag()
                for aid in sorted(ids):b.values+=self.agent_windows[(start,aid)].values
                self.emit('persistence_by_cohort',{**context,'cohort':cohort,'agents':len(ids),**b.summary()})
            previous=active
        for aid,a in sorted(self.agents.items()):
            self.emit('persistence_by_agent',{'agent_id':aid,'generation':a.generation,'first_tick':a.first_tick,'last_tick':a.last_tick,
                'first_age':a.first_age,'last_age':a.last_age,'death_tick':a.death_tick,'end_reason':a.end_reason,
                'episodes':a.allocations,'actual_evictions':a.evictions,**a.bag.summary()})
        for (start,aid),b in sorted(self.agent_windows.items()):
            self.emit('persistence_by_agent_window',{**self.window(start),'agent_id':aid,**b.summary()})
        for (kind,index),b in sorted(self.visits.items()):self.emit('persistence_by_visit_index',{'visit_kind':kind,'visit_index':index,**b.summary()})
        for age,b in sorted(self.age.items()):self.emit('persistence_by_age',{'age_start':age,**b.summary()})
        for kind in ('low','middle','high'):
            self.emit('persistence_by_experience',{'update_bin':kind,**self.experience.get(kind,Bag()).summary()})
        for start in sorted(self.windows):
            for kind in ('first','retained','recreated'):
                self.emit('retained_vs_recreated',{**self.window(start),'encounter_class':kind,
                                                  **self.kinds.get((start,kind),Bag()).summary()})
        for delta,count in sorted(self.intervals.items()):self.emit('state_revisit_intervals',{'elapsed_ticks':delta,'count':count,
                'retained_pairs':self.total.summary()['pairs'],'probability':count/self.total.values[I['pairs']]})
        branch_counts=Counter()
        for (start,branch,a,b),count in self.transitions.items():branch_counts[(start,branch)]+=count
        for (start,branch,a,b),count in sorted(self.transitions.items()):
            self.emit('preference_transitions',{**self.window(start),'branch':branch,'from_action':a,'to_action':b,'count':count,
                'branch_pairs':branch_counts[(start,branch)],'window_id_pairs':int(self.windows[start].values[I['id_pairs']]),
                'probability':count/branch_counts[(start,branch)]})
        origin=Counter()
        for (start,a,b),count in self.silence_transitions.items():origin[(start,a)]+=count
        # Known categorical support: zero outcomes with a positive origin exposure
        # are real zeros. Omitting them would bias equal-seed means upward.
        for start in sorted(self.windows):
            for a in SILENCE_CATEGORIES:
                denominator=origin[(start,a)]
                for b in SILENCE_CATEGORIES:
                    count=self.silence_transitions[(start,a,b)]
                    self.emit('silence_preference_transitions',{
                        **self.window(start),'from_category':a,'to_category':b,'count':count,'origin_pairs':denominator,
                        'probability':count/denominator if denominator else None,
                        'retained_pairs':int(self.windows[start].values[I['pairs']])})
        for age,(n,g,p) in sorted(self.development.items()):self.emit('paired_episode_development',{
            'first_age_start':age,'episodes':n,'mean_gap_change':g/n,'mean_p_silence_change':p/n})
        nd=sum(v[0] for v in self.development.values())
        self.emit('persistence_run_summary',{**self.total.summary(),'agents':len(self.agents),'episodes':self.episode_count,
            'actual_evictions':self.eviction_count,'deaths':sum(a.end_reason=='death' for a in self.agents.values()),
            'run_ends':sum(a.end_reason=='run_end' for a in self.agents.values()),'eligible_paired_episodes':nd,
            'mean_paired_gap_change':sum(v[1] for v in self.development.values())/nd if nd else None,
            'mean_paired_p_silence_change':sum(v[2] for v in self.development.values())/nd if nd else None})
