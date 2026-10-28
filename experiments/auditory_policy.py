"""Exact read-only Physical Q policies under explicit primitive auditory inputs.

No simulation actions, learning, Q copies beyond one row, or RNG calls. The
optional memory tuple is materialized only in an isolated mechanical memory.
"""
from dataclasses import dataclass,asdict,fields,is_dataclass,replace
from collections import OrderedDict,deque
from random import Random
import hashlib
import json
import math

from agents.auditory_memory import AuditoryEvent,AuditoryMemory
from agents.learning import LearningBrain,Physiology
from agents.observation import (AuditoryKind,AuditoryPercept,HeardSignal,Observation,
                                SourceDirection,SILENCE,MASKED)


@dataclass(frozen=True,slots=True)
class Policy:
    state:tuple
    present:bool
    actions:tuple
    values:tuple[float,...]
    probabilities:tuple[float,...]
    maxima:tuple[int,...]
    epsilon:float
    uniform_reason:str

    @property
    def unique_action(self):
        return self.actions[self.maxima[0]].value if len(self.maxima)==1 else None


def categorical(probabilities):
    p=tuple(float(x) for x in probabilities)
    if not p or not all(math.isfinite(x) and x>=0 for x in p) or not math.isclose(sum(p),1.,rel_tol=0,abs_tol=1e-12):
        raise ValueError('Expected finite nonnegative categorical probabilities summing to one')
    return p


def epsilon_policy(values,epsilon):
    q=tuple(values)
    if not q or not all(math.isfinite(v) for v in q) or not math.isfinite(epsilon) or not 0<=epsilon<=1:
        raise ValueError('Invalid Q values or epsilon')
    best=max(q);maxima=tuple(i for i,v in enumerate(q) if v==best)
    p=tuple(epsilon/len(q)+(1-epsilon)*int(v==best)/len(maxima) for v in q)
    return categorical(p),maxima


def validate_percept(percept,vocabulary):
    if not isinstance(percept,AuditoryPercept):raise ValueError('Expected immutable primitive AuditoryPercept')
    if percept.kind is AuditoryKind.IDENTIFIED and not 0<=percept.signal.signal_id<vocabulary:
        raise ValueError('Auditory signal ID outside this brain vocabulary')


def memory_before(brain,tick):
    """Pre-current-event baseline; prevents recalling an already-inserted current cue."""
    return tuple(e for e in brain.auditory_memory.events(tick) if e.received_tick<tick)


def evaluate(brain:LearningBrain,body:Physiology,observation:Observation,memory_events=None):
    if not isinstance(brain,LearningBrain) or not isinstance(body,Physiology) or not isinstance(observation,Observation):
        raise ValueError('Evaluation needs a LearningBrain, frozen Physiology and immutable Observation')
    if type(observation.tick) is not int or observation.tick<0:raise ValueError('Invalid evaluation tick')
    if not all(math.isfinite(v) and v>=0 for v in (body.hunger,body.thirst,body.energy)):
        raise ValueError('Invalid physiological state')
    validate_percept(observation.auditory,brain.vocab_size)
    if memory_events is None:
        # Exactly the existing nonmutating encoder, including memory as held.
        state=brain.state(body,observation)
    else:
        if not isinstance(memory_events,tuple) or len(memory_events)>brain.auditory_memory.horizon:
            raise ValueError('Explicit memory must be an immutable bounded tuple')
        memory=AuditoryMemory(brain.auditory_memory.horizon);last=-1
        for event in memory_events:
            if (not isinstance(event,AuditoryEvent) or type(event.received_tick) is not int
                or not last<event.received_tick<=observation.tick or event.percept.kind is AuditoryKind.SILENCE):
                raise ValueError('Memory events require increasing nonnegative past ticks and nonsilent percepts')
            validate_percept(event.percept,brain.vocab_size)
            memory.update(event.percept,event.received_tick);last=event.received_tick
        original=brain.primitive_state(body,observation)
        state=(*original,memory.feature(observation.auditory,observation.tick)) if brain.learning_uses_auditory else original
    present,values=brain.inspect_physical_q(state)
    if len(values)!=len(brain.actions):raise ValueError('Q row/action-space size mismatch')
    probabilities,maxima=epsilon_policy(values,brain.epsilon)
    reason=('absent_zero_prior' if not present else 'stored_zero' if all(v==0 for v in values)
            else 'stored_tied' if len(maxima)==len(values) else 'exploration_only' if brain.epsilon==1 else 'nonuniform')
    return Policy(state,present,brain.actions,values,probabilities,maxima,brain.epsilon,reason)


def distances(a,b):
    p,q=categorical(a),categorical(b)
    if len(p)!=len(q):raise ValueError('Action distributions must have matching categories')
    tv=.5*sum(abs(x-y) for x,y in zip(p,q))
    js=0.
    for x,y in zip(p,q):
        m=(x+y)/2
        if x:js+=.5*x*math.log2(x/m)
        if y:js+=.5*y*math.log2(y/m)
    return tv,max(0.,js)


def compare(a:Policy,b:Policy):
    if a.actions!=b.actions or a.epsilon!=b.epsilon or a.state[:7]!=b.state[:7]:
        raise ValueError('Counterfactual probes must hold body/visual/action/epsilon features fixed')
    tv,js=distances(a.probabilities,b.probabilities)
    pattern='both_present' if a.present and b.present else 'original_only' if a.present else 'counterfactual_only' if b.present else 'neither_present'
    return {'tv':tv,'jsd_bits':js,'max_set_changed':a.maxima!=b.maxima,
            'unique_greedy_changed':a.unique_action!=b.unique_action if a.unique_action is not None and b.unique_action is not None else None,
            'original_unique_action':a.unique_action,'counterfactual_unique_action':b.unique_action,
            'original_present':a.present,'counterfactual_present':b.present,'row_pattern':pattern,
            'coverage':'one_present' if pattern in ('original_only','counterfactual_only') else pattern,
            'original_uniform':max(a.probabilities)==min(a.probabilities),
            'counterfactual_uniform':max(b.probabilities)==min(b.probabilities),
            'both_uniform':max(a.probabilities)==min(a.probabilities) and max(b.probabilities)==min(b.probabilities),
            'original_uniform_reason':a.uniform_reason,'counterfactual_uniform_reason':b.uniform_reason,
            'delta_probabilities':tuple(y-x for x,y in zip(a.probabilities,b.probabilities))}


def identified(signal,direction=SourceDirection.E):
    return AuditoryPercept(AuditoryKind.IDENTIFIED,HeardSignal(signal,direction))


def interventions(tick,cues=(7,12),vocabulary=16):
    """Fixed bounded alternatives. No semantic interpretation or optimized probe search."""
    if tick<2 or len(set(cues))!=2 or any(type(c) is not int or not 0<=c<vocabulary for c in cues):
        raise ValueError('Two valid distinct cues and evaluation tick >=2 required')
    a,b=cues;empty=();past_a=(AuditoryEvent(tick-1,identified(a)),);past_b=(AuditoryEvent(tick-1,identified(b)),)
    unseen=next((i for i in range(vocabulary) if i not in cues),None)
    result=[('remove_current_empty','current_removal',identified(a),empty,SILENCE,empty),
            ('remove_current_keep_past','current_removal',identified(a),past_b,SILENCE,past_b),
            ('substitute_id_a_b','signal_identity',identified(a),empty,identified(b),empty),
            ('substitute_id_b_a','signal_identity',identified(b),empty,identified(a),empty),
            ('substitute_direction','direction',identified(a),empty,identified(a,SourceDirection.W),empty),
            ('substitute_memory_id','memory_content',SILENCE,past_a,SILENCE,past_b),
            ('empty_encoded_auditory','empty_auditory_baseline',SILENCE,past_a,SILENCE,empty),
            ('silence_masked','current_kind',SILENCE,empty,MASKED,empty)]
    if unseen is not None:result.append(('substitute_untrained_id','signal_identity_unseen',identified(a),empty,identified(unseen),empty))
    return result


def frozen(value):
    """Verification only, outside timed probes; preserve all table/dictionary order."""
    if isinstance(value,Random):return ('Random',frozen(value.getstate()))
    if isinstance(value,LearningBrain):return ('LearningBrain',frozen(vars(value)))
    if is_dataclass(value):return tuple((f.name,frozen(getattr(value,f.name))) for f in fields(value))
    if isinstance(value,dict):return tuple((frozen(k),frozen(v)) for k,v in value.items())
    if isinstance(value,(tuple,list,deque)):return tuple(frozen(x) for x in value)
    if isinstance(value,set):return tuple(sorted(frozen(x) for x in value))
    if isinstance(value,AuditoryMemory):return (value.horizon,frozen(value._events),value._last_tick)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'Unsupported verification state: {type(value)}')


def simulation_fingerprint(sim):
    # Read-only batch-boundary verification, never an every-decision observer.
    bodies=tuple((tuple((f.name,frozen(getattr(h,f.name))) for f in fields(h) if f.name!='brain'),
                  frozen(vars(h.brain))) for h in sim.agents)
    state=(sim.tick,sim._next_id,frozen(sim.config),frozen(sim.statistics),bodies,
           tuple((tile.food,tile.water) for row in sim.world.tiles for tile in row),
           frozen(vars(sim.communication)),frozen(vars(sim.reproduction)),
           tuple(frozen(getattr(sim,name)) for name in ('_order_rng','_offspring_rng','_inheritance_rng','_mutation_rng','_food_rng','_water_rng')))
    return hashlib.sha256(json.dumps(state,separators=(',',':')).encode()).hexdigest()
