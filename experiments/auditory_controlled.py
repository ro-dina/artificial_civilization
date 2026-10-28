"""Predefined small receiver mechanism experiment, using unmodified Q/physics.

Experimenter-supplied cue/context associations are external interventions, not
signal meanings in the learner. Training resets are explicit; pending learning
transitions persist across resets, as in the earlier controlled demonstrations.
"""
from collections import Counter
from dataclasses import dataclass,asdict,replace
import hashlib
import json
from random import Random
import random

from agents.brain import Action
from agents.learning import Physiology
from agents.observation import Observation,TileObservation,SourceDirection
from config import Config
from simulation.simulation import Simulation
from experiments.auditory_policy import (evaluate,identified,interventions,compare,distances,simulation_fingerprint)

CONDITIONS=('predictive_on','predictive_off','independent_on','untrained_on')
CUES=(7,12)


@dataclass(frozen=True)
class Design:
    trials:int=256
    steps_per_trial:int=8
    seeds:tuple=(0,1,2)
    checkpoint_trials:int=32
    correct_probability:float=.5

    def __post_init__(self):
        if type(self.trials) is not int or not 4<=self.trials<=512 or self.trials%4:
            raise ValueError('Controlled trials must be 4..512, divisible by 4')
        if type(self.steps_per_trial) is not int or not 5<=self.steps_per_trial<=8:
            raise ValueError('Use 5..8 steps: previous cue expires before next trial; resets prevent dehydration')
        if not self.seeds or any(type(s) is not int for s in self.seeds) or len(set(self.seeds))!=len(self.seeds):
            raise ValueError('Use distinct integer seeds')
        if type(self.checkpoint_trials) is not int or self.checkpoint_trials<1 or not 0<self.correct_probability<=1:
            raise ValueError('Invalid checkpoint or success threshold')


def config_for(design,seed,condition):
    if condition not in CONDITIONS:raise ValueError('Unknown controlled condition')
    return Config(brain='learning',learning_uses_auditory=condition!='predictive_off',random_seed=seed,
        max_ticks=design.trials*design.steps_per_trial,world_width=3,world_height=1,initial_population=1,
        initial_food=0,initial_water=0,perception_radius=0,hunger_per_tick=0,thirst_per_tick=1,max_age=None)


def schedules(design,seed):
    context_rng=Random(f'v064:{seed}:contexts');cue_rng=Random(f'v064:{seed}:independent-cues')
    contexts=[];independent=[]
    for _ in range(design.trials//4):
        block=[0,0,1,1];context_rng.shuffle(block);cues=[None]*4
        for context in (0,1):
            positions=[i for i,value in enumerate(block) if value==context]
            ids=list(CUES);cue_rng.shuffle(ids)
            for position,cue in zip(positions,ids):cues[position]=cue
        contexts.extend(block);independent.extend(cues)
    predictive=[CUES[c] for c in contexts]
    return tuple(contexts),tuple(predictive),tuple(independent)


def definition(design):
    return {'design':asdict(design),'conditions':CONDITIONS,'cues':CUES,
        'configuration_by_condition':{c:asdict(config_for(design,design.seeds[0],c)) for c in CONDITIONS},
        'external_association':'Context 0 has one water unit at x=2 and cue 7; context 1 at x=0 and cue 12. Receiver resets to x=1, hunger=0, thirst=60, energy=100; radius 0. Both cues originate at x=2, direction E.',
        'independent_control':'Same context schedule. Every randomized four-trial block contains both contexts twice; within each context cues 7/12 occur once each. Independent named schedule RNG; no trial/order/tick input to Q, prior auditory cue expired before next trial.',
        'training':'Ordinary Simulation.step and physiological reward only. No manual Q/action edits, artificial terminal flags or direct update calls. Pending transitions and tables persist through experimental resets.',
        'evaluation':'Same frozen brain/body/visual/tick per pair, no actions or learning; explicit immutable pre-current memory. Missing rows retain zero prior without allocating.',
        'success':'Per cue: expected move (7 east, 12 west) is unique greedy and probability >=0.5; bidirectional requires both present and updated. Report partial/failed. Aggregate support requires predictive bidirectional and better correct-action probability than independent/off/untrained in every seed.',
        'probes':'Two body/visual contexts: thirst60/hunger0/energy100 with current tile empty or water1. Fixed nine interventions; ID substitutions 7<->12 plus lowest nontraining ID, E->W; retained cue age1; explicit empty baseline.',
        'ecological_benefit':'Not tested: no matched frozen ecological rollouts. Training water/first actions are descriptive training outcomes, not Level 4.',
        'interpretation':'Small imposed-cue receiver learning test, not spontaneous signals, speaker adaptation, shared meaning or language.'}


def _j(value):return json.dumps(value,separators=(',',':'))


def root_evaluation(sim,tick):
    h=sim.agents[0];body=Physiology(0,60,100,True);tiles=(TileObservation(0,0,0,0),)
    result=[]
    for cue,action in zip(CUES,(Action.MOVE_EAST,Action.MOVE_WEST)):
        policy=evaluate(h.brain,body,Observation(tick,tiles,identified(cue)),())
        result.append((policy,policy.probabilities[policy.actions.index(action)],action.value))
    return result


def train(design,seed,condition,checkpoint_sink=None):
    cfg=config_for(design,seed,condition);sim=Simulation(cfg);h=sim.agents[0]
    contexts,predictive,independent=schedules(design,seed)
    cues=independent if condition=='independent_on' else predictive
    exposure=Counter();updates=Counter();curves=[];max_memory=0
    block=Counter();checkpoint_hashes=[]

    def checkpoint(trials):
        before=simulation_fingerprint(sim);global_before=random.getstate()
        roots=root_evaluation(sim,max(sim.tick+1,2))
        after=simulation_fingerprint(sim)
        if before!=after or global_before!=random.getstate():raise RuntimeError('Frozen checkpoint altered training state/RNG')
        checkpoint_hashes.append({'trials':trials,'before':before,'after':after})
        n=block['trials']
        row={'experiment':'controlled','evidence':'training_and_frozen_policy','condition':condition,'seed':seed,
            'trials':trials,'ticks':sim.tick,'physical_updates':h.brain.learning_updates,
            'mean_correct_action_probability':sum(x[1] for x in roots)/2,
            'training_first_action_correct':block['correct']/n if n else None,
            'training_water_per_trial':block['water']/n if n else None,
            'training_reward_per_action':block['reward']/block['actions'] if block['actions'] else None,
            'block_trials':n,'q_rows':h.brain.memory_size}
        curves.append(row)
        if checkpoint_sink:checkpoint_sink(row)
        block.clear()

    checkpoint(0)
    if condition!='untrained_on':
        for trial,(context,cue) in enumerate(zip(contexts,cues),1):
            h.x,h.y,h.hunger,h.thirst,h.energy=1,0,0,60,100
            if h.brain.auditory_memory.events(sim.tick+1):raise RuntimeError('Previous-trial auditory memory leaked into root state')
            for tile in sim.world.tiles[0]:tile.water=0
            sim.world.tile_at(2 if context==0 else 0,0).water=1
            sim.communication.emit(cue,99,2,0);sim.communication.advance()
            initial_water=sim.statistics.water_consumed
            for step in range(design.steps_per_trial):
                previous=h.brain._pending.state if h.brain._pending is not None else None
                before_updates=h.brain.learning_updates
                sim.step()
                if not h.alive:raise RuntimeError('Predefined task should not terminate by mortality; report protocol failure')
                state=h.brain.decision_state
                exposure[state]+=1
                if previous is not None:updates[previous]+=h.brain.learning_updates-before_updates
                max_memory=max(max_memory,h.brain.memory_size)
                if step==0:
                    correct=Action.MOVE_EAST if context==0 else Action.MOVE_WEST
                    block['correct']+=h.brain.actions[h.brain._pending.action_index] is correct
                block['actions']+=1;block['reward']+=h.brain.last_reward
            block['trials']+=1;block['water']+=sim.statistics.water_consumed-initial_water
            if trial%design.checkpoint_trials==0 or trial==design.trials:checkpoint(trial)
    if max_memory>=cfg.learning_memory_capacity:
        raise RuntimeError('Design unexpectedly reached capacity; lifetime update coverage would not prove retained-row learning')
    return sim,exposure,updates,curves,checkpoint_hashes


def evaluate_run(sim,exposure,updates,design,seed,condition):
    before=simulation_fingerprint(sim);global_before=random.getstate()
    brain=sim.agents[0].brain;tick=design.trials*design.steps_per_trial+1
    body=Physiology(0,60,100,True);probes=[]
    for name,water in [('root_empty',0),('visible_water',1)]:
        tiles=(TileObservation(0,0,0,water),)
        for probe,intervention,percept_a,memory_a,percept_b,memory_b in interventions(tick,CUES,brain.vocab_size):
            a=evaluate(brain,body,Observation(tick,tiles,percept_a),memory_a)
            b=evaluate(brain,body,Observation(tick,tiles,percept_b),memory_b)
            distance=compare(a,b)
            probes.append({'experiment':'controlled','evidence':'frozen_policy_intervention','condition':condition,'seed':seed,
                'probe':name+':'+probe,'physical_context':name,'intervention':intervention,'evaluation_tick':tick,
                'original_current':_j(asdict(percept_a)),'counterfactual_current':_j(asdict(percept_b)),
                'original_memory':_j([asdict(x) for x in memory_a]),'counterfactual_memory':_j([asdict(x) for x in memory_b]),
                'original_state':_j(a.state),'counterfactual_state':_j(b.state),'original_probabilities':_j(a.probabilities),
                'counterfactual_probabilities':_j(b.probabilities),'delta_probabilities':_j(distance.pop('delta_probabilities')),
                'actions':_j([x.value for x in a.actions]),'original_maxima':_j(a.maxima),'counterfactual_maxima':_j(b.maxima),
                'original_training_visits':exposure[a.state],'counterfactual_training_visits':exposure[b.state],
                'original_training_updates':updates[a.state],'counterfactual_training_updates':updates[b.state],
                'encoded_state_changed':a.state!=b.state,
                'both_updated_present':a.present and b.present and updates[a.state]>0 and updates[b.state]>0,
                'two_distinct_updated_states':a.state!=b.state and a.present and b.present and updates[a.state]>0 and updates[b.state]>0,**distance})
    roots=root_evaluation(sim,tick);correct=[]
    for policy,probability,expected in roots:
        correct.append(policy.present and updates[policy.state]>0 and policy.unique_action==expected and probability>=design.correct_probability)
    cue_tv,cue_js=distances(roots[0][0].probabilities,roots[1][0].probabilities)
    result={'experiment':'controlled','evidence':'frozen_policy','condition':condition,'seed':seed,
        'training_trials':0 if condition=='untrained_on' else design.trials,'training_ticks':sim.tick,
        'cue_7_expected_action':'move_east','cue_12_expected_action':'move_west',
        'cue_7_unique_action':roots[0][0].unique_action,'cue_12_unique_action':roots[1][0].unique_action,
        'cue_7_correct_probability':roots[0][1],'cue_12_correct_probability':roots[1][1],
        'mean_correct_probability':(roots[0][1]+roots[1][1])/2,'root_identity_tv':cue_tv,'root_identity_jsd_bits':cue_js,
        'both_root_states_present':all(x[0].present for x in roots),
        'distinct_root_states':roots[0][0].state!=roots[1][0].state,
        'both_root_states_updated':all(updates[x[0].state]>0 for x in roots),'correct_cues':sum(correct),
        'learning_result':'bidirectional' if all(correct) else 'partial' if any(correct) else 'failed_or_untrained',
        'training_water_consumed':sim.statistics.water_consumed,'physical_updates':brain.learning_updates,
        'physical_q_rows':brain.memory_size,'alive':sim.agents[0].alive}
    after=simulation_fingerprint(sim)
    if before!=after or global_before!=random.getstate():raise RuntimeError('Counterfactual evaluation altered trained state or RNG')
    return probes,result,{'seed':seed,'condition':condition,'before':before,'after':after,'global_rng_unchanged':True,
                           'full_config':asdict(sim.config)}
