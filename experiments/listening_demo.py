"""Repeated arbitrary sound/context association using ordinary delayed Q rewards.

The harness supplies cues and relocates water/receiver between short trials.
Both cues originate from E; neither tells the brain a target direction or meaning.
No Q values, policy choices or rewards are edited. A matched auditory ablation
uses the same trial schedule and physics, with the original state encoder.
"""

from collections import Counter
from dataclasses import asdict
import json

from agents.learning import Physiology
from agents.observation import AuditoryKind, AuditoryPercept, HeardSignal, Observation, SourceDirection
from config import Config
from simulation.simulation import Simulation

CUES = (7, 12)


def run_condition(uses_auditory: bool, trials: int, ticks_per_trial: int) -> dict:
    cfg = Config(brain="learning", learning_uses_auditory=uses_auditory, random_seed=0,
                 world_width=5, world_height=1, initial_population=1, initial_food=0,
                 initial_water=0, perception_radius=0, signal_probability=0,
                 hunger_per_tick=0, thirst_per_tick=1, max_age=None)
    sim = Simulation(cfg)
    human = sim.agents[0]
    human.x, human.y, human.thirst = 2, 0, 60
    body = Physiology.capture(human)
    tiles = sim.observe(human).tiles

    def probe_values():
        return {str(cue): dict(zip((a.value for a in human.brain.actions),
                    human.brain.action_values(body, Observation(sim.tick + 1, tiles,
                        AuditoryPercept(AuditoryKind.IDENTIFIED, HeardSignal(cue, SourceDirection.E))))))
                for cue in CUES}

    initial = probe_values()
    first_actions = {cue: Counter() for cue in CUES}
    last_block_actions = {cue: Counter() for cue in CUES}
    consumed = Counter()
    # The alternating conditions are experimental context interventions, not a
    # signal dictionary inside the simulation. The Brain receives neither this
    # counter nor the water position; both conditions look identical at the start.
    for trial in range(trials):
        cue = CUES[trial % len(CUES)]
        target_x = 4 if trial % len(CUES) == 0 else 0
        human.x, human.y, human.thirst = 2, 0, 60
        for tile in sim.world.tiles[0]:
            tile.water = 0
        sim.world.tile_at(target_x, 0).water = 1
        # An external test emitter: inject one pending cue then prepare the next
        # acting phase exactly as the ordinary end-of-tick swap would do.
        sim.communication.emit(cue, sender_id=99, x=3, y=0)
        sim.communication.advance()
        before = sim.statistics.water_consumed
        for step in range(ticks_per_trial):
            sim.step()
            if step == 0:
                action = human.brain.actions[human.brain._pending.action_index].value
                first_actions[cue][action] += 1
                if trial >= trials * 3 // 4:
                    last_block_actions[cue][action] += 1
        consumed[cue] += sim.statistics.water_consumed - before
    learned = probe_values()
    greedy = {cue: [action for action, value in values.items() if value == max(values.values())]
              for cue, values in learned.items()}
    return {"learning_uses_auditory": uses_auditory, "config": asdict(cfg),
            "initial_values": initial, "learned_values": learned, "learned_greedy_actions": greedy,
            "first_actions": {str(k): dict(v) for k, v in first_actions.items()},
            "last_quarter_first_actions": {str(k): dict(v) for k, v in last_block_actions.items()},
            "water_consumed_by_trial_condition": {str(k): v for k, v in consumed.items()},
            "total_water_consumed": sim.statistics.water_consumed,
            "learning_updates": human.brain.learning_updates, "q_memory_size": human.brain.memory_size}


def run_demo(trials: int = 300, ticks_per_trial: int = 6) -> dict:
    if type(trials) is not int or trials < 2 or type(ticks_per_trial) is not int or not 1 <= ticks_per_trial <= 10:
        raise ValueError("Use at least two trials and 1..10 ordinary actions per trial")
    return {"trials": trials, "ticks_per_trial": ticks_per_trial,
            "intervention": "Alternate two contexts: arbitrary cue 7 or 12 from E, receiver at x=2, thirst=60, one water unit at x=4 or x=0. One cue at trial start; no edited Q values/rewards. Pending transitions persist across trial resets.",
            "interpretation": "A controlled receiver association, not signal semantics, a convention or ecological fitness evidence.",
            "conditions": [run_condition(enabled, trials, ticks_per_trial) for enabled in (True, False)]}


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
