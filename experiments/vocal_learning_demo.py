"""Controlled shared-reward vocal learning, without a receiver or signal meanings.

Only the body and tile are reset between ordinary one-action trials. Q values,
choices, feedback, pending transitions and RNG streams are never edited.
"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json

from agents.learning import Physiology
from config import Config
from simulation.simulation import Simulation


def run_demo(trials: int = 600, seed: int = 0) -> dict:
    if type(trials) is not int or trials < 2:
        raise ValueError("trials must be an integer of at least two")
    config = Config(brain="learning", learning_controls_vocalization=True,
                    learning_uses_auditory=False, random_seed=seed, world_width=1,
                    world_height=1, initial_population=1, initial_food=0, initial_water=0,
                    perception_radius=0, signal_probability=0, max_age=None)
    sim = Simulation(config)
    human = sim.agents[0]
    tile = sim.world.tile_at(0, 0)
    contexts = ("context_a", "context_b")
    probes = {}
    physical = {name: Counter() for name in contexts}
    vocal = {name: Counter() for name in contexts}
    late_vocal = {name: Counter() for name in contexts}

    def reset(name):
        human.hunger, human.thirst, human.energy = ((60, 0, 100) if name == "context_a" else (0, 60, 100))
        tile.food, tile.water = ((1, 0) if name == "context_a" else (0, 1))

    for name in contexts:
        reset(name)
        probes[name] = (Physiology.capture(human), sim.observe(human))

    def values():
        return {name: {"physical": dict(zip((a.value for a in human.brain.actions),
                                            human.brain.action_values(body, replace(obs, tick=sim.tick + 1)))),
                       "vocal": dict(zip((a.label for a in human.brain.vocal_actions),
                                         human.brain.vocal_action_values(body, replace(obs, tick=sim.tick + 1))))}
                for name, (body, obs) in probes.items()}

    initial = values()
    for trial in range(trials):
        name = contexts[trial % 2]
        reset(name)
        sim.step()
        brain = human.brain
        physical[name][brain.actions[brain._pending.action_index].value] += 1
        vocal[name][brain.last_vocal_action.label] += 1
        if trial >= trials * 3 // 4:
            late_vocal[name][brain.last_vocal_action.label] += 1
    learned = values()
    greedy = {name: [action for action, value in row["vocal"].items()
                     if value == max(row["vocal"].values())] for name, row in learned.items()}
    return {"trials": trials, "config": asdict(config),
            "intervention": "Alternate (hunger=60, thirst=0, food=1, water=0) and (hunger=0, thirst=60, food=0, water=1), energy=100, on the current cell. Ordinary actions/physiology/reward; no receivers. Pending transitions persist across resets.",
            "interpretation": "Vocal values reflect coarse shared reward correlations, not signal causality, semantics, communication or conventions.",
            "initial_values": initial, "learned_values": learned, "vocal_greedy_actions": greedy,
            "different_vocal_preferences": greedy[contexts[0]] != greedy[contexts[1]],
            "physical_actions": {k: dict(v) for k, v in physical.items()},
            "vocal_actions": {k: dict(v) for k, v in vocal.items()},
            "last_quarter_vocal_actions": {k: dict(v) for k, v in late_vocal.items()},
            "food_consumed": sim.statistics.food_consumed, "water_consumed": sim.statistics.water_consumed,
            "physical_learning_updates": human.brain.learning_updates,
            "vocal_learning_updates": human.brain.vocal_learning_updates,
            "physical_memory_size": human.brain.memory_size,
            "vocal_memory_size": human.brain.vocal_memory_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.trials, args.seed), indent=2))


if __name__ == "__main__":
    main()
