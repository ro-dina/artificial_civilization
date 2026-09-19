"""Repeat the same local sensory situation to isolate learned action preferences.

Run from the repository root: python3 -m experiments.learning_demo
Needs and one water unit are reset between trials by this experimental harness,
not by the simulation or brain. Each action and reward uses real v0.2 mechanics.
"""

from collections import Counter
from dataclasses import asdict
import json

from agents.learning import Physiology
from config import Config
from simulation.simulation import Simulation


def run_demo(trials: int = 400) -> dict:
    config = Config(brain="learning", random_seed=0, world_width=1, world_height=1,
                    initial_population=1, initial_food=0, initial_water=1, perception_radius=0,
                    hunger_per_tick=0, thirst_per_tick=1, max_age=None)
    counts = Counter()
    def sink(event):
        if event.kind == "agent_step":
            counts[event.details["action"]] += 1

    sim = Simulation(config, event_sink=sink)
    human = sim.agents[0]
    human.thirst = 60
    body, observation = Physiology.capture(human), sim.observe(human)
    initial_values = human.brain.action_values(body, observation)
    for _ in range(trials):
        human.thirst = 60
        sim.world.tile_at(0, 0).water = 1
        sim.step()
    learned_values = human.brain.action_values(body, observation)
    names = [action.value for action in human.brain.actions]
    return {
        "trials": trials, "config": asdict(config),
        "intervention": "Before each trial: set thirst to 60 and current tile water to one unit.",
        "initial_values": dict(zip(names, initial_values)),
        "learned_values": dict(zip(names, learned_values)),
        "initial_greedy_actions": [name for name, value in zip(names, initial_values) if value == max(initial_values)],
        "learned_greedy_actions": [name for name, value in zip(names, learned_values) if value == max(learned_values)],
        "actions": dict(counts), "learning_updates": human.brain.learning_updates,
        "memory_size": human.brain.memory_size,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
