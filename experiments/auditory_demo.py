"""Three-tick hearing mechanism check, not communication learning or evolution."""

from dataclasses import asdict
import json

from agents.brain import Action, Decision
from config import Config
from simulation.simulation import Simulation


class ProbeBrain:
    def __init__(self, signal=None):
        self.signal = signal
        self.observations = []

    def choose_action(self, human, observation):
        self.observations.append(observation)
        return Decision(Action.WAIT, self.signal if observation.tick == 1 else None)


def run_demo():
    sim = Simulation(Config(world_width=12, world_height=3, initial_population=2,
                            initial_food=0, initial_water=0, perception_radius=2,
                            hearing_radius=8, hunger_per_tick=0, thirst_per_tick=0,
                            max_age=None, random_seed=0))
    sender, receiver = sim.agents
    sender.x, sender.y = 6, 1
    receiver.x, receiver.y = 1, 1
    sender.brain, receiver.brain = ProbeBrain(signal=7), ProbeBrain()
    for _ in range(3):
        sim.step()
    records = receiver.brain.observations
    return {"visual_radius": receiver.genome.perception_radius,
            "hearing_radius": sim.config.hearing_radius,
            "separation_for_diagnostics_only": 5,
            "sender_cell_visible": any((t.dx, t.dy) == (5, 0) for t in records[1].tiles),
            "receiver_observations": [{"tick": obs.tick, "signals": [asdict(s) for s in obs.signals]}
                                      for obs in records]}


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
