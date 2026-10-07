"""Small opt-in checks of initial vocal activity and the four learning controls.

No parameter tuning, history retention, semantic labels or long experiments.
"""

from collections import Counter
from dataclasses import asdict
import json
from random import Random
from time import perf_counter

from agents.genome import Genome
from agents.human import Human
from agents.learning import LearningBrain, Physiology
from agents.observation import Observation
from config import Config
from simulation.simulation import Simulation


def initial_rate(decisions: int = 1000, seed: int = 0, vocabulary: int = 16) -> dict:
    if type(decisions) is not int or decisions < 1:
        raise ValueError("decisions must be a positive integer")
    cfg = Config(brain="learning", signal_vocab_size=vocabulary, signal_probability=0,
                 learning_controls_vocalization=True, learning_uses_auditory=False)
    brain = LearningBrain(Random(f"{seed}:brain:0"), cfg)
    human = Human(0, 0, 0, brain, genome=Genome.founder(cfg))
    counts = Counter()
    for tick in range(decisions):
        decision = brain.choose_action(human, Observation(tick, ()))
        counts[decision.vocal_action.label] += 1
        # Exactly zero body change: every Q value remains zero, so each choice
        # measures the initial tied policy rather than its changing learned values.
        brain.observe_outcome(Physiology.capture(human))
    emitted = decisions - counts["SILENCE"]
    return {"seed": seed, "vocabulary": vocabulary, "decisions": decisions,
            "zero_reward_tied_policy": True, "counts": dict(sorted(counts.items())),
            "emitted": emitted, "silent": counts["SILENCE"], "emission_rate": emitted / decisions,
            "expected_emission_rate": vocabulary / (vocabulary + 1)}


def short_condition(listening: bool, vocal_learning: bool, ticks: int = 100, seed: int = 0) -> dict:
    if type(ticks) is not int or not 1 <= ticks <= 100:
        raise ValueError("This mechanism check allows only 1..100 ticks")
    cfg = Config.evolution(brain="learning", random_seed=seed, learning_uses_auditory=listening,
                           learning_controls_vocalization=vocal_learning)
    percepts = Counter()
    vocal_updates = vocal_explore = vocal_exploit = 0
    def sink(event):
        nonlocal vocal_updates, vocal_explore, vocal_exploit
        if event.kind != "agent_step":
            return
        percepts[event.details["observation"].auditory.kind.value] += 1
        diag = event.details["vocal_learning"]
        vocal_updates += diag["updates_this_turn"]
        vocal_explore += int(diag["exploratory"] is True)
        vocal_exploit += int(diag["exploratory"] is False)
    start = perf_counter()
    sim = Simulation(cfg, event_sink=sink)
    for _ in range(ticks):
        sim.step()
    elapsed = perf_counter() - start
    agents = sim.agents
    stats = sim.statistics
    opportunities = sum(percepts.values())
    return {"listening": listening, "vocal_learning": vocal_learning, "config": asdict(cfg),
            "tick": sim.tick, "population": sim.population, "births": stats.births, "deaths": stats.deaths,
            "signals_emitted": stats.signals_emitted, "action_opportunities": opportunities,
            "emission_rate": stats.signals_emitted / opportunities if opportunities else None,
            "percepts": dict(sorted(percepts.items())),
            "masked_fraction_all_observations": percepts["MASKED"] / opportunities if opportunities else None,
            "masked_fraction_nonsilent": percepts["MASKED"] / (opportunities - percepts["SILENCE"])
                    if opportunities > percepts["SILENCE"] else None,
            "physical_updates_cumulative": stats.total_learning_updates,
            "mean_physical_memory_size_living": stats.mean_memory_size,
            "vocal_updates_cumulative": vocal_updates, "vocal_exploratory_actions": vocal_explore,
            "vocal_exploitative_actions": vocal_exploit,
            "mean_vocal_memory_size_living": sum(h.brain.vocal_memory_size for h in agents) / len(agents)
                    if agents and vocal_learning else None,
            "seconds_with_streaming_diagnostics": elapsed}


def run_checks() -> dict:
    return {"interpretation": "One seed, short descriptive mechanism checks; no fitness or communication conclusions.",
            "initial_policy": initial_rate(),
            "conditions": [short_condition(listening, vocal) for listening in (False, True) for vocal in (False, True)]}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
