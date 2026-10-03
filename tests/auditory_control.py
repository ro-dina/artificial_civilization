"""v0.3/v0.4 comparison: ecology and emissions, independent of hearing buckets.

Signal provenance is retained in the digest, but spatial bucket layout and unused
receiver observations are not. Q rows remain ordered to check LRU behavior.
"""

from dataclasses import asdict
import hashlib
import json

from agents.learning import LearningBrain
from tests.test_statistics import dynamics


def ecological_state(sim):
    state = list(dynamics(sim))
    state[2] = asdict(sim.statistics)  # Include every aggregate, including traits.
    for index, buffer in ((6, sim.communication._audible), (7, sim.communication._pending)):
        state[index] = sorted((s.sender_id, s.signal_id, s.x, s.y)
                              for signals in buffer.values() for s in signals)
    learning = tuple(
        (h.id, tuple((key, tuple(values)) for key, values in h.brain._values.items()),
         asdict(h.brain._pending) if h.brain._pending is not None else None,
         h.brain.signal_rng.getstate(), h.brain.learning_updates,
         h.brain.exploratory_actions, h.brain.exploitative_actions,
         h.brain.last_reward, h.brain.last_exploratory)
        for h in sim.agents if isinstance(h.brain, LearningBrain)
    )
    return state, learning


def encoded(value):
    return json.dumps(value, default=asdict, sort_keys=True, separators=(",", ":")).encode()


class ActionDigest:
    """Stream actions/outcomes of every actor, including individuals that die."""

    def __init__(self):
        self.digest = hashlib.sha256()

    def __call__(self, event):
        if event.kind == "agent_step":
            details = {k: v for k, v in event.details.items()
                       if k not in ("observation", "auditory_diagnostics")}
            self.digest.update(encoded((event.tick, event.agent_id, event.x, event.y, details)))


def capture(sim, ticks):
    trajectory = hashlib.sha256()
    for tick in range(ticks + 1):
        if tick:
            sim.step()
        trajectory.update(encoded(ecological_state(sim)))
    return {"trajectory_sha256": trajectory.hexdigest(),
            "final_state_sha256": hashlib.sha256(encoded(ecological_state(sim))).hexdigest(),
            "statistics": asdict(sim.statistics)}
