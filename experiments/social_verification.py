"""Complete state projection for social ON/OFF comparisons, never a Brain input."""
from dataclasses import asdict, fields
import hashlib
import json
from agents.human import Human
from experiments.auditory_policy import frozen


def state(sim, *, include_social=False):
    bodies = tuple((tuple((f.name, frozen(getattr(h, f.name))) for f in fields(Human) if f.name != 'brain'),
                    frozen(vars(h.brain))) for h in sim.agents)
    value = (sim.tick, sim._next_id, asdict(sim.config), asdict(sim.statistics), bodies,
             tuple((t.food, t.water) for row in sim.world.tiles for t in row),
             frozen(vars(sim.communication)), frozen(vars(sim.reproduction)),
             tuple(frozen(getattr(sim, n)) for n in ('_order_rng', '_offspring_rng', '_inheritance_rng',
                   '_mutation_rng', '_food_rng', '_water_rng')))
    if include_social:
        value = (value, tuple((h.id, h.social_memory.capacity, h.social_memory.reunion_min_absence_ticks,
                              h.social_memory._last_observation_tick, h.social_memory.entries()) for h in sim.agents))
    return value


def digest(sim, *, include_social=False):
    return hashlib.sha256(json.dumps(state(sim, include_social=include_social), default=asdict,
                                    separators=(',', ':')).encode()).hexdigest()
