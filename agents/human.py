from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agents.genome import Genome
from config import Config, INITIAL_ENERGY, REPRODUCTIVE_TYPES

if TYPE_CHECKING:
    from agents.brain import Brain


@dataclass(slots=True)
class Human:
    id: int
    x: int
    y: int
    brain: Brain
    reproductive_type: int = 0
    age: int = 0
    hunger: float = 0
    thirst: float = 0
    energy: float = INITIAL_ENERGY
    alive: bool = True
    genome: Genome = field(default_factory=Genome)
    parent_a_id: int | None = None
    parent_b_id: int | None = None
    generation: int = 0
    birth_tick: int = 0
    next_reproduction_tick: int = 0

    def __post_init__(self) -> None:
        if type(self.reproductive_type) is not int or self.reproductive_type not in REPRODUCTIVE_TYPES:
            raise ValueError("reproductive_type must be 0 or 1")
        if not isinstance(self.genome, Genome):
            raise ValueError("genome must be a Genome")

    def update_physiology(self, config: Config) -> str | None:
        """Apply tick costs, then return a cause if this tick kills the agent."""
        if not self.alive:
            return None
        self.hunger += config.hunger_per_tick * self.genome.hunger_multiplier
        self.thirst += config.thirst_per_tick * self.genome.thirst_multiplier
        self.energy = max(0, self.energy - config.energy_per_tick)
        self.age += 1
        cause = None
        if self.hunger >= config.max_hunger:
            cause = "starvation"
        elif self.thirst >= config.max_thirst:
            cause = "dehydration"
        elif config.max_age is not None and self.age >= config.max_age:
            cause = "old_age"
        elif config.energy_depletion_lethal and self.energy <= 0:
            cause = "energy"
        if cause is not None:
            self.alive = False
        return cause
