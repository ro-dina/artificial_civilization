from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import Config, INITIAL_ENERGY

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

    def update_physiology(self, config: Config) -> str | None:
        """Apply tick costs, then return a cause if this tick kills the agent."""
        if not self.alive:
            return None
        self.hunger += config.hunger_per_tick
        self.thirst += config.thirst_per_tick
        self.age += 1
        cause = None
        if self.hunger >= config.max_hunger:
            cause = "starvation"
        elif self.thirst >= config.max_thirst:
            cause = "dehydration"
        elif config.max_age is not None and self.age >= config.max_age:
            cause = "old_age"
        if cause is not None:
            self.alive = False
        return cause
