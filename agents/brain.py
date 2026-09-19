from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import TYPE_CHECKING, Protocol

from agents.observation import Observation
from config import Config

if TYPE_CHECKING:
    from agents.human import Human


class Action(str, Enum):
    MOVE_NORTH = "move_north"
    MOVE_SOUTH = "move_south"
    MOVE_EAST = "move_east"
    MOVE_WEST = "move_west"
    EAT = "eat"
    DRINK = "drink"
    WAIT = "wait"
    REPRODUCE = "reproduce"


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    signal_id: int | None = None


class Brain(Protocol):
    def choose_action(self, human: Human, observation: Observation) -> Decision:
        """Read one's own body and local senses; return an action without mutating them."""
        ...


class RandomBrain:
    """Uniform random actions, with an independent chance of a meaningless signal."""

    def __init__(self, rng: Random, config: Config) -> None:
        self.rng = rng
        self.vocab_size = config.signal_vocab_size
        self.actions = tuple(a for a in Action if config.reproduction_enabled or a is not Action.REPRODUCE)

    def choose_action(self, human: Human, observation: Observation) -> Decision:
        action = self.rng.choice(self.actions)
        signal = None
        if self.rng.random() < human.genome.signal_probability:
            signal = self.rng.randrange(self.vocab_size)
        return Decision(action, signal)
