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


class VocalKind(str, Enum):
    SILENCE = "SILENCE"
    SIGNAL = "SIGNAL"


@dataclass(frozen=True, slots=True)
class VocalAction:
    """A production choice, distinct from the receiver's AuditoryPercept."""

    kind: VocalKind = VocalKind.SILENCE
    signal_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, VocalKind):
            raise ValueError("kind must be a VocalKind")
        if self.kind is VocalKind.SIGNAL:
            if type(self.signal_id) is not int or self.signal_id < 0:
                raise ValueError("SIGNAL requires a nonnegative integer signal ID")
        elif self.signal_id is not None:
            raise ValueError("SILENCE cannot carry a signal ID")

    @property
    def label(self) -> str:
        return "SILENCE" if self.kind is VocalKind.SILENCE else f"SIGNAL_{self.signal_id}"


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    signal_id: int | None = None

    @property
    def vocal_action(self) -> VocalAction:
        """Explicit view preserving the existing Decision(action, signal_id) API."""
        return (VocalAction() if self.signal_id is None
                else VocalAction(VocalKind.SIGNAL, self.signal_id))


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
