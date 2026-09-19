"""Individual, bounded tabular Q-learning from local senses and body outcomes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

from agents.brain import Action, Decision
from agents.observation import Observation
from config import Config

if TYPE_CHECKING:
    from agents.human import Human

State = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Physiology:
    hunger: float
    thirst: float
    energy: float
    alive: bool

    @classmethod
    def capture(cls, human: Human) -> Physiology:
        return cls(human.hunger, human.thirst, human.energy, human.alive)


@dataclass(slots=True)
class Transition:
    state: State
    action_index: int
    before: Physiology
    reward: float | None = None


class LearningBrain:
    """Zero-prior Q values, constant epsilon exploration, and no signal features.

    The next observation arrives only at the next ordinary choose_action call.
    Surviving transitions are then updated; terminal outcomes update immediately.
    """

    def __init__(self, rng: Random, config: Config) -> None:
        self.rng = rng
        # Derived solely from this agent's seeded stream; emissions cannot shift
        # later exploration/tie-breaking draws in this new controller.
        self.signal_rng = Random(rng.getrandbits(64))
        self.vocab_size = config.signal_vocab_size
        self.actions = tuple(a for a in Action if config.reproduction_enabled or a is not Action.REPRODUCE)
        self.learning_rate = config.learning_rate
        self.discount = config.learning_discount
        self.epsilon = config.learning_epsilon
        self.capacity = config.learning_memory_capacity
        self.need_bins = config.learning_need_bins
        self.max_hunger = config.max_hunger
        self.max_thirst = config.max_thirst
        self.max_energy = config.max_energy
        self._values: OrderedDict[State, list[float]] = OrderedDict()
        self._pending: Transition | None = None
        self.learning_updates = 0
        self.exploratory_actions = 0
        self.exploitative_actions = 0
        self.last_exploratory: bool | None = None
        self.last_reward: float | None = None

    @property
    def memory_size(self) -> int:
        return len(self._values)

    def _bin(self, value: float, limit: float) -> int:
        return max(0, min(self.need_bins - 1, int(self.need_bins * value / limit)))

    def state(self, body: Physiology, observation: Observation) -> State:
        """Three body bins and nearest-resource direction categories, nothing global."""
        directions = []
        for resource in ("food", "water"):
            nearest = min(
                (tile for tile in observation.tiles if getattr(tile, resource) > 0),
                key=lambda tile: (abs(tile.dx) + abs(tile.dy), tile.dy, tile.dx),
                default=None,
            )
            if nearest is None:
                directions.extend((2, 2))  # Distinct from the nine sign directions.
            else:
                directions.extend(((nearest.dx > 0) - (nearest.dx < 0),
                                   (nearest.dy > 0) - (nearest.dy < 0)))
        return (self._bin(body.hunger, self.max_hunger), self._bin(body.thirst, self.max_thirst),
                self._bin(body.energy, self.max_energy), *directions)

    def action_values(self, body: Physiology, observation: Observation) -> tuple[float, ...]:
        """Read-only diagnostic; unseen states have all-zero values and are not stored."""
        return tuple(self._values.get(self.state(body, observation), (0.0,) * len(self.actions)))

    def _remember(self, state: State) -> list[float]:
        if state not in self._values:
            if len(self._values) == self.capacity:
                self._values.popitem(last=False)
            self._values[state] = [0.0] * len(self.actions)
        self._values.move_to_end(state)
        return self._values[state]

    def _update(self, next_value: float) -> None:
        transition = self._pending
        if transition is None or transition.reward is None:
            raise RuntimeError("Learning requires a completed action outcome")
        values = self._remember(transition.state)
        index = transition.action_index
        target = transition.reward + self.discount * next_value
        values[index] += self.learning_rate * (target - values[index])
        self.learning_updates += 1
        self._pending = None

    def choose_action(self, human: Human, observation: Observation) -> Decision:
        body = Physiology.capture(human)
        state = self.state(body, observation)
        if self._pending is not None:
            next_value = max(self._values.get(state, (0.0,) * len(self.actions)))
            self._update(next_value)
        values = self._remember(state)
        self.last_exploratory = self.rng.random() < self.epsilon
        if self.last_exploratory:
            index = self.rng.randrange(len(self.actions))
            self.exploratory_actions += 1
        else:
            best = max(values)
            index = self.rng.choice([i for i, value in enumerate(values) if value == best])
            self.exploitative_actions += 1
        self._pending = Transition(state, index, body)
        signal = None
        if self.signal_rng.random() < human.genome.signal_probability:
            signal = self.signal_rng.randrange(self.vocab_size)
        return Decision(self.actions[index], signal)

    def observe_outcome(self, after: Physiology) -> None:
        """Only actual physiological changes supply reward; no resource/action labels."""
        if self._pending is None or self._pending.reward is not None:
            raise RuntimeError("An outcome must follow exactly one decision")
        before = self._pending.before
        reward = ((before.hunger - after.hunger) / self.max_hunger
                  + (before.thirst - after.thirst) / self.max_thirst
                  + (after.energy - before.energy) / self.max_energy)
        self.last_reward = self._pending.reward = reward
        if not after.alive:
            self._update(0.0)
