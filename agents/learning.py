"""Individual, bounded tabular Q-learning from local senses and body outcomes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

from agents.brain import Action, Decision, VocalAction, VocalKind
from agents.auditory_memory import AuditoryFeature, AuditoryMemory
from agents.observation import Observation, SILENCE
from config import Config

if TYPE_CHECKING:
    from agents.human import Human

State = tuple[int | AuditoryFeature, ...]


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
    """Independent bounded physical/vocal Q heads with coarse shared body reward.

    The next observation arrives only at the next ordinary choose_action call.
    Surviving transitions are then updated; terminal outcomes update immediately.
    """

    def __init__(self, rng: Random, config: Config) -> None:
        self.rng = rng
        # Derived solely from this agent's seeded stream; emissions cannot shift
        # later exploration/tie-breaking draws in this new controller.
        signal_seed = rng.getrandbits(64)  # Preserve the single v0.3 construction draw.
        self.signal_rng = Random(signal_seed)
        self.learning_controls_vocalization = config.learning_controls_vocalization
        # No extra draws from either historical stream. The per-agent seed above
        # already derives from Simulation's named brain:<id> stream.
        self.vocal_rng = (Random(f"{signal_seed}:vocal-choice")
                          if self.learning_controls_vocalization else None)
        self.vocab_size = config.signal_vocab_size
        self.vocal_actions = ((VocalAction(), *(VocalAction(VocalKind.SIGNAL, i)
                                               for i in range(self.vocab_size)))
                              if self.learning_controls_vocalization else ())
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
        self.vocal_capacity = config.vocal_learning_memory_capacity
        self._vocal_values: OrderedDict[State, list[float]] = OrderedDict()
        self._vocal_pending: Transition | None = None
        self.vocal_learning_updates = 0
        self.vocal_exploratory_actions = 0
        self.vocal_exploitative_actions = 0
        self.last_vocal_exploratory: bool | None = None
        self.last_vocal_action: VocalAction | None = None
        self.last_vocal_value: float | None = None
        self.learning_updates = 0
        self.exploratory_actions = 0
        self.exploitative_actions = 0
        self.last_exploratory: bool | None = None
        self.last_reward: float | None = None
        self.learning_uses_auditory = config.learning_uses_auditory
        self.auditory_memory = AuditoryMemory(config.auditory_memory_ticks)

    @property
    def memory_size(self) -> int:
        return len(self._values)

    @property
    def vocal_memory_size(self) -> int:
        return len(self._vocal_values)

    def vocal_action_values(self, body: Physiology, observation: Observation) -> tuple[float, ...]:
        """Read-only probe of the independent vocal head; empty in sender control."""
        return tuple(self._vocal_values.get(self.state(body, observation), (0.0,) * len(self.vocal_actions)))

    def vocal_diagnostics(self) -> dict:
        """Opt-in, read-only summary, without whole tables or receiver information."""
        return {"control": "learned" if self.learning_controls_vocalization else "random_control",
                "action": self.last_vocal_action, "exploratory": self.last_vocal_exploratory,
                "updates": self.vocal_learning_updates, "memory_size": self.vocal_memory_size,
                "exploratory_actions": self.vocal_exploratory_actions,
                "exploitative_actions": self.vocal_exploitative_actions,
                "selected_q_value": self.last_vocal_value}

    def _bin(self, value: float, limit: float) -> int:
        return max(0, min(self.need_bins - 1, int(self.need_bins * value / limit)))

    def state(self, body: Physiology, observation: Observation) -> State:
        """Original body/visual features, optionally followed by one auditory feature.

        This diagnostic preview never updates memory, Q rows, LRU order or RNG.
        """
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
        original = (self._bin(body.hunger, self.max_hunger), self._bin(body.thirst, self.max_thirst),
                    self._bin(body.energy, self.max_energy), *directions)
        if not self.learning_uses_auditory:
            return original
        return (*original, self.auditory_memory.feature(observation.auditory, observation.tick))

    def auditory_diagnostics(self, tick: int) -> dict:
        """Optional read-only record of memory and the feature used at this tick."""
        return {"learning_uses_auditory": self.learning_uses_auditory,
                "memory": tuple({"received_tick": event.received_tick, "age": tick - event.received_tick,
                                 "percept": event.percept} for event in self.auditory_memory.events(tick)),
                "feature": self.auditory_memory.feature(SILENCE, tick) if self.learning_uses_auditory else None}

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

    def _remember_vocal(self, state: State) -> list[float]:
        if state not in self._vocal_values:
            if len(self._vocal_values) == self.vocal_capacity:
                self._vocal_values.popitem(last=False)
            self._vocal_values[state] = [0.0] * len(self.vocal_actions)
        self._vocal_values.move_to_end(state)
        return self._vocal_values[state]

    def _update_vocal(self, next_value: float) -> None:
        transition = self._vocal_pending
        if transition is None or transition.reward is None:
            raise RuntimeError("Vocal learning requires a completed action outcome")
        values = self._remember_vocal(transition.state)
        index = transition.action_index
        target = transition.reward + self.discount * next_value
        values[index] += self.learning_rate * (target - values[index])
        self.vocal_learning_updates += 1
        self._vocal_pending = None

    def choose_action(self, human: Human, observation: Observation) -> Decision:
        if self._pending is not None and self._pending.reward is None:
            raise RuntimeError("Learning requires a completed action outcome")
        self.auditory_memory.update(observation.auditory, observation.tick)
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
        if self.learning_controls_vocalization:
            if self._vocal_pending is not None:
                next_value = max(self._vocal_values.get(state, (0.0,) * len(self.vocal_actions)))
                self._update_vocal(next_value)
            vocal_values = self._remember_vocal(state)
            self.last_vocal_exploratory = self.vocal_rng.random() < self.epsilon
            if self.last_vocal_exploratory:
                vocal_index = self.vocal_rng.randrange(len(self.vocal_actions))
                self.vocal_exploratory_actions += 1
            else:
                best = max(vocal_values)
                vocal_index = self.vocal_rng.choice([i for i, value in enumerate(vocal_values) if value == best])
                self.vocal_exploitative_actions += 1
            self._vocal_pending = Transition(state, vocal_index, body)
            self.last_vocal_action = self.vocal_actions[vocal_index]
            self.last_vocal_value = vocal_values[vocal_index]
            signal = self.last_vocal_action.signal_id
        else:
            signal = None
            if self.signal_rng.random() < human.genome.signal_probability:
                signal = self.signal_rng.randrange(self.vocab_size)
            self.last_vocal_action = (VocalAction() if signal is None else VocalAction(VocalKind.SIGNAL, signal))
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
        if self._vocal_pending is not None:
            self._vocal_pending.reward = reward  # Compute once, share exactly; no hearing feedback.
        if not after.alive:
            self._update(0.0)
            if self._vocal_pending is not None:
                self._update_vocal(0.0)
