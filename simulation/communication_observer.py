"""Optional compact, privileged communication records. No behavior or RNG."""

from collections.abc import Callable
import csv
from dataclasses import dataclass, fields
from typing import TextIO

from agents.brain import Decision, RandomBrain
from agents.human import Human
from agents.learning import LearningBrain, Physiology, PrimitiveStateEncoder
from agents.observation import Observation
from config import Config

SCHEMA_VERSION = 1
DENSITY_RADIUS = 2


@dataclass(frozen=True, slots=True)
class CommunicationRecord:
    # A single actor is both receiver of t-1 and producer during t. All body
    # features/age are pre-action. It is not a sender->receiver delivery edge.
    tick: int
    agent_id: int
    brain_type: str
    learning_controls_vocalization: bool
    learning_uses_auditory: bool
    vocal_mechanism: str
    vocal_action: str
    signal_id: int | None
    physical_action: str
    hunger_bin: int
    thirst_bin: int
    energy_bin: int
    food_dx: int
    food_dy: int
    water_dx: int
    water_dy: int
    auditory_kind: str
    heard_signal_id: int | None
    heard_direction: str | None
    encoded_auditory_kind: str | None
    encoded_signal_id: int | None
    encoded_direction: str | None
    encoded_age: int | None
    perception_radius: int
    generation: int
    age: int
    local_density: int


CommunicationSink = Callable[[CommunicationRecord], None]
COLUMNS = tuple(field.name for field in fields(CommunicationRecord))


class CSVCommunicationWriter:
    """No row retention or sampling; the caller owns the text stream."""

    def __init__(self, stream: TextIO) -> None:
        self.writer = csv.writer(stream)
        self.writer.writerow(COLUMNS)

    def __call__(self, record: CommunicationRecord) -> None:
        self.writer.writerow(tuple(getattr(record, name) for name in COLUMNS))


class CommunicationObserver:
    def __init__(self, config: Config, sink: CommunicationSink) -> None:
        self.config = config
        self.sink = sink
        self.encoder = PrimitiveStateEncoder(config)
        self._prefix = []

    def begin_tick(self, agents: list[Human]) -> None:
        """O(world area + population), tick-start density; no live index updates."""
        width, height = self.config.world_width, self.config.world_height
        prefix = [[0] * (width + 1) for _ in range(height + 1)]
        for human in agents:
            prefix[human.y + 1][human.x + 1] += 1
        for y in range(1, height + 1):
            running = 0
            for x in range(1, width + 1):
                running += prefix[y][x]
                prefix[y][x] = running + prefix[y - 1][x]
        self._prefix = prefix

    def density(self, x: int, y: int) -> int:
        radius = DENSITY_RADIUS
        left, top = max(0, x - radius), max(0, y - radius)
        right = min(self.config.world_width, x + radius + 1)
        bottom = min(self.config.world_height, y + radius + 1)
        p = self._prefix
        return p[bottom][right] - p[top][right] - p[bottom][left] + p[top][left] - 1

    def decision(self, human: Human, observation: Observation, decision: Decision) -> None:
        brain = human.brain
        learner = isinstance(brain, LearningBrain)
        state = (brain.decision_state if learner else
                 self.encoder.primitive_state(Physiology.capture(human), observation))
        # Called immediately after choose_action, before execution or feedback.
        if state is None:
            raise RuntimeError("A learner's recorded decision must have a pending state")
        controls = learner and brain.learning_controls_vocalization
        listening = learner and brain.learning_uses_auditory
        feature = state[7] if listening else ()
        kind = feature[0].value if feature else None
        encoded_id = feature[1] if len(feature) == 4 else None
        encoded_direction = feature[2].value if len(feature) == 4 else None
        encoded_age = feature[-1] if len(feature) > 1 else None
        heard = observation.auditory.signal
        self.sink(CommunicationRecord(
            observation.tick, human.id, "learning" if learner else "random" if isinstance(brain, RandomBrain) else "custom",
            controls, listening, "learned" if controls else "random" if learner or isinstance(brain, RandomBrain) else "custom",
            "SILENCE" if decision.signal_id is None else "SIGNAL", decision.signal_id, decision.action.value,
            *state[:7], observation.auditory.kind.value,
            heard.signal_id if heard is not None else None, heard.source_direction.value if heard is not None else None,
            kind, encoded_id, encoded_direction, encoded_age,
            human.genome.perception_radius, human.generation, human.age, self.density(human.x, human.y)))
