"""Immutable local senses; auditory records reveal neither identity nor location."""

from dataclasses import dataclass
from enum import Enum


class SourceDirection(str, Enum):
    N = "N"
    NE = "NE"
    E = "E"
    SE = "SE"
    S = "S"
    SW = "SW"
    W = "W"
    NW = "NW"
    SAME_CELL = "SAME_CELL"

    @classmethod
    def from_offset(cls, dx: int, dy: int) -> "SourceDirection":
        """Sign sectors, not angular octants; north decreases y."""
        if dx == 0 and dy == 0:
            return cls.SAME_CELL
        vertical = "N" if dy < 0 else "S" if dy > 0 else ""
        horizontal = "W" if dx < 0 else "E" if dx > 0 else ""
        return cls(vertical + horizontal)


@dataclass(frozen=True, slots=True)
class TileObservation:
    dx: int
    dy: int
    food: int
    water: int


@dataclass(frozen=True, slots=True)
class HeardSignal:
    signal_id: int
    source_direction: SourceDirection


class AuditoryKind(str, Enum):
    SILENCE = "SILENCE"
    MASKED = "MASKED"
    IDENTIFIED = "IDENTIFIED"


@dataclass(frozen=True, slots=True)
class AuditoryPercept:
    """A resolved sound: only IDENTIFIED can carry an ID and direction."""

    kind: AuditoryKind = AuditoryKind.SILENCE
    signal: HeardSignal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AuditoryKind):
            raise ValueError("kind must be an AuditoryKind")
        if self.kind is AuditoryKind.IDENTIFIED:
            if (not isinstance(self.signal, HeardSignal) or type(self.signal.signal_id) is not int
                    or self.signal.signal_id < 0 or not isinstance(self.signal.source_direction, SourceDirection)):
                raise ValueError("IDENTIFIED requires a nonnegative signal ID and a SourceDirection")
        elif self.signal is not None:
            raise ValueError("SILENCE and MASKED cannot expose a signal or direction")


SILENCE = AuditoryPercept()
MASKED = AuditoryPercept(AuditoryKind.MASKED)


@dataclass(frozen=True, slots=True)
class Observation:
    tick: int
    tiles: tuple[TileObservation, ...]
    auditory: AuditoryPercept = SILENCE

    def __post_init__(self) -> None:
        if not isinstance(self.auditory, AuditoryPercept):
            raise ValueError("auditory must be a resolved AuditoryPercept, not a candidate list")

    @property
    def signals(self) -> tuple[HeardSignal, ...]:
        """Compatibility view: zero or one identified sound, never raw candidates.

        Use auditory.kind to distinguish silence from masking.
        """
        return (self.auditory.signal,) if self.auditory.kind is AuditoryKind.IDENTIFIED else ()
