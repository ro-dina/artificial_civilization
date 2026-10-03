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


@dataclass(frozen=True, slots=True)
class Observation:
    tick: int
    tiles: tuple[TileObservation, ...]
    signals: tuple[HeardSignal, ...] = ()
