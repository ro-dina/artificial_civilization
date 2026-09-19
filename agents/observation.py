"""Immutable, numeric sensory records; no references to the mutable world."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TileObservation:
    dx: int
    dy: int
    food: int
    water: int


@dataclass(frozen=True, slots=True)
class HeardSignal:
    signal_id: int
    sender_id: int
    dx: int
    dy: int


@dataclass(frozen=True, slots=True)
class Observation:
    tick: int
    tiles: tuple[TileObservation, ...]
    signals: tuple[HeardSignal, ...] = ()
