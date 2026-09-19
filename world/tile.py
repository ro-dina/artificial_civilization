from dataclasses import dataclass


@dataclass(slots=True)
class Tile:
    """Counts of consumable resource units; both may occupy the same tile."""

    food: int = 0
    water: int = 0
