import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from math import fsum
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from agents.human import Human

GENOME_TRAITS = (
    "perception_radius", "signal_probability", "hunger_multiplier", "thirst_multiplier",
)


def summarize_genomes(agents: Iterable["Human"]) -> dict[str, float | int | None]:
    """Read living genomes only; empty populations have no mean or extrema."""
    values = {trait: [] for trait in GENOME_TRAITS}
    for human in agents:
        if human.alive:
            for trait in GENOME_TRAITS:
                values[trait].append(getattr(human.genome, trait))
    summary = {}
    for trait, samples in values.items():
        summary[f"mean_{trait}"] = fsum(samples) / len(samples) if samples else None
        summary[f"min_{trait}"] = min(samples) if samples else None
        summary[f"max_{trait}"] = max(samples) if samples else None
    return summary


@dataclass(frozen=True, slots=True)
class Statistics:
    """Population/traits are current; generation is historical and counters cumulative."""

    tick: int
    population: int
    deaths: int = 0
    food_consumed: int = 0
    water_consumed: int = 0
    signals_emitted: int = 0
    births: int = 0
    highest_generation: int = 0
    reproduction_attempts: int = 0
    successful_reproductions: int = 0
    population_limit_blocks: int = 0
    food_regenerated: int = 0
    water_regenerated: int = 0
    starvation_deaths: int = 0
    dehydration_deaths: int = 0
    old_age_deaths: int = 0
    energy_deaths: int = 0
    mean_perception_radius: float | None = None
    mean_signal_probability: float | None = None
    mean_hunger_multiplier: float | None = None
    mean_thirst_multiplier: float | None = None
    min_perception_radius: int | None = None
    max_perception_radius: int | None = None
    min_signal_probability: float | None = None
    max_signal_probability: float | None = None
    min_hunger_multiplier: float | None = None
    max_hunger_multiplier: float | None = None
    min_thirst_multiplier: float | None = None
    max_thirst_multiplier: float | None = None


class CSVStatisticsWriter:
    """Stream snapshots to a caller-owned text file without retaining history."""

    def __init__(self, stream: TextIO) -> None:
        self._writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(Statistics)])
        self._writer.writeheader()

    def write(self, statistics: Statistics) -> None:
        self._writer.writerow(asdict(statistics))
