import csv
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from math import fsum
from typing import TYPE_CHECKING, TextIO

from agents.learning import LearningBrain

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


def summarize_learning(agents: Iterable["Human"]) -> dict[str, float | int | None]:
    count = updates = size = 0
    for human in agents:
        if human.alive and isinstance(human.brain, LearningBrain):
            count += 1
            updates += human.brain.learning_updates
            size += human.brain.memory_size
    return {"learning_agents": count, "mean_learning_updates": updates / count if count else None,
            "mean_memory_size": size / count if count else None}


@dataclass(frozen=True, slots=True)
class Statistics:
    """Population/traits are current; generation is historical and counters cumulative."""

    tick: int
    population: int
    learning_agents: int = 0
    total_learning_updates: int = 0
    mean_learning_updates: float | None = None
    mean_memory_size: float | None = None
    exploratory_actions: int = 0
    exploitative_actions: int = 0
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

    def __init__(self, stream: TextIO, *, metadata: Mapping[str, str | int] | None = None) -> None:
        self._metadata = dict(metadata or {})
        names = [field.name for field in fields(Statistics)]
        if self._metadata.keys() & set(names):
            raise ValueError("CSV metadata must not overwrite statistics columns")
        self._writer = csv.DictWriter(stream, fieldnames=[*self._metadata, *names])
        self._writer.writeheader()

    def write(self, statistics: Statistics) -> None:
        self._writer.writerow({**self._metadata, **asdict(statistics)})
