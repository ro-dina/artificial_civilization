"""Opt-in, read-only perception cohorts. Counters cover one tick, never a history."""
from collections.abc import Callable, Iterable
import csv
from dataclasses import dataclass, fields
from math import fsum
from typing import TextIO

from agents.brain import Action
from agents.human import Human
from config import GENOME_BOUNDS

PERCEPTIONS = tuple(range(GENOME_BOUNDS["perception_radius"][0], GENOME_BOUNDS["perception_radius"][1] + 1))
COUNTERS = ("action_opportunities", "births_as_child", "successful_parent_participations",
            "reproduction_initiations", "failed_reproduction_attempts", "deaths",
            "starvation_deaths", "dehydration_deaths", "old_age_deaths", "energy_deaths")


@dataclass(frozen=True, slots=True)
class CohortStatistics:
    tick: int
    perception_radius: int
    living_population: int = 0
    living_agent_ticks: int = 0
    action_opportunities: int = 0
    births_as_child: int = 0
    successful_parent_participations: int = 0
    reproduction_initiations: int = 0
    failed_reproduction_attempts: int = 0
    deaths: int = 0
    starvation_deaths: int = 0
    dehydration_deaths: int = 0
    old_age_deaths: int = 0
    energy_deaths: int = 0
    mean_signal_probability: float | None = None
    mean_hunger_multiplier: float | None = None
    mean_thirst_multiplier: float | None = None
    mean_generation: float | None = None
    mean_age: float | None = None


CohortSink = Callable[[tuple[CohortStatistics, ...]], None]


class PerceptionCohorts:
    def __init__(self) -> None:
        self.begin_tick()

    def begin_tick(self) -> None:
        self._counts = {p: dict.fromkeys(COUNTERS, 0) for p in PERCEPTIONS}

    def _cohort(self, human: Human) -> dict[str, int]:
        p = human.genome.perception_radius
        if p not in self._counts:
            raise ValueError(f"perception_radius outside cohort bounds: {p}")
        return self._counts[p]

    def action(self, human: Human, action: Action) -> None:
        row = self._cohort(human)
        row["action_opportunities"] += 1
        row["reproduction_initiations"] += int(action is Action.REPRODUCE)

    def birth(self, a: Human, b: Human, child: Human) -> None:
        self._cohort(child)["births_as_child"] += 1
        self._cohort(a)["successful_parent_participations"] += 1
        self._cohort(b)["successful_parent_participations"] += 1

    def failure(self, human: Human) -> None:
        self._cohort(human)["failed_reproduction_attempts"] += 1

    def death(self, human: Human, cause: str) -> None:
        row = self._cohort(human)
        row["deaths"] += 1
        row[f"{cause}_deaths"] += 1

    def snapshot(self, tick: int, humans: Iterable[Human]) -> tuple[CohortStatistics, ...]:
        # Temporary scalar tuples, not retained Human/brain references. No diagnostic
        # reads of learning state (including OrderedDict recency) are necessary.
        values = {p: [] for p in PERCEPTIONS}
        for human in humans:
            if human.alive:
                self._cohort(human)  # Validate before indexing.
                g = human.genome
                values[g.perception_radius].append((g.signal_probability, g.hunger_multiplier,
                                                    g.thirst_multiplier, human.generation, human.age))
        rows = []
        names = ("signal_probability", "hunger_multiplier", "thirst_multiplier", "generation", "age")
        for p, samples in values.items():
            n = len(samples)
            means = {f"mean_{name}": fsum(s[i] for s in samples) / n if n else None
                     for i, name in enumerate(names)}
            rows.append(CohortStatistics(tick, p, living_population=n,
                                        living_agent_ticks=n if tick else 0, **self._counts[p], **means))
        return tuple(rows)


class CSVCohortWriter:
    """Callable streaming sink; the caller owns the file. Blank means = no living cohort."""
    def __init__(self, stream: TextIO, *, metadata: dict[str, object] | None = None) -> None:
        self.metadata = dict(metadata or {})
        self.columns = tuple(f.name for f in fields(CohortStatistics))
        if set(self.columns) & self.metadata.keys():
            raise ValueError("Cohort metadata must not replace statistics columns")
        self.writer = csv.writer(stream)
        self.writer.writerow((*self.metadata, *self.columns))

    def __call__(self, rows: tuple[CohortStatistics, ...]) -> None:
        for row in rows:
            self.writer.writerow((*self.metadata.values(), *(getattr(row, name) for name in self.columns)))
