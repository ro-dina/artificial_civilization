import csv
from dataclasses import asdict, dataclass, fields
from typing import TextIO


@dataclass(frozen=True, slots=True)
class Statistics:
    """Population is current; all consumption, emission, and death counts are cumulative."""

    tick: int
    population: int
    deaths: int = 0
    food_consumed: int = 0
    water_consumed: int = 0
    signals_emitted: int = 0


class CSVStatisticsWriter:
    """Stream snapshots to a caller-owned text file without retaining history."""

    def __init__(self, stream: TextIO) -> None:
        self._writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(Statistics)])
        self._writer.writeheader()

    def write(self, statistics: Statistics) -> None:
        self._writer.writerow(asdict(statistics))
