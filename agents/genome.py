"""Explicit, immutable traits. No brain memory or arbitrary object state is inherited."""

from dataclasses import dataclass, fields
from math import ceil, isfinite
from random import Random

from config import Config, GENOME_BOUNDS, PERCEPTION_RADIUS, SIGNAL_PROBABILITY


@dataclass(frozen=True, slots=True)
class Genome:
    perception_radius: int = PERCEPTION_RADIUS
    signal_probability: float = SIGNAL_PROBABILITY
    hunger_multiplier: float = 1
    thirst_multiplier: float = 1

    def __post_init__(self) -> None:
        for trait in fields(self):
            value = getattr(self, trait.name)
            low, high = GENOME_BOUNDS[trait.name]
            if type(value) not in (int, float) or not isfinite(value) or not low <= value <= high:
                raise ValueError(f"{trait.name} must be finite and in [{low}, {high}]")
        if type(self.perception_radius) is not int:
            raise ValueError("perception_radius must be an integer")

    @classmethod
    def founder(cls, config: Config) -> "Genome":
        return cls(perception_radius=config.perception_radius, signal_probability=config.signal_probability)

    @classmethod
    def inherit(cls, a: "Genome", b: "Genome", config: Config,
                inheritance_rng: Random, mutation_rng: Random) -> "Genome":
        values = {}
        for trait in fields(cls):
            name = trait.name
            value = inheritance_rng.choice((getattr(a, name), getattr(b, name)))
            if config.mutation_rate > 0 and config.mutation_strength > 0 and mutation_rng.random() < config.mutation_rate:
                low, high = GENOME_BOUNDS[name]
                extent = config.mutation_strength * (high - low)
                if name == "perception_radius":
                    value += mutation_rng.randint(-ceil(extent), ceil(extent))
                else:
                    value += mutation_rng.uniform(-extent, extent)
                value = max(low, min(high, value))
            values[name] = value
        return cls(**values)
