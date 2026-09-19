"""Simulation parameters; ticks and resource units are deliberately abstract."""

from dataclasses import dataclass, fields
from math import isfinite

WORLD_WIDTH = 40
WORLD_HEIGHT = 30
INITIAL_POPULATION = 100
INITIAL_FOOD = 600
INITIAL_WATER = 600
PERCEPTION_RADIUS = 2
MAX_HUNGER = 100
MAX_THIRST = 80
MAX_AGE = 1000
HUNGER_PER_TICK = 1
THIRST_PER_TICK = 1
FOOD_HUNGER_RELIEF = 30
WATER_THIRST_RELIEF = 30
INITIAL_ENERGY = 100
REPRODUCTIVE_TYPES = (0, 1)
SIGNAL_VOCAB_SIZE = 16
SIGNAL_RANGE = 3
SIGNAL_PROBABILITY = 0.1
RANDOM_SEED = 42
MAX_TICKS = 1000
STATUS_INTERVAL = 100


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable per-run settings, so independent runs share no mutable config."""

    world_width: int = WORLD_WIDTH
    world_height: int = WORLD_HEIGHT
    initial_population: int = INITIAL_POPULATION
    initial_food: int = INITIAL_FOOD
    initial_water: int = INITIAL_WATER
    perception_radius: int = PERCEPTION_RADIUS
    max_hunger: float = MAX_HUNGER
    max_thirst: float = MAX_THIRST
    max_age: int | None = MAX_AGE
    hunger_per_tick: float = HUNGER_PER_TICK
    thirst_per_tick: float = THIRST_PER_TICK
    food_hunger_relief: float = FOOD_HUNGER_RELIEF
    water_thirst_relief: float = WATER_THIRST_RELIEF
    initial_energy: float = INITIAL_ENERGY
    signal_vocab_size: int = SIGNAL_VOCAB_SIZE
    signal_range: int = SIGNAL_RANGE
    signal_probability: float = SIGNAL_PROBABILITY
    random_seed: int = RANDOM_SEED
    max_ticks: int = MAX_TICKS
    status_interval: int = STATUS_INTERVAL

    def __post_init__(self) -> None:
        integer_fields = {
            "world_width", "world_height", "initial_population", "initial_food",
            "initial_water", "perception_radius", "max_age", "signal_vocab_size",
            "signal_range", "random_seed", "max_ticks", "status_interval",
        }
        positive_fields = {
            "world_width", "world_height", "max_hunger", "max_thirst",
            "max_age", "signal_vocab_size", "status_interval",
        }
        for field in fields(self):
            name, value = field.name, getattr(self, field.name)
            if name == "max_age" and value is None:
                continue
            if name in integer_fields:
                if type(value) is not int:
                    raise ValueError(f"{name} must be an integer")
            elif type(value) not in (int, float) or not isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if name == "random_seed":
                continue
            if value < 0 or (name in positive_fields and value == 0):
                raise ValueError(f"{name} must be {'positive' if name in positive_fields else 'nonnegative'}")
        if self.signal_probability > 1:
            raise ValueError("signal_probability must be between 0 and 1")
