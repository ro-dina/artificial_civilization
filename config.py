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

# Trait limits are experiment constraints, not inherited state.
GENOME_BOUNDS = {
    "perception_radius": (0, 8),
    "signal_probability": (0, 1),
    "hunger_multiplier": (0.5, 1.5),
    "thirst_multiplier": (0.5, 1.5),
}
REPRODUCTION_ENABLED = False
MIN_REPRODUCTIVE_AGE = 20
MAX_REPRODUCTIVE_AGE = None
REPRODUCTION_RADIUS = 1
REPRODUCTION_COOLDOWN = 20
REPRODUCTION_ENERGY_COST = 30
MIN_REPRODUCTION_ENERGY = 35
REPRODUCTION_NEED_FRACTION = 0.75
OFFSPRING_HUNGER = 0
OFFSPRING_THIRST = 0
OFFSPRING_ENERGY = 40
MAX_ENERGY = 100
FOOD_ENERGY_GAIN = 0
ENERGY_PER_TICK = 0
ENERGY_DEPLETION_LETHAL = False
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.05
FOOD_REGENERATION_PROBABILITY = 0
WATER_REGENERATION_PROBABILITY = 0
FOOD_CAPACITY = None
WATER_CAPACITY = None
MAX_POPULATION = None
BRAIN = "random"
LEARNING_RATE = 0.2
LEARNING_DISCOUNT = 0.9
LEARNING_EPSILON = 0.2
LEARNING_MEMORY_CAPACITY = 256
LEARNING_NEED_BINS = 3


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
    reproduction_enabled: bool = REPRODUCTION_ENABLED
    min_reproductive_age: int = MIN_REPRODUCTIVE_AGE
    max_reproductive_age: int | None = MAX_REPRODUCTIVE_AGE
    reproduction_radius: int = REPRODUCTION_RADIUS
    reproduction_cooldown: int = REPRODUCTION_COOLDOWN
    reproduction_energy_cost: float = REPRODUCTION_ENERGY_COST
    min_reproduction_energy: float = MIN_REPRODUCTION_ENERGY
    reproduction_need_fraction: float = REPRODUCTION_NEED_FRACTION
    offspring_hunger: float = OFFSPRING_HUNGER
    offspring_thirst: float = OFFSPRING_THIRST
    offspring_energy: float = OFFSPRING_ENERGY
    max_energy: float = MAX_ENERGY
    food_energy_gain: float = FOOD_ENERGY_GAIN
    energy_per_tick: float = ENERGY_PER_TICK
    energy_depletion_lethal: bool = ENERGY_DEPLETION_LETHAL
    mutation_rate: float = MUTATION_RATE
    mutation_strength: float = MUTATION_STRENGTH
    food_regeneration_probability: float = FOOD_REGENERATION_PROBABILITY
    water_regeneration_probability: float = WATER_REGENERATION_PROBABILITY
    food_capacity: int | None = FOOD_CAPACITY
    water_capacity: int | None = WATER_CAPACITY
    max_population: int | None = MAX_POPULATION
    brain: str = BRAIN
    learning_rate: float = LEARNING_RATE
    learning_discount: float = LEARNING_DISCOUNT
    learning_epsilon: float = LEARNING_EPSILON
    learning_memory_capacity: int = LEARNING_MEMORY_CAPACITY
    learning_need_bins: int = LEARNING_NEED_BINS

    @classmethod
    def evolution(cls, **overrides) -> "Config":
        """A renewable ecology; extinction is still possible. Config() is legacy."""
        settings = dict(
            reproduction_enabled=True, food_energy_gain=20, energy_per_tick=0.2,
            energy_depletion_lethal=True, food_regeneration_probability=0.02,
            water_regeneration_probability=0.02, food_capacity=5, water_capacity=5,
        )
        settings.update(overrides)
        return cls(**settings)

    def __post_init__(self) -> None:
        integer_fields = {
            "world_width", "world_height", "initial_population", "initial_food",
            "initial_water", "perception_radius", "max_age", "signal_vocab_size",
            "signal_range", "random_seed", "max_ticks", "status_interval",
            "min_reproductive_age", "max_reproductive_age", "reproduction_radius",
            "reproduction_cooldown", "food_capacity", "water_capacity", "max_population",
            "learning_memory_capacity", "learning_need_bins",
        }
        positive_fields = {
            "world_width", "world_height", "max_hunger", "max_thirst",
            "max_age", "signal_vocab_size", "status_interval",
            "reproduction_energy_cost", "max_energy", "food_capacity", "water_capacity",
            "learning_rate", "learning_epsilon", "learning_memory_capacity", "learning_need_bins",
        }
        optional_fields = {"max_age", "max_reproductive_age", "food_capacity", "water_capacity", "max_population"}
        boolean_fields = {"reproduction_enabled", "energy_depletion_lethal"}
        for field in fields(self):
            name, value = field.name, getattr(self, field.name)
            if name == "brain":
                if value not in ("random", "learning"):
                    raise ValueError("brain must be 'random' or 'learning'")
                continue
            if name in optional_fields and value is None:
                continue
            if name in boolean_fields:
                if type(value) is not bool:
                    raise ValueError(f"{name} must be a boolean")
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
        for name in ("signal_probability", "mutation_rate", "mutation_strength", "reproduction_need_fraction",
                     "food_regeneration_probability", "water_regeneration_probability",
                     "learning_rate", "learning_discount", "learning_epsilon"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.perception_radius > GENOME_BOUNDS["perception_radius"][1]:
            raise ValueError("perception_radius exceeds the genome limit in GENOME_BOUNDS")
        if self.max_reproductive_age is not None and self.max_reproductive_age < self.min_reproductive_age:
            raise ValueError("max_reproductive_age must be at least min_reproductive_age")
        for name in ("initial_energy", "offspring_energy"):
            if getattr(self, name) > self.max_energy:
                raise ValueError(f"{name} must not exceed max_energy")
        if self.max_population is not None and self.initial_population > self.max_population:
            raise ValueError("initial_population exceeds the computational max_population safeguard")
        for resource in ("food", "water"):
            cap = getattr(self, f"{resource}_capacity")
            if getattr(self, f"{resource}_regeneration_probability") > 0 and cap is None:
                raise ValueError(f"{resource}_capacity is required for regeneration")
            if cap is not None and getattr(self, f"initial_{resource}") > self.world_width * self.world_height * cap:
                raise ValueError(f"initial_{resource} exceeds total tile capacity")
