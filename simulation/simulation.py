from __future__ import annotations

from dataclasses import asdict, replace
from random import Random

from agents.brain import Action, Decision, RandomBrain
from agents.human import Human
from agents.observation import Observation, TileObservation
from config import Config, REPRODUCTIVE_TYPES
from simulation.events import Event, EventSink
from simulation.statistics import Statistics
from systems.communication import Communication
from world.world import World

MOVEMENT = {
    Action.MOVE_NORTH: (0, -1),
    Action.MOVE_SOUTH: (0, 1),
    Action.MOVE_EAST: (1, 0),
    Action.MOVE_WEST: (-1, 0),
}


class Simulation:
    def __init__(self, config: Config, *, event_sink: EventSink | None = None) -> None:
        self.config = config
        self.event_sink = event_sink
        seed = config.random_seed
        # Named streams isolate world generation, placement, ordering, and brains.
        self.world = World.generate(config, Random(f"{seed}:world"))
        spawn_rng = Random(f"{seed}:spawn")
        self._order_rng = Random(f"{seed}:order")
        self.agents = [
            Human(
                id=agent_id,
                x=spawn_rng.randrange(config.world_width),
                y=spawn_rng.randrange(config.world_height),
                brain=RandomBrain(Random(f"{seed}:brain:{agent_id}"), config),
                reproductive_type=spawn_rng.choice(REPRODUCTIVE_TYPES),
                energy=config.initial_energy,
            )
            for agent_id in range(config.initial_population)
        ]
        self.communication = Communication(config.signal_vocab_size, config.signal_range)
        self.tick = 0
        self.statistics = Statistics(tick=0, population=self.population)
        if self.event_sink is not None:
            self.event_sink(Event(0, "run_started", details={"config": asdict(config)}))

    @classmethod
    def create_default(
        cls, seed: int | None = None, *, config: Config | None = None,
        event_sink: EventSink | None = None,
    ) -> Simulation:
        settings = config if config is not None else Config()
        if seed is not None:
            settings = replace(settings, random_seed=seed)
        return cls(settings, event_sink=event_sink)

    @property
    def population(self) -> int:
        return len(self.agents)

    def observe(self, human: Human) -> Observation:
        tiles = tuple(
            TileObservation(x - human.x, y - human.y, tile.food, tile.water)
            for x, y, tile in self.world.nearby_tiles(human.x, human.y, self.config.perception_radius)
        )
        signals = self.communication.hear(human.id, human.x, human.y)
        return Observation(self.tick, tiles, signals)

    def _execute(self, human: Human, action: Action) -> tuple[bool, int, int]:
        """Return action success and the food/water units actually consumed."""
        if action in MOVEMENT:
            return self.world.move(human, *MOVEMENT[action]), 0, 0
        if action is Action.EAT:
            food = self.world.consume_food(human.x, human.y)
            human.hunger = max(0, human.hunger - food * self.config.food_hunger_relief)
            return bool(food), food, 0
        if action is Action.DRINK:
            water = self.world.consume_water(human.x, human.y)
            human.thirst = max(0, human.thirst - water * self.config.water_thirst_relief)
            return bool(water), 0, water
        return True, 0, 0  # WAIT

    def step(self) -> Statistics:
        """Advance once; an extinct simulation is a no-op. No history is retained."""
        if not self.agents:
            return self.statistics
        self.tick += 1
        order = list(self.agents)
        self._order_rng.shuffle(order)
        deaths = food_consumed = water_consumed = signals_emitted = 0

        for human in order:
            observation = self.observe(human)
            if self.event_sink is not None:
                start_x, start_y = human.x, human.y
                before = {"age": human.age, "hunger": human.hunger,
                          "thirst": human.thirst, "energy": human.energy}
            decision = human.brain.choose_action(human, observation)
            if not isinstance(decision, Decision) or not isinstance(decision.action, Action):
                raise ValueError("Brains must return a Decision containing an Action")
            if decision.signal_id is not None:
                self.communication.validate_signal(decision.signal_id)

            success, food, water = self._execute(human, decision.action)
            food_consumed += food
            water_consumed += water
            if decision.signal_id is not None:
                self.communication.emit(decision.signal_id, human.id, human.x, human.y)
                signals_emitted += 1
            cause = human.update_physiology(self.config)
            deaths += int(cause is not None)

            if self.event_sink is not None:
                self.event_sink(Event(
                    tick=self.tick, kind="agent_step", agent_id=human.id,
                    x=start_x, y=start_y,
                    details={
                        "before": before, "observation": observation,
                        "action": decision.action.value, "action_succeeded": success,
                        "signal_emitted": decision.signal_id,
                        "food_consumed": food, "water_consumed": water,
                        "after": {"x": human.x, "y": human.y, "age": human.age,
                                  "hunger": human.hunger, "thirst": human.thirst,
                                  "energy": human.energy, "alive": human.alive},
                        "death_cause": cause,
                    },
                ))

        self.agents = [human for human in self.agents if human.alive]
        self.communication.advance()
        previous = self.statistics
        self.statistics = Statistics(
            tick=self.tick, population=self.population,
            deaths=previous.deaths + deaths,
            food_consumed=previous.food_consumed + food_consumed,
            water_consumed=previous.water_consumed + water_consumed,
            signals_emitted=previous.signals_emitted + signals_emitted,
        )
        return self.statistics

    def print_status(self) -> None:
        stats = self.statistics
        print(
            f"tick={stats.tick} population={stats.population} deaths={stats.deaths} "
            f"food_consumed={stats.food_consumed} water_consumed={stats.water_consumed} "
            f"signals_emitted={stats.signals_emitted}"
        )
