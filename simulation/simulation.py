from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from random import Random

from agents.brain import Action, Brain, Decision, RandomBrain
from agents.genome import Genome
from agents.human import Human
from agents.learning import LearningBrain, Physiology
from agents.observation import Observation, TileObservation
from config import Config, REPRODUCTIVE_TYPES
from simulation.events import Event, EventSink
from simulation.statistics import Statistics, summarize_genomes, summarize_learning
from systems.communication import Communication
from systems.reproduction import Reproduction
from world.world import World

MOVEMENT = {
    Action.MOVE_NORTH: (0, -1),
    Action.MOVE_SOUTH: (0, 1),
    Action.MOVE_EAST: (1, 0),
    Action.MOVE_WEST: (-1, 0),
}


class Simulation:
    def __init__(self, config: Config, *, event_sink: EventSink | None = None,
                 brain_factory: Callable[[Random, Config], Brain] | None = None) -> None:
        self.config = config
        self.event_sink = event_sink
        if brain_factory is None:
            brain_factory = LearningBrain if config.brain == "learning" else RandomBrain
        self.brain_factory = brain_factory
        seed = config.random_seed
        # Named streams isolate world generation, placement, ordering, and brains.
        self.world = World.generate(config, Random(f"{seed}:world"))
        spawn_rng = Random(f"{seed}:spawn")
        self._order_rng = Random(f"{seed}:order")
        self._offspring_rng = Random(f"{seed}:offspring")
        self._inheritance_rng = Random(f"{seed}:inheritance")
        self._mutation_rng = Random(f"{seed}:mutation")
        self._food_rng = Random(f"{seed}:regeneration:food")
        self._water_rng = Random(f"{seed}:regeneration:water")
        self.reproduction = Reproduction(config, Random(f"{seed}:partners"))
        founder_genome = Genome.founder(config)
        self.agents = [
            Human(
                id=agent_id,
                x=spawn_rng.randrange(config.world_width),
                y=spawn_rng.randrange(config.world_height),
                brain=brain_factory(Random(f"{seed}:brain:{agent_id}"), config),
                reproductive_type=spawn_rng.choice(REPRODUCTIVE_TYPES),
                energy=config.initial_energy,
                genome=founder_genome,
            )
            for agent_id in range(config.initial_population)
        ]
        self._next_id = config.initial_population
        self.communication = Communication(config.signal_vocab_size, config.signal_range)
        self.tick = 0
        self.statistics = Statistics(tick=0, population=self.population,
                                     **summarize_genomes(self.agents), **summarize_learning(self.agents))
        if self.event_sink is not None:
            self.event_sink(Event(0, "run_started", details={"config": asdict(config), "founder_genome": founder_genome}))

    @classmethod
    def create_default(
        cls, seed: int | None = None, *, config: Config | None = None,
        event_sink: EventSink | None = None,
        brain_factory: Callable[[Random, Config], Brain] | None = None,
    ) -> Simulation:
        settings = config if config is not None else Config()
        if seed is not None:
            settings = replace(settings, random_seed=seed)
        return cls(settings, event_sink=event_sink, brain_factory=brain_factory)

    @property
    def population(self) -> int:
        return len(self.agents)

    def observe(self, human: Human) -> Observation:
        tiles = tuple(
            TileObservation(x - human.x, y - human.y, tile.food, tile.water)
            for x, y, tile in self.world.nearby_tiles(human.x, human.y, human.genome.perception_radius)
        )
        signals = self.communication.hear(human.id, human.x, human.y)
        return Observation(self.tick, tiles, signals)

    def _execute(self, human: Human, action: Action) -> tuple[bool, int, int]:
        """Return action success and the food/water units actually consumed."""
        if action in MOVEMENT:
            x, y = human.x, human.y
            moved = self.world.move(human, *MOVEMENT[action])
            if moved and self.config.reproduction_enabled:
                self.reproduction.moved(human, x, y)
            return moved, 0, 0
        if action is Action.EAT:
            food = self.world.consume_food(human.x, human.y)
            human.hunger = max(0, human.hunger - food * self.config.food_hunger_relief)
            human.energy = min(self.config.max_energy, human.energy + food * self.config.food_energy_gain)
            return bool(food), food, 0
        if action is Action.DRINK:
            water = self.world.consume_water(human.x, human.y)
            human.thirst = max(0, human.thirst - water * self.config.water_thirst_relief)
            return bool(water), 0, water
        return True, 0, 0  # WAIT

    def _offspring(self, a: Human, b: Human) -> Human:
        child_id = self._next_id
        self._next_id += 1
        return Human(
            id=child_id, x=a.x, y=a.y,
            brain=self.brain_factory(Random(f"{self.config.random_seed}:brain:{child_id}"), self.config),
            reproductive_type=self._offspring_rng.choice(REPRODUCTIVE_TYPES),
            hunger=self.config.offspring_hunger, thirst=self.config.offspring_thirst,
            energy=self.config.offspring_energy,
            genome=Genome.inherit(a.genome, b.genome, self.config, self._inheritance_rng, self._mutation_rng),
            parent_a_id=a.id, parent_b_id=b.id,
            generation=max(a.generation, b.generation) + 1, birth_tick=self.tick,
        )

    def step(self) -> Statistics:
        """Advance once; an extinct simulation is a no-op. No history is retained."""
        if not self.agents:
            return self.statistics
        self.tick += 1
        food_regenerated, water_regenerated = self.world.regenerate(self.config, self._food_rng, self._water_rng)
        order = list(self.agents)
        self._order_rng.shuffle(order)
        if self.config.reproduction_enabled:
            self.reproduction.begin_tick(order)
        deaths = food_consumed = water_consumed = signals_emitted = 0
        reproduction_attempts = population_limit_blocks = 0
        learning_updates = exploratory_actions = exploitative_actions = 0
        death_causes = {"starvation": 0, "dehydration": 0, "old_age": 0, "energy": 0}
        newborns = []

        for human in order:
            learner = human.brain if isinstance(human.brain, LearningBrain) else None
            previous_updates = learner.learning_updates if learner is not None else 0
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

            reproduction_failure = None
            if decision.action is Action.REPRODUCE:
                reproduction_attempts += 1
                partner = self.reproduction.partner_for(human, self.tick)
                success, food, water = False, 0, 0
                if partner is None:
                    reproduction_failure = "ineligible_or_no_partner"
                elif (self.config.max_population is not None
                      and self.population - deaths + len(newborns) >= self.config.max_population):
                    population_limit_blocks += 1
                    reproduction_failure = "population_limit"
                else:
                    newborns.append(self._offspring(human, partner))
                    self.reproduction.charge(human, partner, self.tick)
                    success = True
            else:
                success, food, water = self._execute(human, decision.action)
            food_consumed += food
            water_consumed += water
            if decision.signal_id is not None:
                self.communication.emit(decision.signal_id, human.id, human.x, human.y)
                signals_emitted += 1
            cause = human.update_physiology(self.config)
            deaths += int(cause is not None)
            if cause is not None:
                death_causes[cause] += 1
            observe_outcome = getattr(human.brain, "observe_outcome", None)
            if observe_outcome is not None:
                observe_outcome(Physiology.capture(human))
            if learner is not None:
                learning_updates += learner.learning_updates - previous_updates
                exploratory_actions += int(learner.last_exploratory)
                exploitative_actions += int(not learner.last_exploratory)

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
                        "reproduction_failure": reproduction_failure,
                        **({"learning": {"reward": learner.last_reward,
                                         "exploratory": learner.last_exploratory,
                                         "updates": learner.learning_updates,
                                         "memory_size": learner.memory_size}} if learner is not None else {}),
                    },
                ))

        self.agents = [human for human in self.agents if human.alive]
        self.agents.extend(newborns)
        self.reproduction.clear()
        if self.event_sink is not None:
            for child in newborns:
                self.event_sink(Event(
                    tick=self.tick, kind="birth", agent_id=child.id, x=child.x, y=child.y,
                    details={"child_id": child.id, "parent_a_id": child.parent_a_id,
                             "parent_b_id": child.parent_b_id, "generation": child.generation,
                             "child_genome": child.genome, "reproductive_type": child.reproductive_type,
                             "hunger": child.hunger, "thirst": child.thirst, "energy": child.energy},
                ))
        self.communication.advance()
        previous = self.statistics
        self.statistics = Statistics(
            tick=self.tick, population=self.population,
            total_learning_updates=previous.total_learning_updates + learning_updates,
            exploratory_actions=previous.exploratory_actions + exploratory_actions,
            exploitative_actions=previous.exploitative_actions + exploitative_actions,
            deaths=previous.deaths + deaths,
            food_consumed=previous.food_consumed + food_consumed,
            water_consumed=previous.water_consumed + water_consumed,
            signals_emitted=previous.signals_emitted + signals_emitted,
            births=previous.births + len(newborns),
            highest_generation=max(previous.highest_generation, max((h.generation for h in newborns), default=0)),
            reproduction_attempts=previous.reproduction_attempts + reproduction_attempts,
            successful_reproductions=previous.successful_reproductions + len(newborns),
            population_limit_blocks=previous.population_limit_blocks + population_limit_blocks,
            food_regenerated=previous.food_regenerated + food_regenerated,
            water_regenerated=previous.water_regenerated + water_regenerated,
            starvation_deaths=previous.starvation_deaths + death_causes["starvation"],
            dehydration_deaths=previous.dehydration_deaths + death_causes["dehydration"],
            old_age_deaths=previous.old_age_deaths + death_causes["old_age"],
            energy_deaths=previous.energy_deaths + death_causes["energy"],
            **summarize_genomes(self.agents),
            **summarize_learning(self.agents),
        )
        assert self.population == self.config.initial_population + self.statistics.births - self.statistics.deaths
        return self.statistics

    def print_status(self) -> None:
        stats = self.statistics
        print(
            f"tick={stats.tick} population={stats.population} deaths={stats.deaths} "
            f"food_consumed={stats.food_consumed} water_consumed={stats.water_consumed} "
            f"signals_emitted={stats.signals_emitted} births={stats.births} "
            f"highest_generation={stats.highest_generation} reproduction_attempts={stats.reproduction_attempts} "
            f"population_limit_blocks={stats.population_limit_blocks}"
        )
