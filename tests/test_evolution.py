from dataclasses import FrozenInstanceError, asdict, fields, replace
import io
import json
import random
import unittest

from agents.brain import Action, Decision, RandomBrain
from agents.genome import Genome
from config import Config, GENOME_BOUNDS
from simulation.events import JSONLEventWriter
from simulation.simulation import Simulation
from systems.reproduction import Reproduction
from world.world import World


class ConstantBrain:
    def __init__(self, action=Action.WAIT):
        self.action = action
        self.calls = 0

    def choose_action(self, human, observation):
        self.calls += 1
        return Decision(self.action)


def pair(**overrides):
    settings = dict(
        world_width=5, world_height=5, initial_population=2, initial_food=0,
        initial_water=0, reproduction_enabled=True, reproduction_energy_cost=10,
        min_reproduction_energy=10, reproduction_cooldown=3, max_age=None,
        hunger_per_tick=0, thirst_per_tick=0, offspring_energy=20,
    )
    settings.update(overrides)
    sim = Simulation(Config(**settings), brain_factory=lambda rng, cfg: ConstantBrain())
    for index, human in enumerate(sim.agents):
        human.x, human.y = 1, 1
        human.reproductive_type = index % 2
        human.age = sim.config.min_reproductive_age
        human.brain = ConstantBrain(Action.REPRODUCE)
    return sim


def snapshot(sim):
    return (
        sim.statistics, sim._next_id,
        tuple(tuple(getattr(h, f.name) for f in fields(h) if f.name != "brain") for h in sim.agents),
        tuple((t.food, t.water) for row in sim.world.tiles for t in row),
        sim.communication._audible, sim.communication._pending,
    )


class ReproductionTests(unittest.TestCase):
    def test_compatible_parents_create_child_with_lineage_and_cost(self):
        sim = pair(offspring_hunger=3, offspring_thirst=4, offspring_energy=25)
        parents = list(sim.agents)
        parents[0].generation, parents[1].generation = 2, 5
        stats = sim.step()
        child = sim.agents[-1]
        self.assertEqual((stats.population, stats.births, stats.successful_reproductions), (3, 1, 1))
        self.assertEqual(stats.reproduction_attempts, 2)
        self.assertEqual(child.id, 2)
        self.assertEqual({child.parent_a_id, child.parent_b_id}, {0, 1})
        self.assertEqual((child.generation, child.birth_tick, child.age), (6, 1, 0))
        self.assertEqual((child.hunger, child.thirst, child.energy), (3, 4, 25))
        self.assertEqual((child.x, child.y), (1, 1))
        self.assertIn(child.reproductive_type, (0, 1))
        self.assertTrue(all(h.energy == 90 for h in parents))
        self.assertEqual(stats.highest_generation, 6)

    def test_newborn_waits_until_next_tick_and_has_fresh_brain(self):
        sim = pair()
        sim.step()
        child = sim.agents[-1]
        self.assertEqual((child.age, child.brain.calls), (0, 0))
        self.assertTrue(all(child.brain is not h.brain for h in sim.agents[:-1]))
        sim.step()
        self.assertEqual((child.age, child.brain.calls), (1, 1))

    def test_incompatible_types_and_self_reproduction_fail(self):
        for population in (1, 2):
            sim = pair(initial_population=population)
            for h in sim.agents:
                h.reproductive_type = 0
            self.assertEqual(sim.step().births, 0)
            self.assertTrue(all(h.energy == 100 and h.next_reproduction_tick == 0 for h in sim.agents))

    def test_both_parents_must_be_eligible(self):
        for field, value in (("age", 18), ("energy", 9), ("hunger", 76), ("thirst", 61),
                             ("next_reproduction_tick", 10)):
            with self.subTest(field=field):
                sim = pair()
                setattr(sim.agents[0], field, value)
                self.assertEqual(sim.step().births, 0)

    def test_reproductive_age_bounds_are_inclusive_and_optional(self):
        sim = pair(max_reproductive_age=30)
        human = sim.agents[0]
        for age, expected in ((19, False), (20, True), (30, True), (31, False)):
            human.age = age
            self.assertEqual(sim.reproduction.eligible(human, 1), expected)
        unlimited = pair(max_reproductive_age=None)
        unlimited.agents[0].age = 10000
        self.assertTrue(unlimited.reproduction.eligible(unlimited.agents[0], 1))

    def test_dead_partner_is_ineligible(self):
        sim = pair()
        sim.agents[1].alive = False
        sim.reproduction.begin_tick(sim.agents)
        self.assertIsNone(sim.reproduction.partner_for(sim.agents[0], 1))

    def test_distance_boundary_and_out_of_range(self):
        for position, births in (((2, 2), 1), ((3, 1), 0)):
            sim = pair(reproduction_radius=1)
            sim.agents[1].x, sim.agents[1].y = position
            self.assertEqual(sim.step().births, births)
        sim = pair(reproduction_radius=0)
        self.assertEqual(sim.step().births, 1)

    def test_cooldown_prevents_repeated_pairing(self):
        sim = pair(reproduction_cooldown=3)
        self.assertEqual(sim.step().births, 1)
        self.assertEqual(sim.step().births, 1)
        self.assertEqual(sim.step().births, 1)
        self.assertEqual(sim.step().births, 2)

    def test_zero_cooldown_still_allows_only_one_pairing_per_tick(self):
        sim = pair(reproduction_cooldown=0)
        self.assertEqual(sim.step().births, 1)
        self.assertEqual(sim.step().births, 2)

    def test_no_parent_can_pair_twice_even_with_multiple_partners(self):
        sim = pair(initial_population=4, reproduction_cooldown=0)
        self.assertEqual(sim.step().births, 2)
        self.assertEqual([h.energy for h in sim.agents[:4]], [90] * 4)

    def test_queued_birth_survives_parents_dying_in_same_tick(self):
        sim = pair(max_age=21)
        stats = sim.step()
        self.assertEqual((stats.population, stats.births, stats.deaths), (1, 1, 2))
        self.assertEqual((sim.agents[0].age, sim.agents[0].generation), (0, 1))

    def test_reproduction_cost_can_increase_energy_mortality(self):
        sim = pair(initial_energy=70, reproduction_energy_cost=60,
                   min_reproduction_energy=0, energy_per_tick=20, energy_depletion_lethal=True)
        stats = sim.step()
        self.assertEqual((stats.births, stats.energy_deaths, stats.population), (1, 2, 1))

    def test_insufficient_energy_does_not_charge_either_parent(self):
        sim = pair(reproduction_energy_cost=30, min_reproduction_energy=0)
        sim.agents[0].energy = 29
        self.assertEqual(sim.step().births, 0)
        self.assertEqual([h.energy for h in sim.agents], [29, 100])

    def test_energy_lethal_mode_requires_positive_reserve_after_cost(self):
        sim = pair(energy_depletion_lethal=True)
        human = sim.agents[0]
        human.energy = 10
        self.assertFalse(sim.reproduction.eligible(human, 1))
        human.energy = 10.1
        self.assertTrue(sim.reproduction.eligible(human, 1))

    def test_disabled_reproduction_fails_even_for_custom_brain(self):
        sim = pair(reproduction_enabled=False)
        self.assertEqual(sim.step().births, 0)
        self.assertEqual(sim.statistics.reproduction_attempts, 2)

    def test_population_safeguard_counts_blocks_without_charging(self):
        sim = pair(max_population=2)
        stats = sim.step()
        self.assertEqual((stats.population, stats.births, stats.population_limit_blocks), (2, 0, 2))
        self.assertTrue(all(h.energy == 100 and h.next_reproduction_tick == 0 for h in sim.agents))

    def test_pending_births_count_toward_population_safeguard(self):
        sim = pair(initial_population=4, max_population=5)
        stats = sim.step()
        self.assertEqual((stats.births, stats.population), (1, 5))
        self.assertGreater(stats.population_limit_blocks, 0)

    def test_death_earlier_in_tick_releases_safeguard_space(self):
        sim = pair(initial_population=3, max_population=3)
        sim._order_rng.shuffle = lambda order: None
        sim.agents[0].hunger = sim.config.max_hunger
        sim.agents[0].brain = ConstantBrain()
        stats = sim.step()
        self.assertEqual((stats.deaths, stats.births, stats.population), (1, 1, 3))

    def test_partner_index_tracks_movement_between_buckets(self):
        sim = pair(reproduction_radius=1)
        a, b = sim.agents
        b.x = 4
        sim.reproduction.begin_tick(sim.agents)
        self.assertIsNone(sim.reproduction.partner_for(a, 1))
        old_x = b.x
        b.x = 2
        sim.reproduction.moved(b, old_x, b.y)
        self.assertIs(sim.reproduction.partner_for(a, 1), b)
        b.alive = False
        self.assertIsNone(sim.reproduction.partner_for(a, 1))

    def test_partner_choice_is_seeded_and_not_always_lowest_id(self):
        sim = pair(initial_population=4)
        sim.agents[2].reproductive_type = 1
        choices = []
        for seed in range(20):
            system = Reproduction(sim.config, random.Random(seed))
            system.begin_tick(sim.agents)
            choices.append(system.partner_for(sim.agents[0], 1).id)
        self.assertEqual(set(choices), {1, 2, 3})

    def test_birth_events_reconstruct_lineage_and_ids_are_never_reused(self):
        events = []
        config = Config(
            world_width=1, world_height=1, initial_population=6, initial_food=0, initial_water=0,
            reproduction_enabled=True, min_reproductive_age=0, reproduction_cooldown=1,
            reproduction_energy_cost=1, min_reproduction_energy=1, max_age=3,
            hunger_per_tick=0, thirst_per_tick=0, max_population=30,
        )
        sim = Simulation(config, event_sink=events.append,
                         brain_factory=lambda rng, cfg: ConstantBrain(Action.REPRODUCE))
        for _ in range(12):
            stats = sim.step()
            self.assertEqual(len({h.id for h in sim.agents}), sim.population)
            self.assertEqual(config.initial_population + stats.births - stats.deaths, sim.population)
            self.assertFalse(sim.reproduction._buckets)
        births = [event for event in events if event.kind == "birth"]
        ids = [event.agent_id for event in births]
        self.assertEqual(ids, list(range(6, 6 + len(ids))))
        self.assertGreater(sim.statistics.deaths, 0)
        self.assertGreater(sim.statistics.highest_generation, 1)
        lineage = {agent_id: 0 for agent_id in range(6)}
        for event in births:
            data = event.details
            generation = max(lineage[data["parent_a_id"]], lineage[data["parent_b_id"]]) + 1
            self.assertEqual(data["generation"], generation)
            self.assertEqual(data["child_id"], event.agent_id)
            self.assertIsInstance(data["child_genome"], Genome)
            lineage[event.agent_id] = generation
            self.assertTrue(sim.world.is_valid(event.x, event.y))


class GenomeTests(unittest.TestCase):
    def test_founders_share_identical_immutable_baseline_and_metadata(self):
        sim = Simulation(Config(initial_population=3))
        for human in sim.agents:
            self.assertEqual(human.genome, Genome.founder(sim.config))
            self.assertEqual((human.parent_a_id, human.parent_b_id, human.generation, human.birth_tick), (None, None, 0, 0))
        with self.assertRaises(FrozenInstanceError):
            sim.agents[0].genome.signal_probability = 0.4

    def test_unmutated_traits_each_come_from_one_parent(self):
        a, b = Genome(0, 0, 0.5, 0.5), Genome(8, 1, 1.5, 1.5)
        config = Config(mutation_rate=0)
        mutation_rng = random.Random(2)
        before = mutation_rng.getstate()
        mixed = False
        for seed in range(30):
            child = Genome.inherit(a, b, config, random.Random(seed), mutation_rng)
            for name in GENOME_BOUNDS:
                self.assertIn(getattr(child, name), (getattr(a, name), getattr(b, name)))
            mixed |= child not in (a, b)
        self.assertTrue(mixed)
        self.assertEqual(mutation_rng.getstate(), before)

    def test_mutations_replay_and_different_seeds_can_differ(self):
        config = Config(mutation_rate=1, mutation_strength=0.1)
        def child(seed):
            return Genome.inherit(Genome(), Genome(), config, random.Random(1), random.Random(seed))
        self.assertEqual(child(123), child(123))
        self.assertNotEqual(child(123), child(124))
        self.assertNotEqual(child(123), Genome())

    def test_mutations_stay_in_bounds_including_maximum_strength(self):
        config = Config(mutation_rate=1, mutation_strength=1)
        a, b = Genome(), Genome(8, 1, 1.5, 0.5)
        inheritance, mutation = random.Random(1), random.Random(2)
        for _ in range(300):
            child = Genome.inherit(a, b, config, inheritance, mutation)
            for name, (low, high) in GENOME_BOUNDS.items():
                self.assertLessEqual(low, getattr(child, name))
                self.assertLessEqual(getattr(child, name), high)
            self.assertIs(type(child.perception_radius), int)
            a, b = b, child

    def test_zero_strength_keeps_inherited_traits_and_does_not_draw_mutation_rng(self):
        rng = random.Random(1)
        before = rng.getstate()
        child = Genome.inherit(Genome(), Genome(), Config(mutation_rate=1, mutation_strength=0), random.Random(2), rng)
        self.assertEqual(child, Genome())
        self.assertEqual(rng.getstate(), before)

    def test_invalid_traits_are_rejected(self):
        for settings in ({"perception_radius": -1}, {"perception_radius": 9}, {"perception_radius": 1.2},
                         {"signal_probability": 1.1}, {"hunger_multiplier": 0},
                         {"thirst_multiplier": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Genome(**settings)

    def test_invalid_reproductive_types_are_rejected(self):
        human = pair().agents[0]
        for value in (-1, 2, True, 0.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(human, reproductive_type=value)

    def test_traits_affect_perception_physiology_and_random_signalling(self):
        sim = pair(hunger_per_tick=2, thirst_per_tick=2)
        human = sim.agents[0]
        human.genome = Genome(0, 1, 0.5, 1.5)
        human.brain = RandomBrain(random.Random(42), sim.config)
        observation = sim.observe(human)
        self.assertEqual(len(observation.tiles), 1)
        self.assertIsNotNone(human.brain.choose_action(human, observation).signal_id)
        human.update_physiology(sim.config)
        self.assertEqual((human.hunger, human.thirst), (1, 3))


class RenewableResourceTests(unittest.TestCase):
    def test_initial_placement_and_regeneration_respect_caps(self):
        config = Config(world_width=2, world_height=2, initial_food=8, initial_water=12,
                        food_capacity=2, water_capacity=3,
                        food_regeneration_probability=1, water_regeneration_probability=1)
        world = World.generate(config, random.Random(1))
        self.assertTrue(all(t.food == 2 and t.water == 3 for row in world.tiles for t in row))
        world.consume_food(0, 0)
        world.consume_water(1, 1)
        self.assertEqual(world.regenerate(config, random.Random(2), random.Random(3)), (1, 1))
        self.assertEqual(world.regenerate(config, random.Random(2), random.Random(3)), (0, 0))
        self.assertTrue(all(t.food == 2 and t.water == 3 for row in world.tiles for t in row))

    def test_empty_tiles_gain_at_most_one_unit_per_tick(self):
        config = Config(world_width=2, world_height=2, initial_food=0, initial_water=0,
                        food_capacity=2, water_capacity=3,
                        food_regeneration_probability=1, water_regeneration_probability=1)
        world = World.generate(config, random.Random(1))
        for food, water in ((1, 1), (2, 2), (2, 3), (2, 3)):
            world.regenerate(config, random.Random(2), random.Random(3))
            self.assertTrue(all(t.food == food and t.water == water for row in world.tiles for t in row))

    def test_regeneration_can_be_disabled_without_random_draws(self):
        config = Config(initial_food=0, initial_water=0)
        world = World.generate(config, random.Random(1))
        food_rng, water_rng = random.Random(2), random.Random(3)
        before = food_rng.getstate(), water_rng.getstate()
        self.assertEqual(world.regenerate(config, food_rng, water_rng), (0, 0))
        self.assertEqual((food_rng.getstate(), water_rng.getstate()), before)
        self.assertTrue(all(t.food == 0 and t.water == 0 for row in world.tiles for t in row))

    def test_seeded_regeneration_replays_and_resource_streams_are_independent(self):
        config = Config.evolution(world_width=4, world_height=4, initial_food=0, initial_water=0,
                                  food_regeneration_probability=0.2, water_regeneration_probability=0.3)
        worlds = [World.generate(config, random.Random(1)) for _ in range(3)]
        rngs = [(random.Random(2), random.Random(3)) for _ in worlds]
        for _ in range(10):
            for index, world in enumerate(worlds):
                cfg = replace(config, food_regeneration_probability=0) if index == 2 else config
                world.regenerate(cfg, *rngs[index])
            self.assertEqual(worlds[0].tiles, worlds[1].tiles)
            self.assertEqual([t.water for row in worlds[0].tiles for t in row],
                             [t.water for row in worlds[2].tiles for t in row])

    def test_regeneration_precedes_actions_and_statistics_count_actual_units(self):
        sim = Simulation(Config(world_width=1, world_height=1, initial_population=1,
                                initial_food=0, initial_water=0, food_capacity=1,
                                food_regeneration_probability=1),
                         brain_factory=lambda rng, cfg: ConstantBrain(Action.EAT))
        sim.step()
        self.assertEqual((sim.statistics.food_consumed, sim.statistics.food_regenerated), (1, 1))
        self.assertEqual(sim.world.tile_at(0, 0).food, 0)


class EvolutionSimulationTests(unittest.TestCase):
    @staticmethod
    def ecology():
        return Config.evolution(world_width=5, world_height=5, initial_population=12,
                                initial_food=20, initial_water=20, min_reproductive_age=4,
                                reproduction_cooldown=5, mutation_rate=0.5)

    def test_complete_multigeneration_replay_and_logging_independence(self):
        global_before = random.getstate()
        stream = io.StringIO()
        first = Simulation(self.ecology())
        second = Simulation(self.ecology(), event_sink=JSONLEventWriter(stream))
        for _ in range(180):
            first.step()
            second.step()
            self.assertEqual(snapshot(first), snapshot(second))
            stats = first.statistics
            self.assertEqual(stats.population, first.config.initial_population + stats.births - stats.deaths)
            self.assertEqual(len({h.id for h in first.agents}), stats.population)
            self.assertTrue(all(0 <= h.energy <= first.config.max_energy for h in first.agents))
            self.assertTrue(all(0 <= t.food <= first.config.food_capacity and 0 <= t.water <= first.config.water_capacity
                                for row in first.world.tiles for t in row))
        self.assertGreater(first.statistics.births, 0)
        self.assertGreater(first.statistics.highest_generation, 1)
        self.assertGreater(first.statistics.food_regenerated, 0)
        self.assertEqual(random.getstate(), global_before)
        for name in ("_order_rng", "_offspring_rng", "_inheritance_rng", "_mutation_rng", "_food_rng", "_water_rng"):
            self.assertEqual(getattr(first, name).getstate(), getattr(second, name).getstate())
        self.assertEqual(first.reproduction.rng.getstate(), second.reproduction.rng.getstate())
        for a, b in zip(first.agents, second.agents):
            self.assertEqual(a.brain.rng.getstate(), b.brain.rng.getstate())
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        births = [e for e in events if e["kind"] == "birth"]
        self.assertEqual(len(births), first.statistics.births)
        self.assertTrue(any(e["details"]["child_genome"] != asdict(Genome.founder(first.config)) for e in births))
        for resource in ("food", "water"):
            remaining = sum(getattr(t, resource) for row in first.world.tiles for t in row)
            self.assertEqual(remaining, getattr(first.config, f"initial_{resource}")
                             + getattr(first.statistics, f"{resource}_regenerated")
                             - getattr(first.statistics, f"{resource}_consumed"))

    def test_energy_gain_cap_and_consumption_requirement(self):
        sim = pair(food_energy_gain=20)
        human = sim.agents[0]
        human.brain = ConstantBrain(Action.EAT)
        sim.agents[1].brain = ConstantBrain()
        human.energy = 95
        sim.world.tile_at(human.x, human.y).food = 1
        sim.step()
        self.assertEqual(human.energy, 100)
        human.energy = 50
        sim.step()
        self.assertEqual(human.energy, 50)

    def test_energy_death_and_disabled_energy_death(self):
        for lethal in (False, True):
            sim = pair(initial_population=1, energy_per_tick=2, energy_depletion_lethal=lethal)
            human = sim.agents[0]
            human.brain = ConstantBrain()
            human.energy = 1
            sim.step()
            self.assertEqual(human.energy, 0)
            self.assertEqual(human.alive, not lethal)
            self.assertEqual(sim.statistics.energy_deaths, int(lethal))

    def test_death_cause_priority_is_preserved_with_energy_last(self):
        for hunger, thirst, age, expected in ((100, 80, 2, "starvation"), (0, 80, 2, "dehydration"),
                                            (0, 0, 2, "old_age"), (0, 0, 0, "energy")):
            sim = pair(max_age=2, energy_depletion_lethal=True)
            human = sim.agents[0]
            human.hunger, human.thirst, human.age, human.energy = hunger, thirst, age, 0
            self.assertEqual(human.update_physiology(sim.config), expected)

    def test_extinction_is_possible_with_reproduction_and_regeneration_enabled(self):
        config = Config.evolution(initial_population=1, initial_food=0, initial_water=0, max_thirst=1)
        sim = Simulation(config)
        sim.step()
        self.assertEqual((sim.population, sim.statistics.deaths, sim.statistics.births), (0, 1, 0))
        stats = sim.statistics
        self.assertEqual(sim.step(), stats)

    def test_legacy_seed_42_regression(self):
        sim = Simulation(Config())
        for _ in range(1000):
            if not sim.population:
                break
            sim.step()
        stats = sim.statistics
        self.assertEqual((stats.tick, stats.population, stats.deaths), (309, 0, 100))
        self.assertEqual((stats.food_consumed, stats.water_consumed, stats.signals_emitted), (361, 389, 1425))
        self.assertEqual((stats.births, stats.food_regenerated, stats.water_regenerated), (0, 0, 0))

    def test_invalid_evolution_settings_fail_validation(self):
        for settings in (
            {"reproduction_enabled": 1}, {"energy_depletion_lethal": 1}, {"min_reproductive_age": -1},
            {"max_reproductive_age": 19}, {"reproduction_radius": -1}, {"reproduction_cooldown": -1},
            {"reproduction_energy_cost": 0}, {"offspring_energy": -1}, {"offspring_energy": 101},
            {"initial_energy": 101}, {"food_energy_gain": -1}, {"energy_per_tick": -1},
            {"mutation_rate": 1.1}, {"mutation_strength": -1}, {"mutation_strength": 1.1},
            {"mutation_strength": float("inf")},
            {"food_regeneration_probability": 0.1}, {"water_regeneration_probability": 0.1},
            {"food_capacity": 0}, {"water_capacity": -1}, {"max_population": 99},
            {"reproduction_need_fraction": 1.1},
            {"world_width": 1, "world_height": 1, "food_capacity": 2, "initial_food": 3},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Config(**settings)


if __name__ == "__main__":
    unittest.main()
