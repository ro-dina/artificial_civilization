import csv
from dataclasses import asdict, fields
import io
import random
import unittest
from unittest.mock import patch

from agents.brain import Action, Decision
from agents.genome import Genome
from config import Config
from simulation.simulation import Simulation
from simulation.statistics import CSVStatisticsWriter, GENOME_TRAITS, summarize_genomes

TRAIT_COLUMNS = tuple(f"{prefix}_{trait}" for trait in GENOME_TRAITS for prefix in ("mean", "min", "max"))


class FixedBrain:
    def __init__(self, action=Action.WAIT):
        self.action = action

    def choose_action(self, human, observation):
        return Decision(self.action)


def dynamics(sim):
    """Include all agent fields, ordering, resources, signals, counters, and RNG states."""
    return (
        sim.tick, sim._next_id,
        {name: value for name, value in asdict(sim.statistics).items() if name not in TRAIT_COLUMNS},
        tuple(tuple(asdict(h.genome) if f.name == "genome" else getattr(h, f.name)
                    for f in fields(h) if f.name != "brain") for h in sim.agents),
        tuple((h.id, h.brain.rng.getstate(), h.brain.vocab_size, h.brain.actions) for h in sim.agents),
        tuple((t.food, t.water) for row in sim.world.tiles for t in row),
        tuple((cell, tuple(signals)) for cell, signals in sim.communication._audible.items()),
        tuple((cell, tuple(signals)) for cell, signals in sim.communication._pending.items()),
        tuple(getattr(sim, name).getstate() for name in (
            "_order_rng", "_offspring_rng", "_inheritance_rng", "_mutation_rng", "_food_rng", "_water_rng")),
        sim.reproduction.rng.getstate(),
    )


class GenomeStatisticsTests(unittest.TestCase):
    def assert_traits(self, stats, genome):
        for trait in GENOME_TRAITS:
            for prefix in ("mean", "min", "max"):
                self.assertEqual(getattr(stats, f"{prefix}_{trait}"), getattr(genome, trait))

    def test_founder_statistics_at_tick_zero(self):
        for config in (Config(), Config(initial_population=17, perception_radius=4, signal_probability=0.3)):
            with self.subTest(config=config):
                sim = Simulation(config)
                self.assertEqual(sim.statistics.tick, 0)
                self.assert_traits(sim.statistics, Genome.founder(config))

    def test_mixed_genome_mean_min_max_exclude_dead_agents(self):
        sim = Simulation(Config(initial_population=4))
        genomes = (Genome(1, 0.1, 0.5, 1.5), Genome(3, 0.5, 1, 1),
                   Genome(5, 0.9, 1.5, 0.5), Genome(8, 1, 1.5, 1.5))
        for human, genome in zip(sim.agents, genomes):
            human.genome = genome
        sim.agents[-1].alive = False
        expected = {
            "mean_perception_radius": 3, "min_perception_radius": 1, "max_perception_radius": 5,
            "mean_signal_probability": 0.5, "min_signal_probability": 0.1, "max_signal_probability": 0.9,
            "mean_hunger_multiplier": 1, "min_hunger_multiplier": 0.5, "max_hunger_multiplier": 1.5,
            "mean_thirst_multiplier": 1, "min_thirst_multiplier": 0.5, "max_thirst_multiplier": 1.5,
        }
        before = dynamics(sim)
        self.assertEqual(summarize_genomes(iter(sim.agents)), expected)
        self.assertEqual(dynamics(sim), before)

    def test_deaths_update_current_statistics_without_changing_old_snapshot(self):
        sim = Simulation(Config(initial_population=2, initial_food=0, initial_water=0,
                                hunger_per_tick=0, thirst_per_tick=0, max_age=None),
                         brain_factory=lambda rng, config: FixedBrain())
        sim.agents[0].genome = Genome(1, 0.1, 0.5, 1.5)
        survivor_genome = Genome(5, 0.9, 1.5, 0.5)
        sim.agents[1].genome = survivor_genome
        previous = sim.step()
        self.assertEqual(previous.mean_perception_radius, 3)
        self.assertEqual(previous.mean_signal_probability, 0.5)
        sim.agents[0].hunger = sim.config.max_hunger
        current = sim.step()
        self.assertEqual((current.population, current.deaths), (1, 1))
        self.assert_traits(current, survivor_genome)
        self.assertEqual(previous.mean_perception_radius, 3)
        self.assertEqual(previous.population, 2)

    def test_newborn_genomes_are_included_in_birth_tick_statistics(self):
        sim = Simulation(Config(world_width=1, world_height=1, initial_population=2,
                                initial_food=0, initial_water=0, hunger_per_tick=0,
                                thirst_per_tick=0, max_age=None, reproduction_enabled=True,
                                min_reproductive_age=0, mutation_rate=0),
                         brain_factory=lambda rng, config: FixedBrain())
        a, b = sim.agents
        a.reproductive_type, b.reproductive_type = 0, 1
        a.genome, b.genome = Genome(0, 0, 0.5, 1.5), Genome(8, 1, 1.5, 0.5)
        before = sim.step()
        a.brain.action = b.brain.action = Action.REPRODUCE
        after = sim.step()
        child = sim.agents[-1]
        self.assertEqual((after.population, after.births, child.age), (3, 1, 0))
        self.assertNotEqual(after.mean_perception_radius, before.mean_perception_radius)
        for trait in GENOME_TRAITS:
            expected_mean = (getattr(a.genome, trait) + getattr(b.genome, trait) + getattr(child.genome, trait)) / 3
            self.assertAlmostEqual(getattr(after, f"mean_{trait}"), expected_mean)
            self.assertEqual(getattr(after, f"min_{trait}"), min(getattr(a.genome, trait), getattr(b.genome, trait)))
            self.assertEqual(getattr(after, f"max_{trait}"), max(getattr(a.genome, trait), getattr(b.genome, trait)))

    def test_extinct_and_initially_empty_populations_use_none_and_blank_csv_cells(self):
        for population in (0, 1):
            sim = Simulation(Config(initial_population=population, max_hunger=1))
            stats = sim.step()
            self.assertEqual(stats.population, 0)
            for column in TRAIT_COLUMNS:
                self.assertIsNone(getattr(stats, column))
            self.assertEqual(sim.step(), stats)
            stream = io.StringIO()
            writer = CSVStatisticsWriter(stream)
            writer.write(stats)
            row = next(csv.DictReader(io.StringIO(stream.getvalue())))
            self.assertTrue(all(row[column] == "" for column in TRAIT_COLUMNS))

    def test_zero_valued_traits_are_not_missing(self):
        sim = Simulation(Config(initial_population=1, perception_radius=0, signal_probability=0))
        self.assert_traits(sim.statistics, Genome(0, 0, 1, 1))

    def test_csv_appends_all_twelve_trait_columns_and_preserves_precision(self):
        sim = Simulation(Config(initial_population=2, signal_probability=0.123456789))
        stream = io.StringIO()
        writer = CSVStatisticsWriter(stream)
        writer.write(sim.statistics)
        reader = csv.DictReader(io.StringIO(stream.getvalue()))
        row = next(reader)
        self.assertEqual(set(reader.fieldnames[-12:]), set(TRAIT_COLUMNS))
        self.assertEqual(reader.fieldnames[0], "tick")
        self.assertEqual(reader.fieldnames[-13], "energy_deaths")
        for column in TRAIT_COLUMNS:
            self.assertEqual(float(row[column]), getattr(sim.statistics, column))

    def test_summaries_and_csv_do_not_change_state_or_rng(self):
        config = Config.evolution(world_width=5, world_height=5, initial_population=12,
                                  initial_food=20, initial_water=20, min_reproductive_age=4,
                                  reproduction_cooldown=5, mutation_rate=0.5)
        global_before = random.getstate()
        # Disable only the observational helper in the reference run. No product flag
        # or alternative biological implementation is needed for this comparison.
        with patch("simulation.simulation.summarize_genomes", return_value={}):
            reference = Simulation(config)
        observed = Simulation(config)
        writer = CSVStatisticsWriter(io.StringIO())
        writer.write(observed.statistics)
        self.assertEqual(dynamics(reference), dynamics(observed))
        for _ in range(180):
            with patch("simulation.simulation.summarize_genomes", return_value={}):
                reference.step()
            observed.step()
            summarize_genomes(observed.agents)  # Extra reads must also be harmless.
            writer.write(observed.statistics)
            self.assertEqual(dynamics(reference), dynamics(observed))
        self.assertGreater(observed.statistics.births, 0)
        self.assertGreater(observed.statistics.highest_generation, 1)
        self.assertEqual(random.getstate(), global_before)


if __name__ == "__main__":
    unittest.main()
