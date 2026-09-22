import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from agents.genome import Genome
from config import Config
from simulation.events import JSONLEventWriter
from simulation.simulation import Simulation
from tests.test_learning import learning_state


class FixedPerceptionTests(unittest.TestCase):
    def test_configuration_accepts_none_and_bounded_integers(self):
        self.assertIsNone(Config().fixed_perception_radius)
        for radius in (None, 0, 2, 8):
            self.assertEqual(Config(fixed_perception_radius=radius).fixed_perception_radius, radius)
        for radius in (-1, 9, 2.0, True, "2", float("nan"), float("inf")):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                Config(fixed_perception_radius=radius)

    def test_fixed_founders_override_baseline_and_control_actual_observation(self):
        for radius in (0, 2, 8):
            sim = Simulation(Config(perception_radius=1, fixed_perception_radius=radius,
                                    world_width=19, world_height=19, initial_population=1,
                                    initial_food=0, initial_water=0))
            human = sim.agents[0]
            human.x = human.y = 9
            self.assertEqual(human.genome.perception_radius, radius)
            self.assertEqual(len(sim.observe(human).tiles), (2 * radius + 1) ** 2)
            for prefix in ("mean", "min", "max"):
                self.assertEqual(getattr(sim.statistics, f"{prefix}_perception_radius"), radius)

    def test_inheritance_fixes_all_generations_even_with_maximum_mutation(self):
        global_before = random.getstate()
        for radius in (0, 2, 8):
            config = Config(fixed_perception_radius=radius, mutation_rate=1, mutation_strength=1)
            inheritance_rng, mutation_rng = random.Random(10), random.Random(20)
            a, b = Genome(0, 0, 0.5, 1.5), Genome(8, 1, 1.5, 0.5)
            for _ in range(100):
                child = Genome.inherit(a, b, config, inheritance_rng, mutation_rng)
                self.assertEqual(child.perception_radius, radius)
                a, b = b, child
        self.assertEqual(random.getstate(), global_before)

    def test_other_traits_and_rng_draws_match_ordinary_inheritance(self):
        a, b = Genome(0, 0.2, 0.6, 1.4), Genome(8, 0.8, 1.4, 0.6)
        for mutation_rate, mutation_strength in ((0, 1), (1, 0), (0.5, 0.1), (1, 1)):
            config = Config(mutation_rate=mutation_rate, mutation_strength=mutation_strength)
            fixed = replace(config, fixed_perception_radius=2)
            original_inheritance, fixed_inheritance = random.Random(4), random.Random(4)
            original_mutation, fixed_mutation = random.Random(5), random.Random(5)
            varied = set()
            for _ in range(50):
                original_child = Genome.inherit(a, b, config, original_inheritance, original_mutation)
                fixed_child = Genome.inherit(a, b, fixed, fixed_inheritance, fixed_mutation)
                self.assertEqual(fixed_child, replace(original_child, perception_radius=2))
                self.assertEqual(original_inheritance.getstate(), fixed_inheritance.getstate())
                self.assertEqual(original_mutation.getstate(), fixed_mutation.getstate())
                varied.add(fixed_child.signal_probability)
            if mutation_rate and mutation_strength:
                self.assertGreater(len(varied), 2)

    def test_unspecified_radius_remains_heritable_and_mutable(self):
        config = Config(perception_radius=3, mutation_rate=1, mutation_strength=1)
        self.assertEqual(Genome.founder(config).perception_radius, 3)
        inheritance_rng, mutation_rng = random.Random(6), random.Random(7)
        radii = {Genome.inherit(Genome(), Genome(), config, inheritance_rng, mutation_rng).perception_radius
                 for _ in range(50)}
        self.assertGreater(len(radii), 1)

    def test_fixed_learning_replays_across_generations_and_logging(self):
        config = Config.evolution(brain="learning", random_seed=0, fixed_perception_radius=2,
                                  world_width=5, world_height=5, initial_population=12,
                                  initial_food=20, initial_water=20, min_reproductive_age=4,
                                  reproduction_cooldown=5, mutation_rate=1)
        stream = io.StringIO()
        first = Simulation(config)
        second = Simulation(config, event_sink=JSONLEventWriter(stream))
        for _ in range(100):
            first.step()
            second.step()
            self.assertEqual(learning_state(first), learning_state(second))
            self.assertTrue(all(h.genome.perception_radius == 2 for h in first.agents))
            for prefix in ("mean", "min", "max"):
                self.assertEqual(getattr(first.statistics, f"{prefix}_perception_radius"),
                                 2 if first.population else None)
        self.assertGreaterEqual(first.statistics.highest_generation, 2)
        births = [json.loads(line) for line in stream.getvalue().splitlines()
                  if json.loads(line)["kind"] == "birth"]
        self.assertEqual(len(births), first.statistics.births)
        self.assertTrue(all(event["details"]["child_genome"]["perception_radius"] == 2 for event in births))

    def test_cli_fixed_zero_is_recorded_in_csv_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.csv"
            result = subprocess.run(
                [sys.executable, "main.py", "--mode", "evolution", "--brain", "learning",
                 "--fixed-perception-radius", "0", "--seed", "0", "--ticks", "3",
                 "--population", "2", "--csv", str(path)],
                check=True, text=True, capture_output=True,
            )
            self.assertIn("fixed_perception_radius=0", result.stdout)
            metadata = json.loads(path.with_suffix(".csv.metadata.json").read_text())
            self.assertEqual(metadata["config"]["fixed_perception_radius"], 0)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row["config_id"], metadata["config_id"])
                for prefix in ("mean", "min", "max"):
                    self.assertEqual(float(row[f"{prefix}_perception_radius"]), 0)

    def test_cli_rejects_invalid_radius_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            for radius in ("-1", "9", "2.5"):
                result = subprocess.run(
                    [sys.executable, "main.py", "--fixed-perception-radius", radius, "--csv", str(path)],
                    text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertFalse(path.exists())
                self.assertFalse(path.with_suffix(".csv.metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
