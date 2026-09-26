import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agents.brain import Action
from agents.genome import Genome
from config import Config
from simulation.cohorts import COUNTERS, CSVCohortWriter, PerceptionCohorts
from simulation.events import JSONLEventWriter
from simulation.simulation import Simulation
from tests.test_learning import learning_state
from tests.test_statistics import FixedBrain, dynamics


class CohortTests(unittest.TestCase):
    def make_sim(self, **overrides):
        self.snapshots, self.events = [], []
        settings = dict(world_width=1, world_height=1, initial_population=2, initial_food=0,
                        initial_water=0, hunger_per_tick=0, thirst_per_tick=0, energy_per_tick=0,
                        max_age=None, reproduction_enabled=True, min_reproductive_age=0,
                        mutation_rate=0, reproduction_cooldown=3)
        settings.update(overrides)
        return Simulation(Config(**settings), cohort_sink=self.snapshots.append, lineage_sink=self.events.append,
                          brain_factory=lambda rng, config: FixedBrain())

    def test_initial_snapshot_zero_exposure_and_founders(self):
        sim = self.make_sim()
        rows = self.snapshots[0]
        self.assertEqual(len(rows), 9)
        self.assertEqual(sum(r.living_population for r in rows), 2)
        self.assertTrue(all(getattr(r, field) == 0 for r in rows for field in COUNTERS))
        self.assertEqual(rows[2].mean_hunger_multiplier, 1)
        self.assertIsNone(rows[0].mean_hunger_multiplier)
        self.assertEqual([e.kind for e in self.events], ["run_started", "founder", "founder"])
        self.assertTrue(all(e.details["genome"] == h.genome for e, h in zip(self.events[1:], sim.agents)))

    def test_mixed_means_bounds_and_read_only_snapshot(self):
        sim = self.make_sim(initial_population=3)
        sim.agents[0].genome = Genome(0, 0.1, 0.5, 1.5)
        sim.agents[1].genome = Genome(0, 0.5, 1.5, 0.5)
        sim.agents[2].alive = False
        rows = sim.cohorts.snapshot(0, sim.agents)
        self.assertAlmostEqual(rows[0].mean_signal_probability, 0.3)
        self.assertEqual((rows[0].mean_hunger_multiplier, rows[0].mean_thirst_multiplier), (1, 1))
        self.assertEqual(sum(r.living_population for r in rows), 2)
        for p in (-1, 9):
            with self.assertRaises(ValueError):
                Genome(perception_radius=p)

    def test_birth_both_parents_initiator_and_newborn_exposure(self):
        sim = self.make_sim()
        a, b = sim.agents
        a.reproductive_type, b.reproductive_type = 0, 1
        a.genome, b.genome = Genome(0), Genome(2)
        a.brain.action = Action.REPRODUCE
        sim.step()
        child = sim.agents[-1]
        rows = self.snapshots[-1]
        self.assertEqual(sum(r.births_as_child for r in rows), 1)
        self.assertEqual(rows[child.genome.perception_radius].births_as_child, 1)
        self.assertEqual((rows[0].successful_parent_participations, rows[2].successful_parent_participations), (1, 1))
        self.assertEqual((rows[0].reproduction_initiations, rows[2].reproduction_initiations), (1, 0))
        self.assertEqual(sum(r.action_opportunities for r in rows), 2)
        self.assertEqual(sum(r.living_agent_ticks for r in rows), 3)
        self.assertEqual(child.age, 0)
        sim.step()
        self.assertEqual(sum(r.action_opportunities for r in self.snapshots[-1]), 3)
        self.assertEqual(sum(r.failed_reproduction_attempts for r in self.snapshots[-1]), 1)

    def test_same_cohort_parents_count_twice_and_birth_event_reused(self):
        sim = self.make_sim()
        detailed = []
        sim.event_sink = detailed.append
        sim.agents[0].reproductive_type, sim.agents[1].reproductive_type = 0, 1
        sim.agents[0].brain.action = Action.REPRODUCE
        sim.step()
        self.assertEqual(self.snapshots[-1][2].successful_parent_participations, 2)
        self.assertIs(next(e for e in detailed if e.kind == "birth"), next(e for e in self.events if e.kind == "birth"))

    def test_parents_can_die_on_birth_tick_and_cost_is_not_reclassified(self):
        sim = self.make_sim(max_age=1)
        for p, human in enumerate(sim.agents):
            human.reproductive_type = p
            human.genome = Genome(p)
            human.brain.action = Action.REPRODUCE
        sim.step()
        rows = self.snapshots[-1]
        self.assertEqual((sim.population, sim.statistics.births, sim.statistics.deaths), (1, 1, 2))
        self.assertEqual(sum(r.successful_parent_participations for r in rows), 2)
        self.assertEqual(sum(r.old_age_deaths for r in rows), 2)
        self.assertEqual(sum(r.action_opportunities for r in rows), 2)
        self.assertEqual(sum(r.living_agent_ticks for r in rows), 1)

    def test_safeguard_failure_counted_only_as_initiation_and_failure(self):
        sim = self.make_sim(max_population=2)
        sim.agents[0].reproductive_type, sim.agents[1].reproductive_type = 0, 1
        sim.agents[0].brain.action = Action.REPRODUCE
        sim.step()
        row = self.snapshots[-1][2]
        self.assertEqual((row.reproduction_initiations, row.failed_reproduction_attempts,
                          row.successful_parent_participations, row.births_as_child), (1, 1, 0, 0))

    def test_death_causes_and_extinction_action_exposure(self):
        sim = self.make_sim(initial_population=4, max_age=20, energy_depletion_lethal=True)
        a, b, c, d = sim.agents
        for p, human in enumerate(sim.agents):
            human.genome = Genome(p)
        a.hunger, a.thirst = sim.config.max_hunger, sim.config.max_thirst
        b.thirst, c.age, d.energy = sim.config.max_thirst, 20, 0
        sim.step()
        self.assertEqual(sim.population, 0)
        rows = self.snapshots[-1]
        for p, cause in enumerate(("starvation", "dehydration", "old_age", "energy")):
            self.assertEqual(getattr(rows[p], f"{cause}_deaths"), 1)
            self.assertEqual((rows[p].deaths, rows[p].action_opportunities, rows[p].living_agent_ticks), (1, 1, 0))
        deaths = [e for e in self.events if e.kind == "death"]
        self.assertEqual({e.details["genome_at_death"].perception_radius for e in deaths}, {0, 1, 2, 3})
        self.assertTrue(all(e.tick == 1 for e in deaths))
        sim.step()
        self.assertEqual(len(self.snapshots), 2)
        self.assertTrue(all(r.mean_age is None for r in rows))

    def test_initially_empty_and_streaming_blank_means(self):
        sim = self.make_sim(initial_population=0)
        stream = io.StringIO()
        writer = CSVCohortWriter(stream, metadata={"seed": 0})
        writer(self.snapshots[0])
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(r["mean_age"] == "" and r["seed"] == "0" for r in rows))
        sim.step()
        self.assertEqual(len(self.snapshots), 1)
        with self.assertRaises(ValueError):
            CSVCohortWriter(io.StringIO(), metadata={"tick": 999})

    def test_disabled_does_not_construct_payloads(self):
        with patch("simulation.simulation.Event", side_effect=AssertionError("disabled logging")):
            sim = Simulation(Config(initial_population=2))
            sim.step()
        self.assertIsNone(sim.cohorts)

    def test_all_four_observer_modes_preserve_every_tick_in_all_modes(self):
        global_state = random.getstate()
        for evolution in (False, True):
            for brain in ("random", "learning"):
                with self.subTest(evolution=evolution, brain=brain):
                    factory = Config.evolution if evolution else Config
                    cfg = factory(world_width=4, world_height=4, initial_population=12, initial_food=15,
                                  initial_water=15, brain=brain, min_reproductive_age=3,
                                  reproduction_cooldown=4, mutation_rate=0.6, random_seed=0)
                    sims = []
                    for cohorts, lineage in ((False, False), (True, False), (False, True), (True, True)):
                        sims.append(Simulation(cfg, cohort_sink=CSVCohortWriter(io.StringIO()) if cohorts else None,
                                               lineage_sink=JSONLEventWriter(io.StringIO()) if lineage else None))
                    state = learning_state if brain == "learning" else dynamics
                    totals = {name: 0 for name in COUNTERS}
                    for tick in range(121):
                        if tick:
                            for sim in sims:
                                sim.step()
                        reference = state(sims[0])
                        for sim in sims[1:]:
                            self.assertEqual(state(sim), reference)
                            self.assertEqual(sim.statistics, sims[0].statistics)
                        observed = sims[-1]
                        before = state(observed)
                        rows = observed.cohorts.snapshot(observed.tick, observed.agents)
                        self.assertEqual(state(observed), before)  # Also checks learning recency/pending/RNG.
                        self.assertEqual(sum(r.living_population for r in rows), observed.population)
                        self.assertEqual(sum(r.successful_parent_participations for r in rows), 2 * sum(r.births_as_child for r in rows))
                        if tick == observed.tick:  # Extinction no-op must not be accumulated again.
                            for name in COUNTERS:
                                totals[name] += sum(getattr(r, name) for r in rows)
                    self.assertEqual(totals["births_as_child"], observed.statistics.births)
                    self.assertEqual(totals["deaths"], observed.statistics.deaths)
                    self.assertEqual(totals["reproduction_initiations"], observed.statistics.reproduction_attempts)
                    if evolution:
                        self.assertGreater(observed.statistics.births, 0)
        self.assertEqual(random.getstate(), global_state)

    def test_cli_streams_all_outputs_and_checks_collisions_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate, cohorts, lineage = (root / n for n in ("run.csv", "cohorts.csv", "lineage.jsonl"))
            args = [sys.executable, "main.py", "--mode", "evolution", "--brain", "learning", "--seed", "0", "--ticks", "3",
                    "--csv", str(aggregate), "--cohort-csv", str(cohorts), "--lineage-events", str(lineage)]
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with cohorts.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 36)
            records = [json.loads(line) for line in lineage.read_text().splitlines()]
            self.assertEqual(records[-1]["kind"], "run_finished")
            self.assertNotIn("agent_step", {r["kind"] for r in records})
            sidecar = cohorts.with_suffix(".csv.metadata.json")
            self.assertEqual(json.loads(sidecar.read_text())["cohort_schema_version"], 1)
            before = aggregate.read_bytes()
            result = subprocess.run(args[:-2] + ["--lineage-events", str(aggregate)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different files", result.stderr)
            self.assertEqual(aggregate.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
