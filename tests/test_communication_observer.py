"""Observational isolation: complete states, decision timing and compact schema."""

import csv
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agents.brain import Action, Decision
from agents.learning import LearningBrain, Physiology, PrimitiveStateEncoder
from agents.observation import AuditoryKind, Observation, SourceDirection
from config import Config
from simulation.communication_observer import COLUMNS, CSVCommunicationWriter, CommunicationObserver
from simulation.simulation import Simulation
from tests.auditory_control import encoded
from tests.test_statistics import FixedBrain
from tests.test_vocalization import vocal_state

FIXTURE = Path(__file__).parent / "fixtures/v06_vocal_seed7_tick80.json"


class ObserverIsolationTests(unittest.TestCase):
    def test_v06_fixture_bytes_are_unchanged(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                         "0098529d6812003f97d718b3e1f4927b88fd371775047f82b640df96ec048dcd")

    def compare(self, cfg, ticks):
        rows, events_off, events_on = [], [], []
        off = Simulation(cfg, event_sink=events_off.append)
        on = Simulation(cfg, event_sink=events_on.append, communication_sink=rows.append)
        self.assertIsNone(off.communication_observer)
        self.assertEqual(vocal_state(off), vocal_state(on))
        self.assertEqual(rows, [])  # tick 0 is a snapshot, never a decision
        for _ in range(ticks):
            off.step()
            on.step()
            self.assertEqual(vocal_state(off), vocal_state(on))
            self.assertEqual(events_off, events_on)  # includes dying brains' diagnostics
        actor_events = [e for e in events_on if e.kind == "agent_step"]
        self.assertEqual([(r.tick, r.agent_id) for r in rows], [(e.tick, e.agent_id) for e in actor_events])
        self.assertEqual(sum(r.signal_id is not None for r in rows), on.statistics.signals_emitted)
        for row, event in zip(rows, actor_events):
            obs = event.details["observation"]
            self.assertEqual(row.auditory_kind, obs.auditory.kind.value)
            self.assertEqual(row.heard_signal_id, obs.auditory.signal.signal_id if obs.auditory.signal else None)
            self.assertEqual(row.physical_action, event.details["action"])
        return on, rows

    def test_all_four_ablations_preserve_every_tick_state_and_full_events(self):
        original = Config(**json.loads(FIXTURE.read_text())["config"])
        global_before = random.getstate()
        for listening in (False, True):
            for production in (False, True):
                with self.subTest(listening=listening, production=production):
                    sim, rows = self.compare(replace(original, learning_uses_auditory=listening,
                                                   learning_controls_vocalization=production), 40)
                    self.assertGreater(sim.statistics.births, 0)
                    self.assertTrue(all(r.vocal_mechanism == ("learned" if production else "random") for r in rows))
                    self.assertTrue(all((r.encoded_auditory_kind is not None) == listening for r in rows))
        self.assertEqual(random.getstate(), global_before)

    def test_random_and_legacy_controls_preserve_rng_and_body_trajectory(self):
        for brain in ("random", "learning"):
            with self.subTest(brain=brain):
                self.compare(Config(brain=brain, initial_population=6, world_width=5, world_height=5), 12)

    def test_observer_on_matches_the_unmodified_v06_historical_fixture(self):
        fixture = json.loads(FIXTURE.read_text())
        trajectory, events = hashlib.sha256(), hashlib.sha256()
        sim = Simulation(Config(**fixture["config"]), communication_sink=lambda row: None,
                         event_sink=lambda e: events.update(encoded(asdict(e))))
        for tick in range(fixture["ticks"] + 1):
            if tick:
                sim.step()
            trajectory.update(encoded(vocal_state(sim)))
        self.assertEqual(trajectory.hexdigest(), fixture["result"]["trajectory_sha256"])
        self.assertEqual(events.hexdigest(), fixture["result"]["events_sha256"])
        self.assertEqual(hashlib.sha256(encoded(vocal_state(sim))).hexdigest(), fixture["result"]["final_state_sha256"])
        self.assertEqual(asdict(sim.statistics), fixture["result"]["statistics"])

    def test_encoded_state_is_read_only_and_matches_actual_pending_state(self):
        sim = Simulation(Config(brain="learning", initial_population=1))
        h = sim.agents[0]
        obs = replace(sim.observe(h), tick=1)
        decision = h.brain.choose_action(h, obs)
        rows = []
        observer = CommunicationObserver(sim.config, rows.append)
        observer.begin_tick(sim.agents)
        before = vocal_state(sim)
        with patch.object(random.Random, "random", side_effect=AssertionError("observer RNG")):
            for _ in range(3):
                observer.decision(h, obs, decision)
                self.assertIs(h.brain.decision_state, h.brain._pending.state)
        self.assertEqual(vocal_state(sim), before)
        r = rows[0]
        self.assertEqual((r.hunger_bin, r.thirst_bin, r.energy_bin, r.food_dx, r.food_dy, r.water_dx, r.water_dy),
                         h.brain._pending.state[:7])
        self.assertEqual(r.age, h.age)
        with self.assertRaises(FrozenInstanceError):
            r.signal_id = 3
        with self.assertRaises(AttributeError):
            h.brain.decision_state = ()

    def test_random_body_diagnostics_use_the_shared_primitive_encoder(self):
        rows = []
        sim = Simulation(Config(initial_population=1), communication_sink=rows.append)
        h = sim.agents[0]
        body, obs = Physiology.capture(h), sim.observe(h)
        expected = PrimitiveStateEncoder(sim.config).primitive_state(body, obs)
        sim.step()
        r = rows[0]
        self.assertEqual((r.hunger_bin, r.thirst_bin, r.energy_bin, r.food_dx, r.food_dy, r.water_dx, r.water_dy), expected)
        self.assertEqual((r.brain_type, r.vocal_mechanism), ("random", "random"))
        self.assertFalse(r.learning_controls_vocalization)
        self.assertFalse(r.learning_uses_auditory)

    def test_density_is_tick_start_chebyshev_count_including_shared_cells(self):
        sim = Simulation(Config(world_width=8, world_height=6, initial_population=5))
        for h, (x, y) in zip(sim.agents, ((0, 0), (0, 0), (2, 2), (3, 0), (7, 5))):
            h.x, h.y = x, y
        observer = CommunicationObserver(sim.config, lambda r: None)
        observer.begin_tick(sim.agents)
        self.assertEqual(observer.density(0, 0), 2)
        self.assertEqual(observer.density(7, 5), 0)
        sim.agents[1].x = 7
        self.assertEqual(observer.density(0, 0), 2)  # frozen tick-start density

    def test_next_tick_reception_and_masking_are_distinct_from_silence(self):
        rows = []
        sim = Simulation(Config(world_width=1, world_height=1, initial_population=3),
                         communication_sink=rows.append,
                         brain_factory=lambda rng, cfg: type("Emitter", (), {"choose_action": lambda self, h, o: Decision(Action.WAIT, 12)})())
        sim.step()
        self.assertTrue(all(r.auditory_kind == "SILENCE" for r in rows))
        sim.step()
        self.assertTrue(all(r.auditory_kind == "MASKED" for r in rows if r.tick == 2))
        self.assertTrue(all(r.signal_id == 12 for r in rows))
        self.assertEqual(sim.statistics.signals_emitted, 6)

    def test_identified_arbitrary_id_direction_without_brain_identity(self):
        rows = []
        sim = Simulation(Config(world_width=2, world_height=1, initial_population=2), communication_sink=rows.append,
                         brain_factory=lambda rng, cfg: FixedBrain())
        sim.agents[0].x, sim.agents[1].x = 0, 1
        sim.communication.emit(12, 100, 1, 0)
        sim.communication.advance()
        sim.step()
        first = next(r for r in rows if r.agent_id == 0)
        self.assertEqual((first.auditory_kind, first.heard_signal_id, first.heard_direction), ("IDENTIFIED", 12, "E"))
        self.assertFalse(any(name in COLUMNS for name in ("sender_id", "sender_x", "distance", "loudness")))
        self.assertFalse(hasattr(sim.observe(sim.agents[0]).auditory.signal, "sender_id"))

    def test_terminal_decisions_are_recorded_extinction_creates_no_future_rows(self):
        rows = []
        sim = Simulation(Config(initial_population=2, max_age=1), communication_sink=rows.append)
        sim.step()
        self.assertEqual((sim.population, len(rows)), (0, 2))
        sim.step()
        self.assertEqual(len(rows), 2)

    def test_birth_tick_has_no_newborn_decisions(self):
        cfg = Config(**json.loads(FIXTURE.read_text())["config"])
        rows, births = [], []
        sim = Simulation(cfg, communication_sink=rows.append,
                         event_sink=lambda e: births.append(e) if e.kind == "birth" else None)
        for _ in range(12):
            sim.step()
        self.assertGreater(len(births), 0)
        decisions = {(r.tick, r.agent_id) for r in rows}
        for e in births:
            self.assertNotIn((e.tick, e.agent_id), decisions)

    def test_csv_writer_is_streaming_and_silence_is_not_a_signal(self):
        stream = io.StringIO()
        writer = CSVCommunicationWriter(stream)
        sim = Simulation(Config(initial_population=1, signal_probability=0), communication_sink=writer)
        sim.step()
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(tuple(rows[0]), COLUMNS)
        self.assertEqual(rows[0]["vocal_action"], "SILENCE")
        self.assertEqual(rows[0]["signal_id"], "")
        self.assertEqual(sim.statistics.signals_emitted, 0)
        self.assertEqual(set(writer.__dict__), {"writer"})

    def test_cli_recording_preserves_aggregate_csv_config_id_and_writes_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = [sys.executable, "main.py", "--mode", "evolution", "--brain", "learning", "--seed", "7",
                    "--population", "3", "--ticks", "5"]
            subprocess.run([*base, "--csv", str(root / "off.csv")], check=True, capture_output=True)
            subprocess.run([*base, "--csv", str(root / "on.csv"), "--communication-csv", str(root / "comm.csv")], check=True, capture_output=True)
            self.assertEqual((root / "off.csv").read_bytes(), (root / "on.csv").read_bytes())
            metadata = json.loads((root / "comm.csv.metadata.json").read_text())
            normal = json.loads((root / "on.csv.metadata.json").read_text())
            self.assertEqual(metadata["config_id"], normal["config_id"])
            self.assertEqual(metadata["final_tick"], 5)
            self.assertEqual(tuple(metadata["columns"]), COLUMNS)
            self.assertEqual(metadata["schema_version"], 1)
            self.assertEqual(metadata["density_radius"], 2)
            self.assertIn("agents/learning.py", metadata["source_sha256"])
            failed = subprocess.run([*base, "--csv", str(root / "same.csv"), "--communication-csv", str(root / "same.csv")], capture_output=True)
            self.assertNotEqual(failed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
