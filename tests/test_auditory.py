"""Perfect hearing, sensory privacy, unchanged behavior, and historical replay."""

from dataclasses import FrozenInstanceError, asdict, fields, replace
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agents.genome import Genome
from agents.learning import Physiology
from agents.observation import HeardSignal, SILENCE, SourceDirection
from config import Config
from experiments.auditory_demo import run_demo
from simulation.events import JSONLEventWriter
from simulation.simulation import Simulation
from systems.communication import Communication
from tests.auditory_control import ActionDigest, capture, ecological_state
from tests.test_basic import FixedBrain, make_sim


class AuditoryTests(unittest.TestCase):
    def test_all_nine_sign_sectors_including_unequal_offsets(self):
        offsets = {(0, -7): "N", (1, -7): "NE", (7, 0): "E", (7, 1): "SE",
                   (0, 7): "S", (-1, 7): "SW", (-7, 0): "W", (-7, -1): "NW", (0, 0): "SAME_CELL"}
        for (dx, dy), direction in offsets.items():
            with self.subTest(offset=(dx, dy)):
                comm = Communication(16, 7)
                comm.emit(4, 99, 10 + dx, 10 + dy)
                comm.advance()
                self.assertEqual(comm.hear(1, 10, 10), (HeardSignal(4, SourceDirection(direction)),))

    def test_inclusive_chebyshev_boundary_and_outside(self):
        comm = Communication(16, 8)
        comm.emit(1, 1, 8, 8)  # Square corner is inside, despite Euclidean distance.
        comm.emit(2, 2, 9, 0)
        comm.advance()
        self.assertEqual(comm.hear(0, 0, 0), (HeardSignal(1, SourceDirection.SE),))

    def test_hearing_is_independent_of_visual_genome(self):
        sim = make_sim(world_width=12, hearing_radius=8)
        human = sim.agents[0]
        human.x = human.y = 0
        sim.communication.emit(7, 99, 6, 0)
        sim.communication.advance()
        for visual_radius in (0, 1, 2, 8):
            human.genome = replace(human.genome, perception_radius=visual_radius)
            obs = sim.observe(human)
            self.assertEqual(obs.signals, (HeardSignal(7, SourceDirection.E),))
            self.assertEqual(any(t.dx == 6 for t in obs.tiles), visual_radius >= 6)

    def test_hearing_change_does_not_change_visual_tiles(self):
        a = make_sim(hearing_radius=0)
        b = make_sim(hearing_radius=8)
        self.assertEqual(a.observe(a.agents[0]).tiles, b.observe(b.agents[0]).tiles)
        self.assertEqual(a.agents[0].genome, b.agents[0].genome)
        self.assertNotIn("hearing_radius", {f.name for f in fields(Genome)})

    def test_zero_hearing_only_receives_other_agents_in_same_cell(self):
        comm = Communication(16, 0)
        comm.emit(1, 1, 3, 3)
        comm.emit(2, 2, 3, 3)
        comm.emit(3, 3, 3, 4)
        comm.advance()
        self.assertEqual(comm.hear(1, 3, 3), (HeardSignal(2, SourceDirection.SAME_CELL),))

    def test_brain_record_is_immutable_and_has_no_identity_position_or_distance(self):
        comm = Communication(16, 8)
        comm.emit(7, 123, 5, 0)
        comm.advance()
        heard = comm.hear(0, 0, 0)[0]
        self.assertEqual(asdict(heard), {"signal_id": 7, "source_direction": "E"})
        self.assertEqual({f.name for f in fields(heard)}, {"signal_id", "source_direction"})
        for name in ("sender_id", "x", "y", "dx", "dy", "distance", "strength", "volume", "__dict__"):
            self.assertFalse(hasattr(heard, name), name)
        with self.assertRaises(FrozenInstanceError):
            heard.signal_id = 3

    def test_same_sector_signals_do_not_reveal_near_versus_far_or_identity(self):
        a, b = Communication(16, 8), Communication(16, 8)
        a.emit(7, 1, 1, 0)
        b.emit(7, 999, 8, 0)
        a.advance()
        b.advance()
        self.assertEqual(a.hear(0, 0, 0), b.hear(0, 0, 0))

    def test_duplicate_vocalizations_are_not_deduplicated_or_given_identity(self):
        comm = Communication(16, 8)
        for sender in (1, 2):
            comm.emit(7, sender, sender, 0)
        comm.advance()
        self.assertEqual(comm.hear(0, 0, 0), (HeardSignal(7, SourceDirection.E),) * 2)

    def test_sender_death_keeps_signal_for_next_tick_only(self):
        sim = make_sim(initial_population=2, max_age=2, hearing_radius=8)
        sender, receiver = sim.agents
        sender.x, sender.y, sender.age = 4, 0, 1
        receiver.x = receiver.y = 0
        sender.brain = FixedBrain(signal=7)
        sim.step()
        self.assertFalse(sender.alive)
        self.assertNotIn(sender, sim.agents)
        self.assertEqual(receiver.brain.observations[0].signals, ())
        sim.step()
        self.assertEqual(receiver.brain.observations[1].signals, (HeardSignal(7, SourceDirection.E),))
        self.assertEqual(sim.communication.hear(receiver.id, 0, 0), ())

    def test_three_tick_demo_hears_beyond_vision_with_delay_and_expiry(self):
        result = run_demo()
        self.assertFalse(result["sender_cell_visible"])
        self.assertEqual((result["visual_radius"], result["hearing_radius"]), (2, 8))
        self.assertEqual(result["receiver_observations"], [
            {"tick": 1, "signals": []},
            {"tick": 2, "signals": [{"signal_id": 7, "source_direction": "E"}]},
            {"tick": 3, "signals": []},
        ])

    def test_logging_separates_privileged_diagnostics_and_uses_pre_action_receiver_position(self):
        stream = io.StringIO()
        sim = Simulation(Config(initial_population=2, initial_food=0, initial_water=0,
                                hearing_radius=8), event_sink=JSONLEventWriter(stream))
        sender, receiver = sim.agents
        sender.x, sender.y = 6, 1
        receiver.x, receiver.y = 1, 1
        sender.brain, receiver.brain = FixedBrain(signal=7), FixedBrain()
        sim.step()
        from agents.brain import Action
        receiver.brain.action = Action.MOVE_WEST
        sim.step()
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        event = next(e for e in events if e["tick"] == 2 and e["agent_id"] == receiver.id)
        self.assertEqual(events[0]["details"]["auditory_schema_version"], 3)
        self.assertEqual(event["details"]["observation"]["auditory"],
                         {"kind": "IDENTIFIED", "signal": {"signal_id": 7, "source_direction": "E"}})
        self.assertEqual(event["details"]["auditory_diagnostics"], [{
            "signal_id": 7, "sender_id": sender.id, "sender_position": {"x": 6, "y": 1},
            "receiver_id": receiver.id, "receiver_position": {"x": 1, "y": 1},
            "source_direction": "E", "chebyshev_distance": 5, "apparent_loudness": 1/36, "hearing_radius": 8,
            "inside_hearing_range": True, "emitted_tick": 1, "receiving_tick": 2,
        }])
        self.assertEqual(event["details"]["after"]["x"], 0)

    def test_no_diagnostic_payloads_without_full_logging(self):
        with patch.object(Communication, "diagnostics", side_effect=AssertionError("unexpected diagnostic")):
            sim = Simulation(Config(initial_population=2), lineage_sink=lambda e: None)
            sim.step()

    def test_observation_and_diagnostics_consume_no_rng_or_mutate_state(self):
        sim = Simulation(Config.evolution(brain="learning", initial_population=10, random_seed=0))
        for _ in range(5):
            sim.step()
        before, global_before = ecological_state(sim), random.getstate()
        for human in sim.agents:
            for _ in range(3):
                sim.observe(human)
                sim.communication.diagnostics(human.id, human.x, human.y, sim.tick + 1)
        self.assertEqual(ecological_state(sim), before)
        self.assertEqual(random.getstate(), global_before)

    def test_same_seed_replays_nonempty_auditory_observations(self):
        config = Config(initial_population=12, world_width=5, world_height=5,
                        signal_probability=1, random_seed=0)
        a, b = Simulation(config), Simulation(config)
        heard_count = 0
        for _ in range(5):
            a.step()
            b.step()
            first = [a.observe(h) for h in a.agents]
            self.assertEqual(first, [b.observe(h) for h in b.agents])
            heard_count += sum(len(o.signals) for o in first)
        self.assertGreater(heard_count, 0)

    def test_ignoring_all_auditory_input_keeps_complete_learning_and_random_state(self):
        for brain in ("random", "learning"):
            config = Config.evolution(brain=brain, world_width=5, world_height=5,
                                      initial_population=12, initial_food=20, initial_water=20,
                                      min_reproductive_age=3, mutation_rate=0.5, random_seed=0,
                                      learning_uses_auditory=False)
            a, b = Simulation(config), Simulation(config, event_sink=JSONLEventWriter(io.StringIO()))
            with patch.object(a.communication, "perceive", return_value=SILENCE):
                for _ in range(60):
                    a.step()
                    b.step()
                    self.assertEqual(ecological_state(a), ecological_state(b))
                    if brain == "learning":
                        for h in b.agents:
                            obs = b.observe(h)
                            body = Physiology.capture(h)
                            self.assertEqual(h.brain.state(body, obs), h.brain.state(body, replace(obs, auditory=SILENCE)))
            self.assertGreater(b.statistics.births, 0)

    def test_defaults_validation_and_old_config_alias(self):
        self.assertEqual((Config().hearing_radius, Config().signal_vocab_size), (8, 16))
        for radius in (0, 3, 8, 40):
            self.assertEqual(Simulation(Config(hearing_radius=radius, initial_population=0)).communication.hearing_radius, radius)
        for bad in (-1, 2.5, True, "8", None, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Config(hearing_radius=bad)
        for radius in (0, 3):
            old = Config(signal_range=radius)
            self.assertEqual(old.hearing_radius, radius)
            self.assertEqual(Config(**asdict(old)), old)
            self.assertEqual(replace(old, signal_range=None, hearing_radius=8).hearing_radius, 8)
        with self.assertRaises(ValueError):
            Config(signal_range=3, hearing_radius=4)

    def test_cli_config_and_validation_before_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            result = subprocess.run([sys.executable, "main.py", "--hearing-radius", "0",
                                     "--signal-vocab-size", "40", "--ticks", "2", "--population", "2",
                                     "--csv", str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(path.with_suffix(".csv.metadata.json").read_text())
            self.assertEqual(metadata["version"], "0.6")
            self.assertEqual(metadata["config"]["hearing_radius"], 0)
            self.assertEqual(metadata["config"]["signal_vocab_size"], 40)
            missing = Path(directory) / "invalid.csv"
            for radius in ("-1", "1.5"):
                result = subprocess.run([sys.executable, "main.py", "--hearing-radius", radius,
                                         "--csv", str(missing)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(missing.exists())


class HistoricalAuditoryControlTests(unittest.TestCase):
    def test_pre_v04_complete_trajectories_and_actions_at_old_and_new_hearing_ranges(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/v03_auditory_control.json").read_text())
        global_before = random.getstate()
        for run in fixture["runs"]:
            for radius in (3, 8):
                with self.subTest(mode=run["mode"], brain=run["brain"], radius=radius):
                    settings = {**run["config"], "signal_range": None, "hearing_radius": radius,
                                "learning_uses_auditory": False, "learning_controls_vocalization": False}
                    sink = ActionDigest()
                    sim = Simulation(Config(**settings), event_sink=sink)
                    actual = capture(sim, fixture["ticks"])
                    for key, value in actual.items():
                        self.assertEqual(value, run[key], key)
                    self.assertEqual(sink.digest.hexdigest(), run["actions_sha256"])
        self.assertEqual(random.getstate(), global_before)


if __name__ == "__main__":
    unittest.main()
