"""v0.5 masking, mechanical memory, learned listening and historical ablations."""

from dataclasses import FrozenInstanceError, asdict, fields, replace
import hashlib
import io
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agents.auditory_memory import AuditoryMemory
from agents.brain import Action
from agents.learning import Physiology
from agents.observation import (AuditoryKind, AuditoryPercept, HeardSignal, MASKED,
                                Observation, SILENCE, SourceDirection)
from config import Config
from experiments.listening_demo import run_demo as listening_demo
from experiments.masking_demo import run_demo as masking_demo
from simulation.events import JSONLEventWriter
from simulation.simulation import Simulation
from systems.communication import Communication, apparent_loudness
from tests.auditory_control import ActionDigest, capture, ecological_state, encoded
from tests.test_learning import make_brain, learning_state


def identified(signal_id=7, direction=SourceDirection.E):
    return AuditoryPercept(AuditoryKind.IDENTIFIED, HeardSignal(signal_id, direction))


def channel_percept(sources, ratio=3):
    channel = Communication(40, 8, ratio)
    for sender, signal_id, x, y in sources:
        channel.emit(signal_id, sender, x, y)
    channel.advance()
    return channel.perceive(0, 0, 0)


def capture_listening(config, ticks):
    """Version-specific full state/memory and resolved/privileged event digests."""
    events = hashlib.sha256()
    def sink(event):
        events.update(encoded(asdict(event)))
    sim = Simulation(config, event_sink=sink)
    trajectory = hashlib.sha256()
    for tick in range(ticks + 1):
        if tick:
            sim.step()
        state = (learning_state(sim), asdict(sim.statistics))
        trajectory.update(encoded(state))
    return {"trajectory_sha256": trajectory.hexdigest(),
            "final_state_sha256": hashlib.sha256(encoded(state)).hexdigest(),
            "events_sha256": events.hexdigest(), "statistics": asdict(sim.statistics)}


class MaskingTests(unittest.TestCase):
    def test_loudness_values_and_chebyshev_corner(self):
        for d, expected in enumerate((1, 0.25, 1/9, 1/16, 1/25)):
            self.assertEqual(apparent_loudness(d), expected)
        channel = Communication(16, 8)
        channel.emit(7, 1, 8, 8)
        channel.advance()
        self.assertEqual(channel.perceive(0, 0, 0), identified(7, SourceDirection.SE))
        self.assertEqual(channel.diagnostics(0, 0, 0, 1)[0]["apparent_loudness"], 1/81)

    def test_five_examples_and_inclusive_dominance_threshold(self):
        cases = [([], SILENCE), ([(1, 7, 2, 0)], identified()),
                 ([(1, 7, 1, 0), (2, 12, 3, 0)], identified()),
                 ([(1, 7, 1, 0), (2, 12, 2, 0)], MASKED),
                 ([(1, 7, 2, 0), (2, 12, -2, 0)], MASKED)]
        for sources, expected in cases:
            self.assertEqual(channel_percept(sources), expected)
        sources = [(1, 7, 1, 0), (2, 12, 2, 0)]
        self.assertEqual(channel_percept(sources, ratio=2.25), identified())
        self.assertEqual(channel_percept(sources, ratio=2.25001), MASKED)
        demo = masking_demo()
        self.assertEqual([demo[k]["resolved_percept"].kind.value for k in "ABCDE"],
                         ["IDENTIFIED", "MASKED", "MASKED", "IDENTIFIED", "SILENCE"])
        self.assertEqual((demo["A"]["dominance_ratio"], demo["B"]["dominance_ratio"]), (4, 2.25))

    def test_three_or_more_sources_use_only_top_two_not_count_or_total_loudness(self):
        dominant = [(1, 7, 0, -1), (2, 12, 3, 0), (3, 9, -3, 0), (4, 4, 3, 3)]
        self.assertEqual(channel_percept(dominant), identified(7, SourceDirection.N))
        dominant.append((5, 3, 0, 1))
        self.assertEqual(channel_percept(dominant), MASKED)
        self.assertEqual(channel_percept([(1, 7, 0, 0), (2, 7, 0, 0), (3, 12, 8, 8)]), MASKED)

    def test_order_and_sender_id_do_not_resolve_equal_strength_ties(self):
        for sources, expected in (([(1, 7, 1, 0), (2, 12, -1, 0), (3, 3, 7, 7)], MASKED),
                                  ([(1, 7, 1, 0), (2, 12, -3, 0), (3, 3, 7, 7)], identified())):
            for order in itertools.permutations(sources):
                self.assertEqual(channel_percept(order), expected)
                changed_ids = [(sender + 100, signal, x, y) for sender, signal, x, y in order]
                self.assertEqual(channel_percept(changed_ids), expected)

    def test_self_and_outside_signals_cannot_mask(self):
        self.assertEqual(channel_percept([(0, 3, 0, 0)]), SILENCE)
        self.assertEqual(channel_percept([(0, 3, 0, 0), (1, 7, 1, 0), (2, 12, 9, 0)]), identified())

    def test_masked_and_silence_records_hide_identity_direction_distance_loudness(self):
        self.assertNotEqual(MASKED, SILENCE)
        for percept in (MASKED, SILENCE, identified()):
            self.assertEqual({f.name for f in fields(percept)}, {"kind", "signal"})
            for item in (percept, percept.signal) if percept.signal is not None else (percept,):
                for name in ("sender_id", "dx", "dy", "x", "y", "distance", "loudness", "strength", "lineage"):
                    self.assertFalse(hasattr(item, name), name)
            with self.assertRaises(FrozenInstanceError):
                percept.kind = AuditoryKind.SILENCE
        self.assertIsNone(MASKED.signal)
        self.assertEqual(Observation(1, (), MASKED).signals, ())
        for bad in ((AuditoryKind.MASKED, HeardSignal(7, SourceDirection.E)),
                    (AuditoryKind.IDENTIFIED, None), ("MASKED", None)):
            with self.assertRaises(ValueError):
                AuditoryPercept(*bad)
        with self.assertRaises(ValueError):
            Observation(1, (), (HeardSignal(7, SourceDirection.E),))

    def test_masking_is_no_rng_and_simulation_brain_sees_only_resolved_percept(self):
        sim = Simulation(Config(initial_population=1, initial_food=0, initial_water=0))
        receiver = sim.agents[0]
        receiver.x = receiver.y = 0
        sim.communication.emit(7, 99, 1, 0)
        sim.communication.emit(12, 100, 2, 0)
        sim.communication.advance()
        before, global_before = ecological_state(sim), random.getstate()
        with patch.object(random.Random, "random", side_effect=AssertionError("unexpected RNG")):
            for _ in range(3):
                self.assertEqual(sim.observe(receiver).auditory, MASKED)
                sim.communication.diagnostics(receiver.id, 0, 0, 1)
                sim.communication.resolution_diagnostics(receiver.id, 0, 0)
        self.assertEqual(ecological_state(sim), before)
        self.assertEqual(random.getstate(), global_before)
        self.assertFalse(hasattr(receiver.brain, "auditory_memory"))


class AuditoryMemoryTests(unittest.TestCase):
    def test_identified_is_retained_during_silence_and_expires_at_exact_horizon(self):
        memory = AuditoryMemory(4)
        memory.update(identified(12), 100)
        for tick in range(100, 104):
            memory.update(SILENCE, tick)
            self.assertEqual(memory.feature(SILENCE, tick), (AuditoryKind.IDENTIFIED, 12, SourceDirection.E, tick-100))
            self.assertEqual(len(memory.events(tick)), 1)
        memory.update(SILENCE, 104)
        self.assertEqual(memory.events(104), ())
        self.assertEqual(memory.feature(SILENCE, 104), (AuditoryKind.SILENCE,))

    def test_masked_is_remembered_and_silence_never_fills_memory(self):
        memory = AuditoryMemory(4)
        for tick in range(20):
            memory.update(SILENCE, tick)
        self.assertEqual(len(memory._events), 0)
        memory.update(MASKED, 20)
        memory.update(SILENCE, 22)
        self.assertEqual(memory.feature(SILENCE, 22), (AuditoryKind.MASKED, 2))
        memory.update(SILENCE, 24)
        self.assertEqual(len(memory._events), 0)

    def test_raw_sequence_remains_bounded_latest_event_is_used_and_gaps_expire(self):
        memory = AuditoryMemory(4)
        for tick in range(100):
            memory.update(identified(tick % 16), tick)
            self.assertLessEqual(len(memory._events), 4)
        self.assertEqual([e.received_tick for e in memory.events(99)], [96, 97, 98, 99])
        self.assertEqual(memory.feature(SILENCE, 99), (AuditoryKind.IDENTIFIED, 3, SourceDirection.E, 0))
        memory.update(MASKED, 100)
        self.assertEqual(memory.feature(SILENCE, 101), (AuditoryKind.MASKED, 1))
        memory.update(SILENCE, 200)
        self.assertEqual(memory._events, memory._events.__class__())

    def test_horizon_one_repeated_same_tick_and_rewinding_validation(self):
        memory = AuditoryMemory(1)
        memory.update(identified(), 10)
        memory.update(MASKED, 10)
        memory.update(SILENCE, 10)
        self.assertEqual(len(memory.events(10)), 1)
        self.assertEqual(memory.feature(SILENCE, 10), (AuditoryKind.MASKED, 0))
        memory.update(SILENCE, 11)
        self.assertEqual(memory.events(11), ())
        for tick in (9, -1, True, 1.5):
            with self.assertRaises(ValueError):
                memory.update(MASKED, tick)

    def test_read_only_preview_does_not_update_memory_or_rng(self):
        brain, human, _ = make_brain()
        obs = Observation(100, (), identified())
        before_rng, global_before = brain.rng.getstate(), random.getstate()
        body = Physiology.capture(human)
        with patch.object(random.Random, "random", side_effect=AssertionError("unexpected RNG")):
            for _ in range(3):
                self.assertEqual(brain.state(body, obs)[-1], (AuditoryKind.IDENTIFIED, 7, SourceDirection.E, 0))
                brain.action_values(body, obs)
                brain.auditory_diagnostics(obs.tick)
        self.assertEqual(tuple(brain.auditory_memory._events), ())
        self.assertIsNone(brain.auditory_memory._last_tick)
        self.assertEqual(brain.memory_size, 0)
        self.assertEqual(brain.rng.getstate(), before_rng)
        self.assertEqual(random.getstate(), global_before)


class ListeningBrainTests(unittest.TestCase):
    def test_original_features_plus_one_nested_auditory_feature_and_ablation(self):
        brain, human, _ = make_brain()
        old, other, _ = make_brain(learning_uses_auditory=False)
        body = Physiology.capture(human)
        for percept, feature in ((SILENCE, (AuditoryKind.SILENCE,)), (MASKED, (AuditoryKind.MASKED, 0)),
                                  (identified(12, SourceDirection.NW), (AuditoryKind.IDENTIFIED, 12, SourceDirection.NW, 0))):
            obs = Observation(1, (), percept)
            self.assertEqual(old.state(body, obs), (1, 2, 1, 2, 2, 2, 2))
            self.assertEqual(brain.state(body, obs), (*old.state(body, obs), feature))
        for tick, percept in ((10, identified(7)), (11, identified(12)), (12, MASKED)):
            brain.choose_action(human, Observation(tick, (), percept))
            brain.observe_outcome(body)
        self.assertEqual(len(brain.auditory_memory.events(12)), 3)
        self.assertEqual(brain.state(body, Observation(13, ()))[-1], (AuditoryKind.MASKED, 1))
        self.assertEqual(len(brain.state(body, Observation(13, ()))), 8)
        self.assertEqual(brain.state(body, Observation(16, ()))[-1], (AuditoryKind.SILENCE,))

    def test_q_encoder_uses_latest_not_entire_history(self):
        a, human, _ = make_brain()
        b, _, _ = make_brain()
        a.auditory_memory.update(identified(1), 1)
        b.auditory_memory.update(MASKED, 1)
        for brain in (a, b):
            brain.auditory_memory.update(identified(7), 2)
        self.assertNotEqual(a.auditory_memory.events(3), b.auditory_memory.events(3))
        obs = Observation(3, ())
        self.assertEqual(a.state(Physiology.capture(human), obs), b.state(Physiology.capture(human), obs))

    def test_reward_formula_is_identical_for_silence_identified_and_masked(self):
        for percept in (SILENCE, identified(), MASKED):
            brain, human, _ = make_brain(learning_rate=0.5)
            brain.choose_action(human, Observation(1, (), percept))
            brain.observe_outcome(Physiology(30, 40, 70, True))
            self.assertAlmostEqual(brain.last_reward, 0.55)
            neutral, human, _ = make_brain()
            neutral.choose_action(human, Observation(1, (), percept))
            neutral.observe_outcome(Physiology.capture(human))
            self.assertEqual(neutral.last_reward, 0)

    def test_emission_stream_is_independent_of_q_values_auditory_and_body(self):
        first, a, _ = make_brain(seed=19, signal_probability=0.8)
        second, b, _ = make_brain(seed=19, signal_probability=0.8, learning_uses_auditory=False)
        reference = random.Random()
        reference.setstate(first.signal_rng.getstate())
        for tick in range(30):
            obs = Observation(tick, (), identified(tick % 16))
            b.hunger, b.thirst = tick, tick
            state = first.state(Physiology.capture(a), obs)
            first._values[state] = [float(i) for i in range(len(first.actions))]
            expected = reference.randrange(16) if reference.random() < 0.8 else None
            self.assertEqual(first.choose_action(a, obs).signal_id, expected)
            self.assertEqual(second.choose_action(b, Observation(tick, (), MASKED)).signal_id, expected)
            first.observe_outcome(Physiology.capture(a))
            second.observe_outcome(Physiology.capture(b))
        self.assertEqual(first.signal_rng.getstate(), reference.getstate())
        self.assertEqual(second.signal_rng.getstate(), reference.getstate())

    def test_rejected_decision_does_not_update_auditory_memory(self):
        brain, human, _ = make_brain()
        brain.choose_action(human, Observation(1, (), identified()))
        before = tuple(brain.auditory_memory._events)
        with self.assertRaises(RuntimeError):
            brain.choose_action(human, Observation(2, (), MASKED))
        self.assertEqual(tuple(brain.auditory_memory._events), before)

    def test_newborn_has_empty_uninherited_auditory_and_q_memory(self):
        cfg = Config.evolution(brain="learning", world_width=1, world_height=1, initial_population=2,
                               initial_food=0, initial_water=0, min_reproductive_age=0,
                               hunger_per_tick=0, thirst_per_tick=0, energy_per_tick=0, signal_probability=0)
        sim = Simulation(cfg)
        a, b = sim.agents
        a.reproductive_type, b.reproductive_type = 0, 1
        for parent in sim.agents:
            parent.brain.auditory_memory.update(identified(), 0)
            observation = replace(sim.observe(parent), tick=1)
            values = parent.brain._remember(parent.brain.state(Physiology.capture(parent), observation))
            values[parent.brain.actions.index(Action.REPRODUCE)] = 10
        with patch.object(a.brain.rng, "random", return_value=0.9), patch.object(b.brain.rng, "random", return_value=0.9):
            sim.step()
        self.assertEqual(sim.statistics.births, 1)
        child = sim.agents[-1]
        self.assertEqual(child.age, 0)
        self.assertEqual(child.brain.auditory_memory.events(sim.tick), ())
        self.assertIsNone(child.brain.auditory_memory._last_tick)
        self.assertEqual(child.brain.memory_size, 0)
        self.assertIsNone(child.brain._pending)
        self.assertIsNot(child.brain.auditory_memory, a.brain.auditory_memory)

    def test_config_validation_and_fields_not_genetic(self):
        self.assertEqual((Config().auditory_memory_ticks, Config().auditory_masking_ratio, Config().learning_uses_auditory),
                         (4, 3.0, True))
        for setting in ({"auditory_memory_ticks": 0}, {"auditory_memory_ticks": -1},
                        {"auditory_memory_ticks": 2.5}, {"auditory_memory_ticks": True},
                        {"auditory_masking_ratio": 1}, {"auditory_masking_ratio": -1},
                        {"auditory_masking_ratio": float("inf")}, {"auditory_masking_ratio": float("nan")},
                        {"auditory_masking_ratio": True}, {"learning_uses_auditory": 1}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                Config(**setting)
        Config(auditory_memory_ticks=1, auditory_masking_ratio=1.001, learning_uses_auditory=False)
        for ratio in (1, 0, float("inf"), float("nan"), True):
            with self.assertRaises(ValueError):
                Communication(16, 8, ratio)


class ListeningIntegrationTests(unittest.TestCase):
    def test_v04_random_and_disabled_learning_match_frozen_pre_change_controls(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/v04_listening_control.json").read_text())
        for run in fixture["runs"]:
            sink = ActionDigest()
            sim = Simulation(Config(**run["config"], learning_uses_auditory=run["config"]["brain"] == "random"), event_sink=sink)
            actual = capture(sim, fixture["ticks"])
            for key in actual:
                self.assertEqual(actual[key], run[key], (run["config"]["brain"], key))
            self.assertEqual(sink.digest.hexdigest(), run["actions_sha256"])

    def test_v05_enabled_memory_percepts_learning_and_rng_match_separate_fixture(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/v05_listening_seed7_tick80.json").read_text())
        actual = capture_listening(Config(**fixture["config"]), fixture["ticks"])
        self.assertEqual(actual, fixture["result"])

    def test_cli_auditory_ablation_and_both_demonstrations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.csv"
            result = subprocess.run([sys.executable, "main.py", "--brain", "learning", "--ticks", "2",
                                     "--population", "1", "--no-learning-uses-auditory", "--csv", str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(path.with_suffix(".csv.metadata.json").read_text())
            self.assertIs(metadata["config"]["learning_uses_auditory"], False)
            for name in ("masking_demo", "listening_demo"):
                result = subprocess.run([sys.executable, "-m", f"experiments.{name}"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsInstance(json.loads(result.stdout), dict)

    def test_same_physics_when_cognitive_access_changes_without_action_bias(self):
        cfg = Config.evolution(brain="learning", world_width=4, world_height=4, initial_population=8,
                               initial_food=12, initial_water=12, learning_epsilon=1, random_seed=0)
        a, b = Simulation(cfg), Simulation(replace(cfg, learning_uses_auditory=False))
        for _ in range(20):
            a.step()
            b.step()
            # Q state counts intentionally differ with cognitive access even when
            # epsilon=1 keeps every action and ecological quantity identical.
            for name, value in asdict(a.statistics).items():
                if name != "mean_memory_size":
                    self.assertEqual(value, getattr(b.statistics, name), name)
            self.assertEqual([a.observe(h) for h in a.agents], [b.observe(h) for h in b.agents])
            for x, y in zip(a.agents, b.agents):
                self.assertEqual(x.brain.rng.getstate(), y.brain.rng.getstate())
                self.assertEqual(x.brain.signal_rng.getstate(), y.brain.signal_rng.getstate())
                self.assertEqual(len(x.brain._pending.state), 8)
                self.assertEqual(len(y.brain._pending.state), 7)

    def test_enabled_replay_full_logging_and_extra_reads_leave_all_memory_rng_unchanged(self):
        cfg = Config.evolution(brain="learning", random_seed=7, world_width=6, world_height=5,
                               initial_population=12, initial_food=25, initial_water=25,
                               min_reproductive_age=3, reproduction_cooldown=5, mutation_rate=0.5)
        stream = io.StringIO()
        a, b = Simulation(cfg), Simulation(cfg, event_sink=JSONLEventWriter(stream))
        global_before = random.getstate()
        for _ in range(80):
            a.step()
            b.step()
            before = learning_state(b)
            for human in b.agents:
                human.brain.state(Physiology.capture(human), b.observe(human))
                human.brain.auditory_diagnostics(b.tick)
            self.assertEqual(learning_state(b), before)
            self.assertEqual(learning_state(a), learning_state(b))
        self.assertEqual(random.getstate(), global_before)
        self.assertGreater(a.statistics.births, 0)
        event = next(json.loads(line) for line in stream.getvalue().splitlines()
                     if '"auditory_learning"' in line)
        self.assertIn("memory", event["details"]["auditory_learning"])
        self.assertIn("resolved_percept", event["details"]["auditory_resolution"])

    def test_controlled_listening_changes_sound_conditioned_preferences(self):
        result = listening_demo()
        enabled, disabled = result["conditions"]
        self.assertEqual(enabled["learned_greedy_actions"]["7"], ["move_east"])
        self.assertGreater(enabled["learned_values"]["7"]["move_east"], 0)
        self.assertNotEqual(enabled["learned_values"]["7"], enabled["learned_values"]["12"])
        self.assertEqual(disabled["learned_values"]["7"], disabled["learned_values"]["12"])
        self.assertGreater(enabled["water_consumed_by_trial_condition"]["7"], 0)
        self.assertTrue(all(value == 0 for row in enabled["initial_values"].values() for value in row.values()))
        self.assertEqual(enabled["learning_updates"], result["trials"] * result["ticks_per_trial"] - 1)


if __name__ == "__main__":
    unittest.main()
