"""v0.6 dual-head learning, sender controls, complete replay and mechanism checks."""

from collections import Counter, OrderedDict
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, asdict, fields, replace
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agents.brain import Action, Decision, RandomBrain, VocalAction, VocalKind
from agents.genome import Genome
from agents.learning import LearningBrain, Physiology
from agents.observation import AuditoryKind, Observation, SILENCE, SourceDirection, TileObservation
from config import Config
from experiments.vocal_learning_demo import run_demo
from experiments.vocal_verification import initial_rate, short_condition
from simulation.events import JSONLEventWriter
from simulation.simulation import Simulation
from simulation.statistics import CSVStatisticsWriter, summarize_genomes, summarize_learning
from tests.auditory_control import ecological_state, encoded
from tests.test_learning import make_brain
from tests.test_listening import identified
from tests.test_statistics import FixedBrain

FIXTURES = Path(__file__).parent / "fixtures"


def vocal_state(sim):
    """Historical full ecological/physical state, plus every new mutable head field."""
    return (ecological_state(sim), tuple(
        (h.id, tuple(h.brain.auditory_memory._events), h.brain.auditory_memory._last_tick,
         tuple((key, tuple(values)) for key, values in h.brain._vocal_values.items()),
         asdict(h.brain._vocal_pending) if h.brain._vocal_pending is not None else None,
         h.brain.vocal_rng.getstate() if h.brain.vocal_rng is not None else None,
         h.brain.vocal_learning_updates, h.brain.vocal_exploratory_actions,
         h.brain.vocal_exploitative_actions, h.brain.last_vocal_action,
         h.brain.last_vocal_exploratory, h.brain.last_vocal_value)
        for h in sim.agents if isinstance(h.brain, LearningBrain)))


def capture_vocal(config, ticks):
    events, trajectory = hashlib.sha256(), hashlib.sha256()
    sim = Simulation(config, event_sink=lambda event: events.update(encoded(asdict(event))))
    for tick in range(ticks + 1):
        if tick:
            sim.step()
        state = vocal_state(sim)
        trajectory.update(encoded(state))
    return {"trajectory_sha256": trajectory.hexdigest(),
            "final_state_sha256": hashlib.sha256(encoded(state)).hexdigest(),
            "events_sha256": events.hexdigest(), "statistics": asdict(sim.statistics)}


def force_choices(human, sim, action, signal_id):
    """Test-only selection control; demonstrations never edit Q values."""
    brain = human.brain
    state = brain.state(Physiology.capture(human), replace(sim.observe(human), tick=sim.tick + 1))
    row = brain._remember(state)
    row[:] = [0.0] * len(row)
    row[brain.actions.index(action)] = 100
    vocal = brain._remember_vocal(state)
    vocal[:] = [0.0] * len(vocal)
    vocal[0 if signal_id is None else signal_id + 1] = 100


class VocalActionTests(unittest.TestCase):
    def test_explicit_silence_is_immutable_and_not_an_integer_signal(self):
        silence = VocalAction()
        self.assertIs(silence.kind, VocalKind.SILENCE)
        self.assertIsNone(silence.signal_id)
        self.assertEqual(silence.label, "SILENCE")
        with self.assertRaises(FrozenInstanceError):
            silence.signal_id = 0
        self.assertNotEqual(silence, VocalAction(VocalKind.SIGNAL, 0))

    def test_invalid_vocal_records_rejected(self):
        for kind, signal in (("SILENCE", None), (VocalKind.SILENCE, 0),
                             (VocalKind.SIGNAL, None), (VocalKind.SIGNAL, -1), (VocalKind.SIGNAL, True)):
            with self.subTest(kind=kind, signal=signal), self.assertRaises(ValueError):
                VocalAction(kind, signal)

    def test_decision_preserves_existing_api_and_adds_explicit_vocal_view(self):
        self.assertEqual(Decision(Action.WAIT).vocal_action, VocalAction())
        self.assertEqual(Decision(Action.EAT, 3).vocal_action, VocalAction(VocalKind.SIGNAL, 3))
        self.assertEqual([f.name for f in fields(Decision)], ["action", "signal_id"])

    def test_vocabulary_sizes_include_each_id_once_and_one_silence(self):
        for size in (4, 16, 40, 256):
            brain, _, _ = make_brain(signal_vocab_size=size)
            self.assertEqual(len(brain.vocal_actions), size + 1)
            self.assertEqual(brain.vocal_actions[0], VocalAction())
            self.assertEqual([a.signal_id for a in brain.vocal_actions[1:]], list(range(size)))

    def test_physical_actions_and_indices_unchanged(self):
        for reproduction in (False, True):
            brain, _, _ = make_brain(reproduction_enabled=reproduction)
            self.assertEqual(brain.actions, tuple(Action) if reproduction else tuple(Action)[:-1])
            self.assertNotIn(VocalKind.SILENCE, brain.actions)

    def test_new_config_defaults_validation_and_nonheritability(self):
        self.assertIs(Config().learning_controls_vocalization, True)
        self.assertEqual(Config().vocal_learning_memory_capacity, 256)
        for settings in ({"learning_controls_vocalization": 1}, {"learning_controls_vocalization": None},
                         {"vocal_learning_memory_capacity": 0}, {"vocal_learning_memory_capacity": -1},
                         {"vocal_learning_memory_capacity": True}, {"vocal_learning_memory_capacity": 1.5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Config(**settings)
        Config(vocal_learning_memory_capacity=1, learning_controls_vocalization=False)
        self.assertEqual({f.name for f in fields(Genome)},
                         {"perception_radius", "signal_probability", "hunger_multiplier", "thirst_multiplier"})


class VocalLearningTests(unittest.TestCase):
    def test_zero_prior_reads_do_not_allocate_or_consume_rng(self):
        brain, human, _ = make_brain()
        rngs = (brain.rng.getstate(), brain.signal_rng.getstate(), brain.vocal_rng.getstate(), random.getstate())
        with patch.object(random.Random, "random", side_effect=AssertionError("unexpected RNG")):
            for _ in range(3):
                self.assertEqual(brain.vocal_action_values(Physiology.capture(human), Observation(1, ())), (0.0,) * 17)
                brain.vocal_diagnostics()
        self.assertEqual((brain.memory_size, brain.vocal_memory_size), (0, 0))
        self.assertEqual(rngs, (brain.rng.getstate(), brain.signal_rng.getstate(), brain.vocal_rng.getstate(), random.getstate()))

    def test_same_encoded_state_and_shared_before_object(self):
        for listening in (False, True):
            brain, human, _ = make_brain(learning_uses_auditory=listening)
            brain.choose_action(human, Observation(1, (), identified(12)))
            self.assertIs(brain._pending.state, brain._vocal_pending.state)
            self.assertIs(brain._pending.before, brain._vocal_pending.before)
            self.assertEqual(len(brain._vocal_pending.state), 8 if listening else 7)
            if listening:
                self.assertEqual(brain._vocal_pending.state[-1], (AuditoryKind.IDENTIFIED, 12, SourceDirection.E, 0))

    def test_both_heads_receive_exact_same_reward_once(self):
        brain, human, _ = make_brain(learning_rate=0.5)
        brain.choose_action(human, Observation(1, ()))
        brain.observe_outcome(Physiology(30, 40, 70, True))
        self.assertAlmostEqual(brain.last_reward, 0.55)
        self.assertEqual(brain._pending.reward, brain._vocal_pending.reward)
        self.assertEqual(brain._pending.reward, brain.last_reward)
        self.assertEqual((brain.learning_updates, brain.vocal_learning_updates), (0, 0))

    def test_discounted_vocal_bootstrap_uses_own_next_values(self):
        brain, human, _ = make_brain(learning_rate=0.5, learning_discount=0.9)
        brain.choose_action(human, Observation(1, ()))
        transition = brain._vocal_pending
        brain.observe_outcome(Physiology(30, 40, 70, True))
        human.hunger, human.thirst, human.energy = 30, 40, 70
        next_state = brain.state(Physiology.capture(human), Observation(2, ()))
        brain._values[next_state] = [2.0] * len(brain.actions)
        brain._vocal_values[next_state] = [4.0] * len(brain.vocal_actions)
        brain.choose_action(human, Observation(2, ()))
        self.assertAlmostEqual(brain._vocal_values[transition.state][transition.action_index], 0.5 * (0.55 + 0.9 * 4))
        self.assertEqual((brain.learning_updates, brain.vocal_learning_updates), (1, 1))

    def test_terminal_both_heads_bootstrap_zero_and_clear_pending(self):
        brain, human, _ = make_brain(learning_rate=0.5)
        brain.choose_action(human, Observation(1, ()))
        physical, vocal = brain._pending, brain._vocal_pending
        brain.observe_outcome(Physiology(50, 80, 0, False))
        for rows, transition in ((brain._values, physical), (brain._vocal_values, vocal)):
            self.assertAlmostEqual(rows[transition.state][transition.action_index], -0.425)
        self.assertIsNone(brain._pending)
        self.assertIsNone(brain._vocal_pending)
        self.assertEqual((brain.learning_updates, brain.vocal_learning_updates), (1, 1))

    def test_outcome_contract_applies_to_both_heads_without_partial_mutation(self):
        brain, human, _ = make_brain()
        with self.assertRaises(RuntimeError):
            brain.observe_outcome(Physiology.capture(human))
        brain.choose_action(human, Observation(1, ()))
        before = (brain.vocal_rng.getstate(), list(brain._vocal_values), tuple(brain.auditory_memory._events))
        with self.assertRaises(RuntimeError):
            brain.choose_action(human, Observation(2, (), identified()))
        self.assertEqual(before, (brain.vocal_rng.getstate(), list(brain._vocal_values), tuple(brain.auditory_memory._events)))
        brain.observe_outcome(Physiology.capture(human))
        with self.assertRaises(RuntimeError):
            brain.observe_outcome(Physiology.capture(human))

    def test_no_signal_or_hearing_terms_enter_reward(self):
        rewards = []
        for signal in (None, 0, 15):
            brain, human, _ = make_brain()
            with patch.object(brain.vocal_rng, "random", return_value=0), \
                 patch.object(brain.vocal_rng, "randrange", return_value=0 if signal is None else signal + 1):
                decision = brain.choose_action(human, Observation(1, (), identified()))
            self.assertEqual(decision.signal_id, signal)
            brain.observe_outcome(Physiology(30, 40, 70, True))
            rewards.append(brain._vocal_pending.reward)
        self.assertEqual(rewards, [0.55] * 3)

    def test_receiver_identity_target_and_world_are_not_required(self):
        brain, human, _ = make_brain()
        body = SimpleNamespace(hunger=40, thirst=60, energy=50, alive=True)
        brain.choose_action(body, Observation(1, (), identified()))
        self.assertFalse(any(name in vars(brain) for name in ("world", "simulation", "receiver", "sender_id")))
        self.assertEqual(len(brain._vocal_pending.state), 8)
        self.assertEqual({f.name for f in fields(VocalAction)}, {"kind", "signal_id"})

    def test_learned_vocal_choices_ignore_genome_signal_probability(self):
        a, x, _ = make_brain(signal_probability=0)
        b, y, _ = make_brain(signal_probability=1)
        for tick in range(100):
            self.assertEqual(a.choose_action(x, Observation(tick, ())), b.choose_action(y, Observation(tick, ())))
            a.observe_outcome(Physiology.capture(x))
            b.observe_outcome(Physiology.capture(y))
        self.assertEqual(a.vocal_rng.getstate(), b.vocal_rng.getstate())

    def test_learned_emission_never_consumes_legacy_signal_rng(self):
        brain, human, _ = make_brain()
        before = brain.signal_rng.getstate()
        with patch.object(brain.signal_rng, "random", side_effect=AssertionError("random emission gate")):
            brain.choose_action(human, Observation(1, ()))
        self.assertEqual(brain.signal_rng.getstate(), before)

    def test_disabled_vocal_head_keeps_exact_v05_signal_and_physical_rng(self):
        brain, human, cfg = make_brain(seed=19, learning_controls_vocalization=False)
        physical = random.Random(19)
        legacy_signal = random.Random(physical.getrandbits(64))
        for tick in range(50):
            exploratory = physical.random() < cfg.learning_epsilon
            index = physical.randrange(len(brain.actions)) if exploratory else physical.choice(list(range(len(brain.actions))))
            signal = legacy_signal.randrange(cfg.signal_vocab_size) if legacy_signal.random() < human.genome.signal_probability else None
            self.assertEqual(brain.choose_action(human, Observation(tick, ())), Decision(brain.actions[index], signal))
            brain.observe_outcome(Physiology.capture(human))
        self.assertEqual(brain.rng.getstate(), physical.getstate())
        self.assertEqual(brain.signal_rng.getstate(), legacy_signal.getstate())
        self.assertIsNone(brain.vocal_rng)
        self.assertEqual((brain.vocal_memory_size, brain.vocal_learning_updates), (0, 0))
        self.assertEqual(brain.vocal_action_values(Physiology.capture(human), Observation(50, ())), ())

    def test_random_control_probability_zero_and_one(self):
        for probability in (0, 1):
            brain, human, _ = make_brain(signal_probability=probability, learning_controls_vocalization=False)
            decision = brain.choose_action(human, Observation(1, ()))
            self.assertEqual(decision.signal_id is None, probability == 0)
            self.assertIsNone(brain._vocal_pending)

    def test_vocal_draws_do_not_shift_physical_rng_or_q_on_matched_observations(self):
        a, x, _ = make_brain(seed=19, learning_controls_vocalization=False)
        b, y, _ = make_brain(seed=19)
        for tick in range(100):
            obs = Observation(tick, (), identified(tick % 16))
            self.assertEqual(a.choose_action(x, obs).action, b.choose_action(y, obs).action)
            after = Physiology(30 + tick % 3, 40, 60, True)
            a.observe_outcome(after)
            b.observe_outcome(after)
            self.assertEqual(a._values, b._values)
            self.assertEqual(a._pending, b._pending)
            self.assertEqual(a.rng.getstate(), b.rng.getstate())

    def test_named_vocal_rng_derivation_preserves_single_construction_draw(self):
        brain, _, _ = make_brain(seed=42)
        reference = random.Random(42)
        seed = reference.getrandbits(64)
        self.assertEqual(brain.rng.getstate(), reference.getstate())
        self.assertEqual(brain.vocal_rng.getstate(), random.Random(f"{seed}:vocal-choice").getstate())

    def test_exploration_uniformly_includes_silence_and_every_signal(self):
        brain, human, _ = make_brain(learning_epsilon=1, signal_vocab_size=4)
        seen = set()
        reference = random.Random()
        reference.setstate(brain.vocal_rng.getstate())
        for tick in range(200):
            reference.random()
            expected = reference.randrange(5)
            action = brain.choose_action(human, Observation(tick, ())).vocal_action
            self.assertEqual(action, brain.vocal_actions[expected])
            seen.add(action)
            brain.observe_outcome(Physiology.capture(human))
        self.assertEqual(seen, set(brain.vocal_actions))
        self.assertEqual((brain.vocal_exploratory_actions, brain.vocal_exploitative_actions), (200, 0))

    def test_exploit_max_ties_use_uniform_rng_choice_including_silence(self):
        brain, human, _ = make_brain(signal_vocab_size=4)
        state = brain.state(Physiology.capture(human), Observation(0, ()))
        brain._remember_vocal(state)[:] = [2, -1, 2, -1, 2]
        seen = set()
        reference = random.Random()
        reference.setstate(brain.vocal_rng.getstate())
        with patch.object(brain.vocal_rng, "random", return_value=0.9):
            for tick in range(100):
                expected = reference.choice([0, 2, 4])
                decision = brain.choose_action(human, Observation(tick, ()))
                self.assertEqual(decision.vocal_action, brain.vocal_actions[expected])
                seen.add(decision.vocal_action)
                # Terminal, zero-discount continuation and the exact target keep
                # all three maxima tied for this selection-mechanism test.
                brain.observe_outcome(Physiology(40, 60, 100, False))
                brain._remember_vocal(state)[:] = [2, -1, 2, -1, 2]
        self.assertEqual(seen, {brain.vocal_actions[i] for i in (0, 2, 4)})
        self.assertEqual(brain.vocal_exploitative_actions, 100)

    def test_head_tables_do_not_share_rows_or_values(self):
        brain, human, _ = make_brain()
        brain.choose_action(human, Observation(1, ()))
        self.assertIsInstance(brain._values, OrderedDict)
        self.assertIsInstance(brain._vocal_values, OrderedDict)
        state = brain._pending.state
        self.assertIsNot(brain._values[state], brain._vocal_values[state])
        brain._vocal_values[state][0] = 99
        self.assertEqual(brain._values[state], [0.0] * 7)

    def test_independent_lru_caps_and_recency(self):
        brain, human, _ = make_brain(learning_memory_capacity=2, vocal_learning_memory_capacity=3)
        states = []
        for tick, (dx, dy) in enumerate(((0, -1), (0, 1), (1, 0), (-1, 0))):
            obs = Observation(tick, (TileObservation(dx, dy, 1, 0),))
            states.append(brain.state(Physiology.capture(human), obs))
            brain.choose_action(human, obs)
            brain.observe_outcome(Physiology.capture(human))
        self.assertEqual(list(brain._values), states[-2:])
        self.assertEqual(list(brain._vocal_values), states[-3:])
        physical_order = list(brain._values)
        brain._remember_vocal(states[-3])
        self.assertEqual(list(brain._vocal_values), [states[-2], states[-1], states[-3]])
        self.assertEqual(list(brain._values), physical_order)

    def test_capacity_one_both_heads_handles_switching_states(self):
        brain, human, _ = make_brain(learning_memory_capacity=1, vocal_learning_memory_capacity=1)
        for tick in range(20):
            brain.choose_action(human, Observation(tick, (TileObservation(tick % 2, 0, 1, 0),)))
            brain.observe_outcome(Physiology.capture(human))
            self.assertEqual((brain.memory_size, brain.vocal_memory_size), (1, 1))
        self.assertEqual((brain.learning_updates, brain.vocal_learning_updates), (19, 19))

    def test_auditory_expiration_and_ablation_are_same_for_both_heads(self):
        for listening in (False, True):
            brain, human, _ = make_brain(learning_uses_auditory=listening)
            for tick in range(6):
                brain.choose_action(human, Observation(tick, (), identified(7) if tick == 0 else SILENCE))
                self.assertEqual(brain._pending.state, brain._vocal_pending.state)
                if listening:
                    self.assertEqual(brain._vocal_pending.state[-1],
                                     (AuditoryKind.IDENTIFIED, 7, SourceDirection.E, tick) if tick < 4
                                     else (AuditoryKind.SILENCE,))
                else:
                    self.assertEqual(len(brain._vocal_pending.state), 7)
                brain.observe_outcome(Physiology.capture(human))


class VocalIntegrationTests(unittest.TestCase):
    def simulation(self, **settings):
        cfg = Config(brain="learning", learning_uses_auditory=False, world_width=5, world_height=5,
                     initial_population=2, initial_food=0, initial_water=0,
                     hunger_per_tick=0, thirst_per_tick=0, min_reproductive_age=0, **settings)
        events = []
        sim = Simulation(cfg, event_sink=events.append)
        a, b = sim.agents
        a.x = b.x = a.y = b.y = 2
        a.reproductive_type, b.reproductive_type = 0, 1
        b.brain = FixedBrain()
        return sim, a, b, events

    def test_each_physical_action_can_execute_with_learned_signal(self):
        for action in Action:
            with self.subTest(action=action):
                sim, a, _, events = self.simulation(reproduction_enabled=True)
                tile = sim.world.tile_at(a.x, a.y)
                tile.food = tile.water = 1
                a.hunger = a.thirst = 50
                force_choices(a, sim, action, 7)
                with patch.object(a.brain.rng, "random", return_value=0.9), \
                     patch.object(a.brain.vocal_rng, "random", return_value=0.9):
                    sim.step()
                step = next(e for e in events if e.kind == "agent_step" and e.agent_id == a.id)
                self.assertTrue(step.details["action_succeeded"])
                self.assertEqual(step.details["signal_emitted"], 7)
                signals = [s for bucket in sim.communication._audible.values() for s in bucket]
                self.assertEqual([(s.signal_id, s.x, s.y) for s in signals], [(7, a.x, a.y)])
                if action is Action.REPRODUCE:
                    self.assertEqual(sim.statistics.births, 1)

    def test_delay_no_same_tick_hearing_and_expiration_with_silence(self):
        sim, a, b, events = self.simulation()
        force_choices(a, sim, Action.MOVE_EAST, 7)
        with patch.object(a.brain.rng, "random", return_value=0.9), \
             patch.object(a.brain.vocal_rng, "random", return_value=0.9):
            sim.step()
            self.assertTrue(all(e.details["observation"].auditory is SILENCE for e in events if e.kind == "agent_step"))
            self.assertEqual(sim.observe(b).auditory, identified(7))
            force_choices(a, sim, Action.WAIT, None)
            sim.step()
            receiver = next(e for e in events if e.tick == 2 and e.kind == "agent_step" and e.agent_id == b.id)
            self.assertEqual(receiver.details["observation"].auditory, identified(7))
            self.assertEqual(sim.observe(b).auditory, SILENCE)

    def test_silence_does_not_emit_or_increment_counter(self):
        sim, a, _, _ = self.simulation()
        force_choices(a, sim, Action.WAIT, None)
        with patch.object(a.brain.rng, "random", return_value=0.9), \
             patch.object(a.brain.vocal_rng, "random", return_value=0.9):
            sim.step()
        self.assertEqual(sim.statistics.signals_emitted, 0)
        self.assertFalse(sim.communication._audible)
        self.assertEqual(a.brain.last_vocal_action, VocalAction())

    def test_death_retains_already_emitted_signal_for_receiver_next_tick(self):
        sim, a, b, _ = self.simulation(max_hunger=1)
        a.hunger = 1
        force_choices(a, sim, Action.WAIT, 7)
        with patch.object(a.brain.rng, "random", return_value=0.9), \
             patch.object(a.brain.vocal_rng, "random", return_value=0.9):
            sim.step()
        self.assertNotIn(a, sim.agents)
        self.assertEqual(sim.observe(b).auditory, identified(7, SourceDirection.SAME_CELL))
        self.assertEqual((a.brain.learning_updates, a.brain.vocal_learning_updates), (1, 1))

    def test_newborn_has_two_empty_uninherited_heads_and_no_birth_tick_decision(self):
        sim, a, _, _ = self.simulation(reproduction_enabled=True)
        force_choices(a, sim, Action.REPRODUCE, 7)
        with patch.object(a.brain.rng, "random", return_value=0.9), \
             patch.object(a.brain.vocal_rng, "random", return_value=0.9):
            sim.step()
        child = sim.agents[-1]
        self.assertEqual(child.parent_a_id, a.id)
        self.assertEqual((child.age, child.brain.memory_size, child.brain.vocal_memory_size), (0, 0, 0))
        self.assertEqual((child.brain.learning_updates, child.brain.vocal_learning_updates), (0, 0))
        self.assertIsNone(child.brain._pending)
        self.assertIsNone(child.brain._vocal_pending)
        self.assertIsNot(child.brain._vocal_values, a.brain._vocal_values)
        self.assertNotEqual(child.brain.vocal_rng.getstate(), a.brain.vocal_rng.getstate())

    def test_sender_reward_unchanged_by_delivery_or_receiver_behavior(self):
        for in_range in (False, True):
            sim, a, b, _ = self.simulation(hearing_radius=0)
            if not in_range:
                b.x = b.y = 0
            tile = sim.world.tile_at(2, 2)
            tile.water = 1
            a.thirst = 60
            b.brain = FixedBrain(Action.EAT)
            force_choices(a, sim, Action.DRINK, 12)
            with patch.object(a.brain.rng, "random", return_value=0.9), \
                 patch.object(a.brain.vocal_rng, "random", return_value=0.9):
                sim.step()
            self.assertEqual(a.brain.last_reward, 30 / 80)
            self.assertEqual(a.brain._vocal_pending.reward, a.brain._pending.reward)
            self.assertEqual(sim.observe(b).auditory.kind is AuditoryKind.IDENTIFIED, in_range)

    def test_opt_in_events_distinguish_both_control_paths_and_heads(self):
        for vocal in (False, True):
            events = []
            sim = Simulation(Config(brain="learning", initial_population=1,
                                    learning_controls_vocalization=vocal), event_sink=events.append)
            sim.step()
            details = events[-1].details
            diag = details["vocal_learning"]
            self.assertEqual(diag["control"], "learned" if vocal else "random_control")
            self.assertEqual(diag["action"].signal_id, details["signal_emitted"])
            self.assertIn("exploratory", details["learning"])
            self.assertIn("memory_size", diag)
            self.assertEqual(diag["exploratory"] is None, not vocal)
            self.assertNotIn("q_table", diag)

    def test_no_diagnostic_payloads_built_without_full_logging(self):
        sim = Simulation(Config(brain="learning", initial_population=1))
        with patch.object(LearningBrain, "vocal_diagnostics", side_effect=AssertionError("diagnostics disabled")):
            sim.step()

    def test_four_ablations_execute_independently_and_randombrain_ignores_switch(self):
        for listening in (False, True):
            for vocal in (False, True):
                cfg = Config(brain="learning", initial_population=2, learning_uses_auditory=listening,
                             learning_controls_vocalization=vocal)
                sim = Simulation(cfg)
                sim.step()
                for human in sim.agents:
                    self.assertEqual(len(human.brain._pending.state), 8 if listening else 7)
                    self.assertEqual(human.brain.vocal_memory_size, 1 if vocal else 0)
        cfg = Config(initial_population=2)
        a, b = Simulation(cfg), Simulation(replace(cfg, learning_controls_vocalization=False))
        for _ in range(10):
            a.step()
            b.step()
            self.assertEqual(vocal_state(a), vocal_state(b))
        self.assertIsInstance(a.agents[0].brain, RandomBrain)

    def test_full_replay_logging_inspection_and_csv_preserve_both_heads_all_rng(self):
        cfg = Config.evolution(brain="learning", random_seed=7, world_width=6, world_height=5,
                               initial_population=12, initial_food=25, initial_water=25,
                               min_reproductive_age=3, reproduction_cooldown=5, mutation_rate=0.5)
        global_before = random.getstate()
        a, b = Simulation(cfg), Simulation(cfg, event_sink=JSONLEventWriter(io.StringIO()))
        writer = CSVStatisticsWriter(io.StringIO())
        for _ in range(80):
            a.step()
            b.step()
            before = vocal_state(b)
            writer.write(b.statistics)
            summarize_genomes(b.agents)
            summarize_learning(b.agents)
            for human in b.agents:
                body, obs = Physiology.capture(human), b.observe(human)
                human.brain.vocal_action_values(body, obs)
                human.brain.action_values(body, obs)
                human.brain.auditory_diagnostics(b.tick)
                human.brain.vocal_diagnostics()
            self.assertEqual(vocal_state(b), before)
            self.assertEqual(vocal_state(a), vocal_state(b))
            self.assertEqual(b.population, cfg.initial_population + b.statistics.births - b.statistics.deaths)
        self.assertEqual(random.getstate(), global_before)

    def test_v06_full_fixture_and_pre_change_baseline_record(self):
        fixture = json.loads((FIXTURES / "v06_vocal_seed7_tick80.json").read_text())
        self.assertEqual(fixture["pre_change_tests"]["count"], 194)
        self.assertEqual(fixture["pre_change_tests"]["result"], "OK")
        self.assertEqual(capture_vocal(Config(**fixture["config"]), fixture["ticks"]), fixture["result"])

    def test_historical_fixtures_have_not_been_regenerated(self):
        fixture = json.loads((FIXTURES / "v06_vocal_seed7_tick80.json").read_text())
        for filename, expected in fixture["historical_fixture_hashes"].items():
            self.assertEqual(hashlib.sha256(Path(filename).read_bytes()).hexdigest(), expected)

    def test_vocal_demo_only_ordinary_reward_learns_different_arbitrary_preferences(self):
        result = run_demo()
        self.assertTrue(all(value == 0 for row in result["initial_values"].values()
                            for head in row.values() for value in head.values()))
        self.assertTrue(result["different_vocal_preferences"])
        self.assertNotEqual(result["learned_values"]["context_a"]["vocal"],
                            result["learned_values"]["context_b"]["vocal"])
        self.assertEqual((result["physical_learning_updates"], result["vocal_learning_updates"]), (599, 599))
        self.assertGreater(result["food_consumed"], 0)
        self.assertGreater(result["water_consumed"], 0)
        self.assertEqual(result["config"]["initial_population"], 1)

    def test_initial_policy_rate_reports_actual_unbiased_counts_without_gating(self):
        first, second = initial_rate(), initial_rate()
        self.assertEqual(first, second)
        self.assertEqual(first["emitted"] + first["silent"], 1000)
        self.assertEqual(first["expected_emission_rate"], 16 / 17)
        self.assertGreater(first["silent"], 0)
        self.assertGreater(first["emitted"], 900)
        self.assertEqual(len(first["counts"]), 17)

    def test_short_masking_check_counts_observations_and_emissions(self):
        result = short_condition(True, True, ticks=10)
        self.assertEqual(sum(result["percepts"].values()), result["action_opportunities"])
        self.assertGreater(result["percepts"]["MASKED"], 0)
        self.assertEqual(result["emission_rate"], result["signals_emitted"] / result["action_opportunities"])
        self.assertGreater(result["vocal_updates_cumulative"], 0)

    def test_cli_independent_sender_flags_metadata_and_demo_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            result = subprocess.run([sys.executable, "main.py", "--brain", "learning", "--ticks", "2",
                                     "--population", "1", "--no-learning-controls-vocalization", "--csv", str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads(path.with_suffix(".csv.metadata.json").read_text())
            self.assertEqual(metadata["version"], "0.6.1")
            self.assertIs(metadata["config"]["learning_controls_vocalization"], False)
            self.assertIs(metadata["config"]["learning_uses_auditory"], True)
        result = subprocess.run([sys.executable, "-m", "experiments.vocal_learning_demo", "--trials", "20"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["trials"], 20)


if __name__ == "__main__":
    unittest.main()
