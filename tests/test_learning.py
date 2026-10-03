from dataclasses import asdict
import hashlib
import inspect
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

from agents.brain import Action, RandomBrain
from agents.genome import Genome
from agents.human import Human
from agents.learning import LearningBrain, Physiology
from agents.observation import HeardSignal, Observation, SourceDirection, TileObservation
from config import Config
from experiments.learning_demo import run_demo
from simulation.events import JSONLEventWriter
from simulation.simulation import Simulation
from simulation.statistics import CSVStatisticsWriter, summarize_genomes, summarize_learning
from tests.test_statistics import dynamics


def make_brain(seed=0, **settings):
    config = Config(brain="learning", **settings)
    brain = LearningBrain(random.Random(seed), config)
    human = Human(0, 0, 0, brain, hunger=40, thirst=60, energy=50, genome=Genome.founder(config))
    return brain, human, config


def learning_state(sim):
    return (
        dynamics(sim),
        tuple((h.id, tuple((key, tuple(values)) for key, values in h.brain._values.items()),
               asdict(h.brain._pending) if h.brain._pending is not None else None,
               h.brain.signal_rng.getstate(), h.brain.learning_updates,
               h.brain.exploratory_actions, h.brain.exploitative_actions,
               h.brain.last_reward, h.brain.last_exploratory) for h in sim.agents),
    )


class LearningBrainTests(unittest.TestCase):
    def test_random_brain_ignores_observation_and_keeps_original_draw_sequence(self):
        config = Config.evolution()
        brain = RandomBrain(random.Random(42), config)
        reference = random.Random(42)
        human = Human(0, 0, 0, brain)
        empty = Observation(0, ())
        noisy = Observation(900, (TileObservation(1, 0, 999, 999),), (HeardSignal(15, SourceDirection.W),))
        for tick in range(100):
            action = reference.choice(tuple(Action))
            signal = reference.randrange(config.signal_vocab_size) if reference.random() < human.genome.signal_probability else None
            decision = brain.choose_action(human, empty if tick % 2 else noisy)
            self.assertEqual((decision.action, decision.signal_id), (action, signal))
        self.assertEqual(brain.rng.getstate(), reference.getstate())

    def test_memory_starts_empty_and_diagnostics_do_not_create_state(self):
        brain, human, _ = make_brain()
        before_rng = brain.rng.getstate()
        self.assertEqual(brain.memory_size, 0)
        self.assertEqual(brain.action_values(Physiology.capture(human), Observation(0, ())), (0.0,) * 7)
        self.assertEqual(brain.memory_size, 0)
        self.assertEqual(brain.rng.getstate(), before_rng)
        self.assertIsNone(brain._pending)

    def test_body_and_observation_are_sufficient_without_world_or_position(self):
        brain, human, _ = make_brain()
        # Deliberately lacks x, y, id, age, world, and any simulation reference.
        body_only = SimpleNamespace(hunger=human.hunger, thirst=human.thirst, energy=human.energy,
                                    alive=True, genome=human.genome)
        self.assertIn(brain.choose_action(body_only, Observation(0, ())).action, brain.actions)
        self.assertEqual(list(inspect.signature(brain.choose_action).parameters), ["human", "observation"])
        self.assertFalse(any(name in vars(brain) for name in ("world", "simulation", "human")))

    def test_reward_and_q_update_are_exact_and_wait_for_real_next_observation(self):
        brain, human, _ = make_brain(learning_rate=0.5, learning_discount=0.9)
        observation = Observation(1, ())
        state = brain.state(Physiology.capture(human), observation)
        action = brain.choose_action(human, observation).action
        brain.observe_outcome(Physiology(30, 40, 70, True))
        self.assertAlmostEqual(brain.last_reward, 0.55)
        self.assertEqual(brain.learning_updates, 0)
        self.assertTrue(all(value == 0 for value in brain._values[state]))
        human.hunger, human.thirst, human.energy = 30, 40, 70
        next_state = brain.state(Physiology.capture(human), observation)
        brain._values[next_state] = [2.0] * len(brain.actions)
        brain.choose_action(human, observation)
        self.assertAlmostEqual(brain._values[state][brain.actions.index(action)], 0.5 * (0.55 + 0.9 * 2))
        self.assertEqual(brain.learning_updates, 1)

    def test_terminal_update_has_no_future_bootstrap(self):
        brain, human, _ = make_brain(learning_rate=0.5)
        observation = Observation(1, ())
        state = brain.state(Physiology.capture(human), observation)
        action = brain.choose_action(human, observation).action
        brain.observe_outcome(Physiology(50, 80, 0, False))
        self.assertAlmostEqual(brain.last_reward, -0.85)
        self.assertAlmostEqual(brain._values[state][brain.actions.index(action)], -0.425)
        self.assertEqual(brain.learning_updates, 1)
        self.assertIsNone(brain._pending)

    def test_outcome_contract_rejects_missing_or_duplicate_feedback(self):
        brain, human, _ = make_brain()
        observation = Observation(1, ())
        with self.assertRaises(RuntimeError):
            brain.observe_outcome(Physiology.capture(human))
        brain.choose_action(human, observation)
        with self.assertRaises(RuntimeError):
            brain.choose_action(human, Observation(999, ()))
        brain.observe_outcome(Physiology.capture(human))
        with self.assertRaises(RuntimeError):
            brain.observe_outcome(Physiology.capture(human))

    def test_memory_capacity_and_lru_eviction(self):
        brain, human, _ = make_brain(learning_memory_capacity=2)
        states = []
        for x, y in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            observation = Observation(1, (TileObservation(x, y, 1, 0),))
            states.append(brain.state(Physiology.capture(human), observation))
            brain.choose_action(human, observation)
            brain.observe_outcome(Physiology.capture(human))
            self.assertLessEqual(brain.memory_size, 2)
        self.assertEqual(list(brain._values), states[-2:])
        self.assertNotIn(states[0], brain._values)

    def test_capacity_one_handles_transition_to_new_state(self):
        brain, human, _ = make_brain(learning_memory_capacity=1)
        for tick in range(20):
            observation = Observation(tick, (TileObservation(tick % 2, 0, 1, 0),))
            brain.choose_action(human, observation)
            brain.observe_outcome(Physiology.capture(human))
            self.assertEqual(brain.memory_size, 1)
        self.assertEqual(brain.learning_updates, 19)

    def test_observed_resources_condition_learned_values(self):
        brain, human, _ = make_brain()
        here = Observation(0, (TileObservation(0, 0, 0, 1),))
        elsewhere = Observation(0, (TileObservation(1, 0, 0, 1),))
        action = brain.choose_action(human, here).action
        brain.observe_outcome(Physiology(40, 30, 50, False))
        values = brain.action_values(Physiology.capture(human), here)
        self.assertGreater(values[brain.actions.index(action)], 0)
        self.assertEqual(brain.action_values(Physiology.capture(human), elsewhere), (0.0,) * len(brain.actions))

    def test_nearest_resource_feature_is_order_independent_and_has_no_action_prior(self):
        brain, human, _ = make_brain()
        tiles = (TileObservation(0, -1, 1, 0), TileObservation(1, 0, 1, 0), TileObservation(0, 0, 0, 2))
        body = Physiology.capture(human)
        self.assertEqual(brain.state(body, Observation(1, tiles)), brain.state(body, Observation(1, tuple(reversed(tiles)))))
        self.assertEqual(brain.action_values(body, Observation(1, tiles)), (0.0,) * len(brain.actions))

    def test_no_information_about_resources_outside_perception(self):
        sim = Simulation(Config(brain="learning", initial_population=1, initial_food=0, initial_water=0,
                                world_width=5, world_height=5, perception_radius=1))
        human = sim.agents[0]
        human.x = human.y = 0
        before = sim.observe(human)
        sim.world.tile_at(4, 4).food = sim.world.tile_at(4, 4).water = 100
        self.assertEqual(sim.observe(human), before)
        state = human.brain.state(Physiology.capture(human), before)
        sim.world.tile_at(1, 0).water = 1
        self.assertNotEqual(human.brain.state(Physiology.capture(human), sim.observe(human)), state)

    def test_zero_radius_and_empty_edge_observations_are_safe(self):
        sim = Simulation(Config(brain="learning", world_width=1, world_height=1, initial_population=1,
                                initial_food=0, initial_water=0, perception_radius=0))
        human = sim.agents[0]
        self.assertEqual(len(sim.observe(human).tiles), 1)
        sim.step()
        self.assertEqual((human.x, human.y), (0, 0))
        brain, human, _ = make_brain()
        self.assertIn(brain.choose_action(human, Observation(0, ())).action, brain.actions)

    def test_signals_and_tick_numbers_do_not_enter_learning_state(self):
        first, a, _ = make_brain(seed=10)
        second, b, _ = make_brain(seed=10)
        for tick in range(20):
            empty = Observation(tick, ())
            signals = Observation(tick + 1000, (), (HeardSignal(tick % 16, tuple(SourceDirection)[tick % 9]),))
            self.assertEqual(first.choose_action(a, empty), second.choose_action(b, signals))
            first.observe_outcome(Physiology.capture(a))
            second.observe_outcome(Physiology.capture(b))
        self.assertEqual(first._values, second._values)

    def test_signal_emission_does_not_shift_exploration_stream(self):
        first, a, _ = make_brain(seed=10, signal_probability=0)
        second, b, _ = make_brain(seed=10, signal_probability=1, signal_vocab_size=4000)
        for _ in range(30):
            quiet = first.choose_action(a, Observation(0, ()))
            loud = second.choose_action(b, Observation(0, ()))
            self.assertEqual(quiet.action, loud.action)
            self.assertIsNone(quiet.signal_id)
            self.assertTrue(0 <= loud.signal_id < 4000)
            first.observe_outcome(Physiology.capture(a))
            second.observe_outcome(Physiology.capture(b))
        self.assertEqual(first.rng.getstate(), second.rng.getstate())

    def test_seeded_exploration_is_reproducible_and_uses_no_global_randomness(self):
        global_before = random.getstate()
        histories = []
        for seed in (3, 3, 4):
            brain, human, _ = make_brain(seed=seed, learning_epsilon=1)
            history = []
            for _ in range(30):
                history.append(brain.choose_action(human, Observation(0, ())))
                brain.observe_outcome(Physiology.capture(human))
            self.assertEqual(brain.exploratory_actions, 30)
            self.assertEqual(brain.exploitative_actions, 0)
            histories.append(history)
        self.assertEqual(histories[0], histories[1])
        self.assertNotEqual(histories[0], histories[2])
        self.assertEqual(random.getstate(), global_before)

    def test_exploitation_breaks_untrained_ties_without_permanent_first_action_bias(self):
        selected = set()
        for seed in range(40):
            brain, human, _ = make_brain(seed=seed)
            with patch.object(brain.rng, "random", return_value=0.9):
                selected.add(brain.choose_action(human, Observation(0, ())).action)
            self.assertEqual(brain.exploitative_actions, 1)
        self.assertEqual(selected, set(Action) - {Action.REPRODUCE})

    def test_controlled_actual_physiology_changes_action_preference(self):
        result = run_demo()
        self.assertTrue(all(value == 0 for value in result["initial_values"].values()))
        self.assertEqual(len(result["initial_greedy_actions"]), 7)
        self.assertEqual(result["learned_greedy_actions"], ["drink"])
        self.assertGreater(result["learned_values"]["drink"], 0)
        self.assertGreater(result["actions"]["drink"], result["trials"] / 2)
        self.assertEqual(result["learning_updates"], result["trials"] - 1)


class LearningSimulationTests(unittest.TestCase):
    def test_random_seed_zero_matches_pre_v03_state_and_rng_fixture(self):
        fixture = json.loads((Path(__file__).parent / "fixtures" / "v02_seed0_tick100.json").read_text())
        # Keep the original hearing/bucket radius: the historical digest includes
        # exact bucket keys, in addition to all original state and RNG checks.
        sim = Simulation(Config.evolution(random_seed=fixture["seed"], signal_range=3))
        for _ in range(fixture["ticks"]):
            sim.step()
        state = list(dynamics(sim))
        state[2] = {name: state[2][name] for name in fixture["statistics"]}
        self.assertEqual(state[2], fixture["statistics"])
        digest = hashlib.sha256(json.dumps(state, default=asdict, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest, fixture["state_sha256"])

    def test_config_validation_and_default_random_selection(self):
        self.assertIsInstance(Simulation(Config(initial_population=1)).agents[0].brain, RandomBrain)
        for settings in ({"brain": "other"}, {"learning_rate": 0}, {"learning_rate": 1.1},
                         {"learning_discount": -1}, {"learning_discount": 1.1},
                         {"learning_epsilon": 0}, {"learning_epsilon": 1.1},
                         {"learning_memory_capacity": 0}, {"learning_memory_capacity": 1.5},
                         {"learning_need_bins": 0}, {"learning_rate": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Config(**settings)

    def test_agents_and_offspring_have_distinct_fresh_memory(self):
        config = Config.evolution(brain="learning", world_width=1, world_height=1, initial_population=2,
                                  initial_food=0, initial_water=0, min_reproductive_age=0,
                                  hunger_per_tick=0, thirst_per_tick=0, energy_per_tick=0)
        sim = Simulation(config)
        a, b = sim.agents
        a.reproductive_type, b.reproductive_type = 0, 1
        self.assertIsNot(a.brain._values, b.brain._values)
        for human in sim.agents:
            state = human.brain.state(Physiology.capture(human), sim.observe(human))
            row = human.brain._remember(state)
            row[human.brain.actions.index(Action.REPRODUCE)] = 10
        with patch.object(a.brain.rng, "random", return_value=0.9), patch.object(b.brain.rng, "random", return_value=0.9):
            sim.step()
        self.assertEqual(sim.statistics.births, 1)
        child = sim.agents[-1]
        self.assertIsInstance(child.brain, LearningBrain)
        self.assertEqual((child.age, child.brain.memory_size, child.brain.learning_updates), (0, 0, 0))
        self.assertIsNone(child.brain._pending)
        self.assertIsNot(child.brain._values, a.brain._values)
        self.assertIsNot(child.brain._values, b.brain._values)

    def test_full_learning_replay_logging_statistics_and_invariants(self):
        config = Config.evolution(brain="learning", world_width=5, world_height=5, initial_population=12,
                                  initial_food=20, initial_water=20, min_reproductive_age=4,
                                  reproduction_cooldown=5, mutation_rate=0.5)
        global_before = random.getstate()
        stream = io.StringIO()
        first = Simulation(config)
        second = Simulation(config, event_sink=JSONLEventWriter(stream))
        writer = CSVStatisticsWriter(io.StringIO())
        expected_actions = 0
        for _ in range(180):
            expected_actions += first.population
            first.step()
            second.step()
            before = learning_state(second)
            summarize_learning(second.agents)
            summarize_genomes(second.agents)
            writer.write(second.statistics)
            self.assertEqual(learning_state(second), before)
            self.assertEqual(learning_state(first), learning_state(second))
            stats = first.statistics
            self.assertEqual(stats.learning_agents, first.population)
            self.assertEqual(stats.exploratory_actions + stats.exploitative_actions, expected_actions)
            self.assertEqual(stats.population, config.initial_population + stats.births - stats.deaths)
            self.assertEqual(len({h.id for h in first.agents}), first.population)
            for resource in ("food", "water"):
                remaining = sum(getattr(t, resource) for row in first.world.tiles for t in row)
                self.assertEqual(remaining, getattr(config, f"initial_{resource}")
                                 + getattr(stats, f"{resource}_regenerated") - getattr(stats, f"{resource}_consumed"))
            for name, value in summarize_genomes(first.agents).items():
                self.assertEqual(getattr(stats, name), value)
            self.assertTrue(all(h.brain.memory_size <= config.learning_memory_capacity for h in first.agents))
        self.assertGreater(first.statistics.total_learning_updates, 0)
        self.assertEqual(random.getstate(), global_before)
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertTrue(any("learning" in e["details"] for e in events if e["kind"] == "agent_step"))

    def test_death_preserves_cumulative_learning_counts_and_empty_means(self):
        sim = Simulation(Config(brain="learning", initial_population=1, max_hunger=1))
        stats = sim.step()
        self.assertEqual((stats.population, stats.learning_agents, stats.total_learning_updates), (0, 0, 1))
        self.assertEqual(stats.exploratory_actions + stats.exploitative_actions, 1)
        self.assertIsNone(stats.mean_learning_updates)
        self.assertIsNone(stats.mean_memory_size)
        self.assertEqual(sim.step(), stats)
        empty = Simulation(Config(brain="learning", initial_population=0))
        self.assertEqual(empty.step().total_learning_updates, 0)

    def test_mixed_population_learning_means_count_only_living_learners(self):
        sim = Simulation(Config(brain="learning", initial_population=3))
        sim.agents[0].brain = RandomBrain(random.Random(1), sim.config)
        sim.step()
        stats = sim.step()
        self.assertEqual(stats.learning_agents, 2)
        self.assertEqual(stats.total_learning_updates, 2)
        self.assertEqual(stats.mean_learning_updates, 1)
        self.assertGreaterEqual(stats.mean_memory_size, 1)

    def test_cli_metadata_identifies_mode_brain_seed_and_full_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            subprocess.run([sys.executable, "main.py", "--mode", "evolution", "--brain", "learning",
                            "--seed", "7", "--ticks", "2", "--population", "2", "--csv", str(path)],
                           check=True, capture_output=True, text=True)
            import csv
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            metadata = json.loads(path.with_suffix(".csv.metadata.json").read_text())
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertEqual((row["mode"], row["brain"], row["seed"]), ("evolution", "learning", "7"))
                self.assertEqual(row["config_id"], metadata["config_id"])
            self.assertEqual(metadata["config"]["learning_memory_capacity"], Config().learning_memory_capacity)
            self.assertEqual(metadata["config"]["brain"], "learning")

    def test_csv_metadata_cannot_overwrite_metrics(self):
        with self.assertRaises(ValueError):
            CSVStatisticsWriter(io.StringIO(), metadata={"population": 999})


if __name__ == "__main__":
    unittest.main()
