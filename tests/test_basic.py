import csv
from dataclasses import FrozenInstanceError, asdict, replace
import io
import json
import random
import unittest

from agents.brain import Action, Decision
from agents.observation import HeardSignal, SourceDirection
from config import Config
from simulation.events import JSONLEventWriter
from simulation.simulation import Simulation
from simulation.statistics import CSVStatisticsWriter
from systems.communication import Communication
from world.world import World


class FixedBrain:
    def __init__(self, action=Action.WAIT, signal=None):
        self.action = action
        self.signal = signal
        self.observations = []

    def choose_action(self, human, observation):
        self.observations.append(observation)
        return Decision(self.action, self.signal if observation.tick == 1 else None)


def make_sim(**overrides):
    config = Config(
        world_width=5, world_height=5, initial_population=1,
        initial_food=0, initial_water=0, hunger_per_tick=0,
        thirst_per_tick=0, max_age=None, signal_probability=0,
    )
    sim = Simulation(replace(config, **overrides))
    for human in sim.agents:
        human.brain = FixedBrain()
    return sim


def resource_state(world):
    return tuple((tile.food, tile.water) for row in world.tiles for tile in row)


def body_state(sim):
    return tuple(
        (h.id, h.x, h.y, h.age, h.hunger, h.thirst, h.energy, h.alive, h.reproductive_type)
        for h in sim.agents
    )


class WorldTests(unittest.TestCase):
    def test_world_generation_is_deterministic_and_preserves_resource_totals(self):
        config = Config(world_width=3, world_height=2, initial_food=100, initial_water=90)
        first = World.generate(config, random.Random(123))
        second = World.generate(config, random.Random(123))
        self.assertEqual(resource_state(first), resource_state(second))
        self.assertEqual(sum(t.food for row in first.tiles for t in row), 100)
        self.assertEqual(sum(t.water for row in first.tiles for t in row), 90)
        self.assertNotEqual(resource_state(first), resource_state(World.generate(config, random.Random(124))))

    def test_movement_cannot_leave_any_world_edge(self):
        sim = make_sim()
        human = sim.agents[0]
        for x, y, dx, dy in ((0, 0, -1, 0), (0, 0, 0, -1), (4, 4, 1, 0), (4, 4, 0, 1)):
            with self.subTest(x=x, y=y, dx=dx, dy=dy):
                human.x, human.y = x, y
                self.assertFalse(sim.world.move(human, dx, dy))
                self.assertEqual((human.x, human.y), (x, y))
        human.x, human.y = 2, 2
        self.assertTrue(sim.world.move(human, 1, 0))
        self.assertEqual((human.x, human.y), (3, 2))
        with self.assertRaises(ValueError):
            sim.world.move(human, 2, 0)
        with self.assertRaises(IndexError):
            sim.world.tile_at(-1, 0)

    def test_one_tile_world_blocks_all_moves(self):
        sim = make_sim(world_width=1, world_height=1)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            self.assertFalse(sim.world.move(sim.agents[0], dx, dy))

    def test_observation_is_local_relative_and_immutable(self):
        sim = make_sim(perception_radius=1)
        human = sim.agents[0]
        human.x, human.y = 0, 0
        sim.world.tile_at(0, 0).food = 2
        sim.world.tile_at(2, 0).food = 99
        observation = sim.observe(human)
        self.assertEqual({(t.dx, t.dy) for t in observation.tiles}, {(0, 0), (1, 0), (0, 1), (1, 1)})
        self.assertFalse(any(t.food == 99 for t in observation.tiles))
        sim.world.consume_food(0, 0)
        self.assertEqual(observation.tiles[0].food, 2)
        with self.assertRaises(FrozenInstanceError):
            observation.tiles[0].food = 10

    def test_zero_perception_radius_sees_only_current_tile(self):
        sim = make_sim(perception_radius=0)
        observation = sim.observe(sim.agents[0])
        self.assertEqual(len(observation.tiles), 1)
        self.assertEqual((observation.tiles[0].dx, observation.tiles[0].dy), (0, 0))


class SurvivalTests(unittest.TestCase):
    def test_eating_reduces_hunger_and_consumes_one_unit_before_tick_cost(self):
        sim = make_sim(hunger_per_tick=1)
        human = sim.agents[0]
        human.brain = FixedBrain(Action.EAT)
        human.hunger = 40
        sim.world.tile_at(human.x, human.y).food = 2
        stats = sim.step()
        self.assertEqual(human.hunger, 11)
        self.assertEqual(sim.world.tile_at(human.x, human.y).food, 1)
        self.assertEqual(stats.food_consumed, 1)
        self.assertEqual(human.age, 1)

    def test_drinking_reduces_thirst_and_consumes_one_unit_before_tick_cost(self):
        sim = make_sim(thirst_per_tick=1)
        human = sim.agents[0]
        human.brain = FixedBrain(Action.DRINK)
        human.thirst = 40
        sim.world.tile_at(human.x, human.y).water = 2
        self.assertEqual(sim.step().water_consumed, 1)
        self.assertEqual(human.thirst, 11)
        self.assertEqual(sim.world.tile_at(human.x, human.y).water, 1)

    def test_empty_resources_do_not_reduce_needs_or_count_consumption(self):
        for action in (Action.EAT, Action.DRINK):
            with self.subTest(action=action):
                sim = make_sim()
                human = sim.agents[0]
                human.hunger = human.thirst = 10
                human.brain = FixedBrain(action)
                sim.step()
                self.assertEqual((human.hunger, human.thirst), (10, 10))
                self.assertEqual((sim.statistics.food_consumed, sim.statistics.water_consumed), (0, 0))

    def test_needs_are_clamped_at_zero(self):
        for action, resource, need in ((Action.EAT, "food", "hunger"), (Action.DRINK, "water", "thirst")):
            with self.subTest(action=action):
                sim = make_sim()
                human = sim.agents[0]
                human.brain = FixedBrain(action)
                setattr(human, need, 1)
                setattr(sim.world.tile_at(human.x, human.y), resource, 1)
                sim.step()
                self.assertEqual(getattr(human, need), 0)

    def test_starvation_and_dehydration_at_exact_threshold(self):
        for settings in ({"max_hunger": 2, "hunger_per_tick": 1}, {"max_thirst": 2, "thirst_per_tick": 1}):
            with self.subTest(settings=settings):
                sim = make_sim(**settings)
                human = sim.agents[0]
                sim.step()
                self.assertTrue(human.alive)
                sim.step()
                self.assertFalse(human.alive)
                self.assertEqual((sim.population, sim.statistics.deaths), (0, 1))
                stats = sim.statistics
                self.assertEqual(sim.step(), stats)
                self.assertEqual(human.age, 2)

    def test_consumption_can_prevent_death_on_threshold_tick(self):
        sim = make_sim(max_hunger=5, hunger_per_tick=1)
        human = sim.agents[0]
        human.hunger = 4
        human.brain = FixedBrain(Action.EAT)
        sim.world.tile_at(human.x, human.y).food = 1
        sim.step()
        self.assertTrue(human.alive)
        self.assertEqual(human.hunger, 1)

    def test_maximum_age_and_disabled_age_limit(self):
        for maximum, expected_alive in ((2, False), (None, True)):
            sim = make_sim(max_age=maximum)
            human = sim.agents[0]
            sim.step()
            sim.step()
            self.assertEqual(human.alive, expected_alive)

    def test_shared_resource_is_consumed_only_once(self):
        sim = make_sim(world_width=1, world_height=1, initial_population=2)
        sim.world.tile_at(0, 0).food = 1
        for human in sim.agents:
            human.hunger = 40
            human.brain = FixedBrain(Action.EAT)
        self.assertEqual(sim.step().food_consumed, 1)
        self.assertEqual(sorted(h.hunger for h in sim.agents), [10, 40])
        self.assertEqual(sim.world.tile_at(0, 0).food, 0)


class CommunicationTests(unittest.TestCase):
    def test_signal_range_inclusive_boundary_self_exclusion_and_expiry(self):
        comm = Communication(16, 2)
        comm.emit(15, 1, 3, 3)
        self.assertEqual(comm.hear(2, 3, 3), ())
        comm.advance()
        heard = comm.hear(2, 5, 5)
        self.assertEqual(len(heard), 1)
        self.assertEqual(heard[0], HeardSignal(15, SourceDirection.NW))
        diagnostic = comm.diagnostics(2, 5, 5, receiving_tick=2)[0]
        self.assertEqual((diagnostic["sender_id"], diagnostic["sender_position"]), (1, {"x": 3, "y": 3}))
        self.assertEqual(comm.hear(2, 6, 3), ())
        self.assertEqual(comm.hear(1, 3, 3), ())
        comm.advance()
        self.assertEqual(comm.hear(2, 3, 3), ())

    def test_spatial_buckets_match_distance_rule(self):
        positions = [(x, y) for y in range(7) for x in range(7)]
        for radius in (0, 1, 2, 3):
            # Distinct meaningless IDs let this mechanism test identify deliveries
            # without giving sender identity to real Brain-facing observations.
            comm = Communication(len(positions), radius)
            for sender, (x, y) in enumerate(positions):
                comm.emit(sender, sender, x, y)
            comm.advance()
            for receiver, (x, y) in enumerate(positions):
                expected = {sender for sender, (sx, sy) in enumerate(positions)
                            if sender != receiver and max(abs(sx - x), abs(sy - y)) <= radius}
                self.assertEqual({s.signal_id for s in comm.hear(receiver, x, y)}, expected)

    def test_vocabulary_sizes_and_invalid_signals(self):
        for size in (4, 16, 40, 256, 4000):
            comm = Communication(size, 0)
            comm.emit(size - 1, 1, 0, 0)
            comm.advance()
            self.assertEqual(comm.hear(2, 0, 0)[0].signal_id, size - 1)
            for invalid in (-1, size, 1.5, True):
                with self.assertRaises(ValueError):
                    comm.emit(invalid, 1, 0, 0)

    def test_agents_hear_signals_only_on_next_tick(self):
        sim = make_sim(initial_population=2, signal_range=0)
        sender, receiver = sim.agents
        sender.x = sender.y = receiver.x = receiver.y = 0
        sender.brain = FixedBrain(signal=3)
        receiver.brain = FixedBrain()
        sim.step()
        sim.step()
        sim.step()
        observations = receiver.brain.observations
        self.assertEqual(observations[0].signals, ())
        self.assertEqual(observations[1].signals[0].signal_id, 3)
        self.assertEqual(observations[2].signals, ())
        self.assertEqual(sim.statistics.signals_emitted, 1)

    def test_signal_origin_is_position_after_movement(self):
        sim = make_sim(initial_population=2, signal_range=0)
        sender, receiver = sim.agents
        sender.x, sender.y = 0, 0
        receiver.x, receiver.y = 1, 0
        sender.brain = FixedBrain(Action.MOVE_EAST, signal=2)
        sim.step()
        sim.step()
        self.assertEqual(receiver.brain.observations[1].signals[0].source_direction, SourceDirection.SAME_CELL)


class SimulationTests(unittest.TestCase):
    def test_identical_seeds_replay_full_state_and_events(self):
        config = Config(world_width=8, world_height=6, initial_population=12, initial_food=80, initial_water=80)
        first_events, second_events = [], []
        first = Simulation(config, event_sink=first_events.append)
        second = Simulation(config, event_sink=second_events.append)
        self.assertEqual(body_state(first), body_state(second))
        for _ in range(160):
            self.assertEqual(first.step(), second.step())
            self.assertEqual(body_state(first), body_state(second))
            self.assertEqual(resource_state(first.world), resource_state(second.world))
        self.assertEqual(first_events, second_events)
        self.assertEqual(first.population + first.statistics.deaths, config.initial_population)
        self.assertEqual(sum(t.food for row in first.world.tiles for t in row) + first.statistics.food_consumed, config.initial_food)
        self.assertEqual(sum(t.water for row in first.world.tiles for t in row) + first.statistics.water_consumed, config.initial_water)

    def test_global_random_state_is_untouched_and_logging_does_not_change_run(self):
        before = random.getstate()
        events = []
        first = Simulation.create_default(seed=0)
        second = Simulation.create_default(seed=0, event_sink=events.append)
        for _ in range(20):
            first.step()
            second.step()
        self.assertEqual(random.getstate(), before)
        self.assertEqual(first.config.random_seed, 0)
        self.assertEqual(first.statistics, second.statistics)
        self.assertEqual(body_state(first), body_state(second))
        self.assertEqual(resource_state(first.world), resource_state(second.world))
        self.assertEqual(len({h.id for h in first.agents}), first.population)

    def test_csv_and_optional_jsonl_events(self):
        csv_stream, event_stream = io.StringIO(), io.StringIO()
        config = Config(initial_population=2, max_hunger=1, max_thirst=1, signal_probability=1)
        sim = Simulation(config, event_sink=JSONLEventWriter(event_stream))
        writer = CSVStatisticsWriter(csv_stream)
        writer.write(sim.statistics)
        writer.write(sim.step())
        rows = list(csv.DictReader(io.StringIO(csv_stream.getvalue())))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1], {name: "" if value is None else str(value)
                                  for name, value in asdict(sim.statistics).items()})
        events = [json.loads(line) for line in event_stream.getvalue().splitlines()]
        self.assertEqual(events[0]["details"]["config"]["random_seed"], config.random_seed)
        self.assertEqual(len(events), 3)
        detail = events[1]["details"]
        self.assertIn("tiles", detail["observation"])
        self.assertIn("signals", detail["observation"])
        self.assertIn("action", detail)
        self.assertIsInstance(detail["signal_emitted"], int)
        self.assertEqual(detail["death_cause"], "starvation")

    def test_empty_simulation_is_noop(self):
        sim = make_sim(initial_population=0)
        self.assertEqual(sim.step().tick, 0)
        self.assertEqual(sim.statistics.deaths, 0)

    def test_invalid_configuration_is_rejected(self):
        invalid_settings = (
            {"world_width": 0}, {"world_height": -1}, {"initial_population": -1},
            {"initial_food": -1}, {"initial_water": 1.5}, {"perception_radius": -1},
            {"max_hunger": 0}, {"max_thirst": 0}, {"max_age": 0},
            {"hunger_per_tick": float("nan")}, {"thirst_per_tick": -1},
            {"signal_vocab_size": 0}, {"signal_range": -1}, {"signal_probability": 1.1},
            {"random_seed": None}, {"max_ticks": -1}, {"status_interval": 0},
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                replace(Config(), **settings)


if __name__ == "__main__":
    unittest.main()
