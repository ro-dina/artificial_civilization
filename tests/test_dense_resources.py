"""v082 passive snapshots, exact labels, fixed regression and stage gates."""
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from random import Random
import gzip
import json
import random
import tempfile
import unittest

import numpy as np
import pandas as pd

from config import Config
from simulation.simulation import Simulation
from simulation.dense_resources import DenseResourceWriter, DenseResourceObserver, read_dense, PHASE
from world.resource_ecology import PatchGeometry
from experiments.dense_resource_data import environment_labels, DESIGN, plan, exact_counts
from experiments.dense_resource_models import Encoding, fit_logistic, objective_gradient_hessian, hierarchy_from_patterns, feature_counts, feature_support
from experiments.dense_resource_study import settings, allowed, preregister
from experiments.resource_ecology_experiments import ecology_config, fingerprint
from experiments.future_resource_targets import ADMIN_END, NOT_AT_RISK, future_events, behavioral, Target, DEATH
from experiments.future_resource_io import digest, save_json
from experiments.nonstationary_prediction import Hierarchy
from experiments.future_resource_prediction import keys, score


X = json.dumps([0, 1, 2, 0, 0, 1, -1, ['SILENCE']], separators=(',', ':'))


def examples():
    return pd.DataFrame({'x_key': [X] * 4, 'action': [0, 1, 0, 1], 'b_token': [1, 2, 1, 2], 'rd_token': [0, 1, 0, 1],
        'sequence': [0, 1, 2, 3], 'agent_id': [1] * 4, 'own_index': [0, 1, 2, 3], 'tick': [1, 2, 3, 4],
        'x': [0] * 4, 'y': [0] * 4, 'post_x': [1] * 4, 'water_here': [1, 0, 2, 0],
        'food_consumed': [0] * 4, 'water_consumed': [0] * 4, 'acquired': [0] * 4, 'terminal': [0, 0, 0, 1]})


class DenseTests(unittest.TestCase):
    def test_binary_roundtrip_tick_zero_shape_and_accounting(self):
        cfg = Config.evolution(brain='learning', initial_population=2, random_seed=4)
        sim = Simulation(cfg)
        before = fingerprint(sim)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.gz'
            with DenseResourceWriter(path, cfg) as writer:
                writer.snapshot(sim)
                for _ in range(3): sim.step(); writer.snapshot(sim)
            frames = list(read_dense(path)); meta = json.loads(Path(str(path) + '.json').read_text())
            self.assertEqual([t for t, f, w in frames], list(range(4)))
            self.assertEqual(len(frames[0][1]), 1200)
            self.assertEqual(sum(frames[0][1]), 600)
            self.assertEqual(frames[-1][1], bytes(t.food for row in sim.world.tiles for t in row))
            self.assertEqual(meta['shape'], [2, 30, 40]); self.assertEqual(meta['phase'], PHASE)
            self.assertEqual(meta['records'], 4); self.assertTrue(meta['all_mass_balance_checks_passed'])

    def test_observer_on_off_full_state_all_modes_and_ecologies(self):
        cfg = Config.evolution(brain='learning', initial_population=4, random_seed=8)
        for ecology in ('E0', 'E3'):
            for mode in 'ABC':
                eco = ecology_config(ecology)
                with self.subTest(ecology=ecology, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    rows = []
                    with DenseResourceWriter(Path(tmp) / 'data.gz', cfg) as writer:
                        observer = DenseResourceObserver(lambda k, r: rows.append((k, r)), mode, ecology, PatchGeometry.generate(cfg, eco), writer)
                        baseline = Simulation(cfg, **settings(mode), resource_ecology=eco)
                        observed = Simulation(cfg, **settings(mode), resource_ecology=eco, resource_observer=observer, social_learning_observer=observer)
                        self.assertEqual(fingerprint(baseline), fingerprint(observed))
                        for _ in range(8):
                            baseline.step(); observed.step(); self.assertEqual(fingerprint(baseline), fingerprint(observed))
                        observer.finish(observed)
                        self.assertEqual(fingerprint(baseline), fingerprint(observed))
                    self.assertEqual(len(list(read_dense(Path(tmp) / 'data.gz'))), 9)

    def test_disabled_observer_creates_no_new_stream_or_rng(self):
        cfg = Config.evolution(initial_population=2)
        a, b = Simulation(cfg), Simulation.create_default(config=cfg)
        for _ in range(4): a.step(); b.step()
        self.assertEqual(fingerprint(a), fingerprint(b))

    def test_snapshot_requires_statistics_finalized(self):
        cfg = Config.evolution(initial_population=0)
        sim = Simulation(cfg)
        with tempfile.TemporaryDirectory() as tmp, DenseResourceWriter(Path(tmp) / 'd.gz', cfg) as writer:
            original = sim.statistics; sim.statistics = replace(original, tick=1)
            with self.assertRaises(ValueError): writer.snapshot(sim)
            sim.statistics = original; writer.snapshot(sim)

    def test_quantity_capacity_and_mass_balance_fail_closed(self):
        cfg = Config.evolution(initial_population=0); sim = Simulation(cfg)
        with tempfile.TemporaryDirectory() as tmp, DenseResourceWriter(Path(tmp) / 'd.gz', cfg) as writer:
            tile = sim.world.tile_at(0, 0); old = tile.food
            for invalid in (-1, 6, 1.2):
                tile.food = invalid
                with self.assertRaises(ValueError): writer.snapshot(sim)
            tile.food = old
            writer.snapshot(sim)
            sim.step()  # extinct step is a no-op, so directly test changed stock
            sim.world.tile_at(0, 0).water += 1
            with self.assertRaises(ValueError): writer.snapshot(sim)

    def test_mass_balance_checked_independently_for_water(self):
        cfg = Config.evolution(initial_population=0); sim = Simulation(cfg)
        tile = next(t for row in sim.world.tiles for t in row if t.water < 5)
        with tempfile.TemporaryDirectory() as tmp, DenseResourceWriter(Path(tmp) / 'd.gz', cfg) as writer:
            tile.water += 1
            with self.assertRaisesRegex(ValueError, 'mass balance failed: water'): writer.snapshot(sim)
            tile.water -= 1; writer.snapshot(sim)

    def test_uint8_capacity_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            for capacity in (None, 256):
                cfg = Config(food_capacity=capacity)
                with self.assertRaises(ValueError): DenseResourceWriter(Path(tmp) / 'd.gz', cfg)

    def test_checksums_and_compression_corruption(self):
        cfg = Config.evolution(initial_population=0); sim = Simulation(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'd.gz'
            with DenseResourceWriter(path, cfg) as writer: writer.snapshot(sim)
            data = path.read_bytes(); self.assertEqual(gzip.decompress(data)[:8], b'ACRD082\n')
            path.write_bytes(data[:-4])
            with self.assertRaises(ValueError): list(read_dense(path))

    def test_phase_rejected_and_output_exclusive(self):
        cfg = Config.evolution(initial_population=0); sim = Simulation(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'd.gz'
            with DenseResourceWriter(path, cfg) as writer: writer.snapshot(sim)
            with self.assertRaises(ValueError): DenseResourceWriter(path, cfg)
            meta = Path(str(path) + '.json'); value = json.loads(meta.read_text()); value['phase'] = 'tick_start'; meta.write_text(json.dumps(value))
            with self.assertRaises(ValueError): list(read_dense(path))

    def test_binary_is_deterministic_and_rng_pure(self):
        cfg = Config.evolution(initial_population=0); sim = Simulation(cfg); rng = random.getstate()
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('a.gz', 'b.gz'):
                with DenseResourceWriter(Path(tmp) / name, cfg) as writer: writer.snapshot(sim)
            self.assertEqual((Path(tmp) / 'a.gz').read_bytes(), (Path(tmp) / 'b.gz').read_bytes())
        self.assertEqual(rng, random.getstate())

    def test_exact_future_join_frozen_reference_and_death_independence(self):
        f = examples(); dense = np.zeros((10, 2, 1, 2), dtype=np.uint8); dense[6, 1, 0, 0] = 1; dense[6, 1, 0, 1] = 0
        labels, quantity = environment_labels(f, dense)
        self.assertEqual(labels['T1'][0].tolist(), [1, 0, 0, 0]); self.assertEqual(quantity[0], 1)
        f['post_x'] = 0; f['terminal'] = 1
        self.assertEqual(labels['T1'][0].tolist(), environment_labels(f, dense)[0]['T1'][0].tolist())

    def test_depletion_only_when_present_and_admin_boundary(self):
        f = examples(); dense = np.zeros((8, 2, 1, 2), dtype=np.uint8)
        labels, _ = environment_labels(f, dense)
        self.assertEqual(labels['T1'][0].tolist(), [0, 0, -1, -1])
        self.assertEqual(labels['T2'][0].tolist(), [1, -1, -1, -1])
        self.assertEqual(labels['T2'][1][1], NOT_AT_RISK); self.assertEqual(labels['T1'][1][3], ADMIN_END)

    def test_invalid_origin_dtype_coordinate_and_duplicate(self):
        f = examples(); dense = np.zeros((10, 2, 1, 2), dtype=np.uint8)
        for changed in (f.assign(x=3), f.assign(x=.5), f.assign(sequence=0)):
            with self.assertRaises(ValueError): environment_labels(changed, dense)
        with self.assertRaises(ValueError): environment_labels(f, dense.astype(float))

    def test_behavior_reference_death_competes(self):
        f = examples()
        labels, reasons = behavioral(f, future_events(f, 4), Target('B10', 'T5', 'behavioral', 'either', 10))
        self.assertTrue((labels == -1).all()); self.assertTrue((reasons == DEATH).all())

    def test_fixed_nested_encoding_dimensions_and_no_privilege(self):
        f = examples()
        for level, dimension in enumerate((69, 80, 86)):
            encoder = Encoding(level); data = encoder.transform(f)
            self.assertEqual(encoder.dimension, dimension)
            self.assertTrue((data[:, 0] == 0).all())
            self.assertEqual(data.shape, (4, 13 + level))
            changed = f.assign(x=999, post_x=-999, water_here=55, terminal=1)
            np.testing.assert_array_equal(data, encoder.transform(changed))

    def test_all_auditory_categories_are_origin_features(self):
        f = examples()
        for auditory in (['SILENCE'], ['MASKED', 3], ['IDENTIFIED', 15, 'NW', 2]):
            f.x_key = json.dumps([0, 1, 2, 0, 0, 1, -1, auditory])
            self.assertEqual(Encoding(2).transform(f).shape, (4, 15))
        f.x_key = json.dumps([0, 1, 2, 0, 0, 1, -1, ['IDENTIFIED', 16, 'NW', 2]])
        with self.assertRaises(ValueError): Encoding(2).transform(f)

    def test_hierarchical_sufficient_counts_equal_legacy(self):
        patterns = examples()[['x_key', 'action', 'b_token', 'rd_token']].iloc[:2].copy()
        patterns['count'] = [30, 25]; patterns['positive'] = [12, 13]
        a = hierarchy_from_patterns(patterns); b = Hierarchy(('X_A', 'X_B_A', 'X_C_A'), 2)
        for row in patterns.itertuples(index=False):
            key = keys(row.x_key, row.action, row.b_token, row.rd_token)
            for target, n in [(0, row.count - row.positive), (1, row.positive)]:
                for _ in range(n): b.add(key, row.action, target)
        b.freeze(); self.assertEqual(a.cells, b.cells); self.assertEqual(a.actions, b.actions)

    def test_logistic_gradient_and_hessian(self):
        indices = Encoding(0).transform(examples()); weights = np.array([10., 20., 15., 25.]); positive = np.array([2., 15., 3., 20.]); coefficient = np.linspace(-.1, .1, 69)
        objective, gradient, hessian = objective_gradient_hessian(coefficient, indices, positive, weights)
        for index in (0, 2, 65):
            delta = np.zeros(69); delta[index] = 1e-5
            plus = objective_gradient_hessian(coefficient + delta, indices, positive, weights)
            minus = objective_gradient_hessian(coefficient - delta, indices, positive, weights)
            self.assertAlmostEqual(gradient[index], (plus[0] - minus[0]) / 2e-5, places=7)
            np.testing.assert_allclose(hessian[:, index], (plus[1] - minus[1]) / 2e-5, atol=1e-7)

    def test_regularized_fit_convergence_determinism_and_rng(self):
        matrix = Encoding(2).transform(examples()); before = random.getstate(); numpy_before = np.random.get_state()
        a = fit_logistic(matrix, [2, 18, 3, 20], [10, 25, 15, 30], 86)
        b = fit_logistic(matrix, [2, 18, 3, 20], [10, 25, 15, 30], 86)
        self.assertTrue(a['converged']); self.assertLessEqual(a['gradient_infinity'], 1e-8); self.assertEqual(a, b)
        self.assertEqual(before, random.getstate()); np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])

    def test_nonconvergence_and_single_class_explicit(self):
        matrix = Encoding(0).transform(examples())
        model = fit_logistic(matrix, [1, 9, 1, 9], [10] * 4, 69, iterations=0)
        self.assertFalse(model['converged'])
        with self.assertRaises(ValueError): fit_logistic(matrix, [0] * 4, [10] * 4, 69)

    def test_feature_support_distinct_from_exact_cells(self):
        matrix = Encoding(2).transform(examples()); counts = feature_counts(matrix, 86, [20] * 4)
        self.assertTrue(feature_support(matrix, counts).all())
        new = matrix.copy(); new[0, -1] = 85
        self.assertFalse(feature_support(new, counts)[0])

    def test_same_decision_scoring_and_seed_split(self):
        f = examples(); values = np.array([0, 1, 0, 1]); supported = np.ones(4, bool)
        a, _ = score(np.array([.2, .8, .3, .7]), values, f.agent_id, supported)
        b, _ = score(np.array([.3, .7, .4, .6]), values, f.agent_id, supported)
        self.assertEqual(a['decisions'], b['decisions'])
        rules = json.loads(DESIGN.read_text()); self.assertFalse(set(rules['train_seeds']) & set(rules['evaluation_seeds']))

    def test_frozen_source_and_gate_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); preregister(root); self.assertEqual(plan(root)['horizon'], 500)
            save_json(root / 'validation.json', {'passed': False, 'inputs': {}, 'preregistration_sha256': digest(root / 'preregistration.json')})
            self.assertFalse(allowed(root, 'pilot'))
            source = root / 'gate.txt'; source.write_text('original')
            save_json(root / 'pilot_review.json', {'allowed': True, 'inputs': {str(source): digest(source)}, 'preregistration_sha256': digest(root / 'preregistration.json')})
            self.assertTrue(allowed(root, 'training')); source.write_text('changed')
            with self.assertRaises(ValueError): allowed(root, 'training')


if __name__ == '__main__': unittest.main()
