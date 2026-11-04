"""Exact passive archive labels, support gates and frozen retrospective models."""
from pathlib import Path
from hashlib import sha256
import json
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.future_resource_targets import (Target, definitions, snapshots, environmental, future_events, behavioral,
    competing_curve, OBSERVED, NO_SNAPSHOT, ADMIN_END, HISTORY_GAP, DEATH, NO_REFERENCE)
from experiments.future_resource_io import (DESIGN, decisions, digest, feature_counts, lookup_counts,
    missing_fields, receipt, save_csv, save_json, verify_hashes)
from experiments.future_resource_analysis import support, choose_target
from experiments.future_resource_prediction import Predictor, keys, score, split_seeds, summarize
from experiments.future_resource_labels import next_sequences


def frame():
    return pd.DataFrame({'sequence': range(4), 'tick': [1, 2, 3, 4], 'agent_id': [7] * 4,
        'own_index': range(4), 'x': [0, 1, 1, 0], 'y': [0] * 4, 'post_x': [1, 1, 0, 0], 'post_y': [0] * 4,
        'food_here': [2, 0, 0, 1], 'water_here': [0] * 4, 'food_consumed': [1, 0, 0, 1],
        'water_consumed': [0] * 4, 'acquired': [1, 0, 0, 1], 'terminal': [0] * 4,
        'nearest_visible_dx': [1., np.nan, 0., 0.], 'nearest_visible_dy': [0., np.nan, 0., 0.],
        'x_key': ['x'] * 4, 'action': [4, 6, 6, 4], 'b_token': [1, 2, 1, 2], 'rd_token': [0, 1, 0, 1]})


class FutureResourceTests(unittest.TestCase):
    def test_candidate_names_unique_and_fixed_alias(self):
        names = [t.name for t in definitions()]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 37)
        self.assertNotIn('T2_food_r0_h5', names)

    def test_exact_endpoint_alignment_no_interpolation(self):
        f = frame(); t = Target('test', 'T1', 'environmental', 'food', 1)
        labels, reasons = environmental(f, {2: {'food': np.array([[0, 5]])}}, t, 4)
        self.assertEqual(labels.tolist(), [0, -1, -1, -1])
        self.assertEqual(reasons.tolist(), [OBSERVED, NO_SNAPSHOT, NO_SNAPSHOT, ADMIN_END])

    def test_frozen_reference_not_future_agent_position(self):
        f = frame(); t = Target('test', 'T1', 'environmental', 'food', 1)
        grids = {2: {'food': np.array([[0, 5]])}}
        first = environmental(f, grids, t, 4)[0]
        f.loc[:, 'post_x'] = 1; f.loc[1:, 'x'] = 0
        self.assertEqual(first[0], environmental(f, grids, t, 4)[0][0])
        self.assertEqual(first[0], 0)

    def test_neighbor_reference_uses_current_visible_location(self):
        t = Target('test', 'T1', 'environmental', 'food', 1, location='neighbor')
        labels, reasons = environmental(frame(), {2: {'food': np.array([[0, 5]])}}, t, 4)
        self.assertEqual(labels[0], 1)
        self.assertEqual(reasons[1], NO_REFERENCE)

    def test_local_radius_does_not_wrap_world(self):
        t = Target('test', 'T2', 'environmental', 'food', 1, 1)
        f = frame(); f.x = 0
        labels, _ = environmental(f, {2: {'food': np.array([[0, 0, 1]])}}, t, 4)
        self.assertEqual(labels[0], 0)

    def test_quantity_change_not_inferred_from_presence(self):
        f = frame(); grids = {2: {'food': np.array([[1, 0]])}}
        dec = Target('d', 'T3', 'environmental', 'food', 1, transition='decrease')
        dep = Target('p', 'T3', 'environmental', 'food', 1, transition='depletion')
        self.assertEqual(environmental(f, grids, dec, 4)[0][0], 1)
        self.assertEqual(environmental(f, grids, dep, 4)[0][0], 0)

    def test_recovery_conditional_on_absence(self):
        t = Target('r', 'T3', 'environmental', 'water', 1, transition='recovery')
        self.assertEqual(environmental(frame(), {2: {'water': np.array([[2, 0]])}}, t, 4)[0][0], 1)

    def test_snapshot_complete_unique_grid_required(self):
        incomplete = pd.DataFrame({'tick': [1], 'x': [0], 'y': [0], 'food': [1], 'water': [0]})
        with self.assertRaises(ValueError):
            snapshots(incomplete, 2, 1)
        complete = pd.concat([incomplete, incomplete.assign(x=1)], ignore_index=True)
        self.assertEqual(snapshots(complete, 2, 1)[1]['food'].tolist(), [[1, 1]])

    def test_next_actual_event_excludes_current(self):
        elapsed, event = future_events(frame(), 4)['food']
        self.assertEqual(elapsed.tolist(), [3, 2, 1, 0])
        self.assertEqual(event.tolist(), [1, 1, 1, 3])

    def test_death_competes_and_acquisition_wins_future_tie(self):
        f = frame(); f.loc[3, 'terminal'] = 1
        elapsed, event = future_events(f, 4)['food']
        self.assertEqual(event.tolist(), [1, 1, 1, 2])
        self.assertEqual(elapsed[-1], 0)
        self.assertEqual(future_events(f, 4)['water'][1].tolist(), [2] * 4)

    def test_death_is_not_binary_negative_or_ordinary_censor(self):
        f = frame(); f.loc[3, 'terminal'] = 1
        target = Target('k', 'T5', 'behavioral', 'water', 5)
        labels, reasons = behavioral(f, future_events(f, 4), target)
        self.assertTrue((labels == -1).all())
        self.assertTrue((reasons == DEATH).all())

    def test_acquisition_before_admin_end_is_observed(self):
        f = frame(); target = Target('k', 'T5', 'behavioral', 'food', 10)
        labels, reasons = behavioral(f, future_events(f, 4), target)
        self.assertEqual(labels.tolist(), [1, 1, 1, -1])
        self.assertEqual(reasons[-1], ADMIN_END)

    def test_negative_requires_complete_horizon(self):
        f = frame(); target = Target('k', 'T5', 'behavioral', 'water', 2)
        labels, reasons = behavioral(f, future_events(f, 4), target)
        self.assertEqual(labels.tolist(), [0, 0, -1, -1])

    def test_missing_future_history_never_bridged(self):
        f = frame().drop(index=2).reset_index(drop=True)
        elapsed, event = future_events(f, 4)['food']
        self.assertEqual(event.tolist(), [4, 4, 3])
        labels, reasons = behavioral(f, future_events(f, 4), Target('k', 'T5', 'behavioral', 'food', 5))
        self.assertEqual(labels.tolist(), [-1, -1, -1])
        self.assertEqual(reasons[0], HISTORY_GAP)

    def test_competing_risk_curve_accounts_for_death(self):
        rows = competing_curve(np.array([1, 1, 2, 2]), np.array([1, 2, 3, 1]), 2)
        self.assertAlmostEqual(rows[-1]['acquisition_cif'], .5)
        self.assertAlmostEqual(rows[-1]['death_cif'], .25)
        self.assertAlmostEqual(rows[-1]['event_free'], .25)

    def test_target_prevalence_counts_and_conditional_support(self):
        f = frame(); labels = np.array([0, 1, 0, 1]); reasons = np.zeros(4, dtype=np.uint8)
        thresholds = json.loads(DESIGN.read_text())['thresholds']
        result, cells, histogram = support(f, labels, reasons, feature_counts(f, labels), thresholds)
        self.assertEqual(result['prevalence'], .5)
        self.assertEqual(result['positive'], 2)
        self.assertEqual(result['negative'], 2)
        self.assertEqual(sum(histogram.strata), 2)
        self.assertEqual(cells[0]['test_decisions'], 4)
        self.assertFalse(result['eligible'])

    def test_degenerate_and_unavailable_targets(self):
        f = frame(); thresholds = json.loads(DESIGN.read_text())['thresholds']
        for value, classification in [(1, 'degenerate'), (-1, 'unavailable')]:
            labels = np.full(4, value); result, _, _ = support(f, labels, np.zeros(4, dtype=np.uint8), feature_counts(f, labels), thresholds)
            self.assertEqual(result['classification'], classification)

    def test_training_cell_support_is_seed_comparable_not_local_catalog_id(self):
        f = frame(); labels = np.array([0, 1, 0, 1])
        counts = feature_counts(f, labels)
        lookup = lookup_counts(f, counts)
        self.assertEqual(lookup['M0'].tolist(), [2, 2, 2, 2])

    def test_predictor_backoff_and_incremental_memory_parent(self):
        p = Predictor()
        f = pd.concat([frame().iloc[[0]]] * 20, ignore_index=True)
        p.add(f, np.ones(20, dtype=int)); p.freeze()
        test = frame().iloc[[0]].copy(); test.rd_token = 5
        probabilities, supported = p.predict(test)
        self.assertTrue(supported['M1'][0])
        self.assertFalse(supported['M2'][0])
        self.assertEqual(probabilities['M1'][0], probabilities['M2'][0])
        with self.assertRaises(RuntimeError):
            p.add(f, np.ones(20, dtype=int))

    def test_seed_separation_rejects_overlap(self):
        split_seeds([3, 4], [0, 1, 2])
        with self.assertRaises(ValueError):
            split_seeds([3, 4], [4, 5])

    def test_features_contain_no_future_or_privileged_fields(self):
        self.assertEqual(keys('X', 4, 2, 1), {'X_A': ('X', 4), 'X_B_A': ('X', 4, 2), 'X_C_A': ('X', 4, 2, 1)})

    def test_binary_brier_calibration_equal_agent_scores(self):
        values, bins = score(np.array([.25, .75]), np.array([0, 1]), np.array([1, 2]), np.array([True, False]))
        self.assertAlmostEqual(values['brier'], .0625)
        self.assertEqual(values['decisions'], 2)
        self.assertEqual(sum(b['n'] for b in bins), 2)
        self.assertEqual(values['log_loss'], values['equal_agent_log_loss'])

    def test_selection_uses_only_frozen_eligibility_not_predictive_gain(self):
        f = pd.DataFrame({'target': ['x'] * 9 + ['y'] * 9, 'ecology': ['E0'] * 18, 'eligible': [False] * 9 + [True] * 9})
        self.assertEqual(choose_target(f, ['x', 'y']), 'y')
        self.assertIsNone(choose_target(f, ['x']))

    def test_missing_field_classification(self):
        self.assertEqual(missing_fields(['tick', 'food'], ['tick', 'water']), ['water'])

    def test_analysis_consumes_no_rng_or_simulation(self):
        before = random.getstate(); numpy_before = np.random.get_state()
        with patch('simulation.simulation.Simulation.__init__', side_effect=AssertionError('No new worlds')):
            future_events(frame(), 4)
            p = Predictor(); p.add(frame(), np.array([0, 1, 0, 1])); p.freeze(); p.predict(frame())
        self.assertEqual(before, random.getstate())
        now = np.random.get_state()
        self.assertEqual(numpy_before[0], now[0]); self.assertTrue(np.array_equal(numpy_before[1], now[1])); self.assertEqual(numpy_before[2:], now[2:])

    def test_csv_output_deterministic_and_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'a.csv.gz', Path(tmp) / 'b.csv.gz'
            save_csv(first, frame()); save_csv(second, frame())
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with self.assertRaises(FileExistsError):
                save_csv(first, frame())

    def test_frozen_plan_detects_changed_sources_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'input.txt'; source.write_text('original')
            save_json(root / 'design_receipt.json', {'design': json.loads(DESIGN.read_text()), 'design_sha256': digest(DESIGN)})
            plan = {'inputs': {str(source): digest(source)}, 'source_sha256': {}}
            save_json(root / 'frozen.json', plan)
            self.assertEqual(receipt(root / 'frozen.json', root), plan)
            source.write_text('changed')
            with self.assertRaises(ValueError):
                receipt(root / 'frozen.json', root)

    def test_archive_reads_do_not_write_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'input.csv'; save_csv(p, frame()); original = digest(p)
            f = pd.read_csv(p); future_events(f, 4)
            self.assertEqual(original, digest(p))

    def test_equal_seed_summary_no_decision_pooling(self):
        f = pd.DataFrame({'mode': ['A', 'A'], 'seed': [0, 1], 'loss': [1., 3.]})
        row = summarize(f, ['mode'], ['loss']).iloc[0]
        self.assertEqual(row['mean'], 2.)
        self.assertAlmostEqual(row['std'], np.sqrt(2))
        self.assertEqual(row['n'], 2)

    def test_paired_summary_keeps_outcome_metric_separate_from_measure(self):
        f = pd.DataFrame({'mode': ['A', 'A', 'A', 'A'], 'seed': [0, 1, 0, 1],
                          'metric': ['loss', 'loss', 'brier', 'brier'], 'raw_difference': [1., 3., 2., 4.]})
        s = summarize(f, ['mode', 'metric'], ['raw_difference'])
        self.assertEqual(s.set_index('metric').loc['loss', 'mean'], 2.)
        self.assertEqual(s.set_index('metric').loc['brier', 'mean'], 3.)
        self.assertTrue(s.measure.eq('raw_difference').all())

    def test_terminal_history_cannot_continue(self):
        f = frame(); f.loc[1, 'terminal'] = 1
        with self.assertRaises(ValueError):
            future_events(f, 4)

    def test_different_agent_histories_never_link(self):
        f = frame(); f.loc[2:, 'agent_id'] = 8; f.loc[2:, 'own_index'] = [0, 1]
        self.assertEqual(future_events(f, 4)['food'][1][:2].tolist(), [4, 4])

    def test_compact_future_event_links_actual_next_sequence(self):
        f = frame(); elapsed, event = future_events(f, 4)['food']
        self.assertEqual(next_sequences(f, elapsed, event, 'food').tolist(), [3, 3, 3, -1])
        bad = event.copy(); bad[-1] = 1
        with self.assertRaises(ValueError):
            next_sequences(f, elapsed, bad, 'food')


if __name__ == '__main__':
    unittest.main()
