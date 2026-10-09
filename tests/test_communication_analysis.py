"""Categorical toy controls, sparse streaming inputs and deterministic artifacts."""

from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from config import Config
from experiments.analyze_communication import Options, SCHEMAS, aggregate_time_series, analyze, analyze_window
from experiments.communication_data import discover, file_hash, read_windows, validate_frame
from experiments.communication_metrics import (analysis_rng, convergence, entropy, information_report,
                                               jensen_shannon, mutual_information, pair_sample)
from experiments.communication_pilot import run_pilot
from simulation.communication_observer import COLUMNS, CSVCommunicationWriter, CommunicationRecord

TOYS = json.loads((Path(__file__).parent / "fixtures/v061_communication_toys.json").read_text())


def record(tick=1, agent=0, signal=3, *, listening=False, production=False, brain="learning", **changes):
    r = CommunicationRecord(tick, agent, brain, production, listening, "learned" if production else "random",
        "SIGNAL" if signal is not None else "SILENCE", signal, "wait", tick % 2, 0, 2, 2, 2, 2, 2,
        "SILENCE", None, None, "SILENCE" if listening else None, None, None, None, 2, 0, tick-1, 1)
    return replace(r, **changes)


def fixture_file(directory, name="toy_communication.csv", *, seed=0, listening=False, production=False,
                 brain="learning", horizon=5, rows=None):
    cfg = Config(brain=brain, random_seed=seed, learning_uses_auditory=listening,
                 learning_controls_vocalization=production, initial_population=2, max_ticks=max(1, horizon))
    rows = rows if rows is not None else [record(t, a, signal=(3 if (t+a) % 2 else None),
        listening=listening, production=production, brain=brain,
        auditory_kind="IDENTIFIED" if t>1 and a else "SILENCE", heard_signal_id=12 if t>1 and a else None,
        heard_direction="E" if t>1 and a else None,
        physical_action="wait" if a else "eat") for t in range(1, horizon+1) for a in range(2)]
    path = Path(directory) / name
    with path.open("w", newline="") as stream:
        writer = CSVCommunicationWriter(stream)
        for row in rows:
            writer(row)
    meta = {"version": "0.6.1", "schema_version": 1, "columns": COLUMNS, "config": asdict(cfg), "seed": seed,
            "final_tick": horizon, "statistics": {"population": 0 if horizon == 0 else 2, "births": 0, "deaths": 0,
            "exploratory_actions": 0, "exploitative_actions": len(rows)}, "source_sha256": {}}
    path.with_suffix(".csv.metadata.json").write_text(json.dumps(meta, indent=2))
    return path


class MemoryOutput:
    def __init__(self):
        self.rows = defaultdict(list)

    def add(self, name, row):
        self.rows[name].append(row)


class InformationTests(unittest.TestCase):
    def test_entropy_known_values_and_empty(self):
        self.assertEqual(entropy([1, 1]), 1)
        self.assertEqual(entropy([1, 1, 1, 1]), 2)
        self.assertEqual(entropy([0, 5, 0]), 0)
        self.assertIsNone(entropy([]))
        self.assertIsNone(entropy([0, 0]))
        for counts in ([-1], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                entropy(counts)

    def test_independent_and_dependent_sender_toys(self):
        for name in ("A_independent", "B_dependent"):
            toy = TOYS[name]
            self.assertEqual(mutual_information(toy["v"], toy["x"]), toy["expected_mi_bits"])
        self.assertIsNone(mutual_information([], []))
        with self.assertRaises(ValueError):
            mutual_information([1], [])

    def test_receiver_independent_and_dependent_toys(self):
        for name in ("F_receiver_independent", "G_receiver_dependent"):
            toy = TOYS[name]
            self.assertEqual(mutual_information(toy["a"], toy["percept"]), toy["expected_mi_bits"])

    def test_signal_id_relabeling_changes_no_information_or_null_estimate(self):
        v, x = [3, 3, 12, 12]*20, [0, 1, 0, 1]*20
        relabeled = [37 if value == 3 else 0 for value in v]
        a = information_report(v, x, permutations=20, rng=analysis_rng(8))
        b = information_report(relabeled, x, permutations=20, rng=analysis_rng(8))
        self.assertEqual(a, b)  # IDs have categorical identity, no numeric proximity/meaning.

    def test_identity_confounding_pooled_vs_within_agent(self):
        toy = TOYS["C_identity"]
        pooled, _ = information_report(toy["v"], toy["x"], permutations=9, rng=analysis_rng(0))
        within, individuals = information_report(toy["v"], toy["x"], groups=toy["agent_id"],
                                                permutations=9, rng=analysis_rng(0))
        self.assertEqual(pooled["mi_bits"], 1)
        self.assertEqual(within["mi_bits"], 0)
        self.assertEqual(within["corrected_mi_bits"], 0)
        self.assertEqual(within["strata"], 2)
        self.assertTrue(all(row["mi_bits"] == 0 for row in individuals))
        self.assertIsNone(within["nmi"])

    def test_conditional_mi_weights_individual_exposure_within_a_run(self):
        report, individuals = information_report([0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
            groups=[1, 1, 2, 2, 2, 2, 2, 2], permutations=3, rng=analysis_rng(2))
        self.assertEqual(report["mi_bits"], 0.25)
        self.assertEqual([r["n"] for r in individuals], [2, 6])

    def test_permutation_is_deterministic_and_preserves_group_marginals(self):
        class CheckingRNG:
            def __init__(self):
                self.calls = []
                self.rng = analysis_rng(3)
            def permuted(self, values, axis):
                self.calls.append(values.copy())
                result = self.rng.permuted(values, axis=axis)
                np.testing.assert_array_equal(np.sort(values, axis=axis), np.sort(result, axis=axis))
                return result
        rng = CheckingRNG()
        values, states, ids = [3, 12, 3, 12], [0, 1, 1, 0], [0, 0, 1, 1]
        a, _ = information_report(values, states, groups=ids, permutations=7, rng=rng)
        b, _ = information_report(values, states, groups=ids, permutations=7, rng=analysis_rng(3))
        self.assertEqual(a, b)
        self.assertEqual(len(rng.calls), 7)
        self.assertTrue(all(call.tolist() == [[0, 1], [0, 1]] for call in rng.calls))

    def test_analysis_rng_does_not_touch_python_or_numpy_global_state(self):
        py_before, np_before = random.getstate(), np.random.get_state()
        information_report([0, 1]*20, [0, 0, 1, 1]*10, permutations=10, rng=analysis_rng(9, "toy"))
        self.assertEqual(random.getstate(), py_before)
        np_after = np.random.get_state()
        self.assertEqual(np_before[0], np_after[0])
        np.testing.assert_array_equal(np_before[1], np_after[1])
        self.assertEqual(np_before[2:], np_after[2:])

    def test_normalization_undefined_for_constant_categories_and_empty(self):
        report, _ = information_report([7]*3, [1, 2, 3], permutations=1, rng=analysis_rng(0))
        self.assertIsNone(report["nmi"])
        self.assertIsNone(report["permuted_std_bits"])
        empty, rows = information_report([], [], permutations=3, rng=analysis_rng(0))
        self.assertEqual(empty["n"], 0)
        self.assertIsNone(empty["mi_bits"])
        self.assertEqual(rows, [])

    def test_perfect_information_nmi_one_and_correction_can_be_negative(self):
        report, _ = information_report([3, 3, 12, 12]*20, [0, 0, 1, 1]*20,
                                      permutations=20, rng=analysis_rng(3))
        self.assertEqual(report["nmi"], 1)
        self.assertGreater(report["corrected_mi_bits"], 0.8)
        independent, _ = information_report([3, 3, 12, 12], [0, 1, 0, 1], permutations=20, rng=analysis_rng(0))
        self.assertLess(independent["corrected_mi_bits"], 0)

    def test_invalid_permutation_arguments(self):
        for count in (0, -1, True):
            with self.assertRaises(ValueError):
                information_report([1], [2], permutations=count, rng=analysis_rng(0))
        with self.assertRaises(ValueError):
            information_report([1, 2], [2, 3], groups=[1], permutations=1, rng=analysis_rng(0))

    def test_jsd_identical_disjoint_symmetric_bounded_and_empty(self):
        for name in ("D_identical", "E_disjoint"):
            toy = TOYS[name]
            self.assertEqual(jensen_shannon(toy["p"], toy["q"]), toy["expected_jsd_bits"])
        self.assertEqual(jensen_shannon([1, 4, 2], [3, 2, 4]), jensen_shannon([3, 2, 4], [1, 4, 2]))
        self.assertIsNone(jensen_shannon([0, 0], [1, 2]))
        with self.assertRaises(ValueError):
            jensen_shannon([1], [1, 2])
        for i in range(1, 20):
            self.assertTrue(0 <= jensen_shannon([i, 1, 0], [0, 3, i]) <= 1)

    def test_pair_sampling_exhaustive_and_large_is_unique_deterministic(self):
        pairs, n = pair_sample(4, 100, analysis_rng(0))
        self.assertEqual(n, 6)
        self.assertEqual(pairs.tolist(), [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        a, n = pair_sample(10000, 80, analysis_rng(7))
        b, _ = pair_sample(10000, 80, analysis_rng(7))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(map(tuple, a))), 80)
        self.assertTrue((a[:, 0] < a[:, 1]).all())
        self.assertTrue((a >= 0).all() and (a < 10000).all())

    def test_convergence_frequency_null_and_sparse_agent_threshold(self):
        result = convergence([3, 12]*20, [0]*20+[1]*20, vocabulary=16, min_samples=10,
                             max_pairs=50, permutations=5, rng=analysis_rng(0))
        self.assertEqual((result["eligible_agents"], result["pairs"], result["mean_jsd_bits"]), (2, 1, 0))
        self.assertIsNotNone(result["null_mean_jsd_bits"])
        sparse = convergence([3, 12], [0, 1], vocabulary=16, min_samples=10,
                             max_pairs=50, permutations=5, rng=analysis_rng(0))
        self.assertEqual(sparse["eligible_agents"], 0)
        self.assertIsNone(sparse["mean_jsd_bits"])


class AnalysisInputTests(unittest.TestCase):
    def test_pilot_rejects_long_runs_duplicate_seeds_and_existing_destinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ticks, seeds in ((5000, (0,)), (1, (0, 0)), (0, (0,))):
                with self.assertRaises(ValueError):
                    run_pilot(Path(tmp), ticks, seeds)
            (Path(tmp) / "pilot_manifest.json").touch()
            with self.assertRaisesRegex(ValueError, "already exist"):
                run_pilot(Path(tmp), 1, (0,))

    def test_tiny_pilot_runs_all_conditions_with_identical_ecology(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            report = run_pilot(Path(tmp), 2, (7,))
            self.assertEqual([r["condition"] for r in report["runs"]], list("ABCD"))
            ecologies = []
            for run in report["runs"]:
                cfg = run["config"].copy()
                cfg.pop("learning_uses_auditory")
                cfg.pop("learning_controls_vocalization")
                ecologies.append(cfg)
                self.assertEqual(run["final_tick"], 2)
                self.assertGreater(run["communication_bytes"], 0)
            self.assertTrue(all(c == ecologies[0] for c in ecologies))

    def test_module_and_direct_script_entry_points(self):
        for command in ([sys.executable, "-m", "experiments.analyze_communication", "--help"],
                        [sys.executable, "experiments/analyze_communication.py", "--help"]):
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertIn("--analysis-seed", result.stdout)

    def test_discovery_all_four_ablations_and_random_sorted_with_missing_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            for code, listen, produce in (("D", True, True), ("B", True, False), ("C", False, True), ("A", False, False)):
                fixture_file(tmp, f"{code}_communication.csv", seed=7, listening=listen, production=produce)
            fixture_file(tmp, "random_communication.csv", brain="random")
            runs = discover(tmp, "*_communication.csv")
            self.assertEqual([r["condition"] for r in runs], ["A", "B", "C", "D", "random"])
            self.assertEqual([r["seed"] for r in runs[:-1]], [7]*4)

    def test_missing_sidecar_duplicate_seed_and_bad_schema_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = fixture_file(tmp)
            fixture_file(tmp, "duplicate_communication.csv")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                discover(tmp, "*_communication.csv")
            p.with_suffix(".csv.metadata.json").unlink()
            with self.assertRaisesRegex(ValueError, "sidecar"):
                discover(tmp, p.name)
            with self.assertRaisesRegex(ValueError, "No communication"):
                discover(tmp, "absent*")

    def test_tick_windows_chunk_boundaries_partial_and_extinction_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_file(tmp, horizon=5)
            run = discover(tmp, "*.csv")[0]
            windows = list(read_windows(run, 2, chunksize=3))
            self.assertEqual([i for i, _ in windows], [0, 1, 2])
            self.assertEqual([len(f) for _, f in windows], [4, 4, 2])
            self.assertEqual(windows[-1][1].tick.unique().tolist(), [5])

    def test_empty_and_missing_windows_do_not_fill_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_file(tmp, horizon=5, rows=[record(1), record(5)])
            windows = list(read_windows(discover(tmp, "*.csv")[0], 2, chunksize=1))
            self.assertEqual([len(f) for _, f in windows], [1, 0, 1])
            empty = fixture_file(tmp, "empty_communication.csv", seed=1, horizon=0, rows=[])
            run = discover(tmp, empty.name)[0]
            self.assertEqual(list(read_windows(run, 100)), [])

    def test_horizons_100_1000_5000_need_no_fixed_length_assumptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            for horizon in (100, 1000, 5000):
                path = fixture_file(tmp, f"h{horizon}_communication.csv", horizon=horizon,
                                    rows=[record(1), record(horizon)])
                windows = list(read_windows(discover(tmp, path.name)[0], 100))
                self.assertEqual(len(windows), horizon // 100)
                self.assertEqual(sum(len(f) for _, f in windows), 2)
                self.assertEqual(int(windows[-1][1].tick.max()), horizon)

    def test_duplicate_across_chunks_and_unsorted_ticks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_file(tmp, rows=[record(), record()])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                list(read_windows(discover(tmp, "*.csv")[0], 2, chunksize=1))
            fixture_file(tmp, rows=[record(2), record(1)])
            with self.assertRaisesRegex(ValueError, "streaming order"):
                list(read_windows(discover(tmp, "*.csv")[0], 2))

    def test_malformed_payloads_and_encoded_tags_are_rejected(self):
        cfg = asdict(Config())
        for row in (record(signal=20), record(signal=None, vocal_action="SIGNAL"), record(auditory_kind="MASKED", heard_signal_id=3),
                    record(listening=True, encoded_auditory_kind="UNKNOWN"), record(hunger_bin=3),
                    record(food_dx=2, food_dy=0), record(listening=True, brain="random")):
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_frame(pd.DataFrame([asdict(row)]), cfg, "toy")

    def test_options_validation(self):
        for kwargs in ({"permutations": 0}, {"window_size": True}, {"min_agent_emissions": 0}):
            with self.assertRaises(ValueError):
                Options(**kwargs)
        self.assertEqual(Options(analysis_seed=-3).analysis_seed, -3)


class AnalysisCalculationTests(unittest.TestCase):
    def calculate(self, rows, options=None, end=4):
        output = MemoryOutput()
        context = {"run": "toy", "condition": "D", "seed": 0, "window_start": 0, "window_end": end,
            "actual_window_end": end, "complete_window": True, "partial_window": False,
            "observed_ticks": len({r.tick for r in rows})}
        frame = pd.DataFrame([asdict(r) for r in rows], columns=COLUMNS)
        analyze_window(frame, context, asdict(Config()), options or Options(permutations=3), output, {})
        return output

    def test_silence_masking_denominators_arbitrary_ids(self):
        out = self.calculate([record(1, signal=None), record(2, signal=12, auditory_kind="MASKED"),
                             record(3, signal=3, auditory_kind="IDENTIFIED", heard_signal_id=12, heard_direction="NE")])
        rate = out.rows["silence_over_time"][0]
        self.assertEqual((rate["n_decisions"], rate["emitted"], rate["silent"]), (3, 2, 1))
        mask = out.rows["masking_over_time"][0]
        self.assertAlmostEqual(mask["masked_rate"] + mask["identified_rate"] + mask["silence_percept_rate"], 1)
        self.assertEqual(mask["identified_per_emitted"], 0.5)
        freq = {r["vocal_action"]: r for r in out.rows["vocal_frequency_over_time"]}
        self.assertEqual(freq["SIGNAL_12"]["count"], 1)
        self.assertEqual(freq["SIGNAL_12"]["prob_given_emitted"], 0.5)
        self.assertIsNone(freq["SILENCE"]["prob_given_emitted"])

    def test_empty_window_rates_entropy_and_information_are_undefined(self):
        out = self.calculate([])
        self.assertIsNone(out.rows["silence_over_time"][0]["emission_rate"])
        self.assertTrue(all(r["mi_bits"] is None and r["n"] == 0 for r in out.rows["sender_information"]))
        self.assertTrue(all(r["probability"] is None for r in out.rows["vocal_frequency_over_time"]))

    def test_individual_thresholds_and_sparse_preferences_are_explicit(self):
        out = self.calculate([record(1, 0), record(2, 0), record(1, 1, signal=None)],
                             Options(permutations=3, min_agent_decisions=2, min_agent_emissions=2))
        within = [r for r in out.rows["sender_information"] if r["conditioning"] == "within_agent_eligible"]
        self.assertTrue(all(r["eligible_agents"] == 1 and r["n"] == 2 for r in within))
        individuals = out.rows["sender_information_by_agent"]
        self.assertEqual({r["agent_id"] for r in individuals}, {0})
        preferences = out.rows["vocal_preferences_by_agent"]
        self.assertFalse(preferences[-1]["eligible_decisions"])
        self.assertIsNone(preferences[-1]["most_frequent_signal"])

    def test_receiver_body_visual_conditioning_reports_exposure(self):
        out = self.calculate([record(t, hunger_bin=0, signal=3,
            auditory_kind="IDENTIFIED" if t%2 else "SILENCE", heard_signal_id=12 if t%2 else None,
            heard_direction="E" if t%2 else None, physical_action="eat" if t%2 else "wait") for t in range(1, 9)],
            Options(permutations=3, min_receiver_stratum=2))
        for row in out.rows["receiver_responsiveness"]:
            self.assertEqual(row["mi_bits"], 1)
            self.assertEqual(row["n"], 8)
            self.assertEqual(row["strata"], 1)

    def test_equal_seed_weighting_sample_sd_and_finite_n(self):
        tables = {name: pd.DataFrame(columns=SCHEMAS[name]) for name in
            ("silence_over_time", "masking_over_time", "population_convergence", "sender_information", "receiver_responsiveness")}
        tables["silence_over_time"] = pd.DataFrame([{"condition": "A", "seed": seed, "window_start": 0,
            "window_end": 100, "actual_window_end": 100, "emission_rate": value, "silence_rate": 1-value,
            "n_decisions": n} for seed, value, n in ((0, 0, 10000), (1, 1, 2))])
        summary = aggregate_time_series(tables)
        row = summary.loc[summary.metric.eq("emission_rate")].iloc[0]
        self.assertEqual(row["mean"], 0.5)
        self.assertAlmostEqual(row["std"], 2**-0.5)
        self.assertEqual(row["n"], 2)
        tables["silence_over_time"].loc[1, "emission_rate"] = np.nan
        row = aggregate_time_series(tables).query("metric == 'emission_rate'").iloc[0]
        self.assertEqual(row["n"], 1)
        self.assertTrue(pd.isna(row["std"]))


class AnalysisArtifactTests(unittest.TestCase):
    def test_all_csv_json_png_repeat_exactly_and_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "in"
            inputs.mkdir()
            for code, listening, production, seed in (("D", True, True, 1), ("A", False, False, 0),
                ("C", False, True, 0), ("B", True, False, 0), ("D", True, True, 0)):
                fixture_file(inputs, f"{code}_seed{seed}_communication.csv", seed=seed, listening=listening, production=production)
            before = {p.name: file_hash(p) for p in inputs.iterdir()}
            opt = Options(window_size=2, permutations=3, min_agent_decisions=2, min_agent_emissions=1, min_receiver_stratum=2)
            with redirect_stdout(io.StringIO()):
                a = analyze(inputs, root / "a", options=opt)
                b = analyze(inputs, root / "b", options=opt)
            self.assertEqual(a, b)
            self.assertEqual(len(a["csv_files"]), 15)
            self.assertEqual(len(a["png_files"]), 10)
            for name in [*a["csv_files"], *a["png_files"], "analysis_metadata.json"]:
                self.assertEqual((root / "a" / name).read_bytes(), (root / "b" / name).read_bytes(), name)
            self.assertEqual(before, {p.name: file_hash(p) for p in inputs.iterdir()})
            summary = pd.read_csv(root / "a/silence_over_time.csv")
            self.assertTrue(summary.loc[summary.window_start.eq(4), "partial_window"].all())
            self.assertFalse(summary.loc[summary.window_start.eq(4), "complete_window"].any())
            self.assertEqual(a["seeds_by_condition"]["D"], [0, 1])

    def test_tick_zero_empty_run_can_generate_all_plots_and_blank_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_file(root, horizon=0, rows=[])
            with redirect_stdout(io.StringIO()):
                metadata = analyze(root, root / "out", options=Options(permutations=1))
            self.assertEqual(len(metadata["png_files"]), 10)
            summary = pd.read_csv(root / "out/communication_run_summary.csv")
            self.assertEqual(summary.n_decisions.iloc[0], 0)
            self.assertTrue(pd.isna(summary.emission_rate.iloc[0]))

    def test_output_input_collision_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = fixture_file(tmp, "silence_over_time.csv")
            digest = file_hash(p)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                analyze(tmp, tmp, pattern=p.name)
            self.assertEqual(file_hash(p), digest)


if __name__ == "__main__":
    unittest.main()
