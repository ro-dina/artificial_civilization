from contextlib import redirect_stdout
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import warnings

HAS_ANALYSIS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("pandas", "matplotlib"))
if HAS_ANALYSIS_DEPS:
    import numpy as np
    import pandas as pd
    from experiments.analyze_fixed_perception import (
        FINAL_METRICS, analyze, build_tables, discover_runs, load_runs, parse_filename,
    )


@unittest.skipUnless(HAS_ANALYSIS_DEPS, "Optional analysis dependencies: pip install -r requirements-analysis.txt")
class FixedPerceptionAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_run(self, perception=2, seed=0, rows=None, columns=None, name=None):
        path = self.root / (name or f"learning_fixed_p{perception}_seed_{seed}.csv")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns or ["tick", *FINAL_METRICS])
            writer.writeheader()
            for row in rows if rows is not None else [{"tick": 0}]:
                values = {name: 0 for name in FINAL_METRICS}
                values.update(population=10, mean_memory_size=2, mean_learning_updates=3)
                values.update(row)
                writer.writerow({name: values.get(name) for name in writer.fieldnames})
        return path

    def metric(self, table, metric, perception=2, tick=None):
        selected = table[(table["metric"] == metric) & (table["perception"] == perception)]
        if tick is not None:
            selected = selected[selected["tick"] == tick]
        return selected.iloc[0]

    def test_filename_parsing_and_discovery_support_new_radii_and_missing_seeds(self):
        for name, expected in (("learning_fixed_p2_seed_7.csv", (2, 7)),
                               ("learning_fixed_p16_seed_103.csv", (16, 103)),
                               ("learning_fixed_p0_seed_-2.csv", (0, -2))):
            self.assertEqual(parse_filename(Path(name)), expected)
        self.write_run(16, 103)
        self.write_run(2, 7)
        self.write_run(2, 1)
        self.write_run(name="unrelated.csv")
        self.assertEqual([parse_filename(p) for p in discover_runs(self.root)], [(2, 1), (2, 7), (16, 103)])
        with self.assertRaises(ValueError):
            parse_filename(Path("learning_fixed_p2_seed_7.csv.backup"))

    def test_malformed_names_warn_and_empty_discovery_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "No matching runs"):
            discover_runs(self.root)
        self.write_run(name="learning_fixed_pwrong_seed_1.csv")
        expected = self.write_run()
        with self.assertWarnsRegex(UserWarning, "expected learning_fixed"):
            self.assertEqual(discover_runs(self.root), [expected])

    def test_final_row_is_maximum_tick_and_does_not_backfill_nan(self):
        path = self.write_run(rows=[{"tick": 9, "population": None, "mean_memory_size": None},
                                    {"tick": 0, "mean_memory_size": 7}, {"tick": 4, "mean_memory_size": 8}])
        runs, final, _ = build_tables(load_runs([path]))
        self.assertEqual(runs.loc[0, "final_tick"], 9)
        self.assertTrue(pd.isna(runs.loc[0, "final_population"]))
        self.assertTrue(pd.isna(runs.loc[0, "mean_memory_size"]))
        self.assertEqual(self.metric(final, "mean_memory_size")["n"], 0)

    def test_mean_and_sample_sd_weight_runs_equally(self):
        a = self.write_run(seed=0, rows=[{"tick": 0, "population": 10, "mean_memory_size": 2}])
        b = self.write_run(seed=9, rows=[{"tick": 0, "population": 100, "mean_memory_size": 8}])
        runs, final, series = build_tables(load_runs([a, b]))
        self.assertEqual(len(runs), 2)
        row = self.metric(final, "mean_memory_size")
        self.assertEqual((row["mean"], row["min"], row["max"], row["n"]), (5, 2, 8, 2))
        self.assertAlmostEqual(row["std"], math.sqrt(18))
        row = self.metric(series, "population", tick=0)
        self.assertEqual(row["mean"], 55)
        self.assertAlmostEqual(row["std"], math.sqrt(4050))

    def test_missing_ticks_use_only_observations_and_metric_specific_counts(self):
        a = self.write_run(seed=1, rows=[{"tick": 0, "population": 10},
                                       {"tick": 2, "population": 20, "mean_memory_size": None}])
        b = self.write_run(seed=8, rows=[{"tick": 1, "population": 30},
                                       {"tick": 2, "population": 40, "mean_memory_size": 6}])
        _, _, series = build_tables(load_runs([a, b]))
        self.assertEqual(self.metric(series, "population", tick=0)["n"], 1)
        self.assertTrue(pd.isna(self.metric(series, "population", tick=1)["std"]))
        self.assertEqual(self.metric(series, "population", tick=2)["mean"], 30)
        memory = self.metric(series, "mean_memory_size", tick=2)
        self.assertEqual((memory["mean"], memory["n"]), (6, 1))

    def test_file_order_does_not_change_tables(self):
        paths = [self.write_run(4, 9), self.write_run(0, 2), self.write_run(4, 1)]
        a, b = load_runs(paths), load_runs(list(reversed(paths)))
        pd.testing.assert_frame_equal(a, b)
        for first, second in zip(build_tables(a), build_tables(b)):
            pd.testing.assert_frame_equal(first, second)

    def test_duplicate_ticks_and_run_identities_are_rejected(self):
        a = self.write_run(rows=[{"tick": 0}, {"tick": 0}])
        with self.assertRaisesRegex(ValueError, "duplicate ticks"):
            load_runs([a])
        a = self.write_run()
        b = self.write_run(name="learning_fixed_p02_seed_00.csv")
        with self.assertRaisesRegex(ValueError, "Duplicate run identity"):
            load_runs([a, b])

    def test_missing_required_columns_and_invalid_ticks_are_rejected(self):
        path = self.write_run(columns=["tick", "population", "mean_memory_size", "total_learning_updates"])
        with self.assertRaisesRegex(ValueError, "missing required columns: mean_learning_updates"):
            load_runs([path])
        for tick in (None, -1, 1.5, "bad", "inf"):
            with self.subTest(tick=tick):
                path = self.write_run(rows=[{"tick": tick}])
                with self.assertRaisesRegex(ValueError, "tick must contain"):
                    load_runs([path])
        with self.assertRaisesRegex(ValueError, "no data rows"):
            load_runs([self.write_run(rows=[])])

    def test_optional_columns_warn_and_remain_unavailable(self):
        path = self.write_run(columns=["tick", "population", "mean_memory_size", "mean_learning_updates"])
        with self.assertWarnsRegex(UserWarning, "missing optional metrics"):
            data = load_runs([path])
        runs, final, _ = build_tables(data)
        self.assertTrue(pd.isna(runs.loc[0, "births"]))
        row = self.metric(final, "births")
        self.assertEqual(row["n"], 0)
        self.assertTrue(pd.isna(row["mean"]))

    def test_invalid_numeric_measurements_warn_and_are_excluded(self):
        path = self.write_run(rows=[{"tick": 0, "mean_memory_size": "bad", "births": "inf"}])
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            data = load_runs([path])
        self.assertEqual(len(recorded), 2)
        self.assertTrue(pd.isna(data.loc[0, "mean_memory_size"]))
        self.assertTrue(pd.isna(data.loc[0, "births"]))

    def test_extinction_keeps_zero_population_and_counters_but_not_living_means(self):
        a = self.write_run(seed=0, rows=[{"tick": 1, "population": 0, "mean_memory_size": 0,
                                       "total_learning_updates": 123}])
        b = self.write_run(seed=1, rows=[{"tick": 1, "population": 20, "mean_memory_size": 10},
                                       {"tick": 2, "population": 30}])
        with self.assertWarnsRegex(UserWarning, "population=0"):
            runs, final, series = build_tables(load_runs([a, b]))
        self.assertEqual(runs.loc[0, "total_learning_updates"], 123)
        self.assertEqual(self.metric(final, "population")["mean"], 15)
        self.assertEqual(self.metric(final, "population")["n"], 2)
        self.assertEqual(self.metric(series, "mean_memory_size", tick=1)["mean"], 10)
        self.assertEqual(self.metric(series, "population", tick=1)["mean"], 10)
        self.assertEqual(self.metric(series, "population", tick=2)["n"], 1)  # No post-extinction extension.

    def test_outputs_are_reproducible_preserve_inputs_and_use_no_randomness(self):
        paths = [self.write_run(0, 1), self.write_run(0, 9, rows=[{"tick": 1}]), self.write_run(4, 2)]
        before_files = {p: p.read_bytes() for p in paths}
        before_random, before_numpy = random.getstate(), np.random.get_state()
        first, second = self.root / "out1", self.root / "out2"
        console = io.StringIO()
        with warnings.catch_warnings(record=True) as recorded, redirect_stdout(console):
            warnings.simplefilter("always")
            analyze(self.root, first)
            analyze(self.root, second)
        self.assertTrue(any("Tick grids differ" in str(w.message) for w in recorded))
        self.assertTrue(any("Final ticks differ" in str(w.message) for w in recorded))
        self.assertIn("Found 3 runs", console.getvalue())
        self.assertIn("perception=0: 2 seeds", console.getvalue())
        self.assertEqual(random.getstate(), before_random)
        self.assertEqual(np.random.get_state()[0], before_numpy[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before_numpy[1])
        self.assertEqual(np.random.get_state()[2:], before_numpy[2:])
        for path, original in before_files.items():
            self.assertEqual(path.read_bytes(), original)
        names = {p.name for p in first.iterdir()}
        self.assertEqual(names, {"population_over_time.png", "memory_size_over_time.png",
                                "learning_updates_over_time.png", "final_population.png", "death_causes_final.png",
                                "traits_over_time.png", "final_metrics.csv", "run_summary.csv",
                                "time_series_metrics.csv", "analysis_metadata.json"})
        for name in names:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
        metadata = json.loads((first / "analysis_metadata.json").read_text())
        self.assertEqual(metadata["inputs"][0]["sha256"], hashlib.sha256(paths[0].read_bytes()).hexdigest())

    def test_direct_script_and_module_entrypoints_handle_all_extinct_data(self):
        self.write_run(rows=[{"tick": 1, "population": 0, "mean_memory_size": None,
                             "mean_learning_updates": None, "mean_hunger_multiplier": None,
                             "mean_thirst_multiplier": None, "mean_signal_probability": None}])
        for command in (["-m", "experiments.analyze_fixed_perception"], ["experiments/analyze_fixed_perception.py"]):
            result = subprocess.run([sys.executable, *command, "--data-dir", str(self.root),
                                     "--output-dir", str(self.root / "output")],
                                    check=True, capture_output=True, text=True)
            self.assertIn("Found 1 runs", result.stdout)
            self.assertIn("mean=nan, std=nan, n=0", result.stdout)
            self.assertTrue((self.root / "output" / "traits_over_time.png").is_file())


if __name__ == "__main__":
    unittest.main()
