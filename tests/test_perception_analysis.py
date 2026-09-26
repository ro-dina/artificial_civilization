from contextlib import redirect_stdout
from dataclasses import asdict
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

from agents.genome import Genome
from simulation.cohorts import CohortStatistics

HAS_ANALYSIS = all(importlib.util.find_spec(m) is not None for m in ("pandas", "matplotlib"))
if HAS_ANALYSIS:
    import pandas as pd
    from experiments import analyze_evolving_perception as evolving
    from experiments import analyze_perception_cohorts as cohorts
    from experiments import analyze_lineages as lineages
    from experiments.perception_analysis_common import aggregate_metrics, seed_files


@unittest.skipUnless(HAS_ANALYSIS, "Optional analysis dependencies not installed")
class PerceptionAnalysisTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def evolving_file(self, seed, rows):
        path = self.root / f"v03_learning_seed_{seed}_5000.csv"
        pd.DataFrame([{**dict.fromkeys(evolving.OPTIONAL, 1), "brain": "learning", "seed": seed, **row}
                      for row in rows]).to_csv(path, index=False)
        return seed, path

    def cohort_file(self, seed, rows=None):
        path = self.root / f"evolving_seed_{seed}_cohorts.csv"
        if rows is None:
            rows = [asdict(CohortStatistics(tick=tick, perception_radius=p, living_population=10 if p == 0 else 0,
                    living_agent_ticks=10 if tick and p == 0 else 0, action_opportunities=10 if tick and p == 0 else 0,
                    mean_hunger_multiplier=1 if p == 0 else None, mean_thirst_multiplier=1 if p == 0 else None,
                    mean_signal_probability=0.1 if p == 0 else None)) for tick in (0, 1, 2) for p in range(9)]
        pd.DataFrame(rows).to_csv(path, index=False)
        return seed, path

    def test_seed_discovery_numeric_order_duplicates_and_missing_seeds(self):
        for seed in (9, 2):
            self.cohort_file(seed)
        files = seed_files(self.root, "*_cohorts.csv")
        self.assertEqual([seed for seed, _ in files], [2, 9])
        files[0][1].with_name("copy_seed_2_cohorts.csv").write_bytes(files[0][1].read_bytes())
        with self.assertRaisesRegex(ValueError, "Multiple inputs"):
            seed_files(self.root, "*_cohorts.csv")

    def test_evolving_final_row_nan_file_order_and_sample_sd(self):
        a = self.evolving_file(3, [{"tick": 2, "population": 0, "mean_perception_radius": 5},
                                    {"tick": 0, "population": 100, "mean_perception_radius": 2}])
        b = self.evolving_file(7, [{"tick": 0, "population": 2, "mean_perception_radius": 4}])
        frame = evolving.load_evolving([a, b])
        pd.testing.assert_frame_equal(frame, evolving.load_evolving([b, a]))
        summary = aggregate_metrics(frame, ["tick"], ("mean_perception_radius",))
        initial = summary[summary.tick.eq(0)].iloc[0]
        self.assertEqual((initial["mean"], initial.n), (3, 2))
        self.assertAlmostEqual(initial["std"], math.sqrt(2))
        last = frame[frame.seed.eq(3)].iloc[-1]
        self.assertEqual(last.tick, 2)
        self.assertTrue(math.isnan(last.mean_perception_radius))

    def test_evolving_required_columns_and_wrong_experiment_errors(self):
        seed, path = self.evolving_file(0, [{"tick": 0, "population": 10, "mean_perception_radius": 2}])
        sidecar = path.with_suffix(".csv.metadata.json")
        sidecar.write_text(json.dumps({"config": {"fixed_perception_radius": 2}}))
        with self.assertRaisesRegex(ValueError, "not free evolution"):
            evolving.load_evolving([(seed, path)])
        sidecar.unlink()
        pd.DataFrame([{"tick": 0, "population": 10}]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "missing required"):
            evolving.load_evolving([(seed, path)])

    def test_cohort_zero_exposure_and_zero_population_are_nan(self):
        frame = cohorts.load_cohorts([self.cohort_file(0)])
        summary, windows, window_summary, series = cohorts.build_tables(frame, 2)
        self.assertTrue(summary.loc[summary.perception_radius.eq(1), "death_rate"].isna().all())
        self.assertEqual(summary.loc[summary.perception_radius.eq(0), "action_opportunities"].iloc[0], 20)
        self.assertTrue(windows.complete_window.all())
        self.assertTrue(window_summary["std"].isna().all())
        self.assertTrue(series.loc[series.metric.eq("mean_hunger_multiplier") & series.perception_radius.eq(1), "n"].eq(0).all())
        dead = frame.drop(columns="population_share").copy()
        dead[["living_population", "living_agent_ticks", "action_opportunities"]] = 0
        path = self.root / "evolving_seed_0_cohorts.csv"
        dead.to_csv(path, index=False)
        extinct = cohorts.load_cohorts([(0, path)])
        self.assertTrue(extinct.population_share.isna().all())

    def test_window_boundaries_tick_zero_and_partial_windows(self):
        frame = cohorts.load_cohorts([self.cohort_file(0)])
        extra = frame[frame.tick.eq(2)].assign(tick=3)
        frame = pd.concat([frame, extra], ignore_index=True)
        _, windows, _, _ = cohorts.build_tables(frame, 2)
        p0 = windows[windows.perception_radius.eq(0)]
        self.assertEqual(list(p0.window_start), [0, 2])
        self.assertEqual(list(p0.action_opportunities), [20, 10])
        self.assertEqual(list(p0.complete_window), [True, False])
        with self.assertRaises(ValueError):
            cohorts.build_tables(frame, 0)

    def test_seeds_are_independent_before_equal_weight_rate_mean(self):
        a = cohorts.load_cohorts([self.cohort_file(1)])
        b = a.assign(seed=8).copy()
        a.loc[a.tick.gt(0) & a.perception_radius.eq(0), "deaths"] = 2
        b.loc[b.tick.gt(0) & b.perception_radius.eq(0), ["action_opportunities", "deaths"]] = [100, 4]
        data = pd.concat([b, a], ignore_index=True)
        summary, _, windows, _ = cohorts.build_tables(data, 2)
        self.assertEqual(set(summary.seed), {1, 8})
        rates = windows[windows.perception_radius.eq(0) & windows.metric.eq("death_rate")].iloc[0]
        self.assertAlmostEqual(rates["mean"], (0.2 + 0.04) / 2)
        self.assertAlmostEqual(rates["std"], pd.Series([0.2, 0.04]).std(ddof=1))
        self.assertNotAlmostEqual(rates["mean"], 12 / 220)
        shuffled = data.iloc[::-1]
        for expected, actual in zip(cohorts.build_tables(data, 2), cohorts.build_tables(shuffled, 2)):
            pd.testing.assert_frame_equal(expected, actual)

    def test_missing_ticks_warn_and_are_not_imputed(self):
        seed, path = self.cohort_file(0)
        data = pd.read_csv(path)
        data[data.tick.ne(1)].to_csv(path, index=False)
        with self.assertWarnsRegex(UserWarning, "missing ticks"):
            frame = cohorts.load_cohorts([(seed, path)])
        _, windows, _, _ = cohorts.build_tables(frame, 2)
        self.assertTrue(windows.observed_ticks.eq(1).all())
        self.assertFalse(windows.complete_window.any())
        self.assertEqual(windows.action_opportunities.sum(), 10)

    def test_reject_missing_cohorts_duplicates_and_bad_counts(self):
        seed, path = self.cohort_file(0)
        data = pd.read_csv(path)
        for invalid in (data.iloc[1:], pd.concat([data, data.iloc[:1]]), data.assign(action_opportunities=-1),
                        data.assign(successful_parent_participations=1), data.assign(perception_radius=9)):
            invalid.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                cohorts.load_cohorts([(seed, path)])

    def test_tick_zero_only_is_supported(self):
        seed, path = self.cohort_file(0)
        data = pd.read_csv(path)
        data[data.tick.eq(0)].to_csv(path, index=False)
        summary, windows, _, _ = cohorts.build_tables(cohorts.load_cohorts([(seed, path)]))
        self.assertTrue(summary.death_rate.isna().all())
        self.assertTrue(windows.empty)

    def test_deterministic_analysis_outputs_and_rng_unchanged(self):
        self.cohort_file(2)
        self.evolving_file(2, [{"tick": 0, "population": 10, "mean_perception_radius": 2},
                                {"tick": 1, "population": 9, "mean_perception_radius": 1}])
        state = random.getstate()
        import numpy as np
        numpy_state = np.random.get_state()
        for module in (evolving, cohorts):
            outputs = []
            for name in ("first", "second"):
                output = self.root / module.__name__ / name
                with redirect_stdout(io.StringIO()):
                    module.analyze(self.root, output)
                outputs.append({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()})
            self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(random.getstate(), state)
        self.assertEqual(np.random.get_state()[0], numpy_state[0])
        self.assertTrue((np.random.get_state()[1] == numpy_state[1]).all())
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])

    def lineage_file(self, bad_parent=False):
        path = self.root / "evolving_seed_0_lineage.jsonl"
        genome = asdict(Genome())
        records = [dict(tick=0, kind="founder", agent_id=i, details={"generation": 0, "genome": genome}) for i in range(2)]
        records += [dict(tick=1, kind="birth", agent_id=2, details={"parent_a_id": 99 if bad_parent else 0,
            "parent_b_id": 1, "generation": 1, "child_genome": {**genome, "perception_radius": 0}}),
            dict(tick=2, kind="death", agent_id=0, details={"genome_at_death": genome, "death_cause": "energy"}),
            dict(tick=2, kind="birth", agent_id=3, details={"parent_a_id": 0, "parent_b_id": 2,
                "generation": 2, "child_genome": genome}),
            dict(tick=5, kind="run_finished", details={"population": 3})]
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return path

    def test_lineage_both_parents_overlap_death_birth_same_tick_and_final_horizon(self):
        records, founders, edges, horizon = lineages.read_lineage(self.lineage_file())
        self.assertEqual(horizon, 5)  # Last birth tick 2 is not the end of this run.
        self.assertEqual(records[3]["ancestors"], 3)
        self.assertEqual((edges[1]["parent_a_id"], edges[1]["parent_b_id"]), (0, 2))
        descendants, traits, mixing = lineages.summarize(records, founders, [1, 2, 5], 0)
        last = [r for r in descendants if r["tick"] == 5]
        self.assertEqual([r["living_descendants"] for r in last], [2, 2])
        self.assertGreater(sum(r["living_descendants"] for r in last), mixing[-1]["living_population"])
        self.assertAlmostEqual(mixing[-1]["mean_founder_ancestors"], 5 / 3)
        self.assertTrue(any(r["perception_radius"] == 0 and r["living_descendants"] == 1 for r in traits))

    def test_lineage_rejects_missing_parent_and_warns_truncated_horizon(self):
        with self.assertRaisesRegex(ValueError, "recorded parents"):
            lineages.read_lineage(self.lineage_file(bad_parent=True))
        path = self.lineage_file()
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        with self.assertWarnsRegex(UserWarning, "no run_finished"):
            self.assertEqual(lineages.read_lineage(path)[-1], 2)

    def test_lineage_outputs_are_deterministic(self):
        self.lineage_file()
        outputs = []
        for name in ("a", "b"):
            output = self.root / name
            with redirect_stdout(io.StringIO()):
                lineages.analyze(self.root, output, ticks=(0, 1, 5))
            outputs.append({p.name: p.read_bytes() for p in output.iterdir()})
        self.assertEqual(outputs[0], outputs[1])

    def test_both_entrypoints_for_all_three_analyses(self):
        for name in ("analyze_evolving_perception", "analyze_perception_cohorts", "analyze_lineages"):
            for args in (["-m", f"experiments.{name}"], [f"experiments/{name}.py"]):
                result = subprocess.run([sys.executable, *args, "--help"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--pattern", result.stdout)


if __name__ == "__main__":
    unittest.main()
