"""Synthetic arithmetic checks only. No simulation imports or execution."""
import importlib.util
import unittest

HAS_ANALYSIS = all(importlib.util.find_spec(name) is not None for name in ("pandas", "matplotlib"))
if HAS_ANALYSIS:
    import pandas as pd
    from experiments.analyze_perception_transition import (
        METRICS, TRAITS, EXPOSURES, window_metrics, summarize, differences, early_changes,
        matched_tick_differences, joint_patterns,
    )


@unittest.skipUnless(HAS_ANALYSIS, "Optional analysis dependencies unavailable")
class PerceptionTransitionTests(unittest.TestCase):
    def data(self):
        rows = []
        for seed in (0, 9):
            for p in (0, 1, 2):
                for tick, population, trait in ((0, 100, 100), (1, 1, 1), (1000, 3, 2), (1001, 2, 3), (2000, 2, 4)):
                    rows.append(dict(seed=seed, perception_radius=p, tick=tick, living_population=population,
                        living_agent_ticks=population if tick else 0, action_opportunities=population if tick else 0,
                        successful_parent_participations=1 if tick else 0, deaths=0,
                        population_share=0.2 if tick==1 else 0.4, **dict.fromkeys(TRAITS, trait)))
        return pd.DataFrame(rows)

    def test_boundaries_exposure_weighted_means_and_share_time_mean(self):
        result = window_metrics(self.data())
        row = result[(result.seed==0)&(result.perception_radius==0)&(result.window_end==1000)].iloc[0]
        self.assertEqual(row.observed_ticks, 2)
        self.assertFalse(row.complete_window)
        self.assertEqual(row.action_opportunities, 4)
        self.assertEqual(row.reproductive_participation_rate, 2/4)
        self.assertAlmostEqual(row.population_share, 0.3)
        self.assertEqual(row.mean_age, 7/4)
        self.assertEqual(row.mean_age_equal_present_tick_mean, 1.5)
        later = result[(result.seed==0)&(result.perception_radius==0)&(result.window_end==2000)].iloc[0]
        self.assertEqual(later.mean_age, 3.5)

    def test_absence_zero_denominator_and_missing_window(self):
        frame = self.data()
        frame.loc[frame.perception_radius.eq(0), ["living_population", "living_agent_ticks", "action_opportunities"]] = 0
        frame.loc[frame.perception_radius.eq(0), "population_share"] = 0
        result = window_metrics(frame)
        rows = result[result.perception_radius.eq(0)]
        self.assertTrue(rows.death_rate.isna().all())
        self.assertTrue(rows.mean_age.isna().all())
        self.assertTrue(rows[rows.window_end.le(2000)].population_share.eq(0).all())
        self.assertTrue(rows[rows.window_end.gt(2000)].population_share.isna().all())
        self.assertTrue(rows[rows.window_end.gt(2000)].action_opportunities.isna().all())

    def test_seed_equal_weight_sample_sd_and_sign_missing_counts(self):
        frame = pd.DataFrame([dict(seed=seed, group="test", value=value, action_opportunities=exposure)
                              for seed,value,exposure in ((0,0.1,10),(1,0.3,1000),(2,float("nan"),0))])
        row = summarize(frame, ["group"], metrics=("value",), exposure_columns=("action_opportunities",), signs=True).iloc[0]
        self.assertAlmostEqual(row["mean"], 0.2)
        self.assertAlmostEqual(row["std"], 0.02**0.5)
        self.assertEqual((row.n,row.positive,row.negative,row.zero,row.missing,row.expected_seeds), (2,2,0,0,1,3))
        self.assertEqual(row.action_opportunities_min, 10)

    def test_paired_subtraction_preserves_seed_and_missing(self):
        frame = window_metrics(self.data())
        frame.loc[frame.seed.eq(9)&frame.perception_radius.eq(0), "death_rate"] = 0.1
        frame.loc[frame.seed.eq(0)&frame.perception_radius.eq(2), "death_rate"] = float("nan")
        gaps = differences(frame)
        self.assertTrue(gaps[gaps.seed.eq(0)].death_rate.isna().all())
        self.assertAlmostEqual(gaps[(gaps.seed==9)&(gaps.window_end==1000)].death_rate.iloc[0], 0.1)
        self.assertEqual(len(gaps), 10)

    def test_early_change_is_second_minus_first_and_matched_by_seed(self):
        frame = window_metrics(self.data())
        result = early_changes(frame, ["seed", "perception_radius"], EXPOSURES)
        self.assertTrue(result.mean_age.eq(1.75).all())
        self.assertTrue(result.mean_age_first.eq(1.75).all())
        self.assertTrue(result.mean_age_second.eq(3.5).all())

    def test_matched_ticks_remove_temporal_weighting_difference(self):
        frame = self.data()
        frame.loc[frame.perception_radius.eq(0)&frame.tick.eq(1), "living_population"] = 10
        frame.loc[frame.perception_radius.eq(0)&frame.tick.eq(1000), "living_population"] = 1
        primary = differences(window_metrics(frame))
        self.assertLess(primary[primary.window_end.eq(1000)].mean_generation.iloc[0], 0)
        matched = matched_tick_differences(frame)
        self.assertTrue(matched[matched.window_end.eq(1000)].mean_generation.eq(0).all())
        self.assertTrue(matched[matched.window_end.eq(1000)].shared_present_ticks.eq(2).all())
        frame.loc[frame.perception_radius.eq(0)&frame.tick.eq(1), "living_population"] = 0
        matched = matched_tick_differences(frame)
        self.assertTrue(matched[matched.window_end.eq(1000)].shared_present_ticks.eq(1).all())

    def test_order_independence(self):
        frame = self.data()
        a, b = window_metrics(frame), window_metrics(frame.iloc[::-1])
        pd.testing.assert_frame_equal(a,b)
        pd.testing.assert_frame_equal(differences(a), differences(b))
        pd.testing.assert_frame_equal(matched_tick_differences(frame), matched_tick_differences(frame.iloc[::-1]))

    def test_joint_sign_counts_do_not_treat_missing_as_false(self):
        rows = [dict(seed=0,window_end=2000,reproductive_participation_rate=0.1,death_rate=-0.1,
                     mean_hunger_multiplier=0.1,mean_thirst_multiplier=0.1),
                dict(seed=1,window_end=2000,reproductive_participation_rate=float("nan"),death_rate=-0.1,
                     mean_hunger_multiplier=0.1,mean_thirst_multiplier=0.1)]
        result = joint_patterns(pd.DataFrame(rows)).iloc[0]
        self.assertEqual((result.n,result.expected_seeds,result.both_rate_directions_with_neither_metabolic_lower), (1,2,1))


if __name__ == "__main__":
    unittest.main()
