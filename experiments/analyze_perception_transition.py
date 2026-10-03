"""Offline paired-seed comparisons of perception cohorts. Never runs a simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .analyze_perception_cohorts import load_cohorts
    from .perception_analysis_common import plotting, save_plot, seed_files, write_tables, metadata
else:
    from analyze_perception_cohorts import load_cohorts
    from perception_analysis_common import plotting, save_plot, seed_files, write_tables, metadata

TRAITS = ("mean_hunger_multiplier", "mean_thirst_multiplier", "mean_signal_probability", "mean_age", "mean_generation")
METRICS = ("population_share", "reproductive_participation_rate", "death_rate", *TRAITS)
LABELS = ("Population share", "Successful parent participations / action opportunity", "Deaths / action opportunity",
          "Hunger multiplier", "Thirst multiplier", "Signal probability", "Age (ticks)", "Generation")
TITLES = dict(zip(METRICS, LABELS))
WINDOWS = (1000, 2000, 3000, 4000, 5000)
WINDOW_LABELS = [f"({end-1000},{end}]" for end in WINDOWS]
EXPOSURES = ("action_opportunities", "living_agent_ticks")
KEYS = ["seed", "perception_radius", "window_end"]


def window_metrics(data):
    """One observation per seed/cohort/window. Tick 0 never contributes exposure."""
    rows = []
    for seed in sorted(data.seed.unique()):
        for p in (0, 1, 2):
            cohort = data[data.seed.eq(seed) & data.perception_radius.eq(p)]
            for end in WINDOWS:
                part = cohort[cohort.tick.gt(end - 1000) & cohort.tick.le(end)]
                n_ticks = part.tick.nunique()
                actions, living = (part[name].sum() if n_ticks else float("nan") for name in EXPOSURES)
                parents = part.successful_parent_participations.sum() if n_ticks else float("nan")
                deaths = part.deaths.sum() if n_ticks else float("nan")
                row = dict(seed=seed, perception_radius=p, window_start=end-1000, window_end=end,
                    observed_ticks=n_ticks, complete_window=n_ticks == 1000,
                    present_ticks=int(part.living_population.gt(0).sum()),
                    action_opportunities=actions, living_agent_ticks=living,
                    successful_parent_participations=parents, deaths=deaths,
                    population_share=part.population_share.mean(),
                    population_share_valid_ticks=int(part.population_share.notna().sum()),
                    reproductive_participation_rate=parents / actions if actions > 0 else float("nan"),
                    death_rate=deaths / actions if actions > 0 else float("nan"))
                for trait in TRAITS:
                    valid = part[part[trait].notna() & part.living_population.gt(0)]
                    support = valid.living_population.sum()
                    row[trait] = (valid[trait] * valid.living_population).sum() / support if support else float("nan")
                    row[trait + "_valid_living_exposure"] = support
                    row[trait + "_valid_ticks"] = len(valid)
                    row[trait + "_equal_present_tick_mean"] = valid[trait].mean()
                rows.append(row)
    return pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)


def describe(values):
    values = values.dropna()
    return {"mean": values.mean(), "std": values.std(ddof=1), "min": values.min(),
            "max": values.max(), "n": len(values), "positive": int(values.gt(0).sum()),
            "negative": int(values.lt(0).sum()), "zero": int(values.eq(0).sum())}


def summarize(table, keys, metrics=METRICS, exposure_columns=EXPOSURES, signs=False):
    rows = []
    for identity, group in table.groupby(keys, sort=True, dropna=False):
        if len(keys) == 1:
            identity = (identity,) if not isinstance(identity, tuple) else identity
        for metric in metrics:
            stats = describe(group[metric])
            if not signs:
                for key in ("positive", "negative", "zero"):
                    stats.pop(key)
            row = {**dict(zip(keys, identity)), "metric": metric, **stats,
                   "expected_seeds": group.seed.nunique(), "missing": int(group[metric].isna().sum())}
            valid = group[group[metric].notna()]
            for name in exposure_columns:
                row.update({f"{name}_{suffix}": getattr(valid[name], suffix)() for suffix in ("median", "min", "max")})
            rows.append(row)
    return pd.DataFrame(rows)


def differences(table):
    a = table[table.perception_radius.eq(0)].set_index(["seed", "window_end"])
    b = table[table.perception_radius.eq(2)].set_index(["seed", "window_end"])
    result = a[list(METRICS)].subtract(b[list(METRICS)])
    for name in EXPOSURES:
        result[f"p0_{name}"] = a[name]
        result[f"p2_{name}"] = b[name]
    for trait in TRAITS:
        name = trait + "_equal_present_tick_mean"
        result[name] = a[name] - b[name]
    return result.reset_index()


def early_changes(table, keys, exposures):
    first = table[table.window_end.eq(1000)].set_index(keys)
    second = table[table.window_end.eq(2000)].set_index(keys)
    result = second[list(METRICS)].subtract(first[list(METRICS)])
    for metric in METRICS:
        result[metric + "_first"] = first[metric]
        result[metric + "_second"] = second[metric]
    for name in exposures:
        result[name + "_first"] = first[name]
        result[name + "_second"] = second[name]
    return result.reset_index()


def joint_patterns(gaps):
    rows = []
    for end, group in gaps.groupby("window_end", sort=True):
        valid = group.dropna(subset=["reproductive_participation_rate", "death_rate",
                                   "mean_hunger_multiplier", "mean_thirst_multiplier"])
        higher = valid.reproductive_participation_rate.gt(0)
        lower = valid.death_rate.lt(0)
        both_metabolic = valid.mean_hunger_multiplier.lt(0) & valid.mean_thirst_multiplier.lt(0)
        neither_metabolic = valid.mean_hunger_multiplier.ge(0) & valid.mean_thirst_multiplier.ge(0)
        rows.append(dict(window_end=end, n=len(valid), expected_seeds=group.seed.nunique(),
            higher_reproduction_and_lower_death=int((higher & lower).sum()),
            both_metabolic_multipliers_lower=int(both_metabolic.sum()),
            neither_metabolic_multiplier_lower=int(neither_metabolic.sum()),
            both_rate_directions_with_neither_metabolic_lower=int((higher & lower & neither_metabolic).sum()),
            seeds_both_rate_directions_with_neither_metabolic_lower=",".join(map(str, valid.loc[higher & lower & neither_metabolic, "seed"]))))
    return pd.DataFrame(rows)


def matched_tick_differences(data):
    """Sensitivity to differing temporal composition: compare living cohorts at SAME ticks."""
    a = data[data.perception_radius.eq(0)].set_index(["seed", "tick"])
    b = data[data.perception_radius.eq(2)].set_index(["seed", "tick"])
    common = a.living_population.gt(0) & b.living_population.gt(0)
    delta = (a[list(TRAITS)] - b[list(TRAITS)]).where(common, float("nan"))
    rows = []
    for seed in sorted(data.seed.unique()):
        for end in WINDOWS:
            index = (a.index.get_level_values("seed") == seed) & (a.index.get_level_values("tick") > end-1000) & (a.index.get_level_values("tick") <= end)
            valid = index & common
            row = dict(seed=seed, window_end=end, shared_present_ticks=int(valid.sum()))
            for metric in TRAITS:
                values = delta.loc[valid, metric]
                row[metric] = values.mean()
                row[metric + "_valid_ticks"] = int(values.notna().sum())
            for p, frame in ((0, a), (2, b)):
                row[f"p{p}_action_opportunities"] = frame.loc[valid, "action_opportunities"].sum()
            rows.append(row)
    return pd.DataFrame(rows)


def exposure_label(group, metric, paired=False):
    valid = group[group[metric].notna()]
    names = ("p0_action_opportunities", "p2_action_opportunities") if paired else ("action_opportunities",)
    text = f"n={len(valid)}/{len(group)}"
    for name in names:
        values = valid[name].dropna()
        prefix = name[:2] + " E" if paired else "E"
        text += f"\n{prefix}: {values.median():,.0f} [{values.min():,.0f}, {values.max():,.0f}]" if len(values) else f"\n{prefix}: NA"
    return text


def plot_metrics(per_seed, summary, gaps, gap_summary, matched, matched_summary, output):
    plt, style = plotting()
    with style:
        for metric in METRICS:
            fig, (ax, support) = plt.subplots(2, 1, figsize=(13, 7.2), gridspec_kw={"height_ratios": [4, 1.8]})
            for p in (0, 1, 2):
                points = per_seed[per_seed.perception_radius.eq(p)]
                rows = summary[summary.perception_radius.eq(p) & summary.metric.eq(metric)].sort_values("window_end")
                x = np.arange(5) + (p-1)*0.22
                for i, end in enumerate(WINDOWS):
                    samples = points[points.window_end.eq(end)].sort_values("seed")
                    ax.scatter(x[i] + np.linspace(-0.05, 0.05, len(samples)), samples[metric], color=f"C{p}", s=13, alpha=0.45)
                ax.errorbar(x, rows["mean"], yerr=rows["std"], marker="o", color=f"C{p}", capsize=3, label=f"p{p}")
            ax.set(xticks=np.arange(5), xticklabels=WINDOW_LABELS, ylabel=TITLES[metric],
                   title=f"{TITLES[metric]} by perception cohort\nEqual-seed mean ±1 sample SD; dots are individual seeds")
            ax.legend(ncol=3)
            ax.grid(alpha=0.2)
            support.axis("off")
            cells = [[exposure_label(per_seed[per_seed.perception_radius.eq(p) & per_seed.window_end.eq(end)], metric)
                      for end in WINDOWS] for p in (0, 1, 2)]
            table = support.table(cellText=cells, rowLabels=["p0", "p1", "p2"], colLabels=WINDOW_LABELS,
                                  cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8.5)
            table.scale(1, 2.2)
            support.set_title("Finite seed n; action exposure E = median [min, max] across those seeds", fontsize=9, pad=20)
            save_plot(plt, fig, output / f"{metric}_by_window.png")

        for metric in METRICS:
            rows = gap_summary[gap_summary.metric.eq(metric)].sort_values("window_end")
            fig, (ax, support) = plt.subplots(2, 1, figsize=(13, 6.6), gridspec_kw={"height_ratios": [4, 1.5]})
            for seed, group in gaps.groupby("seed", sort=True):
                ax.plot(np.arange(5), group.sort_values("window_end")[metric], linewidth=0.8, alpha=0.45)
            ax.errorbar(np.arange(5), rows["mean"], yerr=rows["std"], color="black", marker="o", capsize=4, label="Mean ±1 SD")
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set(xticks=np.arange(5), xticklabels=WINDOW_LABELS, ylabel=f"p0 − p2: {TITLES[metric]}",
                   title=f"Paired p0 − p2 difference: {TITLES[metric]}\nThin lines are seeds; signs describe values, not significance")
            ax.legend()
            ax.grid(alpha=0.2)
            cells = []
            for row in rows.itertuples():
                group = gaps[gaps.window_end.eq(row.window_end)]
                cells.append(f"+ / − / 0 = {row.positive} / {row.negative} / {row.zero}\n" + exposure_label(group, metric, paired=True))
            support.axis("off")
            table = support.table(cellText=[cells], colLabels=WINDOW_LABELS, cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8.5)
            for (row, col), cell in table.get_celld().items():
                cell.set_height(0.68 if row else 0.23)
            support.set_title("Paired finite n; p0/p2 action exposure E = median [min, max]", fontsize=9, pad=5)
            save_plot(plt, fig, output / f"p0_minus_p2_{metric}.png")

        fig = plt.figure(figsize=(14, 17))
        grid = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 0.8])
        axes = [fig.add_subplot(grid[i//2, i%2]) for i in range(8)]
        for ax, metric in zip(axes, METRICS):
            for p in (0, 1, 2):
                values = per_seed[per_seed.perception_radius.eq(p) & per_seed.window_end.le(2000)]
                for seed, group in values.groupby("seed", sort=True):
                    ax.plot([0, 1], group.sort_values("window_end")[metric], color=f"C{p}", alpha=0.18, linewidth=0.7)
                group = summary[summary.perception_radius.eq(p) & summary.window_end.le(2000) & summary.metric.eq(metric)].sort_values("window_end")
                ax.errorbar([0, 1], group["mean"], yerr=group["std"], color=f"C{p}", marker="o", capsize=3,
                            label=f"p{p}: n={','.join(map(str, group.n))}")
            ax.set(title=TITLES[metric], xticks=[0,1], xticklabels=WINDOW_LABELS[:2])
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        support = fig.add_subplot(grid[4, :])
        support.axis("off")
        cells = [[exposure_label(per_seed[per_seed.perception_radius.eq(p) & per_seed.window_end.eq(end)], "population_share")
                  for end in WINDOWS[:2]] for p in (0, 1, 2)]
        table = support.table(cellText=cells, rowLabels=["p0", "p1", "p2"], colLabels=WINDOW_LABELS[:2], cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.2)
        support.set_title("Recorded seed count and action E median [min, max]; finite metric n in each legend", fontsize=10, pad=20)
        fig.suptitle("First two windows: equal-seed mean ±1 SD", fontsize=13)
        save_plot(plt, fig, output / "early_transition.png")

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        for ax, metric in zip(axes.flat, TRAITS):
            for table, label, color in ((gap_summary, "Separate cohort exposure weights", "C0"),
                                        (matched_summary, "Shared ticks, equal tick weights", "C1")):
                group = table[table.metric.eq(metric)].sort_values("window_end")
                ax.errorbar(np.arange(5), group["mean"], yerr=group["std"], marker="o", color=color, capsize=3, label=label)
            ax.axhline(0, color="gray", linewidth=1, linestyle="--")
            ax.set(title=f"p0 − p2: {TITLES[metric]}", xticks=np.arange(5), xticklabels=WINDOW_LABELS)
            ax.tick_params(axis="x", labelsize=8)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        support = axes.flat[-1]
        support.axis("off")
        lines = ["Shared-tick comparison support", "Finite n; action E median [min, max]"]
        for end in WINDOWS:
            lines.append(WINDOW_LABELS[WINDOWS.index(end)] + ": " + exposure_label(matched[matched.window_end.eq(end)], "mean_generation", paired=True))
        support.text(0, 1, "\n".join(lines), va="top", fontsize=9, linespacing=1.25)
        fig.suptitle("Temporal composition sensitivity; seed mean ±1 SD\nShared-tick differences are descriptive, not an age/trait-adjusted causal model")
        save_plot(plt, fig, output / "matched_tick_sensitivity.png")


def analyze(data_dir=Path("data/cohort_replicates"), output_dir=Path("analysis/perception_transition")):
    files = seed_files(data_dir, "evolving_seed_*_cohorts.csv")
    files = [(seed, path) for seed, path in files if seed in range(10)]
    if [seed for seed, _ in files] != list(range(10)):
        raise ValueError("This analysis requires the existing seed 0–9 cohort CSVs")
    fingerprints = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for _, path in files}
    configs, audit = [], []
    for seed, path in files:
        sidecar = json.loads(path.with_suffix(".csv.metadata.json").read_text())
        cfg = sidecar["config"].copy()
        if cfg.pop("random_seed") != seed or cfg.get("brain") != "learning" or cfg.get("fixed_perception_radius") is not None:
            raise ValueError(f"{path}: expected free-evolution LearningBrain with matching seed")
        if sidecar.get("cohort_schema_version") != 1:
            raise ValueError(f"{path}: unsupported cohort schema")
        configs.append(cfg)
    if any(cfg != configs[0] for cfg in configs[1:]):
        raise ValueError("Configurations differ beyond random_seed; do not combine unlike experiments")
    data = load_cohorts(files)
    for seed, group in data.groupby("seed", sort=True):
        audit.append(dict(seed=seed, rows=len(group), first_tick=group.tick.min(), last_tick=group.tick.max(),
                          unique_ticks=group.tick.nunique(), complete_0_to_5000=set(group.tick)==set(range(5001))))
    per_seed = window_metrics(data)
    summary = summarize(per_seed, ["window_end", "perception_radius"])
    wide = summary.set_index(["window_end", "metric", "perception_radius"])[["mean", "std", "min", "max", "n",
        "action_opportunities_median", "action_opportunities_min", "action_opportunities_max"]].unstack("perception_radius")
    wide.columns = [f"p{p}_{name}" for name, p in wide.columns]
    paired_exposures = tuple(f"p{p}_{name}" for p in (0,2) for name in EXPOSURES)
    gaps = differences(per_seed)
    gap_summary = summarize(gaps, ["window_end"], exposure_columns=paired_exposures, signs=True)
    changes = early_changes(per_seed, ["seed", "perception_radius"], EXPOSURES)
    change_exposures = tuple(name + suffix for name in EXPOSURES for suffix in ("_first", "_second"))
    change_summary = summarize(changes, ["perception_radius"], exposure_columns=change_exposures, signs=True)
    gap_changes = early_changes(gaps, ["seed"], paired_exposures).assign(comparison="second_window_gap_minus_first_window_gap")
    gap_change_exposures = tuple(name + suffix for name in paired_exposures for suffix in ("_first", "_second"))
    gap_change_summary = summarize(gap_changes, ["comparison"], exposure_columns=gap_change_exposures, signs=True)
    sensitivity_metrics = tuple(name + "_equal_present_tick_mean" for name in TRAITS)
    sensitivity = summarize(gaps, ["window_end"], metrics=sensitivity_metrics, exposure_columns=paired_exposures, signs=True)
    matched = matched_tick_differences(data)
    matched_summary = summarize(matched, ["window_end"], metrics=TRAITS,
                                exposure_columns=("p0_action_opportunities", "p2_action_opportunities"), signs=True)
    output = Path(output_dir)
    write_tables(output, window_metrics_by_seed=per_seed, window_metrics_summary=summary,
        window_comparison=wide.reset_index(), p0_minus_p2_by_seed=gaps, p0_minus_p2_summary=gap_summary,
        early_transition_by_seed=changes, early_transition_summary=change_summary,
        paired_gap_transition_by_seed=gap_changes, paired_gap_transition_summary=gap_change_summary,
        equal_tick_weighting_sensitivity=sensitivity, matched_tick_differences_by_seed=matched,
        matched_tick_difference_summary=matched_summary, joint_sign_patterns=joint_patterns(gaps), input_audit=pd.DataFrame(audit))
    plot_metrics(per_seed, summary, gaps, gap_summary, matched, matched_summary, output)
    metadata(output, files, __file__, source_directory=str(Path(data_dir)), seeds=list(range(10)),
        windows="(0,1000], (1000,2000], (2000,3000], (3000,4000], (4000,5000]",
        population_share="Arithmetic time mean of cohort_population / total_population over recorded ticks; absent cohort is zero when total>0",
        traits_age_generation="Sum(recorded living-cohort mean * living_population) / sum(living_population) over finite observations within seed/window",
        rates="Sum(events) / sum(action_opportunities) within seed/window; zero denominator is NaN",
        sensitivity="Equal-weight time means over cohort-present finite ticks, then paired p0-p2 differences",
        matched_tick_sensitivity="Mean(p0_cohort_mean(t) - p2_cohort_mean(t)) over shared living ticks within seed/window; same time weights for both cohorts",
        signs="Strictly positive/negative/exact zero, paired within seed; missing excluded with n reported; no significance test",
        learning="No memory or learning-update columns recorded in cohort CSV; no cohort-level learning inference",
        simulation_executed=False)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in fingerprints.items()):
        raise RuntimeError("Input changed during analysis")
    print("Existing seeds 0–9 only. No simulation executed. Cohort learning metrics unavailable.")
    print(gap_summary[["window_end", "metric", "mean", "std", "n", "positive", "negative", "zero"]].to_string(index=False))
    print(f"Output: {output}")
    return per_seed, summary, gaps, gap_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/cohort_replicates"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/perception_transition"))
    args = parser.parse_args()
    try:
        analyze(args.data_dir, args.output_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
