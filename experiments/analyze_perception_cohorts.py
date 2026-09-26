"""Offline, equal-seed cohort summaries. Rates use actual action opportunities."""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

if __package__:
    from .perception_analysis_common import (pd, aggregate_metrics, seed_files, numeric_csv, check_seed,
        write_tables, plotting, save_plot, mean_band, metadata, warn_grids)
else:
    from perception_analysis_common import (pd, aggregate_metrics, seed_files, numeric_csv, check_seed,
        write_tables, plotting, save_plot, mean_band, metadata, warn_grids)

COUNTERS = ("living_agent_ticks", "action_opportunities", "births_as_child", "successful_parent_participations",
            "reproduction_initiations", "failed_reproduction_attempts", "deaths", "starvation_deaths",
            "dehydration_deaths", "old_age_deaths", "energy_deaths")
TRAITS = ("mean_hunger_multiplier", "mean_thirst_multiplier", "mean_signal_probability")
OPTIONAL = ("mean_generation", "mean_age")
REQUIRED = ("tick", "perception_radius", "living_population", *COUNTERS, *TRAITS)
RATE_NUMERATORS = {"reproductive_participation_rate": "successful_parent_participations",
                   "death_rate": "deaths", "reproduction_initiation_rate": "reproduction_initiations"}


def load_cohorts(files):
    frames = []
    for seed, path in files:
        frame = numeric_csv(path, REQUIRED, OPTIONAL, keys=("tick", "perception_radius"))
        check_seed(frame, seed, path)
        if not frame.perception_radius.between(0, 8).all():
            raise ValueError(f"{path}: perception_radius must be in [0, 8]")
        if not frame.groupby("tick").perception_radius.nunique().eq(9).all():
            raise ValueError(f"{path}: each recorded tick needs all nine cohorts (including zero counts)")
        counts = frame[["living_population", *COUNTERS]]
        if (counts.isna() | (counts < 0) | (counts % 1 != 0)).any().any():
            raise ValueError(f"{path}: cohort counts must be finite nonnegative integers")
        if frame.loc[frame.tick.eq(0), list(COUNTERS)].ne(0).any().any():
            raise ValueError(f"{path}: tick 0 must have zero event counts and exposure")
        after_zero = frame.tick.gt(0)
        if not frame.loc[after_zero, "living_agent_ticks"].eq(frame.loc[after_zero, "living_population"]).all():
            raise ValueError(f"{path}: living_agent_ticks must equal the completed living snapshot")
        causes = frame[["starvation_deaths", "dehydration_deaths", "old_age_deaths", "energy_deaths"]].sum(axis=1)
        if not causes.eq(frame.deaths).all():
            raise ValueError(f"{path}: death causes do not sum to deaths")
        total = frame.groupby("tick")[["births_as_child", "successful_parent_participations",
                                      "reproduction_initiations", "failed_reproduction_attempts"]].sum()
        if (not total.successful_parent_participations.eq(2 * total.births_as_child).all()
                or not (total.reproduction_initiations - total.failed_reproduction_attempts).eq(total.births_as_child).all()):
            raise ValueError(f"{path}: births, parent participations, and initiation counts disagree")
        frame.loc[frame.living_population.eq(0), [*TRAITS, *OPTIONAL]] = float("nan")
        denominator = frame.groupby("tick").living_population.transform("sum").replace(0, float("nan"))
        frame["population_share"] = frame.living_population / denominator
        ticks = sorted(frame.tick.unique())
        if any(b != a + 1 for a, b in zip(ticks, ticks[1:])):
            warnings.warn(f"{path.name}: missing ticks; exposures/events are totals over recorded ticks only", stacklevel=2)
        frames.append(frame[["seed", *REQUIRED, *OPTIONAL, "population_share"]])
    return pd.concat(frames, ignore_index=True).sort_values(["seed", "tick", "perception_radius"]).reset_index(drop=True)


def add_rates(table):
    table = table.copy()
    denominator = table.action_opportunities.replace(0, float("nan"))
    for rate, numerator in RATE_NUMERATORS.items():
        table[rate] = table[numerator] / denominator
    return table


def build_tables(data, window_size=1000):
    if type(window_size) is not int or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    # Tick zero is an initial condition, not an elapsed agent-tick.
    elapsed = data[data.tick.gt(0)].copy()
    keys = ["seed", "perception_radius"]
    summary = data.groupby(keys, sort=True)[list(COUNTERS)].sum().reset_index()
    coverage = data.groupby(keys, sort=True).agg(first_tick=("tick", "min"), final_tick=("tick", "max"))
    summary = summary.merge(coverage.reset_index(), on=keys)
    observed = elapsed.groupby(keys).tick.nunique()
    summary["observed_action_ticks"] = [int(observed.get((r.seed, r.perception_radius), 0)) for r in summary.itertuples()]
    summary = add_rates(summary)
    elapsed["window_start"] = ((elapsed.tick - 1) // window_size) * window_size
    window_keys = [*keys, "window_start"]
    windows = elapsed.groupby(window_keys, sort=True)[list(COUNTERS)].sum().reset_index()
    coverage = elapsed.groupby(window_keys, sort=True).agg(observed_ticks=("tick", "nunique"),
                                                         first_observed_tick=("tick", "min"), last_observed_tick=("tick", "max"))
    windows = windows.merge(coverage.reset_index(), on=window_keys)
    windows["window_end"] = windows.window_start + window_size
    windows["complete_window"] = windows.observed_ticks.eq(window_size)
    windows = add_rates(windows)
    window_summary = aggregate_metrics(windows, ["perception_radius", "window_start", "window_end"],
                                       (*COUNTERS, *RATE_NUMERATORS, "observed_ticks"))
    series = aggregate_metrics(data, ["perception_radius", "tick"],
                               ("living_population", "population_share", *TRAITS, *OPTIONAL))
    return summary, windows, window_summary, series


def plot_results(series, windows, output):
    plt, style = plotting()
    with style:
        def panel(ax, table, metric, x="tick", max_perception=8):
            for p, group in table[table.metric.eq(metric)].groupby("perception_radius", sort=True):
                if p > max_perception:
                    continue
                label = f"p{p}" if group["n"].gt(0).any() else f"p{p} (n=0)"
                mean_band(ax, group, x=x, label=label, color=f"C{p}", annotate_empty=False,
                          linewidth=1.7 if p <= 2 else 0.8, linestyle="-" if p <= 2 else "--")
            if not table.loc[table.metric.eq(metric) & table.perception_radius.le(max_perception), "n"].gt(0).any():
                ax.text(0.5, 0.5, "No finite observations", transform=ax.transAxes, ha="center")
            ax.set(xlabel="tick" if x == "tick" else "window end tick", ylabel=metric)
            ax.legend(ncol=5, fontsize=8)

        for metric, filename, title in (
            ("population_share", "perception_frequency_over_time.png", "Living cohort shares (equal seed weights)"),
            ("living_population", "perception_population_over_time.png", "Living cohort populations"),
            ("reproductive_participation_rate", "perception_reproductive_rate.png", "Successful parent participations / action opportunities"),
            ("death_rate", "perception_death_rate.png", "Deaths / action opportunities"),
            ("action_opportunities", "perception_exposure.png", "Recorded action exposure per window; inspect n in CSV"),
        ):
            by_window = metric in RATE_NUMERATORS or metric == "action_opportunities"
            fig, ax = plt.subplots(figsize=(11, 5))
            panel(ax, windows if by_window else series, metric, "window_end" if by_window else "tick")
            ax.set_title(title + "\nSeed mean ±1 sample SD (undefined for n < 2)")
            save_plot(plt, fig, output / filename)
        for max_p, filename in ((2, "perception_traits_over_time.png"), (8, "perception_traits_all_cohorts.png")):
            fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
            for ax, metric in zip(axes, TRAITS):
                panel(ax, series, metric, max_perception=max_p)
            fig.suptitle(f"Living cohort traits (p0–p{max_p}); empty cohorts excluded\nSparse cohorts: inspect exposure plot and per-metric n in CSV")
            save_plot(plt, fig, output / filename)


def analyze(data_dir=Path("data"), output_dir=Path("analysis/perception_cohorts"),
            pattern="evolving_seed_*_cohorts.csv", window_size=1000):
    files = seed_files(data_dir, pattern)
    data = load_cohorts(files)
    warn_grids(data)
    summary, windows, window_summary, series = build_tables(data, window_size)
    output = Path(output_dir)
    write_tables(output, cohort_summary=summary, cohort_window_rates=windows,
                 cohort_window_summary=window_summary, cohort_time_series=series)
    plot_results(series, window_summary, output)
    metadata(output, files, __file__, window_size=window_size,
             windows="(start, end]; sum counts / sum action opportunities within each seed, then equal-seed summaries",
             gaps="Missing ticks are not filled; partial window coverage is reported, not extrapolated",
             exposure="Living snapshots include newborns/exclude dead; action opportunities exclude newborns/include deaths",
             attribution="Child births, two successful parent participations per birth, REPRODUCE chooser initiations")
    print(f"Found {len(files)} cohort runs; seeds: {[seed for seed, _ in files]}")
    print(summary[["seed", "perception_radius", "action_opportunities", "births_as_child",
                   "successful_parent_participations", "reproductive_participation_rate", "death_rate"]].to_string(index=False))
    print(f"Output written to: {output}")
    return summary, windows, window_summary, series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/perception_cohorts"))
    parser.add_argument("--pattern", default="evolving_seed_*_cohorts.csv")
    parser.add_argument("--window-size", type=int, default=1000)
    args = parser.parse_args()
    try:
        analyze(args.data_dir, args.output_dir, args.pattern, args.window_size)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
