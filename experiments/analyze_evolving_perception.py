"""Analyze existing free-evolution LearningBrain CSVs; no simulation execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

if __package__:
    from .perception_analysis_common import (pd, aggregate_metrics, seed_files, numeric_csv, check_seed,
        write_tables, plotting, save_plot, mean_band, metadata, warn_grids)
else:
    from perception_analysis_common import (pd, aggregate_metrics, seed_files, numeric_csv, check_seed,
        write_tables, plotting, save_plot, mean_band, metadata, warn_grids)

REQUIRED = ("tick", "population", "mean_perception_radius")
OPTIONAL = ("min_perception_radius", "max_perception_radius", "mean_hunger_multiplier", "mean_thirst_multiplier",
            "mean_signal_probability", "mean_memory_size", "mean_learning_updates", "births", "deaths",
            "starvation_deaths", "dehydration_deaths", "old_age_deaths", "energy_deaths", "highest_generation")
METRICS = (*REQUIRED[1:], *OPTIONAL)


def load_evolving(files):
    frames = []
    for seed, path in files:
        frame = numeric_csv(path, REQUIRED, OPTIONAL)
        check_seed(frame, seed, path)
        if "brain" in frame and not frame.brain.eq("learning").all():
            raise ValueError(f"{path}: expected LearningBrain runs")
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        if sidecar.exists() and json.loads(sidecar.read_text()).get("config", {}).get("fixed_perception_radius") is not None:
            raise ValueError(f"{path}: fixed-perception experiment is not free evolution")
        means = [name for name in METRICS if name.startswith(("mean_", "min_perception", "max_perception"))]
        frame.loc[frame.population.eq(0), means] = float("nan")
        frame["source_file"] = path.name
        frames.append(frame[["seed", "source_file", "tick", *METRICS]])
    return pd.concat(frames, ignore_index=True).sort_values(["seed", "tick"]).reset_index(drop=True)


def analyze(data_dir=Path("data"), output_dir=Path("analysis/evolving_perception"),
            pattern="v03_learning_seed_*_5000.csv"):
    files = seed_files(data_dir, pattern)
    data = load_evolving(files)
    warn_grids(data)
    series = aggregate_metrics(data, ["tick"], METRICS)
    runs = data.groupby("seed", sort=True).tail(1).rename(columns={"tick": "final_tick"})
    if runs.final_tick.nunique() > 1:
        warnings.warn("Final ticks differ; final metrics use each run's own last row, not a common horizon", stacklevel=2)
    final = aggregate_metrics(runs.assign(experiment="free_learning"), ["experiment"], METRICS).drop(columns="experiment")
    output = Path(output_dir)
    write_tables(output, evolving_time_series=series, evolving_final_metrics=final, evolving_run_summary=runs)
    plt, style = plotting()
    with style:
        for names, filename, title in (
            (("mean_perception_radius",), "perception_over_time.png", "Free-evolution perception"),
            (("population",), "population_over_time.png", "Free-evolution population"),
            (("mean_memory_size", "mean_learning_updates"), "learning_over_time.png", "Living learners"),
            (("mean_hunger_multiplier", "mean_thirst_multiplier", "mean_signal_probability"), "traits_over_time.png", "Living genome traits"),
        ):
            fig, axes = plt.subplots(len(names), 1, figsize=(10, 4 * len(names)), squeeze=False)
            for ax, name in zip(axes.flat, names):
                mean_band(ax, series[series.metric.eq(name)])
                ax.set(xlabel="tick", ylabel=name, title=f"{title}: {name}")
                ax.legend()
            save_plot(plt, fig, output / filename)
        columns = min(5, len(files))
        rows = (len(files) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3 * rows), squeeze=False, sharex=True, sharey=True)
        for ax, (seed, group) in zip(axes.flat, data.groupby("seed", sort=True)):
            ax.plot(group.tick, group.mean_perception_radius, linewidth=1)
            ax.set(title=f"seed {seed}", xlabel="tick", ylabel="mean perception")
            ax.grid(alpha=0.2)
        for ax in list(axes.flat)[len(files):]:
            ax.set_visible(False)
        save_plot(plt, fig, output / "perception_by_seed.png")
        means = series.pivot(index="tick", columns="metric", values="mean")
        fig, ax = plt.subplots(figsize=(8, 5))
        points = ax.scatter(means.mean_perception_radius, means.population, c=means.index, cmap="viridis", s=6, alpha=0.65)
        fig.colorbar(points, ax=ax, label="tick")
        ax.set(xlabel="mean perception radius across runs", ylabel="mean population across runs",
               title="Perception and population at matching ticks\nDescriptive time trajectory")
        save_plot(plt, fig, output / "perception_population_relationship.png")
    metadata(output, files, __file__, required_columns=REQUIRED, optional_columns=OPTIONAL,
             final_rows="Maximum recorded tick per seed, preserving NaNs")
    print(f"Found {len(files)} free-evolution runs; seeds: {[seed for seed, _ in files]}")
    print(series[series.metric.eq("mean_perception_radius") & series.tick.isin([0,500,1000,2000,3000,4000,5000])].to_string(index=False))
    print(f"Output written to: {output}")
    return data, series, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/evolving_perception"))
    parser.add_argument("--pattern", default="v03_learning_seed_*_5000.csv")
    args = parser.parse_args()
    try:
        analyze(args.data_dir, args.output_dir, args.pattern)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
