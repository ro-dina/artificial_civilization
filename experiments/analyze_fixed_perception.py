"""Post-process fixed-perception CSVs; never import or run the simulation.

Each run has equal weight. Only observed ticks contribute; missing observations
and undefined living-population means are not interpolated or carried forward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import warnings

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

FILE_PATTERN = "learning_fixed_p*_seed_*.csv"
FILE_RE = re.compile(r"learning_fixed_p(?P<perception>\d+)_seed_(?P<seed>-?\d+)\.csv")
TIME_METRICS = ("population", "mean_memory_size", "mean_learning_updates")
TRAIT_METRICS = ("mean_hunger_multiplier", "mean_thirst_multiplier", "mean_signal_probability")
DEATH_METRICS = ("starvation_deaths", "dehydration_deaths", "energy_deaths")
FINAL_METRICS = (
    "population", "births", "deaths", "highest_generation", "food_consumed", "water_consumed",
    *DEATH_METRICS, "old_age_deaths", *TRAIT_METRICS,
    "mean_memory_size", "mean_learning_updates", "total_learning_updates",
    "exploratory_actions", "exploitative_actions",
)
INSTALL_HELP = "Install analysis dependencies: python3 -m pip install -r requirements-analysis.txt"


def parse_filename(path: Path) -> tuple[int, int]:
    match = FILE_RE.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"{path}: expected learning_fixed_p<RADIUS>_seed_<SEED>.csv")
    return int(match["perception"]), int(match["seed"])


def discover_runs(data_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(Path(data_dir).glob(FILE_PATTERN)):
        if not path.is_file():
            continue
        try:
            parse_filename(path)
        except ValueError as error:
            warnings.warn(str(error), stacklevel=2)
        else:
            paths.append(path)
    if not paths:
        raise ValueError(f"No matching runs in {data_dir}: {FILE_PATTERN}")
    return sorted(paths, key=lambda p: (*parse_filename(p), p.name))


def load_runs(paths: list[Path]) -> pd.DataFrame:
    """Validate run identities/ticks; retain NaNs without substituting other metrics."""
    frames = []
    identities = set()
    for path in sorted(paths, key=lambda p: (*parse_filename(p), p.name)):
        perception, seed = parse_filename(path)
        if (perception, seed) in identities:
            raise ValueError(f"Duplicate run identity: perception={perception}, seed={seed} ({path})")
        identities.add((perception, seed))
        frame = pd.read_csv(path)
        required = [name for name in ("tick", *TIME_METRICS) if name not in frame]
        if required:
            raise ValueError(f"{path}: missing required columns: {', '.join(required)}; no aliases are substituted")
        if frame.empty:
            raise ValueError(f"{path}: no data rows")
        ticks = pd.to_numeric(frame["tick"], errors="coerce")
        if (ticks.isna() | ~ticks.between(0, 2**63 - 1) | (ticks % 1 != 0)).any():
            raise ValueError(f"{path}: tick must contain finite, nonnegative integers without missing values")
        frame["tick"] = ticks.astype("int64")
        if frame["tick"].duplicated().any():
            raise ValueError(f"{path}: duplicate ticks would count a seed more than once")
        if "seed" in frame and not pd.to_numeric(frame["seed"], errors="coerce").eq(seed).all():
            raise ValueError(f"{path}: CSV seed column disagrees with filename seed={seed}")
        missing = [name for name in FINAL_METRICS if name not in frame]
        if missing:
            warnings.warn(f"{path.name}: missing optional metrics: {', '.join(missing)}; recorded as unavailable",
                          stacklevel=2)
        for name in FINAL_METRICS:
            if name not in frame:
                frame[name] = float("nan")
                continue
            numeric = pd.to_numeric(frame[name], errors="coerce")
            invalid = (frame[name].notna() & numeric.isna()) | numeric.isin([math.inf, -math.inf])
            if invalid.any():
                warnings.warn(f"{path.name}: {int(invalid.sum())} invalid {name} values treated as NaN", stacklevel=2)
            frame[name] = numeric.mask(invalid)
        if frame["population"].dropna().lt(0).any():
            raise ValueError(f"{path}: population must not be negative")
        living_means = [*TIME_METRICS[1:], *TRAIT_METRICS]
        extinct = frame["population"].eq(0)
        if frame.loc[extinct, living_means].notna().any().any():
            warnings.warn(f"{path.name}: living-population means at population=0 treated as undefined", stacklevel=2)
        frame.loc[extinct, living_means] = float("nan")
        frame = frame[["tick", *FINAL_METRICS]].sort_values("tick").copy()
        frame.insert(0, "source_file", path.name)
        frame.insert(0, "seed", seed)
        frame.insert(0, "perception", perception)
        frames.append(frame)
    if not frames:
        raise ValueError("No runs to analyze")
    return pd.concat(frames, ignore_index=True)


def summarize_runs(data: pd.DataFrame) -> pd.DataFrame:
    """Select the actual maximum-tick row, including its NaNs (never groupby.last)."""
    ordered = data.sort_values(["perception", "seed", "tick"])
    final = ordered.groupby(["perception", "seed"], sort=True).tail(1).copy()
    return final.rename(columns={"tick": "final_tick", "population": "final_population"}).reset_index(drop=True)


def aggregate_metrics(data: pd.DataFrame, keys: list[str], metrics: tuple[str, ...]) -> pd.DataFrame:
    summaries = []
    grouped = data.groupby(keys, sort=True)
    for metric in metrics:
        values = grouped[metric]
        summary = pd.DataFrame({"mean": values.mean(), "std": values.std(ddof=1),
                                "min": values.min(), "max": values.max(), "n": values.count()}).reset_index()
        summary.insert(len(keys), "metric", metric)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True).sort_values([*keys, "metric"]).reset_index(drop=True)


def build_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = summarize_runs(data)
    final = aggregate_metrics(runs.rename(columns={"final_population": "population"}),
                              ["perception"], FINAL_METRICS)
    series = aggregate_metrics(data, ["perception", "tick"], (*TIME_METRICS, *TRAIT_METRICS))
    return runs, final, series


def plot_results(runs: pd.DataFrame, final: pd.DataFrame, series: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    perceptions = sorted(runs["perception"].unique())
    colors = {p: plt.get_cmap("tab10")(i % 10) for i, p in enumerate(perceptions)}
    ticks = sorted(series["tick"].unique())

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=180, bbox_inches="tight", metadata={"Software": "fixed-perception analysis"})
        plt.close(fig)

    def time_axes(ax, metric, title):
        for perception in perceptions:
            rows = series.loc[(series["perception"] == perception) & (series["metric"] == metric)]
            rows = rows.set_index("tick").reindex(ticks)
            mean, std = rows["mean"].to_numpy(dtype=float), rows["std"].to_numpy(dtype=float)
            ax.plot(ticks, mean, color=colors[perception], label=f"perception = {perception}",
                    linewidth=1.4, marker="o" if len(ticks) == 1 else None)
            ax.fill_between(ticks, mean - std, mean + std, color=colors[perception], alpha=0.16, linewidth=0)
        if not series.loc[series["metric"] == metric, "mean"].notna().any():
            ax.text(0.5, 0.5, "No finite observations", transform=ax.transAxes, ha="center")
        ax.set(title=title, xlabel="tick", ylabel=metric)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)

    with plt.rc_context({**matplotlib.rcParamsDefault, "backend": "Agg", "font.family": "DejaVu Sans",
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        for metric, title, filename in (
            ("population", "Population over time", "population_over_time.png"),
            ("mean_memory_size", "Living learners' mean memory size over time", "memory_size_over_time.png"),
            ("mean_learning_updates", "Living learners' mean learning updates over time", "learning_updates_over_time.png"),
        ):
            fig, ax = plt.subplots(figsize=(10, 5.5))
            time_axes(ax, metric, title + "\nAcross runs: mean ±1 sample SD")
            save(fig, filename)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        labels = []
        for index, perception in enumerate(perceptions):
            values = runs.loc[runs["perception"] == perception].sort_values("seed")["final_population"].dropna()
            offsets = [0 if len(values) == 1 else 0.18 * (2 * i / (len(values) - 1) - 1)
                       for i in range(len(values))]
            ax.scatter([index + offset for offset in offsets], values, color=colors[perception], alpha=0.75, s=30)
            row = final.loc[(final["perception"] == perception) & (final["metric"] == "population")].iloc[0]
            if pd.notna(row["mean"]):
                ax.plot(index, row["mean"], "D", color="black", markersize=7)
                if pd.notna(row["std"]):
                    ax.errorbar(index, row["mean"], yerr=row["std"], fmt="none", color="black", capsize=5)
            labels.append(f"perception = {perception}\nn = {len(values)}")
        ax.set(xticks=range(len(perceptions)), xticklabels=labels, ylabel="final population",
               title="Final population\nEach run's last recorded tick")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", color="0.5", label="Individual runs"),
                           Line2D([], [], marker="D", linestyle="none", color="black", label="Mean ±1 sample SD")])
        save(fig, "final_population.png")

        fig, ax = plt.subplots(figsize=(10, 5.5))
        for index, metric in enumerate(DEATH_METRICS):
            values = final.loc[final["metric"] == metric].set_index("perception").reindex(perceptions)
            ax.bar([x + (index - 1) * 0.25 for x in range(len(perceptions))], values["mean"],
                   width=0.25, label=metric.removesuffix("_deaths"))
        if not final.loc[final["metric"].isin(DEATH_METRICS), "mean"].notna().any():
            ax.text(0.5, 0.5, "No finite observations", transform=ax.transAxes, ha="center")
        ax.set(xticks=range(len(perceptions)), xticklabels=[f"perception = {p}" for p in perceptions],
               ylabel="mean cumulative deaths per run", title="Selected death causes at each run's final tick")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        save(fig, "death_causes_final.png")

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        for ax, metric in zip(axes, TRAIT_METRICS):
            time_axes(ax, metric, metric + "\nAcross runs: mean ±1 sample SD")
        save(fig, "traits_over_time.png")


def analyze(data_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if pd is None:
        raise ValueError(INSTALL_HELP)
    try:
        import matplotlib
    except ModuleNotFoundError as error:
        raise ValueError(INSTALL_HELP) from error
    paths = discover_runs(data_dir)
    data = load_runs(paths)
    runs, final, series = build_tables(data)
    grids = data.groupby(["perception", "seed"], sort=True)["tick"].apply(tuple)
    if any(grid != grids.iloc[0] for grid in grids.iloc[1:]):
        warnings.warn("Tick grids differ: aggregate observed ticks only, without interpolation or carry-forward; inspect n",
                      stacklevel=2)
    if runs["final_tick"].nunique() > 1:
        warnings.warn("Final ticks differ: final metrics compare each run's own last recorded tick, not a common horizon",
                      stacklevel=2)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for table, name in ((runs, "run_summary.csv"), (final, "final_metrics.csv"), (series, "time_series_metrics.csv")):
        table.to_csv(output_dir / name, index=False, na_rep="", lineterminator="\n")
    plot_results(runs, final, series, output_dir)
    metadata = {
        "python": platform.python_version(), "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "aggregation": "Equal weight per run; sample SD (ddof=1); n counts finite values separately per metric",
        "missing_ticks": "Observed ticks only, no interpolation or carry-forward, including after extinction",
        "final_rows": "Maximum recorded tick per run, preserving NaNs",
        "extinction": "Population zero is included; living-population means are undefined and excluded",
        "nan_output": "Empty CSV cells; SD undefined for n < 2; no SD band/error bar in that case",
        "inputs": [{"file": p.name, "perception": parse_filename(p)[0], "seed": parse_filename(p)[1],
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Found {len(runs)} runs")
    for perception, group in runs.groupby("perception", sort=True):
        print(f"perception={perception}: {len(group)} seeds ({', '.join(map(str, group['seed']))})")
    print(f"Final ticks: {runs['final_tick'].min()} to {runs['final_tick'].max()}")
    for metric, title in (("population", "Final population"), ("mean_memory_size", "Final memory size"),
                          ("mean_learning_updates", "Final mean learning updates")):
        print(f"\n{title}:")
        for row in final.loc[final["metric"] == metric].itertuples():
            print(f"p{row.perception}: mean={row.mean:.6g}, std={row.std:.6g}, n={row.n}")
    print(f"\nOutput written to:\n{output_dir}")
    return runs, final, series


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fixed-perception experiment CSVs (no simulation execution)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/fixed_perception"))
    args = parser.parse_args()
    try:
        analyze(args.data_dir, args.output_dir)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
