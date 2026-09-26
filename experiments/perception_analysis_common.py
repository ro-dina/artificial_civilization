"""Offline analysis helpers. No simulation imports or random draws."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import re
import warnings

import pandas as pd

if __package__:
    from .analyze_fixed_perception import aggregate_metrics
else:
    from analyze_fixed_perception import aggregate_metrics


def seed_files(directory, pattern):
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise ValueError(f"No input files: {directory}/{pattern}")
    found = {}
    for path in paths:
        match = re.search(r"(?:^|_)seed_(-?\d+)(?:_|\.)", path.name)
        if not match:
            raise ValueError(f"Cannot identify seed in {path.name}")
        seed = int(match[1])
        if seed in found:
            raise ValueError(f"Multiple inputs for seed {seed}: {found[seed].name}, {path.name}; narrow --pattern")
        found[seed] = path
    return sorted(found.items())


def numeric_csv(path, required, optional=(), keys=("tick",)):
    frame = pd.read_csv(path)
    missing = [name for name in required if name not in frame]
    if missing or frame.empty:
        raise ValueError(f"{path}: missing required columns {missing} or no data rows")
    absent = [name for name in optional if name not in frame]
    if absent:
        warnings.warn(f"{path.name}: optional columns unavailable: {', '.join(absent)}", stacklevel=2)
    for name in dict.fromkeys((*required, *optional)):
        if name not in frame:
            frame[name] = float("nan")
        original = frame[name]
        values = pd.to_numeric(original, errors="coerce")
        invalid = (original.notna() & values.isna()) | values.isin([float("inf"), -float("inf")])
        if invalid.any():
            warnings.warn(f"{path.name}: invalid {name} values treated as NaN", stacklevel=2)
        frame[name] = values.mask(invalid)
    for key in keys:
        values = frame[key]
        if (values.isna() | (values < 0) | (values % 1 != 0)).any():
            raise ValueError(f"{path}: {key} must be nonnegative integers")
        frame[key] = values.astype("int64")
    if frame.duplicated(list(keys)).any():
        raise ValueError(f"{path}: duplicate {keys}")
    return frame.sort_values(list(keys))


def check_seed(frame, seed, path):
    if "seed" in frame and not pd.to_numeric(frame.seed, errors="coerce").eq(seed).all():
        raise ValueError(f"{path}: seed disagrees with filename")
    frame["seed"] = seed


def write_tables(output, **tables):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, na_rep="", lineterminator="\n")


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt, plt.rc_context({**matplotlib.rcParamsDefault, "backend": "Agg", "font.family": "DejaVu Sans",
                                "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save_plot(plt, fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", metadata={"Software": "perception observation analysis"})
    plt.close(fig)


def mean_band(ax, rows, x="tick", label="Mean ±1 sample SD", *, annotate_empty=True, **kwargs):
    rows = rows.sort_values(x)
    line, = ax.plot(rows[x], rows["mean"], label=label, **kwargs)
    ax.fill_between(rows[x], rows["mean"] - rows["std"], rows["mean"] + rows["std"],
                    color=line.get_color(), alpha=0.13, linewidth=0)
    ax.grid(alpha=0.2)
    if annotate_empty and not rows["mean"].notna().any():
        ax.text(0.5, 0.5, "No finite observations", transform=ax.transAxes, ha="center")


def metadata(output, files, script, **details):
    import matplotlib
    result = {"python": platform.python_version(), "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
              "script_sha256": hashlib.sha256(Path(script).read_bytes()).hexdigest(),
              "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "aggregation_helper_sha256": hashlib.sha256(Path(__file__).with_name("analyze_fixed_perception.py").read_bytes()).hexdigest(),
              "inputs": [{"seed": seed, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for seed, path in files],
              "aggregation": "Equal run weights; sample SD ddof=1; per-metric finite n; no inference of causality",
              "missing": "No interpolation or carry-forward; undefined means/rates are empty CSV cells", **details}
    (Path(output) / "analysis_metadata.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def warn_grids(data):
    grids = data.groupby("seed", sort=True).tick.apply(lambda s: tuple(sorted(s.unique())))
    if any(grid != grids.iloc[0] for grid in grids.iloc[1:]):
        warnings.warn("Tick grids differ; only recorded ticks contribute. Inspect n and final_tick.", stacklevel=2)
