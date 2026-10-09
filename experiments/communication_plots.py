"""Headless, deterministic descriptive plots; no simulation or random jitter."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

CONDITION_NAMES = {"A": "A: listening OFF / vocal learning OFF (random emission)",
                   "B": "B: listening ON / vocal learning OFF (random emission)",
                   "C": "C: listening OFF / vocal learning ON",
                   "D": "D: listening ON / vocal learning ON", "random": "RandomBrain"}


def select(summary, table, metric, **filters):
    rows = summary.loc[summary.table.eq(table) & summary.metric.eq(metric)]
    for key, value in filters.items():
        rows = rows.loc[rows[key].eq(value)]
    return rows.sort_values("actual_window_end")


def line(ax, rows, label, *, color=None, style="-", percent=False):
    x, mean, sd = (rows[name].to_numpy(dtype=float) for name in ("actual_window_end", "mean", "std"))
    counts = rows.loc[rows["n"].gt(0), "n"]
    suffix = f" (n={int(counts.min())}–{int(counts.max())})" if len(counts) else " (n=0)"
    drawn, = ax.plot(x, mean, style, marker=".", label=label + suffix, color=color, linewidth=1.5)
    low, high = mean - sd, mean + sd
    if percent:
        low, high = np.maximum(low, 0), np.minimum(high, 1)
    ax.fill_between(x, low, high, color=drawn.get_color(), alpha=0.12, linewidth=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.grid(alpha=0.2)
    if percent:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))


def seed_probabilities(rows, keys, count="count", denominator="n"):
    groups = ["condition", "seed", *keys]
    grouped = rows.groupby(groups, sort=True)[[count, denominator]].sum().reset_index()
    grouped["probability"] = grouped[count] / grouped[denominator].where(grouped[denominator] > 0)
    return grouped


def collapse(rows, keys, field, keep):
    rows = rows.copy()
    rows[field] = rows[field].where(rows[field].isin(keep), "OTHER")
    return rows.groupby(["condition", "seed", *keys, field], sort=True).probability.sum(min_count=1).reset_index()


def probability_heatmap(ax, rows, row_names, column_names, *, row_field, column_field, title):
    if rows.empty:
        matrix = np.full((len(row_names), len(column_names)), np.nan)
    else:
        means = rows.groupby([row_field, column_field], sort=True).probability.mean()
        matrix = np.array([[means.get((r, c), np.nan) for c in column_names] for r in row_names])
    plot = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(column_names)), [str(c).replace("SIGNAL_", "") for c in column_names], rotation=60, ha="right")
    ax.set_yticks(range(len(row_names)), [str(r) for r in row_names])
    ax.set_title(title, fontsize=10)
    if not np.isfinite(matrix).any():
        ax.text(0.5, 0.5, "No eligible exposure", transform=ax.transAxes, ha="center", color="gray")
    return plot


def plots(tables, summary, references, directory, top_signals):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    names = []
    conditions = sorted(tables["communication_run_summary"].condition.unique())
    colors = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(conditions)}
    width = max(1, len(conditions))

    def save(fig, name):
        if not fig.get_constrained_layout():
            fig.tight_layout()
        descriptions = [CONDITION_NAMES.get(c.split(":")[0], c) for c in conditions]
        # Keep experimental definitions on exported figures, below the axes.
        fig.text(0.5, -0.025, " | ".join(descriptions[:2]) +
                 ("\n" + " | ".join(descriptions[2:]) if len(descriptions) > 2 else ""),
                 ha="center", va="top", fontsize=8)
        fig.savefig(Path(directory) / name, dpi=180, bbox_inches="tight", metadata={"Software": "communication analysis v0.6.1"})
        plt.close(fig)
        names.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(axes, ("emission_rate", "silence_rate"), ("Emission", "SILENCE selection")):
        for c in conditions:
            line(ax, select(summary, "silence_over_time", metric, condition=c), c, color=colors[c], percent=True)
        ax.set(title=title + " rate — equal-seed mean ±1 SD", xlabel="Actual window-end tick", ylabel="Vocal decisions (%)")
        ax.legend(fontsize=8)
    save(fig, "vocal_activity_over_time.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(axes, ("silence_percept_rate", "masked_rate", "identified_rate"), ("SILENCE", "MASKED", "IDENTIFIED")):
        for c in conditions:
            line(ax, select(summary, "masking_over_time", metric, condition=c), c, color=colors[c], percent=True)
        ax.set(title=title + " — equal-seed mean ±1 SD", xlabel="Actual window-end tick", ylabel="Receiver observations (%)")
        ax.legend(fontsize=8)
    save(fig, "auditory_outcomes_over_time.png")

    frequency = tables["vocal_frequency_over_time"]
    emitted = frequency.loc[frequency.vocal_action.ne("SILENCE")]
    per = emitted.groupby(["condition", "seed", "vocal_action"], sort=True)["count"].sum().reset_index()
    denom = per.groupby(["condition", "seed"])["count"].transform("sum")
    per["probability"] = per["count"] / denom.where(denom > 0)
    ranking = per.groupby("vocal_action").probability.mean().reset_index()
    ranking["id"] = ranking.vocal_action.str.removeprefix("SIGNAL_").astype(int)
    top = ranking.sort_values(["probability", "id"], ascending=[False, True], na_position="last").head(top_signals).vocal_action.tolist()
    top = sorted(top, key=lambda label: int(label.removeprefix("SIGNAL_")))
    visible = top + (["OTHER"] if len(ranking) > len(top) or not top else [])
    per = collapse(per, [], "vocal_action", top)
    fig, ax = plt.subplots(figsize=(max(9, len(visible) * 0.65), 4.5))
    bar_width = 0.8 / width
    for i, c in enumerate(conditions):
        rows = per.loc[per.condition.eq(c)].groupby("vocal_action").probability.agg(["mean", "std", "count"]).reindex(visible)
        positions = np.arange(len(visible)) + (i - (width - 1)/2)*bar_width
        ax.bar(positions, rows["mean"], width=bar_width, color=colors[c], label=c)
        finite = rows["std"].notna()
        ax.errorbar(positions[finite], rows.loc[finite, "mean"], yerr=rows.loc[finite, "std"],
                    fmt="none", color="black", capsize=2)
    ax.set_xticks(range(len(visible)), [v.replace("SIGNAL_", "") for v in visible])
    ax.set(title="Emitted arbitrary IDs — equal-run proportions ±1 seed SD (SILENCE excluded)",
           xlabel="Signal ID; OTHER combines remaining IDs only for display", ylabel="Emissions (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    save(fig, "signal_frequency.png")

    features = ("hunger_bin", "thirst_bin", "energy_bin", "food_direction", "water_direction")
    fig, axes = plt.subplots(5, width, figsize=(3.8*width, 12.5), squeeze=False)
    for i, feature in enumerate(features):
        for j, c in enumerate(conditions):
            ax = axes[i, j]
            for metric, conditioning, label, style in (("mi_bits", "population_pooled", "Pooled observed", ":"),
                 ("corrected_mi_bits", "population_pooled", "Pooled minus null", "-"),
                 ("corrected_mi_bits", "within_agent_eligible", "Within-agent minus null", "--")):
                rows = select(summary, "sender_information", metric, condition=c, feature=feature,
                              scope="vocal_including_silence", conditioning=conditioning)
                line(ax, rows, label, style=style)
            ax.set(title=f"{c}: {feature}", xlabel="Actual window-end tick", ylabel="Empirical information (bits)")
            ax.axhline(0, color="gray", linewidth=0.5)
            if i == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Sender V includes SILENCE; corrected MI is descriptive, not signal meaning", y=1.01)
    save(fig, "sender_information_over_time.png")

    fig, axes = plt.subplots(2, width, figsize=(4.5*width, 6.5), squeeze=False, layout="constrained")
    conditional = tables["conditional_vocal"]
    for i, feature in enumerate(("hunger_bin", "thirst_bin")):
        rows = conditional.loc[conditional.feature.eq(feature) & conditional.scope.eq("vocal_including_silence")]
        probabilities = seed_probabilities(rows, ["category", "vocal_action"])
        probabilities = collapse(probabilities, ["category"], "vocal_action", ["SILENCE", *top])
        categories = sorted(rows.category.unique(), key=lambda value: int(value)) or ["No data"]
        for j, c in enumerate(conditions):
            image = probability_heatmap(axes[i,j], probabilities.loc[probabilities.condition.eq(c)], categories,
                ["SILENCE", *visible], row_field="category", column_field="vocal_action", title=f"{c}: {feature}")
            axes[i,j].set(xlabel="SILENCE / arbitrary IDs", ylabel="Primitive need bin")
    fig.suptitle("State-conditioned vocal proportions — equal-run means")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Within-state proportion", fraction=0.015, pad=0.025)
    save(fig, "vocal_by_need_state.png")

    fig, axes = plt.subplots(1, width, figsize=(4.5*width, 7), squeeze=False, layout="constrained")
    images = []
    for j, c in enumerate(conditions):
        ax = axes[0,j]
        reference = references.get(c)
        if reference is None:
            ax.text(0.5, 0.5, "No sufficiently sampled agents", ha="center", transform=ax.transAxes)
            ax.set_title(c)
            ax.set(xlabel="Arbitrary emitted ID", ylabel="Agent ID (analysis only)")
            continue
        agents = sorted(reference["agents"], key=lambda a: a[0])
        rows = []
        for agent, _, counts in agents:
            for column in visible:
                index = int(column.removeprefix("SIGNAL_")) if column != "OTHER" else None
                count = counts[index] if index is not None and index < len(counts) else 0
                if column == "OTHER":
                    count = counts.sum() - sum(counts[int(label.removeprefix("SIGNAL_"))] for label in top
                                               if int(label.removeprefix("SIGNAL_")) < len(counts))
                rows.append({"agent": agent, "signal": column, "probability": count / counts.sum()})
        image = probability_heatmap(ax, pd.DataFrame(rows), [a[0] for a in agents], visible,
                    row_field="agent", column_field="signal",
                    title=f"{c}: seed {reference['seed']}, ({reference['window_start']},{reference['window_end']}]")
        ax.set(xlabel="Arbitrary emitted ID (SILENCE excluded)", ylabel="Agent ID (analysis only)")
        images.append(image)
    fig.suptitle("Individual frequencies; exposure-ranked subset, not shared meanings")
    if images:
        fig.colorbar(images[0], ax=axes.ravel().tolist(), label="Within-agent emitted proportion", fraction=0.02, pad=0.025)
    save(fig, "individual_vocal_preferences.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for c in conditions:
        line(axes[0], select(summary, "population_convergence", "mean_jsd_bits", condition=c), c, color=colors[c])
        line(axes[0], select(summary, "population_convergence", "null_mean_jsd_bits", condition=c), c + " frequency null", color=colors[c], style="--")
        line(axes[1], select(summary, "population_convergence", "observed_minus_null_bits", condition=c), c, color=colors[c])
    axes[0].set(title="Individual distributions — low JSD is similarity, not convention", ylabel="Mean pairwise JSD (bits)", ylim=(0, 1))
    axes[1].set(title="Observed mean JSD minus frequency-matched null", ylabel="JSD difference (bits)")
    for ax in axes:
        ax.set_xlabel("Actual window-end tick")
        ax.legend(fontsize=7)
    save(fig, "population_convergence_over_time.png")

    fig, axes = plt.subplots(1, width, figsize=(4.7*width, 4.7), squeeze=False, layout="constrained")
    rows = tables["receiver_action_distribution"]
    rows = rows.loc[rows.feature.eq("auditory_kind")]
    probabilities = seed_probabilities(rows, ["category", "physical_action"])
    actions = ["move_north", "move_south", "move_east", "move_west", "eat", "drink", "wait", "reproduce"]
    for j, c in enumerate(conditions):
        ax = axes[0,j]
        image = probability_heatmap(ax, probabilities.loc[probabilities.condition.eq(c)], ["SILENCE", "MASKED", "IDENTIFIED"],
                           actions, row_field="category", column_field="physical_action", title=c)
        ax.set(xlabel="Same-tick physical action", ylabel="Actual resolved auditory kind")
    fig.suptitle("Receiver conditional action distributions — association only; equal-run means")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Within-percept action proportion", fraction=0.02, pad=0.025)
    save(fig, "receiver_action_by_auditory.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for c in conditions:
        emissions = select(summary, "silence_over_time", "emission_rate", condition=c)
        masked = select(summary, "masking_over_time", "masked_rate", condition=c)
        paired = emissions.merge(masked, on=["window_start", "window_end", "actual_window_end"], suffixes=("_emit", "_mask"))
        ax.scatter(paired.mean_emit, paired.mean_mask, color=colors[c], label=c, s=30)
    ax.set(title="Acoustic activity and masking — matching-window equal-seed means",
           xlabel="Emission rate per vocal decision (%)", ylabel="MASKED rate per receiver observation (%)", xlim=(0, 1), ylim=(0, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "masking_vs_emission.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    states = tables["silence_by_state"]
    for ax, feature in zip(axes, ("hunger_bin", "thirst_bin", "auditory_kind")):
        source = states.loc[states.feature.eq(feature)]
        probabilities = seed_probabilities(source, ["category"], count="silent")
        categories = sorted(source.category.unique())
        for i, c in enumerate(conditions):
            rows = probabilities.loc[probabilities.condition.eq(c)].groupby("category").probability.agg(["mean", "std"]).reindex(categories)
            positions = np.arange(len(categories)) + (i-(width-1)/2)*0.1
            ax.plot(positions, rows["mean"], "o", markersize=4, color=colors[c], label=c)
            finite = rows["std"].notna()
            ax.errorbar(positions[finite], rows.loc[finite, "mean"], yerr=rows.loc[finite, "std"],
                        fmt="none", color=colors[c], capsize=2)
        ax.set_xticks(range(len(categories)), categories, rotation=20)
        ax.set(title=feature, xlabel="Primitive category", ylabel="SILENCE selection (%)", ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("P(SILENCE | state) — equal-run means ±1 seed SD; no silence reward", y=1.02)
    save(fig, "silence_by_state.png")
    return names
