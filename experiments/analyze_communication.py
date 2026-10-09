"""Offline, seed-separated communication analysis; never starts a simulation."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import pandas as pd

if __package__:
    from .communication_data import FEATURES, discover, file_hash, read_windows
    from .communication_metrics import analysis_rng, convergence, entropy, information_report, jensen_shannon
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from communication_data import FEATURES, discover, file_hash, read_windows
    from communication_metrics import analysis_rng, convergence, entropy, information_report, jensen_shannon

COMMON = ("run", "condition", "seed", "window_start", "window_end", "actual_window_end",
          "complete_window", "partial_window", "observed_ticks")
INFO = ("n", "mi_bits", "h_v_bits", "h_x_bits", "nmi", "permuted_mean_bits", "permuted_std_bits", "corrected_mi_bits")
SCHEMAS = {
    "vocal_frequency_over_time": (*COMMON, "vocal_action", "count", "n_decisions", "n_emitted", "probability", "prob_given_emitted"),
    "silence_over_time": (*COMMON, "n_decisions", "emitted", "silent", "silence_rate", "emission_rate"),
    "masking_over_time": (*COMMON, "n_observations", "silent_percepts", "masked", "identified", "silence_percept_rate",
                         "masked_rate", "identified_rate", "emitted", "emitted_per_recorded_tick", "identified_per_emitted"),
    "conditional_vocal": (*COMMON, "feature", "category", "scope", "vocal_action", "count", "n", "probability"),
    "silence_by_state": (*COMMON, "feature", "category", "silent", "n", "silence_rate"),
    "sender_information": (*COMMON, "feature", "scope", "conditioning", "min_samples", "eligible_agents", "strata", *INFO),
    "sender_information_by_agent": (*COMMON, "agent_id", "feature", "scope", "min_samples", *INFO),
    "vocal_preferences_by_agent": (*COMMON, "agent_id", "n_decisions", "n_emitted", "silence_rate", "most_frequent_signal",
                                   "most_frequent_signal_share", "vocal_entropy_bits", "emitted_entropy_bits",
                                   "eligible_decisions", "eligible_emitted"),
    "vocal_frequency_by_agent": (*COMMON, "agent_id", "signal_id", "count", "n_emitted", "n_decisions", "prob_given_emitted"),
    "population_convergence": (*COMMON, "eligible_agents", "pairs", "possible_pairs", "n_emitted", "pair_method",
                               "min_emitted_samples", "mean_jsd_bits", "median_jsd_bits", "null_mean_jsd_bits",
                               "null_std_jsd_bits", "observed_minus_null_bits"),
    "receiver_action_distribution": (*COMMON, "feature", "category", "physical_action", "count", "n", "probability"),
    "receiver_responsiveness": (*COMMON, "feature", "conditioning", "min_samples", "strata", *INFO),
    "vocal_preference_changes": ("run", "condition", "seed", "agent_id", "first_window_start", "first_window_end",
                                 "last_window_start", "last_window_end", "first_emitted", "last_emitted",
                                 "eligible_windows", "early_late_jsd_bits"),
    "communication_run_summary": ("run", "condition", "seed", "final_tick", "final_population", "births", "deaths",
                                   "n_decisions", "emitted", "silent", "silence_rate", "emission_rate", "masked_rate",
                                   "identified_rate", "silence_percept_rate", "vocal_entropy_bits", "emitted_entropy_bits",
                                   "windows", "complete_windows", "input_bytes"),
}
PLOT_TABLES = set(SCHEMAS) - {"sender_information_by_agent", "vocal_preferences_by_agent", "vocal_frequency_by_agent",
                            "vocal_preference_changes"}


@dataclass(frozen=True)
class Options:
    window_size: int = 100
    permutations: int = 100
    analysis_seed: int = 0
    min_agent_decisions: int = 50
    min_agent_emissions: int = 10
    min_receiver_stratum: int = 20
    max_pairs: int = 1000
    top_signals: int = 12

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or (name != "analysis_seed" and value < 1):
                raise ValueError(f"{name} must be {'an integer' if name == 'analysis_seed' else 'a positive integer'}")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


class Outputs:
    def __init__(self, directory, stack):
        self.plot_rows = {name: [] for name in PLOT_TABLES}
        self.writers = {}
        for name, schema in SCHEMAS.items():
            writer = csv.DictWriter(stack.enter_context((directory / (name + ".csv")).open("w", newline="", encoding="utf-8")),
                                    fieldnames=schema)
            writer.writeheader()
            self.writers[name] = writer

    def add(self, name, row):
        self.writers[name].writerow(row)
        if name in self.plot_rows:
            self.plot_rows[name].append(row)


def features(frame):
    values = {name: frame[name].tolist() for name in FEATURES[:3]}
    for name in ("food", "water"):
        values[name + "_direction"] = list(zip(frame[name + "_dx"], frame[name + "_dy"]))
    return values


def labels(frame):
    return [kind if kind != "IDENTIFIED" else f"ID_{int(signal)}:{direction}"
            for kind, signal, direction in zip(frame.auditory_kind, frame.heard_signal_id, frame.heard_direction)]


def sender_information(frame, context, options, output):
    emitted = frame.loc[frame.signal_id.notna()]
    for scope, source, threshold in (("vocal_including_silence", frame, options.min_agent_decisions),
                                     ("signal_given_emitted", emitted, options.min_agent_emissions)):
        v = (source.signal_id.fillna(-1).astype(int) + 1).tolist()
        ids = source.agent_id.tolist()
        eligible = source.agent_id.map(source.agent_id.value_counts()).ge(threshold)
        within = source.loc[eligible]
        within_ids = within.agent_id.tolist()
        within_v = (within.signal_id.fillna(-1).astype(int) + 1).tolist()
        order = list(dict.fromkeys(within_ids))
        within_features = features(within)
        for feature, x in features(source).items():
            rng = analysis_rng(options.analysis_seed, context["run"], context["window_start"], "sender", scope, feature, "pooled")
            report, _ = information_report(v, x, permutations=options.permutations, rng=rng)
            output.add("sender_information", {**context, "feature": feature, "scope": scope,
                       "conditioning": "population_pooled", "min_samples": None,
                       "eligible_agents": len(set(ids)), **report})
            x_within = within_features[feature]
            rng = analysis_rng(options.analysis_seed, context["run"], context["window_start"], "sender", scope, feature, "within_agent")
            report, individual = information_report(within_v, x_within, groups=within_ids,
                                                    permutations=options.permutations, rng=rng)
            output.add("sender_information", {**context, "feature": feature, "scope": scope,
                       "conditioning": "within_agent_eligible", "min_samples": threshold,
                       "eligible_agents": len(order), **report})
            for agent, row in zip(order, individual):
                output.add("sender_information_by_agent", {**context, "agent_id": agent, "feature": feature,
                           "scope": scope, "min_samples": threshold, **row})


def receiver_information(frame, context, options, output):
    raw = labels(frame)
    needs = list(zip(frame.hunger_bin, frame.thirst_bin, frame.energy_bin))
    visual = list(zip(frame.hunger_bin, frame.thirst_bin, frame.energy_bin,
                      frame.food_dx, frame.food_dy, frame.water_dx, frame.water_dy))
    for feature, values in (("auditory_kind", frame.auditory_kind.tolist()), ("resolved_percept", raw)):
        for conditioning, groups in (("none", None), ("need", needs), ("need_visual", visual)):
            selected = np.arange(len(frame))
            if groups is not None:
                counts = Counter(groups)
                selected = np.asarray([i for i, g in enumerate(groups) if counts[g] >= options.min_receiver_stratum], dtype=int)
            actions = frame.physical_action.iloc[selected].tolist()
            sounds = [values[i] for i in selected]
            strata = None if groups is None else [groups[i] for i in selected]
            rng = analysis_rng(options.analysis_seed, context["run"], context["window_start"], "receiver", feature, conditioning)
            report, _ = information_report(actions, sounds, groups=strata, permutations=options.permutations, rng=rng)
            output.add("receiver_responsiveness", {**context, "feature": feature, "conditioning": conditioning,
                       "min_samples": options.min_receiver_stratum if groups is not None else None, **report})
        counts = Counter(zip(values, frame.physical_action))
        totals = Counter(values)
        categories = ["SILENCE", "MASKED", "IDENTIFIED"] if feature == "auditory_kind" else sorted(totals)
        for category in categories:
            for action in ("move_north", "move_south", "move_east", "move_west", "eat", "drink", "wait", "reproduce"):
                output.add("receiver_action_distribution", {**context, "feature": feature, "category": category,
                           "physical_action": action, "count": counts[category, action], "n": totals[category],
                           "probability": ratio(counts[category, action], totals[category])})


def analyze_window(frame, context, config, options, output, shifts):
    frame = frame.sort_values(["tick", "agent_id"], kind="stable")
    n = len(frame)
    emitted = frame.loc[frame.signal_id.notna()]
    silent = n - len(emitted)
    vocab = config["signal_vocab_size"]
    vocal = np.bincount((frame.signal_id.fillna(-1).astype(int) + 1).to_numpy(dtype=int), minlength=vocab + 1)
    percepts = Counter(frame.auditory_kind)
    output.add("silence_over_time", {**context, "n_decisions": n, "emitted": len(emitted), "silent": silent,
               "silence_rate": ratio(silent, n), "emission_rate": ratio(len(emitted), n)})
    output.add("masking_over_time", {**context, "n_observations": n, "silent_percepts": percepts["SILENCE"],
               "masked": percepts["MASKED"], "identified": percepts["IDENTIFIED"],
               "silence_percept_rate": ratio(percepts["SILENCE"], n), "masked_rate": ratio(percepts["MASKED"], n),
               "identified_rate": ratio(percepts["IDENTIFIED"], n), "emitted": len(emitted),
               "emitted_per_recorded_tick": ratio(len(emitted), context["observed_ticks"]),
               "identified_per_emitted": ratio(percepts["IDENTIFIED"], len(emitted))})
    for index, count in enumerate(vocal):
        output.add("vocal_frequency_over_time", {**context, "vocal_action": "SILENCE" if index == 0 else f"SIGNAL_{index-1}",
                   "count": int(count), "n_decisions": n, "n_emitted": len(emitted), "probability": ratio(int(count), n),
                   "prob_given_emitted": ratio(int(count), len(emitted)) if index else None})
    state_features = features(frame)
    for feature, values in {**state_features, "auditory_kind": frame.auditory_kind.tolist(),
                            "local_density": frame.local_density.tolist()}.items():
        categories = sorted(set(values), key=repr)
        counts = Counter(zip(values, (frame.signal_id.fillna(-1).astype(int) + 1).tolist()))
        totals = Counter(values)
        for category in categories:
            ns = totals[category]
            output.add("silence_by_state", {**context, "feature": feature, "category": str(category),
                       "silent": counts[category, 0], "n": ns, "silence_rate": counts[category, 0] / ns})
            if feature not in FEATURES:
                continue
            n_emitted = ns - counts[category, 0]
            for index in range(vocab + 1):
                action = "SILENCE" if index == 0 else f"SIGNAL_{index-1}"
                output.add("conditional_vocal", {**context, "feature": feature, "category": str(category),
                           "scope": "vocal_including_silence", "vocal_action": action, "count": counts[category, index],
                           "n": ns, "probability": counts[category, index] / ns})
                if index:
                    output.add("conditional_vocal", {**context, "feature": feature, "category": str(category),
                               "scope": "signal_given_emitted", "vocal_action": action, "count": counts[category, index],
                               "n": n_emitted, "probability": ratio(counts[category, index], n_emitted)})
    reference = []
    for agent, block in frame.groupby("agent_id", sort=True):
        counts = np.bincount((block.signal_id.fillna(-1).astype(int) + 1).to_numpy(dtype=int), minlength=vocab + 1)
        ne = int(counts[1:].sum())
        peak = int(np.argmax(counts[1:])) if ne else None
        eligible = ne >= options.min_agent_emissions
        row = {**context, "agent_id": int(agent), "n_decisions": len(block), "n_emitted": ne,
               "silence_rate": counts[0] / len(block), "most_frequent_signal": peak,
               "most_frequent_signal_share": int(counts[peak+1]) / ne if ne else None,
               "vocal_entropy_bits": entropy(counts), "emitted_entropy_bits": entropy(counts[1:]),
               "eligible_decisions": len(block) >= options.min_agent_decisions, "eligible_emitted": eligible}
        output.add("vocal_preferences_by_agent", row)
        for signal in np.flatnonzero(counts[1:]):
            output.add("vocal_frequency_by_agent", {**context, "agent_id": int(agent), "signal_id": int(signal),
                       "count": int(counts[signal+1]), "n_emitted": ne, "n_decisions": len(block),
                       "prob_given_emitted": int(counts[signal+1]) / ne})
        if eligible:
            point = (context["window_start"], context["actual_window_end"], counts[1:].copy())
            if agent not in shifts:
                shifts[agent] = [point, point, 1]
            else:
                shifts[agent][1], shifts[agent][2] = point, shifts[agent][2] + 1
            if len(block) >= options.min_agent_decisions:
                reference.append((int(agent), len(block), counts[1:].copy()))
    sender_information(frame, context, options, output)
    receiver_information(frame, context, options, output)
    rng = analysis_rng(options.analysis_seed, context["run"], context["window_start"], "convergence")
    output.add("population_convergence", {**context, **convergence(emitted.signal_id.astype(int).tolist(),
               emitted.agent_id.tolist(), vocabulary=vocab, min_samples=options.min_agent_emissions,
               max_pairs=options.max_pairs, permutations=options.permutations, rng=rng)})
    return vocal, percepts, reference


def aggregate_time_series(tables):
    rows = []
    specs = {"silence_over_time": ("silence_rate", "emission_rate"),
             "masking_over_time": ("masked_rate", "identified_rate", "silence_percept_rate", "emitted_per_recorded_tick", "identified_per_emitted"),
             "population_convergence": ("mean_jsd_bits", "null_mean_jsd_bits", "observed_minus_null_bits"),
             "sender_information": ("mi_bits", "corrected_mi_bits", "permuted_mean_bits"),
             "receiver_responsiveness": ("mi_bits", "corrected_mi_bits")}
    for table, metrics in specs.items():
        source = tables[table]
        for row in source.to_dict("records"):
            for metric in metrics:
                rows.append({"condition": row["condition"], "seed": row["seed"], "window_start": row["window_start"],
                             "window_end": row["window_end"], "actual_window_end": row["actual_window_end"],
                             "table": table, "metric": metric, "feature": row.get("feature", ""),
                             "scope": row.get("scope", ""), "conditioning": row.get("conditioning", ""), "value": row[metric]})
    keys = ["condition", "window_start", "window_end", "actual_window_end", "table", "metric", "feature", "scope", "conditioning"]
    if not rows:
        return pd.DataFrame(columns=[*keys, "mean", "std", "min", "max", "n"])
    data = pd.DataFrame(rows)
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    if data.duplicated([*keys, "seed"]).any():
        raise ValueError("Each seed must contribute exactly once to each metric/window")
    return data.groupby(keys, sort=True, dropna=False).value.agg(["mean", "std", "min", "max", "count"]).reset_index().rename(columns={"count": "n"})


def analyze(data_dir=Path("data/communication"), output_dir=Path("analysis/communication"),
            pattern="*_communication.csv", options=None):
    options = options or Options()
    runs = discover(data_dir, pattern)
    output_dir = Path(output_dir)
    destinations = {output_dir / (name + ".csv") for name in (*SCHEMAS, "time_series_summary")}
    destinations.add(output_dir / "analysis_metadata.json")
    if {p.resolve() for p in destinations} & {r[k].resolve() for r in runs for k in ("path", "metadata_path")}:
        raise ValueError("Analysis outputs must not overwrite simulation inputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    references, reference_seeds = {}, {}
    with ExitStack() as stack:
        output = Outputs(output_dir, stack)
        for run in runs:
            print(f"Analyzing {run['condition']} seed={run['seed']}: {run['path'].name}", flush=True)
            config, meta = run["metadata"]["config"], run["metadata"]
            shifts, total_percepts = {}, Counter()
            total_vocal = np.zeros(config["signal_vocab_size"] + 1, dtype=np.int64)
            windows = complete = 0
            for index, frame in read_windows(run, options.window_size):
                start, end = index * options.window_size, (index + 1) * options.window_size
                actual_end = min(end, meta["final_tick"])
                observed = int(frame.tick.nunique())
                context = {"run": run["run"], "condition": run["condition"], "seed": run["seed"],
                           "window_start": start, "window_end": end, "actual_window_end": actual_end,
                           "complete_window": actual_end == end and observed == options.window_size,
                           "partial_window": actual_end < end, "observed_ticks": observed}
                if observed < actual_end - start:
                    warnings.warn(f"{run['run']} ({start},{actual_end}]: missing ticks; n refers only to recorded decisions")
                vocal, percepts, reference = analyze_window(frame, context, config, options, output, shifts)
                total_vocal += vocal
                total_percepts.update(percepts)
                windows += 1
                complete += int(context["complete_window"])
                chosen_seed = reference_seeds.setdefault(run["condition"], run["seed"])
                if run["seed"] == chosen_seed and reference:
                    reference = sorted(reference, key=lambda r: (-r[1], r[0]))[:30]
                    references[run["condition"]] = {"seed": run["seed"], "window_start": start,
                        "window_end": actual_end, "agents": reference}
            for agent, (first, last, count) in sorted(shifts.items()):
                output.add("vocal_preference_changes", {"run": run["run"], "condition": run["condition"], "seed": run["seed"],
                           "agent_id": int(agent), "first_window_start": first[0], "first_window_end": first[1],
                           "last_window_start": last[0], "last_window_end": last[1], "first_emitted": int(first[2].sum()),
                           "last_emitted": int(last[2].sum()), "eligible_windows": count,
                           "early_late_jsd_bits": jensen_shannon(first[2], last[2]) if count >= 2 else None})
            n, emitted = int(total_vocal.sum()), int(total_vocal[1:].sum())
            stats = meta.get("statistics", {})
            if stats and (n != int(stats.get("exploratory_actions", 0)) + int(stats.get("exploitative_actions", 0))) and config["brain"] == "learning":
                warnings.warn(f"{run['run']}: decisions differ from cumulative learner actions (missing data or mixed/custom brains)")
            output.add("communication_run_summary", {"run": run["run"], "condition": run["condition"], "seed": run["seed"],
                       "final_tick": meta["final_tick"], "final_population": stats.get("population"), "births": stats.get("births"),
                       "deaths": stats.get("deaths"), "n_decisions": n, "emitted": emitted, "silent": int(total_vocal[0]),
                       "silence_rate": ratio(int(total_vocal[0]), n), "emission_rate": ratio(emitted, n),
                       "masked_rate": ratio(total_percepts["MASKED"], n), "identified_rate": ratio(total_percepts["IDENTIFIED"], n),
                       "silence_percept_rate": ratio(total_percepts["SILENCE"], n), "vocal_entropy_bits": entropy(total_vocal),
                       "emitted_entropy_bits": entropy(total_vocal[1:]), "windows": windows, "complete_windows": complete,
                       "input_bytes": run["path"].stat().st_size})
        tables = {name: pd.DataFrame(rows, columns=SCHEMAS[name]) for name, rows in output.plot_rows.items()}
    summary = aggregate_time_series(tables)
    summary.to_csv(output_dir / "time_series_summary.csv", index=False, float_format="%.12g")
    if __package__:
        from .communication_plots import plots
    else:
        from communication_plots import plots
    image_names = plots(tables, summary, references, output_dir, options.top_signals)
    import matplotlib
    metadata = {"version": "0.6.1", "options": asdict(options), "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                "inputs": [{"file": r["path"].name, "sha256": r["sha256"], "metadata_sha256": r["metadata_sha256"],
                            "condition": r["condition"], "seed": r["seed"], "ecology_id": r["ecology_id"],
                            "simulation_metadata": r["metadata"]} for r in runs],
                "seeds_by_condition": {c: sorted(r["seed"] for r in runs if r["condition"] == c) for c in sorted({r["condition"] for r in runs})},
                "source_sha256": {p.name: file_hash(p) for p in [Path(__file__), Path(__file__).with_name("communication_data.py"),
                                  Path(__file__).with_name("communication_metrics.py"), Path(__file__).with_name("communication_plots.py")]},
                "definitions": {"entropy_mi": "Empirical discrete frequencies; log2 bits. V includes SILENCE; S is conditioned on emitted records only.",
                    "nmi": "MI/min(H(V),H(X)); conditional version uses weighted within-stratum entropies. Zero denominator is null.",
                    "permutations": "Pooled shuffle preserves window marginals but not identity. Within-agent shuffle preserves each eligible agent's marginals; receiver conditioned shuffles preserve each eligible need/visual stratum. No p-values; time autocorrelation is broken, not controlled.",
                    "individual_conditioning": "Empirical I(V;X|agent_id), weighted by eligible agents' decision counts within each run/window. Per-agent reports include their own within-agent shuffled baselines.",
                    "convergence_null": "Shuffle emitted IDs across eligible agents preserving global ID frequencies and each agent's emission count; evaluate identical exhaustive or seeded sampled pairs. JSD log2, 0..1 bits; no semantics implied.",
                    "receiver": "Same receiving tick's actual resolved percept and selected physical action; includes conditioned need and need+visual estimates. Listening OFF/ON is the existing control; correlations do not prove response causality. Retained-memory features are recorded but this analysis uses current resolved sound.",
                    "windows": "(k*size,(k+1)*size], tick 0 has no decisions; actual final endpoints separate partial windows. No interpolation/carry-forward. Empty windows have undefined rates/entropy/MI.",
                    "aggregation": "Every run/seed computed first, equal-seed mean, sample SD ddof=1, finite metric n. Seeds never pool into one population. Input ecologies separated and duplicate condition/seed inputs rejected.",
                    "delivery_ratio": "IDENTIFIED observations/emitted signals in that window, not a transmission success probability. One emission can reach multiple receivers; reception crosses window boundaries.",
                    "missing": "Blank CSV/null JSON for undefined estimates. Zero frequency is valid only with positive exposure. Conditional state categories absent from records are not inferred.",
                    "preferences": "Per-run/window observed choices, not Q-policy preferences. Most frequent signal share conditions on emission; ties choose lowest ID for reporting. First/last eligible-window JSD needs two windows.",
                    "visual_selection": "Signal plots use equal-run frequencies, top-N then OTHER; complete frequencies remain in CSV. Individual heatmaps use the lowest seed per condition, last window with eligible emitters, top 30 by decision exposure then ID; not selected by visual pattern.",
                    "inference": "No significance test, causal interpretation, semantic assignment, fitness ranking or convention claim."},
                "individual_plot_selection": {c: {k: v for k, v in r.items() if k != "agents"} | {"agent_ids": [a[0] for a in r["agents"]]} for c, r in references.items()},
                "csv_files": [name + ".csv" for name in (*SCHEMAS, "time_series_summary")], "png_files": image_names}
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Found {len(runs)} runs; " + "; ".join(f"{c}: {len(seeds)} seeds" for c, seeds in metadata["seeds_by_condition"].items()))
    print(tables["communication_run_summary"][["condition", "seed", "n_decisions", "emission_rate", "masked_rate"]].to_string(index=False))
    print(f"Output written to {output_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/communication"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/communication"))
    parser.add_argument("--pattern", default="*_communication.csv")
    for name, default in asdict(Options()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    args = parser.parse_args()
    try:
        options = Options(**{name: getattr(args, name) for name in asdict(Options())})
        analyze(args.data_dir, args.output_dir, args.pattern, options)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
