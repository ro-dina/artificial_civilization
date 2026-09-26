"""Two-parent founder ancestry reachability, not exclusive lineages/genetic contributions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

if __package__:
    from .perception_analysis_common import pd, seed_files, write_tables, metadata
else:
    from perception_analysis_common import pd, seed_files, write_tables, metadata

TRAITS = ("hunger_multiplier", "thirst_multiplier", "signal_probability")


def read_lineage(path, expected_seed=None):
    """Offline records only. Bit sets preserve both-parent DAG reachability compactly."""
    records, founders, edges = {}, [], []
    final_tick, finished, previous_tick = 0, False, 0
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
                tick, kind, agent_id, detail = event["tick"], event["kind"], event.get("agent_id"), event["details"]
                if type(tick) is not int or tick < previous_tick or finished:
                    raise ValueError("ticks must be ordered; run_finished must be last")
                previous_tick = final_tick = tick
                if kind in ("founder", "birth"):
                    if type(agent_id) is not int or agent_id in records:
                        raise ValueError("duplicate/invalid agent ID")
                    if kind == "founder":
                        if tick != 0 or detail["generation"] != 0:
                            raise ValueError("founders must be generation 0 at tick 0")
                        mask = 1 << len(founders)
                        founders.append(agent_id)
                        genome = detail["genome"]
                    else:
                        a, b = detail["parent_a_id"], detail["parent_b_id"]
                        if a == b or a not in records or b not in records:
                            raise ValueError("birth needs two distinct, already recorded parents")
                        if any(records[p]["birth_tick"] >= tick or (records[p]["death_tick"] is not None
                               and records[p]["death_tick"] < tick) for p in (a, b)):
                            raise ValueError("parent lifetime is incompatible with birth tick")
                        if detail["generation"] != max(records[a]["generation"], records[b]["generation"]) + 1:
                            raise ValueError("inconsistent child generation")
                        mask = records[a]["ancestors"] | records[b]["ancestors"]
                        genome = detail["child_genome"]
                        edges.append({"child_id": agent_id, "parent_a_id": a, "parent_b_id": b, "birth_tick": tick})
                    records[agent_id] = {"birth_tick": tick, "death_tick": None, "generation": detail["generation"],
                                         "genome": genome, "ancestors": mask, "founder": kind == "founder"}
                elif kind == "death":
                    if agent_id not in records or records[agent_id]["death_tick"] is not None:
                        raise ValueError("death must reference a recorded living agent")
                    if records[agent_id]["genome"] != detail["genome_at_death"]:
                        raise ValueError("death genome differs from lifetime genome")
                    if detail["death_cause"] not in ("starvation", "dehydration", "old_age", "energy"):
                        raise ValueError("unknown death cause")
                    records[agent_id]["death_tick"] = tick
                elif kind == "run_finished":
                    actual = sum(r["death_tick"] is None for r in records.values())
                    if actual != detail["population"]:
                        raise ValueError("run_finished population disagrees with birth/death log")
                    finished = True
                elif kind == "run_started":
                    if expected_seed is not None and detail["config"]["random_seed"] != expected_seed:
                        raise ValueError("run_started seed disagrees with filename")
                else:
                    raise ValueError(f"unexpected event kind {kind!r}; use --lineage-events")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not finished:
        warnings.warn(f"{path}: no run_finished marker; horizon is last event tick ({final_tick})", stacklevel=2)
    return records, founders, edges, final_tick


def summarize(records, founders, ticks, seed):
    descendants, distributions, mixing = [], [], []
    for tick in sorted(set(ticks)):
        born = [(agent_id, r) for agent_id, r in records.items() if r["birth_tick"] <= tick]
        living = [(agent_id, r) for agent_id, r in born if r["death_tick"] is None or r["death_tick"] > tick]
        counts = [r["ancestors"].bit_count() for _, r in living]
        mixing.append({"seed": seed, "tick": tick, "living_population": len(living),
                       "mean_founder_ancestors": sum(counts) / len(counts) if counts else None,
                       "min_founder_ancestors": min(counts, default=None), "max_founder_ancestors": max(counts, default=None)})
        for index, founder_id in enumerate(founders):
            bit = 1 << index
            members = [r for agent_id, r in living if agent_id != founder_id and r["ancestors"] & bit]
            descendants.append({"seed": seed, "tick": tick, "founder_id": founder_id,
                "descendants_born": sum(agent_id != founder_id and bool(r["ancestors"] & bit) for agent_id, r in born),
                "living_descendants": len(members)})
            for p in range(9):
                cohort = [r["genome"] for r in members if r["genome"]["perception_radius"] == p]
                row = {"seed": seed, "tick": tick, "founder_id": founder_id, "perception_radius": p,
                       "living_descendants": len(cohort)}
                for trait in TRAITS:
                    values = [g[trait] for g in cohort]
                    row.update({f"mean_{trait}": sum(values) / len(values) if values else None,
                                f"min_{trait}": min(values, default=None), f"max_{trait}": max(values, default=None)})
                distributions.append(row)
    return descendants, distributions, mixing


def analyze(data_dir=Path("data"), output_dir=Path("analysis/lineages"),
            pattern="evolving_seed_*_lineage.jsonl", ticks=(0, 1000, 2000, 3000, 4000, 5000)):
    if any(type(t) is not int or t < 0 for t in ticks):
        raise ValueError("selected ticks must be nonnegative integers")
    files = seed_files(data_dir, pattern)
    descendants, distributions, mixing, all_edges = [], [], [], []
    for seed, path in files:
        records, founders, edges, horizon = read_lineage(path, expected_seed=seed)
        selected = sorted({horizon, *(t for t in ticks if t <= horizon)})
        a, b, c = summarize(records, founders, selected, seed)
        descendants.extend(a)
        distributions.extend(b)
        mixing.extend(c)
        all_edges.extend({"seed": seed, **edge} for edge in edges)
        print(f"seed={seed}: {len(founders)} founders, {len(edges)} births, final_tick={horizon}")
    write_tables(output_dir,
        founder_descendants=pd.DataFrame(descendants, columns=["seed", "tick", "founder_id", "descendants_born", "living_descendants"]),
        founder_trait_distributions=pd.DataFrame(distributions, columns=["seed", "tick", "founder_id", "perception_radius", "living_descendants",
            *(f"{prefix}_{trait}" for trait in TRAITS for prefix in ("mean", "min", "max"))]),
        ancestry_mixing=pd.DataFrame(mixing),
        parent_child_edges=pd.DataFrame(all_edges, columns=["seed", "child_id", "parent_a_id", "parent_b_id", "birth_tick"]))
    metadata(output_dir, files, __file__, definition="Both-parent DAG founder reachability; memberships overlap, exclude founder itself from descendants",
             selected_ticks=ticks, limitation="Genealogical reachability is not genetic contribution, independent lineage membership, or evidence of hitchhiking")
    print("Founder memberships overlap. Counts must not be summed into a population or treated as independent lineages.")
    print(f"Output written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/lineages"))
    parser.add_argument("--pattern", default="evolving_seed_*_lineage.jsonl")
    parser.add_argument("--ticks", type=int, nargs="+", default=[0, 1000, 2000, 3000, 4000, 5000])
    args = parser.parse_args()
    try:
        analyze(args.data_dir, args.output_dir, args.pattern, args.ticks)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
