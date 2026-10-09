"""Explicit short 2x2 pilot, using the unchanged evolution preset through main.py.

This command runs simulations; analyze_communication only reads their files.
Existing destinations are rejected rather than overwritten. No ecology tuning.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

from config import Config

CONDITIONS = (("A", False, False), ("B", True, False), ("C", False, True), ("D", True, True))


def run_pilot(output_dir=Path("data/communication"), ticks=1000, seeds=(0, 1, 2)):
    if type(ticks) is not int or not 1 <= ticks <= 1000:
        raise ValueError("This short pilot permits only 1..1000 ticks; no long experiments")
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct integers")
    directory = Path(output_dir).resolve()
    jobs, paths = [], [directory / "pilot_manifest.json"]
    for condition, listening, production in CONDITIONS:
        for seed in sorted(seeds):
            prefix = directory / f"{condition}_seed_{seed}"
            comm = prefix.with_name(prefix.name + "_communication.csv")
            stats = prefix.with_name(prefix.name + "_statistics.csv")
            log = prefix.with_suffix(".log")
            paths.extend([comm, comm.with_suffix(".csv.metadata.json"), stats, stats.with_suffix(".csv.metadata.json"), log])
            jobs.append((condition, listening, production, seed, comm, stats, log))
    if any(path.exists() for path in paths):
        raise ValueError("Pilot destinations already exist; choose another output directory")
    directory.mkdir(parents=True, exist_ok=True)
    reports = []
    for condition, listening, production, seed, comm, stats, log in jobs:
        command = [sys.executable, str(ROOT / "main.py"), "--mode", "evolution", "--brain", "learning",
            "--learning-uses-auditory" if listening else "--no-learning-uses-auditory",
            "--learning-controls-vocalization" if production else "--no-learning-controls-vocalization",
            "--seed", str(seed), "--ticks", str(ticks), "--csv", str(stats), "--communication-csv", str(comm)]
        print(f"Running {condition} seed={seed}, up to {ticks} ticks", flush=True)
        start = perf_counter()
        with log.open("w") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        meta = json.loads(comm.with_suffix(".csv.metadata.json").read_text())
        reports.append({"condition": condition, "seed": seed, "command": command, "seconds": perf_counter()-start,
            "communication_file": comm.name, "communication_bytes": comm.stat().st_size,
            "statistics_file": stats.name, "final_tick": meta["final_tick"], "statistics": meta["statistics"],
            "config_id": meta["config_id"], "config": meta["config"]})
    report = {"version": "0.6.1", "ticks_requested": ticks, "seeds": sorted(seeds), "runs": reports,
              "ecology": asdict(Config.evolution(brain="learning")),
              "interpretation": "Short descriptive pilot, not evidence of meaning, communication or ecological advantage."}
    (directory / "pilot_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(reports)} runs; {sum(r['communication_bytes'] for r in reports):,} communication bytes")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/communication"))
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    try:
        run_pilot(args.output_dir, args.ticks, tuple(args.seeds))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
