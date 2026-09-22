"""Small command-line runner; simulation rules live in dedicated modules."""

import argparse
import hashlib
import json
import platform
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

from config import Config
from simulation.events import JSONLEventWriter
from simulation.simulation import Simulation
from simulation.statistics import CSVStatisticsWriter


def main() -> None:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Artificial Civilization v0.3: headless artificial life")
    parser.add_argument("--mode", choices=("legacy", "evolution"), default="legacy")
    parser.add_argument("--brain", choices=("random", "learning"), default=defaults.brain)
    parser.add_argument("--fixed-perception-radius", type=int, default=defaults.fixed_perception_radius,
                        help="fix the genome's perception radius for founders and all descendants")
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--ticks", type=int, default=defaults.max_ticks, help="maximum ticks to run")
    parser.add_argument("--population", type=int, default=defaults.initial_population)
    parser.add_argument("--status-every", type=int, default=defaults.status_interval)
    parser.add_argument("--csv", type=Path, help="stream CSV and .metadata.json sidecar (overwrites destinations)")
    parser.add_argument("--events", type=Path, help="opt-in detailed JSONL (overwrites destination)")
    args = parser.parse_args()
    try:
        if args.mode == "evolution":
            defaults = Config.evolution()
        config = replace(defaults, random_seed=args.seed, max_ticks=args.ticks,
                         initial_population=args.population, status_interval=args.status_every, brain=args.brain,
                         fixed_perception_radius=args.fixed_perception_radius)
    except ValueError as error:
        parser.error(str(error))
    metadata_path = args.csv.with_suffix(args.csv.suffix + ".metadata.json") if args.csv else None
    paths = [path.resolve() for path in (args.csv, args.events, metadata_path) if path is not None]
    if len(set(paths)) != len(paths):
        parser.error("CSV, metadata, and events must use different files")
    config_id = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    metadata = {"mode": args.mode, "brain": config.brain, "seed": config.random_seed, "config_id": config_id}

    with ExitStack() as stack:
        event_sink = None
        csv_writer = None
        if args.events:
            args.events.parent.mkdir(parents=True, exist_ok=True)
            event_sink = JSONLEventWriter(stack.enter_context(args.events.open("w", encoding="utf-8")))
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            csv_writer = CSVStatisticsWriter(
                stack.enter_context(args.csv.open("w", newline="", encoding="utf-8")), metadata=metadata)
            metadata_path.write_text(json.dumps({**metadata, "version": "0.3", "python": platform.python_version(),
                                                 "config": asdict(config)}, indent=2) + "\n", encoding="utf-8")
        sim = Simulation(config, event_sink=event_sink)
        if csv_writer is not None:
            csv_writer.write(sim.statistics)
        fixed_label = (f" fixed_perception_radius={config.fixed_perception_radius}"
                       if config.fixed_perception_radius is not None else "")
        print(f"mode={args.mode} brain={config.brain} seed={config.random_seed}{fixed_label}")
        last_status_tick = None
        for _ in range(config.max_ticks):
            if sim.population == 0:
                break
            stats = sim.step()
            if csv_writer is not None:
                csv_writer.write(stats)
            if sim.tick % config.status_interval == 0:
                sim.print_status()
                last_status_tick = sim.tick
        if last_status_tick != sim.tick:
            sim.print_status()


if __name__ == "__main__":
    main()
