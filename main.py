"""Small command-line runner; simulation rules live in dedicated modules."""

import argparse
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from config import Config
from simulation.events import JSONLEventWriter
from simulation.simulation import Simulation
from simulation.statistics import CSVStatisticsWriter


def main() -> None:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Artificial Civilization v0.1: headless artificial life")
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--ticks", type=int, default=defaults.max_ticks, help="maximum ticks to run")
    parser.add_argument("--population", type=int, default=defaults.initial_population)
    parser.add_argument("--status-every", type=int, default=defaults.status_interval)
    parser.add_argument("--csv", type=Path, help="stream aggregate CSV (overwrites destination)")
    parser.add_argument("--events", type=Path, help="opt-in detailed JSONL (overwrites destination)")
    args = parser.parse_args()
    try:
        config = replace(defaults, random_seed=args.seed, max_ticks=args.ticks,
                         initial_population=args.population, status_interval=args.status_every)
    except ValueError as error:
        parser.error(str(error))
    if args.csv and args.events and args.csv.resolve() == args.events.resolve():
        parser.error("--csv and --events must use different files")

    with ExitStack() as stack:
        event_sink = None
        csv_writer = None
        if args.events:
            args.events.parent.mkdir(parents=True, exist_ok=True)
            event_sink = JSONLEventWriter(stack.enter_context(args.events.open("w", encoding="utf-8")))
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            csv_writer = CSVStatisticsWriter(stack.enter_context(args.csv.open("w", newline="", encoding="utf-8")))
        sim = Simulation(config, event_sink=event_sink)
        if csv_writer is not None:
            csv_writer.write(sim.statistics)
        print(f"seed={config.random_seed}")
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
