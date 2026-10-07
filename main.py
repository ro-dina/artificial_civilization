"""Small command-line runner; simulation rules live in dedicated modules."""

import argparse
import hashlib
import json
import platform
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

from config import Config
from simulation.cohorts import CSVCohortWriter
from simulation.events import Event, JSONLEventWriter
from simulation.simulation import Simulation
from simulation.statistics import CSVStatisticsWriter


def main() -> None:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Artificial Civilization v0.6: headless artificial life")
    parser.add_argument("--mode", choices=("legacy", "evolution"), default="legacy")
    parser.add_argument("--brain", choices=("random", "learning"), default=defaults.brain)
    parser.add_argument("--learning-uses-auditory", action=argparse.BooleanOptionalAction,
                        default=defaults.learning_uses_auditory,
                        help="enable auditory Q-state features; combine both --no-learning-* flags for the historical control")
    parser.add_argument("--learning-controls-vocalization", action=argparse.BooleanOptionalAction,
                        default=defaults.learning_controls_vocalization,
                        help="enable Vocal Q; --no-learning-controls-vocalization preserves v0.5 random emission")
    parser.add_argument("--hearing-radius", type=int, default=defaults.hearing_radius,
                        help="perfect auditory range in cells (independent of visual perception)")
    parser.add_argument("--signal-vocab-size", type=int, default=defaults.signal_vocab_size,
                        help="meaningless integer signal types; 16 retains the historical vocabulary")
    parser.add_argument("--fixed-perception-radius", type=int, default=defaults.fixed_perception_radius,
                        help="fix the genome's perception radius for founders and all descendants")
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--ticks", type=int, default=defaults.max_ticks, help="maximum ticks to run")
    parser.add_argument("--population", type=int, default=defaults.initial_population)
    parser.add_argument("--status-every", type=int, default=defaults.status_interval)
    parser.add_argument("--csv", type=Path, help="stream CSV and .metadata.json sidecar (overwrites destinations)")
    parser.add_argument("--events", type=Path, help="opt-in detailed JSONL (overwrites destination)")
    parser.add_argument("--cohort-csv", type=Path, help="opt-in per-tick perception cohorts with metadata sidecar")
    parser.add_argument("--lineage-events", type=Path, help="opt-in compact founder/birth/death JSONL")
    args = parser.parse_args()
    try:
        if args.mode == "evolution":
            defaults = Config.evolution()
        config = replace(defaults, random_seed=args.seed, max_ticks=args.ticks,
                         initial_population=args.population, status_interval=args.status_every, brain=args.brain,
                         fixed_perception_radius=args.fixed_perception_radius,
                         hearing_radius=args.hearing_radius, signal_vocab_size=args.signal_vocab_size,
                         learning_uses_auditory=args.learning_uses_auditory,
                         learning_controls_vocalization=args.learning_controls_vocalization)
    except ValueError as error:
        parser.error(str(error))
    metadata_path = args.csv.with_suffix(args.csv.suffix + ".metadata.json") if args.csv else None
    cohort_metadata_path = args.cohort_csv.with_suffix(args.cohort_csv.suffix + ".metadata.json") if args.cohort_csv else None
    paths = [path.resolve() for path in (args.csv, args.events, metadata_path, args.cohort_csv,
                                       cohort_metadata_path, args.lineage_events) if path is not None]
    if len(set(paths)) != len(paths):
        parser.error("CSV, metadata, and events must use different files (including cohort/lineage outputs)")
    config_id = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    metadata = {"mode": args.mode, "brain": config.brain, "seed": config.random_seed, "config_id": config_id}

    with ExitStack() as stack:
        event_sink = None
        lineage_sink = None
        cohort_sink = None
        csv_writer = None
        if args.events:
            args.events.parent.mkdir(parents=True, exist_ok=True)
            event_sink = JSONLEventWriter(stack.enter_context(args.events.open("w", encoding="utf-8")))
        if args.lineage_events:
            args.lineage_events.parent.mkdir(parents=True, exist_ok=True)
            lineage_sink = JSONLEventWriter(stack.enter_context(args.lineage_events.open("w", encoding="utf-8")))
        if args.cohort_csv:
            args.cohort_csv.parent.mkdir(parents=True, exist_ok=True)
            cohort_sink = CSVCohortWriter(
                stack.enter_context(args.cohort_csv.open("w", newline="", encoding="utf-8")), metadata=metadata)
            cohort_metadata_path.write_text(json.dumps({**metadata, "cohort_schema_version": 1,
                "config": asdict(config), "counters": "per tick, not cumulative; tick 0 has zero exposure/events",
                "living_agent_ticks": "completed-tick living population, including newborns",
                "action_opportunities": "agents that acted, excluding newborns, including deaths during tick",
                "births_as_child": "child perception at birth",
                "successful_parent_participations": "both parents at successful reproduction; two per birth",
                "reproduction_initiations": "REPRODUCE chooser only; includes failed attempts",
                "deaths": "genome at death; starvation, dehydration, old_age, energy priority",
                "empty_cohort_means": None}, indent=2) + "\n", encoding="utf-8")
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            csv_writer = CSVStatisticsWriter(
                stack.enter_context(args.csv.open("w", newline="", encoding="utf-8")), metadata=metadata)
            metadata_path.write_text(json.dumps({**metadata, "version": "0.6", "python": platform.python_version(),
                                                 "config": asdict(config)}, indent=2) + "\n", encoding="utf-8")
        sim = Simulation(config, event_sink=event_sink, cohort_sink=cohort_sink, lineage_sink=lineage_sink)
        if csv_writer is not None:
            csv_writer.write(sim.statistics)
        fixed_label = (f" fixed_perception_radius={config.fixed_perception_radius}"
                       if config.fixed_perception_radius is not None else "")
        print(f"mode={args.mode} brain={config.brain} seed={config.random_seed}{fixed_label} "
              f"hearing_radius={config.hearing_radius} signal_vocab_size={config.signal_vocab_size} "
              f"learning_uses_auditory={config.learning_uses_auditory} "
              f"learning_controls_vocalization={config.learning_controls_vocalization}")
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
        if lineage_sink is not None:
            lineage_sink(Event(sim.tick, "run_finished", details={"population": sim.population}))


if __name__ == "__main__":
    main()
