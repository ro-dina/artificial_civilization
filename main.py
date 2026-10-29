"""Small command-line runner; simulation rules live in dedicated modules."""

import argparse
import hashlib
import json
import platform
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path

from config import Config, SocialConfig
from simulation.cohorts import CSVCohortWriter
from simulation.communication_observer import COLUMNS, DENSITY_RADIUS, SCHEMA_VERSION, CSVCommunicationWriter
from simulation.events import Event, JSONLEventWriter
from simulation.simulation import Simulation
from simulation.statistics import CSVStatisticsWriter
from simulation.social_observer import CSVSocialWriter, write_metadata as write_social_metadata
from simulation.vocal_learning_observer import (SCHEMAS as VOCAL_SCHEMAS, CSVVocalLearningWriter,
    VocalLearningObserver, hash_file, recording_paths)


def main() -> None:
    defaults = Config()
    social_defaults = SocialConfig()
    parser = argparse.ArgumentParser(description="Artificial Civilization v0.7.0: headless artificial life")
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
    parser.add_argument("--communication-csv", type=Path, help="opt-in compact decision/perception CSV with metadata sidecar")
    parser.add_argument("--vocal-learning-csv", type=Path,
                        help="opt-in Vocal Q decisions, updates, row events and key catalog (.csv or .csv.gz)")
    parser.add_argument('--social-memory', action=argparse.BooleanOptionalAction, default=social_defaults.social_memory_enabled,
                        help='opt-in visual identity and bounded memory; current brains ignore it')
    parser.add_argument('--social-memory-capacity', type=int, default=social_defaults.social_memory_capacity)
    parser.add_argument('--social-reunion-min-absence-ticks', type=int, default=social_defaults.social_reunion_min_absence_ticks,
                        help='inclusive elapsed-sighting threshold; continuous contact is never a reunion')
    parser.add_argument('--social-csv', type=Path, help='exclusive social recording (.csv or .csv.gz) with completion sidecar')
    args = parser.parse_args()
    try:
        social_config = SocialConfig(args.social_memory, args.social_memory_capacity, args.social_reunion_min_absence_ticks)
        if args.social_csv is not None and not args.social_memory:
            raise ValueError('--social-csv requires --social-memory')
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
    communication_metadata_path = args.communication_csv.with_suffix(args.communication_csv.suffix + ".metadata.json") if args.communication_csv else None
    social_metadata_path = Path(str(args.social_csv) + ".metadata.json") if args.social_csv else None
    vocal_paths = recording_paths(args.vocal_learning_csv) if args.vocal_learning_csv else {}
    vocal_metadata_path = Path(str(args.vocal_learning_csv) + ".metadata.json") if args.vocal_learning_csv else None
    paths = [path.resolve() for path in (args.csv, args.events, metadata_path, args.cohort_csv,
                                       cohort_metadata_path, args.lineage_events, args.communication_csv,
                                       communication_metadata_path, vocal_metadata_path, args.social_csv, social_metadata_path, *vocal_paths.values()) if path is not None]
    if len(set(paths)) != len(paths):
        parser.error("CSV, metadata, and events must use different files (including communication/cohort/lineage outputs)")
    if args.social_csv and any(path.exists() for path in paths):
        parser.error('Social recordings refuse existing output/sidecar destinations; choose new paths')
    config_id = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    metadata = {"mode": args.mode, "brain": config.brain, "seed": config.random_seed, "config_id": config_id}

    with ExitStack() as stack:
        event_sink = None
        lineage_sink = None
        cohort_sink = None
        csv_writer = None
        communication_sink = None
        vocal_observer = None
        social_writer = CSVSocialWriter.open(stack, args.social_csv) if args.social_csv else None
        if args.vocal_learning_csv:
            vocal_writer = CSVVocalLearningWriter.open(stack, args.vocal_learning_csv)
            vocal_observer = VocalLearningObserver(config, vocal_writer)
        if args.communication_csv:
            args.communication_csv.parent.mkdir(parents=True, exist_ok=True)
            communication_sink = CSVCommunicationWriter(
                stack.enter_context(args.communication_csv.open("w", newline="", encoding="utf-8")))
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
            metadata_path.write_text(json.dumps({**metadata, "version": "0.6.1", "python": platform.python_version(),
                                                 "config": asdict(config)}, indent=2) + "\n", encoding="utf-8")
        sim = Simulation(config, event_sink=event_sink, cohort_sink=cohort_sink, lineage_sink=lineage_sink,
                         communication_sink=communication_sink, vocal_learning_observer=vocal_observer,
                         social_config=social_config, social_sink=social_writer)
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
        if sim.social_observer is not None:
            sim.social_observer.finish(sim.statistics)
        if vocal_observer is not None:
            vocal_observer.finish(sim.agents, sim.tick)
        if communication_metadata_path is not None:
            root = Path(__file__).parent
            sources = ("main.py", "config.py", "agents/brain.py", "agents/learning.py", "agents/observation.py",
                       "agents/auditory_memory.py", "simulation/simulation.py", "simulation/communication_observer.py",
                       "systems/communication.py", "systems/reproduction.py", "agents/genome.py", "agents/human.py",
                       "world/world.py", "simulation/statistics.py")
            communication_metadata_path.write_text(json.dumps({**metadata, "version": "0.6.1",
                "python": platform.python_version(), "schema_version": SCHEMA_VERSION, "columns": COLUMNS,
                "config": asdict(config), "final_tick": sim.tick, "statistics": asdict(sim.statistics),
                "density_radius": DENSITY_RADIUS,
                "density_definition": "Other existing agents within Chebyshev radius at tick start, before any actions; includes subsequent deaths and excludes newborns. Privileged, not Brain-visible.",
                "timing": "No decisions at tick 0. Row t combines the actual resolved reception of t-1 emissions with that actor's same-turn choices at t; body/age are pre-action.",
                "state_definition": "LearningBrain's already encoded pending state; RandomBrain/custom body-visual diagnostics use the same primitive encoder, not a cognitive state of those brains.",
                "sampling": "None; every ordinary actor decision, including dying actors, excluding newborns during birth tick.",
                "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}},
                indent=2) + "\n", encoding="utf-8")

    if args.social_csv:
        write_social_metadata(args.social_csv, sim, social_writer)

    # Closed/flushed files only: a sidecar certifies a completed recording.
    if vocal_metadata_path is not None:
        root = Path(__file__).parent
        vocal_metadata_path.write_text(json.dumps({**metadata, "version": "0.6.2", "schema_version": 1,
            "completed": True, "config": asdict(config), "final_tick": sim.tick,
            "statistics": asdict(sim.statistics), "schemas": VOCAL_SCHEMAS,
            "files": {k: {"file": p.name, "sha256": hash_file(p), "bytes": p.stat().st_size,
                           "records": vocal_writer.counts[k]} for k, p in vocal_paths.items()},
            "unique_keys": len(vocal_observer.keys), "peak_agent_state_histories": vocal_observer.peak_histories,
            "communication": ({"file": str(args.communication_csv.resolve()),
                               "sha256": hash_file(args.communication_csv),
                               "metadata_sha256": hash_file(communication_metadata_path)} if args.communication_csv else None),
            "source_sha256": {p: hash_file(root / p) for p in ("agents/learning.py", "main.py",
                "simulation/simulation.py", "simulation/vocal_learning_observer.py")},
            "sampling": "None: every decision, every actual update and row event; exact keys interned once.",
            "missing": "Blank: no Vocal Q in A/B, RandomBrain or custom brains.",
            "timing": "Choice diagnostics after previous transition update, before execution. Start-presence precedes that update; selection-presence follows it. Terminal update follows execution.",
            "novelty": "Per individual: first ever chosen key; stored at turn start; previously chosen and absent at turn start (recreated).",
            "memory": "Optional observer retains exact key catalog and alive agents' per-key scalar histories; released at death/end. No Brain history added."},
            indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
