"""Validated compact inputs, bounded-by-window loading and run identities."""

import hashlib
import json
from pathlib import Path
import warnings

import pandas as pd

from simulation.communication_observer import COLUMNS, SCHEMA_VERSION

STRINGS = {"brain_type", "vocal_mechanism", "vocal_action", "physical_action", "auditory_kind",
           "heard_direction", "encoded_auditory_kind", "encoded_direction"}
FLAGS = {"learning_controls_vocalization", "learning_uses_auditory"}
OPTIONAL_INTS = {"signal_id", "heard_signal_id", "encoded_signal_id", "encoded_age"}
DIRECTIONS = {"N", "NE", "E", "SE", "S", "SW", "W", "NW", "SAME_CELL"}
ACTIONS = {"move_north", "move_south", "move_east", "move_west", "eat", "drink", "wait", "reproduce"}
FEATURES = ("hunger_bin", "thirst_bin", "energy_bin", "food_direction", "water_direction")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(data_dir, pattern):
    paths = sorted(Path(data_dir).glob(pattern))
    if not paths:
        raise ValueError(f"No communication inputs: {data_dir}/{pattern}")
    runs, identities = [], set()
    for path in paths:
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        if not sidecar.is_file():
            raise ValueError(f"{path}: requires its communication .metadata.json sidecar")
        meta = json.loads(sidecar.read_text())
        if meta.get("schema_version") != SCHEMA_VERSION or tuple(meta.get("columns", ())) != COLUMNS:
            raise ValueError(f"{path}: incompatible communication schema")
        cfg = meta["config"]
        ecology = {k: v for k, v in cfg.items() if k not in {"random_seed", "max_ticks", "status_interval",
                   "brain", "learning_uses_auditory", "learning_controls_vocalization"}}
        ecology_id = hashlib.sha256(json.dumps(ecology, sort_keys=True).encode()).hexdigest()[:12]
        condition = ("ABCD"[int(cfg["learning_uses_auditory"]) + 2 * int(cfg["learning_controls_vocalization"])]
                     if cfg["brain"] == "learning" else cfg["brain"])
        identity = (ecology_id, condition, meta["seed"])
        if identity in identities:
            raise ValueError(f"Duplicate condition/seed input: {identity}; do not double-weight one run")
        identities.add(identity)
        if type(meta.get("final_tick")) is not int or meta["final_tick"] < 0 or meta["seed"] != cfg["random_seed"]:
            raise ValueError(f"{path}: invalid final_tick or inconsistent seed")
        runs.append({"path": path, "metadata_path": sidecar, "metadata": meta, "ecology_id": ecology_id,
                     "condition": condition, "seed": meta["seed"], "run": path.stem,
                     "sha256": file_hash(path), "metadata_sha256": file_hash(sidecar)})
    if len({r["ecology_id"] for r in runs}) > 1:
        for run in runs:
            run["condition"] += ":" + run["ecology_id"][:6]
        warnings.warn("Multiple ecologies: kept as separate conditions, never seed-pooled")
    return sorted(runs, key=lambda r: (r["condition"], r["seed"], r["run"]))


def validate_frame(frame, cfg, path):
    if set(frame) != set(COLUMNS):
        raise ValueError(f"{path}: columns must match the compact communication schema")
    required = set(COLUMNS) - OPTIONAL_INTS - {"heard_direction", "encoded_auditory_kind", "encoded_direction"}
    if frame[list(required)].isna().any().any() or not frame.tick.is_monotonic_increasing:
        raise ValueError(f"{path}: missing required fields or ticks not in streaming order")
    nonnegative = set(COLUMNS) - STRINGS - FLAGS - {"food_dx", "food_dy", "water_dx", "water_dy"}
    if frame[list(nonnegative)].lt(0).any().any() or frame.tick.eq(0).any():
        raise ValueError(f"{path}: decisions require positive ticks and nonnegative numeric fields")
    if frame.duplicated(["tick", "agent_id"]).any() or not frame.physical_action.isin(ACTIONS).all():
        raise ValueError(f"{path}: duplicate decisions or invalid physical action")
    for name in ("hunger_bin", "thirst_bin", "energy_bin"):
        if not frame[name].lt(cfg["learning_need_bins"]).all():
            raise ValueError(f"{path}: {name} outside the actual encoder bins")
    for name in ("food", "water"):
        dx, dy = frame[name + "_dx"], frame[name + "_dy"]
        if not ((dx.isin([-1, 0, 1]) & dy.isin([-1, 0, 1])) | (dx.eq(2) & dy.eq(2))).all():
            raise ValueError(f"{path}: invalid {name} direction; absent is exactly (2,2)")
    if not frame.vocal_action.isin(["SILENCE", "SIGNAL"]).all() or not frame.auditory_kind.isin(["SILENCE", "MASKED", "IDENTIFIED"]).all():
        raise ValueError(f"{path}: invalid vocal/auditory tag")
    for tag, id_col, direction_col in ((frame.vocal_action.eq("SIGNAL"), "signal_id", None),
                                      (frame.auditory_kind.eq("IDENTIFIED"), "heard_signal_id", "heard_direction")):
        if not frame[id_col].notna().eq(tag).all() or frame[id_col].dropna().ge(cfg["signal_vocab_size"]).any():
            raise ValueError(f"{path}: tag/payload or vocabulary mismatch for {id_col}")
        if direction_col and (not frame[direction_col].notna().eq(tag).all() or not frame.loc[tag, direction_col].isin(DIRECTIONS).all()):
            raise ValueError(f"{path}: invalid resolved source direction")
    if not frame.brain_type.isin(["learning", "random", "custom"]).all() or not frame.vocal_mechanism.isin(["learned", "random", "custom"]).all():
        raise ValueError(f"{path}: invalid brain/emission mechanism")
    if not frame.vocal_mechanism.eq("learned").eq(frame.learning_controls_vocalization).all():
        raise ValueError(f"{path}: learned and random emission mechanisms must remain distinct")
    active = frame.learning_uses_auditory
    if ((frame.learning_controls_vocalization | active) & frame.brain_type.ne("learning")).any():
        raise ValueError(f"{path}: only LearningBrain can use either learning head")
    if not frame.encoded_auditory_kind.notna().eq(active).all():
        raise ValueError(f"{path}: auditory encoder ablation mismatch")
    if not frame.loc[active, "encoded_auditory_kind"].isin(["SILENCE", "MASKED", "IDENTIFIED"]).all():
        raise ValueError(f"{path}: invalid retained auditory kind")
    identified = frame.encoded_auditory_kind.eq("IDENTIFIED").fillna(False)
    aged = frame.encoded_auditory_kind.isin(["MASKED", "IDENTIFIED"])
    if (not frame.encoded_signal_id.notna().eq(identified).all()
            or not frame.encoded_direction.notna().eq(identified).all()
            or not frame.encoded_age.notna().eq(aged).all()
            or not frame.loc[identified, "encoded_direction"].isin(DIRECTIONS).all()
            or frame.encoded_signal_id.dropna().ge(cfg["signal_vocab_size"]).any()
            or frame.encoded_age.dropna().ge(cfg["auditory_memory_ticks"]).any()):
        raise ValueError(f"{path}: invalid retained auditory feature")


def read_windows(run, size, chunksize=50000):
    """One window at a time. Never interpolate or carry observations past extinction."""
    dtypes = {name: "string" if name in STRINGS else "boolean" if name in FLAGS
              else "Int64" if name in OPTIONAL_INTS else "int64" for name in COLUMNS}
    horizon = run["metadata"]["final_tick"]
    current, parts, last_tick, last_ids = 0, [], 0, set()
    with pd.read_csv(run["path"], dtype=dtypes, chunksize=chunksize) as reader:
        for frame in reader:
            if frame.empty:
                continue
            validate_frame(frame, run["metadata"]["config"], run["path"])
            first, final = int(frame.tick.iloc[0]), int(frame.tick.iloc[-1])
            if first < last_tick or final > horizon:
                raise ValueError(f"{run['path']}: tick ordering/horizon mismatch")
            ids = set(frame.loc[frame.tick.eq(first), "agent_id"])
            if first == last_tick and ids & last_ids:
                raise ValueError(f"{run['path']}: duplicate decision across chunks")
            final_ids = set(frame.loc[frame.tick.eq(final), "agent_id"])
            last_ids = last_ids | final_ids if first == final == last_tick else final_ids
            last_tick = final
            for index, block in frame.groupby((frame.tick - 1) // size, sort=True):
                while current < index:
                    yield current, pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
                    current, parts = current + 1, []
                parts.append(block)
    if horizon and last_tick != horizon:
        warnings.warn(f"{run['path']}: no decisions at recorded final tick; missing observations are not filled")
    while current * size < horizon:
        yield current, pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
        current, parts = current + 1, []
