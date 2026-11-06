"""Exact dense labels, immutable case loading and declared support gates."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from experiments.future_resource_io import digest, save_csv, save_json, decisions
from experiments.future_resource_targets import future_events, behavioral, Target, OBSERVED, ADMIN_END, NOT_AT_RISK
from experiments.future_resource_analysis import support as exact_support, integrity
from experiments.dense_resource_models import Encoding, feature_counts, feature_support
from simulation.dense_resources import read_dense, PHASE

ROOT = Path('data/dense_resource_v082')
OUTPUT = Path('analysis/dense_resource_v082')
DESIGN = Path('experiments/dense_resource_design.json')


def scientific_sources():
    names = ['simulation/dense_resources.py', 'experiments/dense_resource_models.py',
             'experiments/dense_resource_data.py', 'experiments/dense_resource_study.py',
             'experiments/dense_resource_analysis.py', 'config.py']
    for folder in ('agents', 'systems', 'world', 'simulation'):
        names.extend(str(p) for p in sorted(Path(folder).glob('*.py')) if str(p) not in names)
    return {p: digest(p) for p in names}


def plan(root=ROOT):
    root = Path(root)
    value = json.loads((root / 'preregistration.json').read_text())
    if digest(DESIGN) != value['design_sha256']:
        raise ValueError('Scientific design changed after freeze')
    for path, sha in value['source_sha256'].items():
        if digest(path) != sha:
            raise ValueError('Frozen scientific source changed: ' + path)
    return value['design']


def case_path(root, phase, ecology, mode, seed):
    return Path(root) / phase / f'{ecology}_{mode}_seed_{seed}.csv.gz.metadata.json'


def cases(root, phase, seeds):
    return [case_path(root, phase, ecology, mode, seed) for ecology in ('E0', 'E1', 'E2', 'E3') for mode in 'ABC' for seed in seeds]


def load_case(path):
    path = Path(path); meta = json.loads(path.read_text())
    if meta.get('version') != '0.8.2' or not meta.get('completed'):
        raise ValueError('Completed dense case required')
    files = {}
    for role, record in meta['files'].items():
        if Path(record['file']).name != record['file']:
            raise ValueError('Invalid case file path')
        p = path.parent / record['file']
        if digest(p) != record['sha256']:
            raise ValueError('Case checksum mismatch: ' + str(p))
        files[role] = p
    return meta, files


def arrays(path):
    meta = json.loads(Path(str(path) + '.json').read_text())
    if meta['phase'] != PHASE:
        raise ValueError('End-of-tick resources required')
    frames = [np.frombuffer(food + water, dtype=np.uint8).reshape(meta['shape']) for tick, food, water in read_dense(path)]
    return np.stack(frames)


def environment_labels(frame, dense, horizon=5):
    if horizon != 5:
        raise ValueError('Frozen environmental horizon is5')
    if frame.sequence.duplicated().any() or frame.duplicated(['agent_id', 'own_index']).any():
        raise ValueError('Duplicate origin decisions')
    if dense.ndim != 4 or dense.shape[1] != 2 or dense.dtype != np.uint8:
        raise ValueError('Exact uint8 tick/resource/y/x arrays required')
    _, _, height, width = dense.shape
    x, y, future = frame.x.to_numpy(), frame.y.to_numpy(), frame.tick.to_numpy() + horizon
    if any(not np.issubdtype(a.dtype, np.integer) for a in (x, y, future)):
        raise ValueError('Integer origin coordinates and ticks required')
    if not ((x >= 0) & (x < width) & (y >= 0) & (y < height)).all() or (frame.tick < 1).any():
        raise ValueError('Invalid origin coordinate/tick')
    valid = future < len(dense)
    endpoint = np.full(len(frame), -1, dtype=np.int16)
    endpoint[valid] = dense[future[valid], 1, y[valid], x[valid]]
    primary = np.full(len(frame), -1, dtype=np.int8)
    primary[valid] = endpoint[valid] > 0
    reason = np.where(valid, OBSERVED, ADMIN_END).astype(np.uint8)
    present = frame.water_here.to_numpy() > 0
    depletion = np.full(len(frame), -1, dtype=np.int8)
    depletion[valid & present] = endpoint[valid & present] == 0
    dep_reason = np.where(~present, NOT_AT_RISK, reason).astype(np.uint8)
    return {'T1': (primary, reason), 'T2': (depletion, dep_reason)}, endpoint


def prepare(path, audit=False):
    path = Path(path); meta, files = load_case(path)
    frame = decisions(files)
    if audit:
        integrity(meta, files, frame)
    dense_path = path.parent / meta['dense']['file']
    dense = arrays(dense_path)
    if len(dense) != meta['tick'] + 1:
        raise ValueError('Incomplete dense tick coverage')
    labels, endpoint = environment_labels(frame, dense)
    # Independent serialized stock accounting, including all intermediate ticks.
    ticks = pd.read_csv(files['ticks'])
    cfg = meta['config']['simulation']
    for i, resource in enumerate(('food', 'water')):
        expected = cfg['initial_' + resource] + ticks[resource + '_regenerated'] - ticks[resource + '_consumed']
        if not np.array_equal(expected, dense[:, i].sum(axis=(1, 2))):
            raise ValueError('Dense/readback mass balance failed: ' + resource)
    label_path = path.with_name(path.name.replace('.csv.gz.metadata.json', '.labels.csv.gz'))
    payload = pd.DataFrame({'sequence': frame.sequence, 'endpoint_water': endpoint})
    for name, (values, reasons) in labels.items():
        payload[name + '_label'] = values; payload[name + '_reason'] = reasons
    if label_path.exists():
        old = pd.read_csv(label_path)
        if not old.equals(payload.astype(old.dtypes.to_dict())):
            raise ValueError('Existing label cache differs')
    else:
        save_csv(label_path, payload)
        save_json(Path(str(label_path) + '.json'), {'schema_version': 1, 'phase': PHASE, 'source_metadata': str(path),
            'acquisitions_sha256': digest(files['acquisitions']), 'dense_sha256': digest(dense_path),
            'sha256': digest(label_path), 'records': len(frame), 'reference': 'frozen pre-action x,y; no future position; t+5 end tick',
            'targets': ['T1 water present', 'T2 water depletion conditional origin present'], 'reason_codes': {'0': 'observed', '2': 'administrative end', '6': 'not at risk'}})
    return meta, files, frame, labels


def patterns(frame, labels):
    f = frame.loc[labels >= 0, ['x_key', 'action', 'b_token', 'rd_token']].copy()
    f['positive'] = labels[labels >= 0]
    return f.groupby(['x_key', 'action', 'b_token', 'rd_token'], sort=True).positive.agg(['sum', 'size']).reset_index().rename(columns={'sum': 'positive', 'size': 'count'})


def combine_patterns(tables):
    return pd.concat(tables, ignore_index=True).groupby(['x_key', 'action', 'b_token', 'rd_token'], sort=True)[['positive', 'count']].sum().reset_index()


def exact_counts(pattern):
    return {name: pattern.groupby(columns)['count'].sum() for name, columns in {
        'M0': ['x_key', 'action'], 'M1': ['x_key', 'action', 'b_token'], 'M2': ['x_key', 'action', 'b_token', 'rd_token']}.items()}


def assess(frame, values, reasons, training_pattern, rules):
    exact, cells, histogram = exact_support(frame, values, reasons, exact_counts(training_pattern), rules['thresholds'])
    # Feature-level support is a separate qualification, never a replacement
    # silently attributed to exact cells.
    observed = frame.loc[values >= 0]
    encoder = Encoding(2)
    training_indices = encoder.transform(training_pattern)
    level_counts = feature_counts(training_indices, encoder.dimension, training_pattern['count'].to_numpy())
    indices = encoder.transform(observed)
    level = {}
    for name, columns in [('R0', 13), ('R1', 14), ('R2', 15)]:
        mask = feature_support(indices[:, :columns], level_counts)
        level[name + '_feature_n'] = int(mask.sum())
        level[name + '_feature_coverage'] = float(mask.mean()) if len(mask) else 0.
    noncell_failures = [s for s in exact['failed_rules'].split(',') if s and s not in ('M01_common', 'social_variation')]
    support01 = feature_support(indices[:, :14], level_counts)
    support012 = feature_support(indices, level_counts)
    level['R_supported_B_levels'] = int(observed.loc[support01, 'b_token'].nunique())
    level['R_supported_RD_levels'] = int(observed.loc[support012, 'rd_token'].nunique())
    limit = rules['regularized_support']['minimum_supported_row_fraction']
    level['R1_eligible'] = not noncell_failures and level['R1_feature_coverage'] >= limit and level['R_supported_B_levels'] >= 2
    level['R2_eligible'] = level['R1_eligible'] and level['R2_feature_coverage'] >= limit and level['R_supported_RD_levels'] >= 2
    return {**exact, **level}, cells


def behavior_reference(frame, end_tick):
    events = future_events(frame, end_tick)
    return behavioral(frame, events, Target('B10', 'T5', 'behavioral', 'either', 10))
