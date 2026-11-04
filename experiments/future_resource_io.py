"""Exclusive outputs and immutable archive/protocol receipts for v081."""
from hashlib import sha256
from pathlib import Path
import gzip
import io
import json

import pandas as pd

ROOT = Path('data/future_resource_v081')
OUTPUT = Path('analysis/future_resource_v081')
ARCHIVE = Path('data/resource_ecology_v080')
DESIGN = Path('experiments/future_resource_design.json')


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sources():
    return {str(p): digest(p) for p in sorted(Path('experiments').glob('future_resource*')) if p.suffix in ('.py', '.json')}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def save_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.gz':
        with path.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped, io.TextIOWrapper(zipped, newline='') as stream:
            frame.to_csv(stream, index=False, float_format='%.12g', lineterminator='\n')
    else:
        with path.open('x', newline='') as stream:
            frame.to_csv(stream, index=False, float_format='%.12g', lineterminator='\n')


def verify_hashes(items):
    for path, expected in items.items():
        if digest(path) != expected:
            raise ValueError('Frozen input/source changed: ' + path)


def design(root=ROOT):
    receipt = json.loads((Path(root) / 'design_receipt.json').read_text())
    if digest(DESIGN) != receipt['design_sha256']:
        raise ValueError('Design changed after rules receipt')
    return receipt['design']


def receipt(path, root=ROOT):
    value = json.loads(Path(path).read_text())
    design(root)
    verify_hashes(value['inputs'])
    expected = dict(value['source_sha256'])
    correction_path = Path(root) / 'reporting_correction.json'
    if correction_path.exists():
        correction = json.loads(correction_path.read_text())
        if digest(Path(root) / 'frozen_evaluation.json') != correction['original_frozen_plan_sha256']:
            raise ValueError('Original frozen decisions changed after reporting correction')
        for source, record in correction['sources'].items():
            if expected.get(source) == record['original_sha256']:
                expected[source] = record['corrected_sha256']
    verify_hashes(expected)
    return value


def archive_case(path):
    path = Path(path)
    meta = json.loads(path.read_text())
    if meta.get('version') != '0.8.0' or not meta.get('completed'):
        raise ValueError('Completed v080 case required')
    files = {}
    for role, record in meta['files'].items():
        if Path(record['file']).name != record['file']:
            raise ValueError('Invalid archive stream path')
        p = path.parent / record['file']
        if digest(p) != record['sha256']:
            raise ValueError('Archive checksum mismatch: ' + str(p))
        files[role] = p
    return meta, files


def missing_fields(columns, required):
    return sorted(set(required) - set(columns))


def decisions(files):
    frame = pd.read_csv(files['acquisitions'], float_precision='round_trip')
    required = ('sequence', 'tick', 'agent_id', 'own_index', 'base_id', 'b_token', 'c_token', 'action',
                'x', 'y', 'food_here', 'water_here', 'food_consumed', 'water_consumed', 'acquired', 'terminal')
    absent = missing_fields(frame.columns, required)
    if absent:
        raise ValueError('Unavailable target fields: ' + ', '.join(absent))
    catalog = pd.read_csv(files['states'])
    if catalog.base_id.duplicated().any():
        raise ValueError('Duplicate original-state catalog ID')
    mapping = catalog.set_index('base_id').key_json
    frame['x_key'] = frame.base_id.map(mapping)
    if frame.x_key.isna().any():
        raise ValueError('Missing original-state catalog entry')
    from simulation.controls_observer import C_PAIRS
    if not frame.c_token.between(0, len(C_PAIRS) - 1).all():
        raise ValueError('Invalid memory token')
    pairs = [C_PAIRS[int(v)] for v in frame.c_token]
    if any(b != int(row) for (b, rd), row in zip(pairs, frame.b_token)):
        raise ValueError('Visible/memory token mismatch')
    frame['rd_token'] = [rd for b, rd in pairs]
    return frame


def training_paths(archive, ecology):
    from experiments.resource_ecology_prediction import training_path
    frozen = json.loads((Path(archive) / 'frozen.json').read_text())
    return [training_path(archive, frozen, ecology, seed) for seed in (3, 4)]


def case_paths(archive=ARCHIVE):
    return [Path(archive) / 'primary' / f'{e}_{mode}_seed_{seed}.csv.gz.metadata.json'
            for e in ('E0', 'E1', 'E2', 'E3') for mode in 'ABC' for seed in (0, 1, 2)]


FEATURES = {'M0': ['x_key', 'action'], 'M1': ['x_key', 'action', 'b_token'],
            'M2': ['x_key', 'action', 'b_token', 'rd_token']}


def feature_counts(frame, labels):
    observed = frame.loc[labels >= 0]
    return {name: observed.groupby(columns, sort=False).size() for name, columns in FEATURES.items()}


def lookup_counts(frame, counts):
    return {name: cells.reindex(pd.MultiIndex.from_frame(frame[FEATURES[name]]), fill_value=0).to_numpy()
            for name, cells in counts.items()}
