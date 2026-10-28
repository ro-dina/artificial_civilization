"""Completed Vocal Q recordings: strict identities, hashes and streaming chunks."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from config import Config
from simulation.vocal_learning_observer import SCHEMAS, condition, hash_file


def discover(directory, pattern='*.csv*'):
    runs, identities = [], set()
    for sidecar in sorted(Path(directory).glob(pattern + '.metadata.json')):
        meta = json.loads(sidecar.read_text())
        if meta.get('version') != '0.6.2':
            continue
        if meta.get('schema_version') != 1 or not meta.get('completed'):
            raise ValueError(f'{sidecar}: requires a completed diagnostic recording, schema 1')
        if meta.get('schemas') != {k: list(v) for k, v in SCHEMAS.items()}:
            raise ValueError(f'{sidecar}: incompatible diagnostic schemas')
        cfg = Config(**meta['config'])
        config_id = hashlib.sha256(json.dumps(meta['config'], sort_keys=True).encode()).hexdigest()
        if meta['config_id'] != config_id or meta['seed'] != cfg.random_seed:
            raise ValueError(f'{sidecar}: inconsistent seed/config hash')
        if type(meta.get('final_tick')) is not int or not 0 <= meta['final_tick'] <= cfg.max_ticks:
            raise ValueError(f'{sidecar}: invalid final tick')
        ecology = {k: v for k, v in meta['config'].items() if k not in
                   {'random_seed', 'max_ticks', 'status_interval', 'learning_uses_auditory', 'learning_controls_vocalization'}}
        ecology_id = hashlib.sha256(json.dumps(ecology, sort_keys=True).encode()).hexdigest()[:12]
        identity = (ecology_id, condition(cfg), cfg.random_seed)
        if identity in identities:
            raise ValueError('Duplicate ecology/condition/seed; cannot double-weight runs')
        identities.add(identity)
        paths = {}
        for kind in SCHEMAS:
            info = meta['files'][kind]
            name = info['file']
            if Path(name).name != name or not name:
                raise ValueError(f'{sidecar}: unsafe file name')
            path = sidecar.parent / name
            if hash_file(path) != info['sha256'] or path.stat().st_size != info['bytes']:
                raise ValueError(f'{path}: hash/size mismatch; incomplete or modified recording')
            if tuple(pd.read_csv(path, nrows=0).columns) != SCHEMAS[kind]:
                raise ValueError(f'{path}: column mismatch')
            paths[kind] = path
        runs.append({'metadata': meta, 'metadata_path': sidecar, 'paths': paths,
                     'run': paths['decisions'].name.split('.csv')[0], 'condition': condition(cfg),
                     'seed': cfg.random_seed, 'ecology_id': ecology_id,
                     'applicable': cfg.brain == 'learning' and cfg.learning_controls_vocalization})
    if not runs:
        raise ValueError(f'No completed v0.6.2 diagnostic sidecars in {directory}')
    if len({r['ecology_id'] for r in runs}) > 1:
        for r in runs:
            r['condition'] += ':' + r['ecology_id'][:6]
    return sorted(runs, key=lambda r: (r['condition'], r['seed'], r['run']))


def read_chunks(run, kind, chunksize=40000):
    count, last_tick, last_ids = 0, 0, set()
    catalog = run.get('catalog', set())
    with pd.read_csv(run['paths'][kind], chunksize=chunksize, float_precision='round_trip') as reader:
        for frame in reader:
            if frame.empty:
                continue
            count += len(frame)
            if kind == 'states':
                if frame.state_id.duplicated().any():
                    raise ValueError('Duplicate catalog ID')
                yield frame
                continue
            if frame[['tick', 'agent_id']].isna().any().any() or not frame.tick.is_monotonic_increasing:
                raise ValueError(f'{kind}: missing identity or unordered tick')
            if (frame.tick < 1).any() or (frame.tick > run['metadata']['final_tick']).any() or frame.tick.iloc[0] < last_tick:
                raise ValueError(f'{kind}: tick outside completed horizon or out of order')
            if not frame.state_id.dropna().isin(catalog).all():
                raise ValueError(f'{kind}: state not in exact key catalog')
            if kind == 'decisions':
                if frame.duplicated(['tick', 'agent_id']).any():
                    raise ValueError('Duplicate decision')
                first = int(frame.tick.iloc[0])
                ids = set(frame.loc[frame.tick.eq(first), 'agent_id'])
                if first == last_tick and ids & last_ids:
                    raise ValueError('Duplicate decision across chunks')
                final_ids = set(frame.loc[frame.tick.eq(frame.tick.iloc[-1]), 'agent_id'])
                last_ids = last_ids | final_ids if first == last_tick == frame.tick.iloc[-1] else final_ids
                original_condition = condition(Config(**run['metadata']['config']))
                if not frame.condition.eq(original_condition).all():
                    raise ValueError('Condition differs from sidecar')
                numeric = list(SCHEMAS['decisions'][5:])
                numeric.remove('novelty')
                if run['applicable']:
                    if frame[numeric + ['novelty']].isna().any().any():
                        raise ValueError('Missing applicable Vocal Q diagnostics')
                    n = run['metadata']['config']['signal_vocab_size'] + 1
                    if (not frame.n_actions.eq(n).all() or not frame.max_ties.between(1, n).all()
                            or not frame.selected_action.between(0, n-1).all()
                            or not frame.p_silence.between(0, 1).all()
                            or not frame.novelty.isin(['first', 'stored', 'recreated']).all()):
                        raise ValueError('Invalid Vocal Q choice metrics')
                    numbers = frame[numeric].to_numpy(float)
                    if not np.isfinite(numbers).all() or not frame.rows.between(1,run['metadata']['config']['vocal_learning_memory_capacity']).all():
                        raise ValueError('Nonfinite diagnostics or Q capacity exceeded')
                    flags = ['present_at_start','present_before_selection','newly_allocated','all_equal',
                             'silence_is_max','exploratory','at_capacity']
                    if not frame[flags].isin([0,1]).all().all():
                        raise ValueError('Invalid decision flags')
                    expected = frame.epsilon/n + (1-frame.epsilon)*frame.silence_is_max/frame.max_ties
                    if not np.allclose(frame.p_silence,expected,rtol=0,atol=1e-15):
                        raise ValueError('Incorrect implied SILENCE probability')
                elif frame[list(SCHEMAS['decisions'][5:])].notna().any().any():
                    raise ValueError('A/B/random metrics must be N/A')
            elif kind == 'updates':
                if frame.isna().any().any() or not np.isfinite(frame.to_numpy(float)).all():
                    raise ValueError('Incomplete/nonfinite update event')
                if not frame[['terminal','recreated']].isin([0,1]).all().all():
                    raise ValueError('Invalid update flag')
            elif kind == 'rows':
                active = frame.event.isin(['allocate','evict'])
                if not frame.state_id.notna().eq(active).all():
                    raise ValueError('Row event state ID must match event type')
            last_tick = int(frame.tick.iloc[-1])
            yield frame
    if count != run['metadata']['files'][kind]['records']:
        raise ValueError(f'{kind}: row count differs from completed metadata')
