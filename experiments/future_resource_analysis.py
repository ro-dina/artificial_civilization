"""Archive feasibility first, frozen selection second; never starts a simulation."""
from collections import Counter
from pathlib import Path
from time import perf_counter
import argparse
import json

import numpy as np
import pandas as pd

from experiments.future_resource_io import (ROOT, OUTPUT, ARCHIVE, DESIGN, archive_case, case_paths, decisions,
    design, digest, feature_counts, lookup_counts, missing_fields, receipt, save_csv, save_json, sources, training_paths, verify_hashes, FEATURES)
from experiments.future_resource_targets import (definitions, target_table, snapshots, environmental, future_events,
    behavioral, competing_curve, REASONS, OBSERVED, NO_SNAPSHOT, ADMIN_END, HISTORY_GAP, DEATH, NO_REFERENCE, NOT_AT_RISK)
from experiments.resource_ecology_experiments import peak


def tags(meta):
    return {k: meta[k] for k in ('ecology', 'mode', 'seed')}


def stream_semantics(role):
    return {
        'acquisitions': 'Ordered actual decisions: sequential pre-action positions/quantities/social tokens; actual post-action consumption, physiology, terminal. *_tickstart distances frozen AFTER regeneration, before actors.',
        'outcomes': 'Same ordered decision key, actual existing reward and terminal outcome; no exact next world.',
        'spatial': 'Full integer food/water units at END tick0/100/200/300/400/500; NOT tick-start, NOT every tick.',
        'individuals': 'One life row per founder/child, death or administrative censor; duration age, birth tick; never a complete world event log.',
        'states': 'Original X catalog, including old auditory feature; catalog IDs local to run, canonical key joins across seeds.',
        'resource_ticks': 'End-tick aggregate stocks and regeneration budgets; distances summarize tick-start maps, NOT exact maps.',
        'actions': 'Grouped actions/contexts, not an ordered per-decision history.',
        'ticks': 'End-tick cumulative aggregate statistics, no individual resource events.'
    }.get(role, 'Grouped optional learning/exposure diagnostics; not an exact resource event stream.')


def integrity(meta, files, frame):
    n = len(frame)
    if not np.array_equal(frame.sequence, np.arange(n)) or frame.duplicated(['agent_id', 'own_index']).any():
        raise ValueError('Duplicate/missing decision identifiers')
    if not frame.tick.is_monotonic_increasing:
        raise ValueError('Invalid global tick order')
    if not np.array_equal(frame.tick, frame.birth_tick + frame.own_index + 1) or not np.array_equal(frame.age, frame.own_index):
        raise ValueError('Invalid birth/own-decision alignment')
    grouped = frame.groupby('agent_id', sort=False)
    if grouped.own_index.min().ne(0).any() or not (grouped.own_index.max() + 1).equals(grouped.size()):
        raise ValueError('Missing agent history')
    if frame.groupby('agent_id').terminal.sum().gt(1).any():
        raise ValueError('Multiple terminal rows')
    last = grouped.tail(1).set_index('agent_id')
    if int(frame.terminal.sum()) != int(last.terminal.sum()):
        raise ValueError('Invalid terminal transition')
    for resource, action in (('food', 4), ('water', 5)):
        units = frame[resource + '_consumed']
        oracle = (frame.action.eq(action) & frame[resource + '_here'].gt(0)).astype(int)
        if not units.isin([0, 1]).all() or not units.equals(oracle):
            raise ValueError('Impossible actual consumption')
        if int(units.sum()) != meta['statistics'][resource + '_consumed']:
            raise ValueError('Resource consumption aggregate mismatch')
    if not frame.acquired.equals((frame.food_consumed + frame.water_consumed).astype(int)):
        raise ValueError('Invalid acquisition flag')
    consume = frame.action.isin([4, 5])
    if not np.array_equal(frame.loc[consume, 'action_succeeded'], frame.loc[consume, 'acquired']):
        raise ValueError('Failed consumption mislabeled')
    life = pd.read_csv(files['individuals']).set_index('agent_id')
    if life.index.duplicated().any() or len(life) != meta['config']['simulation']['initial_population'] + meta['statistics']['births']:
        raise ValueError('Missing/duplicate individual lifetime')
    if not life.opportunities.equals(grouped.size().reindex(life.index, fill_value=0)):
        raise ValueError('Individual opportunities mismatch')
    linked = life.loc[last.index]
    if not np.array_equal(last.terminal, 1 - linked.censored):
        raise ValueError('Death/censor mismatch')
    if not np.array_equal(last.tick, linked.birth_tick + linked.duration):
        raise ValueError('Death/end follow-up alignment mismatch')
    if not last.loc[linked.censored.eq(1), 'tick'].eq(meta['tick']).all():
        raise ValueError('Incomplete administrative follow-up')
    for resource in ('food', 'water'):
        if not np.array_equal(life[resource], grouped[resource + '_consumed'].sum().reindex(life.index, fill_value=0)):
            raise ValueError('Individual resource accounting mismatch')
    outcomes = pd.read_csv(files['outcomes'], usecols=['sequence', 'tick', 'agent_id', 'base_id', 'b_token', 'c_token', 'action', 'reward', 'terminal'], float_precision='round_trip')
    for col in outcomes:
        if not np.array_equal(outcomes[col], frame[col]):
            raise ValueError('Ordered outcome linkage mismatch: ' + col)
    ticks = pd.read_csv(files['ticks'])
    resource_ticks = pd.read_csv(files['resource_ticks'])
    spatial = pd.read_csv(files['spatial'])
    cfg = meta['config']['simulation']
    grids = snapshots(spatial, cfg['world_width'], cfg['world_height'])
    for resource in ('food', 'water'):
        expected = cfg['initial_' + resource] + ticks[resource + '_regenerated'] - ticks[resource + '_consumed']
        if not np.array_equal(expected, resource_ticks[resource + '_stock']):
            raise ValueError('Resource mass balance mismatch')
        for tick, grid in grids.items():
            quantity = grid[resource]
            if (quantity < 0).any() or (quantity > cfg[resource + '_capacity']).any() or int(quantity.sum()) != int(resource_ticks.loc[resource_ticks.tick.eq(tick), resource + '_stock'].iloc[0]):
                raise ValueError('Snapshot quantity/accounting mismatch')
    # Independent current X/action sufficiency oracle, not a future predictor.
    catalog = {key: json.loads(key) for key in frame.x_key.unique()}
    here = np.array([int(a == 4 and catalog[x][3:5] == [0, 0] or a == 5 and catalog[x][5:7] == [0, 0]) for x, a in zip(frame.x_key, frame.action)])
    if not np.array_equal(here, frame.acquired):
        raise ValueError('Original X/action immediate-acquisition oracle failed')
    return {'decisions': n, 'agents_with_decisions': frame.agent_id.nunique(), 'individuals': len(life),
            'terminal_decisions': int(frame.terminal.sum()), 'censored_individuals': int(life.censored.sum()),
            'snapshot_ticks': json.dumps(sorted(grids)), 'all_checks_passed': True}


def audit(archive=ARCHIVE, root=ROOT, output=OUTPUT):
    design(root)
    out = Path(output) / 'audit'
    if out.exists():
        raise ValueError('Refusing completed audit overwrite')
    start = perf_counter()
    inventory, checks, inputs = [], [], {}
    for path in case_paths(archive):
        meta, files = archive_case(path)
        inputs[str(path)] = digest(path)
        if meta['tick'] != 500 or meta['schema_version'] != 2:
            raise ValueError('Expected complete500 schema2 archive')
        for role, p in files.items():
            columns = list(pd.read_csv(p, nrows=0).columns)
            absent = missing_fields(columns, meta['schemas'][role])
            if absent:
                raise ValueError('Missing recorded schema fields: ' + str(p) + ' ' + str(absent))
            total = 0
            ticks, identities = [], set()
            nulls = Counter()
            for chunk in pd.read_csv(p, chunksize=50000):
                total += len(chunk)
                nulls.update({k: int(v) for k, v in chunk.isna().sum().items()})
                if 'tick' in chunk and len(chunk):
                    ticks.extend([int(chunk.tick.min()), int(chunk.tick.max())])
                if 'agent_id' in chunk:
                    identities.update(int(x) for x in chunk.agent_id.dropna().unique())
            if total != meta['files'][role]['records']:
                raise ValueError('Record count mismatch: ' + str(p))
            inputs[str(p)] = digest(p)
            inventory.append({**tags(meta), 'role': role, 'filename': str(p), 'schema_version': meta['schema_version'],
                'records': total, 'bytes': p.stat().st_size, 'tick_min': min(ticks) if ticks else None, 'tick_max': max(ticks) if ticks else None,
                'agents': len(identities) if 'agent_id' in columns else None, 'columns_json': json.dumps(columns), 'missing_schema_fields': json.dumps(absent),
                'null_counts_json': json.dumps(dict(nulls), sort_keys=True), 'identifiers': json.dumps([c for c in ('sequence', 'agent_id', 'base_id', 'x', 'y') if c in columns]),
                'timestamps': json.dumps([c for c in ('tick', 'own_index', 'birth_tick', 'duration') if c in columns]),
                'resource_units': 'integer consumable tile units' if role in ('acquisitions', 'spatial', 'individuals', 'resource_ticks', 'ticks') else 'not exact per-tile quantities',
                'social_representation': 'B visible token + C pair token (RD decoded); no memory identities as features' if role in ('acquisitions', 'outcomes', 'actions') else 'see columns',
                'temporal_semantics': stream_semantics(role), 'future_link': 'agent/own_index for behavioral; fixed coordinate/saved end tick for environmental' if role == 'acquisitions' else 'not standalone future-label source'})
        frame = decisions(files)
        checks.append({**tags(meta), **integrity(meta, files, frame)})
        print('Audited', meta['ecology'], meta['mode'], meta['seed'], len(frame), flush=True)
    save_csv(out / 'field_inventory.csv', pd.DataFrame(inventory))
    save_csv(out / 'integrity.csv', pd.DataFrame(checks))
    save_csv(out / 'target_definitions.csv', target_table())
    save_json(out / 'receipt.json', {'stage': 'archive audit; previously examined seeds', 'inputs': inputs, 'source_sha256': sources(),
        'wall_seconds': perf_counter() - start, 'peak_memory_bytes': peak(), 'cases': 36,
        'input_bytes': sum(Path(p).stat().st_size for p in inputs), 'no_simulations': True})


def labels_for(frame, grids, events, target, end_tick):
    return environmental(frame, grids, target, end_tick) if target.kind == 'environmental' else behavioral(frame, events, target)


def load_labels(path):
    meta, files = archive_case(path)
    frame = decisions(files)
    cfg = meta['config']['simulation']
    grids = snapshots(pd.read_csv(files['spatial']), cfg['world_width'], cfg['world_height'])
    return meta, files, frame, grids, future_events(frame, meta['tick'])


def effective(values):
    counts = pd.Series(values).value_counts().to_numpy(dtype=float)
    if not len(counts):
        return 0.0
    p = counts / counts.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def support(frame, labels, reasons, counts, thresholds):
    valid = labels >= 0
    observed = frame.loc[valid].copy()
    observed['label'] = labels[valid]
    n = len(observed)
    positive = int(labels[valid].sum())
    reference_n = int((~np.isin(reasons, [NO_REFERENCE, NOT_AT_RISK])).sum())
    train = lookup_counts(observed, counts)
    minimum = thresholds['minimum_training_cell']
    common01 = (train['M0'] >= minimum) & (train['M1'] >= minimum)
    common012 = common01 & (train['M2'] >= minimum)
    baseline = observed.groupby(FEATURES['M0']).label.agg(['sum', 'size'])
    baseline['negative'] = baseline['size'] - baseline['sum']
    mixed = (baseline['sum'] >= thresholds['minimum_class_in_mixed_stratum']) & (baseline.negative >= thresholds['minimum_class_in_mixed_stratum'])
    result = {'source_decisions': len(frame), 'reference_observations': reference_n, 'observed': n, 'positive': positive, 'negative': n - positive,
        'prevalence': positive / n if n else None, 'label_coverage': n / reference_n if reference_n else 0.0,
        'agents': observed.agent_id.nunique(), 'original_states': observed.x_key.nunique(), 'social_tokens': observed.b_token.nunique(),
        'memory_tokens': observed.rd_token.nunique(), 'social_effective_categories': effective(observed.b_token), 'memory_effective_categories': effective(observed.rd_token),
        'adequately_mixed_baseline_strata': int(mixed.sum()), 'mixed_baseline_decisions': int(baseline.loc[mixed, 'size'].sum()),
        'deterministic_in_observed_XA_strata': bool(n and not ((baseline['sum'] > 0) & (baseline.negative > 0)).any()),
        'M01_common_n': int(common01.sum()), 'M01_common_coverage': float(common01.mean()) if n else 0.,
        'M012_common_n': int(common012.sum()), 'M012_common_coverage': float(common012.mean()) if n else 0.,
        'supported_social_tokens': observed.loc[common01, 'b_token'].nunique(), 'supported_memory_tokens': observed.loc[common012, 'rd_token'].nunique()}
    result.update({reason: int((reasons == code).sum()) for code, reason in REASONS.items() if code != OBSERVED})
    failures = []
    tests = {
        'prevalence': n > 0 and thresholds['prevalence_min'] <= positive / n <= thresholds['prevalence_max'],
        'class_counts': min(positive, n - positive) >= thresholds['minimum_each_class'],
        'agents': result['agents'] >= thresholds['minimum_agents'],
        'label_coverage': result['label_coverage'] >= thresholds['minimum_label_coverage'],
        'conditional_variation': result['adequately_mixed_baseline_strata'] >= thresholds['minimum_mixed_baseline_strata'],
        'M01_common': result['M01_common_coverage'] >= thresholds['minimum_common_coverage'],
        'social_variation': result['supported_social_tokens'] >= thresholds['minimum_supported_social_tokens']}
    failures = [key for key, passed in tests.items() if not passed]
    result['eligible'] = not failures
    result['memory_eligible'] = not failures and result['M012_common_coverage'] >= thresholds['minimum_common_coverage'] and result['supported_memory_tokens'] >= thresholds['minimum_supported_memory_tokens']
    result['classification'] = 'unavailable' if not n else 'degenerate' if not tests['prevalence'] or not tests['class_counts'] else 'supported' if not failures else 'exploratory_only'
    result['failed_rules'] = ','.join(failures)
    cells = []
    for name, columns in FEATURES.items():
        grouped = observed.groupby(columns).label.agg(['sum', 'size'])
        cells.append({'model': name, 'test_cells': len(grouped), 'test_decisions': n, 'train_cells': len(counts[name]),
            'train_supported_cells': int(counts[name].ge(minimum).sum()), 'supported_test_decisions': int((train[name] >= minimum).sum()),
            'two_class_cells': int(((grouped['sum'] > 0) & (grouped['sum'] < grouped['size'])).sum()),
            'adequately_two_class_cells': int(((grouped['sum'] >= thresholds['minimum_class_in_mixed_stratum']) & (grouped['size'] - grouped['sum'] >= thresholds['minimum_class_in_mixed_stratum'])).sum())})
    histogram = baseline.groupby(['sum', 'negative']).size().reset_index(name='strata').rename(columns={'sum': 'positive'})
    return result, cells, histogram


def feasibility(archive=ARCHIVE, root=ROOT, output=OUTPUT):
    rules = design(root)
    audit_receipt = receipt(Path(output) / 'audit/receipt.json', root)
    out = Path(output) / 'feasibility'
    if out.exists():
        raise ValueError('Refusing feasibility overwrite')
    start = perf_counter()
    inputs = dict(audit_receipt['inputs'])
    rows, cells, histograms, events_out, curves = [], [], [], [], []
    binary = [t for t in definitions() if t.kind != 'time_to_event']
    for ecology in rules['ecologies']:
        training = {}
        for path in training_paths(archive, ecology):
            meta, files, frame, grids, events = load_labels(path)
            inputs[str(path)] = digest(path)
            inputs.update({str(p): digest(p) for p in files.values()})
            integrity(meta, files, frame)
            for target in binary:
                labels, reasons = labels_for(frame, grids, events, target, meta['tick'])
                counts = feature_counts(frame, labels)
                if target.name not in training:
                    training[target.name] = counts
                else:
                    training[target.name] = {name: training[target.name][name].add(v, fill_value=0).astype(int) for name, v in counts.items()}
        for path in [p for p in case_paths(archive) if p.name.startswith(ecology + '_')]:
            meta, files, frame, grids, events = load_labels(path)
            for target in binary:
                labels, reasons = labels_for(frame, grids, events, target, meta['tick'])
                result, conditional, histogram = support(frame, labels, reasons, training[target.name], rules['thresholds'])
                tag = {**tags(meta), 'target': target.name, 'kind': target.kind, 'resource': target.resource, 'horizon': target.horizon, 'radius': target.radius}
                rows.append({**tag, **result})
                cells.extend({**tag, **r} for r in conditional)
                histograms.append(histogram.assign(**tag))
            for resource, (elapsed, event) in events.items():
                tag = {**tags(meta), 'target': 'T4_' + resource, 'kind': 'time_to_event', 'resource': resource}
                events_out.append({**tag, 'origins': len(frame), 'agents': frame.agent_id.nunique(),
                    'acquisitions': int((event == 1).sum()), 'competing_deaths': int((event == 2).sum()),
                    'administrative_censors': int((event == 3).sum()), 'missing_history': int((event == 4).sum()),
                    'mean_observed_elapsed': float(elapsed.mean()), 'interpretation': 'overlapping origins, competing risks; not independent individuals or binary negatives'})
                curves.extend({**tag, **r} for r in competing_curve(elapsed, event))
            print('Feasibility', ecology, meta['mode'], meta['seed'], flush=True)
    f = pd.DataFrame(rows)
    save_csv(out / 'target_support.csv', f)
    save_csv(out / 'conditional_feature_support.csv', pd.DataFrame(cells))
    save_csv(out / 'conditional_outcome_histogram.csv.gz', pd.concat(histograms, ignore_index=True))
    save_csv(out / 'competing_events.csv', pd.DataFrame(events_out))
    save_csv(out / 'competing_curves.csv', pd.DataFrame(curves))
    save_csv(out / 'censoring.csv', f[['ecology', 'mode', 'seed', 'target', 'kind', 'source_decisions', 'reference_observations', 'observed', 'missing_snapshot', 'administrative_end', 'missing_history', 'death_competing', 'no_reference', 'not_at_risk']])
    save_csv(out / 'reconstructability.csv', f[['ecology', 'mode', 'seed', 'target', 'kind', 'source_decisions', 'observed', 'label_coverage', 'classification', 'failed_rules']])
    save_json(out / 'receipt.json', {'stage': 'descriptive feasibility; no model losses evaluated', 'inputs': inputs, 'source_sha256': sources(),
        'wall_seconds': perf_counter() - start, 'peak_memory_bytes': peak(), 'candidate_binary_variants': len(binary), 'time_to_event_variants': 3,
        'no_simulations': True, 'design_sha256': digest(DESIGN)})


def choose_target(table, priorities):
    for target in priorities:
        f = table[table.target.eq(target)]
        for ecology, group in f.groupby('ecology', sort=True):
            if len(group) == 9 and group.eligible.all():
                return target
    return None


def freeze(archive=ARCHIVE, root=ROOT, output=OUTPUT):
    rules = design(root)
    evidence = Path(output) / 'feasibility'
    previous = receipt(evidence / 'receipt.json', root)
    support_table = pd.read_csv(evidence / 'target_support.csv')
    primary = choose_target(support_table, rules['selection']['environment_priority'])
    secondary = choose_target(support_table, rules['selection']['behavior_priority'])
    inputs = {**previous['inputs'], str(evidence / 'target_support.csv'): digest(evidence / 'target_support.csv'), str(evidence / 'receipt.json'): digest(evidence / 'receipt.json')}
    value = {'version': '0.8.1', 'stage': 'frozen AFTER archived feasibility, BEFORE retrospective losses; not confirmation preregistration',
        'primary_environment': primary, 'secondary_behavior': secondary,
        'retrospective_environment': primary or rules['selection']['environment_priority'][0],
        'retrospective_behavior': secondary or rules['selection']['behavior_priority'][0],
        'fallbacks_are_exploratory_only': True, 'thresholds': rules['thresholds'], 'model': rules['models'],
        'train_seeds': rules['train_seeds'], 'evaluation_seeds': rules['test_seeds'], 'train_policy': rules['train_policy'], 'test_policy': rules['test_policy'],
        'confirmation': {'executed': False, 'reason': 'No qualifying environmental target, so no new-seed confirmation justified.' if primary is None else 'Archive selection remains retrospective; any confirmation requires a separate frozen prospective new-seed plan.'},
        'inputs': inputs, 'source_sha256': sources(), 'no_simulations': True}
    save_json(Path(root) / 'frozen_evaluation.json', value)
    print('FROZEN: environmental=', primary, 'behavioral=', secondary, 'no new simulations', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('audit', 'feasibility', 'freeze'))
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = vars(parser.parse_args())
    phase = args.pop('phase')
    try:
        globals()[phase](**args)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
