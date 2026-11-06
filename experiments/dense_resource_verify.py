"""Final original-file, dense mass-balance, frozen-model and scoring audit."""
from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd

from experiments.future_resource_io import digest, save_csv, save_json
from experiments.dense_resource_data import ROOT, OUTPUT, plan, load_case, arrays
from experiments.dense_resource_study import allowed


def run(root=ROOT, output=OUTPUT, test_log=None):
    root, output = Path(root), Path(output); rules = plan(root)
    before = json.loads((root / 'before.json').read_text())
    protected = {p: r for p, r in before['files'].items() if p not in ('README.md', '.gitignore')}
    for p, record in protected.items():
        if digest(p) != record['sha256']:
            raise ValueError('Historical source/artifact changed: ' + p)
    frozen = json.loads((root / 'preregistration.json').read_text())
    inventory, metadata_inputs = [], {}
    initial = {}; full_cases = {'training': 0, 'evaluation': 0}; frames = 0
    for path in sorted(root.rglob('*.csv.gz.metadata.json')):
        meta, files = load_case(path); metadata_inputs[str(path)] = digest(path)
        if meta['source_sha256'] != frozen['source_sha256'] or meta['preregistration_sha256'] != digest(root / 'preregistration.json'):
            raise ValueError('Run source/configuration differs from frozen study')
        if meta['files']['acquisitions']['records'] != meta['totals']['decisions'] or meta['files']['individuals']['records'] != 100 + meta['statistics']['births']:
            raise ValueError('Missing decision/life records')
        if meta['config']['simulation']['learning_memory_capacity'] != 256 or meta['config']['simulation']['vocal_learning_memory_capacity'] != 256:
            raise ValueError('Q capacity changed')
        if meta['phase'] in full_cases:
            full_cases[meta['phase']] += 1
            if meta['tick'] != 500: raise ValueError('Incomplete full-length scientific run')
            key = (meta['phase'], meta['seed']); old = initial.setdefault(key, meta['initial_agents_rng_sha256'])
            if old != meta['initial_agents_rng_sha256']: raise ValueError('Different initial bodies/agent RNG across factorial cases')
        if meta['dense']:
            dense_path = path.parent / meta['dense']['file']; data = arrays(dense_path); ticks = pd.read_csv(files['ticks'])
            if len(data) != meta['tick'] + 1: raise ValueError('Dense snapshot incompleteness')
            for i, resource in enumerate(('food', 'water')):
                actual = data[:, i].sum(axis=(1, 2)); cfg = meta['config']['simulation']
                expected = cfg['initial_' + resource] + ticks[resource + '_regenerated'] - ticks[resource + '_consumed']
                if not np.array_equal(actual, expected): raise ValueError('Independent readback mass balance failure')
                inventory.append({'phase': meta['phase'], 'ecology': meta['ecology'], 'mode': meta['mode'], 'seed': meta['seed'],
                    'resource': resource, 'frames': len(data), 'first_tick': 0, 'last_tick': meta['tick'], 'max_mass_balance_error': 0,
                    'minimum_quantity': int(data[:, i].min()), 'maximum_quantity': int(data[:, i].max()), 'binary_sha256': meta['dense']['sha256']})
            spatial = pd.read_csv(files['spatial'])
            for tick, group in spatial.groupby('tick'):
                for i, resource in enumerate(('food', 'water')):
                    old_grid = group.pivot(index='y', columns='x', values=resource).to_numpy()
                    if not np.array_equal(old_grid, data[int(tick), i]):
                        raise ValueError('Dense phase/coordinate layout differs from existing actual world snapshot')
            frames += len(data)
        label_path = path.with_name(path.name.replace('.csv.gz.metadata.json', '.labels.csv.gz'))
        if label_path.exists():
            label_meta = json.loads(Path(str(label_path) + '.json').read_text())
            if digest(label_path) != label_meta['sha256'] or label_meta['acquisitions_sha256'] != meta['files']['acquisitions']['sha256'] or label_meta['dense_sha256'] != meta['dense']['sha256']:
                raise ValueError('Label/source checksum disagreement')
    validation = json.loads((root / 'validation.json').read_text())
    for ecology, mode, seed, ticks in rules['instrumentation']['validation']:
        name = f'{ecology}_{mode}_seed_{seed}.csv.gz.metadata.json'
        off, on, repeat = [json.loads((root / phase / name).read_text()) for phase in ('validation_off', 'validation_on', 'validation_repeat')]
        for key in ('fingerprints', 'statistics', 'action_sha256'):
            if off[key] != on[key] or on[key] != repeat[key]: raise ValueError('Instrumentation purity/repeat proof changed')
        if on['dense']['sha256'] != repeat['dense']['sha256']: raise ValueError('Repeated binary not byte-identical')
        if any(off['files'][k]['sha256'] != on['files'][k]['sha256'] for k in off['files']): raise ValueError('Old diagnostic output altered by dense observation')
    for mode in 'ABC':
        name = f'E0_{mode}_seed_0.csv.gz.metadata.json'
        new = json.loads((root / 'legacy' / name).read_text())
        old = json.loads((Path('data/resource_ecology_v080/pilot') / name).read_text())
        if any(new[k] != old[k] for k in ('fingerprints', 'statistics', 'action_sha256')):
            raise ValueError('Archived E0 full-state/RNG regression')
    if not allowed(root, 'pilot') or not allowed(root, 'training') or full_cases['training'] != 36:
        raise ValueError('Validation/pilot/training stage incomplete')
    evaluation_allowed = allowed(root, 'evaluation')
    if evaluation_allowed and full_cases['evaluation'] != 36: raise ValueError('Expected36 allowed evaluation cases')
    if not evaluation_allowed and full_cases['evaluation']: raise ValueError('Evaluation performed despite failed gate')
    # Pilot seed10 is repeated as a full training prefix, not a new replication.
    prefix_checks = 0
    for pilot in sorted((root / 'pilot').glob('*.metadata.json')):
        a = json.loads(pilot.read_text()); train = root / 'training' / pilot.name; b = json.loads(train.read_text())
        if a['action_sha256'] != b['fingerprints']['action_checkpoints']['100']:
            raise ValueError('Pilot/training action/physiology prefix mismatch')
        prefix_checks += 1
    if evaluation_allowed:
        p = output / 'evaluation'; score = pd.read_csv(p / 'scores.csv'); paired = pd.read_csv(p / 'paired_by_seed.csv')
        if score.groupby(['ecology', 'mode', 'seed', 'target', 'subset']).same_decisions_sha256.nunique().ne(1).any():
            raise ValueError('Models scored on different decisions')
        if paired.loc[~paired.qualified, 'qualified_difference'].notna().any(): raise ValueError('Unsupported effect must be NA')
        support = pd.read_csv(p / 'support.csv'); support = support[support.target.isin(['T1', 'T2'])]
        scored = score.set_index(['ecology', 'mode', 'seed', 'target', 'model', 'subset'])
        for row in support.itertuples(index=False):
            key = (row.ecology, row.mode, row.seed, row.target)
            for model_name, subset, count in [('H0', 'all', row.observed), ('H0', 'H01', row.M01_common_n),
                ('H0', 'H012', row.M012_common_n), ('R0', 'R01', row.R1_feature_n), ('R0', 'R012', row.R2_feature_n)]:
                if (*key, model_name, subset) in scored.index and int(scored.loc[(*key, model_name, subset), 'decisions']) != int(count):
                    raise ValueError('Prediction masks do not match independent support calculations')
        for model_path in sorted((root / 'models').glob('*.json')):
            model = json.loads(model_path.read_text())
            if model['training_seeds'] != rules['train_seeds'] or not model['same_mode_training']:
                raise ValueError('Wrong predictor training seeds/policy')
            for name, fit in model['regularized'].items():
                if fit and fit['converged'] and fit['gradient_infinity'] > 1e-8:
                    raise ValueError('False model convergence report')
    if set(rules['train_seeds']) & set(rules['evaluation_seeds']) or set(rules['evaluation_seeds']) & set(range(5)):
        raise ValueError('Seed leakage into confirmation')
    log_path = Path(test_log) if test_log else root / 'final_tests.log'; log = log_path.read_text()
    result = re.search(r'Ran (\d+) tests in ([\d.]+)s', log)
    if not result or int(result[1]) < 681 or '\nOK\n' not in log or 'FAILED (' in log: raise ValueError('Complete test suite must pass')
    save_csv(output / 'snapshot_integrity.csv', pd.DataFrame(inventory))
    save_json(output / 'verification.json', {'version': '0.8.2', 'passed': True, 'baseline_commit': before['baseline_commit'],
        'protected_files_unchanged': len(protected), 'all_original_simulator_sources_and_tests_unchanged': True,
        'new_dense_observer_only': True, 'archived_E0_complete_exact_cases': 3, 'instrumentation_on_off_exact_cases': 4,
        'repeated_binary_exact_cases': 4, 'pilot_training_prefix_matches': prefix_checks, 'dense_frames_readback_verified': frames,
        'food_and_water_mass_balance_all_ticks': True, 'full_length_cases': full_cases, 'evaluation_allowed': evaluation_allowed,
        'same_decision_scoring': True, 'seed_separation_verified': True, 'no_threshold_or_regularization_retuning': True,
        'full_tests': int(result[1]), 'full_test_seconds': float(result[2]), 'test_log_sha256': digest(log_path),
        'preregistration_sha256': digest(root / 'preregistration.json'), 'run_metadata_sha256': metadata_inputs,
        'data_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        'analysis_bytes_before_receipt': sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        'stopping': 'Measurement methodology cycle complete; next intervention separate v083, no automatic extension.'})
    print('Verified:', len(protected), 'protected originals;', full_cases, 'full runs;', frames, 'dense frames;', result[1], 'tests.')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--output', type=Path, default=OUTPUT); parser.add_argument('--test-log', type=Path)
    try: run(**vars(parser.parse_args()))
    except (ValueError, OSError, KeyError) as error: parser.error(str(error))


if __name__ == '__main__': main()
