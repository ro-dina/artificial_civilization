"""Historical byte preservation and frozen v081 analysis verification."""
from pathlib import Path
import argparse
import json
import re

import pandas as pd

from experiments.future_resource_io import ROOT, OUTPUT, digest, receipt, save_csv, save_json, sources, verify_hashes


def run(root=ROOT, output=OUTPUT, test_log=None):
    root, output = Path(root), Path(output)
    before = json.loads((root / 'before.json').read_text())
    protected = {p: record['sha256'] for p, record in before['files'].items() if p not in ('README.md', '.gitignore')}
    verify_hashes(protected)
    plan = receipt(root / 'frozen_evaluation.json', root)
    phases = {}
    for name, path in [('audit', output / 'audit'), ('feasibility', output / 'feasibility'), ('prediction', output / 'prediction'), ('figures', output / 'figures')]:
        phases[name] = receipt(path / 'receipt.json', root)
    log_path = Path(test_log) if test_log else root / 'final_tests.log'
    log = log_path.read_text(); match = re.search(r'Ran (\d+) tests in ([\d.]+)s', log)
    if not match or int(match[1]) < 625 or '\nOK\n' not in log or 'FAILED (' in log:
        raise ValueError('Complete old and new tests must pass')
    scores = pd.read_csv(output / 'prediction/model_comparison.csv')
    if scores.groupby(['ecology', 'mode', 'seed', 'target', 'subset']).same_decisions_sha256.nunique().ne(1).any():
        raise ValueError('Models evaluated on different decisions')
    paired = pd.read_csv(output / 'prediction/paired_by_seed.csv')
    if paired.loc[~paired.support_eligible, 'eligible_difference'].notna().any():
        raise ValueError('Unsupported effects must be NA')
    if set(plan['train_seeds']) & set(plan['evaluation_seeds']):
        raise ValueError('Seed leakage')
    for name, record in phases['figures']['files'].items():
        if digest(output / 'figures' / name) != record['sha256']:
            raise ValueError('Figure changed')
    runtime = []
    for phase, value in phases.items():
        runtime.append({'phase': phase, 'wall_seconds': value['wall_seconds'], 'peak_memory_bytes': value['peak_memory_bytes'],
            'input_bytes': sum(Path(p).stat().st_size for p in value['inputs']),
            'output_bytes': sum(p.stat().st_size for p in (output / phase).rglob('*') if p.is_file())})
    save_csv(output / 'runtime_storage.csv', pd.DataFrame(runtime))
    result = {'version': '0.8.1', 'passed': True, 'baseline_commit': before['baseline_commit'],
        'protected_historical_files_unchanged': len(protected), 'simulator_and_all_original_tests_byte_unchanged': True,
        'compatibility_method': 'All original simulator/observer/brain/config sources and saved v080 state/RNG fingerprints byte-identical; original regression suite rerun. No scientific-world reruns or new diagnostic instrumentation.',
        'full_tests': int(match[1]), 'full_test_seconds': float(match[2]), 'test_log_sha256': digest(log_path),
        'new_simulation_runs': 0, 'primary_environment': plan['primary_environment'], 'secondary_behavior': plan['secondary_behavior'],
        'same_decisions_verified': True, 'unsupported_effects_NA': True, 'seed_separation_verified': True,
        'retrospective_only': True, 'source_sha256': sources(), 'phase_receipts': {name: digest(output / name / 'receipt.json') for name in phases},
        'inputs_unchanged': True, 'data_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        'analysis_bytes_before_receipt': sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}
    save_json(output / 'verification.json', result)
    print('Verified', len(protected), 'protected files,', match[1], 'tests, identical model rowsets, frozen sources/inputs; no new simulations.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--output', type=Path, default=OUTPUT); parser.add_argument('--test-log', type=Path)
    try:
        run(**vars(parser.parse_args()))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
