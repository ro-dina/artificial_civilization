"""Descriptive seed summaries only; never selects or redefines a target."""
from pathlib import Path
import argparse

import pandas as pd

from experiments.future_resource_io import ROOT, OUTPUT, digest, receipt, save_csv, save_json, sources
from experiments.future_resource_prediction import summarize


def run(root=ROOT, output=OUTPUT):
    root, output = Path(root), Path(output)
    receipt(root / 'frozen_evaluation.json', root)
    p = output / 'feasibility/target_support.csv'
    support = pd.read_csv(p)
    metrics = ['source_decisions', 'reference_observations', 'observed', 'positive', 'negative', 'prevalence', 'label_coverage',
               'agents', 'original_states', 'social_tokens', 'memory_tokens', 'social_effective_categories', 'memory_effective_categories',
               'adequately_mixed_baseline_strata', 'mixed_baseline_decisions', 'M01_common_coverage', 'M012_common_coverage',
               'missing_snapshot', 'administrative_end', 'missing_history', 'death_competing']
    summary = summarize(support, ['ecology', 'mode', 'target', 'kind', 'resource', 'horizon', 'radius'], metrics)
    save_csv(output / 'feasibility/seed_summary.csv', summary)
    save_csv(output / 'feasibility/target_eligibility.csv', support[['ecology', 'mode', 'seed', 'target', 'kind', 'classification', 'eligible', 'memory_eligible', 'failed_rules']])
    rows = []
    for (target, kind), group in support.groupby(['target', 'kind'], sort=True):
        failures = group.failed_rules.fillna('').str.split(',').explode()
        rows.append({'target': target, 'kind': kind, 'cases': len(group), 'eligible_cases': int(group.eligible.sum()),
            'memory_eligible_cases': int(group.memory_eligible.sum()),
            **{flag + '_failed_cases': int((failures == flag).sum()) for flag in ('prevalence', 'class_counts', 'agents', 'label_coverage', 'conditional_variation', 'M01_common', 'social_variation')}})
    save_csv(output / 'feasibility/eligibility_summary.csv', pd.DataFrame(rows))
    save_json(output / 'descriptive_tables_receipt.json', {'inputs': {str(p): digest(p)}, 'source_sha256': sources(),
        'interpretation': 'Within-run counts/rates retained, then equal-seed mean/sampleSD/min/max/n separately by ecology/mode. No decision pooling. Does not alter frozen selection or model scores.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--output', type=Path, default=OUTPUT)
    try:
        run(**vars(parser.parse_args()))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
