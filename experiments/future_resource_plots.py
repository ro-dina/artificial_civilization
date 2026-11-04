"""Seed-level descriptive target-support figures; unsupported effects stay NA."""
from pathlib import Path
from time import perf_counter
import argparse
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.future_resource_io import ROOT, OUTPUT, digest, receipt, save_json, sources
from experiments.resource_ecology_experiments import peak


def run(root=ROOT, output=OUTPUT):
    root, output = Path(root), Path(output)
    plan = receipt(root / 'frozen_evaluation.json', root)
    out = output / 'figures'
    if out.exists():
        raise ValueError('Refusing existing figures')
    out.mkdir(parents=True)
    start = perf_counter()
    inputs, skipped = {}, {}
    def read(path):
        path = output / path
        inputs[str(path)] = digest(path)
        return pd.read_csv(path)
    support = read('feasibility/target_support.csv')
    scores = read('prediction/model_comparison.csv')
    paired = read('prediction/paired_by_seed.csv')
    calibration = read('prediction/calibration.csv')
    curves = read('feasibility/competing_curves.csv')
    definitions = read('audit/target_definitions.csv')
    ecologies = ('E0', 'E1', 'E2', 'E3')
    colors = dict(zip(('M0', 'M1', 'M2', 'X_only'), plt.get_cmap('tab10').colors[:4]))
    targets = (plan['retrospective_environment'], plan['retrospective_behavior'])
    scope = 'Retrospective archives: ecological seeds0/1/2, n≤3 per ecology/mode. Seed means ± sample SD. Support gates frozen; no causal/policy-benefit inference.'
    def finish(fig, name, title, extra=''):
        fig.suptitle(title, fontsize=12)
        fig.text(.5, .01, '\n'.join(textwrap.wrap(scope + ' ' + extra, 135)), ha='center', va='bottom', fontsize=8)
        fig.tight_layout(rect=(0, .11, 1, .93))
        fig.savefig(out / (name + '.png'), dpi=180)
        plt.close(fig)
    def heatmap(metric, name, title, vmax=1):
        fig, axes = plt.subplots(1, 3, figsize=(16, 13))
        order = definitions.loc[definitions.kind.ne('time_to_event'), 'name'].tolist()
        for ax, mode in zip(axes, 'ABC'):
            grouped = support[support['mode'].eq(mode)].groupby(['target', 'ecology'])[metric].mean().unstack().reindex(index=order, columns=ecologies)
            im = ax.imshow(grouped.to_numpy(), aspect='auto', vmin=0, vmax=vmax, cmap='viridis')
            ax.set_xticks(range(4), ecologies); ax.set_yticks(range(len(order)), order, fontsize=7); ax.set_title(mode)
            fig.colorbar(im, ax=ax, fraction=.04)
        finish(fig, name, title, 'Environmental values use exact snapshot-aligned labels only; other timestamps remain missing.')
    heatmap('prevalence', 'target_prevalence', 'Observed positive prevalence by candidate and ecology')
    support['mixed_fraction'] = support.mixed_baseline_decisions / support.observed.replace(0, np.nan)
    heatmap('mixed_fraction', 'conditional_outcome_variation', 'Fraction in X/action strata with ≥20 of EACH observed class')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    merged = support.merge(definitions[['name', 'family']], left_on='target', right_on='name')
    selected = merged[merged['mode'].eq('A') & merged.family.isin(['T1', 'T2']) & ~merged.target.str.contains('neighbor')]
    for i, resource in enumerate(('food', 'water')):
        for j, radius in enumerate((0, 1, 2)):
            ax = axes[i, j]
            for ecology, group in selected[selected.resource.eq(resource) & selected.radius.eq(radius)].groupby('ecology'):
                s = group.groupby('horizon').prevalence.agg(['mean', 'std'])
                ax.errorbar(s.index, s['mean'], yerr=s['std'], marker='o', capsize=3, label=ecology)
            ax.set(title=f'A: {resource}, frozen radius{radius}', xlabel='Future ticks to END snapshot', ylabel='Observed prevalence', ylim=(0, 1)); ax.legend(fontsize=8)
    finish(fig, 'variation_by_horizon', 'Future resource variation: exact sparse calendar endpoints', 'Mode A shown; this is not all-calendar prevalence and no future-agent recentering is used.')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, target in enumerate(targets):
        for ax, mode in zip(axes[i], 'ABC'):
            f = support[support.target.eq(target) & support['mode'].eq(mode)].copy()
            fields = ['observed', 'missing_snapshot', 'death_competing', 'administrative_end', 'missing_history']
            bottom = np.zeros(4)
            for field in fields:
                f[field + '_fraction'] = f[field] / f.reference_observations.replace(0, np.nan)
                values = f.groupby('ecology')[field + '_fraction'].mean().reindex(ecologies).to_numpy()
                ax.bar(range(4), values, bottom=bottom, label=field); bottom += values
            ax.set_xticks(range(4), ecologies); ax.set(title=f'{mode}: {target}', ylabel='Fraction of reference-eligible origins', ylim=(0, 1.03))
    axes[0, -1].legend(fontsize=7)
    finish(fig, 'target_censoring', 'Observed, unrecorded and competing-event outcomes', 'Missing snapshots are missing observations, not survival censoring. Deaths are not ordinary negative labels.')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, target in enumerate(targets):
        for ax, mode in zip(axes[i], 'ABC'):
            f = support[support.target.eq(target) & support['mode'].eq(mode)]
            for metric, label, offset in [('M01_common_coverage', 'M0/M1', -.06), ('M012_common_coverage', 'M0/M1/M2', .06)]:
                s = f.groupby('ecology')[metric].agg(['mean', 'std']).reindex(ecologies)
                ax.errorbar(np.arange(4) + offset, s['mean'], yerr=s['std'], marker='o', capsize=3, label=label)
            ax.axhline(.2, color='grey', ls='--'); ax.set_xticks(range(4), ecologies)
            ax.set(title=f'{mode}: {target}', ylabel='Common training-cell coverage (min20)', ylim=(-.01, 1)); ax.legend(fontsize=7)
    finish(fig, 'social_feature_support', 'Observed-label common support, distinct from target coverage')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, target in enumerate(targets):
        for ax, mode in zip(axes[i], 'ABC'):
            f = scores[scores.target.eq(target) & scores['mode'].eq(mode) & scores.subset.eq('all')]
            for j, model in enumerate(f.model.unique()):
                s = f[f.model.eq(model)].groupby('ecology').log_loss.agg(['mean', 'std']).reindex(ecologies)
                ax.errorbar(np.arange(4) + (j - 1.5) * .09, s['mean'], yerr=s['std'], marker='o', capsize=2, color=colors[model], label=model)
            ax.set_xticks(range(4), ecologies); ax.set(title=f'{mode}: {target}', ylabel='Binary log loss'); ax.legend(fontsize=7)
    finish(fig, 'prediction_loss', 'All-observed-label loss: including unsupported backoff diagnostics', 'Fallback/unsupported scores are not evidence of incremental information; eligibility and common subsets remain in CSV.')
    def differences(contrast, name, title):
        f = paired[paired.contrast.eq(contrast) & paired.metric.eq('log_loss') & paired.subset.eq('common012' if contrast == 'M2-M1' else 'common01')].dropna(subset=['eligible_difference'])
        if f.empty:
            skipped[name] = 'NA: no case meets frozen target and feature/common-support criteria; no zero-effect plot generated.'
            return
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for ax, target in zip(axes, targets):
            g = f[f.target.eq(target)]
            for j, mode in enumerate('ABC'):
                for k, ecology in enumerate(ecologies):
                    values = g[g['mode'].eq(mode) & g.ecology.eq(ecology)].eligible_difference.to_numpy()
                    x = k + (j - 1) * .2
                    if len(values):
                        ax.scatter(x + np.linspace(-.025, .025, len(values)), values, s=12, label=mode if k == 0 else None)
                        ax.errorbar(x, values.mean(), yerr=values.std(ddof=1) if len(values) > 1 else None, fmt='k_', capsize=3)
                    else:
                        ax.text(x, .5, 'NA', fontsize=6, rotation=90, transform=ax.get_xaxis_transform(), ha='center')
            ax.axhline(0, color='grey', lw=.7); ax.set_xticks(range(4), ecologies); ax.set(title=target, ylabel=contrast + ' log-loss difference')
        finish(fig, name, title, 'Only frozen-support-eligible seed cases; unsupported cells NA. Raw per-seed backoff differences are retained separately.')
    differences('M1-M0', 'per_seed_contrasts', 'Paired perception contrasts: exact same decisions')
    differences('M2-M1', 'memory_increment', 'Incremental remembered direction, holding actual B fixed')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, target in enumerate(targets):
        for ax, mode in zip(axes[i], 'ABC'):
            # Show E3 explicitly, not a mixture of ecological distributions.
            f = calibration[calibration.target.eq(target) & calibration['mode'].eq(mode) & calibration.ecology.eq('E3') & calibration.subset.eq('all')]
            for model, group in f.groupby('model'):
                s = group.groupby('bin')[['mean_probability', 'positive_fraction']].mean()
                ax.plot(s.mean_probability, s.positive_fraction, 'o-', color=colors[model], label=model, ms=3)
            ax.plot([0, 1], [0, 1], color='grey', ls='--'); ax.set(title=f'E3/{mode}: {target}', xlabel='Mean predicted positive probability', ylabel='Observed positive fraction', xlim=(0, 1), ylim=(0, 1)); ax.legend(fontsize=7)
    finish(fig, 'calibration', 'Descriptive E3 calibration: equal-seed bin means', 'Bin sample counts and contributing seeds vary; full calibration for every ecology/seed is in CSV. No decision-level uncertainty interval.')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, ['prevalence', 'label_coverage', 'M01_common_coverage']):
        for target in targets:
            f = support[support.target.eq(target) & support['mode'].eq('A')]
            s = f.groupby('ecology')[metric].agg(['mean', 'std']).reindex(ecologies)
            ax.errorbar(range(4), s['mean'], yerr=s['std'], marker='o', capsize=3, label=target)
        ax.set_xticks(range(4), ecologies); ax.set(title='A: ' + metric, ylim=(0, 1.03)); ax.legend(fontsize=6)
    finish(fig, 'environmental_vs_behavioral', 'Distinct target populations: environmental and behavioral measurements', 'Parallel support summaries only; these outcomes are not combined into a single information-value score.')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, mode in zip(axes, 'ABC'):
        for ecology, group in curves[curves.target.eq('T4_either') & curves['mode'].eq(mode)].groupby('ecology'):
            s = group.groupby('elapsed')[['acquisition_cif', 'death_cif']].mean()
            line, = ax.plot(s.index, s.acquisition_cif, label=ecology + ' acquire')
            ax.plot(s.index, s.death_cif, '--', color=line.get_color(), label=ecology + ' death')
        ax.set(title=mode, xlabel='Own opportunities after origin', ylabel='Descriptive competing cumulative incidence', ylim=(0, 1)); ax.legend(fontsize=6, ncol=2)
    finish(fig, 'competing_events', 'Next actual acquisition and competing death', 'Discrete Aalen–Johansen; overlapping origins and administrative calendar selection limit interpretation; not independent agent survival.')
    save_json(out / 'receipt.json', {'version': '0.8.1', 'inputs': inputs, 'source_sha256': sources(),
        'wall_seconds': perf_counter() - start, 'peak_memory_bytes': peak(), 'skipped_figures': skipped,
        'files': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(out.glob('*.png'))}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--output', type=Path, default=OUTPUT)
    try:
        run(**vars(parser.parse_args()))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
