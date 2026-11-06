"""Seed-level v082 tables and figures, only for actually executed experiments."""
from pathlib import Path
from time import perf_counter
import argparse
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.future_resource_io import digest, save_json, save_csv as exclusive_csv
from experiments.future_resource_prediction import summarize
from experiments.dense_resource_data import ROOT, OUTPUT, plan, cases
from experiments.resource_ecology_experiments import peak


def save_csv(path, frame):
    """Presentation rerenders reuse byte-identical tables; never replace data."""
    path = Path(path)
    if path.exists():
        if path.read_text() != frame.to_csv(index=False, float_format='%.12g', lineterminator='\n'):
            raise ValueError('Existing descriptive table differs; refusing overwrite')
    else:
        exclusive_csv(path, frame)


def run(root=ROOT, output=OUTPUT):
    root, output = Path(root), Path(output); rules = plan(root); start = perf_counter()
    out = output / 'figures'
    if out.exists(): raise ValueError('Refusing figure overwrite')
    out.mkdir(parents=True); inputs = {}; skipped = {}
    evaluated = (output / 'evaluation/receipt.json').exists()
    phase = 'evaluation' if evaluated else 'training'
    seeds = rules['evaluation_seeds'] if evaluated else rules['train_seeds']
    folder = output / phase
    def read(name):
        p = folder / (name + '.csv'); inputs[str(p)] = digest(p); return pd.read_csv(p)
    support = read('support'); support = support[support.target.isin(['T1', 'T2'])]
    numeric = [c for c in support.select_dtypes('number') if c != 'seed']
    save_csv(output / 'support_seed_summary.csv', summarize(support, ['ecology', 'mode', 'target'], numeric))
    runtime = []
    for p in sorted(root.rglob('*.csv.gz.metadata.json')):
        meta = json.loads(p.read_text())
        runtime.append({'phase': meta['phase'], 'ecology': meta['ecology'], 'mode': meta['mode'], 'seed': meta['seed'],
            'ticks': meta['tick'], **meta['runtime'], 'decision_stream_bytes': sum(r['bytes'] for r in meta['files'].values()),
            'dense_bytes': meta['dense']['bytes'] if meta['dense'] else 0, 'dense_frames': meta['dense']['records'] if meta['dense'] else 0,
            'decisions': meta['totals']['decisions'], 'population': meta['statistics']['population']})
    save_csv(output / 'runtime_by_run.csv', pd.DataFrame(runtime))
    ecologies = ('E0', 'E1', 'E2', 'E3'); colors = plt.get_cmap('tab10').colors
    scope = f'{phase}: ecological seeds{seeds}, n≤3 per ecology/mode; equal-seed mean ± sample SD. Frozen plan, no decision-level significance or causal policy-value inference.'
    def finish(fig, name, title, extra=''):
        fig.suptitle(title, fontsize=12)
        fig.text(.5, .01, '\n'.join(textwrap.wrap(scope + ' ' + extra, 135)), ha='center', va='bottom', fontsize=8)
        fig.tight_layout(rect=(0, .12, 1, .93)); fig.savefig(out / (name + '.png'), dpi=180); plt.close(fig)
    def metric_plot(frame, metrics, name, title):
        fig, axes = plt.subplots(len(metrics), 3, figsize=(15, 4 * len(metrics) + 1), squeeze=False)
        for row, metric in enumerate(metrics):
            for ax, mode in zip(axes[row], 'ABC'):
                f = frame[frame['mode'].eq(mode)]
                for j, (target, group) in enumerate(f.groupby('target')):
                    s = group.groupby('ecology')[metric].agg(['mean', 'std']).reindex(ecologies)
                    ax.errorbar(np.arange(4) + (j - .5) * .08, s['mean'], yerr=s['std'], fmt='o-', capsize=3, label=target)
                ax.set_xticks(range(4), ecologies); ax.set(title=mode + ': ' + metric, ylabel=metric); ax.legend(fontsize=8)
        finish(fig, name, title)
    metric_plot(support[support.target.eq('T1')], ['prevalence'], 'future_water_prevalence', 'Primary T1: water at frozen origin tile, end tick t+5')
    metric_plot(support[support.target.eq('T2')], ['prevalence'], 'water_depletion_prevalence', 'Secondary T2: depletion, conditional origin water present')
    metric_plot(support, ['label_coverage'], 'label_coverage', 'Exact environmental label coverage; death does not remove an origin')
    metric_plot(support, ['M01_common_coverage', 'M012_common_coverage', 'R1_feature_coverage', 'R2_feature_coverage'], 'conditional_feature_support', 'Strict exact cells and feature-level support are distinct')
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for j, ecology in enumerate(ecologies):
        p = cases(root, phase, [seeds[0]])[3 * j]
        meta = json.loads(p.read_text()); dense = p.parent / meta['dense']['file']; stock = pd.read_csv(str(dense) + '.stocks.csv')
        inputs[str(dense) + '.stocks.csv'] = digest(str(dense) + '.stocks.csv')
        for i, resource in enumerate(('food', 'water')):
            ax = axes[i, j]; error = int((stock[resource + '_stock'] - stock[resource + '_expected']).abs().max())
            ax.plot(stock.tick, stock[resource + '_stock'], label='actual units'); ax.plot(stock.tick, stock[resource + '_expected'], '--', label='mass balance')
            ax.set(title=f'{ecology}/A seed{seeds[0]} {resource}; max error={error}', xlabel='Tick', ylabel='Stock units'); ax.legend(fontsize=7)
    finish(fig, 'snapshot_integrity', 'Dense resource stock equals accounting on every recorded tick', 'Example first seed/A; independent verification covers all cases, both resources and all frames.')
    if not evaluated:
        for name in ('hierarchical_log_loss', 'regularized_log_loss', 'social_increment', 'memory_increment', 'per_seed_pairs', 'calibration', 'q_fragmentation'):
            skipped[name] = 'Evaluation not executed; no invented predictive result.'
    else:
        scores, paired, calibration, diagnostics, probes = [read(n) for n in ('scores', 'paired_by_seed', 'calibration', 'diagnostics', 'policy_probes')]
        qualified = paired[paired.qualified].copy()
        sign_counts = qualified.groupby(['ecology', 'mode', 'target', 'subset', 'contrast', 'metric']).qualified_difference.agg(
            lower_loss_seeds=lambda v: int((v < 0).sum()), higher_loss_seeds=lambda v: int((v > 0).sum()),
            equal_loss_seeds=lambda v: int((v == 0).sum()), n='count').reset_index()
        save_csv(output / 'paired_sign_counts.csv', sign_counts)
        joint = []
        for row in support.itertuples(index=False):
            model = json.loads((root / 'models' / f'{row.ecology}_{row.mode}_{row.target}.json').read_text())
            for increment, test in [('H1', row.eligible), ('H2', row.memory_eligible), ('R1', row.R1_eligible), ('R2', row.R2_eligible)]:
                training = bool(model['training_qualification'][increment])
                joint.append({'ecology': row.ecology, 'mode': row.mode, 'seed': row.seed, 'target': row.target, 'increment': increment,
                    'training_gate': training, 'heldout_gate': bool(test), 'combined_target_model_gate': training and bool(test),
                    'interpretation': 'exact-cell qualified' if increment.startswith('H') else 'feature-level model-dependent generalization'})
        save_csv(output / 'joint_qualification.csv', pd.DataFrame(joint))
        for family, filename in [('H', 'hierarchical_log_loss'), ('R', 'regularized_log_loss')]:
            eligible = paired[paired.contrast.eq(family + '1-' + family + '0') & paired.subset.eq('all') & paired.metric.eq('log_loss') & paired.qualified]
            if eligible.empty:
                skipped[filename] = 'NA: no qualified perception comparison in this model family.'; continue
            keys = eligible[['ecology', 'mode', 'seed', 'target']].drop_duplicates()
            f = scores.merge(keys, on=['ecology', 'mode', 'seed', 'target']); f = f[f.subset.eq('all') & f.model.isin([family + '0', family + '1'])]
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for i, target in enumerate(('T1', 'T2')):
                for ax, mode in zip(axes[i], 'ABC'):
                    for j, model in enumerate((family + '0', family + '1')):
                        s = f[f.target.eq(target) & f['mode'].eq(mode) & f.model.eq(model)].groupby('ecology').log_loss.agg(['mean', 'std', 'count']).reindex(ecologies)
                        ax.errorbar(np.arange(4) + (j - .5) * .08, s['mean'], yerr=s['std'], fmt='o-', capsize=3, label=model)
                    ax.set_xticks(range(4), ecologies); ax.set(title=target + '/' + mode, ylabel='Log loss', xlim=(-.5, 3.5)); ax.legend(fontsize=8)
            finish(fig, filename, family + ' baseline/social loss on identical all-label decisions', 'Only qualified perception seed cases shown; corresponding baseline uses exactly the same cases. Raw unsupported scores stay in CSV.')
        def differences(suffix, filename, title, outside=False):
            f = paired[paired.contrast.str.contains(suffix, regex=False) & paired.metric.eq('log_loss') & paired.qualified]
            if outside:
                f = f[f.contrast.eq('R1-R0') & f.subset.eq('R01_outside_H01')]
            else:
                f = f[f.subset.eq('all')]
            if f.empty:
                skipped[filename] = 'NA: no frozen-support-qualified paired effect.'; return
            families = ('R',) if outside else ('H', 'R')
            fig, axes = plt.subplots(2, len(families), figsize=(7 * len(families), 9), squeeze=False)
            for i, target in enumerate(('T1', 'T2')):
                for ax, family in zip(axes[i], families):
                    g = f[f.target.eq(target) & f.contrast.str.startswith(family)]
                    for j, mode in enumerate('ABC'):
                        for k, ecology in enumerate(ecologies):
                            values = g[g['mode'].eq(mode) & g.ecology.eq(ecology)].qualified_difference.dropna().to_numpy()
                            x = k + (j - 1) * .23
                            if len(values):
                                ax.errorbar(x, values.mean(), yerr=values.std(ddof=1) if len(values) > 1 else None, fmt='o', color=colors[j], capsize=3)
                                ax.scatter(x + np.linspace(-.035, .035, len(values)), values, s=9, color=colors[j])
                            else: ax.text(x, .5, 'NA', transform=ax.get_xaxis_transform(), fontsize=6, rotation=90, ha='center')
                    ax.axhline(0, color='grey', lw=.7); ax.set_xticks(range(4), ecologies); ax.set(title=target + '/' + family, ylabel='Paired log-loss difference', xlim=(-.5, 3.5))
            handles = [plt.Line2D([], [], marker='o', color=colors[i], label=m) for i, m in enumerate('ABC')]
            axes[0, -1].legend(handles=handles, fontsize=8)
            finish(fig, filename, title, 'Negative means lower loss. Dots are independent ecological seeds; unsupported cases NA. R is model-dependent generalization, not causal benefit.')
        # Match suffix independently of family prefix: "1-H0" and "1-R0".
        for number, filename, title in [(1, 'social_increment', 'Perception increment beyond original X and selected action'), (2, 'memory_increment', 'Remembered-direction increment holding actual B fixed')]:
            subset = paired[paired.contrast.str[1].eq(str(number))].copy()
            saved = paired; paired = subset
            differences('-', filename, title)
            paired = saved
        saved = paired; paired = paired[paired.contrast.eq('R1-R0')]
        differences('-', 'generalization_outside_exact', 'R1−R0 outside H exact common support: model-dependent generalization', True); paired = saved
        f = paired[paired.subset.eq('all') & paired.metric.eq('log_loss') & paired.qualified]
        if len(f):
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            for ax, contrast in zip(axes.ravel(), ('H1-H0', 'H2-H1', 'R1-R0', 'R2-R1')):
                for i, (ecology, mode) in enumerate((e, m) for e in ecologies for m in 'ABC'):
                    for j, target in enumerate(('T1', 'T2')):
                        g = f[f.contrast.eq(contrast) & f.ecology.eq(ecology) & f['mode'].eq(mode) & f.target.eq(target)]
                        ax.scatter(i + (j - .5) * .16 + np.linspace(-.025, .025, len(g)), g.qualified_difference, s=12, color=colors[j], label=target if i == 0 else None)
                ax.axhline(0, color='grey', lw=.7); ax.set_xticks(range(12), [e + '/' + m for e in ecologies for m in 'ABC'], rotation=45); ax.set_title(contrast); ax.legend(fontsize=8)
            finish(fig, 'per_seed_pairs', 'Per-seed prediction contrasts, qualified estimates only')
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, target in zip(axes, ('T1', 'T2')):
            f = calibration[calibration.ecology.eq('E3') & calibration['mode'].eq('C') & calibration.target.eq(target) & calibration.subset.eq('all')]
            for j, (model, g) in enumerate(f.groupby('model')):
                s = g.groupby('bin')[['mean_probability', 'positive_fraction']].mean()
                ax.plot(s.mean_probability, s.positive_fraction, 'o-' if model[0] == 'H' else 's--', color=colors[j], label=model, ms=3)
            ax.plot([0, 1], [0, 1], ':', color='grey'); ax.set(title='E3/C ' + target, xlabel='Mean predicted positive probability', ylabel='Observed positive fraction', xlim=(0, 1), ylim=(0, 1)); ax.legend(fontsize=7)
        finish(fig, 'calibration', 'Descriptive E3/C calibration: equal-seed bin means', 'H/T1 in E3 is an unsupported diagnostic; R remains feature-qualified. Bin counts/contributing seeds vary; no decision-level confidence interval.')
        diagnostics['target'] = 'actual run'
        metric_plot(diagnostics, ['first_state_fraction', 'mean_visits_per_agent_state', 'q_occupancy', 'evictions'], 'q_fragmentation', 'Unchanged Physical Q: novelty, revisits, occupancy and evictions')
        metric_plot(diagnostics, ['emission_rate', 'identified_fraction', 'masked_fraction', 'vocal_q_rows_final'], 'vocal_observations', 'Unchanged Vocal Q and auditory exposure: no semantic interpretation')
        probes['policy_tv'] = probes.tv_sum / probes.pairs.replace(0, np.nan)
        save_csv(output / 'policy_probe_values.csv', probes)
        save_csv(output / 'calibration_seed_summary.csv', summarize(calibration, ['ecology', 'mode', 'target', 'model', 'subset', 'bin'], ['n', 'mean_probability', 'positive_fraction']))
    save_json(out / 'receipt.json', {'version': '0.8.2', 'phase': phase, 'inputs': inputs, 'source_sha256': {__file__: digest(__file__)},
        'wall_seconds': perf_counter() - start, 'peak_memory_bytes': peak(), 'skipped': skipped,
        'files': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(out.glob('*.png'))}})


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--output', type=Path, default=OUTPUT)
    try: run(**vars(parser.parse_args()))
    except (ValueError, OSError, KeyError) as error: parser.error(str(error))


if __name__ == '__main__': main()
