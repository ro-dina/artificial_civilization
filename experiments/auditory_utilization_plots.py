"""Headless, deterministic v0.6.4 figures; points denote independent runs."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.auditory_controlled import CONDITIONS

COLORS = dict(zip(CONDITIONS, ('#0072B2', '#D55E00', '#009E73', '#777777')))


def points(ax, frame, x, y, groups, *, order=None):
    """Equal-seed mean/SD with deterministic offsets and no random jitter."""
    order = sorted(frame[x].unique()) if order is None else order
    names = sorted(frame[groups].unique())
    width = .7 / max(1, len(names))
    for j, name in enumerate(names):
        part = frame[frame[groups] == name]
        labeled = False
        for i, value in enumerate(order):
            b = part[part[x] == value].sort_values('seed')[y].dropna()
            center = i + (j - (len(names)-1)/2)*width
            if b.empty:
                continue
            offsets = np.linspace(-width*.22, width*.22, len(b)) if len(b)>1 else np.array([0.])
            color = COLORS.get(name, f'C{j}')
            ax.scatter(center+offsets, b, s=16, color=color, alpha=.7)
            ax.errorbar(center, b.mean(), yerr=b.std(ddof=1) if len(b)>1 else None,
                        fmt='D', ms=4, capsize=3, color=color,
                        label=str(name) if not labeled else None)
            labeled = True
    ax.set_xticks(range(len(order)), [str(x).replace('_', '\n') for x in order])
    ax.grid(axis='y', alpha=.2)


def plots(tables, directory):
    """Empty components are skipped; unsupported distances remain gaps."""
    files = []
    with plt.rc_context({'font.size': 9, 'axes.titlesize': 10}):
        def save(fig, name):
            fig.tight_layout()
            fig.savefig(directory/name, dpi=180, metadata={'Software': 'Artificial Civilization v0.6.4'})
            plt.close(fig)
            files.append(name)

        exposure = tables['retrospective_exposure_by_window']
        if not exposure.empty:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
            for ax, metric in zip(axes, ('current_identified_rate', 'encoded_nonempty_rate', 'silent_with_retained_event_rate')):
                for condition, b in exposure.groupby('condition', sort=True):
                    g = b.groupby('actual_window_end')[metric].agg(['mean', 'std'])
                    ax.plot(g.index, g['mean'], label=condition)
                    ax.fill_between(g.index, g['mean']-g['std'], g['mean']+g['std'], alpha=.15)
                ax.set(title=metric.replace('_', ' '), xlabel='tick', ylabel='Fraction of decisions')
                ax.legend(title='Condition'); ax.set_ylim(0, 1)
            fig.suptitle('Observational auditory exposure: equal-seed mean ± sample SD')
            save(fig, 'retrospective_exposure_memory.png')

        matched = tables['retrospective_matched_by_run']
        matched = matched[matched.contrast.eq('all')] if not matched.empty else matched
        for metric, filename, title in (
            ('coverage', 'retrospective_matched_coverage.png', 'Observational matched-decision coverage'),
            ('mean_context_tv', 'retrospective_action_distances.png', 'Observational action TV: matched contexts')):
            if matched.empty:
                continue
            fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
            for ax, scope in zip(axes, ('current', 'encoded', 'current_given_encoded', 'encoded_given_current')):
                b = matched[matched.scope.eq(scope)]
                points(ax, b, 'threshold', metric, 'condition')
                ax.set(title=scope.replace('_', ' '), xlabel='Minimum decisions per exact audio cell', ylabel=metric)
                if ax.get_legend_handles_labels()[0]:ax.legend(title='Condition', fontsize=7)
            fig.suptitle(title+' (points = seeds; mean ± sample SD)')
            save(fig, filename)

        cf = tables['counterfactual_summary_by_run']
        if not cf.empty:
            interventions = sorted(cf.intervention.unique())
            for metric, filename in (('mean_tv', 'counterfactual_policy_tv.png'), ('mean_jsd_bits', 'counterfactual_policy_jsd.png')):
                # Physical contexts are never pooled to increase seed counts.
                fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
                for ax, coverage in zip(axes, ('both_present', 'one_present', 'neither_present')):
                    b = cf[cf.coverage.eq(coverage) & cf.physical_context.eq('root_empty')]
                    points(ax, b, 'intervention', metric, 'condition', order=interventions)
                    ax.set(title=coverage.replace('_', ' '), ylabel=metric)
                    ax.tick_params(axis='x', labelsize=7)
                    if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
                fig.suptitle('Frozen-policy intervention: root body/visual context; seed means ± sample SD')
                save(fig, filename)

        coverage = tables['counterfactual_state_coverage']
        if not coverage.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            points(ax, coverage, 'coverage', 'fraction', 'condition', order=['both_present','one_present','neither_present'])
            ax.set(title='Frozen-policy probes: Q-row presence (18 probes per receiver)', ylabel='Fraction of probes')
            ax.legend(); save(fig, 'counterfactual_state_coverage.png')

        if not cf.empty:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, metric in zip(axes, ('policy_changed_fraction','max_set_changed_fraction','unique_greedy_changed_fraction')):
                # One mean per condition/seed, not individual probes as replicates.
                raw = tables['counterfactual_probes']
                if metric == 'policy_changed_fraction':
                    b = raw.assign(value=raw.tv.gt(0).astype(float))
                else:
                    source = 'max_set_changed' if metric.startswith('max_set') else 'unique_greedy_changed'
                    b = raw.assign(value=raw[source].map({True:1.,False:0.}))
                b = b.groupby(['condition','seed'], as_index=False).value.mean()
                points(ax,b,'condition','value','condition',order=list(CONDITIONS))
                ax.set(title=metric.replace('_',' '),ylabel='Fraction (defined comparisons only)',ylim=(-.03,1.03))
                ax.tick_params(axis='x',labelsize=7)
            fig.suptitle('Frozen-policy interventions: full distribution versus greedy changes')
            save(fig,'greedy_vs_policy_changes.png')

        curves = tables['training_learning_curves']
        if not curves.empty:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
            for ax, metric in zip(axes, ('mean_correct_action_probability','training_first_action_correct','training_water_per_trial')):
                for condition, b in curves.groupby('condition',sort=True):
                    g=b.groupby('trials')[metric].agg(['mean','std'])
                    ax.plot(g.index,g['mean'],label=condition,color=COLORS[condition])
                    ax.fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],alpha=.15,color=COLORS[condition])
                ax.set(title=metric.replace('_',' '),xlabel='Completed training trials',ylabel='Probability / fraction')
                ax.legend(fontsize=7)
            fig.suptitle('Controlled acquisition: frozen root probes / preceding training blocks (mean ± SD)')
            save(fig,'controlled_learning_curves.png')

        results=tables['controlled_experiment_results']
        if not results.empty:
            fig, axes=plt.subplots(1,3,figsize=(12,4))
            for ax, metric in zip(axes, ('root_identity_tv','root_identity_jsd_bits','mean_correct_probability')):
                points(ax,results,'condition',metric,'condition',order=list(CONDITIONS))
                ax.set(title=metric.replace('_',' '),ylabel='Frozen-policy measurement',ylim=(-.03,1.03))
                ax.tick_params(axis='x',labelsize=7)
            fig.suptitle('Controlled receiver: every seed and equal-seed mean ± sample SD')
            save(fig,'controlled_condition_comparison.png')
    return files
