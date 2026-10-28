"""Deterministic, headless diagnostic figures (no random jitter or inference)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def plots(tables, runs, directory, min_decisions):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9, 'axes.spines.top':False,'axes.spines.right':False})
    conditions = sorted({r['condition'] for r in runs})
    colors = {c:plt.get_cmap('tab10')(i) for i,c in enumerate(conditions)}
    names, selection = [], {}
    summary = tables['time_series_summary']

    def draw(ax, source, metric, x='actual_window_end', style='-', label_suffix=''):
        if not {'metric','condition',x,'mean','std','n'} <= set(source):
            return
        for c in conditions:
            rows = source.loc[source.condition.eq(c) & source.metric.eq(metric)].sort_values(x)
            if rows.empty or not rows.n.gt(0).any():
                continue
            xx, mean, sd = (rows[col].to_numpy(float) for col in (x,'mean','std'))
            n = rows.loc[rows.n.gt(0),'n']
            ax.plot(xx,mean,style,color=colors[c],label=f'{c}{label_suffix} (n={int(n.min())}–{int(n.max())} seeds)')
            ax.fill_between(xx,mean-sd,mean+sd,color=colors[c],alpha=.13,linewidth=0)

    def decorate(ax, title, ylabel, percent=False, xlabel='Simulation tick'):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
        ax.grid(alpha=.2)
        if percent:
            ax.set_ylim(-.02,1.02)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
        else:
            ax.text(.5,.5,'Not applicable / no observed Vocal Q decisions',ha='center',transform=ax.transAxes)

    def save(fig,name):
        fig.tight_layout()
        fig.text(.5,-.025,'C: listening OFF; D: listening ON. Both Vocal Q ON. A/B: Vocal Q N/A, random emission.\n'
                 'Lines: equal-seed means; bands: sample SD. Descriptive diagnostics, no semantic or causal claim.',
                 ha='center',va='top',fontsize=8)
        fig.savefig(Path(directory)/name,dpi=180,bbox_inches='tight',metadata={'Software':'Vocal Q diagnostics v0.6.2'})
        plt.close(fig)
        names.append(name)

    specs = [('hit_rate','Vocal Q state hit rate','Proportion present before selection','vocal_state_hit_rate.png',True),
             ('eviction_rate','Actual Vocal Q LRU evictions','Evictions / Vocal Q decision','vocal_eviction_rate.png',False),
             ('mean_occupancy','Vocal Q table occupancy','Stored rows at selection','vocal_q_occupancy.png',False),
             ('mean_q_gap','Vocal Q preference gap','Maximum Q − second-largest Q','vocal_q_preference_gap.png',False),
             ('multiple_max_fraction','Vocal Q maximum ties','Proportion with multiple maxima','vocal_q_ties.png',True),
             ('mean_abs_td_error','Vocal Q TD error','Mean absolute TD error / update','vocal_td_error.png',False)]
    for metric,title,label,name,percent in specs:
        fig, ax = plt.subplots(figsize=(8,4))
        draw(ax,summary,metric)
        if metric == 'mean_occupancy':
            for limit in sorted({r['metadata']['config']['vocal_learning_memory_capacity'] for r in runs if r['applicable']}):
                ax.axhline(limit,color='gray',linestyle=':',label=f'Capacity {limit}')
        decorate(ax,title,label,percent)
        save(fig,name)
    fig,ax = plt.subplots(figsize=(8,4))
    draw(ax,summary,'mean_p_silence',label_suffix=' implied')
    draw(ax,summary,'observed_silence_rate',style='--',label_suffix=' observed')
    for n in sorted({r['metadata']['config']['signal_vocab_size']+1 for r in runs if r['applicable']}):
        ax.axhline(1/n,color='gray',ls=':',label=f'Uniform 1/{n}')
    decorate(ax,'SILENCE policy probability and observed choices','Probability / frequency')
    save(fig,'silence_policy_probability.png')
    fig,axes = plt.subplots(1,3,figsize=(14,4))
    for ax,metric,title in zip(axes,['hit_rate','mean_q_gap','mean_p_silence'],['State hit rate','Q preference gap','Implied SILENCE probability']):
        draw(ax,tables['age_summary'],metric,x='age_start')
        decorate(ax,title,metric,metric=='hit_rate',xlabel='Agent age (bin start)')
    save(fig,'vocal_learning_by_age.png')
    fig,axes = plt.subplots(1,max(1,len(conditions)),figsize=(6*max(1,len(conditions)),4),squeeze=False)
    for ax,c in zip(axes[0],conditions):
        source=summary.loc[summary.condition.eq(c)]
        for metric,label,color in [('first_fraction','First ever','tab:blue'),('stored_fraction','Stored revisit','tab:green'),('recreated_fraction','Post-eviction revisit','tab:red')]:
            rows=source.loc[source.metric.eq(metric)].sort_values('actual_window_end')
            if len(rows):
                ax.plot(rows.actual_window_end,rows['mean'],label=label,color=color)
                xx,yy,sd=(rows[col].to_numpy(float) for col in ('actual_window_end','mean','std'))
                ax.fill_between(xx,yy-sd,yy+sd,color=color,alpha=.12)
        decorate(ax,f'{c}: state encounter categories','Proportion (categories sum to 1)',True)
    save(fig,'vocal_state_novelty.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for c in conditions:
        eligible=tables['individuals'].loc[tables['individuals'].condition.eq(c) & tables['individuals'].eligible]
        if eligible.empty:
            selection[c]={'agent_ids':[]}
            continue
        seed=int(eligible.seed.min())
        eligible=eligible.loc[eligible.seed.eq(seed)].sort_values(['decisions','agent_id'],ascending=[False,True]).head(4)
        ids=[int(x) for x in eligible.agent_id]
        selection[c]={'seed':seed,'agent_ids':ids,'rule':f'Lowest seed; >= {min_decisions} decisions; top four decision exposures, then ID. All selected, including single age-bin records.'}
        for aid in ids:
            rows=tables['individual_ages'].loc[tables['individual_ages'].condition.eq(c)&tables['individual_ages'].seed.eq(seed)&tables['individual_ages'].agent_id.eq(aid)].sort_values('age_start')
            for ax,metric in zip(axes,['mean_q_gap','mean_p_silence']):
                ax.plot(rows.age_start,rows[metric],marker='.',label=f'{c} seed {seed}, ID {aid}',linewidth=1)
    for ax,metric in zip(axes,['mean_q_gap','mean_p_silence']):
        decorate(ax,'Individual '+metric,metric,xlabel='Agent age (bin start)')
    save(fig,'individual_vocal_learning.png')
    if 'communication_comparison' in tables and not tables['communication_comparison'].empty:
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        joined=tables['communication_comparison']
        food=joined.loc[joined.feature.eq('food_direction')]
        for ax,metric in zip(axes,['hit_rate','eviction_rate','mean_q_gap']):
            for c in conditions:
                block=food.loc[food.condition.eq(c)]
                ax.scatter(block[metric],block.corrected_mi_bits,s=18,color=colors[c],alpha=.7,label=c)
            decorate(ax,'Matched run/windows', 'Corrected within-agent food MI (bits)',xlabel=metric)
        save(fig,'vocal_learning_and_sender_mi.png')
    if 'age_sender_mi_summary' in tables:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,feature in zip(axes,['food_direction','water_direction']):
            rows=tables['age_sender_mi_summary'].loc[tables['age_sender_mi_summary'].feature.eq(feature)]
            draw(ax,rows,'mean_corrected_mi_bits',x='mean_age_bin')
            decorate(ax,'Matched individual-window '+feature,'Mean corrected within-agent MI (bits)',xlabel='Mean agent age in window (bin start)')
        save(fig,'age_and_sender_mi.png')
    return names,selection
