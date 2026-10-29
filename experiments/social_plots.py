"""Deterministic observational figures, 180 dpi, no seaborn or random jitter."""
from collections import Counter,defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plots(folder):
    windows=pd.read_csv(folder/'social_by_window.csv')
    runs=pd.read_csv(folder/'social_by_run.csv')
    intervals=pd.read_csv(folder/'social_interval_distribution.csv')
    conditions=sorted(runs.condition.unique())
    colors={name:f'C{i%10}' for i,name in enumerate(conditions)}
    with plt.rc_context({'font.size':9}):
        def save(fig,name):
            fig.tight_layout();fig.savefig(folder/name,dpi=180,metadata={'Software':'Artificial Civilization v0.7.0'})
            plt.close(fig)
        def line(ax,metric,title,ylabel):
            if windows.empty:
                ax.text(.5,.5,'No observation windows',ha='center',transform=ax.transAxes)
            for condition,b in windows.groupby('condition',sort=True):
                g=b.groupby('actual_window_end')[metric].agg(['mean','std','count'])
                ax.plot(g.index,g['mean'],marker='o',label=condition,color=colors[condition])
                ax.fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],alpha=.15,color=colors[condition])
                defined=g['std'].notna()
                if defined.any():
                    ax.errorbar(g.index[defined],g.loc[defined,'mean'],yerr=g.loc[defined,'std'],
                                fmt='none',capsize=3,color=colors[condition],alpha=.8)
                ax.scatter(b.actual_window_end,b[metric],s=12,color=colors[condition],alpha=.6)
            ax.set(title=title,xlabel='Completed tick (actual window end)',ylabel=ylabel)
            ax.grid(alpha=.2)
            ax.ticklabel_format(axis='y',useOffset=False)
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        line(axes[0],'first_encounters','Bounded-memory unknown contacts','Directed contacts / window')
        line(axes[1],'reunions','Remembered qualifying reunions','Directed reunions / window')
        fig.suptitle('Observed social contact: equal-seed mean ± sample SD')
        save(fig,'encounters_reunions_over_time.png')
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        line(axes[0],'mean_memory_occupancy','Post-observation occupancy','Entries / observation opportunity')
        line(axes[1],'full_memory_fraction','Full memories','Fraction of observation opportunities')
        save(fig,'memory_occupancy_over_time.png')
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        line(axes[0],'memory_hit_rate','Known visible contacts','Hits / directed visible contacts')
        line(axes[1],'evictions_per_observation','Memory evictions','Evictions / observation opportunity')
        save(fig,'memory_hits_evictions.png')
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        line(axes[0],'known_visible_fraction','Known before observation batch','Known / directed visible contacts')
        line(axes[1],'unknown_visible_fraction','Unknown before observation batch','Unknown / directed visible contacts')
        save(fig,'known_unknown_visible.png')
        fig,ax=plt.subplots(figsize=(7.5,3.8))
        b=intervals[intervals.scope.eq('run')&intervals.interval_kind.eq('reunion')]
        if b.empty:ax.text(.5,.5,'No qualifying recorded reunions',ha='center',transform=ax.transAxes)
        for number,((condition,seed),part) in enumerate(b.groupby(['condition','seed'],sort=True)):
            ax.step(part.elapsed_ticks,part['count'],where='mid',label=f'{condition}, seed {seed}',color=colors[condition],alpha=.7,linestyle=('-','--',':','-.')[number%4])
        ax.set(title='Observed reunion gaps (not inferred survival)',xlabel='Ticks since last direct sighting',ylabel='Directed reunion events per seed')
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        save(fig,'reunion_interval_distribution.png')
        hist=defaultdict(Counter);examples={}
        for chunk in pd.read_csv(folder/'social_by_individual.csv',chunksize=100000):
            for (condition,seed),part in chunk.groupby(['condition','seed'],sort=True):
                hist[(condition,seed)].update(part.observed_individuals)
                eligible=part[part.observations.gt(0)]
                if not eligible.empty:
                    candidate=(int(seed),int(eligible.individual_id.min()))
                    if condition not in examples or candidate<examples[condition]:examples[condition]=candidate
        fig,ax=plt.subplots(figsize=(8,3.8))
        for condition in conditions:
            for number,((name,seed),counts) in enumerate(sorted(hist.items())):
                if name!=condition:continue
                x=np.array(sorted(counts));cdf=np.cumsum([counts[k] for k in x])/sum(counts.values())
                ax.step(x,cdf,where='post',color=colors[condition],alpha=.7,label=f'{condition}, seed {seed}',linestyle=('-','--',':','-.')[number%4])
        ax.set(title='Distinct visually observed partners per individual',xlabel='Distinct visually observed partners',ylabel='Fraction of recorded individuals')
        fig.suptitle('Per-seed distributions; includes newborns with no observation opportunities')
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        save(fig,'distinct_social_contacts.png')
        fig,axes=plt.subplots(1,3,figsize=(12,3.8))
        for ax,metric,title in zip(axes,('population_end','births','deaths'),('Living population','Actual births','Actual deaths')):
            line(ax,metric,title,'Completed population' if metric=='population_end' else 'Explicit lifecycle events / window')
        fig.suptitle('Observed demography: no death inference from absence')
        save(fig,'social_demography.png')
        pieces=[]
        for chunk in pd.read_csv(folder/'social_individual_windows.csv',chunksize=100000):
            for condition,(seed,identity) in sorted(examples.items()):
                pieces.append(chunk[chunk.condition.eq(condition)&chunk.seed.eq(seed)&chunk.individual_id.eq(identity)])
        chosen=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()
        fig,axes=plt.subplots(1,3,figsize=(12,3.8))
        for condition,(seed,identity) in sorted(examples.items()):
            b=chosen[chosen.condition.eq(condition)].sort_values('actual_window_end')
            label=f'{condition}: seed {seed}, ID {identity}'
            for ax,metric in zip(axes,('mean_visible_count','mean_memory_occupancy','reunions')):
                ax.plot(b.actual_window_end,b[metric],marker='o',label=label,color=colors[condition])
        for ax,title in zip(axes,('Contacts / opportunity','Entries / opportunity','Reunions / window')):
            ax.set(title=title,xlabel='Actual window end');ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        fig.suptitle('Observed examples: lowest seed, then lowest ID with an observation; no outcome selection')
        save(fig,'individual_social_trajectories.png')
    return {condition:{'seed':seed,'individual_id':identity} for condition,(seed,identity) in sorted(examples.items())}
