"""Static scientific figures. Realizations are grouped within ecological seed."""
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS=dict(zip('ABCDE',plt.get_cmap('tab10').colors[:5]))


def plots(folder,limits):
    folder=Path(folder);runs=pd.read_csv(folder/'runs.csv');ticks=pd.read_csv(folder/'timeseries.csv')
    labels={c:f'{c} (eco n={g.seed.nunique()}, nuisance/eco={g.groupby("seed").realization.nunique().max() if c in "DE" else 0})' for c,g in runs.groupby('condition')}
    replication='; nuisance realizations/eco: '+', '.join(f'{c}={g.groupby("seed").realization.nunique().max() if c in "DE" else 0}' for c,g in runs.groupby('condition'))
    def finish(fig,name,title):
        fig.suptitle(title,fontsize=12)
        fig.text(.5,.01,'\n'.join(textwrap.wrap(limits+replication,110)),ha='center',va='bottom',fontsize=8)
        fig.tight_layout(rect=(0,.10,1,.93));fig.savefig(folder/(name+'.png'),dpi=180);plt.close(fig)
    def bar(ax,metric,title):
        values=runs.groupby(['condition','seed'])[metric].mean().reset_index()
        for i,(c,g) in enumerate(values.groupby('condition')):
            ax.errorbar(i,g[metric].mean(),yerr=g[metric].std(ddof=1),fmt='o',capsize=4,color=COLORS[c],label=labels[c])
            ax.scatter(i+np.linspace(-.12,.12,len(g)),g[metric],s=14,color=COLORS[c],alpha=.6)
        ax.set_xticks(range(len(labels)),labels.keys());ax.set_title(title);ax.set_ylabel(metric);ax.grid(alpha=.15)
    def bars(name,metrics,title):
        fig,axes=plt.subplots(1,len(metrics),figsize=(max(7,4*len(metrics)),4.8),squeeze=False)
        for ax,m in zip(axes[0],metrics):bar(ax,m,m)
        handles=[plt.Line2D([],[],marker='o',color=COLORS[c],label=labels[c]) for c in labels]
        axes[0,-1].legend(handles=handles,fontsize=7)
        finish(fig,name,title)
    fig,ax=plt.subplots(figsize=(9,5))
    for c,g in ticks.groupby('condition'):
        seed=g.groupby(['seed','tick']).population.mean().reset_index();a=seed.groupby('tick').population.agg(['mean','std','count'])
        x=a.index.to_numpy();mean=a['mean'].to_numpy();sd=a['std'].to_numpy()
        ax.plot(x,mean,label=labels[c],color=COLORS[c]);ax.fill_between(x,mean-sd,mean+sd,alpha=.12,color=COLORS[c])
    ax.set(xlabel='Tick',ylabel='Living population');ax.legend(fontsize=8);finish(fig,'population','Population: equal-seed mean ± sample SD')
    bars('births_deaths',['births','deaths'],'Cumulative births and deaths')
    bars('first_state_fraction',['first_state_fraction','fraction_states_once','mean_visits'],'Agent-state fragmentation; points are ecological seeds')
    bars('q_occupancy_eviction',['q_occupancy','evictions_per_decision','q_updates'],'Physical Q occupancy, churn and actual TD updates')
    bars('token_entropy',['token_entropy','effective_tokens','conditional_entropy'],'Empirical additional-token entropy (bits); conditional on original X')
    bars('resources',['food_consumed','water_consumed'],'Cumulative resource consumption')
    bars('movement',['toward_fraction','away_fraction','equal_fraction'],'Actual moved actors vs frozen visible target; not social preference')
    def hist(name,file,x,y,title,log=False):
        f=pd.read_csv(folder/file);fig,ax=plt.subplots(figsize=(9,5))
        for c,g in f.groupby('condition'):
            # Missing histogram bins within a run are real zero counts.
            pivot=g.groupby(['seed','realization',x])[y].sum().unstack(x,fill_value=0)
            pivot=pivot.groupby(level='seed').mean();mean=pivot.mean(axis=0)
            ax.plot(mean.index,mean,marker='.' if len(mean)<30 else None,label=labels[c],color=COLORS[c])
        if log:ax.set_xscale('log')
        ax.set(xlabel=x,ylabel=y);ax.legend(fontsize=8);finish(fig,name,title)
    hist('state_visits','state_visits.csv','visits','state_fraction','Lifetime visits per agent-state (death/end histories included)',True)
    hist('token_persistence','token_runs.csv','length','fraction','Observed token segments; death/end right-censored, not complete lifetimes',True)
    # Per-seed action rates; no decision pooling across seeds.
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(8)
    for j,(c,g) in enumerate(runs.groupby('condition')):
        rates=g[[f'physical_action_{i}' for i in range(8)]].div(g.decisions,axis=0)
        rates['seed']=g.seed.to_numpy();perseed=rates.groupby('seed').mean();means=perseed.mean();sd=perseed.std(ddof=1)
        ax.bar(x+(j-2)*.15,means,width=.14,yerr=sd,capsize=2,label=labels[c],color=COLORS[c])
    ax.set_xticks(x,['N','S','E','W','eat','drink','wait','reproduce']);ax.set_ylabel('Actions / decisions');ax.legend(fontsize=7)
    finish(fig,'physical_actions','Physical action distributions; original action ordering')
    q=pd.read_csv(folder/'probes.csv');fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,minimum in zip(axes,(2,5)):
        for c,g in q[q.min_visits.eq(minimum)].groupby('condition'):
            a=g.groupby(['seed','tick']).policy_tv.mean().reset_index()
            h=a.groupby('tick').policy_tv.agg(['mean','std','count']);ax.errorbar(h.index,h['mean'],yerr=h['std'],label=labels[c],color=COLORS[c],marker='.',capsize=2)
        ax.set(title=f'≥{minimum} row visits + ≥1 actual update',xlabel='Tick',ylabel='Supported pair policy TV');ax.legend(fontsize=6)
    finish(fig,'policy_dependence','Read-only fixed-original-state Q dependence; unsupported comparisons = NA')
    match=pd.read_csv(folder/'matching.csv');fig,axes=plt.subplots(1,5,figsize=(17,5))
    for ax,metric in zip(axes,('first_state_fraction','relative_mean_visits','token_entropy','conditional_entropy','same_token')):
        for j,c in enumerate(('D','E')):
            g=match[match.condition.eq(c)&match.metric.eq(metric)];v=g.groupby('seed').difference.mean()
            ax.scatter(np.full(len(v),j)+np.linspace(-.08,.08,len(v)),v,color=COLORS[c]);ax.errorbar(j,v.mean(),yerr=v.std(ddof=1),fmt='o',color='black')
            if len(g):ax.axhspan(-g.tolerance.iloc[0],g.tolerance.iloc[0],alpha=.1,color='grey')
        ax.set_xticks((0,1),('D − B','E − C'));ax.set_title(metric);ax.axhline(0,color='grey',lw=.8)
    finish(fig,'matching_quality','Predeclared matching tolerances (shaded); all mismatches retained')
    if runs[runs.condition.isin(['D','E'])].realization.nunique()>1:
        fig,axes=plt.subplots(1,2,figsize=(10,5))
        for ax,c in zip(axes,'DE'):
            for seed,g in runs[runs.condition.eq(c)].groupby('seed'):
                ax.plot(g.realization,g.population,marker='o',label=f'eco seed {seed}')
            ax.set(title=labels[c],xlabel='Independent nuisance realization',ylabel='Final population');ax.legend()
        finish(fig,'nuisance_sensitivity','Within-ecological-seed sensitivity; not additional independent replication')
    hist('vocal_actions','vocal_actions.csv','vocal_action','n','Passive Vocal Q choices (0=silence); no communication-benefit claim')


def prediction_plot(frame,folder,common_support=False):
    fig,axes=plt.subplots(1,2,figsize=(10,5))
    for ax,c in zip(axes,'BC'):
        for j,name in enumerate(('X_A','X_social_A','X_nuisance_A')):
            g=frame[frame.source_condition.eq(c)&frame.model.eq(name)]
            ax.errorbar(j,g.log_loss.mean(),yerr=g.log_loss.std(ddof=1),fmt='o',capsize=3)
            ax.scatter(j+np.linspace(-.08,.08,len(g)),g.log_loss,s=14)
        ax.set_xticks(range(3),('X,A','X,S,A','X,U,A'));ax.set(title=f'Fixed {c} experience',ylabel='Held-out reward-bin log loss (lower better)')
    fig.suptitle('Action-conditioned reward prediction; '+('exploratory common-support subset' if common_support else 'all held-out decisions'))
    fig.text(.5,.01,'Train eco seeds 3/4; test eco n=3 (0/1/2), one attached nuisance/seed; Q=256, 500 ticks. Sparse-cell backoff; unmatched nuisance.',ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.08,1,.93));fig.savefig(Path(folder)/'predictive_information.png',dpi=180);plt.close(fig)
