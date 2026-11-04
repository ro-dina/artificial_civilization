"""Deterministic headless scientific figures; seeds remain replication units."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS={'A':'#0072B2','B':'#009E73','C':'#D55E00','C_decay':'#CC79A7','B_large':'#56B4E9','C_large':'#E69F00'}
ACTIONS=('N','S','E','W','eat','drink','wait','reproduce')


def plots(folder):
    run=pd.read_csv(folder/'run_summary.csv');ticks=pd.read_csv(folder/'timeseries.csv');windows=pd.read_csv(folder/'windows.csv')
    dist=pd.read_csv(folder/'distributions.csv');q=pd.read_csv(folder/'probes.csv');actions=pd.read_csv(folder/'action_probabilities.csv')
    vocal=pd.read_csv(folder/'vocal_actions.csv');conditions=sorted(run.condition.unique());files=[]
    with plt.rc_context({'font.size':8}):
        def save(fig,name):
            fig.text(.5,.01,'Seeds are replications (n≤'+str(run.seed.nunique())+'). Means ± sample SD where shown; ECDFs are per-seed.\nCounts/denominators in CSV. No social rewards; no preference/communication claim.',ha='center',fontsize=6)
            fig.tight_layout(rect=(0,.05,1,1));fig.savefig(folder/name,dpi=180,metadata={'Software':'Artificial Civilization v0.7.2'});plt.close(fig);files.append(name)
        def categorical(ax,frame,metric,title,ylabel):
            for j,c in enumerate(conditions):
                y=frame[frame.condition.eq(c)][metric].dropna();color=COLORS.get(c,'grey')
                if y.empty:ax.text(j,.02,'NA',transform=ax.get_xaxis_transform(),ha='center');continue
                ax.scatter(j+np.linspace(-.07,.07,len(y)),y,s=12,color=color,alpha=.65)
                ax.errorbar(j,y.mean(),yerr=y.std(ddof=1) if len(y)>1 else None,fmt='D',capsize=3,color=color,ms=4)
            ax.set_xticks(range(len(conditions)),conditions);ax.set(title=title,ylabel=ylabel);ax.grid(axis='y',alpha=.2)
        def temporal(ax,frame,metric,title,ylabel,continuous=False):
            if frame.empty:ax.text(.5,.5,'No observations',transform=ax.transAxes,ha='center');return
            if frame.tick.nunique()==1:
                categorical(ax,frame,metric,title+' (one observation window)',ylabel);return
            for c,b in frame.groupby('condition',sort=True):
                g=b.groupby('tick')[metric].agg(['mean','std','count']);color=COLORS.get(c,'grey')
                if continuous:
                    ax.plot(g.index,g['mean'],color=color,label=c)
                    ax.fill_between(g.index,g['mean']-g['std'],g['mean']+g['std'],color=color,alpha=.15)
                else:ax.errorbar(g.index,g['mean'],yerr=g['std'],fmt='o',ms=3,capsize=2,label=c,color=color)
            ax.set(title=title,xlabel='Tick' if continuous else 'Completed discrete window end',ylabel=ylabel);ax.legend(fontsize=7);ax.grid(alpha=.2)
        for metric,name,title,y in (('population','population_over_time.png','Population','Living agents'),
                                    ('distinct_global_states','physical_state_diversity.png','Distinct physical decision keys across agents','Cumulative distinct keys')):
            fig,ax=plt.subplots(figsize=(9,4));temporal(ax,ticks,metric,title,y,True);save(fig,name)
        mortality=ticks.copy();mortality['mortality']=mortality.deaths/(mortality.initial_population+mortality.births).replace(0,np.nan)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        temporal(axes[0],mortality,'mortality','Cumulative deaths / entrants (not a hazard)','Deaths / (founders + births)',True)
        for c,g in dist[dist.metric.eq('survival_age')&dist.reason.ne('run_end_censored')].groupby('condition',sort=True):
            for seed,b in g.groupby('seed',sort=True):
                b=b.groupby('value')['count'].sum().sort_index();axes[1].step(b.index,b.cumsum()/b.sum(),where='post',color=COLORS.get(c),alpha=.6,label=f'{c} seed {seed}')
        axes[1].set(title='Observed death age, conditional on death',xlabel='Age at death',ylabel='Per-seed ECDF; survivors excluded/right-censored')
        if axes[1].get_legend_handles_labels()[0]:axes[1].legend(fontsize=6)
        else:axes[1].text(.5,.5,'No observed deaths',ha='center',transform=axes[1].transAxes)
        save(fig,'survival_and_mortality.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        temporal(axes[0],windows,'mean_physical_occupancy','Physical Q occupancy','Rows / actor decision')
        temporal(axes[1],windows,'q_evictions_per_decision','Physical Q eviction','Eviction events / decisions')
        save(fig,'q_occupancy_and_evictions.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        temporal(axes[0],windows,'visible_fraction','Visible-individual exposure','Decisions with visible others / decisions')
        temporal(axes[1],windows,'remembered_fraction','Pre-update remembered exposure','Decisions with remembered others / decisions')
        save(fig,'social_exposure.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,prefix,title in zip(axes,('','remembered_'),('Nearest visible','Nearest pre-remembered')):
            for j,metric in enumerate(('toward_fraction','away_fraction')):
                for i,c in enumerate(conditions):
                    y=run[run.condition.eq(c)][prefix+metric].dropna()
                    if y.empty:continue
                    ax.errorbar(i+(j-.5)*.2,y.mean(),yerr=y.std(ddof=1) if len(y)>1 else None,fmt=('o' if j==0 else 's'),capsize=3,color=COLORS.get(c),label=('toward' if j==0 else 'away') if i==0 else None)
            ax.set_xticks(range(len(conditions)),conditions);ax.set(title=title+' frozen target',ylabel='Actual target movements: toward/away fraction')
            if ax.get_legend_handles_labels()[0]:ax.legend()
        save(fig,'movement_relative.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,reason,title in zip(axes,('observed_absence','run_end_censored'),('Completed contact episodes','End-censored ongoing contacts')):
            for (c,seed),b in dist[dist.metric.eq('contact_length')&dist.reason.eq(reason)].groupby(['condition','seed'],sort=True):
                b=b.sort_values('value');ax.step(b.value,b['count'].cumsum()/b['count'].sum(),where='post',color=COLORS.get(c),alpha=.6,label=f'{c} seed {seed}')
            ax.set(title=title,xlabel='Consecutive actual visual opportunities',ylabel='Per-seed conditional ECDF')
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6)
        save(fig,'contact_durations.png')
        fig,ax=plt.subplots(figsize=(8,4));categorical(ax,run,'reunion_recognition','Long-gap recognition (shared definition, differing histories)','Pre-remembered / historical long-gap contacts')
        save(fig,'reunion_recognition.png')
        fig,axes=plt.subplots(1,len(conditions),figsize=(max(10,4*len(conditions)),4),squeeze=False)
        for ax,c in zip(axes[0],conditions):
            b=actions[actions.condition.eq(c)]
            for j,category in enumerate(('none','unremembered','remembered')):
                g=b[b.social_category.eq(category)].groupby('action').probability.agg(['mean','std'])
                if g.empty:continue
                ax.errorbar(g.index+(j-1)*.15,g['mean'],yerr=g['std'],fmt='o',ms=3,capsize=2,label=category)
            ax.set_xticks(range(8),ACTIONS,rotation=45);ax.set(title=c+' observed exposure associations',ylabel='Per-seed conditional action probability')
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6)
        save(fig,'actions_by_social_state.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        temporal(axes[0],windows,'signals_per_decision','Vocal production','Signals / decisions')
        temporal(axes[1],windows,'identified_fraction','Current resolved auditory exposure','IDENTIFIED / decisions')
        save(fig,'vocal_and_auditory.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,kind in zip(axes,('all_social','memory_only')):
            b=q[q.kind.eq(kind)&q.min_visits.eq(5)] if not q.empty else q
            if not b.empty and b.tick.nunique()==1:
                categorical(ax,b,'tv_mean',kind+f': checkpoint {int(b.tick.iloc[0])}','Policy TV; NA without supported pairs')
            elif b.empty or not b.tv_mean.notna().any():
                ax.text(.5,.5,'No qualified pairs at any checkpoint\nRate unavailable, not zero',transform=ax.transAxes,ha='center')
                ax.set(title=kind,ylabel='Mean policy TV across supported seeds')
            else:
                for index,(c,g) in enumerate(b.groupby('condition',sort=True)):
                    grouped=g.groupby('tick').tv_mean.agg(['mean','std','count']);grouped=grouped[grouped['count'].gt(0)]
                    if grouped.empty:continue
                    x=grouped.index+(index-(b.condition.nunique()-1)/2)*12
                    ax.errorbar(x,grouped['mean'],yerr=grouped['std'],fmt='o',ms=3,capsize=2,label=c,color=COLORS.get(c))
                    for xx,(tick,row) in zip(x,grouped.iterrows()):
                        n=int(row['count']);pairs=int(g[g.tick.eq(tick)].pairs.sum())
                        ax.annotate(f'n={n}; pairs={pairs}',(xx,row['mean']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=5,color=COLORS.get(c))
                ax.set(title=kind,xlabel='Observed checkpoints; no interpolation',ylabel='Mean policy TV across supported seeds')
                ax.set_xticks(sorted(b.tick.unique()));ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.suptitle('Trained retained Q rows: ≥5 current-row visits and ≥1 TD update; support changes over time',fontsize=9)
        save(fig,'counterfactual_policy_sensitivity.png')
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,metric in zip(axes,('hunger','thirst','energy')):
            b=dist[dist.metric.eq(metric)]
            for c,g in b.groupby('condition',sort=True):
                cell=g.groupby(['seed','value'])['count'].sum().reindex(pd.MultiIndex.from_product([g.seed.unique(),range(10)],names=['seed','value']),fill_value=0)
                norm=cell/cell.groupby(level=0).transform('sum');avg=norm.groupby(level=1).mean()
                ax.plot(avg.index,avg.values,marker='o',label=c,color=COLORS.get(c))
            ax.set(title='Pre-action '+metric,xlabel='Decile of configured maximum',ylabel='Equal-seed decision-weighted fraction')
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        save(fig,'physiology_distributions.png')
        fig,ax=plt.subplots(figsize=(10,4))
        if not vocal.empty:
            for index,c in enumerate(conditions):
                part=vocal[vocal.condition.eq(c)]
                if part.empty:continue
                g=part.groupby(['seed','vocal_action']).n.sum().reindex(pd.MultiIndex.from_product([part.seed.unique(),range(int(part.vocal_action_count.max()))],names=['seed','vocal_action']),fill_value=0)
                prob=g/g.groupby(level=0).transform('sum')
                g=prob.groupby(level=1).agg(['mean','std'])
                ax.errorbar(g.index+(index-(len(conditions)-1)/2)*.12,g['mean'],yerr=g['std'],fmt='o',ms=3,capsize=2,label=c,color=COLORS.get(c))
            ax.legend()
        ax.set(title='Vocal choices: 0=SILENCE, i+1=meaningless signal i',xlabel='Categorical vocal action ID; no similarity between neighbors',ylabel='Equal-seed choice frequency')
        save(fig,'vocal_action_distribution.png')
    return files
