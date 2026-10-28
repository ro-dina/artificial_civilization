"""Headless observational figures. Conditional identity coverage is always visible."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

COLORS={'C':'tab:blue','D':'tab:orange'}


def plots(tables,summary,mi,trace,selection,options,directory):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    conditions=sorted(tables['persistence_run_summary'].condition.unique())
    colors={c:COLORS.get(c.split(':')[0],'tab:gray') for c in conditions}
    names=[]
    def save(fig,name):
        fig.tight_layout()
        fig.text(.5,-.025,'C: listening OFF; D: listening ON; both Vocal Q ON. Equal-seed means ± sample SD; no interpolation.\n'
                 'Same-state retained episodes only unless labeled otherwise. Identity rates describe the identifiable subset, not all policies.',
                 ha='center',va='top',fontsize=8)
        fig.savefig(Path(directory)/name,dpi=180,bbox_inches='tight',metadata={'Software':'Vocal persistence v0.6.3'})
        plt.close(fig);names.append(name)

    def draw(ax,frame,x,metric,label='',eligible=None,style='-',points=False):
        if frame.empty or metric not in frame:return
        for c in conditions:
            rows=frame.loc[frame.condition.eq(c)].copy()
            if eligible is not None:rows.loc[~eligible(rows),metric]=np.nan
            grouped=rows.groupby(x,sort=True)[metric].agg(['mean','std','count'])
            good=grouped['count'].gt(0)
            if not good.any():continue
            xx=grouped.index.to_numpy(float);yy=grouped['mean'].to_numpy(float);sd=grouped['std'].to_numpy(float)
            n=grouped.loc[good,'count']
            ax.plot(xx,yy,marker='o' if points else '.',linestyle='None' if points else style,color=colors[c],
                    label=f'{c} {label} (n={int(n.min())}–{int(n.max())} seeds)')
            finite=np.isfinite(sd)&np.isfinite(yy)
            if points:
                ax.errorbar(xx[finite],yy[finite],yerr=sd[finite],fmt='none',color=colors[c],capsize=3)
            else:ax.fill_between(xx,yy-sd,yy+sd,color=colors[c],alpha=.12,linewidth=0)

    def decorate(ax,title,xlabel,ylabel,percent=False):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2)
        if percent:ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylim(-.02,1.02)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
        else:ax.text(.5,.5,'No eligible / identifiable observations',ha='center',transform=ax.transAxes)

    w=tables['persistence_by_window']
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    draw(axes[0],w,'actual_window_end','preferred_action_match_rate',points=True,
         eligible=lambda f:f.id_pairs.ge(options.min_identity_pairs))
    decorate(axes[0],'Unique preferred-action persistence (conditional subset)','Simulation tick','Same preferred action / identifiable pairs',True)
    draw(axes[1],w,'actual_window_end','identity_coverage',points=True)
    decorate(axes[1],f'Coverage; upper panel needs ≥{options.min_identity_pairs} identifiable pairs per seed/window',
             'Simulation tick','Identifiable unique pairs / all retained pairs',True)
    save(fig,'preference_persistence_over_time.png')
    fig,ax=plt.subplots(figsize=(9,4))
    for metric,label,style in [('choice_match_rate','all pairs','-'),('exploit_choice_match_rate','both exploited','--'),
                              ('explore_choice_match_rate','≥1 explored',':')]:draw(ax,w,'actual_window_end',metric,label,style=style)
    decorate(ax,'Behavioral choice consistency; ties/exploration can change samples','Simulation tick','Matching choices / corresponding pairs',True)
    save(fig,'choice_consistency_over_time.png')
    visits=tables['persistence_by_visit_index']
    visits=visits.loc[visits.visit_kind.eq('episode')&visits.visit_index.le(options.plot_visits)]
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    draw(axes[0],visits,'visit_index','mean_gap');decorate(axes[0],'Preference gap by episode visit','Visit number','Mean top-two Q gap')
    draw(axes[1],visits,'visit_index','decisions');decorate(axes[1],'Exposure falls at later visits; CSV retains every visit index','Visit number','Encounter count / seed')
    axes[1].set_xticks(sorted({1,2,3,4,5,10,15,options.plot_visits}&set(range(1,options.plot_visits+1))))
    if not visits.empty:axes[1].set_yscale('log')
    save(fig,'preference_gap_by_visit.png')
    intervals=tables['state_revisit_intervals'].copy();fig,ax=plt.subplots(figsize=(10,4))
    if not intervals.empty:
        intervals['bin']=np.floor(np.log2(intervals.elapsed_ticks)).astype(int)
        b=intervals.groupby(['condition','seed','bin'],sort=True)['count'].sum().reset_index()
        bins=range(int(b.bin.max())+1)
        for ci,c in enumerate(conditions):
            means=[];sds=[]
            for i in bins:
                values=[]
                for seed,rows in intervals.loc[intervals.condition.eq(c)].groupby('seed',sort=True):
                    denom=rows['count'].sum();values.append(rows.loc[rows.bin.eq(i),'count'].sum()/denom if denom else np.nan)
                means.append(np.mean(values) if values else np.nan);sds.append(np.std(values,ddof=1) if len(values)>1 else np.nan)
            xx=np.arange(len(bins))+(ci-(len(conditions)-1)/2)*.35
            ax.bar(xx,means,.35,color=colors[c],label=c)
            valid=np.isfinite(sds);ax.errorbar(xx[valid],np.array(means)[valid],yerr=np.array(sds)[valid],fmt='none',color='black',capsize=2)
        ax.set_xticks(range(len(bins)),['1' if i==0 else f'{2**i}–{2**(i+1)-1}' for i in bins])
    decorate(ax,'Actual intervals between retained encounters (log2 bins)','Elapsed ticks','Proportion of retained intervals')
    save(fig,'state_revisit_intervals.png')
    fig,axes=plt.subplots(1,max(1,len(conditions)),figsize=(6*max(1,len(conditions)),4),squeeze=False)
    s=tables['silence_preference_transitions'];categories=['unique_max','tied_max','not_max']
    for ax,c in zip(axes[0],conditions):
        # Aggregate transition counts within each seed first, then equal-seed rates.
        rows=s.loc[s.condition.eq(c)];matrix=np.full((3,3),np.nan)
        for i,a in enumerate(categories):
            for j,b in enumerate(categories):
                vals=[]
                for seed,block in rows.groupby('seed',sort=True):
                    origin=block.loc[block.from_category.eq(a)]
                    denom=origin['count'].sum()
                    if denom:vals.append(origin.loc[origin.to_category.eq(b),'count'].sum()/denom)
                if vals:matrix[i,j]=np.mean(vals)
        im=ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=1,cmap='viridis')
        for i in range(3):
            for j in range(3):
                if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]<.5 else 'black')
        origin_labels=[]
        for a in categories:
            counts=[int(block.loc[block.from_category.eq(a),'count'].sum()) for _,block in rows.groupby('seed',sort=True)]
            counts=[n for n in counts if n>0]
            origin_labels.append(a+f'\nn={len(counts)} seeds; '+(f'{min(counts)}–{max(counts)} pairs' if counts else 'no origin pairs'))
        ax.set_xticks(range(3),categories,rotation=20);ax.set_yticks(range(3),origin_labels)
        ax.set(title=f'{c}: SILENCE maximum status (not full policy)',xlabel='Next encounter',ylabel='Previous encounter')
        fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    save(fig,'silence_preference_transitions.png')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    ages=tables['persistence_by_age']
    draw(axes[0],ages,'age_start','preferred_action_match_rate',points=True,eligible=lambda f:f.id_pairs.ge(options.min_identity_pairs))
    decorate(axes[0],'Cross-sectional age profile; conditional identities','Age at second encounter (bin start)','Preferred-action match rate',True)
    dev=tables['paired_episode_development']
    draw(axes[1],dev,'first_age_start','mean_gap_change',points=True)
    decorate(axes[1],f'Paired same-episode first/last {options.paired_block} encounters (≥{options.min_episode_encounters})',
             'Age at first encounter (bin start)','Mean late − early Q gap')
    save(fig,'preference_stability_by_age.png')
    kinds=tables['retained_vs_recreated'];fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    if not kinds.empty:
        counts=kinds.pivot_table(index=['run','condition','seed','actual_window_end'],columns='encounter_class',values='decisions',aggfunc='sum',fill_value=0).reset_index()
        for col in ('retained','recreated'):
            if col not in counts:counts[col]=0
        den=counts.retained+counts.recreated
        counts['post_eviction_fraction']=counts.recreated/den.where(den>0)
        draw(axes[0],counts,'actual_window_end','post_eviction_fraction',points=True)
        draw(axes[1],counts,'actual_window_end','recreated',points=True)
    decorate(axes[0],'Post-eviction revisits are separate episodes','Simulation tick','Recreated / all revisits')
    decorate(axes[1],'Sparse recreation exposure','Simulation tick','Recreated encounters / seed')
    save(fig,'retained_vs_recreated.png')
    fig,ax=plt.subplots(figsize=(10,4));turn=tables['agent_turnover']
    for metric,label,style in [('active_agents','active','-'),('first_time_learners','first-time','--'),('actual_deaths','actual deaths',':')]:
        draw(ax,turn,'actual_window_end',metric,label,style=style)
    decorate(ax,'Agent participation and actual deaths; absence alone is not death','Simulation tick','Agents / seed/window')
    save(fig,'agent_turnover_over_time.png')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    if not mi.empty:
        rows=mi.loc[mi.feature.eq('food_direction')]
        draw(axes[0],rows,'actual_window_end','corrected_within_agent_mi_bits','corrected')
        draw(axes[0],rows,'actual_window_end','raw_within_agent_mi_bits','raw',style='--')
        for c in conditions:
            b=rows.loc[rows.condition.eq(c)&rows.id_pairs.ge(options.min_identity_pairs)]
            axes[1].scatter(b.preferred_action_match_rate,b.corrected_within_agent_mi_bits,color=colors[c],s=18,alpha=.7,label=c)
    decorate(axes[0],'Verified same-run within-agent V/food MI','Simulation tick','MI (bits)')
    decorate(axes[1],'Matched windows; conditional subset, no causal fit','Identifiable preferred-action match rate','Corrected within-agent MI (bits)')
    save(fig,'persistence_vs_sender_mi.png')
    # Two exposure-selected state episodes per condition; no interesting-outcome selection.
    cols=max(1,len(conditions)*options.plot_episodes);fig,axes=plt.subplots(3,cols,figsize=(5*cols,9),squeeze=False)
    col=0
    for c in conditions:
        selected=selection.get(c,{});episodes=selected.get('episodes',[])
        for ep in episodes:
            rows=trace.loc[trace.run.eq(selected['run'])&trace.agent_id.eq(ep['agent_id'])&trace.state_id.eq(ep['state_id'])&trace.episode.eq(ep['episode'])].sort_values('tick')
            if col>=cols:break
            title=f"{c} seed {selected['seed']} / ID {ep['agent_id']}\nstate {ep['state_id']}, episode {ep['episode']} ({ep['encounters']} encounters)"
            axes[0,col].plot(rows.age,rows.gap,'.-',color=colors[c]);axes[0,col].set(title=title,ylabel='Q gap')
            axes[1,col].plot(rows.age,rows.p_silence,'.-',color=colors[c]);axes[1,col].set(ylabel='Implied P(SILENCE)')
            for exploratory,marker,label in [(False,'.','exploited choice'),(True,'o','explored choice')]:
                b=rows.loc[rows.exploratory.eq(exploratory)]
                axes[2,col].scatter(b.age,b.selected_action,marker=marker,s=12,color=colors[c],alpha=.6,label=label)
            known=rows.loc[rows.preferred_action.notna()]
            axes[2,col].scatter(known.age,known.preferred_action,marker='x',s=20,color='black',label='identified unique max')
            axes[2,col].set(xlabel='Agent age at recorded encounter',ylabel='Action index (0=SILENCE)');axes[2,col].legend(fontsize=7)
            for ax in axes[:,col]:ax.grid(alpha=.2)
            col+=1
    for j in range(col,cols):
        for ax in axes[:,j]:ax.text(.5,.5,'No eligible episode',ha='center',transform=ax.transAxes)
    save(fig,'individual_preference_trajectories.png')
    return names
