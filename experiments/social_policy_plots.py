"""Discrete-window policy diagnostics; all uncertainty is across independent seeds."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS={'A':'#0072B2','B':'#009E73','C':'#D55E00','D':'#CC79A7'}


def label(b):
    r=b.iloc[0]
    return f'{r.condition}: N={r.formation_threshold}, {r.eviction_policy}, K={r.capacity}, Nc={r.candidate_capacity}, tau={r.tau:g}'


def plots(folder):
    run=pd.read_csv(folder/'run_summary.csv');windows=pd.read_csv(folder/'window_summary.csv')
    age=pd.read_csv(folder/'age_summary.csv');pairs=pd.read_csv(folder/'paired_by_seed.csv');dist=pd.read_csv(folder/'distributions.csv')
    files=[]
    with plt.rc_context({'font.size':8}):
        def save(fig,name):
            keys=run[['condition','formation_threshold','eviction_policy']].drop_duplicates().sort_values('condition')
            key='; '.join(f'{r.condition}=N{r.formation_threshold}/{r.eviction_policy}' for r in keys.itertuples(index=False))
            footer=(key+'\n'+f'K={sorted(map(int,run.capacity.unique()))}; Nc={sorted(map(int,run.candidate_capacity.unique()))}; '
                    f'tau={sorted(map(float,run.tau.unique()))}; seeds={run.seed.nunique()}; sample SD ddof=1. Tau is unused for LRU.')
            fig.text(.5,.01,footer,ha='center',va='bottom',fontsize=6)
            fig.tight_layout(rect=(0,.08,1,1));fig.savefig(folder/name,dpi=180,metadata={'Software':'Artificial Civilization v0.7.1'});plt.close(fig);files.append(name)
        def categorical(ax,frame,metric,title,units):
            groups=sorted(frame.groupby('treatment',sort=False),key=lambda item:(item[1].condition.iloc[0],item[1].capacity.iloc[0],item[1].tau.iloc[0]));ticks=[]
            for j,(key,b) in enumerate(groups):
                x=j;y=b[metric].dropna();ticks.append(f'{b.condition.iloc[0]}\nK{b.capacity.iloc[0]} T{b.tau.iloc[0]:g}')
                if y.empty:ax.text(x,.02,'NA',ha='center',transform=ax.get_xaxis_transform());continue
                offset=np.linspace(-.1,.1,len(y)) if len(y)>1 else np.zeros(1)
                color=COLORS[b.condition.iloc[0]]
                ax.scatter(x+offset,y,s=15,color=color,alpha=.6)
                ax.errorbar(x,y.mean(),yerr=y.std(ddof=1) if len(y)>1 else None,fmt='D',ms=4,capsize=3,color=color)
            ax.set_xticks(range(len(groups)),ticks,fontsize=7);ax.set(title=title,ylabel=units);ax.grid(axis='y',alpha=.2)
        def temporal(metric,name,title,units):
            if windows.empty:return
            if windows.window_start.nunique()==1:
                fig,ax=plt.subplots(figsize=(max(7,len(run.treatment.unique())*.6),4))
                categorical(ax,run,metric,title+f' (one window through tick {int(run.final_tick.max())})',units)
                fig.suptitle('Points=seeds; diamond=equal-seed mean ± sample SD; n='+str(run.seed.nunique()))
            else:
                fig,ax=plt.subplots(figsize=(11,5))
                spans=sorted(windows[['window_start','actual_window_end']].drop_duplicates().itertuples(index=False,name=None))
                keys=list(windows.treatment.unique());width=.7/max(1,len(keys))
                for j,(key,b) in enumerate(windows.groupby('treatment',sort=True)):
                    color=COLORS[b.condition.iloc[0]];labeled=False
                    for n,span in enumerate(spans):
                        y=b[b.window_start.eq(span[0])&b.actual_window_end.eq(span[1])][metric].dropna()
                        if y.empty:continue
                        x=n+(j-(len(keys)-1)/2)*width
                        ax.errorbar(x,y.mean(),yerr=y.std(ddof=1) if len(y)>1 else None,fmt='o',ms=3,capsize=2,color=color,label=label(b) if not labeled else None)
                        labeled=True
                ax.set_xticks(range(len(spans)),[f'({a},{b}]' for a,b in spans]);ax.set(title=title,xlabel='Discrete observed windows; no interpolation',ylabel=units)
                ax.grid(axis='y',alpha=.2)
                if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6,loc='best',title=f'Equal seeds, SD ddof=1, n≤{run.seed.nunique()}')
            save(fig,name)
        temporal('mean_occupancy','memory_occupancy.png','Long-term occupancy','Entries / observation opportunity')
        temporal('fraction_observations_full','memory_at_capacity.png','Full long-term memories','Full / observation opportunities')
        temporal('evictions','long_term_evictions.png','Long-term memory evictions','Eviction events / window')
        temporal('recognition_eligible','eligible_recognition.png','Recognition among historically eligible contacts','Pre-update hits / prior-five-consecutive eligible contacts')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        categorical(axes[0],run,'cumulative_ever_committed_fraction','Visually observed pairs ever committed','Unique committed / unique observed observer-target pairs')
        categorical(axes[1],run,'cumulative_candidate_conversion','Temporary candidate conversion (NA if immediate)','Candidate commits / candidates created, cumulative')
        fig.suptitle('Formation: equal-seed mean ± sample SD; points=seeds; n='+str(run.seed.nunique()))
        save(fig,'candidate_formation.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        frame=run.copy()
        frame['eligible_eviction_failure']=frame.eligible_forgotten_misses/frame.eligible_contacts.replace(0,np.nan)
        frame['eligible_formation_failure']=frame.eligible_never_committed_misses/frame.eligible_contacts.replace(0,np.nan)
        categorical(axes[0],frame,'eligible_eviction_failure','Eligible, previously committed but forgotten','Failures / same historically eligible contacts')
        categorical(axes[1],frame,'eligible_formation_failure','Eligible, never committed','Failures / same historically eligible contacts')
        save(fig,'recognition_failure_causes.png')
        fig,ax=plt.subplots(figsize=(9,4))
        if run.capacity.nunique()==1:
            categorical(ax,run,'recognition_eligible',f'Fixed K={run.capacity.iloc[0]}: capacity effect NOT measured','Eligible recognition rate')
        else:
            capacities=sorted(run.capacity.unique())
            for condition,b in run.groupby('condition',sort=True):
                for j,k in enumerate(capacities):
                    y=b[b.capacity.eq(k)].recognition_eligible.dropna();x=j+(0 if condition=='A' else .15)
                    if y.empty:continue
                    r=b.iloc[0]
                    description=f'{condition}: N={r.formation_threshold}, {r.eviction_policy}, Nc={r.candidate_capacity}, tau={r.tau:g}'
                    ax.errorbar(x,y.mean(),yerr=y.std(ddof=1) if len(y)>1 else None,fmt='o',capsize=3,color=COLORS[condition],label=description if j==0 else None)
            ax.set_xticks(range(len(capacities)),capacities);ax.set(xlabel='Discrete long-term K (other policy settings in metadata)',ylabel='Eligible recognition rate',title='Capacity sensitivity: equal seeds ± sample SD')
            ax.legend(fontsize=7)
        save(fig,'capacity_sensitivity.png')
        fig,ax=plt.subplots(figsize=(11,4.5))
        if age.empty:ax.text(.5,.5,'No observation-age exposure',ha='center')
        else:
            starts=sorted(age.age_start.unique());keys=sorted(age.treatment.unique());width=.7/max(1,len(keys))
            for j,(key,b) in enumerate(age.groupby('treatment',sort=True)):
                labeled=False
                for n,start in enumerate(starts):
                    d=b[b.age_start.eq(start)];y=d.recognition_eligible.dropna()
                    if y.empty:continue
                    ax.errorbar(n+(j-(len(keys)-1)/2)*width,y.mean(),yerr=y.std(ddof=1) if len(y)>1 else None,fmt='o',ms=3,capsize=2,color=COLORS[b.condition.iloc[0]],label=label(b) if not labeled else None)
                    labeled=True
            ax.set_xticks(range(len(starts)),[f'[{a},{a+int(age.age_end.iloc[0]-age.age_start.iloc[0])})' for a in starts]);ax.set(xlabel='Observer pre-action age bins',ylabel='Eligible recognition (finite seed n in age CSV)',title='Age and exposure: current commitments never counted as hits')
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6)
        save(fig,'recognition_by_age.png')
        capacities=sorted(run.capacity.unique())
        fig,axes=plt.subplots(1,len(capacities),figsize=(max(10,4*len(capacities)),4.5),squeeze=False)
        b=dist[dist.metric.eq('strength_at_eviction')&dist.reason.eq('eviction')]
        for ax,k in zip(axes[0],capacities):
            subset=b[b.capacity.eq(k)]
            if subset.empty:ax.text(.5,.5,'No observed long-term evictions',transform=ax.transAxes,ha='center')
            for j,((treatment,seed),part) in enumerate(subset.groupby(['treatment','seed'],sort=True)):
                part=part.sort_values('value');cdf=part['count'].cumsum()/part['count'].sum()
                ax.step(part.value,cdf,where='post',alpha=.6,color=COLORS[part.condition.iloc[0]],linestyle=('-','--',':')[j%3],label=f'{part.condition.iloc[0]} T{part.tau.iloc[0]:g}, seed {seed}')
            ax.set(xlabel='Sightings in evicted residence (includes formation)',ylabel='Per-seed eviction ECDF',title=f'Strength at eviction, K={k}')
            if not subset.empty:ax.legend(fontsize=6,ncol=2)
        fig.suptitle('Conditional eviction distributions, not memory quality; no evictions means no ECDF')
        save(fig,'strength_at_eviction.png')
        fig,ax=plt.subplots(figsize=(max(8,len(pairs.alternative.unique())*.6) if not pairs.empty else 8,4.5))
        b=pairs[pairs.metric.eq('recognition_eligible')]
        if b.empty:ax.text(.5,.5,'No paired eligible-rate comparisons',transform=ax.transAxes,ha='center')
        names=[]
        groups=sorted(b.groupby(['alternative','control'],sort=False),key=lambda item:(item[1].capacity.iloc[0],item[1].tau.iloc[0],item[0]))
        for j,((alternative,control),part) in enumerate(groups):
            y=part.difference.dropna();names.append(part.contrast.iloc[0]+f'\nK{part.capacity.iloc[0]} T{part.tau.iloc[0]:g}')
            if y.empty:
                ax.text(j,.02,'NA',ha='center',transform=ax.get_xaxis_transform());continue
            ax.scatter(j+np.linspace(-.08,.08,len(y)),y,s=15,color='C0');ax.errorbar(j,y.mean(),yerr=y.std(ddof=1) if len(y)>1 else None,fmt='D',capsize=3,color='black')
        ax.set_xticks(range(len(names)),names,fontsize=7);ax.axhline(0,color='grey',lw=.8)
        ax.set(title='Paired-seed recognition differences: equal-seed mean ± sample SD',ylabel='Alternative minus control, identical eligible exposure')
        save(fig,'paired_recognition.png')
    return files
