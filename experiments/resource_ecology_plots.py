"""Headless v080 figures with independent-seed means and sample SD."""
from pathlib import Path
from time import perf_counter
import argparse,textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.resource_ecology_experiments import ROOT,ANALYSIS,save_json,peak
from simulation.vocal_learning_observer import hash_file

COLORS=dict(zip(('E0','E1','E2','E3'),plt.get_cmap('tab10').colors[:4]))
MODES=dict(zip('ABC',plt.get_cmap('tab10').colors[:3]))


def plots(phase='primary',output=None,data_dir=None):
    folder=Path(data_dir) if data_dir else ANALYSIS/phase;out=Path(output) if output else folder/'figures'
    if out.exists():raise ValueError('Refusing existing figures')
    out.mkdir(parents=True);start=perf_counter();inputs={}
    def read(name):
        p=folder/(name+'.csv');inputs[str(p)]=hash_file(p);return pd.read_csv(p)
    run=read('runs');tick=read('timeseries');resource=read('resource_timeseries');spatial=read('spatial');life=read('individuals');horizon=int(run.horizon.iloc[0])
    scope=f'{phase}: ecological seeds0/1/2, n=3 per ecology/mode; {horizon} ticks; Q capacity256; mean ± sample SD; A/B/C fragmentation remains.'
    def finish(fig,name,title,extra=''):
        fig.suptitle(title,fontsize=12);fig.text(.5,.01,'\n'.join(textwrap.wrap(scope+' '+extra,125)),ha='center',va='bottom',fontsize=8)
        fig.tight_layout(rect=(0,.11,1,.93));fig.savefig(out/(name+'.png'),dpi=180);plt.close(fig)
    def trajectories(frame,metrics,name,title):
        fig,axes=plt.subplots(len(metrics),3,figsize=(15,4*len(metrics)+1),squeeze=False)
        for j,mode in enumerate('ABC'):
            for i,metric in enumerate(metrics):
                ax=axes[i,j]
                for e,g in frame[frame['mode'].eq(mode)].groupby('ecology'):
                    s=g.groupby('tick')[metric].agg(['mean','std','count'])
                    x=s.index.to_numpy();y=s['mean'].to_numpy();sd=s['std'].to_numpy()
                    ax.plot(x,y,color=COLORS[e],label=e);ax.fill_between(x,y-sd,y+sd,color=COLORS[e],alpha=.12)
                ax.set(title=mode+': '+metric,xlabel='Tick',ylabel=metric);ax.legend(fontsize=8)
        finish(fig,name,title,'Resource series use actual recorded ticks; no invented post-extinction supply.')
    trajectories(resource,['food_stock','water_stock'],'resource_availability','Actual food and water availability')
    trajectories(resource,['food_effective_cells','water_effective_cells'],'resource_concentration','Effective occupied resource cells (2**entropy)')
    trajectories(tick,['population'],'population','Living population trajectories')
    resource['patch_density_ratio']=resource.patch_agent_share/(resource.patch_cells/1200)
    trajectories(resource,['patch_density_ratio'],'patch_agent_density','Living agent concentration in patch regions')
    for name in ('food','water'):
        fig,axes=plt.subplots(2,4,figsize=(15,8))
        for j,e in enumerate(('E0','E1','E2','E3')):
            g=spatial[spatial.ecology.eq(e)&spatial['mode'].eq('A')&spatial.seed.eq(0)];last=int(g.tick.max())
            for i,t in enumerate((0,last)):
                s=g[g.tick.eq(t)];array=s.pivot(index='y',columns='x',values=name).to_numpy()
                im=axes[i,j].imshow(array,origin='upper',vmin=0,vmax=5,cmap='viridis')
                if e in ('E2','E3'):
                    mask=s.pivot(index='y',columns='x',values='patch').to_numpy();axes[i,j].contour(mask,levels=[.5],colors='white',linewidths=.6)
                axes[i,j].set(title=f'{e}/A seed0 tick{t}',xlabel='x',ylabel='y');fig.colorbar(im,ax=axes[i,j],fraction=.046)
        finish(fig,'spatial_'+name,'Spatial '+name+' units: shared fixed geometry, independent placements','Example seed0/A; white contours only for actual E2/E3 patches. No averaging over different geographic seeds.')
    def bars(metrics,name,title):
        fig,axes=plt.subplots(1,len(metrics),figsize=(max(8,5*len(metrics)),5),squeeze=False)
        for ax,metric in zip(axes[0],metrics):
            for j,mode in enumerate('ABC'):
                for i,e in enumerate(('E0','E1','E2','E3')):
                    s=run[run.ecology.eq(e)&run['mode'].eq(mode)][metric].dropna();x=i+(j-1)*.22
                    if len(s):
                        ax.errorbar(x,s.mean(),yerr=s.std(ddof=1),fmt='o',capsize=3,color=MODES[mode]);ax.scatter(x+np.linspace(-.035,.035,len(s)),s,s=9,color=MODES[mode],alpha=.6)
            ax.set_xticks(range(4),('E0','E1','E2','E3'));ax.set(title=metric,ylabel=metric)
        handles=[plt.Line2D([],[],marker='o',color=MODES[m],label=m) for m in 'ABC'];axes[0,-1].legend(handles=handles,fontsize=8)
        finish(fig,name,title)
    bars(['births','deaths'],'births_deaths','Cumulative births and deaths; dots are ecological seeds')
    bars(['mean_individual_acquisition_rate','acquisitions_per_attempt'],'acquisition_success','Actual acquisitions: distinct actor and attempt denominators')
    bars(['starvation_deaths','dehydration_deaths','resource_deaths_per_opportunity'],'resource_deaths','Observed resource-related death causes and exposure rates')
    bars(['visible_fraction','remembered_visible_fraction','toward_fraction'],'social_exposure','Social observations and actual movement relative to a frozen visible target')
    bars(['first_state_fraction','states_once_fraction','q_mean_occupancy','q_evictions_per_opportunity'],'q_fragmentation','Physical Q visitation and retention diagnostics')
    bars(['emission_rate','identified_fraction','masked_fraction','mean_vocal_q_rows_final'],'vocal_exposure','Unchanged Vocal Q: passive emission, hearing/masking and retained rows')
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,mode in zip(axes,'ABC'):
        for e in ('E0','E1','E2','E3'):
            curves=[]
            for seed in (0,1,2):
                g=life[life.ecology.eq(e)&life['mode'].eq(mode)&life.seed.eq(seed)&life.founder.eq(1)];duration=g.duration.to_numpy();censored=g.censored.to_numpy().astype(bool)
                curves.append([np.mean((duration>t)|(censored&(duration>=t))) for t in range(horizon+1)])
            values=np.asarray(curves);mean=values.mean(axis=0);sd=values.std(axis=0,ddof=1);x=np.arange(horizon+1)
            ax.plot(x,mean,label=e,color=COLORS[e]);ax.fill_between(x,mean-sd,mean+sd,alpha=.1,color=COLORS[e])
        ax.set(title=mode,xlabel='Founder follow-up ticks',ylabel='Founder survival fraction',ylim=(-.03,1.03));ax.legend(fontsize=8)
    finish(fig,'individual_survival','Original founder survival: administrative survivors right-censored','100 founders/run; seed-level curves, not pooled-agent confidence intervals. Newborn follow-up remains separate.')
    fig,ax=plt.subplots(figsize=(12,6));data=[];labels=[]
    for e in ('E0','E1','E2','E3'):
        for mode in 'ABC':
            g=run[run.ecology.eq(e)&run['mode'].eq(mode)]
            data.append([(g['diagnostic_physical_action_'+str(i)]/g.diagnostic_decisions).mean() for i in range(8)]);labels.append(e+'/'+mode)
    im=ax.imshow(np.asarray(data),aspect='auto',cmap='magma',vmin=0);ax.set_yticks(range(12),labels);ax.set_xticks(range(8),['N','S','E','W','eat','drink','wait','reproduce']);fig.colorbar(im,ax=ax,label='Choices / actual opportunities')
    finish(fig,'physical_actions','Physical action distributions; equal-seed means')
    contrasts=read('contrasts_by_seed');fig,axes=plt.subplots(1,4,figsize=(17,5))
    metrics=['mean_individual_acquisition_rate','founder_rmst','resource_deaths_per_opportunity','population']
    for ax,metric in zip(axes,metrics):
        g=contrasts[contrasts.family.eq('interaction')&contrasts.metric.eq(metric)]
        for i,mode in enumerate('ABC'):
            values=g[g.stratum.eq(mode)].difference.dropna();ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),fmt='o',capsize=3,color=MODES[mode]);ax.scatter(i+np.linspace(-.05,.05,len(values)),values,s=10,color=MODES[mode])
        ax.axhline(0,color='grey',lw=.8);ax.set_xticks(range(3),list('ABC'));ax.set_title(metric,fontsize=9);ax.set_ylabel('(E3−E2) − (E1−E0)')
    finish(fig,'factorial_contrasts','Scarcity × localization interaction: paired ecological seeds','Absolute effect sizes; all ecological/social paired contrasts and moderation are in CSV, no significance claim.')
    q=read('probes');fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,minimum in zip(axes,(2,5)):
        for j,mode in enumerate('BC'):
            for i,e in enumerate(('E0','E1','E2','E3')):
                g=q[q.ecology.eq(e)&q['mode'].eq(mode)&q.min_visits.eq(minimum)];last=g.groupby('seed').tick.transform('max');g=g[g.tick.eq(last)];v=g.policy_tv.dropna()
                x=i+(j-.5)*.15
                if len(v):ax.errorbar(x,v.mean(),yerr=v.std(ddof=1) if len(v)>1 else None,fmt='o',capsize=3,color=MODES[mode])
                else:ax.text(x,.5,'NA',transform=ax.get_xaxis_transform(),ha='center',fontsize=7)
        ax.set_xticks(range(4),('E0','E1','E2','E3'));ax.set(title=f'≥{minimum} visits, ≥1 actual TD update',ylabel='Trained retained-pair policy TV')
    finish(fig,'physical_policy_dependence','Read-only fixed-original-state Q dependence','B/C observed retained rows only; support counts and supported ecological n in CSV; no absent zero priors.')
    save_json(out/'metadata.json',{'version':'0.8.0','phase':phase,'inputs':inputs,'wall_seconds':perf_counter()-start,'peak_memory_bytes':peak(),
        'source_sha256':{__file__:hash_file(__file__)},'outputs':{p.name:{'sha256':hash_file(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir())}})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('pilot','primary'),default='primary');p.add_argument('--output',type=Path);p.add_argument('--data-dir',type=Path)
    try:plots(**vars(p.parse_args()))
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
