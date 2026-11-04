"""v074 static figures; ecological seeds, not decisions, are replication units."""
import argparse,textwrap
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.nonstationary_controls import ROOT,ANALYSIS,save_json,peak_bytes
from simulation.vocal_learning_observer import hash_file

COLORS=dict(zip('ABCDE',plt.get_cmap('tab10').colors[:5]))


def aggregate(frame,keys,metric):
    seed=frame.groupby([*keys,'seed'],sort=True,dropna=False)[metric].mean().reset_index()
    return seed.groupby(keys,sort=True,dropna=False)[metric].agg(['mean','std','count'])


def plots(root=ROOT,output=ANALYSIS/'figures'):
    root=Path(root);output=Path(output)
    if output.exists():raise ValueError('Refusing existing figures')
    output.mkdir(parents=True);start=perf_counter();inputs={}
    def read(path):
        path=Path(path);inputs[str(path)]=hash_file(path);return pd.read_csv(path)
    def finish(fig,name,title,scope):
        fig.suptitle(title,fontsize=12)
        fig.text(.5,.01,'\n'.join(textwrap.wrap(scope+'; Physical/Vocal Q capacity256; fixed nuisance schedule. Unmatched contrasts are exploratory; no policy-value inference.',125)),ha='center',va='bottom',fontsize=8)
        fig.tight_layout(rect=(0,.11,1,.92));fig.savefig(output/(name+'.png'),dpi=180);plt.close(fig)
    def draw(ax,x,mean,sd,**kwargs):
        kwargs.setdefault('capsize',2)
        ax.errorbar(x,mean,yerr=np.where(np.isfinite(sd),sd,0),**kwargs)
    social=read(ANALYSIS/'nonstationarity/windows.csv');cal=read(ANALYSIS/'calibration/windows.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,metric in zip(axes,('token_entropy','conditional_entropy')):
        for c,g in social[social.window.ne('full')].groupby('condition'):
            s=aggregate(g,['window'],metric);order=[str(i) for i in range(0,500,100)];s=s.reindex(order)
            draw(ax,range(5),s['mean'],s['std'],label=c,color=COLORS[c],marker='o')
        ax.set_xticks(range(5),['0–99','100–199','200–299','300–399','400–499'],rotation=15);ax.set(title=metric,ylabel='Bits',xlabel='Calendar tick window');ax.legend()
    finish(fig,'social_token_entropy_windows','Archived social-token nonstationarity: equal-seed mean ± sample SD','Calibration eco seeds3/4, n=2, 500 ticks; no nuisance realizations')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,reference,control in zip(axes,'BC','DE'):
        for c in (reference,control):
            g=cal[cal.condition.eq(c)&cal.window.ne('full')];s=aggregate(g,['window'],'token_entropy').reindex([str(i) for i in range(0,500,100)])
            draw(ax,range(5),s['mean'],s['std'],label=c,color=COLORS[c],marker='o',linestyle='-' if c==reference else '--')
        ax.set_xticks(range(5),['0–99','100–199','200–299','300–399','400–499'],rotation=15);ax.set(title=reference+' vs '+control,ylabel='Marginal entropy (bits)');ax.legend()
    finish(fig,'nuisance_entropy_windows','Frozen nonstationary nuisance on fixed calibration experience','Calibration eco n=2, nuisance realizations/eco=2, 500 ticks')
    matching=read(ANALYSIS/'calibration/matching.csv').assign(phase='calibration')
    pilot=read(ANALYSIS/'pilot/matching.csv').assign(phase='pilot');matching=pd.concat([matching,pilot[pilot.complete_window]],ignore_index=True)
    metrics=['token_entropy','conditional_entropy','marginal_tv','conditional_tv_common_X','first_state_fraction','relative_mean_visits']
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    order=[('calibration',str(i)) for i in range(0,500,100)]+[('calibration','full'),('pilot','0'),('pilot','full')]
    for ax,metric in zip(axes.flat,metrics):
        g=matching[matching.metric.eq(metric)]
        for j,c in enumerate('DE'):
            a=g[g.condition.eq(c)];s=aggregate(a,['phase','window'],'difference').reindex(pd.MultiIndex.from_tuples(order,names=['phase','window']))
            draw(ax,np.arange(8)+(j-.5)*.08,s['mean'],s['std'],label=c,color=COLORS[c],marker='.',linestyle='-')
        t=g.target.iloc[0];ax.axhspan(-t if metric not in ('marginal_tv','conditional_tv_common_X') else 0,t,color='grey',alpha=.12)
        ax.axhline(0,color='grey',lw=.7);ax.set_xticks(range(8),['cal0','cal100','cal200','cal300','cal400','calFull','pilot0','pilotFull'],rotation=45,fontsize=7);ax.set_title(metric);ax.legend(fontsize=7)
    finish(fig,'matching_quality','Marginal, conditional and fragmentation matching (target bands shaded)','Calibration eco n=2/500 ticks; pilot eco n=3/100 ticks; nuisance realizations/eco=2; NA if common-X support insufficient')
    segments=read(ANALYSIS/'calibration/run_segments.csv');fig,axes=plt.subplots(1,3,figsize=(14,5))
    for c,g in cal[cal.window.ne('full')].groupby('condition'):
        s=aggregate(g,['window'],'same_token').reindex([str(i) for i in range(0,500,100)])
        draw(axes[0],range(5),s['mean'],s['std'],label=c,color=COLORS[c],marker='.')
    axes[0].set_xticks(range(5),['0','100','200','300','400']);axes[0].set(xlabel='Window start',ylabel='Same-token transition probability')
    for ax,window in zip(axes[1:],('0','full')):
        for c,g in segments[segments.window.eq(window)].groupby('condition'):
            pivot=g.groupby(['seed','realization','length']).segments.sum().unstack('length',fill_value=0)
            cdf=pivot.cumsum(axis=1).div(pivot.sum(axis=1),axis=0);mean=cdf.groupby(level='seed').mean().mean()
            ax.plot(mean.index,mean,label=c,color=COLORS[c])
        ax.set(xlabel='Observed clipped segment length',ylabel='Empirical CDF',title='Window '+window);ax.set_xscale('log')
    for ax in axes:ax.legend(fontsize=7)
    finish(fig,'token_persistence_run_lengths','Temporal persistence and observed run segments','Calibration eco n=2, nuisance/eco=2, 500 ticks; left/death/window-end censoring retained; not uncensored run lifetimes')
    r=read(ANALYSIS/'replay/runs.csv');representations=['A','B','C','D','E','B_Umemory']
    def bars(ax,frame,metric,source_condition):
        g=frame[frame.source_condition.eq(source_condition)];seed=g.groupby(['representation','seed'])[metric].mean().reset_index()
        for i,rep in enumerate(representations):
            a=seed[seed.representation.eq(rep)][metric].dropna()
            if not len(a):ax.text(i,.5,'NA',transform=ax.get_xaxis_transform(),ha='center');continue
            draw(ax,[i],[a.mean()],[a.std(ddof=1)],fmt='o',color=COLORS.get(rep,'grey'))
            ax.scatter(i+np.linspace(-.08,.08,len(a)),a,s=10,color=COLORS.get(rep,'grey'),alpha=.5)
        ax.set_xticks(range(6),['X','X+B','X+C','X+U11','X+U37','X+B+Urd'],rotation=25,fontsize=8);ax.set_title(source_condition+' experience: '+metric)
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for j,c in enumerate('BC'):
        bars(axes[0,j],r,'first_state_fraction',c);bars(axes[1,j],r,'mean_visits',c)
    finish(fig,'q_state_fragmentation','Fixed-experience representation fragmentation','Held-out source B/C eco n=3, nuisance/eco=2, 500 ticks; B+Urd has66 possible combinations (unmatched)')
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for j,c in enumerate('BC'):
        bars(axes[0,j],r,'mean_retained_q_range',c);bars(axes[1,j],r,'retention_fraction',c)
    finish(fig,'fixed_experience_q_comparison','Same actual actions/rewards: retained Q dispersion and retention','Held-out source B/C eco n=3, nuisance/eco=2, 500 ticks; diagnostic representations, not alternative-policy outcomes')
    q=read(ANALYSIS/'replay/probes.csv');q=q[q.min_visits.eq(5)]
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for j,c in enumerate('BC'):
        for ax,kind in zip(axes[:,j],('all_social','memory_only')):
            g=q[q.source_condition.eq(c)&q.kind.eq(kind)];s=g.groupby(['representation','seed']).policy_tv.mean().reset_index()
            for i,rep in enumerate(representations):
                a=s[s.representation.eq(rep)].policy_tv.dropna()
                if len(a):draw(ax,[i],[a.mean()],[a.std(ddof=1)],fmt='o',color=COLORS.get(rep,'grey'))
                else:ax.text(i,.5,'NA',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
            labels=[rep+'\nn='+str(s[s.representation.eq(rep)].policy_tv.count()) for rep in representations]
            ax.set_xticks(range(6),labels,rotation=20,fontsize=7);ax.set(title=c+' '+kind,ylabel='Policy TV (qualified retained row pairs)')
    finish(fig,'learned_policy_dependence','Fixed-X dependence; memory-only additionally holds actual B fixed','Source eco n≤3 (supported n shown), nuisance/eco=2, 500 ticks; row visits≥5 and TD updates≥1, unsupported=NA')
    pred=read(ANALYSIS/'prediction/runs.csv');fig,axes=plt.subplots(3,2,figsize=(14,12))
    targets=['immediate_reward','future_reward_5','survival_5']
    for row,target in enumerate(targets):
        for j,c in enumerate('BC'):
            g=pred[pred.source_condition.eq(c)&pred.target.eq(target)];names=list(dict.fromkeys(g.model))
            for offset,subset,style in [(-.06,'all','o'),(.06,'common','s')]:
                a=g[g.subset.eq(subset)];s=aggregate(a,['model'],'log_loss').reindex(names)
                draw(axes[row,j],np.arange(len(names))+offset,s['mean'],s['std'],label=subset,fmt=style)
            axes[row,j].set_xticks(range(len(names)),names,rotation=30,fontsize=7);axes[row,j].set(title=c+' '+target,ylabel='Held-out log loss (lower better)');axes[row,j].legend(fontsize=8)
    finish(fig,'heldout_predictive_performance','Prespecified hierarchical prediction on identical held-out decisions','Train eco3/4; test eco n=3, attached nuisance/eco=2, 500 ticks; all and identical common-support subsets separate')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,c in zip(axes,'BC'):
        g=pred[pred.source_condition.eq(c)&pred.target.eq('immediate_reward')&pred.subset.eq('all')]
        for metric in ('coverage','common_coverage'):
            names=list(dict.fromkeys(g.model));s=aggregate(g,['model'],metric).reindex(names)
            draw(ax,np.arange(len(names)),s['mean'],s['std'],marker='o',label=metric)
        ax.set_xticks(range(len(names)),names,rotation=30,fontsize=8);ax.set_title(c);ax.set_ylabel('Fraction of eligible test decisions');ax.legend(fontsize=8)
    finish(fig,'prediction_support','Training-cell support vs common-support exposure','Test eco n=3, nuisance/eco=2, 500 ticks; min20 training cell observations, counts in CSV; missing support is not zero effect')
    pairs=read(ANALYSIS/'prediction/paired_by_seed.csv');fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,subset in zip(axes,('all','common')):
        g=pairs[pairs.source_condition.eq('C')&pairs.subset.eq(subset)&pairs.metric.eq('log_loss')]
        for contrast in ('X_C_A-X_B_A','X_C_A-X_B_URD_A'):
            s=aggregate(g[g.contrast.eq(contrast)],['target'],'difference').reindex(targets)
            draw(ax,range(3),s['mean'],s['std'],marker='o',label=contrast)
        ax.axhline(0,color='grey',lw=.8);ax.set_xticks(range(3),targets,rotation=20,fontsize=8);ax.set(title=subset,ylabel='Paired log-loss difference (negative=better)');ax.legend(fontsize=8)
    finish(fig,'memory_specific_prediction','Additional remembered-direction prediction with actual B held fixed','Source C test eco n=3, nuisance/eco=2, 500 ticks; observational prediction, no causal/policy/survival claim')
    online=read(ANALYSIS/'pilot/online_runs.csv');fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,c in zip(axes,'DE'):
        for seed,g in online[online.condition.eq(c)].groupby('seed'):ax.plot(g.realization,g.population,marker='o',label=f'eco seed{seed}')
        ax.set(title=c,xlabel='Independent nuisance realization',ylabel='Final population');ax.set_xticks([0,1]);ax.legend(fontsize=8)
    finish(fig,'nuisance_realization_sensitivity','Bounded pilot sensitivity: within-ecological-seed differences','Pilot eco n=3, nuisance/eco=2, 100 ticks; not independent n=6')
    save_json(output/'metadata.json',{'version':'0.7.4','inputs':inputs,'wall_seconds':perf_counter()-start,'peak_memory_bytes':peak_bytes(),
        'source_sha256':{__file__:hash_file(__file__)},'outputs':{p.name:{'sha256':hash_file(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir())}})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=ANALYSIS/'figures')
    try:plots(**vars(p.parse_args()))
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
