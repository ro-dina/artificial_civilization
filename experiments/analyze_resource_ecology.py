"""Resource ecology summaries and paired factorial effects; ecological seed n=3."""
from pathlib import Path
from time import perf_counter
import argparse,json,math
import numpy as np
import pandas as pd

from experiments.resource_ecology_experiments import ROOT,ANALYSIS,load,case_path,verify_frozen,save_json,peak
from simulation.vocal_learning_observer import hash_file


def gini(values):
    x=sorted(float(v) for v in values if math.isfinite(v) and v>=0);n=len(x)
    if not n:return None
    total=sum(x)
    return max(0.,min(1.,sum((2*i-n-1)*v for i,v in enumerate(x,1))/(n*total))) if total else 0.


def founder_survival(people,horizon):
    founders=people[people.founder.eq(1)]
    if founders.empty:return {'founder_rmst':None,'founder_n':0,'founder_censored':0}
    # Every original founder is observed from0; survivors censor at H, all other
    # durations are observed deaths. This empirical RMST needs no censor imputation.
    if (founders.censored.eq(1)&founders.duration.lt(horizon)).any():raise ValueError('Incomplete founder administrative follow-up')
    return {'founder_rmst':float(founders.duration.clip(upper=horizon).mean()),'founder_n':len(founders),'founder_censored':int(founders.censored.sum())}


def summary(frame,keys,metrics):
    t=frame.melt(id_vars=[*keys,'seed'],value_vars=metrics,var_name='metric',value_name='measurement')
    if t.duplicated([*keys,'seed','metric']).any():raise ValueError('Repeated independent seed contribution')
    return t.groupby([*keys,'metric'],sort=True,dropna=False).measurement.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'})


def factorial(frame,metrics):
    result=[]
    for seed,g in frame.groupby('seed',sort=True):
        g=g.set_index(['ecology','mode'])
        for metric in metrics:
            interactions={}
            for mode in 'ABC':
                for a,b in [('E1','E0'),('E2','E0'),('E3','E1'),('E3','E2')]:
                    result.append({'seed':seed,'family':'ecology','stratum':mode,'contrast':a+'-'+b,'metric':metric,'difference':g.loc[(a,mode),metric]-g.loc[(b,mode),metric]})
                interactions[mode]=(g.loc[('E3',mode),metric]-g.loc[('E2',mode),metric])-(g.loc[('E1',mode),metric]-g.loc[('E0',mode),metric])
                result.append({'seed':seed,'family':'interaction','stratum':mode,'contrast':'scarcity_x_localization','metric':metric,'difference':interactions[mode]})
            for e in ('E0','E1','E2','E3'):
                for a,b in [('B','A'),('C','B'),('C','A')]:
                    result.append({'seed':seed,'family':'social','stratum':e,'contrast':a+'-'+b,'metric':metric,'difference':g.loc[(e,a),metric]-g.loc[(e,b),metric]})
            for a,b in [('B','A'),('C','B'),('C','A')]:
                result.append({'seed':seed,'family':'moderation','stratum':a+'-'+b,'contrast':'interaction_difference','metric':metric,'difference':interactions[a]-interactions[b]})
    return pd.DataFrame(result)


def analyze(root=ROOT,phase='primary',output=None):
    root=Path(root);output=Path(output) if output else ANALYSIS/phase;verify_frozen(root);start=perf_counter()
    if (output/'runs.csv').exists():raise ValueError('Refusing completed analysis overwrite')
    horizon=100 if phase=='pilot' else 500;rows=[];times=[];resources=[];grids=[];people_out=[];cues=[];probes=[];visits=[];vocal=[];inputs={}
    for e in ('E0','E1','E2','E3'):
        for mode in 'ABC':
            for seed in (0,1,2):
                p=Path(str(case_path(root,phase,e,mode,seed))+'.metadata.json');m,files=load(p);inputs[str(p)]=hash_file(p)
                tag={'ecology':e,'mode':mode,'seed':seed};s=m['statistics'];t=m['totals'];life=pd.read_csv(files['individuals'],float_precision='round_trip')
                obs=life[life.opportunities.gt(0)].copy();obs['acq_rate']=obs.acquisitions/obs.opportunities
                den=t['decisions'];attempts=int(life.eat_attempts.sum()+life.drink_attempts.sum());success=int(life.acquisitions.sum())
                r={**tag,**s,**{'diagnostic_'+k:v for k,v in t.items()},**founder_survival(life,horizon),
                    'horizon':horizon,'actual_final_tick':m['tick'],'extinct':int(s['population']==0),'actor_n':len(obs),'individual_n':len(life),
                    'mean_individual_acquisition_rate':float(obs.acq_rate.mean()) if len(obs) else None,'acquisition_rate_gini':gini(obs.acq_rate),
                    'founder_mean_acquisition_rate':float(obs.loc[obs.founder.eq(1),'acq_rate'].mean()) if obs.founder.eq(1).any() else None,
                    'acquisitions':success,'acquisition_attempts':attempts,'acquisitions_per_attempt':success/attempts if attempts else None,'acquisitions_per_opportunity':success/den if den else None,
                    'failed_eating':int(life.eat_attempts.sum()-life.food.sum()),'failed_drinking':int(life.drink_attempts.sum()-life.water.sum()),
                    'resource_deaths':s['starvation_deaths']+s['dehydration_deaths'],'resource_deaths_per_opportunity':(s['starvation_deaths']+s['dehydration_deaths'])/den if den else None,
                    'mean_reward':float(life.reward.sum())/den if den else None,'newborn_censored':int(life[life.founder.eq(0)].censored.sum()),
                    'newborn_mean_observed_followup':float(life[life.founder.eq(0)].duration.mean()) if life.founder.eq(0).any() else None,
                    'first_state_fraction':t['first_agent_states']/den,'q_evictions_per_opportunity':t.get('evictions',0)/den,'q_mean_occupancy':t['occupancy_sum']/den,
                    'visible_fraction':t.get('visible_decisions',0)/den,'remembered_visible_fraction':t.get('remembered_decisions',0)/den,
                    'toward_fraction':t.get('toward',0)/t['target_moves'] if t.get('target_moves',0) else None,
                    'emission_rate':s['signals_emitted']/den,'identified_fraction':t.get('identified',0)/den,'masked_fraction':t.get('masked',0)/den,
                    'eligible_reproduction_opportunities':int(life.eligible_opportunities.sum()),'successful_reproduction_choices':int(life.successful_reproduction_choices.sum())}
                eligible=r['eligible_reproduction_opportunities'];r['reproduction_per_eligible_opportunity']=r['successful_reproduction_choices']/eligible if eligible else None
                d=pd.read_csv(files['distributions']);v=d[d.metric.eq('physical_state_visits')]
                r['states_once_fraction']=float(v.loc[v.value.eq(1),'count'].sum()/v['count'].sum()) if v['count'].sum() else None
                if int((v.value*v['count']).sum())!=den or success!=s['food_consumed']+s['water_consumed']:raise ValueError('Histogram/acquisition accounting failed')
                visits.append(v.assign(**tag));q=pd.read_csv(files['probes']);q['policy_tv']=q.tv_sum/q.pairs.replace(0,np.nan);probes.append(q.assign(**tag))
                rt=pd.read_csv(files['resource_ticks']);resources.append(rt.assign(**tag))
                r['mean_vocal_q_rows_final']=float(rt.mean_vocal_q_rows.iloc[-1]) if 'mean_vocal_q_rows' in rt and pd.notna(rt.mean_vocal_q_rows.iloc[-1]) else None
                for resource in ('food','water'):
                    r[resource+'_mean_stock']=rt[resource+'_stock'].mean();r[resource+'_attempted']=rt[resource+'_attempted'].sum();r[resource+'_realized']=rt[resource+'_realized'].sum()
                    r[resource+'_capacity_censored_expected']=rt[resource+'_capacity_censored_expected'].sum();r[resource+'_mean_effective_cells']=rt[resource+'_effective_cells'].mean()
                rows.append(r);times.append(pd.read_csv(files['ticks']).assign(**tag));grids.append(pd.read_csv(files['spatial']).assign(**tag));people_out.append(life.assign(**tag));vocal.append(pd.read_csv(files['vocal']).assign(**tag))
                columns=['visible_individuals','remembered_visible_count','food_here','water_here','food_near3_tickstart','water_near3_tickstart','acquired','action','reward','food_at_visible_neighbor','water_at_visible_neighbor']
                a=pd.read_csv(files['acquisitions'],usecols=columns)
                attempts_mask=a.action.isin([4,5]);acquired_mask=a.acquired.gt(0)
                for name,mask in [('consuming_turn_reward',acquired_mask),('failed_consumption_turn_reward',attempts_mask&~acquired_mask),('food_consuming_turn_reward',a.action.eq(4)&acquired_mask),('water_consuming_turn_reward',a.action.eq(5)&acquired_mask)]:
                    r[name]=float(a.loc[mask,'reward'].mean()) if mask.any() else None
                a['visible_bin']=np.where(a.visible_individuals.eq(0),0,np.where(a.visible_individuals.eq(1),1,2));a['remembered']=a.remembered_visible_count.gt(0).astype(int)
                for (visible,remembered),g in a.groupby(['visible_bin','remembered'],sort=True):
                    n=len(g);att=g.action.isin([4,5]);successes=int(g.acquired.sum())
                    cues.append({**tag,'visible_bin':visible,'remembered':remembered,'opportunities':n,'attempts':int(att.sum()),'acquisitions':successes,
                        'food_here_fraction':float(g.food_here.gt(0).mean()),'water_here_fraction':float(g.water_here.gt(0).mean()),
                        'food_near3_fraction':float(g.food_near3_tickstart.mean()),'water_near3_fraction':float(g.water_near3_tickstart.mean()),
                        'neighbor_food_fraction':float(g.food_at_visible_neighbor.dropna().gt(0).mean()) if g.food_at_visible_neighbor.notna().any() else None,
                        'neighbor_water_fraction':float(g.water_at_visible_neighbor.dropna().gt(0).mean()) if g.water_at_visible_neighbor.notna().any() else None})
    run=pd.DataFrame(rows);metrics=['mean_individual_acquisition_rate','founder_rmst','resource_deaths_per_opportunity','population','births','deaths','acquisitions_per_attempt','acquisitions_per_opportunity','mean_reward','acquisition_rate_gini','first_state_fraction','q_evictions_per_opportunity','visible_fraction','toward_fraction','emission_rate','food_consumed','water_consumed']
    contrasts=factorial(run,metrics);output.mkdir(parents=True,exist_ok=True)
    tables={'runs':run,'summary':summary(run,['ecology','mode'],[c for c in run.select_dtypes('number') if c!='seed']),
        'timeseries':pd.concat(times,ignore_index=True),'resource_timeseries':pd.concat(resources,ignore_index=True),'spatial':pd.concat(grids,ignore_index=True),
        'individuals':pd.concat(people_out,ignore_index=True),'social_resource_cues':pd.DataFrame(cues),'probes':pd.concat(probes,ignore_index=True),'state_visits':pd.concat(visits,ignore_index=True),'vocal_actions':pd.concat(vocal,ignore_index=True),
        'contrasts_by_seed':contrasts,'contrasts_summary':contrasts.groupby(['family','stratum','contrast','metric']).difference.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'})}
    for name,f in tables.items():f.to_csv(output/(name+'.csv'),index=False,float_format='%.12g')
    save_json(output/'analysis_metadata.json',{'version':'0.8.0','phase':phase,'horizon':horizon,'ecological_seeds':[0,1,2],'inputs':inputs,
        'wall_seconds':perf_counter()-start,'peak_memory_bytes':peak(),'source_sha256':{__file__:hash_file(__file__)},
        'definitions':{'replication':'Per-run rates first, equal ecological-seed mean/sample SD. No decision-level experimental confidence intervals.',
        'survival':'Founder RMST to requested horizon, survivors right-censored. Newborn followup and censoring separate; not naive lifespan.',
        'acquisition':'Actual food/water units; individual rates over each actual exposure, unweighted actors mean primary, aggregate numerator/denominator also retained.',
        'interpretation':'Factorial manipulations include capacity-mediated supply/consumption feedback. A/B/C state fragmentation remains. Proximity, prediction and population differences do not establish cooperation or causal social-information benefit.'}})
    print(run[['ecology','mode','seed','population','mean_individual_acquisition_rate','founder_rmst','resource_deaths','first_state_fraction']].to_string(index=False),flush=True)
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('pilot','primary'),default='primary');p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path)
    try:analyze(**vars(p.parse_args()))
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
