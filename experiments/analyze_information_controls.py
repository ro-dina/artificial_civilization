"""Read-only, equal ecological-seed analysis; never pools decisions as replicates."""
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.controls_metrics import load,read_csv,original_frame,sequence_metrics,tokens_for
from experiments.calibrate_information_controls import TARGETS
from experiments.analyze_social_learning import RATES
from simulation.vocal_learning_observer import hash_file


def grouped(frame,keys,metrics):
    """Average nuisance realizations WITHIN ecological seed before between-seed SD."""
    seed=frame.groupby([*keys,'seed'],sort=True,dropna=False)[metrics].mean().reset_index()
    tidy=seed.melt(id_vars=[*keys,'seed'],value_vars=metrics,var_name='metric',value_name='__sample_value')
    return tidy.groupby([*keys,'metric'],sort=True,dropna=False).__sample_value.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'})


def matching(runs):
    rows=[]
    for control,target in (('D','B'),('E','C')):
        for _,r in runs[runs.condition.eq(control)].iterrows():
            g=runs[runs.condition.eq(target)&runs.seed.eq(r.seed)]
            if len(g)!=1:raise ValueError('Exactly one reference per ecological seed required')
            b=g.iloc[0]
            for metric,tolerance in TARGETS.items():
                difference=r.mean_visits/b.mean_visits-1 if metric=='relative_mean_visits' else r[metric]-b[metric]
                gross=.05 if metric=='first_state_fraction' else .75 if metric=='token_entropy' else .15 if metric=='same_token' else np.nan
                rows.append({'condition':control,'reference':target,'seed':r.seed,'realization':r.realization,'metric':metric,'difference':difference,
                    'tolerance':tolerance,'within_target':abs(difference)<=tolerance,'gross_failure':False if pd.isna(gross) else abs(difference)>gross,
                    'decisions_control':r.decisions,'decisions_reference':b.decisions})
    return pd.DataFrame(rows)


def analyze(root=Path('data/social_learning_v073'),output=Path('analysis/social_learning_v073/pilot'),phase='pilot'):
    root=Path(root);output=Path(output);horizon=100 if phase=='pilot' else 500
    if output.exists():raise ValueError('Refusing existing analysis directory')
    if output.resolve().is_relative_to(root.resolve()):raise ValueError('Output overlaps data')
    sources=[]
    for c in 'ABC':
        for seed in (0,1,2):sources.append(Path('data/social_learning_v072')/phase/f'{c}_seed_{seed}.csv.gz.metadata.json')
    for sub in (['pilot','pilot_robustness'] if phase=='pilot' else ['primary','robustness']):
        sources.extend(sorted((root/sub).glob('*.metadata.json')))
    baselines=[];identities=[]
    for path in sources:
        m,_=load(path);cfg=m['config']['simulation']
        if cfg['learning_memory_capacity']!=256 or cfg['vocal_learning_memory_capacity']!=256 or m['tick']!=horizon:raise ValueError('Primary architecture/horizon mismatch')
        baselines.append({k:v for k,v in cfg.items() if k!='random_seed'})
        identities.append((m['condition'],m['seed'],m.get('realization',0)))
    if len(set(identities))!=len(identities) or any(x!=baselines[0] for x in baselines):raise ValueError('Duplicate replication or incompatible ecology')
    runs=[];times=[];probes=[];visits=[];runs_hist=[];token_freq=[];conditional=[];retained=[];updates_hist=[];distributions=[];vocals=[]
    for path in sources:
        m,files=load(path);c=m['condition'];seed=m['seed'];r=m.get('realization',0)
        tag={'condition':c,'seed':seed,'realization':r,'nuisance_seed':m.get('nuisance_seed')}
        reference=root/'references'/f'{c}_seed_{seed}_r0.csv.gz.metadata.json' if c in 'ABC' else path
        om,of,f=original_frame(reference);f=f[f.tick.le(horizon)]
        metrics,counts,joint,rhist,vhist=sequence_metrics(f,tokens_for(f,c))
        total=m['totals'];s=m['statistics'];row={**tag,**total,**metrics,**s,'decisions':len(f),'final_tick':m['tick'],'theoretical_multiplier':{'A':1,'B':11,'C':37,'D':11,'E':37}[c]}
        if total['decisions']!=len(f) or total['first_agent_states']!=metrics['distinct_agent_states']:raise ValueError('Ordered stream/archived fragmentation mismatch')
        if s['population']!=m['config']['simulation']['initial_population']+s['births']-s['deaths']:raise ValueError('Population accounting failed')
        for name,(a,b) in RATES.items():row[name]=total.get(a,0)/total[b] if total.get(b,0) else np.nan
        row['evictions_per_decision']=total.get('evictions',0)/len(f);row['q_updates']=total.get('updates',0)
        row['q_occupancy']=total['occupancy_sum']/len(f)
        d=read_csv(files['distributions']);distributions.append(d.assign(**tag))
        for label,subset in [('death_age',d[d.metric.eq('survival_age')&d.reason.ne('run_end_censored')]),('survivor_age',d[d.metric.eq('survival_age')&d.reason.eq('run_end_censored')])]:
            n=subset['count'].sum();row['mean_'+label]=(subset.value*subset['count']).sum()/n if n else np.nan;row[label+'_n']=n
        runs.append(row)
        times.append(read_csv(files['ticks']).assign(**tag))
        q=read_csv(files['probes']);q=q[q.kind.eq('all_social')];q['policy_tv']=q.tv_sum/q.pairs.replace(0,np.nan);q['greedy_change_fraction']=q.changed_greedy_sets/q.pairs.replace(0,np.nan)
        probes.append(q.assign(**tag))
        vocals.append(read_csv(files['vocal']).assign(**tag))
        for value,n in vhist.items():visits.append({**tag,'visits':value,'states':n,'state_fraction':n/metrics['distinct_agent_states']})
        for (length,reason),n in rhist.items():runs_hist.append({**tag,'length':length,'reason':reason,'runs':n,'fraction':n/sum(rhist.values())})
        for token,n in counts.items():token_freq.append({**tag,'token':token,'decisions':n,'frequency':n/len(f)})
        catalog=read_csv(of['states']).set_index('base_id').key_json.to_dict()
        for base,ct in sorted(joint.items()):
            den=sum(ct.values())
            for token,n in sorted(ct.items()):conditional.append({**tag,'original_key':catalog[base],'token':token,'n':n,'exposure':den,'frequency':n/den,'supported_min20':den>=20})
        retained.append(read_csv(of['retained']).query('tick <= @horizon').assign(**tag))
        # Whole-run lifetime update histogram is comparable only at its recorded horizon.
        if om['tick']==horizon:updates_hist.append(read_csv(of['lifetime_updates']).assign(**tag))
    run=pd.DataFrame(runs);ticks=pd.concat(times,ignore_index=True);match=matching(run)
    pairs=[];averaged=run.groupby(['condition','seed'],sort=True).mean(numeric_only=True).reset_index()
    for seed in (0,1,2):
        g=averaged[averaged.seed.eq(seed)].set_index('condition')
        for alt,ref in [('B','A'),('C','A'),('D','B'),('E','C'),('D','A'),('E','A')]:
            if alt not in g.index:continue
            for metric in ('population','births','deaths','first_state_fraction','evictions_per_decision','toward_fraction','food_consumed','water_consumed'):
                pairs.append({'seed':seed,'contrast':alt+'-'+ref,'metric':metric,'difference':g.loc[alt,metric]-g.loc[ref,metric]})
    numeric=[x for x in run.select_dtypes('number') if x not in ('seed','realization','nuisance_seed')]
    tables={'runs':run,'seed_summary':grouped(run,['condition'],numeric),'matching':match,'timeseries':ticks,
            'paired_by_seed':pd.DataFrame(pairs),'probes':pd.concat(probes,ignore_index=True),'state_visits':pd.DataFrame(visits),
            'token_runs':pd.DataFrame(runs_hist),'token_frequencies':pd.DataFrame(token_freq),'conditional_token_frequencies':pd.DataFrame(conditional),
            'retained_rows':pd.concat(retained,ignore_index=True),'distributions':pd.concat(distributions,ignore_index=True),'vocal_actions':pd.concat(vocals,ignore_index=True)}
    if updates_hist:tables['lifetime_updates']=pd.concat(updates_hist,ignore_index=True)
    tables['probe_seed_summary']=grouped(tables['probes'],['condition','tick','min_visits'],['policy_tv','greedy_change_fraction','pairs','qualified_rows','supported_agents'])
    differences=tables['paired_by_seed']
    tables['paired_summary']=differences.groupby(['contrast','metric']).difference.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'})
    matched_seed=match.groupby(['condition','metric','seed']).difference.mean().reset_index()
    tables['matching_seed_summary']=matched_seed.groupby(['condition','metric']).difference.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'})
    output.mkdir(parents=True)
    for name,f in tables.items():f.to_csv(output/(name+'.csv'),index=False,float_format='%.12g')
    limits='Exploratory: stationary nuisance does not match every conditional distribution/run length; n=3 ecological seeds, Q=256, '+str(horizon)+' ticks'
    from experiments.controls_plots import plots
    plots(output,limits)
    review={'phase':phase,'horizon':horizon,'all_targets_met':bool(match.within_target.all()),'gross_failures':int(match.gross_failure.sum()),
            'failed_targets':match[~match.within_target].to_dict(orient='records'),
            'decision':'primary_blocked' if match.gross_failure.any() else 'review_required','interpretation':'Category count alone is not matching; no automatic primary on poor matching.'}
    (output/'matching_review.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n')
    metadata={'version':'0.7.3','phase':phase,'horizon':horizon,'ecological_seeds':[0,1,2],'q_capacity':256,
        'inputs':{str(p):hash_file(p) for p in sources},'reference_traces':{str(p):hash_file(p) for p in sorted((root/'references').glob('*.metadata.json'))},
        'frozen_models':json.loads((root/'models/frozen.json').read_text())['models'],
        'definitions':{'n':'Nuisance realizations averaged within ecological seed, then equal-seed mean/sample SD (ddof=1). No significance tests.',
          'runs':'Per actual consecutive own decision opportunities; terminal and end segments are right-censored, not completed runs.',
          'prediction':'Separate B/C same-experience held-out-seed analysis; immediate reward conditioned on actual action, not policy improvement.',
          'limits':limits},'source_sha256':{str(p):hash_file(p) for p in sorted(Path('experiments').glob('*controls*.py'))},
        'outputs':{p.name:{'sha256':hash_file(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir())}}
    (output/'analysis_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(run[['condition','seed','realization','population','first_state_fraction','mean_visits','same_token','token_entropy','evictions_per_decision']].to_string(index=False),flush=True)
    print('Matching:',review['all_targets_met'],'gross failures:',review['gross_failures'],flush=True)
    return review


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('data/social_learning_v073'));p.add_argument('--phase',choices=('pilot','primary'),default='pilot');p.add_argument('--output',type=Path,default=Path('analysis/social_learning_v073/pilot'))
    try:analyze(**vars(p.parse_args()))
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
