"""v074 read-only clock/lifetime diagnostics on actual ordered decisions."""
from collections import Counter,defaultdict
import json,math
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.controls_metrics import original_frame,read_csv,sequence_metrics,entropy
from simulation.vocal_learning_observer import hash_file

OLD=Path('data/social_learning_v073')
RUN_BINS=(1,2,3,4,5,8,16,32,math.inf)


def source(condition,seed):
    return OLD/('calibration' if seed in (3,4) else 'references')/f'{condition}_seed_{seed}_r0.csv.gz.metadata.json'


def prepared(path):
    m,files,f=original_frame(path)
    f=f.copy();f['opportunity_index']=f.groupby('agent_id',sort=False).cumcount()
    f['first_tick']=f.groupby('agent_id',sort=False).tick.transform('min')
    # Under the current complete-from-birth records, age advances once per own
    # opportunity from zero; age and opportunity index are exactly collinear.
    f['derived_pre_age']=f.opportunity_index
    return m,files,f


def novelty(frame,tokens):
    seen={};flags=[]
    for (identity,x,terminal),token in zip(frame[['agent_id','base_id','terminal']].itertuples(index=False,name=None),tokens):
        history=seen.setdefault(identity,set());key=(int(x),int(token));flags.append(key not in history);history.add(key)
        if terminal:seen.pop(identity)
    return np.asarray(flags,dtype=bool)


def run_segments(frame,tokens,has_prior=None):
    """Window-clipped observed segments; left and right censoring explicit."""
    last={};hist=Counter()
    has_prior={} if has_prior is None else has_prior
    for (identity,terminal),token in zip(frame[['agent_id','terminal']].itertuples(index=False,name=None),tokens):
        old=last.get(identity)
        if old and old[0]!=token:
            hist[old[1],old[2],'observed_change']+=1
        length=old[1]+1 if old and old[0]==token else 1
        left=old[2] if old and old[0]==token else bool(has_prior.get(identity,False)) if old is None else False
        if terminal:hist[length,left,'death_censored']+=1;last.pop(identity,None)
        else:last[identity]=(token,length,left)
    for token,length,left in last.values():hist[length,left,'window_end_censored']+=1
    return hist


def run_cdf(hist):
    n=sum(hist.values())
    return [sum(count for (length,*_),count in hist.items() if length<=edge)/n if n else np.nan for edge in RUN_BINS]


def profiles(frame,tokens,final_tick=500):
    tokens=np.asarray(tokens);first=novelty(frame,tokens);rows=[];frequencies=[];segments=[];visits=[]
    windows=[(str(start),start,start+100) for start in range(0,500,100) if start<=final_tick]+[('full',0,final_tick+1)]
    for label,start,stop in windows:
        mask=(frame.tick.ge(start)&frame.tick.lt(stop)).to_numpy();f=frame.loc[mask];u=tokens[mask]
        if not len(f):continue
        result,counts,joint,_,vhist=sequence_metrics(f,u)
        result['window_first_state_fraction']=result['first_state_fraction']
        result['first_state_fraction']=float(first[mask].mean())
        prior=f.groupby('agent_id',sort=False).opportunity_index.first().gt(0).to_dict()
        hist=run_segments(f,u,prior);cdf=run_cdf(hist)
        rows.append({'window':label,'start_tick':start,'stop_tick_exclusive':stop,'complete_window':label=='full' or final_tick>=stop-1,
            **result,'agents':f.agent_id.nunique(),'founder_decision_fraction':float(f.first_tick.eq(1).mean()),
            'mean_pre_age':float(f.derived_pre_age.mean()),'mean_opportunity_index':float(f.opportunity_index.mean()),
            'transitions':result['transitions'],**{f'run_cdf_{i}':v for i,v in enumerate(cdf)}})
        for token,n in sorted(counts.items()):frequencies.append({'window':label,'token':token,'n':n,'decisions':len(f),'frequency':n/len(f)})
        for (length,left,reason),n in sorted(hist.items()):segments.append({'window':label,'length':length,'left_censored':left,'right_reason':reason,'segments':n})
        for value,n in sorted(vhist.items()):visits.append({'window':label,'visits':value,'states':n})
    return pd.DataFrame(rows),pd.DataFrame(frequencies),pd.DataFrame(segments),pd.DataFrame(visits)


def summarize(frame,keys,metrics):
    # Realizations first, ecological seeds second; missing support remains NA.
    s=frame.groupby([*keys,'seed'],sort=True,dropna=False)[metrics].mean().reset_index()
    t=s.melt(id_vars=[*keys,'seed'],value_vars=metrics,var_name='metric',value_name='measurement')
    return t.groupby([*keys,'metric'],sort=True,dropna=False).measurement.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'})


def diagnose(root=Path('data/social_learning_v074'),output=Path('analysis/social_learning_v074/nonstationarity')):
    output=Path(output)
    if output.exists():raise ValueError('Refusing existing diagnostic output')
    rows=[];freq=[];runs=[];visit=[];strata=[];inputs={}
    for c in 'BC':
        for seed in (3,4):
            path=source(c,seed);m,files,f=prepared(path);inputs[str(path)]=hash_file(path)
            tokens=f.b_token if c=='B' else f.c_token
            frames=profiles(f,tokens,m['tick']);tag={'condition':c,'seed':seed,'realization':0,'phase':'calibration'}
            ticks=read_csv(files['ticks']).set_index('tick').population.to_dict()
            frames[0]['mean_start_population']=[float(f.loc[f.tick.ge(r.start_tick)&f.tick.lt(r.stop_tick_exclusive),'tick'].map(lambda t:ticks[t-1]).mean()) for r in frames[0].itertuples()]
            rows.append(frames[0].assign(**tag));freq.append(frames[1].assign(**tag));runs.append(frames[2].assign(**tag));visit.append(frames[3].assign(**tag))
            catalog={r.base_id:json.loads(r.key_json) for r in read_csv(files['states']).itertuples(index=False)}
            contexts=pd.DataFrame({'token':tokens,'window':(f.tick//100).astype(str),'population_bin':f.tick.map(lambda t:0 if ticks[t-1]<200 else 1 if ticks[t-1]<500 else 2 if ticks[t-1]<1000 else 3),
                'visible_count_bin':np.where(f.b_token.eq(0),0,np.where(f.b_token.le(5),1,2)),
                'pre_age_bin':pd.cut(f.derived_pre_age,[-1,24,49,99,199,10**9],labels=False),
                'opportunity_bin':pd.cut(f.opportunity_index,[-1,24,49,99,199,10**9],labels=False),
                'cohort':np.where(f.first_tick.eq(1),'founder','born_during_run'),
                'food_visible':f.base_id.map(lambda b:int(tuple(catalog[b][3:5])!=(2,2))),
                'water_visible':f.base_id.map(lambda b:int(tuple(catalog[b][5:7])!=(2,2))),
                'food_here':f.base_id.map(lambda b:int(tuple(catalog[b][3:5])==(0,0))),
                'water_here':f.base_id.map(lambda b:int(tuple(catalog[b][5:7])==(0,0)))})
            for name in contexts.columns.drop(['token','window']):
                for (window,value),g in contexts.groupby(['window',name],observed=True,sort=True):
                    counts=g.token.value_counts();h=entropy(counts.values)
                    strata.append({**tag,'window':window,'diagnostic':name,'value':value,'decisions':len(g),'entropy':h,'effective_tokens':2**h})
    output.mkdir(parents=True)
    tables={'windows':pd.concat(rows,ignore_index=True),'frequencies':pd.concat(freq,ignore_index=True),'run_segments':pd.concat(runs,ignore_index=True),'state_visits':pd.concat(visit,ignore_index=True),'diagnostic_strata':pd.DataFrame(strata)}
    tables['summary']=summarize(tables['windows'],['condition','window'],['token_entropy','conditional_entropy','conditional_supported_entropy','first_state_fraction','mean_visits','fraction_states_once','same_token'])
    for name,f in tables.items():f.to_csv(output/f'{name}.csv',index=False,float_format='%.12g')
    report={'phase':'calibration_only','inputs':inputs,'windows':'[0,100),[100,200),...,[400,500); tick 0 has no decisions; tick 500 belongs only to full-run diagnostics.',
            'age':'Derived exactly as own complete-from-birth opportunity index, starting zero. Current physiology increments age once per own decision; age and opportunities are collinear, not independently identified.',
            'resources':'Actual encoded visibility/here indicators; no resource quantities or acquisition outcomes inferred.',
            'limits':'X-independent U has H(U|X,w)=H(U|w) in the ideal population. Social H(S)-H(S|X) diagnoses incompatibility, but plug-in MI is upward biased by sparse X; supported min20 exposure retained. Birth selection and calendar time are descriptive, not causal.'}
    (output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    print(tables['windows'][['condition','seed','window','decisions','mean_start_population','token_entropy','conditional_entropy','same_token','first_state_fraction','mean_pre_age']].to_string(index=False),flush=True)
    return report
