"""Held-out resource-cue prediction using the frozen v074 count-model family."""
from pathlib import Path
from time import perf_counter
import argparse,json
import numpy as np
import pandas as pd

from experiments.nonstationary_prediction import Hierarchy,Score
from experiments.controls_prediction import deep_tuple
from simulation.controls_observer import C_PAIRS
from simulation.vocal_learning_observer import hash_file
from experiments.resource_ecology_experiments import ROOT,ANALYSIS,load,case_path,ecology_config,verify_frozen,save_json,peak
from experiments.analyze_resource_ecology import summary


def keys(row):
    x=row.original;a=int(row.action);b=int(row.b_token);rd=C_PAIRS[int(row.c_token)][1]
    return {'X_A':(x,a),'X_B_A':(x,a,b),'X_C_A':(x,a,b,rd)}


def targets(frame):
    n=len(frame);result={'immediate_reward':np.where(frame.reward.le(-.1),0,np.where(frame.reward.lt(0),1,2)).astype(int),
        'acquisition_attempt':np.where(frame.action.isin([4,5]),frame.acquired,-1).astype(int),
        'food_near3_tickstart':frame.food_near3_tickstart.to_numpy(dtype=int),'water_near3_tickstart':frame.water_near3_tickstart.to_numpy(dtype=int),
        'future5_acquisition':np.full(n,-1,dtype=int)}
    reasons=np.full(n,'end_censored',dtype=object);ticks=frame.tick.to_numpy();acquired=frame.acquired.to_numpy();terminal=frame.terminal.to_numpy()
    for indices in frame.groupby('agent_id',sort=False).indices.values():
        for j,i in enumerate(indices):
            future=indices[j+1:j+6]
            if len(future)==5 and np.array_equal(ticks[future],np.arange(ticks[i]+1,ticks[i]+6)):
                result['future5_acquisition'][i]=int(acquired[future].any());reasons[i]='observed'
            else:reasons[i]='gap_censored' if len(future)==5 else 'death_censored' if terminal[indices[-1]] else 'end_censored'
    return result,reasons


def dataset(path):
    m,files=load(path);window=pd.read_csv(files['windows']);n=int(window.loc[window.tick.le(200),'decisions'].sum())
    columns=['tick','agent_id','base_id','b_token','c_token','action','reward','terminal','acquired','food_near3_tickstart','water_near3_tickstart']
    f=pd.read_csv(files['acquisitions'],usecols=columns,nrows=n,float_precision='round_trip');f=f[f.tick.le(200)].reset_index(drop=True)
    catalog={r.base_id:deep_tuple(json.loads(r.key_json)) for r in pd.read_csv(files['states']).itertuples(index=False)}
    f['original']=f.base_id.map(catalog)
    # A diagnostic sufficiency check, never a Brain rule or predictor feature.
    oracle=[int(row.action==4 and tuple(row.original[3:5])==(0,0) or row.action==5 and tuple(row.original[5:7])==(0,0)) for row in f.itertuples(index=False)]
    if not np.array_equal(oracle,f.acquired.to_numpy()):raise ValueError('Acquisition not consistent with original here-resource state/action')
    labels,reasons=targets(f)
    return f,labels,reasons


def training_path(root,frozen,ecology,seed):
    expected=ecology_config(ecology,frozen['scarcity'],frozen['localization']);candidates=[]
    from dataclasses import asdict
    for path in frozen['calibration_inputs']:
        if path.endswith(f'/{ecology}_A_seed_{seed}.csv.gz.metadata.json'):
            m=json.loads(Path(path).read_text())
            if m['config']['resource_ecology']==asdict(expected):candidates.append(Path(path))
    if len(candidates)!=1:raise ValueError('One selected A-calibration source required')
    return candidates[0]


def run(root=ROOT,output=ANALYSIS/'prediction'):
    root=Path(root);out=Path(output);frozen=verify_frozen(root)
    if out.exists():raise ValueError('Refusing prediction overwrite')
    start=perf_counter();rows=[];bins=[];censor=[];inputs={};names=('X_A','X_B_A','X_C_A')
    for e in ('E0','E1','E2','E3'):
        train=[];test=[]
        for seed in (3,4):
            path=training_path(root,frozen,e,seed);train.append(dataset(path));inputs[str(path)]=hash_file(path)
        for mode in 'ABC':
            for seed in (0,1,2):
                path=Path(str(case_path(root,'primary',e,mode,seed))+'.metadata.json');data=dataset(path);test.append((mode,seed,*data));inputs[str(path)]=hash_file(path)
        for target in train[0][1]:
            predictor=Hierarchy(names,3 if target=='immediate_reward' else 2);train_n=0
            for frame,labels,reasons in train:
                for row,label in zip(frame.itertuples(index=False),labels[target]):
                    if label>=0:predictor.add(keys(row),int(row.action),int(label));train_n+=1
            predictor.freeze()
            for mode,seed,frame,labels,reasons in test:
                scores={(name,subset):Score() for name in names for subset in ('all','common')};eligible=common_n=0
                for row,label in zip(frame.itertuples(index=False),labels[target]):
                    if label<0:continue
                    predictions,support=predictor.predict(keys(row),int(row.action));common=all(support.values());eligible+=1;common_n+=common
                    for name in names:
                        scores[name,'all'].add(row.agent_id,predictions[name],int(label),support[name])
                        if common:scores[name,'common'].add(row.agent_id,predictions[name],int(label),support[name])
                tag={'ecology':e,'mode':mode,'seed':seed,'target':target}
                for (name,subset),score in scores.items():
                    rows.append({**tag,'predictor':name,'subset':subset,**score.result(),'eligible':eligible,'source_decisions':len(frame),'common_coverage':common_n/eligible if eligible else None,'train_decisions':train_n})
                    for index,b in sorted(score.bins.items()):bins.append({**tag,'predictor':name,'subset':subset,'confidence_bin':index,'n':b[0],'confidence':b[1]/b[0],'accuracy':b[2]/b[0]})
                if target=='future5_acquisition':
                    from collections import Counter
                    censor.extend({**tag,'reason':reason,'n':n} for reason,n in Counter(reasons).items())
                print(f'Prediction {e}/{mode} seed={seed} {target} common={common_n}/{eligible}',flush=True)
    out.mkdir(parents=True);f=pd.DataFrame(rows);f.to_csv(out/'by_seed.csv',index=False,float_format='%.12g')
    summary(f,['ecology','mode','target','predictor','subset'],['log_loss','brier','equal_agent_log_loss','coverage','common_coverage']).to_csv(out/'summary.csv',index=False,float_format='%.12g')
    pd.DataFrame(bins).to_csv(out/'calibration_bins.csv',index=False,float_format='%.12g');pd.DataFrame(censor).to_csv(out/'censoring.csv',index=False)
    paired=[]
    for key,g in f.groupby(['ecology','mode','seed','target','subset']):
        e,mode,seed,target,subset=key;g=g.set_index('predictor')
        for a,b in [('X_B_A','X_A'),('X_C_A','X_B_A')]:
            paired.append({'ecology':e,'mode':mode,'seed':seed,'target':target,'subset':subset,'contrast':a+'-'+b,'log_loss_difference':g.loc[a,'log_loss']-g.loc[b,'log_loss'],
                'brier_difference':g.loc[a,'brier']-g.loc[b,'brier'],'common_decisions':int(g.loc[a,'decisions']) if subset=='common' else None})
    pd.DataFrame(paired).to_csv(out/'paired_by_seed.csv',index=False,float_format='%.12g')
    save_json(out/'metadata.json',{'version':'0.8.0','inputs':inputs,'train_seeds':[3,4],'test_seeds':[0,1,2],'train_policy':'A calibration only','test_policy':'Actual A/B/C first200 ticks, same ecology',
        'model':'v074 hierarchical frequency predictor,min20,strength5; no tuning/nuisance/model sweep','wall_seconds':perf_counter()-start,'peak_memory_bytes':peak(),
        'source_sha256':{__file__:hash_file(__file__)},
        'critical_sufficiency':'Current acquisition is deterministic from original X here-resource direction + selected eat/drink action, verified against actual units in every row. Any model score gain for this target is estimation/regularization, not added social information.',
        'limits':'Early200 only; A-trained behavior-policy shift; shared common support identical across predictors per test case. Clock-start near3 targets are privileged diagnostics, not Q inputs. Future acquisition requires5 actual consecutive future rows, otherwise censored. Prediction is not useful policy or causal benefit.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=ANALYSIS/'prediction')
    try:run(**vars(p.parse_args()))
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
