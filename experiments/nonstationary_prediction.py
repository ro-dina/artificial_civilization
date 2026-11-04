"""Prespecified hierarchical outcome prediction. Never used by an agent."""
from collections import Counter,defaultdict
import argparse,math
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd

from experiments.controls_metrics import attach_nuisance,read_csv
from experiments.controls_prediction import deep_tuple
from experiments.information_controls import model_file
from experiments.nonstationary_metrics import source,prepared,summarize
from experiments.nonstationary_controls import ROOT,ANALYSIS,save_json,verify_frozen,peak_bytes
from simulation.controls_observer import C_PAIRS
from simulation.vocal_learning_observer import hash_file
import json

HEADS=('X_A','X_B_A','X_C_A','X_UD_A','X_UE_A','X_B_URD_A')
PARENT={'X_A':None,'X_B_A':'X_A','X_C_A':'X_B_A','X_UD_A':'X_A','X_UE_A':'X_A','X_B_URD_A':'X_B_A'}


def features(x,action,b,c,ud,ue):
    return {'X_A':(x,action),'X_B_A':(x,action,b),'X_C_A':(x,action,*C_PAIRS[c]),
            'X_UD_A':(x,action,ud),'X_UE_A':(x,action,ue),'X_B_URD_A':(x,action,b,C_PAIRS[ue][1])}


class Hierarchy:
    def __init__(self,names,classes,minimum=20,strength=5):
        self.names=names;self.classes=classes;self.minimum=minimum;self.strength=strength
        self.cells={name:{} for name in names};self.actions={};self.frozen=False
    def add(self,keys,action,label):
        if self.frozen:raise RuntimeError('Predictive training is frozen')
        if not 0<=label<self.classes:raise ValueError('Observed target class required')
        counts=self.actions.setdefault(action,[0]*self.classes);counts[label]+=1
        for name in self.names:self.cells[name].setdefault(keys[name],[0]*self.classes)[label]+=1
    def freeze(self):self.frozen=True
    def predict(self,keys,action):
        if not self.frozen:raise RuntimeError('Freeze before evaluation')
        counts=self.actions.get(action,[0]*self.classes);n=sum(counts);prior=[(v+1)/(n+self.classes) for v in counts]
        results={};support={}
        for name in self.names:
            parent=results[PARENT[name]] if PARENT[name] else prior
            cell=self.cells[name].get(keys[name],[0]*self.classes);n=sum(cell);support[name]=n>=self.minimum
            results[name]=[(cell[i]+self.strength*parent[i])/(n+self.strength) for i in range(self.classes)] if support[name] else parent
        return results,support


def outcome_targets(frame,horizon=5):
    """Use actual future rows; no absorbing reward padding or missing-row guess."""
    if type(horizon) is not int or horizon<1:raise ValueError('Positive fixed horizon required')
    n=len(frame);values={name:np.full(n,-1,dtype=np.int8) for name in ('immediate_reward','future_reward_5','survival_5')}
    reasons={name:np.full(n,'run_end_censored',dtype=object) for name in values}
    def bin_reward(r):return 0 if r<=-.1 else 1 if r<0 else 2
    for i,r in enumerate(frame.reward):values['immediate_reward'][i]=bin_reward(r);reasons['immediate_reward'][i]='observed'
    ticks=frame.tick.to_numpy();rewards=frame.reward.to_numpy();terminal=frame.terminal.to_numpy()
    for indices in frame.groupby('agent_id',sort=False).indices.values():
        ts=ticks[indices];last=indices[-1]
        for j,i in enumerate(indices):
            future=indices[j+1:j+1+horizon]
            if len(future)==horizon and np.array_equal(ticks[future],np.arange(ticks[i]+1,ticks[i]+horizon+1)):
                values['future_reward_5'][i]=bin_reward(sum(float(rewards[k]) for k in future));reasons['future_reward_5'][i]='observed'
            else:reasons['future_reward_5'][i]='gap_censored' if len(future)==horizon else 'death_censored' if terminal[last] else 'run_end_censored'
            position=int(np.searchsorted(ts,ticks[i]+horizon))
            if terminal[last] and ticks[last]<=ticks[i]+horizon:
                values['survival_5'][i]=0;reasons['survival_5'][i]='observed_death'
            elif position<len(indices) and ts[position]==ticks[i]+horizon:
                values['survival_5'][i]=int(not terminal[indices[position]]);reasons['survival_5'][i]='observed_horizon'
            elif ts[-1]>=ticks[i]+horizon:reasons['survival_5'][i]='gap_censored'
    return values,reasons


def dataset(condition,seed,root,realization):
    p=source(condition,seed);m,files,f=prepared(p)
    catalog={r.base_id:deep_tuple(json.loads(r.key_json)) for r in read_csv(files['states']).itertuples(index=False)}
    f['original']=f.base_id.map(catalog)
    nseed=60000+100*seed+realization
    f['ud']=attach_nuisance(f,model_file(Path(root)/'models/D.json'),nseed)
    f['ue']=attach_nuisance(f,model_file(Path(root)/'models/E.json'),nseed)
    return f


def row_keys(row):return features(row.original,row.action,row.b_token,row.c_token,row.ud,row.ue)


class Score:
    def __init__(self):
        self.n=0;self.support=0;self.loss=0.;self.brier=0.;self.agents=defaultdict(lambda:[0.,0.,0]);self.bins={}
    def add(self,identity,probabilities,target,supported):
        loss=-math.log(probabilities[target]);brier=sum((p-(i==target))**2 for i,p in enumerate(probabilities))
        self.n+=1;self.support+=supported;self.loss+=loss;self.brier+=brier
        agent=self.agents[identity];agent[0]+=loss;agent[1]+=brier;agent[2]+=1
        confidence=max(probabilities);chosen=probabilities.index(confidence);index=min(9,int(10*confidence))
        b=self.bins.setdefault(index,[0,0.,0]);b[0]+=1;b[1]+=confidence;b[2]+=chosen==target
    def result(self):
        return {'decisions':self.n,'agents':len(self.agents),'supported_decisions':self.support,
            'coverage':self.support/self.n if self.n else None,'log_loss':self.loss/self.n if self.n else None,
            'brier':self.brier/self.n if self.n else None,
            'equal_agent_log_loss':sum(v[0]/v[2] for v in self.agents.values())/len(self.agents) if self.agents else None,
            'equal_agent_brier':sum(v[1]/v[2] for v in self.agents.values())/len(self.agents) if self.agents else None,
            'ECE':sum(abs(b[1]-b[2]) for b in self.bins.values())/self.n if self.n else None}


def run(root=ROOT,output=ANALYSIS/'prediction'):
    root=Path(root);output=Path(output);verify_frozen(root)
    if output.exists():raise ValueError('Refusing prediction overwrite')
    start=perf_counter();rows=[];calibration=[];censoring=[];inputs={};windows=[]
    for condition in 'BC':
        names=('X_A','X_B_A','X_UD_A') if condition=='B' else HEADS
        for realization in (0,1):
            training=[]
            for seed in (3,4):
                f=dataset(condition,seed,root,realization);targets,reasons=outcome_targets(f);training.append((f,targets))
                inputs[str(source(condition,seed))]=hash_file(source(condition,seed))
                for target,labels in reasons.items():
                    censoring.extend({'source_condition':condition,'phase':'train','seed':seed,'realization':realization,'target':target,'reason':reason,'decisions':n} for reason,n in Counter(labels).items())
            # One target at a time bounds the aggregate training-table footprint.
            for target in ('immediate_reward','future_reward_5','survival_5'):
                predictor=Hierarchy(names,2 if target=='survival_5' else 3);train_agents=train_decisions=0
                for f,targets in training:
                    observed=targets[target]>=0;train_agents+=f.loc[observed,'agent_id'].nunique();train_decisions+=int(observed.sum())
                    for row,label in zip(f.itertuples(index=False),targets[target]):
                        if label>=0:predictor.add(row_keys(row),row.action,int(label))
                predictor.freeze()
                for seed in (0,1,2):
                    f=dataset(condition,seed,root,realization);targets,reasons=outcome_targets(f);inputs[str(source(condition,seed))]=hash_file(source(condition,seed))
                    censoring.extend({'source_condition':condition,'phase':'test','seed':seed,'realization':realization,'target':target,'reason':reason,'decisions':n} for reason,n in Counter(reasons[target]).items())
                    scores={(name,subset):Score() for name in names for subset in ('all','common')};window_scores={}
                    available=common_count=0
                    for row,label in zip(f.itertuples(index=False),targets[target]):
                        if label<0:continue
                        available+=1;probabilities,support=predictor.predict(row_keys(row),row.action);common=all(support.values());common_count+=common
                        for name in names:
                            for subset in (('all','common') if common else ('all',)):
                                scores[name,subset].add(row.agent_id,probabilities[name],int(label),support[name])
                            key=(name,row.tick//100)
                            window_scores.setdefault(key,Score()).add(row.agent_id,probabilities[name],int(label),support[name])
                    tag={'source_condition':condition,'seed':seed,'realization':realization,'target':target,'nuisance_seed':60000+100*seed+realization}
                    for (name,subset),score in scores.items():
                        result={**tag,'model':name,'subset':subset,**score.result(),'eligible_decisions':available,'all_source_decisions':len(f),'common_decisions':common_count,
                            'common_coverage':common_count/available if available else None,'target_coverage':available/len(f),'train_agents':train_agents,'train_decisions':train_decisions}
                        rows.append(result)
                        for index,b in sorted(score.bins.items()):calibration.append({**tag,'model':name,'subset':subset,'confidence_bin':index,'decisions':b[0],'mean_confidence':b[1]/b[0],'accuracy':b[2]/b[0]})
                    for (name,window),score in window_scores.items():windows.append({**tag,'model':name,'window':window,**score.result()})
                    print(f'Prediction {condition} r={realization} {target} seed={seed} common={common_count}/{available}',flush=True)
    output.mkdir(parents=True);f=pd.DataFrame(rows);f.to_csv(output/'runs.csv',index=False,float_format='%.12g')
    summarize(f,['source_condition','model','subset','target'],['log_loss','brier','equal_agent_log_loss','equal_agent_brier','ECE','coverage','common_coverage','target_coverage']).to_csv(output/'summary.csv',index=False,float_format='%.12g')
    pd.DataFrame(calibration).to_csv(output/'calibration_bins.csv',index=False,float_format='%.12g');pd.DataFrame(censoring).to_csv(output/'censoring.csv',index=False);pd.DataFrame(windows).to_csv(output/'windows.csv',index=False,float_format='%.12g')
    paired=[]
    for key,g in f.groupby(['source_condition','seed','realization','target','subset']):
        c,seed,r,target,subset=key;g=g.set_index('model')
        contrasts=[('X_B_A','X_A'),('X_B_A','X_UD_A')]+([('X_C_A','X_B_A'),('X_C_A','X_B_URD_A'),('X_C_A','X_UE_A')] if c=='C' else [])
        for a,b in contrasts:
            for metric in ('log_loss','brier','equal_agent_log_loss'):
                paired.append({'source_condition':c,'seed':seed,'realization':r,'target':target,'subset':subset,'contrast':a+'-'+b,'metric':metric,'difference':g.loc[a,metric]-g.loc[b,metric]})
    pd.DataFrame(paired).to_csv(output/'paired_by_seed.csv',index=False,float_format='%.12g')
    save_json(output/'metadata.json',{'version':'0.7.4','phase':'held_out_prediction','train_seeds':[3,4],'test_seeds':[0,1,2],
        'inputs':inputs,'preregistration_sha256':hash_file(root/'preregistration.json'),'frozen_manifest_sha256':hash_file(root/'models/frozen.json'),
        'source_sha256':{__file__:hash_file(__file__)},'minimum_cell':20,'dirichlet_strength':5,'capacity_of_source_policy':256,
        'wall_seconds':perf_counter()-start,'peak_memory_bytes':peak_bytes(),
        'interpretation':'Same held-out decisions for all models, hierarchical parent backoff rather than unrelated prior. Common support is identical within each source/seed/realization/target. Ecological seed is replication; no significance or policy evaluation.',
        'clock':'Frozen U schedule may encode calendar stage; exogenous innovations do not guarantee zero outcome association. Per-calendar-window scores retained.',
        'targets':'Immediate actual reward; next5 ACTUAL future rewards excluding current, no terminal padding; observed survival after tick t+5. Death/end/gap censoring explicit; no fabricated resource acquisition.',
        'memory':'Compare X,B,A vs X,B,RD,A and X,B,U_RD,A. Actual B held fixed; U_RD is independent E projection, diagnostic-only66 possible combinations.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=ANALYSIS/'prediction')
    try:run(**vars(p.parse_args()))
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
