"""Feature-only calibration; reward/action/outcome columns never enter fitting."""
import argparse,json,math
from pathlib import Path
from statistics import mean

from agents.nuisance import NuisanceModel
from experiments.controls_metrics import archived_profile,original_frame,sequence_metrics,attach_nuisance,tokens_for
from simulation.vocal_learning_observer import hash_file

TARGETS={'first_state_fraction':.02,'relative_mean_visits':.10,'token_entropy':.25,'conditional_entropy':.25,'same_token':.05}


def fit(frames,condition):
    if condition not in ('B','C'):raise ValueError('Calibrate B/C only')
    k=11 if condition=='B' else 37
    social=[sequence_metrics(f,tokens_for(f,condition))[0] for f in frames]
    probabilities=[]
    for t in range(k):probabilities.append(mean(sum(x==t for x in tokens_for(f,condition))/len(f) for f in frames))
    # Correct only floating summation remainder, not an empirical model adjustment.
    probabilities[max(range(k),key=probabilities.__getitem__)]+=1-sum(probabilities)
    repeat=mean(r['same_token'] for r in social);independent=sum(p*p for p in probabilities)
    estimate=(repeat-independent)/(1-independent) if independent<1 else 0.
    candidates=sorted({x for d in (-.2,-.1,0,.1,.2) if 0<= (x:=estimate+d)<1} | {0.})
    trials=[]
    # Fitting sees only original-state labels and token sequences, never actions or rewards.
    feature_frames=[f[['agent_id','base_id','b_token','c_token','terminal']].assign(terminal=0) for f in frames]
    for alpha in candidates:
        model=NuisanceModel(tuple(probabilities),alpha);rows=[]
        for f,target in zip(feature_frames,social):
            for nseed in (11,29):
                result=sequence_metrics(f,attach_nuisance(f,model,nseed))[0]
                score=((result['first_state_fraction']-target['first_state_fraction'])/.02)**2
                score+=((result['mean_visits']/target['mean_visits']-1)/.10)**2
                score+=((result['token_entropy']-target['token_entropy'])/.25)**2
                score+=((result['same_token']-target['same_token'])/.05)**2
                rows.append({'nuisance_seed':nseed,'metrics':result,'score':score})
        trials.append({'alpha':alpha,'score':mean(r['score'] for r in rows),'evaluations':rows})
    selected=min(trials,key=lambda x:(x['score'],x['alpha']))
    return {'probabilities':probabilities,'hold_probability':selected['alpha'],'stationary_same_probability':selected['alpha']+(1-selected['alpha'])*independent,
            'estimated_hold_probability':estimate,'trials':trials,'target_metrics':social,'targets':TARGETS,
            'limitations':'Stationary one-parameter persistence cannot reproduce arbitrary state-conditional distributions or heavy-tailed runs. No live X/social/outcome is read. Observed correlations after selection remain possible.'}


def run(root=Path('data/social_learning_v073')):
    root=Path(root);output=root/'models'
    if output.exists():raise ValueError('Refusing to overwrite frozen calibration')
    audit=[];sources={};models={}
    for c in 'BC':
        for s in (0,1,2):
            p=Path('data/social_learning_v072/primary')/f'{c}_seed_{s}.csv.gz.metadata.json'
            r,counts,_=archived_profile(p);audit.append({'condition':c,'seed':s,'metrics':r,'counts':dict(counts),'temporal':'NA: archived action cells are unordered'})
            sources[str(p)]=hash_file(p)
        frames=[]
        for s in (3,4):
            p=root/'calibration'/f'{c}_seed_{s}_r0.csv.gz.metadata.json';m,files,f=original_frame(p)
            frames.append(f);sources[str(p)]=hash_file(p)
        models[{'B':'D','C':'E'}[c]]=fit(frames,c)
    output.mkdir(parents=True)
    for c,model in models.items():
        model.update(version='0.7.3',source_condition={'D':'B','E':'C'}[c],calibration_ecological_seeds=[3,4],evaluation_ecological_seeds=[0,1,2])
        (output/f'{c}.json').write_text(json.dumps(model,indent=2,allow_nan=False)+'\n')
    # Archived temporal NA is serialized as null, never substituted with contact durations.
    def clean(obj):
        if isinstance(obj,float) and not math.isfinite(obj):return None
        if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
        if isinstance(obj,list):return [clean(v) for v in obj]
        return obj
    frozen={'sources':sources,'archive_audit':clean(audit),'models':{c:hash_file(output/f'{c}.json') for c in models},'targets':TARGETS,
            'fit_columns':['agent_id (ownership only)','base_id','b_token/c_token','actual decision order'],
            'excluded_columns':['action','reward','terminal (set zero during generation)','population','births','deaths'],
            'calibration_seeds':[3,4],'evaluation_seeds':[0,1,2],'calibration_nuisance_seeds':[11,29]}
    (output/'frozen.json').write_text(json.dumps(frozen,indent=2,allow_nan=False)+'\n')
    print('Frozen feature-only models:',{c:(m['hold_probability'],m['probabilities']) for c,m in models.items()},flush=True)
    return frozen


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('data/social_learning_v073'))
    try:run(**vars(p.parse_args()))
    except (ValueError,OSError) as error:p.error(str(error))

if __name__=='__main__':main()
