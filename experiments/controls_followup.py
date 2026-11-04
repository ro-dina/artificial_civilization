"""Optional compact fixed-experience learning replay and held-out prediction."""
import argparse,json
from pathlib import Path
from time import perf_counter
import pandas as pd

from experiments.controls_prediction import prediction,replay
from experiments.information_controls import model_file,verify_frozen
from simulation.vocal_learning_observer import hash_file


def run(root=Path('data/social_learning_v073'),output=Path('analysis/social_learning_v073/prediction'),include_replay=False,common_support=False):
    root=Path(root);output=Path(output)
    if output.exists():raise ValueError('Refusing existing follow-up output')
    verify_frozen(root)
    if include_replay:
        review=root/'pilot_review.json'
        if not review.exists() or json.loads(review.read_text()).get('decision')!='primary_allowed':raise ValueError('Replay deferred until primary matching review')
        if not (root/'primary').exists():raise ValueError('Complete primary first')
    start=perf_counter();f=pd.DataFrame(prediction(root,common_support));output.mkdir(parents=True)
    f.to_csv(output/'prediction_by_seed.csv',index=False,float_format='%.12g')
    group=['source_condition','model']+(['subset'] if common_support else [])
    summary=f.melt(id_vars=[*group,'seed'],value_vars=['log_loss','brier','equal_agent_log_loss','support_fraction'],var_name='metric',value_name='value')
    summary=summary.groupby([*group,'metric']).value.agg(['mean','std','min','max','count']).reset_index().rename(columns={'count':'n'})
    summary.to_csv(output/'prediction_summary.csv',index=False,float_format='%.12g')
    differences=[]
    for key,g in f.groupby(['source_condition','seed']+(['subset'] if common_support else [])):
        c,seed=key[:2];subset=key[2] if common_support else 'all'
        g=g.set_index('model')
        for metric in ('log_loss','brier','equal_agent_log_loss'):
            for alt,control in [('X_social_A','X_A'),('X_nuisance_A','X_A'),('X_social_A','X_nuisance_A')]:
                differences.append({'source_condition':c,'seed':seed,'subset':subset,'metric':metric,'contrast':alt+'-'+control,'difference':g.loc[alt,metric]-g.loc[control,metric]})
    pd.DataFrame(differences).to_csv(output/'prediction_paired_by_seed.csv',index=False,float_format='%.12g')
    from experiments.controls_plots import prediction_plot
    prediction_plot(f[f.subset.eq('common_supported')] if common_support else f,output,common_support)
    if include_replay:
        rows=[]
        for source in 'BC':
            for seed in (0,1,2):
                p=root/'references'/f'{source}_seed_{seed}_r0.csv.gz.metadata.json'
                for rep in 'ABCDE':rows.append(replay(p,rep,model_file(root/'models'/f'{rep}.json') if rep in 'DE' else None,80000+seed))
        pd.DataFrame(rows).to_csv(output/'fixed_experience_replay.csv',index=False,float_format='%.12g')
    sources=sorted((root/'calibration').glob('*.metadata.json'))+sorted((root/'references').glob('*.metadata.json'))
    m={'version':'0.7.3','train_ecological_seeds':[3,4],'test_ecological_seeds':[0,1,2],'independent_n':3,
       'fit_support_minimum':20,'dirichlet_strength':5,'reward_bin_edges':[-.1,0.],
       'common_support': 'Exploratory post-hoc diagnostic on identical test decisions with >=20 training observations for all heads. No outcome used to select support; original all-decision results retained separately.' if common_support else False,
       'target':'Existing actual immediate physiological reward, conditioned on actual chosen action. Test-agent repeated decisions remain within their ecological seed.',
       'limits':'Observational prediction with sparse-cell backoff. Features can alias biological eligibility; prediction is not policy/survival benefit. No unreliable finite-sample CMI estimate reported.',
       'replay':'Fixed actual experience, preserves read-before-update and LRU; last survivor transition censored, no synthetic terminal. Not ecological counterfactual.' if include_replay else 'Deferred: not executed.',
       'wall_seconds':perf_counter()-start,'inputs':{str(p):hash_file(p) for p in sources},
       'models':json.loads((root/'models/frozen.json').read_text())['models'],
       'source_sha256':{str(p):hash_file(p) for p in sorted(Path('experiments').glob('*controls*.py'))},
       'outputs':{p.name:{'sha256':hash_file(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir())}}
    (output/'metadata.json').write_text(json.dumps(m,indent=2)+'\n')
    print(f[['source_condition','seed','model','log_loss','support_fraction']].to_string(index=False),flush=True)
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('data/social_learning_v073'));p.add_argument('--output',type=Path,default=Path('analysis/social_learning_v073/prediction'));p.add_argument('--include-replay',action='store_true');p.add_argument('--common-support',action='store_true')
    try:run(**vars(p.parse_args()))
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
