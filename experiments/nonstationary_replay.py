"""Fixed actual B/C experience, independent of whether online matching passes."""
import argparse
from pathlib import Path
from time import perf_counter
import pandas as pd

from experiments.controls_prediction import replay
from experiments.information_controls import model_file
from experiments.nonstationary_metrics import source,summarize
from experiments.nonstationary_controls import ROOT,ANALYSIS,save_json,verify_frozen,peak_bytes
from simulation.vocal_learning_observer import hash_file


def run(root=ROOT,output=ANALYSIS/'replay',validate_only=False):
    root=Path(root);output=Path(output)
    if output.exists():raise ValueError('Refusing existing replay output')
    verify_frozen(root);start=perf_counter();inputs={};validation=[]
    for c in 'ABC':
        for seed in (0,1,2):
            p=source(c,seed);r=replay(p,c);inputs[str(p)]=hash_file(p)
            validation.append(r);print(f'Exact physical Q/LRU/pending projection: {c} seed={seed}',flush=True)
    output.mkdir(parents=True);pd.DataFrame(validation).to_csv(output/'source_validation.csv',index=False)
    rows=[];probes=[];histograms=[]
    if not validate_only:
        for c in 'BC':
            for seed in (0,1,2):
                p=source(c,seed)
                for representation in ('A','B','C','D','E','B_Umemory'):
                    for realization in ((0,1) if representation in ('D','E','B_Umemory') else (0,)):
                        nseed=50000+100*seed+realization;which='D' if representation=='D' else 'E'
                        model=model_file(root/'models'/f'{which}.json') if representation in ('D','E','B_Umemory') else None
                        r=replay(p,representation,model,nseed,diagnostics=True);d=r.pop('diagnostics')
                        tag={'source_condition':c,'seed':seed,'representation':representation,'realization':realization}
                        rows.append({**r,**tag})
                        probes.extend({**tag,**v} for v in d['probes'])
                        for name in ('state_visits','state_updates','retained'):
                            histograms.extend({**tag,'histogram':name,**v} for v in d[name])
                        print(f'Fixed experience {c} seed={seed} representation={representation} r={realization} novelty={r["first_state_fraction"]:.4f}',flush=True)
        f=pd.DataFrame(rows);f.to_csv(output/'runs.csv',index=False,float_format='%.12g')
        metrics=['first_state_fraction','mean_visits','states_once_fraction','updates','evictions','retained_rows','mean_retained_q_range','nonflat_retained_fraction','retention_fraction']
        summarize(f,['source_condition','representation'],metrics).to_csv(output/'summary.csv',index=False,float_format='%.12g')
        q=pd.DataFrame(probes);q['policy_tv']=q.tv_sum/q.pairs.replace(0,float('nan'));q.to_csv(output/'probes.csv',index=False,float_format='%.12g')
        summarize(q,['source_condition','representation','kind','min_visits'],['policy_tv','pairs','qualified_rows','supported_agents']).to_csv(output/'probe_summary.csv',index=False,float_format='%.12g')
        pd.DataFrame(histograms).to_csv(output/'histograms.csv',index=False,float_format='%.12g')
        paired=[]
        averaged=f.groupby(['source_condition','seed','representation']).mean(numeric_only=True)
        for (c,seed),g in averaged.groupby(level=[0,1]):
            for alt,reference in [('B','D'),('C','E'),('C','B'),('C','B_Umemory')]:
                for metric in metrics:
                    paired.append({'source_condition':c,'seed':seed,'contrast':alt+'-'+reference,'metric':metric,'difference':g.loc[(c,seed,alt),metric]-g.loc[(c,seed,reference),metric]})
        pd.DataFrame(paired).to_csv(output/'paired_by_seed.csv',index=False,float_format='%.12g')
    save_json(output/'metadata.json',{'version':'0.7.4','phase':'fixed_experience','exact_source_cases':len(validation),'comparative_replay_cases':len(rows),
        'inputs':inputs,'preregistration_sha256':hash_file(root/'preregistration.json'),'frozen_manifest_sha256':hash_file(root/'models/frozen.json'),
        'source_sha256':{p:hash_file(p) for p in ('experiments/controls_prediction.py','experiments/nonstationary_replay.py')},
        'wall_seconds':perf_counter()-start,'peak_memory_bytes':peak_bytes(),'capacity':256,
        'exactness':'Full archived physical Q values/order and recorded pending(state,action,reward) projection. Pending before-physiology is unavailable, not fabricated, and not needed when replaying actual recorded rewards.',
        'memory':'C-B holds experience fixed. memory_only probes hold original X AND actual B fixed; B_Umemory substitutes an independent canonical RD projection of E (diagnostic66 possible slots, not an online condition).',
        'limits':'No new policy execution or ecological counterfactual; same actual actions/rewards/terminals, censored final pending transitions. Nuisance realizations grouped within ecological seed.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=ANALYSIS/'replay');p.add_argument('--validate-only',action='store_true')
    try:run(**vars(p.parse_args()))
    except (ValueError,OSError,KeyError,RuntimeError) as error:p.error(str(error))

if __name__=='__main__':main()
