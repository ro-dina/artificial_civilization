"""Final v080 preservation, accounting, gates and outcome-support audit."""
from pathlib import Path
import argparse,json,re
import numpy as np
import pandas as pd

from experiments.resource_ecology_experiments import ROOT,ANALYSIS,load,case_path,verify_frozen,pilot_passes,save_json
from experiments.resource_ecology_prediction import dataset
from simulation.vocal_learning_observer import hash_file

ALLOWED={'config.py','simulation/simulation.py','README.md','.gitignore'}


def audit(root=ROOT,output=ANALYSIS/'verification.json',test_log=None):
    root=Path(root);output=Path(output)
    if output.exists():raise ValueError('Refusing verification overwrite')
    before=json.loads((root/'before.json').read_text())['files'];protected=0
    for p,sha in before.items():
        if p not in ALLOWED:
            if hash_file(p)!=sha:raise ValueError('Historical artifact/source modified: '+p)
            protected+=1
    frozen=verify_frozen(root)
    if not pilot_passes(root):raise ValueError('Pilot gates not valid')
    inputs={};runtime=[];e0_exact=0
    for phase in ('pilot','primary'):
        initial={}
        for e in ('E0','E1','E2','E3'):
            for mode in 'ABC':
                for seed in (0,1,2):
                    p=Path(str(case_path(root,phase,e,mode,seed))+'.metadata.json');m,files=load(p);inputs[str(p)]=hash_file(p);s=m['statistics']
                    old=initial.setdefault(seed,m['initial_agents_rng_sha256'])
                    if old!=m['initial_agents_rng_sha256']:raise ValueError('Initial agents/RNG differ across factorial cases')
                    if m['config']['simulation']['learning_memory_capacity']!=256 or m['config']['simulation']['vocal_learning_memory_capacity']!=256:raise ValueError('Q capacity changed')
                    if e=='E0':
                        source=json.loads(Path(m['legacy_comparison']['file']).read_text())
                        if m['fingerprints']!=source['fingerprints'] or s!=source['statistics'] or m['action_sha256']!=source['action_sha256']:raise ValueError('E0 complete regression')
                        e0_exact+=1
                    if m['files']['acquisitions']['records']!=m['totals']['decisions'] or m['files']['individuals']['records']!=100+s['births']:raise ValueError('Missing essential outcomes')
                    life=pd.read_csv(files['individuals'])
                    if life.food.sum()!=s['food_consumed'] or life.water.sum()!=s['water_consumed'] or life.censored.eq(0).sum()!=s['deaths']:raise ValueError('Individual outcome mismatch')
                    r=pd.read_csv(files['resource_ticks']);t=pd.read_csv(files['ticks'])
                    for res in ('food','water'):
                        expected=m['config']['simulation']['initial_'+res]+t[res+'_regenerated']-t[res+'_consumed']
                        if not np.array_equal(expected,r[res+'_stock']):raise ValueError('Resource stock conservation')
                        if not r[res+'_attempted'].fillna(0).equals(r[res+'_realized'].fillna(0)) or r[res+'_rejected'].fillna(0).ne(0).any():raise ValueError('Supply loss')
                    if phase=='primary' and 'mean_vocal_q_rows' not in r:raise ValueError('Missing primary vocal occupancy')
                    runtime.append({'phase':phase,'ecology':e,'mode':mode,'seed':seed,**m['runtime'],'stream_bytes':sum(v['bytes'] for v in m['files'].values())})
    # Repeated schema2 run: all streams byte-identical; old schema1 differs only
    # by added passive vocal-occupancy columns, not trajectories or shared values.
    a,af=load(root/'determinism/E3_A_seed_0.csv.gz.metadata.json');b,bf=load(root/'determinism_repeat/E3_A_seed_0.csv.gz.metadata.json')
    old,of=load(root/'pilot/E3_A_seed_0.csv.gz.metadata.json')
    if a['fingerprints']!=b['fingerprints'] or a['fingerprints']!=old['fingerprints'] or a['statistics']!=old['statistics']:raise ValueError('Deterministic full state/RNG mismatch')
    if any(hash_file(af[k])!=hash_file(bf[k]) for k in af):raise ValueError('Deterministic stream mismatch')
    for k in af:
        if k!='resource_ticks' and hash_file(af[k])!=hash_file(of[k]):raise ValueError('Observational schema upgrade altered old stream')
    x=pd.read_csv(af['resource_ticks'],float_precision='round_trip');y=pd.read_csv(of['resource_ticks'],float_precision='round_trip')
    if not x[list(y)].equals(y):raise ValueError('Old resource snapshot values changed')
    # Validate the saved resource-only audit, not merely equal probabilities.
    supply=pd.read_csv(root/'supply_audit.csv.gz');reset=supply[supply.regime.eq('reset_empty')]
    for seed in (3,4):
        for left,right in [('E0','E2'),('E1','E3')]:
            a=reset[reset.seed.eq(seed)&reset.ecology.eq(left)].sort_values(['tick','resource']).reset_index(drop=True)
            b=reset[reset.seed.eq(seed)&reset.ecology.eq(right)].sort_values(['tick','resource']).reset_index(drop=True)
            if not a[['attempted','realized','budget_rng_sha256']].equals(b[['attempted','realized','budget_rng_sha256']]):raise ValueError('Conditional supply equality failed')
    log_path=Path(test_log) if test_log else root/'final_tests.log'
    log=log_path.read_text();test=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
    if not test or '\nOK\n' not in log or 'FAILED (' in log:raise ValueError('Full test suite must pass')
    # Target prevalence is essential: near-resource presence can be saturated.
    # This is a descriptive outcome-support audit, not a model/parameter refit.
    prevalence=[]
    for p in inputs:
        if '/primary/' not in p:continue
        m=json.loads(Path(p).read_text());frame,labels,reasons=dataset(p)
        for target,values in labels.items():
            observed=values>=0;n=int(observed.sum())
            prevalence.append({'ecology':m['ecology'],'mode':m['mode'],'seed':m['seed'],'target':target,'source_decisions':len(frame),'eligible':n,
                **{'class_'+str(i):int((values==i).sum()) for i in (0,1,2)},'positive_fraction_binary':float(values[observed].mean()) if n and target!='immediate_reward' else None})
    predout=ANALYSIS/'prediction/target_prevalence.csv'
    import io
    for path,frame in [(predout,pd.DataFrame(prevalence)),(ANALYSIS/'runtime_by_case.csv',pd.DataFrame(runtime))]:
        text=frame.to_csv(index=False,float_format='%.12g')
        if path.exists():
            if path.read_text()!=text:raise ValueError('Existing audit table differs; refusing overwrite')
        else:path.write_text(text)
    receipt={'version':'0.8.0','passed':True,'protected_files_unchanged':protected,'baseline_inventory_files':len(before),'allowed_existing_changes':sorted(ALLOWED),
        'all_historical_experiments_preserved':True,'full_tests_passed':int(test[1]),'full_tests_seconds':float(test[2]),'test_log':str(log_path),'test_log_sha256':hash_file(log_path),
        'e0_complete_state_rng_exact':e0_exact,'pilot_cases':36,'primary_cases':36,'all_primary_completed500':all(json.loads(Path(p).read_text())['tick']==500 for p in inputs if '/primary/' in p),
        'deterministic_stream_bytes_exact':True,'schema2_addition_observational_verified':True,'resource_mass_balance_all_cases':True,'individual_acquisition_life_accounting':True,
        'conditional_supply_reset_equality_verified':True,'all_actual_offers_realized':True,'frozen_design_and_parameters_verified':True,'parameters':{k:frozen[k] for k in ('scarcity','localization')},
        'inputs':inputs,'data_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),'analysis_bytes_before_receipt':sum(p.stat().st_size for p in ANALYSIS.rglob('*') if p.is_file()),
        'core_unchanged_sha256':{p:hash_file(p) for p in ('world/world.py','agents/learning.py','agents/social_learning.py','agents/genome.py','systems/reproduction.py','simulation/statistics.py')},
        'source_sha256':{str(p):hash_file(p) for p in sorted(Path('experiments').glob('*resource_ecology*.py'))},
        'limitations':'Supply equality is conditional on eligibility, not unconditional across endogenous spatial saturation. Three ecological seeds. Main A/B/C fragmentation confound remains. Prediction A-trained, early200 only; immediate acquisition already determined by X and action.'}
    save_json(output,receipt);print(f'Verified {protected} protected files, {e0_exact} complete E0 regressions, 72 factorial cases, supply/life accounting, deterministic streams and {test[1]} tests.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=ANALYSIS/'verification.json');p.add_argument('--test-log',type=Path)
    try:audit(**vars(p.parse_args()))
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
