"""v074 final audit; no simulation or model tuning, and no historical writes."""
from collections import Counter
import argparse,json,re
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.controls_metrics import load,read_csv
from experiments.nonstationary_metrics import source,prepared
from experiments.nonstationary_controls import ROOT,ANALYSIS,save_json,verify_frozen,primary_allowed,peak_bytes
from simulation.vocal_learning_observer import hash_file

ALLOWED={'.gitignore','README.md','agents/nuisance.py','main.py','simulation/simulation.py','simulation/controls_observer.py',
    'experiments/information_controls.py','experiments/controls_metrics.py','experiments/controls_prediction.py'}


def audit(root=ROOT,output=ANALYSIS/'verification.json'):
    root=Path(root);output=Path(output)
    if output.exists():raise ValueError('Refusing verification overwrite')
    before=json.loads((root/'before.json').read_text())['files'];protected=0
    for path,sha in before.items():
        if path not in ALLOWED:
            if hash_file(path)!=sha:raise ValueError('Protected source/history changed: '+path)
            protected+=1
    frozen=verify_frozen(root)
    for c in 'ABC':
        m,_=load(root/'compatibility'/f'{c}_seed_0_r0.csv.gz.metadata.json');old,_=load(Path('data/social_learning_v072/pilot')/f'{c}_seed_0.csv.gz.metadata.json')
        if any(m[k]!=old[k] for k in ('fingerprints','statistics','action_sha256')):raise ValueError('Historical ABC regression mismatch')
    for folder,original in [('stationary_compatibility',Path('data/social_learning_v073/pilot')),('determinism',root/'pilot')]:
        for c in 'DE':
            m,files=load(root/folder/f'{c}_seed_0_r0.csv.gz.metadata.json');old,of=load(original/f'{c}_seed_0_r0.csv.gz.metadata.json')
            if m['fingerprints']!=old['fingerprints'] or m['statistics']!=old['statistics'] or any(hash_file(files[k])!=hash_file(of[k]) for k in files):raise ValueError('Byte/state/RNG mismatch')
    for path in sorted((root/'pilot').glob('*.metadata.json')):
        m,_=load(path);seed=m['seed'];ref,_=load(Path('data/social_learning_v072/pilot')/f'A_seed_{seed}.csv.gz.metadata.json')
        if m['fingerprints']['initial_ecology']!=ref['fingerprints']['initial_ecology']:raise ValueError('Initial ecology/RNG mismatch')
        if m['config']['simulation']['learning_memory_capacity']!=256:raise ValueError('Capacity changed')
    validation=read_csv(ANALYSIS/'replay/source_validation.csv')
    if len(validation)!=9 or not validation.exact_source_reconstruction.all():raise ValueError('Missing exact source replay')
    for r in validation.itertuples():
        m,_=load(source(r.source_condition,r.seed))
        if r.physical_q_sha256!=m['physical_q_sha256']:raise ValueError('Source replay digest mismatch')
    review=json.loads((root/'pilot_review.json').read_text());checks=read_csv(review['matching_file']);gated=checks[checks.complete_window]
    if hash_file(review['matching_file'])!=review['matching_sha256']:raise ValueError('Matching evidence changed')
    if review['decision']=='primary_blocked' and ((root/'primary').exists() or primary_allowed(root)):raise ValueError('Failed gate was bypassed')
    log=(root/'final_tests.log').read_text();match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
    if not match or '\nOK\n' not in log or 'FAILED (' in log:raise ValueError('Final full suite must pass first')
    # Supplementary composition diagnostics, never used to fit or refit models.
    selection=[]
    for c in 'BC':
        for seed in (3,4):
            _,_,f=prepared(source(c,seed))
            for start in range(0,500,100):
                g=f[f.tick.ge(start)&f.tick.lt(start+100)]
                selection.append({'condition':c,'seed':seed,'window':start,'decisions':len(g),'agents':g.agent_id.nunique(),
                    'first_observed_agents':int(g.opportunity_index.eq(0).sum()),'newborn_first_observations':int((g.opportunity_index.eq(0)&g.first_tick.gt(1)).sum()),
                    'terminal_observations':int(g.terminal.sum()),'founder_exposure_fraction':g.first_tick.eq(1).mean(),'mean_pre_age':g.derived_pre_age.mean(),
                    'age_opportunity_identical':bool(g.derived_pre_age.eq(g.opportunity_index).all())})
    pd.DataFrame(selection).to_csv(ANALYSIS/'cohort_selection.csv',index=False,float_format='%.12g')
    variability=read_csv(ANALYSIS/'pilot/online_runs.csv').groupby(['condition','seed']).population.agg(['mean','std','min','max','count']).reset_index()
    variability.to_csv(ANALYSIS/'nuisance_variability.csv',index=False,float_format='%.12g')
    times={}
    for phase in ('pilot','determinism','compatibility','stationary_compatibility'):
        receipts=[json.loads(p.read_text()) for p in (root/phase).glob('*.runtime.json')]
        times[phase]={'cases':len(receipts),'per_case_wall_seconds_sum':sum(r['wall_seconds'] for r in receipts),
            'max_recorded_peak_memory_bytes':max((r.get('peak_memory_bytes',0) for r in receipts),default=0) or None}
    for phase in ('calibration','replay','prediction','figures'):
        m=json.loads((ANALYSIS/phase/'metadata.json').read_text());times[phase]={k:m[k] for k in ('wall_seconds','peak_memory_bytes')}
    summary_files=['calibration/windows.csv','pilot/window_summary.csv','replay/summary.csv','replay/probe_summary.csv','prediction/summary.csv']
    for file in summary_files:
        frame=read_csv(ANALYSIS/file)
        if 'n' in frame and frame.n.gt(3).any():raise ValueError('Nuisance seeds incorrectly counted as ecological replication')
    receipt={'version':'0.7.4','passed':True,'baseline_inventory_files':len(before),'protected_existing_files_unchanged':protected,
        'allowed_existing_source_changes':sorted(ALLOWED),'historical_artifacts_preserved':True,'ecological_ABC_exact_cases':3,'full500_source_Q_LRU_pending_exact_cases':9,
        'stationary_v073_exact_reruns':2,'scheduled_full_state_byte_exact_reruns':2,'same_initial_ecology_and_main_rng_all_pilots':True,
        'full_tests_passed':int(match[1]),'full_tests_seconds':float(match[2]),'test_log_sha256':hash_file(root/'final_tests.log'),
        'gate':review,'models':frozen['models'],'runtime':times,'peak_audit_memory_bytes':peak_bytes(),
        'data_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),'analysis_bytes_before_receipt':sum(p.stat().st_size for p in ANALYSIS.rglob('*') if p.is_file()),
        'raw_data_note':'No new B/C500 simulations. Reused v073 ordered outcomes plus v072 ecological references; full resources/lineage fields not invented.',
        'source_sha256':{str(p):hash_file(p) for p in sorted(Path('experiments').glob('nonstationary*')) if p.is_file()},
        'core_unchanged_sha256':{p:hash_file(p) for p in ('config.py','agents/learning.py','agents/social_learning.py','agents/genome.py','systems/reproduction.py','world/world.py')}}
    save_json(output,receipt);print(f'Verified {protected} protected files, all compatibility/replay/determinism evidence, gate and {match[1]} tests.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=ANALYSIS/'verification.json')
    try:audit(**vars(p.parse_args()))
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
