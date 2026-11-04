"""Audit completed v073 files and historical compatibility without new simulations."""
import argparse,json,math
from pathlib import Path
from statistics import mean,median

from experiments.controls_metrics import load,read_csv
from experiments.information_controls import verify_frozen
from simulation.vocal_learning_observer import hash_file


def verify(root=Path('data/social_learning_v073'),output=Path('analysis/social_learning_v073/verification.json')):
    root=Path(root);output=Path(output)
    if output.exists():raise ValueError('Refusing existing verification receipt')
    baseline=json.loads(Path('data/v073_before.json').read_text())['files']
    allowed={'.gitignore','README.md','main.py','simulation/simulation.py'}
    unchanged=[]
    for p,sha in baseline.items():
        if p in allowed:continue
        if hash_file(p)!=sha:raise ValueError('Historical/source file changed unexpectedly: '+p)
        unchanged.append(p)
    verify_frozen(root);references=[];initials={}
    for c in 'ABC':
        for seed in (0,1,2):
            path=root/'references'/f'{c}_seed_{seed}_r0.csv.gz.metadata.json';m,files=load(path)
            old=Path('data/social_learning_v072/primary')/f'{c}_seed_{seed}.csv.gz.metadata.json';om,_=load(old)
            if any(m[k]!=om[k] for k in ('fingerprints','statistics','action_sha256')):raise ValueError('Full v072 trajectory mismatch')
            pilot=Path('data/social_learning_v072/pilot')/f'{c}_seed_{seed}.csv.gz.metadata.json';pm,_=load(pilot)
            if m['fingerprints']['action_checkpoints']['100']!=pm['action_sha256']:raise ValueError('Prefix action mismatch')
            cfg=lambda x:{k:v for k,v in x['config']['simulation'].items() if k!='max_ticks'}
            if cfg(m)!=cfg(pm):raise ValueError('Prefix configuration mismatch beyond inert run horizon')
            row=read_csv(files['ticks']).set_index('tick').loc[100]
            for field,value in pm['statistics'].items():
                if field=='tick':continue
                actual=row[field]
                if not ((value is None and actual!=actual) or actual==value):raise ValueError('Prefix statistic mismatch: '+field)
            references.append({'condition':c,'seed':seed,'primary_exact':True,'pilot_actions_statistics_exact':True,'sha256':hash_file(path)})
            if c=='A':initials[seed]=pm['fingerprints']['initial_ecology']
    repeated=[]
    for c in 'DE':
        p=root/'pilot'/f'{c}_seed_0_r0.csv.gz.metadata.json';m,files=load(p)
        r,rf=load(root/'determinism'/f'{c}_seed_0_r0.csv.gz.metadata.json')
        if m['fingerprints']!=r['fingerprints'] or m['statistics']!=r['statistics'] or any(hash_file(files[k])!=hash_file(rf[k]) for k in files):raise ValueError('Deterministic rerun mismatch')
        repeated.append({'condition':c,'seed':0,'all_streams_byte_identical':True,'full_state_rng_exact':True})
    cases=[]
    for p in sorted(root.glob('*/*.metadata.json')):
        m,files=load(p)
        if m['config']['simulation']['learning_memory_capacity']!=256:raise ValueError('Experimental capacity changed')
        if m['condition'] in 'DE' and m['tick']==100 and m['fingerprints']['initial_ecology']!=initials[m['seed']]:raise ValueError('Initial ecology/main RNG differs')
        runtime=json.loads(Path(str(p).replace('.metadata.json','.runtime.json')).read_text())
        cases.append({'phase':p.parent.name,'condition':m['condition'],'seed':m['seed'],'realization':m['realization'],**runtime})
    stages={}
    for phase in sorted({r['phase'] for r in cases}):
        group=[r for r in cases if r['phase']==phase];times=[r['wall_seconds'] for r in group]
        stages[phase]={'cases':len(group),'case_wall_seconds_sum':sum(times),'case_wall_seconds_median':median(times),'case_wall_seconds_min':min(times),'case_wall_seconds_max':max(times),'stream_bytes':sum(r['bytes'] for r in group)}
    receipt={'version':'0.7.3','passed':True,'baseline_files_checked':len(baseline),'allowed_modified_existing_files':sorted(allowed),
        'unchanged_existing_files':len(unchanged),'historical_v072_preserved':True,'references':references,'deterministic_reruns':repeated,
        'stages':stages,'cases':cases,'primary_gate':json.loads((root/'pilot_review.json').read_text()),'primary_directory_exists':(root/'primary').exists(),
        'models_frozen_verified':True,'data_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        'analysis_bytes_before_receipt':sum(p.stat().st_size for p in output.parent.rglob('*') if p.is_file()),
        'runtime_definition':'Sum of per-case measured wall time; reference cases used concurrent independent CLI processes, so this is not elapsed study wall time.',
        'core_unchanged':{p:hash_file(p) for p in ('config.py','agents/learning.py','agents/social_learning.py','systems/reproduction.py','agents/genome.py','world/world.py')},
        'source_sha256':{str(p):hash_file(p) for p in sorted(Path('experiments').glob('*controls*.py'))}}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(receipt,indent=2)+'\n')
    print('Verified: historical files, all 9 v072 reference trajectories, all pilot prefixes, D/E full-state byte-identical reruns and frozen models.',flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('data/social_learning_v073'));p.add_argument('--output',type=Path,default=Path('analysis/social_learning_v073/verification.json'))
    try:verify(**vars(p.parse_args()))
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
