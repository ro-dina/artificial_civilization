"""Bounded v0.7.1 staged experiment runner. Policies replay one ecology per seed.

Default runs only the 100-tick pilot. Primary 500-tick work is a separate explicit
phase after the pilot receipt exists. No automatic 1000-tick/capacity/tau grid.
"""
import argparse,json,sys,resource
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from experiments.social_history import generate_history,convert_legacy,load_metadata
from experiments.analyze_social_memory_policies import analyze,treatments


def prepare_sources(root,seeds,ticks):
    history=root/'histories';history.mkdir(parents=True,exist_ok=True);paths=[];timings=[]
    for seed in sorted(seeds):
        path=history/f'seed_{seed}_{ticks}.csv.gz'
        if path.exists():
            m=load_metadata(path)
            if m['seed']!=seed or m['requested_horizon']!=ticks:raise ValueError('Cached history mismatch')
            timing={'seed':seed,'cached':True}
        else:
            old=Path(f'data/social_memory/seed_{seed}_social.csv.gz') if ticks<=100 else Path(f'data/social_memory/seed_{seed}_1000ticks_social.csv.gz')
            if old.exists():m,timing=convert_legacy(old,path,ticks);timing.update(seed=seed,source='existing_legacy')
            else:m,timing=generate_history(path,seed,ticks);timing.update(seed=seed,source='new_baseline')
        paths.append(path);timings.append(timing)
    return paths,timings


def run(phase='pilot',seeds=(0,1,2),data_dir=Path('data/social_memory_v071'),output_root=Path('analysis/social_memory_v071'),ticks=None,allow_long=False):
    if phase not in ('pilot','primary','capacity','decay'):raise ValueError('Unknown experiment phase')
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds))!=len(seeds):raise ValueError('Distinct integer seeds required')
    ticks=(500 if phase=='primary' else 100) if ticks is None else ticks
    if type(ticks) is not int or not 1<=ticks<=1000 or ticks>500 and not allow_long:raise ValueError('Use 1..500 ticks; explicit --allow-long permits <=1000')
    if phase=='pilot' and ticks>100:raise ValueError('Pilot limited to 100 ticks')
    data_dir,output_root=Path(data_dir),Path(output_root)
    if data_dir.resolve().is_relative_to(Path('data/social_memory').resolve()) or output_root.resolve().is_relative_to(Path('analysis/social_memory').resolve()):
        raise ValueError('Historical directories protected')
    receipt=data_dir/(phase+f'_{ticks}_runtime.json');destination=output_root/(phase if ticks in (100,500) else phase+f'_{ticks}')
    if receipt.exists() or destination.exists() and any(destination.iterdir()):raise ValueError('Refusing existing experiment outputs')
    if phase=='primary':
        pilot=data_dir/'pilot_100_runtime.json'
        if not pilot.exists() or not json.loads(pilot.read_text()).get('completed'):raise ValueError('Run/review the 100-tick pilot first')
    if phase in ('pilot','primary'):policies=treatments(candidate_capacity=32)
    elif phase=='capacity':policies=treatments(('A','D'),(8,16,32,64,128),(100.,),32)
    else:policies=treatments(('A','D'),(32,),(20.,50.,100.,200.),32)
    data_dir.mkdir(parents=True,exist_ok=True)
    design={'version':'0.7.1','phase':phase,'seeds':sorted(seeds),'horizon':ticks,
            'policies':[{'key':t.key,'social':asdict(t.social),'policy':asdict(t.policy)} for t in policies],
            'scope':'Same ecological/visual histories, no rewards or new behavior; missing/candidate/commitment states separated.'}
    design_path=data_dir/(phase+f'_{ticks}_design.json')
    if design_path.exists():raise ValueError('Experiment design already exists')
    design_path.write_text(json.dumps(design,indent=2)+'\n')
    start=perf_counter();histories,source_times=prepare_sources(data_dir,seeds,ticks)
    metadata,timing=analyze(histories,destination,horizon=ticks,policies=policies)
    result={'completed':True,'design':design,'sources':source_times,'replay':timing,'wall_seconds':perf_counter()-start,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'output_files':len(metadata['outputs'])+1,'output_bytes':sum(p.stat().st_size for p in destination.iterdir()),
            'history_bytes':sum(p.stat().st_size for p in histories)}
    receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'{phase} completed: seconds={result["wall_seconds"]:.3f}, output bytes={result["output_bytes"]}',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('pilot','primary','capacity','decay'),default='pilot')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--data-dir',type=Path,default=Path('data/social_memory_v071'))
    p.add_argument('--output-root',type=Path,default=Path('analysis/social_memory_v071'));p.add_argument('--ticks',type=int);p.add_argument('--allow-long',action='store_true')
    try:run(**vars(p.parse_args()))
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
