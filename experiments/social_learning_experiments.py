"""Explicit bounded online experiments; social inputs can change ecology."""
from contextlib import ExitStack
from dataclasses import asdict
import argparse,hashlib,json,resource,sys
from pathlib import Path
from time import perf_counter

from config import Config,SocialConfig,SocialPolicyConfig,SocialLearningConfig
from simulation.simulation import Simulation
from simulation.social_learning_observer import SocialLearningWriter,SocialLearningObserver,write_metadata
from simulation.vocal_learning_observer import hash_file
from experiments.social_verification import state,digest
from experiments.auditory_policy import frozen


def fingerprints(sim,initial=False):
    value=state(sim)
    if initial:
        value=(*value[:4],tuple((body,tuple((k,v) for k,v in brain if k!='social_mode')) for body,brain in value[4]),*value[5:])
    private=() if initial else tuple((h.id,frozen(vars(h.social_memory))) for h in sim.agents if hasattr(h,'social_memory'))
    return hashlib.sha256(json.dumps((value,private),separators=(',',':')).encode()).hexdigest()


def run_one(folder,condition,seed,ticks,capacity=256):
    if condition not in ('A','B','C','C_decay','B_large','C_large'):raise ValueError('Unsupported condition')
    if type(seed) is not int or type(ticks) is not int or not 1<=ticks<=500:raise ValueError('Integer seed and 1..500 ticks required')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{condition}_seed_{seed}.csv.gz'
    mode='none' if condition=='A' else 'perception' if condition.startswith('B') else 'memory'
    cfg=Config.evolution(brain='learning',random_seed=seed,max_ticks=ticks,learning_memory_capacity=capacity)
    social=SocialConfig(True,32,20)
    policy=SocialPolicyConfig(5,'contact_decay',100,candidate_capacity=32) if condition=='C_decay' else SocialPolicyConfig()
    start=perf_counter();loop=0.;proof={};action_proof={0:hashlib.sha256().hexdigest()}
    with ExitStack() as stack:
        writer=SocialLearningWriter.open(stack,path);observer=SocialLearningObserver(writer)
        sim=Simulation(cfg,social_config=social,social_policy=policy,social_learning=SocialLearningConfig(mode),social_learning_observer=observer)
        initial=fingerprints(sim,True);proof[0]=fingerprints(sim)
        for _ in range(ticks):
            if not sim.population:break
            before=perf_counter();sim.step();loop+=perf_counter()-before
            if sim.tick%100==0:
                proof[sim.tick]=fingerprints(sim)
                action_proof[sim.tick]=observer.action_digest.hexdigest()
                print(f'{condition} seed={seed} tick={sim.tick} population={sim.population} evictions={observer.total["evictions"]}',flush=True)
        proof[sim.tick]=fingerprints(sim)
        final_ecological=digest(sim)
        observer.finish(sim)
    meta=write_metadata(path,sim,observer,writer,condition,{'initial_ecology':initial,'checkpoints':proof,'action_checkpoints':action_proof,'ecological_final':final_ecological})
    timing={'condition':condition,'seed':seed,'ticks':sim.tick,'simulation_seconds':loop,'wall_seconds':perf_counter()-start,
            'bytes':sum(p.stat().st_size for p in folder.glob(f'{condition}_seed_{seed}.*')),'records':sum(writer.counts.values()),
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
    return Path(str(path)+'.metadata.json'),meta,timing


def run(phase='pilot',seeds=(0,1,2),data_dir=Path('data/social_learning_v072'),output_root=Path('analysis/social_learning_v072'),ticks=None):
    if phase not in ('pilot','primary','memory_policy','q_capacity'):raise ValueError('Unknown phase')
    ticks=(100 if phase=='pilot' else 500) if ticks is None else ticks
    if type(ticks) is not int or not 1<=ticks<=500 or phase=='pilot' and ticks>100:raise ValueError('Pilot <=100; other phases <=500 ticks')
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds))!=len(seeds):raise ValueError('Distinct integer seeds required')
    data_dir=Path(data_dir);output_root=Path(output_root)
    for root in (data_dir,output_root):
        for old in ('data/social_memory','data/social_memory_v071','analysis/social_memory','analysis/social_memory_v071'):
            if root.resolve().is_relative_to(Path(old).resolve()):raise ValueError('Historical directory protected')
    destination=data_dir/phase;output=output_root/phase
    if destination.exists() or output.exists():raise ValueError('Refusing existing phase directories')
    if phase!='pilot':
        needed='pilot' if phase=='primary' else 'primary'
        receipt=data_dir/(needed+'_runtime.json')
        if not receipt.exists() or not json.loads(receipt.read_text()).get('verified'):raise ValueError('Complete/review '+needed+' first')
    conditions=('A','B','C') if phase in ('pilot','primary') else ('C_decay',) if phase=='memory_policy' else ('B_large','C_large')
    sources=[]
    if phase in ('memory_policy','q_capacity'):
        for seed in sorted(seeds):
            for c in (('C',) if phase=='memory_policy' else ('B','C')):
                p=data_dir/'primary'/f'{c}_seed_{seed}.csv.gz.metadata.json'
                if not p.exists():raise ValueError('Matching primary source missing')
                m=json.loads(p.read_text())
                if m['config']['simulation']['max_ticks']!=ticks:raise ValueError('Secondary horizon must match primary')
                sources.append(p)
    destination.mkdir(parents=True);start=perf_counter();rows=[]
    design={'version':'0.7.2','phase':phase,'ticks':ticks,'seeds':sorted(seeds),'conditions':conditions,
            'same_ecology_not_same_trajectory':True,'physical_q_capacity':1024 if phase=='q_capacity' else 256,
            'vocal_capacity':256,'source_sha256':{str(p):hash_file(p) for p in sorted(Path('experiments').glob('social_learning*.py'))}}
    (destination/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    initial={}
    for seed in sorted(seeds):
        for condition in conditions:
            p,m,t=run_one(destination,condition,seed,ticks,1024 if phase=='q_capacity' else 256)
            sources.append(p);rows.append(t)
            old=initial.setdefault(seed,m['fingerprints']['initial_ecology'])
            if old!=m['fingerprints']['initial_ecology']:raise RuntimeError('Initial ecology/RNG differs across conditions')
    from experiments.analyze_social_learning import analyze
    analysis,analysis_time=analyze(sources,output)
    result={'verified':True,'design':design,'runs':rows,'analysis_seconds':analysis_time,'wall_seconds':perf_counter()-start,
            'analysis_files':len(analysis['outputs'])+1,'analysis_bytes':sum(p.stat().st_size for p in output.iterdir()),
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
    data_dir.mkdir(parents=True,exist_ok=True)
    with (data_dir/(phase+'_runtime.json')).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(f'{phase} complete: {result["wall_seconds"]:.3f}s, {result["analysis_bytes"]} analysis bytes',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('pilot','primary','memory_policy','q_capacity'),default='pilot')
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--ticks',type=int)
    p.add_argument('--data-dir',type=Path,default=Path('data/social_learning_v072'));p.add_argument('--output-root',type=Path,default=Path('analysis/social_learning_v072'))
    try:run(**vars(p.parse_args()))
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
