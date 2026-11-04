"""Bounded v0.7.3 runs. No ecology, reward or learning parameters are tuned here."""
from contextlib import ExitStack
import argparse,hashlib,json
from pathlib import Path
from time import perf_counter

from config import Config,SocialConfig,SocialPolicyConfig,SocialLearningConfig
from agents.nuisance import NuisanceConfig,NuisanceModel
from simulation.simulation import Simulation
from simulation.controls_observer import ControlsObserver,ControlsWriter,write_metadata
from experiments.social_learning_experiments import fingerprints
from experiments.social_verification import state,digest
from experiments.auditory_policy import frozen
from simulation.vocal_learning_observer import hash_file

ROOT=Path('data/social_learning_v073')


def model_file(path):
    m=json.loads(Path(path).read_text())
    try:return NuisanceModel(tuple(m['probabilities']),m['hold_probability'])
    except (KeyError,TypeError) as error:raise ValueError('Model requires probabilities and hold_probability') from error


def verify_frozen(root):
    frozen=json.loads((Path(root)/'models/frozen.json').read_text())
    for c,sha in frozen['models'].items():
        if hash_file(Path(root)/'models'/f'{c}.json')!=sha:raise ValueError('Frozen calibration model changed')
    for path,sha in frozen['sources'].items():
        if hash_file(path)!=sha:raise ValueError('Calibration source changed')


def initial_fingerprint(sim):
    value=state(sim)
    value=(*value[:4],tuple((body,tuple((k,v) for k,v in brain if k not in ('social_mode','nuisance_channel'))) for body,brain in value[4]),*value[5:])
    return hashlib.sha256(json.dumps((value,()),separators=(',',':')).encode()).hexdigest()


def fingerprint(sim):
    if sim.nuisance is None:return fingerprints(sim)
    return hashlib.sha256(json.dumps((fingerprints(sim),frozen(sim._nuisance_source.rng),sim._nuisance_source.created),separators=(',',':')).encode()).hexdigest()


def run_one(folder,condition,seed,ticks,*,model_path=None,nuisance_seed=0,realization=0,light=False,verify_old=False):
    if condition not in 'ABCDE' or len(condition)!=1:raise ValueError('Expected A..E')
    if type(ticks) is not int or not 1<=ticks<=500:raise ValueError('Bounded runs require 1..500 ticks')
    if type(seed) is not int or type(realization) is not int or realization<0:raise ValueError('Integer ecological seed and nonnegative realization required')
    folder=Path(folder)
    for old in ('data/social_learning_v072','analysis/social_learning_v072','data/social_memory','data/social_memory_v071'):
        if folder.resolve().is_relative_to(Path(old).resolve()):raise ValueError('Historical output directory protected')
    path=folder/f'{condition}_seed_{seed}_r{realization}.csv.gz'
    mode={'A':'none','B':'perception','C':'memory','D':'none','E':'none'}[condition]
    nuisance=None;calibration=None
    if condition in 'DE':
        if model_path is None:raise ValueError('D/E require frozen model')
        model=model_file(model_path)
        if model.categories!=({'D':11,'E':37}[condition]):raise ValueError('Wrong model size')
        nuisance=NuisanceConfig(model,nuisance_seed)
        calibration={'file':str(model_path),'sha256':hash_file(model_path)}
    elif model_path is not None:raise ValueError('A/B/C cannot use nuisance model')
    cfg=Config.evolution(brain='learning',random_seed=seed,max_ticks=ticks)
    start=perf_counter();proof={};ap={0:hashlib.sha256().hexdigest()}
    with ExitStack() as stack:
        writer=ControlsWriter.open(stack,path,light);observer=ControlsObserver(writer,condition,light)
        sim=Simulation(cfg,social_config=SocialConfig(True,32,20),social_policy=SocialPolicyConfig(),
                       social_learning=SocialLearningConfig(mode),social_learning_observer=observer,nuisance=nuisance)
        initial=initial_fingerprint(sim);proof[0]=fingerprint(sim)
        for _ in range(ticks):
            if not sim.population:break
            sim.step()
            if sim.tick%100==0:
                proof[sim.tick]=fingerprint(sim);ap[sim.tick]=observer.action_digest.hexdigest()
                print(f'{condition} eco={seed} nuisance={nuisance_seed if nuisance else "NA"} r={realization} tick={sim.tick} pop={sim.population}',flush=True)
        proof[sim.tick]=fingerprint(sim);ecological=digest(sim)
        observer.finish(sim)
    meta=write_metadata(path,sim,observer,writer,realization=realization,calibration=calibration,
        fingerprints={'initial_ecology':initial,'checkpoints':proof,'action_checkpoints':ap,'ecological_final':ecological})
    receipt={'wall_seconds':perf_counter()-start,'bytes':sum((path.parent/f['file']).stat().st_size for f in meta['files'].values()),'legacy_verified':False}
    if verify_old:
        old=Path('data/social_learning_v072')/('pilot' if ticks==100 else 'primary')/f'{condition}_seed_{seed}.csv.gz.metadata.json'
        m=json.loads(old.read_text())
        if json.loads(json.dumps(meta['fingerprints']))!=m['fingerprints'] or meta['statistics']!=m['statistics'] or meta['action_sha256']!=m['action_sha256']:
            raise RuntimeError('Instrumented trajectory differs from archived v072: '+str(old))
        receipt.update(legacy_verified=True,legacy_metadata=str(old),legacy_sha256=hash_file(old))
    with Path(str(path)+'.runtime.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    return Path(str(path)+'.metadata.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('calibration','references','pilot','pilot_robustness','primary','robustness'),required=True)
    p.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args();folder=args.root/args.phase
    if folder.exists():p.error('Refusing existing phase directory')
    if args.phase=='calibration':
        for seed in (3,4):
            for c in 'BC':run_one(folder,c,seed,500,light=True)
    elif args.phase=='references':
        if not (args.root/'models/frozen.json').exists():p.error('Freeze calibration first')
        for seed in (0,1,2):
            for c in 'ABC':run_one(folder,c,seed,500,light=True,verify_old=True)
    else:
        if not (args.root/'models/frozen.json').exists():p.error('Freeze calibration first')
        verify_frozen(args.root)
        if args.phase=='pilot_robustness':
            review=args.root/'pilot_review.json'
            if not review.exists() or not json.loads(review.read_text()).get('allow_pilot_robustness'):
                p.error('Pilot robustness requires explicit reviewed exploratory permission; not a primary bypass')
        elif args.phase!='pilot':
            review=args.root/'pilot_review.json'
            if not review.exists() or json.loads(review.read_text()).get('decision')!='primary_allowed':
                p.error('Primary blocked: pilot requires an explicit matching review receipt')
        r=int(args.phase in ('robustness','pilot_robustness'));ticks=100 if args.phase in ('pilot','pilot_robustness') else 500
        for seed in (0,1,2):
            for c in 'DE':run_one(folder,c,seed,ticks,model_path=args.root/'models'/f'{c}.json',nuisance_seed=10000+100*seed+r,realization=r)

if __name__=='__main__':main()
