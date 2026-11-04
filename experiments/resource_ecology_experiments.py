"""Bounded preregistered 2x2 resource ecology experiments; no nuisance conditions."""
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from random import Random
from time import perf_counter
import argparse,hashlib,json,math,resource,sys

from config import Config,ResourceEcologyConfig,SocialConfig,SocialPolicyConfig,SocialLearningConfig
from world.world import World
from world.resource_ecology import PatchGeometry,ResourceEcology
from simulation.simulation import Simulation
from simulation.resource_observer import ResourceObserver,ResourceWriter,SCHEMAS,SCHEMA_VERSION,paths
from simulation.controls_observer import physical_fingerprint
from simulation.vocal_learning_observer import hash_file
from experiments.social_learning_experiments import fingerprints
from experiments.social_verification import state,digest
from experiments.auditory_policy import frozen

ROOT=Path('data/resource_ecology_v080');ANALYSIS=Path('analysis/resource_ecology_v080')


def save_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(json.dumps(payload,indent=2,allow_nan=False)+'\n')


def peak():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)


def fingerprint(sim):
    legacy=fingerprints(sim)
    if sim.resource_system is None:return legacy
    return hashlib.sha256(json.dumps((legacy,frozen(vars(sim.resource_system))),separators=(',',':')).encode()).hexdigest()


def initial_agents(sim):
    value=state(sim)
    bodies=tuple((body,tuple((k,v) for k,v in brain if k!='social_mode')) for body,brain in value[4])
    return hashlib.sha256(json.dumps((bodies,value[6:]),separators=(',',':')).encode()).hexdigest()


def ecology_config(ecology,scarcity=.5,localization=.75):
    if ecology not in ('E0','E1','E2','E3'):raise ValueError('Expected E0..E3')
    return ResourceEcologyConfig(scarcity if ecology in ('E1','E3') else 1.,localization if ecology in ('E2','E3') else 0.)


def verify_frozen(root):
    root=Path(root);m=json.loads((root/'frozen.json').read_text())
    if hash_file(root/'preregistration.json')!=m['preregistration_sha256']:raise ValueError('Frozen design changed')
    for p,sha in m['calibration_inputs'].items():
        if hash_file(p)!=sha:raise ValueError('Calibration evidence changed')
    return m


def case_path(root,phase,ecology,mode,seed):return Path(root)/phase/f'{ecology}_{mode}_seed_{seed}.csv.gz'


def run_one(root,phase,ecology,mode,seed,ticks,eco=None,verify_legacy=True):
    root=Path(root)
    if mode not in ('A','B','C') or ecology not in ('E0','E1','E2','E3') or type(ticks) is not int or not 1<=ticks<=500:raise ValueError('Bounded factorial case required')
    for old in ('data/social_memory','data/social_memory_v071','data/social_learning_v072','data/social_learning_v073','data/social_learning_v074'):
        if root.resolve().is_relative_to(Path(old).resolve()):raise ValueError('Historical outputs protected')
    if phase=='primary' and not pilot_passes(root):raise ValueError('Primary requires all pilot gates')
    if phase in ('pilot','primary','determinism'):
        frozen=verify_frozen(root)
        if eco is not None:raise ValueError('Evaluation parameters must come from frozen calibration')
        eco=ecology_config(ecology,frozen['scarcity'],frozen['localization'])
    eco=eco if eco is not None else ecology_config(ecology)
    cfg=Config.evolution(brain='learning',random_seed=seed,max_ticks=ticks)
    path=case_path(root,phase,ecology,mode,seed);start=perf_counter()
    geometry=PatchGeometry.generate(cfg,eco)
    with ExitStack() as stack:
        writer=ResourceWriter.open(stack,path);observer=ResourceObserver(writer,mode,ecology,geometry)
        sim=Simulation(cfg,social_config=SocialConfig(True,32,20),social_policy=SocialPolicyConfig(),
            social_learning=SocialLearningConfig({'A':'none','B':'perception','C':'memory'}[mode]),social_learning_observer=observer,
            resource_ecology=eco,resource_observer=observer)
        initial=fingerprints(sim,True);agent_initial=initial_agents(sim);proof={0:fingerprint(sim)};actions={0:hashlib.sha256().hexdigest()}
        for _ in range(ticks):
            if not sim.population:break
            sim.step()
            if sim.tick%100==0:
                proof[sim.tick]=fingerprint(sim);actions[sim.tick]=observer.action_digest.hexdigest()
                print(f'{phase} {ecology}/{mode} seed={seed} tick={sim.tick} pop={sim.population} food={sim.statistics.food_consumed} water={sim.statistics.water_consumed}',flush=True)
        proof[sim.tick]=fingerprint(sim);legacy_final=digest(sim)
        observer.finish(sim)
    fps={'initial_ecology':initial,'checkpoints':proof,'action_checkpoints':actions,'ecological_final':legacy_final}
    legacy=None
    if ecology=='E0' and verify_legacy and seed in (0,1,2) and ticks in (100,500):
        old=Path('data/social_learning_v072')/('pilot' if ticks==100 else 'primary')/f'{mode}_seed_{seed}.csv.gz.metadata.json'
        previous=json.loads(old.read_text())
        if json.loads(json.dumps(fps))!=previous['fingerprints'] or asdict(sim.statistics)!=previous['statistics'] or observer.action_digest.hexdigest()!=previous['action_sha256']:
            raise RuntimeError('E0 full historical state/RNG regression: '+str(old))
        legacy={'file':str(old),'sha256':hash_file(old),'exact':True}
    if writer.counts['acquisitions']!=observer.total['decisions'] or observer.total['updates']!=sim.statistics.total_learning_updates:raise RuntimeError('Decision/Q accounting mismatch')
    if observer.resource_totals['food']!=sim.statistics.food_consumed or observer.resource_totals['water']!=sim.statistics.water_consumed:raise RuntimeError('Consumption mismatch')
    if writer.counts['individuals']!=cfg.initial_population+sim.statistics.births:raise RuntimeError('Life accounting mismatch')
    files=paths(path);sources=['config.py','simulation/simulation.py','simulation/resource_observer.py','world/resource_ecology.py','world/world.py','agents/learning.py','agents/social_learning.py',__file__]
    meta={'version':'0.8.0','schema_version':SCHEMA_VERSION,'format':'resource_ecology','completed':True,'ecology':ecology,'mode':mode,'seed':seed,'phase':phase,'tick':sim.tick,
        'config':{'simulation':asdict(cfg),'resource_ecology':asdict(eco),'social':asdict(sim.social_config),'social_policy':asdict(sim.social_policy),'social_learning':asdict(sim.social_learning)},
        'geometry':asdict(geometry),'geometry_used_for_resource_generation':bool(eco.localization_strength),'food_water_geometry_shared':True,
        'statistics':asdict(sim.statistics),'totals':dict(observer.total),'resource_totals':dict(observer.resource_totals),
        'fingerprints':fps,'initial_agents_rng_sha256':agent_initial,'physical_q_sha256':physical_fingerprint(sim),'living_ids':[h.id for h in sim.agents],
        'action_sha256':observer.action_digest.hexdigest(),'legacy_comparison':legacy,'schemas':SCHEMAS,
        'files':{k:{'file':p.name,'sha256':hash_file(p),'bytes':p.stat().st_size,'records':writer.counts[k]} for k,p in files.items()},
        'source_sha256':{p:hash_file(p) for p in sources},'runtime':{'wall_seconds':perf_counter()-start,'peak_memory_bytes':peak()},
        'definitions':{'supply':'Original nonfull-tile Bernoulli proposals, p scaled only. Localized proposals reallocated, all offers realized. Equal budget conditional on same eligibility; spatial saturation changes trial counts. Potential expected vs eligible vs actual reported separately.',
            'acquisitions':'Actual _execute food/water units and action success after existing physiology/reward. Pre-action resource values current sequential world; privileged nearest distances frozen just after tick-start regeneration. No extra Brain features.',
            'individuals':'One scalar life row per founder/child, emitted at actual death or administrative censor. Duration=age, founder fixed-horizon RMST identifiable; newborn followup separate. No dead bodies retained.',
            'geometry':'E0/E1 use only a diagnostic reference mask; E2/E3 actually generate within same fixed shared food/water patches with positive background.'}}
    save_json(Path(str(path)+'.metadata.json'),meta)
    return Path(str(path)+'.metadata.json')


def load(path):
    import pandas as pd
    path=Path(path);m=json.loads(path.read_text())
    if m.get('version')!='0.8.0' or not m.get('completed'):raise ValueError('Completed v080 recording required')
    for r in m['files'].values():
        if Path(r['file']).name!=r['file'] or hash_file(path.parent/r['file'])!=r['sha256']:raise ValueError('Resource data checksum mismatch')
    return m,{k:path.parent/r['file'] for k,r in m['files'].items()}


def supply_audit(root,scarcity,localization,ticks=200):
    import pandas as pd
    rows=[]
    for seed in (3,4):
        for regime in ('absent','fixed_withdrawal','reset_empty'):
            for label in ('E0','E1','E2','E3'):
                cfg=Config.evolution(random_seed=seed);eco=ecology_config(label,scarcity,localization)
                world=World.generate(cfg,Random(f'{seed}:world'));system=ResourceEcology(cfg,eco) if not eco.legacy else None
                if system:system.initialize(world)
                rngs=[Random(f'{seed}:regeneration:{r}') for r in ('food','water')]
                for tick in range(1,ticks+1):
                    tiles=[t for row in world.tiles for t in row];withdrawn={r:0 for r in ('food','water')}
                    for r in ('food','water'):
                        if regime=='reset_empty':
                            withdrawn[r]=sum(getattr(t,r) for t in tiles)
                            for t in tiles:setattr(t,r,0)
                        elif regime=='fixed_withdrawal' and tick%10==0:
                            for t in tiles:
                                if getattr(t,r)>0:setattr(t,r,getattr(t,r)-1);withdrawn[r]+=1
                    eligible={r:sum(getattr(t,r)<getattr(cfg,r+'_capacity') for t in tiles) for r in ('food','water')}
                    gains=system.regenerate(world,*rngs) if system else world.regenerate(cfg,*rngs)
                    for r,n,rng in zip(('food','water'),gains,rngs):
                        p=getattr(cfg,r+'_regeneration_probability')*eco.resource_supply_multiplier
                        info=system.last_supply if system else {}
                        rows.append({'seed':seed,'regime':regime,'ecology':label,'tick':tick,'resource':r,'eligible':eligible[r],
                            'potential_expected':cfg.world_width*cfg.world_height*p,'eligible_expected':eligible[r]*p,'attempted':info.get(r+'_attempted',n),'realized':n,
                            'rejected':info.get(r+'_rejected',0),'withdrawn':withdrawn[r],'stock':sum(getattr(t,r) for t in tiles),
                            'budget_rng_sha256':hashlib.sha256(repr(rng.getstate()).encode()).hexdigest()})
    f=pd.DataFrame(rows)
    for seed in (3,4):
        for a,b in [('E0','E2'),('E1','E3')]:
            x=f[f.seed.eq(seed)&f.regime.eq('reset_empty')&f.ecology.eq(a)].sort_values(['tick','resource']).reset_index(drop=True)
            y=f[f.seed.eq(seed)&f.regime.eq('reset_empty')&f.ecology.eq(b)].sort_values(['tick','resource']).reset_index(drop=True)
            if not x[['attempted','realized','budget_rng_sha256']].equals(y[['attempted','realized','budget_rng_sha256']]):raise RuntimeError('Conditional supply coupling failed')
    if f.rejected.any():raise RuntimeError('Supply rejection should be impossible with positive background')
    out=Path(root)/'supply_audit.csv.gz';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise ValueError('Refusing supply audit overwrite')
    f.to_csv(out,index=False,compression={'method':'gzip','mtime':0},float_format='%.12g')
    return out


def calibrate(root=ROOT):
    import pandas as pd
    root=Path(root)
    if (root/'frozen.json').exists():raise ValueError('Refusing calibration overwrite')
    design=json.loads((root/'preregistration.json').read_text());inputs={};records=[]
    def cases(label,eco,tag):
        result=[]
        for seed in (3,4):
            p=run_one(root,'calibration/'+tag,label,'A',seed,200,eco,False);inputs[str(p)]=hash_file(p);m,files=load(p)
            t=pd.read_csv(files['resource_ticks']);s=m['statistics'];sp=pd.read_csv(files['spatial']);first=sp[sp.tick.eq(0)]
            from simulation.resource_observer import concentration
            result.append({'seed':seed,'population100':int(t[t.tick.eq(100)].population.iloc[0]),'population200':s['population'],'decisions':m['totals']['decisions'],
                'food_consumed':s['food_consumed'],'water_consumed':s['water_consumed'],'food_regenerated':s['food_regenerated'],'water_regenerated':s['water_regenerated'],
                'food_mean_stock':float(t[t.tick.ge(100)].food_stock.mean()),'water_mean_stock':float(t[t.tick.ge(100)].water_stock.mean()),
                'food_effective_initial':concentration(first.food.tolist(),first.patch.tolist())['effective_cells'],
                'water_effective_initial':concentration(first.water.tolist(),first.patch.tolist())['effective_cells']})
        return result
    def viable(rs):return all(r['population100']>=20 and r['population200']>=10 and r['decisions']>=5000 and r['food_consumed']>=20 and r['water_consumed']>=20 for r in rs)
    baseline=cases('E0',ecology_config('E0'),'baseline')
    scarcity=None;localization=None
    for m in design['calibration']['scarcity_candidates']:
        rs=cases('E1',ResourceEcologyConfig(m,0),f'scarcity_{m}')
        supply=all(sum(r[resource+'_regenerated'] for r in rs)<=.8*sum(r[resource+'_regenerated'] for r in baseline) for resource in ('food','water'))
        availability=any(abs(sum(r[resource+'_mean_stock'] for r in rs)/sum(r[resource+'_mean_stock'] for r in baseline)-1)>=.1 for resource in ('food','water'))
        accepted=viable(rs) and supply and availability;records.append({'factor':'scarcity','candidate':m,'accepted':accepted,'viable':viable(rs),'supply_manipulated':supply,'availability_changed':availability,'runs':rs})
        if accepted:scarcity=m;break
    for strength in design['calibration']['localization_candidates']:
        rs=cases('E2',ResourceEcologyConfig(1,strength),f'localization_{strength}')
        concentrated=all(sum(r[resource+'_effective_initial'] for r in rs)<=.8*sum(r[resource+'_effective_initial'] for r in baseline) for resource in ('food','water'))
        accepted=viable(rs) and concentrated;records.append({'factor':'localization','candidate':strength,'accepted':accepted,'viable':viable(rs),'spatial_manipulated':concentrated,'runs':rs})
        if accepted:localization=strength;break
    if scarcity is None or localization is None:
        save_json(root/'calibration_failed.json',{'records':records,'baseline':baseline});raise ValueError('Bounded calibration failed; no evaluation')
    combined=cases('E3',ecology_config('E3',scarcity,localization),'combined')
    if not viable(combined):save_json(root/'calibration_failed.json',{'records':records,'combined':combined});raise ValueError('Combined ecology collapses; no retuning')
    audit=supply_audit(root,scarcity,localization);inputs[str(audit)]=hash_file(audit)
    frozen={'version':'0.8.0','scarcity':scarcity,'localization':localization,'calibration_seeds':[3,4],'evaluation_seeds':[0,1,2],
        'preregistration_sha256':hash_file(root/'preregistration.json'),'calibration_inputs':inputs,'baseline':baseline,'candidates':records,'combined':combined,
        'selection':'First feasible candidate by resource manipulation/viability, A only; no B/C scores or benefit tuning.'}
    save_json(root/'frozen.json',frozen);print('FROZEN:',scarcity,localization,flush=True)
    return frozen


def evaluate_pilot(root=ROOT,output=ANALYSIS/'pilot'):
    import pandas as pd
    root=Path(root);verify_frozen(root);checks=[];inputs={}
    for seed in (0,1,2):
        initial=None
        for e in ('E0','E1','E2','E3'):
            for mode in 'ABC':
                p=Path(str(case_path(root,'pilot',e,mode,seed))+'.metadata.json');m,files=load(p);s=m['statistics'];inputs[str(p)]=hash_file(p)
                if initial is None:initial=m['initial_agents_rng_sha256']
                checks.append({'ecology':e,'mode':mode,'seed':seed,'complete':m['tick']==100,'population_viable':s['population']>=10,
                    'opportunities':m['totals']['decisions'],'enough_opportunities':m['totals']['decisions']>=5000,
                    'food_viable':s['food_consumed']>=20,'water_viable':s['water_consumed']>=20,
                    'initial_agents_rng_identical':initial==m['initial_agents_rng_sha256'],
                    'e0_legacy_exact':e!='E0' or bool(m['legacy_comparison']['exact']),
                    'decision_coverage':m['files']['acquisitions']['records']==m['totals']['decisions'],
                    'life_coverage':m['files']['individuals']['records']==m['config']['simulation']['initial_population']+s['births'],
                    'food_accounting':m['resource_totals']['food']==s['food_consumed'],'water_accounting':m['resource_totals']['water']==s['water_consumed']})
    f=pd.DataFrame(checks);flags=[k for k in f if k not in ('ecology','mode','seed','opportunities')];accepted=bool(f[flags].all().all())
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True);out=folder/'gate_checks.csv'
    if out.exists():raise ValueError('Refusing pilot review overwrite')
    f.to_csv(out,index=False)
    review={'decision':'primary_allowed' if accepted else 'primary_blocked','inputs':inputs,'checks_file':str(out),'checks_sha256':hash_file(out),'flags':flags,
            'frozen_sha256':hash_file(root/'frozen.json'),'preregistration_sha256':hash_file(root/'preregistration.json'),'cases':36,'no_social_benefit_gate':True}
    save_json(root/'pilot_review.json',review);print('PILOT:',review['decision'],flush=True);return review


def pilot_passes(root):
    import pandas as pd
    root=Path(root);p=root/'pilot_review.json'
    if not p.exists():return False
    m=json.loads(p.read_text())
    if m.get('decision')!='primary_allowed' or hash_file(root/'frozen.json')!=m['frozen_sha256'] or hash_file(root/'preregistration.json')!=m['preregistration_sha256']:return False
    if any(hash_file(p)!=sha for p,sha in m.get('inputs',{}).items()):return False
    if hash_file(m['checks_file'])!=m['checks_sha256']:return False
    f=pd.read_csv(m['checks_file'])
    return len(f)==36 and bool(f[m['flags']].all().all())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('calibrate','pilot','review','primary'),required=True)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output-root',type=Path,default=ANALYSIS);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);a=p.parse_args()
    try:
        if a.phase=='calibrate':calibrate(a.root)
        elif a.phase=='review':evaluate_pilot(a.root,a.output_root/'pilot')
        else:
            if not a.seeds or len(set(a.seeds))!=len(a.seeds) or any(s not in (0,1,2) for s in a.seeds):raise ValueError('Preregistered distinct evaluation seeds0/1/2 required')
            for seed in sorted(a.seeds):
                for e in ('E0','E1','E2','E3'):
                    for mode in 'ABC':run_one(a.root,a.phase,e,mode,seed,100 if a.phase=='pilot' else 500)
    except (ValueError,OSError,KeyError,RuntimeError) as error:p.error(str(error))

if __name__=='__main__':main()
