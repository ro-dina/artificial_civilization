"""Small online A-D benchmark plus exact final memory/replay comparison."""
from dataclasses import asdict
import hashlib,json,resource,sys
from pathlib import Path
from time import perf_counter
from config import Config
from simulation.simulation import Simulation
from experiments.social_verification import digest
from experiments.auditory_policy import frozen
from experiments.social_policy_metrics import primary,Treatment
from experiments.analyze_social_memory_policies import treatments


def benchmark(output_path=Path('data/social_memory_v071/online_benchmark.json'),seeds=(0,1,2),ticks=100,replay_metadata=Path('analysis/social_memory_v071/pilot/analysis_metadata.json')):
    if type(ticks) is not int or not 1<=ticks<=100:raise ValueError('Online mechanism benchmark limited to 100 ticks')
    output_path=Path(output_path)
    if output_path.exists():raise ValueError('Refusing existing benchmark')
    expected=json.loads(Path(replay_metadata).read_text()) if replay_metadata is not None else None
    rows=[]
    for seed in seeds:
        ecology=[]
        for t in treatments(candidate_capacity=32):
            cfg=Config.evolution(brain='learning',random_seed=seed,max_ticks=ticks)
            start=perf_counter();sim=Simulation(cfg,social_config=t.social,social_policy=t.policy);initial=perf_counter()-start
            start=perf_counter()
            for _ in range(ticks):
                if not sim.population:break
                sim.step()
            elapsed=perf_counter()-start
            eco=digest(sim);ecology.append(eco)
            private=tuple((h.id,frozen(vars(h.social_memory))) for h in sorted(sim.agents,key=lambda h:h.id))
            private_hash=hashlib.sha256(json.dumps(private,separators=(',',':')).encode()).hexdigest()
            if expected is not None:
                source=next(i for i in expected['inputs'] if i['config']['simulation']['random_seed']==seed)
                if private_hash!=source['private_state_sha256'][t.key]:raise RuntimeError('Online/replay full private memory difference')
            row={'seed':seed,**t.fields(),'initialization_seconds':initial,'simulation_seconds':elapsed,'final_tick':sim.tick,
                 'ecological_state_sha256':eco,'private_state_sha256':private_hash,'matches_replay':expected is not None}
            rows.append(row);print(f'Online {t.condition} seed={seed}: {elapsed:.3f}s; memory matches replay',flush=True)
        if len(set(ecology))!=1:raise RuntimeError('Ecology changed across memory conditions')
    report={'version':'0.7.1','ticks':ticks,'seeds':list(seeds),'rows':rows,
        'definitions':'Unrecorded headless simulation loop. Hashing excluded. Same evolution configuration; new memory only. Separate replay timing includes histories/statistics/plots.',
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
    output_path.parent.mkdir(parents=True,exist_ok=True);output_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

if __name__=='__main__':benchmark()
