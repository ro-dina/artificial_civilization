"""Explicit bounded v0.7.0 verification: unchanged evolution, <=100 ticks, <=2 seeds.

Step timing excludes expensive full-state/RNG checks and ordinary CSV writes.
The three independent modes are interleaved in rotating order; this is a local
instrumented comparison, not a portable throughput or memory benchmark.
"""
from contextlib import ExitStack
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
from time import perf_counter

from config import Config,SocialConfig
from simulation.simulation import Simulation
from simulation.statistics import CSVStatisticsWriter
from simulation.social_observer import CSVSocialWriter,write_metadata
from simulation.vocal_learning_observer import hash_file
from experiments.social_verification import digest

MODES=('disabled','enabled_unrecorded','enabled_recorded')


def run_pilot(data_dir=Path('data/social_memory'),ticks=100,seeds=(0,1)):
    if type(ticks) is not int or not 1<=ticks<=100:
        raise ValueError('Pilot requires 1..100 ticks; no long studies')
    if not seeds or len(seeds)>2 or any(type(s) is not int for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('Use at most two distinct integer seeds')
    data_dir=Path(data_dir)
    if data_dir.exists() and any(data_dir.iterdir()):raise ValueError('Pilot refuses nonempty destinations')
    data_dir.mkdir(parents=True,exist_ok=True)
    protocol={'version':'0.7.0','ticks':ticks,'seeds':sorted(seeds),'preset':'Config.evolution(brain=learning)',
              'modes':MODES,'social':asdict(SocialConfig(True)),
              'timing':'Sum ordinary step times, plus construction and recorded finish/gzip close separately. Full-state hashes/ordinary CSV writes/metadata hashing excluded. Rotating per-tick mode order.',
              'no_ecology_tuning':True}
    (data_dir/'pilot_design.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start_all=perf_counter();rng=random.getstate();results=[]
    for seed in sorted(seeds):
        cfg=Config.evolution(random_seed=seed,brain='learning',max_ticks=ticks)
        settings=SocialConfig(True)
        raw=data_dir/f'seed_{seed}_social.csv.gz'
        simulations={};writers={};timings={m:0. for m in MODES};init_times={}
        trajectories={m:hashlib.sha256() for m in MODES};social_trajectories={m:hashlib.sha256() for m in MODES[1:]}
        peak_entries={m:0 for m in MODES};peak_cache_ids=0;maximum_occupancy=0
        social_stack=ExitStack()
        with ExitStack() as stack:
            stack.callback(social_stack.close)
            for mode in MODES:
                begin=perf_counter()
                social_writer=CSVSocialWriter.open(social_stack,raw) if mode=='enabled_recorded' else None
                simulations[mode]=Simulation(cfg,social_config=settings if mode!='disabled' else None,social_sink=social_writer)
                if mode=='enabled_recorded':recorded_writer=social_writer
                init_times[mode]=perf_counter()-begin
                dest=data_dir/'statistics'/f'{mode}_seed_{seed}.csv';dest.parent.mkdir(exist_ok=True)
                writer=CSVStatisticsWriter(stack.enter_context(dest.open('x',encoding='utf-8',newline='')),
                    metadata={'mode':'evolution','brain':'learning','seed':seed,
                              'config_id':hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()})
                writers[mode]=writer;writer.write(simulations[mode].statistics)
            for iteration in range(ticks+1):
                if iteration:
                    rotation=(iteration-1)%len(MODES);order=MODES[rotation:]+MODES[:rotation]
                    for mode in order:
                        begin=perf_counter();simulations[mode].step();timings[mode]+=perf_counter()-begin
                        writers[mode].write(simulations[mode].statistics)
                hashes={mode:digest(sim) for mode,sim in simulations.items()}
                if len(set(hashes.values()))!=1:raise RuntimeError(f'Ecological/RNG difference at seed={seed}, tick={iteration}')
                for mode,h in hashes.items():trajectories[mode].update(h.encode())
                social={mode:digest(simulations[mode],include_social=True) for mode in MODES[1:]}
                if len(set(social.values()))!=1:raise RuntimeError('Recording changed social state')
                for mode,h in social.items():social_trajectories[mode].update(h.encode())
                for mode in MODES[1:]:
                    memories=[h.social_memory for h in simulations[mode].agents]
                    peak_entries[mode]=max(peak_entries[mode],sum(len(m) for m in memories))
                    maximum_occupancy=max(maximum_occupancy,max((len(m) for m in memories),default=0))
                observer=simulations['enabled_recorded'].social_observer
                peak_cache_ids=max(peak_cache_ids,sum(len(cache) for cache in observer._evicted.values()))
                if random.getstate()!=rng:raise RuntimeError('Pilot consumed global random state')
                if simulations['disabled'].population==0:break
            recorded=simulations['enabled_recorded']
            begin=perf_counter();recorded.social_observer.finish(recorded.statistics);social_stack.close()
            finish_seconds=perf_counter()-begin
        metadata=write_metadata(raw,recorded,recorded_writer)
        ordinary_hashes={mode:hash_file(data_dir/'statistics'/f'{mode}_seed_{seed}.csv') for mode in MODES}
        if len(set(ordinary_hashes.values()))!=1:raise RuntimeError('Ordinary CSV changed')
        result={'seed':seed,'final_tick':recorded.tick,'config':asdict(cfg),'statistics':asdict(recorded.statistics),
            'step_wall_seconds':timings,'construction_seconds':init_times,'recorded_finish_close_seconds':finish_seconds,
            'ecological_trajectory_sha256':{m:h.hexdigest() for m,h in trajectories.items()},
            'social_trajectory_sha256':{m:h.hexdigest() for m,h in social_trajectories.items()},
            'aggregate_csv_sha256':ordinary_hashes,'peak_social_entries':peak_entries,'maximum_individual_occupancy':maximum_occupancy,
            'peak_privileged_cache_ids':peak_cache_ids,'record_counts':metadata['event_counts'],
            'raw_bytes':raw.stat().st_size,'metadata_bytes':Path(str(raw)+'.metadata.json').stat().st_size,
            'state_rng_and_recording_equivalence':True}
        results.append(result)
        print(f"seed={seed} tick={recorded.tick} population={recorded.population} births={recorded.statistics.births} deaths={recorded.statistics.deaths} reunions={metadata['event_counts'].get('reunion',0)}",flush=True)
    report={'protocol':protocol,'runs':results,'global_rng_unchanged':random.getstate()==rng,
            'wall_seconds_including_validation':perf_counter()-start_all,
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'platform':sys.platform,'source_sha256':{name:hash_file(name) for name in ('experiments/social_memory_pilot.py','experiments/social_verification.py')},
            'memory_measurement':'Whole-process peak includes three concurrent simulations and full-state verification. Not a per-mode memory estimate. Logical social entry/cache counts reported separately; each is bounded by living population*capacity.'}
    (data_dir/'pilot_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/social_memory'))
    parser.add_argument('--ticks',type=int,default=100)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1])
    try:run_pilot(**vars(parser.parse_args()))
    except (ValueError,OSError) as error:parser.error(str(error))

if __name__=='__main__':main()
