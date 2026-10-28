"""Explicit six-run v0.6.2 pilot, with byte-verified reuse of archived communication.

Unlike the analyser this starts simulations. Only C/D, max 1000 ticks. A fresh
simultaneous communication recording is retained unless an identical archived
recording (full config, statistics, complete decision CSV) is verified. In that
case the diagnostic sidecar references that immutable archive, avoiding a second
permanent copy. No permutation analysis is rerun.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import Config
from simulation.vocal_learning_observer import hash_file, recording_paths

ROOT=Path(__file__).resolve().parents[1]


def run_pilot(output_dir=Path('data/vocal_learning'), ticks=1000, seeds=(0,1,2), archive_dir=Path('data/communication')):
    if type(ticks) is not int or not 1 <= ticks <= 1000 or not seeds or any(type(s) is not int for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('Short pilot: 1..1000 ticks, distinct integer seeds')
    directory=Path(output_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Pilot output must be a new/empty directory')
    directory.mkdir(parents=True,exist_ok=True)
    reports=[]
    for c,listening in [('C',False),('D',True)]:
        for seed in sorted(seeds):
            name=f'{c}_seed_{seed}'
            diag=directory/(name+'.csv.gz')
            comm=directory/(name+'_communication.csv')
            stats=directory/(name+'_statistics.csv')
            cmd=[sys.executable,str(ROOT/'main.py'),'--mode','evolution','--brain','learning','--seed',str(seed),'--ticks',str(ticks),
                 '--learning-uses-auditory' if listening else '--no-learning-uses-auditory',
                 '--learning-controls-vocalization','--csv',str(stats),'--communication-csv',str(comm),'--vocal-learning-csv',str(diag)]
            print(f'Running {name}, {ticks} ticks, both observers',flush=True)
            start=perf_counter()
            with (directory/(name+'.log')).open('w') as stream:
                subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            elapsed=perf_counter()-start
            sidecar=Path(str(diag)+'.metadata.json');meta=json.loads(sidecar.read_text())
            old=Path(archive_dir).resolve()/(name+'_communication.csv')
            old_stats=Path(archive_dir).resolve()/(name+'_statistics.csv')
            receipt=None
            if old.exists() and old_stats.exists():
                old_meta=Path(str(old)+'.metadata.json')
                archived=json.loads(old_meta.read_text())
                if archived['config']!=meta['config'] or archived['statistics']!=meta['statistics'] or archived['final_tick']!=meta['final_tick']:
                    raise ValueError(f'{name}: archived configuration/ecology differs; refusing silent join')
                if hash_file(old)!=hash_file(comm) or hash_file(old_stats)!=hash_file(stats):
                    raise ValueError(f'{name}: actual replay trajectories differ from archived records')
                receipt={'archived_file':str(old),'identical_complete_communication_sha256':hash_file(comm),
                         'identical_statistics_sha256':hash_file(stats),'original_simultaneous_sidecar_sha256':hash_file(Path(str(comm)+'.metadata.json'))}
                meta['communication']={'file':str(old),'sha256':hash_file(old),'metadata_sha256':hash_file(old_meta)}
                meta['archived_replay_verification']=receipt
                sidecar.write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
                comm.unlink()  # Only this command's newly-created, verified redundant copy.
            report={'condition':c,'seed':seed,'seconds':elapsed,'config':meta['config'],'final_tick':meta['final_tick'],
                    'statistics':meta['statistics'],'diagnostic_bytes':sum(p.stat().st_size for p in recording_paths(diag).values()),
                    'records':{k:v['records'] for k,v in meta['files'].items()},'unique_keys':meta['unique_keys'],
                    'peak_agent_state_histories':meta['peak_agent_state_histories'],'archive_match':receipt,
                    'metadata_sha256':hash_file(sidecar),'command':cmd}
            reports.append(report)
            print(f"Completed {name}: {elapsed:.2f}s; {report['diagnostic_bytes']:,} bytes; archive match={receipt is not None}",flush=True)
            (directory/'pilot_manifest.json').write_text(json.dumps({'version':'0.6.2','completed':False,'runs':reports},indent=2)+'\n')
    manifest={'version':'0.6.2','completed':True,'ticks':ticks,'seeds':sorted(seeds),'runs':reports,
              'ecology':asdict(Config.evolution(brain='learning'))}
    (directory/'pilot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('data/vocal_learning'))
    parser.add_argument('--archive-dir',type=Path,default=Path('data/communication'))
    parser.add_argument('--ticks',type=int,default=1000)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    args=parser.parse_args()
    try:run_pilot(**vars(args))
    except (ValueError,OSError,subprocess.CalledProcessError) as error:parser.error(str(error))

if __name__=='__main__':main()
