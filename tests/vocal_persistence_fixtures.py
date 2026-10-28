"""Synthetic diagnostic files, not ecological simulations or historical fixtures."""
from collections import OrderedDict,defaultdict
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from config import Config
from simulation.statistics import Statistics
from simulation.vocal_learning_observer import (CSVVocalLearningWriter,SCHEMAS,hash_file,recording_paths,q_diagnostics)


def values(preferred=None,tied=()):
    q=[0.0]*17
    if preferred is not None:q[preferred]=1.0
    for action in tied:q[action]=1.0
    return q


def write_run(directory,name='C_seed_0',seed=0,listening=False,encounters=None,deaths=None,capacity=256,final_tick=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    encounters=encounters if encounters is not None else [(t,0,0,values(4),4,False) for t in range(1,7)]
    deaths=deaths or {};horizon=max((e[0] for e in encounters),default=0) if final_tick is None else final_tick
    founders={aid for tick,aid,*_ in encounters if tick==1}
    agents={aid for _,aid,*_ in encounters}
    cfg=Config.evolution(brain='learning',random_seed=seed,max_ticks=horizon,initial_population=len(founders),
        learning_uses_auditory=listening,vocal_learning_memory_capacity=capacity)
    config=asdict(cfg);condition='D' if listening else 'C'
    path=directory/(name+'.csv.gz');active=defaultdict(OrderedDict);hist=defaultdict(lambda:defaultdict(lambda:[0,0,0]))
    counts=defaultdict(lambda:[0,0,0,0]);pending={};first={};keys={}
    exploratory=emissions=0
    with ExitStack() as stack:
        writer=CSVVocalLearningWriter.open(stack,path)
        def emit(kind,row):writer(kind,tuple(row.get(k) for k in SCHEMAS[kind]))
        def row_event(event,tick,aid,sid=None):
            if event=='allocate':
                h=hist[aid][sid];h[2]+=1;counts[aid][0]+=1
                active[aid][sid]=[0,0,h[2]]
                emit('rows',dict(event=event,tick=tick,agent_id=aid,state_id=sid,visits=h[0],updates=h[1],epoch=h[2]))
            elif event=='evict':
                visit,update,epoch=active[aid].pop(sid);counts[aid][1]+=1
                emit('rows',dict(event=event,tick=tick,agent_id=aid,state_id=sid,visits=visit,updates=update,epoch=epoch))
            else:
                emit('rows',dict(event=event,tick=tick,agent_id=aid,visits=counts[aid][3],updates=counts[aid][2],rows=len(active[aid])))
        def update(tick,aid,age,previous,current_q=None,terminal=False):
            sid,action,old=previous
            desired=0.8*old if terminal else current_q[action] if current_q is not None else old
            target=old+(desired-old)/cfg.learning_rate
            new=old+cfg.learning_rate*(target-old)
            h=hist[aid][sid];h[1]+=1;active[aid][sid][1]+=1;counts[aid][2]+=1
            emit('updates',dict(tick=tick,agent_id=aid,age=age,state_id=sid,action=action,reward=target,next_value=0.0,
                target=target,old_q=old,new_q=new,td_error=target-old,abs_td_error=abs(target-old),q_change=abs(new-old),
                visits=h[0],updates=h[1],row_updates=active[aid][sid][1],epoch=h[2],recreated=int(h[2]>1),terminal=int(terminal)))
        for tick,aid,sid,q,action,explore in encounters:
            first.setdefault(aid,tick);age=tick-first[aid]
            if sid not in keys:
                index=len(keys);key=[index%3,(index//3)%3,2,(index//9)%3-1,0,0,0]
                if listening:key.append(['SILENCE'])
                keys[sid]=key;emit('states',dict(state_id=sid,key_json=json.dumps(key,separators=(',',':'))))
            allocated,evicted=counts[aid][:2];present=sid in active[aid]
            if aid in pending:
                previous=pending[aid];update(tick,aid,age,previous,q if previous[0]==sid else None)
            before=sid in active[aid]
            if not before:
                if len(active[aid])==capacity:row_event('evict',tick,aid,next(iter(active[aid])))
                row_event('allocate',tick,aid,sid)
            active[aid].move_to_end(sid)
            h=hist[aid][sid];novelty='first' if not h[0] else 'stored' if present else 'recreated'
            h[0]+=1;active[aid][sid][0]+=1;counts[aid][3]+=1
            diag=q_diagnostics(q,cfg.learning_epsilon)
            row=dict(tick=tick,agent_id=aid,age=age,generation=0 if aid in founders else 1,condition=condition,
                state_id=sid,rows=len(active[aid]),present_at_start=int(present),present_before_selection=int(before),
                newly_allocated=int(not before),novelty=novelty,decision_allocations=counts[aid][0]-allocated,
                decision_evictions=counts[aid][1]-evicted,allocations=counts[aid][0],evictions=counts[aid][1],updates=counts[aid][2],
                unique_states=sum(v[0]>0 for v in hist[aid].values()),state_visits=h[0],
                q_max=diag[0],q_min=diag[1],q_mean=diag[2],q_std=diag[3],q_gap=diag[4],max_ties=diag[5],all_equal=diag[6],
                silence_q=diag[7],silence_rank=diag[8],silence_is_max=diag[9],selected_q=q[action],selected_action=action,
                exploratory=int(explore),n_actions=17,epsilon=cfg.learning_epsilon,p_silence=diag[10],at_capacity=int(len(active[aid])==capacity))
            emit('decisions',row);pending[aid]=(sid,action,q[action]);exploratory+=int(explore);emissions+=action!=0
            if deaths.get(aid)==tick:
                update(tick,aid,age,pending[aid],terminal=True);row_event('death',tick,aid);del pending[aid]
        for aid in sorted(agents-set(deaths)):row_event('run_end',horizon,aid)
    stats=Statistics(tick=horizon,population=len(agents-set(deaths)),births=len(agents-founders),deaths=len(deaths),
        total_learning_updates=sum(x[2] for x in counts.values()),exploratory_actions=exploratory,
        exploitative_actions=len(encounters)-exploratory,signals_emitted=emissions)
    meta={'version':'0.6.2','schema_version':1,'completed':True,'synthetic':True,'config':config,'seed':seed,
          'config_id':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),'final_tick':horizon,
          'statistics':asdict(stats),'schemas':{k:list(v) for k,v in SCHEMAS.items()},'unique_keys':len(keys),
          'files':{k:{'file':p.name,'sha256':hash_file(p),'bytes':p.stat().st_size,'records':writer.counts[k]} for k,p in recording_paths(path).items()}}
    Path(str(path)+'.metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    return path


def historical_sources(root,run):
    """Small hash-bound fake archives for join validation, not MI computation."""
    import pandas as pd
    root=Path(root);cdir=root/'communication';vdir=root/'vocal';cdir.mkdir();vdir.mkdir()
    comm=root/'oldC.csv';comm.write_text('synthetic historical communication bytes\n')
    side=Path(str(comm)+'.metadata.json')
    cm={k:run['metadata'][k] for k in ('config','final_tick','statistics')};side.write_text(json.dumps(cm))
    diag_side=run['metadata_path'];meta=json.loads(diag_side.read_text())
    meta['communication']={'file':str(comm),'sha256':hash_file(comm),'metadata_sha256':hash_file(side)}
    diag_side.write_text(json.dumps(meta))
    (cdir/'analysis_metadata.json').write_text(json.dumps({'options':{'window_size':2},'inputs':[{
        'file':comm.name,'sha256':hash_file(comm),'metadata_sha256':hash_file(side),'simulation_metadata':cm}]}))
    (vdir/'analysis_metadata.json').write_text(json.dumps({'window_size':2,'inputs':[{
        'metadata_sha256':hash_file(diag_side),'config':meta['config'],'files':meta['files']}]}))
    for file in ('sender_information.csv','vocal_preferences_by_agent.csv','vocal_preference_changes.csv'):
        (cdir/file).write_text('synthetic_header\n')
    (vdir/'windows.csv').write_text('synthetic_header\n')
    cv=root/'cv.json';vv=root/'vv.json'
    cv.write_text(json.dumps({'output_hashes':{p.name:hash_file(p) for p in cdir.iterdir()}}))
    vv.write_text(json.dumps({'analysis_output_sha256':{p.name:hash_file(p) for p in vdir.iterdir()}}))
    return cdir,vdir,cv,vv
