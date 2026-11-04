"""Pure readers/feature diagnostics, with a run (not a decision) as replication."""
from collections import Counter,defaultdict
import gzip,hashlib,json,math
from pathlib import Path

from agents.nuisance import NuisanceConfig,NuisanceSource
from simulation.controls_observer import C_INDEX


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path,float_precision='round_trip')


def load(path,verify=True):
    path=Path(path);m=json.loads(path.read_text())
    if not m.get('completed') or m.get('version') not in ('0.7.2','0.7.3','0.7.4'):raise ValueError('Expected completed v072/v073/v074 metadata')
    if m.get('format') not in ('social_learning_aggregates','social_information_controls'):raise ValueError('Unexpected diagnostic format')
    if verify:
        for kind,record in m['files'].items():
            if Path(record['file']).name!=record['file']:raise ValueError('Unsafe diagnostic filename')
            p=path.parent/record['file']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Checksum mismatch: '+str(p))
            if p.stat().st_size!=record['bytes']:raise ValueError('File size mismatch: '+str(p))
            with (gzip.open(p,'rt',newline='') if p.suffix=='.gz' else p.open(newline='')) as stream:
                import csv
                header=next(csv.reader(stream))
            if header!=m['schemas'][kind]:raise ValueError('Header mismatch: '+str(p))
    return m,{k:path.parent/v['file'] for k,v in m['files'].items()}


def entropy(counts):
    n=sum(counts)
    return -sum((x/n)*math.log2(x/n) for x in counts if x>0) if n else float('nan')


def token_metrics(base,tokens,weights=None):
    counts=Counter();joint=defaultdict(Counter)
    weights=[1]*len(tokens) if weights is None else weights
    for x,t,n in zip(base,tokens,weights):counts[int(t)]+=int(n);joint[int(x)][int(t)]+=int(n)
    total=sum(counts.values());h=entropy(counts.values())
    conditional=sum(sum(c.values())*entropy(c.values()) for c in joint.values())/total if total else float('nan')
    supported={x:c for x,c in joint.items() if sum(c.values())>=20};supported_n=sum(sum(c.values()) for c in supported.values())
    return {'token_entropy':h,'effective_tokens':2**h,'observed_tokens':len(counts),
            'conditional_entropy':conditional,'conditional_supported_decisions':supported_n,
            'conditional_supported_entropy':sum(sum(c.values())*entropy(c.values()) for c in supported.values())/supported_n if supported_n else float('nan'),
            'decisions':total},counts,joint


def attach_nuisance(frame,model,seed):
    """Ownership lookup only; IDs never seed/enter a channel. Actual order preserved."""
    source=NuisanceSource(NuisanceConfig(model,seed));channels={};tokens=[]
    from agents.nuisance import NonstationaryNuisanceModel
    scheduled=isinstance(model,NonstationaryNuisanceModel)
    columns=['agent_id','terminal','tick'] if scheduled else ['agent_id','terminal']
    for row in frame[columns].itertuples(index=False,name=None):
        identity,terminal=row[:2]
        if identity not in channels:channels[identity]=source.new_channel()
        tokens.append(channels[identity].advance(int(row[2])) if scheduled else channels[identity].advance())
        if terminal:channels.pop(identity)
    return tokens


def sequence_metrics(frame,tokens):
    result,counts,joint=token_metrics(frame.base_id,tokens)
    last={};pairs=set();visits=Counter();runs=Counter();same=transitions=0
    for (identity,x,terminal),t in zip(frame[['agent_id','base_id','terminal']].itertuples(index=False,name=None),tokens):
        visits[identity,int(x),int(t)]+=1
        old=last.get(identity)
        if old:
            transitions+=1;same+=old[0]==t
            if old[0]!=t:runs[old[1],'observed_change']+=1
        length=old[1]+1 if old and old[0]==t else 1
        if terminal:runs[length,'death_censored']+=1;last.pop(identity,None)
        else:last[identity]=(t,length)
    for t,length in last.values():runs[length,'run_end_censored']+=1
    distinct=len(visits);n=len(tokens)
    result.update(first_state_fraction=distinct/n if n else float('nan'),mean_visits=n/distinct if distinct else float('nan'),
                  distinct_agent_states=distinct,states_once=sum(v==1 for v in visits.values()),
                  fraction_states_once=sum(v==1 for v in visits.values())/distinct if distinct else float('nan'),
                  same_token=same/transitions if transitions else float('nan'),same_transitions=same,transitions=transitions,
                  run_count=sum(runs.values()),mean_run_length=sum(k[0]*v for k,v in runs.items())/sum(runs.values()) if runs else float('nan'))
    return result,counts,joint,runs,Counter(visits.values())


def original_frame(path):
    m,files=load(path)
    if 'outcomes' not in files:raise ValueError('Archived aggregates do not contain ordered decisions or rewards')
    f=read_csv(files['outcomes'])
    if len(f)!=m['files']['outcomes']['records']:raise ValueError('Outcome record count mismatch')
    if len(f) and (f.sequence.tolist()!=list(range(len(f))) or not f.tick.is_monotonic_increasing):raise ValueError('Invalid actual decision order')
    if not f.terminal.isin([0,1]).all() or not f.action.between(0,7).all() or not f.reward.map(math.isfinite).all():raise ValueError('Invalid actual outcome')
    return m,files,f


def tokens_for(frame,condition):
    return ([-1]*len(frame) if condition=='A' else frame.b_token.tolist() if condition=='B' else frame.c_token.tolist() if condition=='C' else frame.used_token.tolist())


def archived_profile(path):
    m,files=load(path);f=read_csv(files['actions']);condition=m['condition']
    tokens=f.visible.tolist() if condition=='B' else [C_INDEX[x] for x in zip(f.visible,f.remembered_direction)]
    result,counts,joint=token_metrics(f.base_id,tokens,f.n)
    t=m['totals'];n=t['decisions'];distinct=t['first_agent_states']
    result.update(first_state_fraction=distinct/n,mean_visits=n/distinct,distinct_agent_states=distinct,
                  same_token=float('nan'),mean_run_length=float('nan'))
    return result,counts,joint
