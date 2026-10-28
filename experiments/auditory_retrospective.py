"""Window-bounded matched action frequencies. Observational, never historical Q.

Matches are within run, individual, time window and all seven recorded body/
visual integers. Audio cells are never pooled to enlarge ecological contexts.
"""
from collections import Counter,defaultdict
from dataclasses import dataclass,asdict,fields
from itertools import combinations
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd

from agents.brain import Action
from config import Config
from experiments.communication_data import discover,read_windows,file_hash
from experiments.auditory_policy import distances

BASE=('agent_id','hunger_bin','thirst_bin','energy_bin','food_dx','food_dy','water_dx','water_dy')
SCOPES=('current','encoded','current_given_encoded','encoded_given_current')
COMMON=('evidence','run','condition','seed','window_start','window_end','actual_window_end')
EXPOSURE=('decisions','current_silence','current_masked','current_identified','listening_enabled',
          'current_silence_rate','current_masked_rate','current_identified_rate','encoded_nonempty_rate',
          'current_encoded_difference_rate','silent_with_retained_event_rate')
SUMMARY=('scope','threshold','contrast','available','total_contexts','multi_feature_contexts','matched_contexts',
    'individuals','decisions_used','receiver_decisions','coverage','feature_cells','cell_n_min','cell_n_median','cell_n_max',
    'pairs','mean_context_tv','mean_context_jsd_bits','mean_pair_time_separation','unavailable_reason')
SCHEMAS={
 'retrospective_exposure_by_window':(*COMMON,*EXPOSURE),
 'retrospective_exposure_by_run':('evidence','run','condition','seed',*EXPOSURE),
 'retrospective_feature_frequencies':(*COMMON,'feature_type','feature','count','denominator','probability'),
 'retrospective_matched_contexts':(*COMMON,'scope','threshold','context_id',*BASE,'fixed_audio','feature_cells',
    'decisions_used','pairs','cell_n_min','cell_n_median','cell_n_max','mean_tv','mean_jsd_bits','time_span'),
 'retrospective_action_distances':(*COMMON,'scope','threshold','context_id','contrast','feature_a','feature_b',
    'n_a','n_b','tv','jsd_bits','entropy_a_bits','entropy_b_bits','first_tick_a','last_tick_a','first_tick_b','last_tick_b',
    'mean_time_separation','time_ranges_overlap'),
 'retrospective_matched_summary':(*COMMON,*SUMMARY),
 'retrospective_matched_by_run':('evidence','run','condition','seed',*SUMMARY),
}


class MatchTotals:
    def __init__(self):self.groups={}
    def add(self,row,agents,cells):
        key=(row['scope'],row['threshold'],row['contrast'])
        a=self.groups.setdefault(key,{'counts':Counter(),'agents':set(),'cells':Counter(),'tv':0.,'js':0.,'time':0.,'reason':row['unavailable_reason']})
        for k in ('total_contexts','multi_feature_contexts','matched_contexts','decisions_used','receiver_decisions','feature_cells','pairs'):
            a['counts'][k]+=row[k]
        a['agents'].update(agents);a['cells'].update(cells)
        if row['matched_contexts']:
            a['tv']+=row['mean_context_tv']*row['matched_contexts'];a['js']+=row['mean_context_jsd_bits']*row['matched_contexts']
            a['time']+=row['mean_pair_time_separation']*row['matched_contexts']
    def write(self,context,sink):
        for (scope,threshold,tag),a in sorted(self.groups.items()):
            c=a['counts'];n=c['matched_contexts'];hist=a['cells'];total=sum(hist.values())
            median=None
            if total:
                lo=(total-1)//2;hi=total//2;cum=0;left=right=None
                for size,count in sorted(hist.items()):
                    if cum<=lo<cum+count:left=size
                    if cum<=hi<cum+count:right=size;break
                    cum+=count
                median=(left+right)/2
            sink.add('retrospective_matched_by_run',{**context,'scope':scope,'threshold':threshold,'contrast':tag,
                'available':bool(n),**dict(c),'individuals':len(a['agents']),'coverage':ratio(c['decisions_used'],c['receiver_decisions']),
                'cell_n_min':min(hist) if hist else None,'cell_n_median':median,'cell_n_max':max(hist) if hist else None,
                'mean_context_tv':ratio(a['tv'],n),'mean_context_jsd_bits':ratio(a['js'],n),
                'mean_pair_time_separation':ratio(a['time'],n),'unavailable_reason':None if n else a['reason']})


@dataclass(frozen=True)
class Options:
    window_size:int=100
    thresholds:tuple=(2,5,10)
    def __post_init__(self):
        if type(self.window_size) is not int or self.window_size<1 or not self.thresholds or any(type(n) is not int or n<1 for n in self.thresholds) or len(set(self.thresholds))!=len(self.thresholds):
            raise ValueError('Use a positive window and distinct positive cell thresholds')


def _j(value):return json.dumps(value,separators=(',',':'))
def ratio(a,b):return a/b if b else None

def entropy(p):return -sum(x*math.log2(x) for x in p if x)


def auditory_features(frame,listening):
    current=[];encoded=[]
    for kind,sig,direction,ekind,esig,edir,age in zip(frame.auditory_kind,frame.heard_signal_id,frame.heard_direction,
            frame.encoded_auditory_kind,frame.encoded_signal_id,frame.encoded_direction,frame.encoded_age):
        current.append((kind,) if kind=='SILENCE' else (kind,0) if kind=='MASKED' else (kind,int(sig),direction,0))
        encoded.append(('NOT_ENCODED',) if not listening else (ekind,) if ekind=='SILENCE' else
                       (ekind,int(age)) if ekind=='MASKED' else (ekind,int(esig),edir,int(age)))
    return current,encoded


def factor(keys):
    codes,unique=pd.factorize(pd.Index(keys,tupleize_cols=False),sort=True)
    return codes,tuple(unique)


def contrast(a,b):
    kinds={a[0],b[0]}
    if kinds=={'SILENCE','IDENTIFIED'}:return 'silence_identified'
    if kinds=={'SILENCE','MASKED'}:return 'silence_masked'
    if a[0]==b[0]=='IDENTIFIED':
        if a[1]!=b[1] and a[2:]==b[2:]:return 'different_id_same_direction_age'
        if a[1]==b[1] and a[2]!=b[2] and a[3]==b[3]:return 'different_direction_same_id_age'
        if a[1:3]==b[1:3] and a[3]!=b[3]:return 'different_retained_age'
    return 'other_exact_feature_difference'


def match_scope(frame,current,encoded,context,scope,thresholds,actions,sink,listening,prepared=None,run_totals=None):
    receiver=len(frame)
    if prepared is None:
        base_keys=list(zip(*(frame[c].tolist() for c in BASE)))
        base_codes,base_unique=factor(base_keys)
        current_codes,current_unique=factor(current);encoded_codes,encoded_unique=factor(encoded)
    else:base_codes,base_unique,current_codes,current_unique,encoded_codes,encoded_unique=prepared
    fixed=None;features=current_unique;codes=current_codes;contexts=base_codes
    if scope=='encoded':features=encoded_unique;codes=encoded_codes
    elif scope=='current_given_encoded':
        fixed=encoded_unique;contexts=base_codes*len(encoded_unique)+encoded_codes
    elif scope=='encoded_given_current':
        fixed=current_unique;contexts=base_codes*len(current_unique)+current_codes;features=encoded_unique;codes=encoded_codes
    context_count=len(np.unique(contexts))
    if not listening and scope!='current':
        for threshold in sorted(thresholds):
            row={**context,'scope':scope,'threshold':threshold,
            'contrast':'all','available':False,'total_contexts':context_count,'multi_feature_contexts':0,'matched_contexts':0,
            'individuals':0,'decisions_used':0,'receiver_decisions':receiver,'coverage':0,'feature_cells':0,'pairs':0,
            'unavailable_reason':'auditory_not_encoded'}
            sink.add('retrospective_matched_summary',row)
            if run_totals is not None:run_totals.add(row,set(),[])
        return
    action_map={name:i for i,name in enumerate(actions)}
    action=np.array([action_map[a] for a in frame.physical_action],dtype=np.int64)
    packed=contexts*len(features)+codes
    unique,inverse=np.unique(packed,return_inverse=True)
    counts=np.bincount(inverse*len(actions)+action,minlength=len(unique)*len(actions)).reshape(-1,len(actions))
    sizes=counts.sum(axis=1);ctx=unique//len(features);feat=unique%len(features)
    first=np.full(len(unique),np.iinfo(np.int64).max,dtype=np.int64);last=np.zeros(len(unique),dtype=np.int64)
    ticks=frame.tick.to_numpy(dtype=np.int64)
    np.minimum.at(first,inverse,ticks);np.maximum.at(last,inverse,ticks)
    mean_time=np.bincount(inverse,weights=ticks,minlength=len(unique))/sizes
    _,raw_feature_counts=np.unique(ctx,return_counts=True);multi=int((raw_feature_counts>1).sum())
    for threshold in sorted(thresholds):
        indices=np.flatnonzero(sizes>=threshold)
        boundaries=np.r_[0,np.flatnonzero(np.diff(ctx[indices]))+1,len(indices)] if len(indices) else np.array([0])
        accum=defaultdict(lambda:{'contexts':0,'agents':set(),'decisions':0,'cells':[],'pairs':0,'tv':0.,'js':0.,'time':0.})
        for low,high in zip(boundaries[:-1],boundaries[1:]):
            cell=indices[low:high]
            if len(cell)<2:continue
            code=int(ctx[cell[0]]);base=int(code//len(fixed)) if fixed is not None else code
            fixed_feature=fixed[code%len(fixed)] if fixed is not None else None
            physical=dict(zip(BASE,base_unique[base]));cid=_j([*base_unique[base],fixed_feature])
            grouped=defaultdict(list);used=defaultdict(set)
            for a,b in combinations(cell,2):
                pa,pb=counts[a]/sizes[a],counts[b]/sizes[b];tv,js=distances(pa,pb)
                kind=contrast(features[feat[a]],features[feat[b]])
                separation=abs(mean_time[a]-mean_time[b])
                sink.add('retrospective_action_distances',{**context,'scope':scope,'threshold':threshold,'context_id':cid,
                    'contrast':kind,'feature_a':_j(features[feat[a]]),'feature_b':_j(features[feat[b]]),
                    'n_a':int(sizes[a]),'n_b':int(sizes[b]),'tv':tv,'jsd_bits':js,'entropy_a_bits':entropy(pa),
                    'entropy_b_bits':entropy(pb),'first_tick_a':int(first[a]),'last_tick_a':int(last[a]),
                    'first_tick_b':int(first[b]),'last_tick_b':int(last[b]),'mean_time_separation':separation,
                    'time_ranges_overlap':bool(max(first[a],first[b])<=min(last[a],last[b]))})
                for tag in ('all',kind):grouped[tag].append((tv,js,separation));used[tag].update((int(a),int(b)))
            all_pairs=grouped['all']
            sink.add('retrospective_matched_contexts',{**context,'scope':scope,'threshold':threshold,'context_id':cid,
                **physical,'fixed_audio':_j(fixed_feature) if fixed_feature else None,'feature_cells':len(cell),
                'decisions_used':int(sizes[cell].sum()),'pairs':len(all_pairs),'cell_n_min':int(sizes[cell].min()),
                'cell_n_median':float(np.median(sizes[cell])),'cell_n_max':int(sizes[cell].max()),
                'mean_tv':sum(x[0] for x in all_pairs)/len(all_pairs),'mean_jsd_bits':sum(x[1] for x in all_pairs)/len(all_pairs),
                'time_span':int(last[cell].max()-first[cell].min())})
            for tag,pairs in grouped.items():
                target=accum[tag];target['contexts']+=1;target['agents'].add(int(physical['agent_id']))
                supported=sorted(used[tag]);target['decisions']+=int(sizes[supported].sum());target['cells'].extend(int(sizes[i]) for i in supported)
                target['pairs']+=len(pairs);target['tv']+=sum(x[0] for x in pairs)/len(pairs)
                target['js']+=sum(x[1] for x in pairs)/len(pairs);target['time']+=sum(x[2] for x in pairs)/len(pairs)
        # Fixed contrast support: missing contrasts have no identifiable distance,
        # not a zero effect. Counts/coverage are real zeros for completed windows.
        for tag in ('all','silence_identified','silence_masked','different_id_same_direction_age',
                    'different_direction_same_id_age','different_retained_age','other_exact_feature_difference'):
            a=accum[tag];n=a['contexts'];cells=a['cells']
            row={**context,'scope':scope,'threshold':threshold,'contrast':tag,
                'available':bool(n),'total_contexts':context_count,'multi_feature_contexts':multi,'matched_contexts':n,
                'individuals':len(a['agents']),'decisions_used':a['decisions'],'receiver_decisions':receiver,
                'coverage':ratio(a['decisions'],receiver),'feature_cells':len(cells),'cell_n_min':min(cells) if cells else None,
                'cell_n_median':float(np.median(cells)) if cells else None,'cell_n_max':max(cells) if cells else None,
                'pairs':a['pairs'],'mean_context_tv':ratio(a['tv'],n),'mean_context_jsd_bits':ratio(a['js'],n),
                'mean_pair_time_separation':ratio(a['time'],n),'unavailable_reason':None if n else 'no_two_eligible_audio_cells'}
            sink.add('retrospective_matched_summary',row)
            if run_totals is not None:run_totals.add(row,a['agents'],cells)


def exposure(context,current,encoded,listening,sink,totals):
    n=len(current);counts=Counter(x[0] for x in current)
    nonempty=sum(e[0]!='SILENCE' for e in encoded) if listening else 0
    differs=sum(a!=b for a,b in zip(current,encoded)) if listening else 0
    retained=sum(a[0]=='SILENCE' and b[0]!='SILENCE' for a,b in zip(current,encoded)) if listening else 0
    totals.update({'decisions':n,'current_silence':counts['SILENCE'],'current_masked':counts['MASKED'],
                   'current_identified':counts['IDENTIFIED'],'nonempty':nonempty,'differs':differs,'retained':retained})
    row={**context,'decisions':n,'current_silence':counts['SILENCE'],'current_masked':counts['MASKED'],
         'current_identified':counts['IDENTIFIED'],'listening_enabled':listening,
         'current_silence_rate':ratio(counts['SILENCE'],n),'current_masked_rate':ratio(counts['MASKED'],n),
         'current_identified_rate':ratio(counts['IDENTIFIED'],n),'encoded_nonempty_rate':ratio(nonempty,n) if listening else None,
         'current_encoded_difference_rate':ratio(differs,n) if listening else None,
         'silent_with_retained_event_rate':ratio(retained,n) if listening else None}
    sink.add('retrospective_exposure_by_window',row)
    frequencies={'current_kind':Counter(x[0] for x in current),
                 'current_identified_id':Counter(x[1] for x in current if x[0]=='IDENTIFIED'),
                 'current_identified_direction':Counter(x[2] for x in current if x[0]=='IDENTIFIED'),
                 'encoded_feature':Counter(encoded)}
    for kind,values in frequencies.items():
        denominator=n if kind in ('current_kind','encoded_feature') else counts['IDENTIFIED']
        for feature,count in sorted(values.items(),key=lambda kv:str(kv[0])):
            sink.add('retrospective_feature_frequencies',{**context,'feature_type':kind,'feature':_j(feature),
                'count':count,'denominator':denominator,'probability':ratio(count,denominator) if kind!='encoded_feature' or listening else None})


def run_analysis(data_dir,sink,options,archive_metadata=None,verification=None,pattern='*_communication.csv'):
    runs=discover(data_dir,pattern);registry=None;receipt={}
    if archive_metadata is not None:
        registry=json.loads(Path(archive_metadata).read_text())
        saved=json.loads(Path(verification).read_text())['output_hashes'] if verification else None
        if saved and file_hash(archive_metadata)!=saved['analysis_metadata.json']:
            raise ValueError('Archived communication analysis metadata hash mismatch')
        receipt={'analysis_metadata_sha256':file_hash(archive_metadata),
                 'verification_sha256':file_hash(verification) if verification else None}
    for run in runs:
        meta,cfg=run['metadata'],run['metadata']['config']
        if set(cfg)!={f.name for f in fields(Config)}:raise ValueError('Complete historical Config required')
        Config(**cfg)
        if registry is not None:
            matches=[x for x in registry['inputs'] if x['sha256']==run['sha256'] and x['metadata_sha256']==run['metadata_sha256'] and
                     x['simulation_metadata']['config']==cfg and x['simulation_metadata']['final_tick']==meta['final_tick']]
            if len(matches)!=1:raise ValueError('Historical recording is not an exact verified archived input')
        print(f"Retrospective {run['condition']} seed={run['seed']}",flush=True)
        totals=Counter();matched=MatchTotals();listening=cfg['brain']=='learning' and cfg['learning_uses_auditory']
        actions=tuple(a.value for a in Action if cfg['reproduction_enabled'] or a is not Action.REPRODUCE)
        for index,frame in read_windows(run,options.window_size):
            start=index*options.window_size;actual=min(start+options.window_size,meta['final_tick'])
            context={'evidence':'observational','run':run['run'],'condition':run['condition'],'seed':run['seed'],
                     'window_start':start,'window_end':start+options.window_size,'actual_window_end':actual}
            if not frame.empty and frame.tick.nunique()!=actual-start:
                raise ValueError('Missing recording ticks: incomplete data cannot be matched safely')
            current,encoded=auditory_features(frame,listening)
            exposure(context,current,encoded,listening,sink,totals)
            if frame.empty:continue
            bc,bu=factor(list(zip(*(frame[c].tolist() for c in BASE))))
            cc,cu=factor(current);ec,eu=factor(encoded)
            prepared=(bc,bu,cc,cu,ec,eu)
            for scope in SCOPES:match_scope(frame,current,encoded,context,scope,options.thresholds,actions,sink,listening,prepared,matched)
        n=totals['decisions']
        expected=meta['statistics']['exploratory_actions']+meta['statistics']['exploitative_actions']
        if cfg['brain']=='learning' and n!=expected:raise ValueError('Historical decision count differs from completed statistics')
        sink.add('retrospective_exposure_by_run',{'evidence':'observational','run':run['run'],'condition':run['condition'],'seed':run['seed'],
            **{k:totals[k] for k in ('decisions','current_silence','current_masked','current_identified')},'listening_enabled':listening,
            'current_silence_rate':ratio(totals['current_silence'],n),'current_masked_rate':ratio(totals['current_masked'],n),
            'current_identified_rate':ratio(totals['current_identified'],n),'encoded_nonempty_rate':ratio(totals['nonempty'],n) if listening else None,
            'current_encoded_difference_rate':ratio(totals['differs'],n) if listening else None,
            'silent_with_retained_event_rate':ratio(totals['retained'],n) if listening else None})
        matched.write({'evidence':'observational','run':run['run'],'condition':run['condition'],'seed':run['seed']},sink)
    return runs,receipt
