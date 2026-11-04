"""v074 one preregistered calibration/evaluation cycle, with a fail-closed gate."""
from collections import Counter,defaultdict
from contextlib import ExitStack
from dataclasses import asdict
import argparse,json,math,resource,sys
from pathlib import Path
from statistics import mean
from time import perf_counter
import numpy as np
import pandas as pd

from agents.nuisance import NuisanceConfig,NuisanceModel
from config import Config,SocialConfig,SocialPolicyConfig,SocialLearningConfig
from simulation.simulation import Simulation
from simulation.controls_observer import ControlsObserver,ControlsWriter,write_metadata
from simulation.vocal_learning_observer import hash_file
from experiments.information_controls import model_file,initial_fingerprint,fingerprint
from experiments.controls_metrics import attach_nuisance,read_csv
from experiments.nonstationary_metrics import source,prepared,profiles,summarize

ROOT=Path('data/social_learning_v074')
ANALYSIS=Path('analysis/social_learning_v074')


def peak_bytes():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def save_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(json.dumps(clean(payload),indent=2,allow_nan=False)+'\n')


def verify_frozen(root):
    root=Path(root);m=json.loads((root/'models/frozen.json').read_text())
    for p,sha in {**m['inputs'],str(root/'preregistration.json'):m['preregistration_sha256']}.items():
        if hash_file(p)!=sha:raise ValueError('Frozen input/design changed: '+p)
    for c,sha in m['models'].items():
        if hash_file(root/'models'/f'{c}.json')!=sha:raise ValueError('Frozen model changed')
    return m


def preregister(root,diagnostics):
    """Archive the tracked one-cycle design before any new calibration/evaluation."""
    root=Path(root);diagnostics=Path(diagnostics)
    if not (diagnostics/'metadata.json').exists():raise ValueError('Complete calibration-only diagnosis first')
    design=json.loads(Path('experiments/nonstationary_design.json').read_text())
    design['phaseI_inputs']={str(p):hash_file(p) for p in sorted(diagnostics.iterdir()) if p.is_file()}
    save_json(root/'preregistration.json',design)


def calibrate(root=ROOT,output=ANALYSIS/'calibration'):
    root=Path(root);output=Path(output);start=perf_counter()
    if (root/'models').exists() or output.exists():raise ValueError('Refusing calibration overwrite')
    if not (root/'preregistration.json').exists():raise ValueError('Preregister first')
    inputs={};models={};reference_rows=[]
    for c,control in [('B','D'),('C','E')]:
        frames=[];k=11 if c=='B' else 37;schedule=[]
        for seed in (3,4):
            p=source(c,seed);m,files,f=prepared(p);frames.append(f);inputs[str(p)]=hash_file(p)
        for tick in range(0,500,100):
            pi=[];targets=[]
            for f in frames:
                g=f[f.tick.ge(tick)&f.tick.lt(tick+100)];u=g.b_token if c=='B' else g.c_token
                counts=Counter(u);pi.append([counts[i]/len(g) for i in range(k)])
                # Feature-only temporary frame: terminal status is not a model input.
                features=g[['agent_id','base_id','tick','b_token','c_token']].assign(terminal=0,opportunity_index=0,first_tick=1,derived_pre_age=0)
                from experiments.controls_metrics import sequence_metrics
                targets.append(sequence_metrics(features,u)[0])
            probabilities=[mean(p[i] for p in pi) for i in range(k)]
            probabilities[max(range(k),key=probabilities.__getitem__)]+=1-sum(probabilities)
            independent=sum(p*p for p in probabilities);repeat=mean(r['same_token'] for r in targets)
            raw=(repeat-independent)/(1-independent) if independent<1 else 0.
            alpha=max(0.,min(1.,raw))
            schedule.append({'start_tick':tick,'probabilities':probabilities,'hold_probability':alpha,'moment_alpha_raw':raw,'moment_saturated':raw!=alpha})
        models[control]={'kind':'nonstationary','version':'0.7.4','category_count':k,'schedule':schedule,
            'fit':'Equal calibration seed moments per frozen clock epoch, no parameter search. Cross-epoch persistence retained; finite-transient deviations measured, not repaired.',
            'calibration_seeds':[3,4],'evaluation_seeds':[0,1,2]}
    for c,model in models.items():save_json(root/'models'/f'{c}.json',model)
    manifest={'version':'0.7.4','preregistration_sha256':hash_file(root/'preregistration.json'),'inputs':inputs,'models':{c:hash_file(root/'models'/f'{c}.json') for c in models},
              'calibration_seeds':[3,4],'evaluation_seeds':[0,1,2],'no_outcome_tuning':True,'source_sha256':{p:hash_file(p) for p in ('agents/nuisance.py','experiments/nonstationary_controls.py','experiments/nonstationary_metrics.py')}}
    save_json(root/'models/frozen.json',manifest)
    # Audit frozen family on fixed calibration X after fitting; no second fitting.
    rows=[];frequencies=[];segments=[];comparisons=[]
    for c,control in [('B','D'),('C','E')]:
        model=model_file(root/'models'/f'{control}.json')
        for seed in (3,4):
            m,files,f=prepared(source(c,seed));s=f.b_token.to_numpy() if c=='B' else f.c_token.to_numpy()
            sr,sf,ss,_=profiles(f,s);tag={'condition':c,'seed':seed,'realization':0,'phase':'calibration'}
            rows.append(sr.assign(**tag));frequencies.append(sf.assign(**tag));segments.append(ss.assign(**tag))
            for realization,nseed in enumerate((31,47)):
                u=attach_nuisance(f[['agent_id','tick','terminal']].assign(terminal=0),model,nseed)
                ur,uf,us,_=profiles(f,u);tag={'condition':control,'seed':seed,'realization':realization,'phase':'calibration','nuisance_seed':nseed}
                rows.append(ur.assign(**tag));frequencies.append(uf.assign(**tag));segments.append(us.assign(**tag))
                comparisons.extend(compare(f,s,f,u,files,files,c,control,seed,realization,root,500))
    output.mkdir(parents=True)
    for name,f in {'windows':pd.concat(rows,ignore_index=True),'frequencies':pd.concat(frequencies,ignore_index=True),'run_segments':pd.concat(segments,ignore_index=True),'matching':pd.DataFrame(comparisons)}.items():f.to_csv(output/f'{name}.csv',index=False,float_format='%.12g')
    save_json(output/'metadata.json',{**manifest,'phase':'calibration','wall_seconds':perf_counter()-start,'peak_memory_bytes':peak_bytes(),
        'limitations':'Independent U cannot generally match arbitrary P(S|X,t). Calibration-null comparison holds actual X fixed, reports common supported cells and coverage; entropy/TV are finite-sample diagnostics. No model redesign after this audit.'})
    print('Frozen schedules:',{c:[(e['start_tick'],round(e['hold_probability'],6)) for e in m['schedule']] for c,m in models.items()},flush=True)


def conditional_tv(a,sa,b,ub,files_a,files_b):
    catalogs=[]
    for files in (files_a,files_b):catalogs.append(read_csv(files['states']).set_index('base_id').key_json.to_dict())
    joints=[]
    for f,tokens,catalog in zip((a,b),(sa,ub),catalogs):
        joint=defaultdict(Counter)
        for x,t in zip(f.base_id,tokens):joint[catalog[int(x)]][int(t)]+=1
        joints.append(joint)
    common=sorted(x for x,c in joints[0].items() if sum(c.values())>=20 and sum(joints[1].get(x,{}).values())>=20)
    exposures=[sum(sum(j[x].values()) for x in common) for j in joints];coverage=[exposures[0]/len(a),exposures[1]/len(b)]
    tv=0.
    if common:
        for x in common:
            p,q=joints[0][x],joints[1][x];n,m=sum(p.values()),sum(q.values());weight=(n/exposures[0]+m/exposures[1])/2
            tv+=weight*sum(abs(p[t]/n-q[t]/m) for t in p.keys()|q.keys())/2
    return (tv if common and min(coverage)>=.10 else np.nan),{'common_X_cells':len(common),'reference_supported_decisions':exposures[0],'control_supported_decisions':exposures[1],'reference_coverage':coverage[0],'control_coverage':coverage[1]}


def compare(a,sa,b,ub,files_a,files_b,reference,control,seed,realization,root,horizon):
    ap,af,_,_=profiles(a,sa,horizon);bp,bf,_,_=profiles(b,ub,horizon)
    design=json.loads((Path(root)/'preregistration.json').read_text());rules=design['matching'];rows=[]
    for r in ap.itertuples(index=False):
        alt=bp[bp.window.eq(r.window)]
        if alt.empty:continue
        v=alt.iloc[0];am=a.tick.ge(r.start_tick)&a.tick.lt(r.stop_tick_exclusive);bm=b.tick.ge(r.start_tick)&b.tick.lt(r.stop_tick_exclusive)
        ca=af[af.window.eq(r.window)].set_index('token').frequency.to_dict();cb=bf[bf.window.eq(r.window)].set_index('token').frequency.to_dict()
        ctv,support=conditional_tv(a.loc[am],np.asarray(sa)[am],b.loc[bm],np.asarray(ub)[bm],files_a,files_b)
        differences={k:float(v[k])-getattr(r,k) for k in ('token_entropy','conditional_entropy','same_token','first_state_fraction','fraction_states_once')}
        differences.update(marginal_tv=sum(abs(ca.get(t,0)-cb.get(t,0)) for t in ca.keys()|cb.keys())/2,
            conditional_tv_common_X=ctv,relative_mean_visits=v.mean_visits/r.mean_visits-1,relative_effective_tokens=v.effective_tokens/r.effective_tokens-1,
            run_cdf_distance=max(abs(v[f'run_cdf_{i}']-getattr(r,f'run_cdf_{i}')) for i in range(9)))
        for metric,value in differences.items():
            rule=rules[metric];supported=math.isfinite(value)
            rows.append({'condition':control,'reference':reference,'seed':seed,'realization':realization,'horizon':horizon,'window':r.window,'complete_window':bool(r.complete_window),
                'metric':metric,'difference':value,'target':rule['target'],'gross_threshold':rule['gross'],'supported':supported,
                'within_target':supported and abs(value)<=rule['target'],'gross_failure':supported and abs(value)>rule['gross'],
                'reference_decisions':r.decisions,'control_decisions':v.decisions,**support})
    return rows


def run_one(root,phase,condition,seed,realization,ticks):
    root=Path(root);verify_frozen(root)
    if (phase not in ('pilot','primary','determinism') or condition not in ('D','E')
            or ticks!=(500 if phase=='primary' else 100) or type(seed) is not int or seed not in (0,1,2)
            or type(realization) is not int or realization not in (0,1)):raise ValueError('Preregistered bounded D/E case required')
    if ticks==500 and not primary_allowed(root):raise ValueError('Frozen gate blocks primary')
    path=root/phase/f'{condition}_seed_{seed}_r{realization}.csv.gz'
    modelpath=root/'models'/f'{condition}.json';model=model_file(modelpath);nseed=20000+100*seed+realization;start=perf_counter()
    cfg=Config.evolution(brain='learning',random_seed=seed,max_ticks=ticks);proof={};actions={}
    with ExitStack() as stack:
        writer=ControlsWriter.open(stack,path);o=ControlsObserver(writer,condition)
        s=Simulation(cfg,social_config=SocialConfig(True,32,20),social_policy=SocialPolicyConfig(),social_learning=SocialLearningConfig(),nuisance=NuisanceConfig(model,nseed),social_learning_observer=o)
        initial=initial_fingerprint(s);proof[0]=fingerprint(s)
        for _ in range(ticks):
            if not s.population:break
            s.step()
            if s.tick%100==0:proof[s.tick]=fingerprint(s);actions[s.tick]=o.action_digest.hexdigest();print(f'{condition} eco={seed} nuisance={nseed} tick={s.tick} pop={s.population}',flush=True)
        o.finish(s)
    meta=write_metadata(path,s,o,writer,realization=realization,version='0.7.4',calibration={'file':str(modelpath),'sha256':hash_file(modelpath)},
        fingerprints={'initial_ecology':initial,'checkpoints':proof,'action_checkpoints':actions})
    save_json(Path(str(path)+'.runtime.json'),{'wall_seconds':perf_counter()-start,'peak_memory_bytes':peak_bytes(),'stream_bytes':sum(f['bytes'] for f in meta['files'].values())})
    return Path(str(path)+'.metadata.json')


def evaluate(root=ROOT,output=ANALYSIS/'pilot',phase='pilot'):
    root=Path(root);output=Path(output);verify_frozen(root)
    if output.exists():raise ValueError('Refusing evaluation overwrite')
    horizon=100 if phase=='pilot' else 500;rows=[];windows=[];freq=[];runs=[];ecology=[];inputs={}
    for c in 'DE':
        reference='B' if c=='D' else 'C'
        for seed in (0,1,2):
            _,af,a=prepared(source(reference,seed));a=a[a.tick.le(horizon)]
            sa=a.b_token.to_numpy() if reference=='B' else a.c_token.to_numpy()
            ap,ff,ss,_=profiles(a,sa,horizon);tag={'condition':reference,'seed':seed,'realization':0,'phase':phase}
            windows.append(ap.assign(**tag));freq.append(ff.assign(**tag));runs.append(ss.assign(**tag))
            for realization in (0,1):
                path=root/phase/f'{c}_seed_{seed}_r{realization}.csv.gz.metadata.json';m,bf,b=prepared(path);inputs[str(path)]=hash_file(path);ub=b.used_token.to_numpy()
                rows.extend(compare(a,sa,b,ub,af,bf,reference,c,seed,realization,root,horizon))
                bp,ff,ss,_=profiles(b,ub,horizon);tag={'condition':c,'seed':seed,'realization':realization,'phase':phase}
                windows.append(bp.assign(**tag));freq.append(ff.assign(**tag));runs.append(ss.assign(**tag))
                ecology.append({**tag,'nuisance_seed':m['nuisance_seed'],**m['statistics'],**{'diagnostic_'+k:v for k,v in m['totals'].items()}})
    matching=pd.DataFrame(rows);checks=matching[matching.complete_window]
    # Fail closed: any unsupported or outside-target essential check blocks.
    accepted=gate_passes(checks)
    output.mkdir(parents=True)
    tables={'matching':matching,'windows':pd.concat(windows,ignore_index=True),'frequencies':pd.concat(freq,ignore_index=True),'run_segments':pd.concat(runs,ignore_index=True),'online_runs':pd.DataFrame(ecology)}
    tables['window_summary']=summarize(tables['windows'],['condition','window'],['token_entropy','conditional_entropy','first_state_fraction','mean_visits','same_token','fraction_states_once'])
    for name,f in tables.items():f.to_csv(output/f'{name}.csv',index=False,float_format='%.12g')
    receipt={'version':'0.7.4','phase':phase,'decision':'primary_allowed' if accepted else 'primary_blocked','checks':len(checks),'failed_checks':int((~checks.within_target).sum()),'gross_failures':int(checks.gross_failure.sum()),'unsupported_checks':int((~checks.supported).sum()),
        'preregistration_sha256':hash_file(root/'preregistration.json'),'frozen_manifest_sha256':hash_file(root/'models/frozen.json'),'inputs':inputs,
        'matching_file':str(output/'matching.csv'),'matching_sha256':hash_file(output/'matching.csv'),'interpretation':'One prespecified family; no override or retuning. Matching is diagnostic, not causal equivalence.'}
    save_json(root/(phase+'_review.json'),receipt);save_json(output/'metadata.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True);return receipt


def gate_passes(checks):
    if not len(checks):return False
    return bool(checks.supported.all() and checks.within_target.all() and checks.difference.map(math.isfinite).all()
                and checks.difference.abs().le(checks.target).all() and not checks.gross_failure.any())


def primary_allowed(root):
    root=Path(root);p=root/'pilot_review.json'
    if not p.exists():return False
    m=json.loads(p.read_text())
    if m.get('decision')!='primary_allowed' or m.get('failed_checks')!=0 or m.get('unsupported_checks')!=0 or m.get('gross_failures')!=0:return False
    if hash_file(root/'preregistration.json')!=m['preregistration_sha256'] or hash_file(root/'models/frozen.json')!=m['frozen_manifest_sha256']:return False
    checks=read_csv(m['matching_file']);checks=checks[checks.complete_window]
    return hash_file(m['matching_file'])==m['matching_sha256'] and gate_passes(checks)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('diagnose','preregister','calibrate','pilot','evaluate','primary'),required=True)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output-root',type=Path,default=ANALYSIS);args=p.parse_args()
    try:
        for path in (args.root,args.output_root):
            for old in ('data/social_memory','data/social_memory_v071','data/social_learning_v072','data/social_learning_v073','analysis/social_memory','analysis/social_memory_v071','analysis/social_learning_v072','analysis/social_learning_v073'):
                if path.resolve().is_relative_to(Path(old).resolve()):raise ValueError('Historical output namespace protected')
        if args.phase=='diagnose':
            from experiments.nonstationary_metrics import diagnose
            diagnose(args.root,args.output_root/'nonstationarity')
        elif args.phase=='preregister':preregister(args.root,args.output_root/'nonstationarity')
        elif args.phase=='calibrate':calibrate(args.root,args.output_root/'calibration')
        elif args.phase=='evaluate':evaluate(args.root,args.output_root/'pilot')
        else:
            ticks=100 if args.phase=='pilot' else 500
            if args.phase=='primary' and not primary_allowed(args.root):raise ValueError('Primary blocked by frozen matching gate')
            for seed in (0,1,2):
                for c in 'DE':
                    for r in (0,1):run_one(args.root,args.phase,c,seed,r,ticks)
            evaluate(args.root,args.output_root/args.phase,args.phase)
    except (ValueError,OSError,KeyError) as error:p.error(str(error))

if __name__=='__main__':main()
