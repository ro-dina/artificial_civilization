"""Held-out observational reward prediction on the SAME fixed B/C experience."""
from collections import Counter,defaultdict,OrderedDict
import hashlib,json,math

from experiments.controls_metrics import original_frame,read_csv,attach_nuisance
from experiments.information_controls import model_file
from simulation.controls_observer import C_PAIRS


def reward_bin(reward):return 0 if reward<=-.1 else 1 if reward<0 else 2


class Predictor:
    def __init__(self,minimum=20,strength=5):
        self.minimum=minimum;self.strength=strength;self.cells=defaultdict(Counter);self.actions=defaultdict(Counter)
    def add(self,key,action,reward):
        r=reward_bin(reward);self.cells[key,action][r]+=1;self.actions[action][r]+=1
    def predict(self,key,action):
        prior=self.actions[action];n=sum(prior.values());p=[(prior[i]+1)/(n+3) for i in range(3)]
        cell=self.cells.get((key,action),{});m=sum(cell.values());supported=m>=self.minimum
        return ([(cell.get(i,0)+self.strength*p[i])/(m+self.strength) for i in range(3)] if supported else p),supported,m


def deep_tuple(value):return tuple(map(deep_tuple,value)) if isinstance(value,list) else value


def records(path,model,nseed):
    m,files,f=original_frame(path);catalog={r.base_id:deep_tuple(json.loads(r.key_json)) for r in read_csv(files['states']).itertuples(index=False)}
    social=f.b_token.tolist() if m['condition']=='B' else f.c_token.tolist()
    nuisance=attach_nuisance(f,model,nseed)
    return m,f,[(catalog[x],s,u) for x,s,u in zip(f.base_id,social,nuisance)]


def prediction(root,common_support=False):
    rows=[]
    for condition,control in (('B','D'),('C','E')):
        model=model_file(root/'models'/f'{control}.json');heads={k:Predictor() for k in ('X_A','X_social_A','X_nuisance_A')}
        train_agents=0;train_decisions=0
        for seed in (3,4):
            m,f,keys=records(root/'calibration'/f'{condition}_seed_{seed}_r0.csv.gz.metadata.json',model,90000+seed)
            train_agents+=f.agent_id.nunique();train_decisions+=len(f)
            for (x,s,u),(a,r) in zip(keys,f[['action','reward']].itertuples(index=False,name=None)):
                for name,key in (('X_A',x),('X_social_A',(x,s)),('X_nuisance_A',(x,u))):heads[name].add(key,a,r)
        for seed in (0,1,2):
            m,f,keys=records(root/'references'/f'{condition}_seed_{seed}_r0.csv.gz.metadata.json',model,90000+seed)
            # Optional descriptive subset: identical test decisions supported by
            # all three predictors. No reward enters eligibility or tuning.
            common=[all(sum(heads[name].cells.get((key,a),{}).values())>=heads[name].minimum
                        for name,key in (('X_A',x),('X_social_A',(x,s)),('X_nuisance_A',(x,u))))
                    for (x,s,u),a in zip(keys,f.action)] if common_support else None
            for name,head in heads.items():
                losses=[];briers=[];support=0;agents=defaultdict(list);support_n=0
                for (x,s,u),(identity,a,r) in zip(keys,f[['agent_id','action','reward']].itertuples(index=False,name=None)):
                    key=x if name=='X_A' else (x,s) if name=='X_social_A' else (x,u)
                    p,ok,n=head.predict(key,a);target=reward_bin(r)
                    loss=-math.log(p[target]);losses.append(loss);briers.append(sum((p[i]-(i==target))**2 for i in range(3)))
                    support+=ok;support_n+=n;agents[identity].append(loss)
                rows.append({'source_condition':condition,'seed':seed,'model':name,'log_loss':sum(losses)/len(losses),
                    'brier':sum(briers)/len(briers),'equal_agent_log_loss':sum(sum(v)/len(v) for v in agents.values())/len(agents),
                    'supported_decisions':support,'decisions':len(losses),'agents':len(agents),'support_fraction':support/len(losses),
                    'train_agents':train_agents,'train_decisions':train_decisions,'train_seeds':'3;4','nuisance_seed':90000+seed,
                    'outcome':'actual immediate physiological reward bins <=-.1,(-.1,0),>=0','action_conditioned':True})
                if common_support:
                    rows[-1]['subset']='all'
                    mask=[i for i,ok in enumerate(common) if ok];by_agent=defaultdict(list)
                    for i in mask:by_agent[int(f.agent_id.iloc[i])].append(losses[i])
                    rows.append({**rows[-1],'subset':'common_supported','log_loss':sum(losses[i] for i in mask)/len(mask) if mask else None,
                        'brier':sum(briers[i] for i in mask)/len(mask) if mask else None,
                        'equal_agent_log_loss':sum(sum(v)/len(v) for v in by_agent.values())/len(by_agent) if by_agent else None,
                        'decisions':len(mask),'agents':len(by_agent),'supported_decisions':len(mask),'support_fraction':1. if mask else None,
                        'test_decisions_before_filter':len(f),'common_support_fraction':len(mask)/len(f)})
    return rows


def replay(path,representation,model=None,nseed=0,*,diagnostics=False):
    """Same recorded actions/rewards; no environment execution or policy value claim.

    Bootstrap is read BEFORE previous-row recreation. At cutoff the pending TD
    transition remains unupdated. No next observation is invented.
    """
    if representation not in ('A','B','C','D','E','B_Umemory'):raise ValueError('Unknown replay representation')
    m,files,f=original_frame(path)
    catalog={r.base_id:deep_tuple(json.loads(r.key_json)) for r in read_csv(files['states']).itertuples(index=False)}
    cfg=m['config']['simulation'];capacity=cfg['learning_memory_capacity'];lr=cfg['learning_rate'];discount=cfg['learning_discount'];actions=8 if cfg['reproduction_enabled'] else 7
    if representation in ('D','E','B_Umemory') and model is None:raise ValueError('Nuisance replay needs frozen model')
    tokens=(f.used_token.tolist() if representation==m['condition'] else attach_nuisance(f,model,nseed)) if representation in ('D','E','B_Umemory') else None
    tables={};pending={};histories={};visit_hist=Counter();updates=evictions=allocations=distinct=0
    row_meta={};lifetime_updates={};update_hist=Counter()
    def remember(q,key,identity):
        nonlocal evictions,allocations
        if key not in q:
            if len(q)==capacity:
                evicted,_=q.popitem(last=False);evictions+=1
                if diagnostics:row_meta[identity].pop(evicted,None)
            q[key]=[0.]*actions;allocations+=1
            if diagnostics:row_meta[identity][key]=[0,0]
        q.move_to_end(key);return q[key]
    def update(q,p,next_value,identity):
        nonlocal updates
        key,a,r=p;values=remember(q,key,identity);values[a]+=lr*(r+discount*next_value-values[a]);updates+=1
        if diagnostics:row_meta[identity][key][1]+=1;lifetime_updates[identity][key]+=1
    for i,r in enumerate(f.itertuples(index=False)):
        x=catalog[r.base_id]
        key=x if representation=='A' else (*x,r.b_token) if representation=='B' else (*x,*C_PAIRS[r.c_token]) if representation=='C' else (*x,r.b_token,C_PAIRS[tokens[i]][1]) if representation=='B_Umemory' else (*x,tokens[i])
        q=tables.setdefault(r.agent_id,OrderedDict());visits=histories.setdefault(r.agent_id,Counter())
        if diagnostics:row_meta.setdefault(r.agent_id,{});lifetime_updates.setdefault(r.agent_id,Counter())
        if key not in visits:distinct+=1
        visits[key]+=1
        if r.agent_id in pending:update(q,pending.pop(r.agent_id),max(q.get(key,(0.,)*actions)),r.agent_id)
        remember(q,key,r.agent_id);p=(key,r.action,r.reward)
        if diagnostics:row_meta[r.agent_id][key][0]+=1
        if r.terminal:
            update(q,p,0.,r.agent_id);tables.pop(r.agent_id);visit_hist.update(visits.values());histories.pop(r.agent_id)
            if diagnostics:
                update_hist.update(lifetime_updates[r.agent_id][k] for k in visits)
                lifetime_updates.pop(r.agent_id);row_meta.pop(r.agent_id)
        else:pending[r.agent_id]=p
    for visits in histories.values():visit_hist.update(visits.values())
    rows=[(identity,tuple((k,tuple(v)) for k,v in tables.get(identity,{}).items()),pending.get(identity)) for identity in m['living_ids']]
    sha=hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()
    if representation==m['condition'] and sha!=m['physical_q_sha256']:raise RuntimeError('Fixed-experience replay differs from actual Q/LRU/pending state')
    result={'source_condition':m['condition'],'seed':m['seed'],'representation':representation,'decisions':len(f),'distinct_agent_states':distinct,
            'first_state_fraction':distinct/len(f),'mean_visits':len(f)/distinct,'states_once_fraction':visit_hist[1]/distinct,
            'updates':updates,'evictions':evictions,'allocations':allocations,'pending_cutoff':len(pending),'physical_q_sha256':sha,
            'exact_source_reconstruction':representation==m['condition'],'nuisance_seed':nseed if representation in ('D','E','B_Umemory') else None}
    if diagnostics:
        from types import SimpleNamespace
        from simulation.social_learning_observer import probe,policy
        agents=[];retained=Counter();ranges=[];preferences=[0.]*actions
        for identity in m['living_ids']:
            q=tables.get(identity,{});history=histories.get(identity,{})
            update_hist.update(lifetime_updates.get(identity,{}).get(k,0) for k in history)
            h={k:[v,*row_meta[identity].get(k,(0,0))] for k,v in history.items()}
            brain=SimpleNamespace(_values=q,epsilon=cfg['learning_epsilon'])
            agents.append((SimpleNamespace(id=identity,brain=brain),h))
            for key,values in q.items():
                visits,td=row_meta[identity][key];retained[visits,td]+=1;ranges.append(max(values)-min(values))
                for j,p in enumerate(policy(values,cfg['learning_epsilon'])):preferences[j]+=p
        n=sum(retained.values());mode='none' if representation=='A' else 'memory' if representation in ('C','B_Umemory') else 'perception'
        supported=[probe(agents,mode,m['tick'],minimum,kind) for minimum in (2,5) for kind in ('all_social','memory_only')]
        living_keys=sum(len(h) for h in histories.values())
        result.update(retained_rows=n,living_observed_agent_states=living_keys,retention_fraction=n/living_keys if living_keys else None,
            mean_retained_q_range=sum(ranges)/n if n else None,nonflat_retained_fraction=sum(v>0 for v in ranges)/n if n else None,
            mean_rows_per_living_agent=n/len(m['living_ids']) if m['living_ids'] else None,
            **{f'retained_policy_action_{j}':v/n if n else None for j,v in enumerate(preferences)},
            diagnostics={'probes':supported,'state_visits':[{'visits':k,'states':v} for k,v in sorted(visit_hist.items())],
                'state_updates':[{'updates':k,'states':v} for k,v in sorted(update_hist.items())],
                'retained':[{'visits':k[0],'updates':k[1],'rows':v} for k,v in sorted(retained.items())]})
    return result
