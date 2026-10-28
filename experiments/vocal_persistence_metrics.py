"""Identifiable summaries only: no Q vector reconstruction or stochastic analysis."""
from dataclasses import dataclass
import numpy as np

FIELDS = ('decisions','first','retained','recreated','unique','identified_unique','gap_sum','p_sum','silent',
          'pairs','choice_same','exploit_pairs','exploit_same','explore_pairs','explore_same',
          'both_unique','id_pairs','id_same','set_pairs','set_same','jaccard_sum','tv_pairs','tv_sum',
          'choice_changed','changed_stable','changed_policy','changed_unknown',
          'delta_gap_sum','abs_delta_gap_sum','elapsed_sum','agent_updates_sum','row_updates_sum',
          'delta_p_sum','abs_delta_p_sum','unique_to_tied','tied_to_unique',
          'known_policy_same','known_policy_changed','unknown_policy')
I = {name:i for i,name in enumerate(FIELDS)}
COUNTS = ('decisions','first','retained','recreated','unique','identified_unique','pairs','choice_same',
          'exploit_pairs','exploit_same','explore_pairs','explore_same','both_unique','id_pairs','id_same',
          'set_pairs','set_same','tv_pairs','choice_changed','changed_stable','changed_policy','changed_unknown',
          'unique_to_tied','tied_to_unique','known_policy_same','known_policy_changed','unknown_policy')
RATES = {
 'unique_max_fraction':('unique','decisions'), 'mean_gap':('gap_sum','decisions'),
 'mean_p_silence':('p_sum','decisions'), 'observed_silence_rate':('silent','decisions'),
 'choice_match_rate':('choice_same','pairs'), 'exploit_choice_match_rate':('exploit_same','exploit_pairs'),
 'explore_choice_match_rate':('explore_same','explore_pairs'), 'preferred_action_match_rate':('id_same','id_pairs'),
 'identity_coverage':('id_pairs','pairs'), 'identity_coverage_given_unique':('id_pairs','both_unique'),
 'max_set_coverage':('set_pairs','pairs'), 'mean_max_set_jaccard':('jaccard_sum','set_pairs'),
 'exact_policy_coverage':('tv_pairs','pairs'), 'mean_exact_policy_tv':('tv_sum','tv_pairs'),
 'mean_gap_change':('delta_gap_sum','pairs'), 'mean_abs_gap_change':('abs_delta_gap_sum','pairs'),
 'mean_revisit_interval':('elapsed_sum','pairs'), 'mean_agent_updates_between':('agent_updates_sum','pairs'),
 'mean_state_updates_between':('row_updates_sum','pairs'), 'mean_p_silence_change':('delta_p_sum','pairs'),
 'mean_abs_p_silence_change':('abs_delta_p_sum','pairs'),
 'known_policy_same_fraction':('known_policy_same','pairs'),
 'known_policy_changed_fraction':('known_policy_changed','pairs'),
 'unknown_policy_fraction':('unknown_policy','pairs')}
SUMMARY_COLUMNS = (*COUNTS,*RATES)
SILENCE_CATEGORIES = ('unique_max','tied_max','not_max')


class Bag:
    __slots__=('values',)
    def __init__(self):
        self.values=np.zeros(len(FIELDS),dtype=float)
    def add(self,vector):
        self.values+=vector
    def summary(self):
        result={k:int(self.values[I[k]]) for k in COUNTS}
        for name,(num,den) in RATES.items():
            d=self.values[I[den]]
            result[name]=float(self.values[I[num]]/d) if d else None
        return result


def infer_maxima(d):
    """Exact cardinality + recorded membership facts. Missing sets stay missing.

    No propagation through TD logs, gap guesses, tolerance or imputation.
    """
    n,m=int(d.n_actions),int(d.max_ties)
    if m<1 or m>n or bool(d.all_equal)!=(m==n):
        raise ValueError('Inconsistent maximum tie count')
    if not d.exploratory and d.selected_q!=d.q_max:
        raise ValueError('An exploiting selection must have exactly maximal Q')
    if bool(d.silence_is_max)!=(d.silence_q==d.q_max):
        raise ValueError('SILENCE membership inconsistent with recorded Q')
    if d.selected_q<d.q_min or d.selected_q>d.q_max or d.q_gap<0:
        raise ValueError('Invalid selected Q or preference gap')
    if (m>1 and d.q_gap!=0) or (m==1 and n>1 and d.q_gap<=0):
        raise ValueError('Gap and exact maximum ties disagree')
    inside = 1 if d.silence_is_max else 0
    outside = 0 if d.silence_is_max else 1
    bit=1<<int(d.selected_action)
    if d.selected_q==d.q_max:
        inside |= bit
    else:
        outside |= bit
    if inside&outside or inside.bit_count()>m or outside.bit_count()>n-m:
        raise ValueError('Contradictory recorded maximum membership')
    mask = inside if inside.bit_count()==m else ((1<<n)-1)^outside if outside.bit_count()==n-m else None
    preferred=(mask.bit_length()-1) if m==1 and mask is not None else None
    return preferred,mask


@dataclass(frozen=True,slots=True)
class Encounter:
    tick:int
    age:int
    action:int
    exploratory:bool
    gap:float
    silence_category:str
    p_silence:float
    preferred:int|None
    maxima:int|None
    epoch:int
    state_visits:int
    episode_visit:int
    updates:int
    row_updates:int
    n_actions:int
    epsilon:float
    max_ties:int


def encounter(d,epoch,episode_visit,row_updates):
    preferred,maxima=infer_maxima(d)
    category=('not_max' if not d.silence_is_max else 'unique_max' if d.max_ties==1 else 'tied_max')
    return Encounter(int(d.tick),int(d.age),int(d.selected_action),bool(d.exploratory),float(d.q_gap),
        category,float(d.p_silence),preferred,maxima,epoch,int(d.state_visits),episode_visit,
        int(d.updates),row_updates,int(d.n_actions),float(d.epsilon),int(d.max_ties))


def decision_vector(d,kind,identity_known=None):
    v=np.zeros(len(FIELDS))
    v[I['decisions']]=1;v[I[kind]]=1
    v[I['unique']]=d.max_ties==1
    v[I['identified_unique']]=(infer_maxima(d)[0] is not None) if identity_known is None else identity_known
    v[I['gap_sum']]=d.q_gap;v[I['p_sum']]=d.p_silence;v[I['silent']]=d.selected_action==0
    return v


def pair_vector(a,b,both_unique):
    if b.tick<=a.tick or a.n_actions!=b.n_actions or a.epsilon!=b.epsilon:
        raise ValueError('Consecutive encounters require increasing ticks and identical policy parameters')
    v=np.zeros(len(FIELDS));v[I['pairs']]=1
    same=a.action==b.action
    v[I['choice_same']]=same
    branch='exploit' if not a.exploratory and not b.exploratory else 'explore'
    v[I[branch+'_pairs']]=1;v[I[branch+'_same']]=same
    v[I['both_unique']]=both_unique
    v[I['unique_to_tied']]=a.max_ties==1 and b.max_ties>1
    v[I['tied_to_unique']]=a.max_ties>1 and b.max_ties==1
    if a.preferred is not None and b.preferred is not None:
        v[I['id_pairs']]=1;v[I['id_same']]=a.preferred==b.preferred
    policy_changed=None
    if a.maxima is not None and b.maxima is not None:
        intersection=(a.maxima&b.maxima).bit_count()
        union=(a.maxima|b.maxima).bit_count()
        v[I['set_pairs']]=1;v[I['set_same']]=a.maxima==b.maxima
        v[I['jaccard_sum']]=intersection/union
        tv=(1-a.epsilon)*(1-intersection/max(a.maxima.bit_count(),b.maxima.bit_count()))
        v[I['tv_pairs']]=1;v[I['tv_sum']]=tv
        policy_changed=tv>0
    elif a.epsilon==1:
        v[I['tv_pairs']]=1;policy_changed=False
    elif a.max_ties!=b.max_ties:
        policy_changed=True  # Different set sizes prove a change, not its TV magnitude.
    v[I['unknown_policy' if policy_changed is None else 'known_policy_changed' if policy_changed else 'known_policy_same']]=1
    if not same:
        v[I['choice_changed']]=1
        v[I['changed_unknown' if policy_changed is None else 'changed_policy' if policy_changed else 'changed_stable']]=1
    v[I['delta_gap_sum']]=b.gap-a.gap;v[I['abs_delta_gap_sum']]=abs(b.gap-a.gap)
    v[I['elapsed_sum']]=b.tick-a.tick
    v[I['agent_updates_sum']]=b.updates-a.updates
    v[I['row_updates_sum']]=b.row_updates-a.row_updates
    v[I['delta_p_sum']]=b.p_silence-a.p_silence
    v[I['abs_delta_p_sum']]=abs(b.p_silence-a.p_silence)
    return v
