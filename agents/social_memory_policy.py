"""Experimental bounded formation/retention; no reward, brain, death or RNG access."""
from collections import OrderedDict
from dataclasses import dataclass,replace
import math

from agents.social_memory import SocialMemory,SocialEntry
from config import SocialPolicyConfig


def contact_log(count):
    if type(count) is not int or count <= 0:
        raise ValueError('Contact strength requires positive observed opportunities')
    try:value=math.log1p(count)
    except OverflowError:value=math.log(count)
    return math.log(value)


@dataclass(frozen=True,slots=True)
class RetainedEntry(SocialEntry):
    committed_tick: int
    consecutive_observations: int
    log_strength: float


@dataclass(frozen=True,slots=True)
class Candidate:
    individual_id: int
    first_seen_tick: int
    last_seen_tick: int
    progress: int


@dataclass(frozen=True,slots=True)
class CandidateEvent:
    kind: str
    candidate: Candidate


@dataclass(frozen=True,slots=True)
class PolicyContact:
    individual_id: int
    kind: str
    recognized: bool
    entry: RetainedEntry | None
    progress: int
    elapsed_since_last_seen: int | None
    evicted: RetainedEntry | None


@dataclass(frozen=True,slots=True)
class PolicyUpdate:
    contacts: tuple[PolicyContact,...]
    candidate_events: tuple[CandidateEvent,...]
    known_visible: int
    unknown_visible: int
    occupancy: int
    candidate_occupancy: int


def compare_retention(a,b,tau_ratio):
    """Ascending score, lowest ID exact tie; no exp or absolute-clock division.

    Compare (ta-tb) + tau*(log_strength_a-log_strength_b). Opposing
    terms use integer cross products of the evaluated floats' exact ratios.
    This avoids both exponential underflow and tiny-tau division overflow.
    """
    dt=a.last_seen_tick-b.last_seen_tick
    if dt==0:
        result=(a.encounter_count>b.encounter_count)-(a.encounter_count<b.encounter_count)
    else:
        dl=a.log_strength-b.log_strength
        if dl==0 or (dl>0)==(dt>0):result=1 if dt>0 else -1
        else:
            ln,ld=abs(dl).as_integer_ratio();tn,td=tau_ratio
            time=abs(dt)*td*ld;strength=tn*ln
            result=((time>strength)-(time<strength))*(1 if dt>0 else -1)
    return result or ((a.individual_id>b.individual_id)-(a.individual_id<b.individual_id))


def log_score(entry,tick,tau):
    """Finite diagnostic when representable; components remain exact if absent."""
    try:score=entry.log_strength-(tick-entry.last_seen_tick)/tau
    except OverflowError:return None
    return score if math.isfinite(score) else None


class SelectiveSocialMemory(SocialMemory):
    """Known committed hits, continuing candidates, then new candidates, by ID.

    Candidate capacity and long-term capacity are separate bounds. Missing a
    candidate at an opportunity deletes it; no unseen/lifetime registry exists.
    """
    def __init__(self,capacity=32,reunion_min_absence_ticks=20,policy=None):
        super().__init__(capacity,reunion_min_absence_ticks)
        self.policy=policy if policy is not None else SocialPolicyConfig(formation_threshold=5)
        if not isinstance(self.policy,SocialPolicyConfig):raise ValueError('Expected SocialPolicyConfig')
        self.candidate_capacity=self.policy.resolved_candidate_capacity(capacity)
        self._candidates=OrderedDict()
        self._tau_ratio=float(self.policy.decay_tau).as_integer_ratio()

    def candidates(self):
        return tuple(self._candidates.values())

    def _victim(self):
        if self.policy.eviction_policy=='lru':return next(iter(self._entries))
        iterator=iter(self._entries.values());best=next(iterator)
        for candidate in iterator:
            if compare_retention(candidate,best,self._tau_ratio)<0:best=candidate
        return best.individual_id

    def observe(self,individual_ids,tick,*,capture=False):
        if (type(tick) is not int or tick<0 or self._last_observation_tick is not None and tick<=self._last_observation_tick):
            raise ValueError('Social observation ticks must be strictly increasing nonnegative integers')
        if (not isinstance(individual_ids,tuple) or any(type(i) is not int or i<0 for i in individual_ids)
                or any(a>=b for a,b in zip(individual_ids,individual_ids[1:]))):
            raise ValueError('Visible identities must be a strictly increasing immutable tuple')
        visible=set(individual_ids)
        contacts=[] if capture else None;events=[] if capture else None
        for identity in sorted(set(self._candidates)-visible):
            candidate=self._candidates.pop(identity)
            if capture:events.append(CandidateEvent('candidate_reset',candidate))
        known=[(i,self._entries[i]) for i in individual_ids if i in self._entries]
        unknown=[i for i in individual_ids if i not in self._entries]
        continuing=[i for i in unknown if i in self._candidates]
        new=[i for i in unknown if i not in self._candidates]
        for identity,old in known:
            continuous=old.last_seen_tick==self._last_observation_tick
            count=old.encounter_count+1;elapsed=tick-old.last_seen_tick
            entry=replace(old,last_seen_tick=tick,encounter_count=count,
                          consecutive_observations=old.consecutive_observations+1 if continuous else 1,
                          log_strength=contact_log(count))
            self._entries[identity]=entry;self._entries.move_to_end(identity)
            if capture:
                kind='continuous_contact' if continuous else 'reunion' if elapsed>=self.reunion_min_absence_ticks else 'return_contact'
                contacts.append(PolicyContact(identity,kind,True,entry,entry.consecutive_observations,elapsed,None))
        for identity in (*continuing,*new):
            if self.policy.formation_threshold==1:
                candidate=Candidate(identity,tick,tick,1)
            elif identity in self._candidates:
                old=self._candidates[identity]
                candidate=replace(old,last_seen_tick=tick,progress=old.progress+1)
                self._candidates[identity]=candidate;self._candidates.move_to_end(identity)
                if capture:events.append(CandidateEvent('candidate_progress',candidate))
            else:
                if len(self._candidates)==self.candidate_capacity:
                    victim=min(self._candidates.values(),key=lambda c:(c.last_seen_tick,c.individual_id))
                    del self._candidates[victim.individual_id]
                    if capture:events.append(CandidateEvent('candidate_eviction',victim))
                candidate=Candidate(identity,tick,tick,1);self._candidates[identity]=candidate
                if capture:events.append(CandidateEvent('candidate_created',candidate))
            if candidate.progress>=self.policy.formation_threshold:
                self._candidates.pop(identity,None)
                if capture and self.policy.formation_threshold>1:events.append(CandidateEvent('candidate_committed',candidate))
                entry=RetainedEntry(identity,candidate.first_seen_tick,tick,candidate.progress,tick,candidate.progress,contact_log(candidate.progress))
                evicted=self._entries.pop(self._victim()) if len(self)==self.capacity else None
                self._entries[identity]=entry
                if capture:contacts.append(PolicyContact(identity,'commitment',False,entry,candidate.progress,None,evicted))
            elif capture:contacts.append(PolicyContact(identity,'candidate_contact',False,None,candidate.progress,None,None))
        self._last_observation_tick=tick
        return PolicyUpdate(tuple(contacts),tuple(events),len(known),len(unknown),len(self),len(self._candidates)) if capture else None


def make_memory(social,policy=None):
    if policy is not None and not isinstance(policy,SocialPolicyConfig):raise ValueError('Expected SocialPolicyConfig')
    if policy is None or policy.legacy:
        return SocialMemory(social.social_memory_capacity,social.social_reunion_min_absence_ticks)
    return SelectiveSocialMemory(social.social_memory_capacity,social.social_reunion_min_absence_ticks,policy)
