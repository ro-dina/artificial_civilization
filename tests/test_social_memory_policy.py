"""v0.7.1 formation and numerical retention, independent of ecology."""
from dataclasses import replace,FrozenInstanceError
import math
import random
import unittest
from unittest.mock import patch

from agents.social_memory import SocialMemory
from agents.social_memory_policy import SelectiveSocialMemory,RetainedEntry,contact_log,compare_retention,make_memory,log_score
from config import SocialConfig,SocialPolicyConfig


def new_memory(threshold=5,policy='lru',capacity=32,candidates=None,tau=100):
    return make_memory(SocialConfig(True,capacity,20),SocialPolicyConfig(threshold,policy,tau,candidate_capacity=candidates))


def entry(identity,count,last):
    return RetainedEntry(identity,0,last,count,0,1,contact_log(count))


class FormationTests(unittest.TestCase):
    def test_parameter_validation(self):
        for args in (dict(formation_threshold=0),dict(formation_threshold=True),dict(candidate_capacity=0),
                     dict(candidate_capacity=True),dict(eviction_policy='random'),dict(decay_tau=0),
                     dict(decay_tau=float('inf')),dict(decay_tau=float('nan')),dict(strength_mode='linear')):
            with self.assertRaises(ValueError):SocialPolicyConfig(**args)

    def test_legacy_factory_keeps_original_class_and_state_shape(self):
        m=make_memory(SocialConfig(True),SocialPolicyConfig())
        self.assertIs(type(m),SocialMemory)
        self.assertEqual(set(vars(m)),{'capacity','reunion_min_absence_ticks','_entries','_last_observation_tick'})

    def test_immediate_commit_decay(self):
        m=new_memory(1,'contact_decay');u=m.observe((1,),1,capture=True)
        self.assertEqual(len(m),1);self.assertEqual(m.candidates(),())
        self.assertEqual(m.get(1).encounter_count,1);self.assertEqual(u.contacts[0].kind,'commitment')
        self.assertFalse(u.contacts[0].recognized)

    def test_fifth_opportunity_commits_and_initial_strength_includes_all_five(self):
        m=new_memory()
        for t in range(1,6):
            u=m.observe((1,),t,capture=True)
            self.assertEqual(u.known_visible,0)
            if t<5:self.assertEqual(len(m),0);self.assertEqual(m.candidates()[0].progress,t)
        e=m.get(1);self.assertEqual((e.encounter_count,e.consecutive_observations,e.first_seen_tick,e.committed_tick),(5,5,1,5))
        self.assertEqual(m.candidates(),());self.assertEqual(u.contacts[0].kind,'commitment')
        self.assertFalse(u.contacts[0].recognized)
        self.assertTrue(m.observe((1,),6,capture=True).contacts[0].recognized)

    def test_interruption_resets_deletes_candidate(self):
        m=new_memory();m.observe((1,),1);m.observe((1,),2)
        u=m.observe((),3,capture=True)
        self.assertEqual(m.candidates(),());self.assertEqual(u.candidate_events[0].kind,'candidate_reset')
        m.observe((1,),4);self.assertEqual(m.candidates()[0].progress,1)
        for t in range(5,8):m.observe((1,),t)
        self.assertEqual(len(m),0);m.observe((1,),8)
        self.assertEqual(m.get(1).first_seen_tick,4);self.assertEqual(m.get(1).encounter_count,5)

    def test_opportunities_not_global_elapsed_ticks(self):
        m=new_memory()
        for t in (1,5,100,1000,10000):m.observe((1,),t)
        self.assertEqual(m.get(1).encounter_count,5);self.assertEqual(m.get(1).committed_tick,10000)

    def test_candidates_are_not_known_or_reunions(self):
        m=new_memory();m.observe((1,),1);m.observe((),2)
        u=m.observe((1,),30,capture=True)
        self.assertEqual(u.known_visible,0);self.assertEqual(len(m),0)
        self.assertEqual(u.contacts[0].kind,'candidate_contact')

    def test_candidate_capacity_and_id_tie(self):
        m=new_memory(capacity=3,candidates=2)
        u=m.observe((1,2,3),1,capture=True)
        self.assertEqual([c.individual_id for c in m.candidates()],[2,3])
        evicted=[e.candidate.individual_id for e in u.candidate_events if e.kind=='candidate_eviction']
        self.assertEqual(evicted,[1]);self.assertEqual(len(m),0)

    def test_existing_candidates_progress_before_newcomers(self):
        m=new_memory(capacity=2,candidates=1)
        for t in range(1,5):m.observe((9,),t)
        u=m.observe((1,9),5,capture=True)
        self.assertIsNotNone(m.get(9));self.assertEqual(m.candidates()[0].individual_id,1)
        self.assertEqual([c.individual_id for c in u.contacts],[9,1])

    def test_bounded_candidates_and_ltm_in_dense_batches(self):
        m=new_memory(capacity=2,candidates=2)
        for t in range(1,40):
            m.observe(tuple(range(8)),t)
            self.assertLessEqual(len(m),2);self.assertLessEqual(len(m.candidates()),2)
        self.assertFalse(hasattr(m,'ever_seen'))

    def test_long_term_strength_only_actual_visual_sightings(self):
        m=new_memory()
        for t in range(1,6):m.observe((1,),t)
        before=m.get(1)
        m.observe((),6);m.get(1);m.entries();m.candidates()
        self.assertEqual(before,m.get(1))
        u=m.observe((1,),30,capture=True)
        self.assertEqual(m.get(1).encounter_count,6);self.assertEqual(m.get(1).consecutive_observations,1)
        self.assertEqual(u.contacts[0].kind,'reunion')

    def test_forgetting_requires_relearning(self):
        m=new_memory(capacity=1,candidates=2)
        for t in range(1,6):m.observe((1,),t)
        for t in range(6,11):m.observe((2,),t)
        self.assertIsNone(m.get(1))
        for t in range(11,15):
            u=m.observe((1,),t,capture=True);self.assertFalse(u.contacts[0].recognized)
        m.observe((1,),15);self.assertEqual(m.get(1).committed_tick,15);self.assertEqual(m.get(1).encounter_count,5)

    def test_capture_diagnostic_reads_and_rng_independent(self):
        a,b=new_memory(candidates=2),new_memory(candidates=2);rng=random.getstate()
        with patch.object(random.Random,'random',side_effect=AssertionError):
            for t in range(1,20):
                ids=tuple(range(t%4));a.observe(ids,t);b.observe(ids,t,capture=True)
                self.assertEqual(a.entries(),b.entries());self.assertEqual(a.candidates(),b.candidates())
                a.entries();a.candidates();a.get(1)
        self.assertEqual(rng,random.getstate())

    def test_invalid_input_is_atomic_and_no_shared_memory(self):
        a,b=new_memory(),new_memory();a.observe((1,),1)
        before=(a.entries(),a.candidates(),a._last_observation_tick)
        for ids,t in (((1,1),2),((2,1),2),((-1,),2),((True,),2),((1,),1)):
            with self.assertRaises(ValueError):a.observe(ids,t)
            self.assertEqual(before,(a.entries(),a.candidates(),a._last_observation_tick))
        self.assertEqual(b.candidates(),())


class RetentionTests(unittest.TestCase):
    def test_contact_strength_decay_order(self):
        ratio=(100.).as_integer_ratio()
        self.assertLess(compare_retention(entry(1,1,10),entry(2,50,10),ratio),0)
        self.assertLess(compare_retention(entry(1,50,0),entry(2,1,1000),ratio),0)
        self.assertGreater(compare_retention(entry(1,100,9),entry(2,1,10),ratio),0)

    def test_comparator_matches_explicit_formula_in_normal_range(self):
        for tau in (20.,50.,100.,200.):
            ratio=tau.as_integer_ratio()
            for count in (1,5,50):
                for last in (0,9,100):
                    a,b=entry(1,count,last),entry(2,7,20)
                    difference=contact_log(count)-contact_log(7)+(last-20)/tau
                    expected=(difference>0)-(difference<0) or -1
                    self.assertEqual(compare_retention(a,b,ratio),expected)

    def test_deterministic_exact_score_ties_lowest_id(self):
        a,b=entry(2,5,10),entry(1,5,10)
        self.assertGreater(compare_retention(a,b,(100.).as_integer_ratio()),0)

    def test_no_exponential_underflow_or_absolute_tick_overflow(self):
        huge=10**1000
        for tau in (5e-324,100.,1e308):
            a,b=entry(1,100,huge),entry(2,1,huge+1)
            result=compare_retention(a,b,float(tau).as_integer_ratio())
            self.assertEqual(result>0,tau!=5e-324)
        self.assertLess(compare_retention(entry(1,100,0),entry(2,1,huge),(1e308).as_integer_ratio()),0)
        self.assertIsNone(log_score(entry(1,1,0),huge,100.))

    def test_large_contact_counts_and_invalid_strength(self):
        self.assertTrue(math.isfinite(contact_log(10**1000)))
        for n in (0,-1,True):
            with self.assertRaises(ValueError):contact_log(n)

    def test_decay_and_lru_choose_different_residents(self):
        lru,decay=new_memory(2,'lru',2),new_memory(2,'contact_decay',2)
        for m in (lru,decay):
            for t in range(1,11):m.observe((1,),t)
            m.observe((2,),11);m.observe((2,),12)
            m.observe((3,),13);m.observe((3,),14)
        self.assertIsNone(lru.get(1));self.assertIsNotNone(decay.get(1))
        self.assertIsNone(decay.get(2));self.assertIsNotNone(lru.get(2))

    def test_simultaneous_known_refresh_then_commit_and_tied_eviction(self):
        m=new_memory(1,'contact_decay',2)
        m.observe((2,3),1)
        u=m.observe((1,2,3),2,capture=True)
        self.assertEqual([c.individual_id for c in u.contacts],[2,3,1])
        self.assertEqual(u.contacts[-1].evicted.individual_id,2)
        self.assertTrue(u.contacts[0].recognized)


if __name__=='__main__':unittest.main()
