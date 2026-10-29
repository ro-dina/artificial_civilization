"""Private bounded knowledge, sequential visual geometry and ecological isolation."""
from dataclasses import asdict, fields, replace, FrozenInstanceError
import hashlib
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from agents.brain import Action, Decision
from agents.genome import Genome
from agents.human import Human, SocialHuman
from agents.learning import LearningBrain, Physiology
from agents.observation import (Observation, SocialObservation, VisibleIndividual, HeardSignal,
    AuditoryPercept, SourceDirection)
from agents.social_memory import SocialMemory, SocialEntry
from config import Config, SocialConfig
from simulation.simulation import Simulation
from simulation.social_observer import SocialObserver
from tests.auditory_control import ActionDigest
from tests.test_statistics import FixedBrain
from experiments.social_verification import state, digest


def memory_state(memory):
    return (memory.capacity,memory.reunion_min_absence_ticks,memory._last_observation_tick,memory.entries())


def quiet(**changes):
    defaults=dict(world_width=6,world_height=1,initial_population=2,initial_food=0,initial_water=0,
                  hunger_per_tick=0,thirst_per_tick=0,max_age=None)
    defaults.update(changes)
    return Config(**defaults)


def simulation(config=None, settings=None, sink=None, factory=None):
    return Simulation(config or quiet(),social_config=settings or SocialConfig(True),social_sink=sink,
                      brain_factory=factory or (lambda rng,cfg:FixedBrain()))


class MemoryTests(unittest.TestCase):
    def test_independent_empty_memories(self):
        a,b=SocialMemory(),SocialMemory();a.observe((3,),1)
        self.assertEqual(len(a),1);self.assertEqual(len(b),0)
        self.assertEqual(a.entries(),(SocialEntry(3,1,1,1),))

    def test_first_sighting_and_survival_confirmation(self):
        m=SocialMemory();c=m.observe((1,),1,capture=True).contacts[0]
        self.assertEqual(c.kind,'first_encounter');self.assertIsNone(c.elapsed_since_last_seen)
        self.assertEqual(c.entry.last_confirmed_alive_tick,1)
        self.assertEqual(set(asdict(c.entry)),{'individual_id','first_seen_tick','last_seen_tick','encounter_count'})

    def test_continuous_contact_even_threshold_one(self):
        m=SocialMemory(3,1)
        for t in range(1,5):
            c=m.observe((1,),t,capture=True).contacts[0]
            self.assertEqual(c.kind,'first_encounter' if t==1 else 'continuous_contact')
        self.assertEqual(m.get(1).encounter_count,4)

    def test_consecutive_opportunities_do_not_assert_an_unobserved_absence(self):
        m=SocialMemory(2,20);m.observe((1,),1)
        self.assertEqual(m.observe((1,),100,capture=True).contacts[0].kind,'continuous_contact')

    def test_one_tick_absence_is_return_not_automatic_reunion(self):
        m=SocialMemory();m.observe((1,),1);m.observe((),2)
        self.assertEqual(m.observe((1,),3,capture=True).contacts[0].kind,'return_contact')

    def test_reunion_just_below_threshold(self):
        m=SocialMemory();m.observe((1,),1);m.observe((),2)
        self.assertEqual(m.observe((1,),20,capture=True).contacts[0].kind,'return_contact')

    def test_reunion_exactly_threshold_inclusive(self):
        m=SocialMemory();m.observe((1,),1);m.observe((),2)
        c=m.observe((1,),21,capture=True).contacts[0]
        self.assertEqual(c.kind,'reunion');self.assertEqual(c.elapsed_since_last_seen,20)

    def test_reunion_above_threshold(self):
        m=SocialMemory();m.observe((1,),1);m.observe((),2)
        self.assertEqual(m.observe((1,),22,capture=True).contacts[0].kind,'reunion')

    def test_multiple_contacts_capacity_and_exact_lru_eviction(self):
        m=SocialMemory(2);m.observe((1,2),1)
        u=m.observe((1,3),2,capture=True)
        self.assertEqual(tuple(e.individual_id for e in m.entries()),(1,3))
        self.assertEqual(u.contacts[-1].evicted.individual_id,2)
        self.assertEqual(len(m),2)

    def test_known_batch_membership_is_frozen_before_allocations(self):
        m=SocialMemory(2);m.observe((10,20),1)
        u=m.observe((1,20),2,capture=True)
        self.assertEqual((u.known_visible,u.unknown_visible),(1,1))
        self.assertEqual([c.entry.individual_id for c in u.contacts],[20,1])
        self.assertEqual(u.contacts[0].kind,'continuous_contact')
        self.assertEqual(u.contacts[1].evicted.individual_id,10)

    def test_evicted_identity_becomes_unknown_no_hidden_history(self):
        m=SocialMemory(1);m.observe((1,),1);m.observe((2,),2)
        self.assertIsNone(m.get(1));c=m.observe((1,),3,capture=True).contacts[0]
        self.assertEqual(c.kind,'first_encounter');self.assertEqual(c.entry,SocialEntry(1,3,3,1))
        self.assertEqual(set(vars(m)),{'capacity','reunion_min_absence_ticks','_entries','_last_observation_tick'})

    def test_diagnostic_reads_do_not_refresh_or_draw_randomness(self):
        m=SocialMemory(2);m.observe((1,2),1);before=memory_state(m);rng=random.getstate()
        with patch.object(random.Random,'random',side_effect=AssertionError):
            for _ in range(5):self.assertEqual(m.get(1).individual_id,1);m.entries();len(m)
        self.assertEqual(before,memory_state(m));self.assertEqual(rng,random.getstate())
        with self.assertRaises(FrozenInstanceError):m.entries()[0].last_seen_tick=7

    def test_capture_does_not_change_memory_updates(self):
        a,b=SocialMemory(2),SocialMemory(2)
        for t,ids in enumerate(((1,2),(1,3),(),(1,2,3)),1):
            a.observe(ids,t,capture=False);b.observe(ids,t,capture=True)
            self.assertEqual(memory_state(a),memory_state(b))

    def test_validation_is_atomic(self):
        m=SocialMemory();m.observe((1,),1);before=memory_state(m)
        for ids,t in (((1,1),2),((2,1),2),((-1,),2),((True,),2),([1],2),((1,),1),((1,),-1)):
            with self.assertRaises(ValueError):m.observe(ids,t)
            self.assertEqual(before,memory_state(m))

    def test_positive_settings_validation_and_legacy_config_shape(self):
        for args in ((0,20),(1,0),(True,20),(2,False)):
            with self.assertRaises(ValueError):SocialMemory(*args)
        for kwargs in (dict(social_memory_enabled=1),dict(social_memory_capacity=0),dict(social_reunion_min_absence_ticks=-1)):
            with self.assertRaises(ValueError):SocialConfig(**kwargs)
        self.assertFalse(SocialConfig().social_memory_enabled)
        self.assertEqual(SocialConfig().social_memory_capacity,32)
        self.assertEqual(SocialConfig().social_reunion_min_absence_ticks,20)
        self.assertNotIn('social_memory_enabled',asdict(Config()))


class VisualRecognitionTests(unittest.TestCase):
    def test_disabled_keeps_exact_legacy_types_and_fields(self):
        sim=Simulation(quiet());h=sim.agents[0]
        self.assertIs(type(h),Human);self.assertIs(type(sim.observe(h)),Observation)
        self.assertFalse(hasattr(h,'social_memory'));self.assertIsNone(sim._social_index)
        self.assertIsNone(sim.social_observer)

    def test_only_primitive_immutable_visual_record(self):
        record=VisibleIndividual(2,-1,0)
        self.assertEqual(asdict(record),dict(individual_id=2,dx=-1,dy=0))
        with self.assertRaises(FrozenInstanceError):record.dx=5
        for args in ((-1,0,0),(True,0,0),(1,.5,0)):
            with self.assertRaises(ValueError):VisibleIndividual(*args)
        with self.assertRaises(ValueError):SocialObservation(1,(),individuals=[record])
        with self.assertRaises(ValueError):SocialObservation(1,(),individuals=(record,record))

    def test_inclusive_chebyshev_square_and_relative_position(self):
        sim=simulation(quiet(world_width=8,world_height=8,initial_population=4,perception_radius=2))
        for h,pos in zip(sim.agents,((3,3),(5,5),(6,3),(1,3))):h.x,h.y=pos
        obs=sim.observe(sim.agents[0])
        self.assertEqual(obs.individuals,(VisibleIndividual(1,2,2),VisibleIndividual(3,-2,0)))
        self.assertEqual(len(obs.tiles),25)

    def test_zero_radius_sees_only_same_cell_excludes_self_and_dead(self):
        sim=simulation(quiet(initial_population=4,perception_radius=0))
        for h,x in zip(sim.agents,(1,1,2,1)):h.x,h.y=x,0
        sim.agents[3].alive=False
        self.assertEqual(sim.observe(sim.agents[0]).individuals,(VisibleIndividual(1,0,0),))

    def test_different_inherited_visual_radii(self):
        sim=simulation();sim.agents[0].x=0;sim.agents[1].x=2
        sim.agents[0].genome=replace(sim.agents[0].genome,perception_radius=0)
        sim.agents[1].genome=replace(sim.agents[1].genome,perception_radius=2)
        self.assertEqual(sim.observe(sim.agents[0]).individuals,())
        self.assertEqual(sim.observe(sim.agents[1]).individuals,(VisibleIndividual(0,-2,0),))

    def test_diagnostic_observe_does_not_update_memory_or_persistent_index(self):
        sim=simulation();before=state(sim,include_social=True);buckets=sim.reproduction._buckets.copy()
        for _ in range(3):sim.observe(sim.agents[0])
        self.assertEqual(before,state(sim,include_social=True));self.assertEqual(buckets,sim.reproduction._buckets)

    def test_deterministic_sorted_visible_order(self):
        sim=simulation(quiet(world_width=1,initial_population=4))
        sim.agents.reverse()
        self.assertEqual([r.individual_id for r in sim.observe(sim.agents[-1]).individuals],[1,2,3])

    def test_sequential_movement_both_spatial_paths(self):
        for reproduction in (False,True):
            captured={}
            class Brain:
                def choose_action(self,h,o):
                    captured[h.id]=o
                    return Decision(Action.MOVE_EAST if h.id==0 else Action.WAIT)
            sim=simulation(quiet(world_width=4,perception_radius=1,reproduction_enabled=reproduction),factory=lambda r,c:Brain())
            sim.agents[0].x=0;sim.agents[1].x=2
            with patch.object(sim._order_rng,'shuffle',lambda x:None):sim.step()
            self.assertEqual(captured[1].individuals,(VisibleIndividual(0,-1,0),))
            self.assertEqual(sim._social_index is None,reproduction)

    def test_earlier_death_is_not_visible_to_later_actor(self):
        captured={}
        class Brain:
            def choose_action(self,h,o):captured[h.id]=o;return Decision(Action.WAIT)
        sim=simulation(quiet(world_width=1,max_age=1),factory=lambda r,c:Brain())
        with patch.object(sim._order_rng,'shuffle',lambda x:None):sim.step()
        self.assertEqual(captured[0].individuals,(VisibleIndividual(1,0,0),))
        self.assertEqual(captured[1].individuals,());self.assertEqual(sim.population,0)

    def test_founders_and_newborns_empty_independent_not_inherited(self):
        sim=simulation(quiet(world_width=1,reproduction_enabled=True,min_reproductive_age=0),factory=lambda r,c:FixedBrain(Action.REPRODUCE))
        a,b=sim.agents;a.reproductive_type,b.reproductive_type=0,1
        self.assertEqual([len(h.social_memory) for h in sim.agents],[0,0])
        sim.step();child=sim.agents[-1]
        self.assertEqual(child.birth_tick,1);self.assertEqual(child.id,2);self.assertEqual(len(child.social_memory),0)
        self.assertNotIn(child.id,[e.individual_id for e in a.social_memory.entries()])
        self.assertIsNot(a.social_memory,child.social_memory);self.assertIsNot(b.social_memory,child.social_memory)
        sim.step();self.assertTrue(a.social_memory.get(2));self.assertGreater(len(child.social_memory),0)

    def test_remote_death_never_updates_other_agents_knowledge(self):
        rows=[];sim=simulation(quiet(perception_radius=0),sink=rows.append)
        a,b=sim.agents;a.x=b.x=0;sim.step()
        remembered=a.social_memory.get(b.id)
        b.x=5;b.thirst=sim.config.max_thirst
        sim.step()
        self.assertFalse(b.alive);self.assertEqual(a.social_memory.get(b.id),remembered)
        self.assertTrue(any(r.event_kind=='death' and r.observer_id==b.id for r in rows))
        self.assertFalse(hasattr(a.social_memory.get(b.id),'actual_death_tick'))

    def test_hearing_alone_creates_no_social_knowledge(self):
        sim=simulation(quiet(initial_population=1,perception_radius=0))
        h=sim.agents[0];h.x=0
        sim.communication.emit(7,99,5,0);sim.communication.advance()
        self.assertIsNotNone(sim.observe(h).auditory.signal)
        sim.step();self.assertEqual(len(h.social_memory),0)
        self.assertEqual({f.name for f in fields(HeardSignal)},{'signal_id','source_direction'})
        self.assertEqual({f.name for f in fields(AuditoryPercept)},{'kind','signal'})


class SocialIsolationTests(unittest.TestCase):
    def compare(self,cfg,ticks=30):
        rows=[];a_digest,b_digest,c_digest=ActionDigest(),ActionDigest(),ActionDigest()
        off=Simulation(cfg,event_sink=a_digest)
        on=Simulation(cfg,social_config=SocialConfig(True,3,4),event_sink=b_digest)
        recorded=Simulation(cfg,social_config=SocialConfig(True,3,4),social_sink=rows.append,event_sink=c_digest)
        rng=random.getstate()
        for tick in range(ticks+1):
            if tick:off.step();on.step();recorded.step()
            self.assertEqual(state(off),state(on));self.assertEqual(state(on,include_social=True),state(recorded,include_social=True))
            self.assertEqual(a_digest.digest.digest(),b_digest.digest.digest())
            self.assertEqual(b_digest.digest.digest(),c_digest.digest.digest())
            for a,b in zip(off.agents,on.agents):
                x,y=off.observe(a),on.observe(b)
                self.assertEqual((x.tick,x.tiles,x.auditory),(y.tick,y.tiles,y.auditory))
            self.assertEqual(rng,random.getstate())
        self.assertTrue(rows)
        self.assertTrue(all(len(h.social_memory)<=3 for h in on.agents))
        self.assertTrue(all(len(v)<=3 for v in recorded.social_observer._evicted.values()))
        return off,on,recorded

    def test_random_brain_ecological_rng_and_recording_isolation(self):
        self.compare(Config.evolution(random_seed=7,initial_population=12),25)

    def test_learning_listening_on_ecological_q_lru_rng_isolation(self):
        self.compare(Config.evolution(random_seed=7,initial_population=12,brain='learning'),25)

    def test_learning_listening_off_ecological_q_lru_rng_isolation(self):
        self.compare(Config.evolution(random_seed=7,initial_population=12,brain='learning',learning_uses_auditory=False),25)

    def test_random_emission_learning_control_also_isolated(self):
        self.compare(Config.evolution(random_seed=7,initial_population=8,brain='learning',learning_controls_vocalization=False),15)

    def test_legacy_finite_extinction_is_still_identical(self):
        off,on,recorded=self.compare(Config(initial_population=3,max_age=2),4)
        self.assertEqual(off.population,0);count=len(recorded.social_observer._evicted)
        self.assertEqual(count,0)

    def test_social_features_never_enter_q_keys_or_rewards(self):
        sim=Simulation(quiet(brain='learning'),social_config=SocialConfig(True));h=sim.agents[0]
        normal=Observation(1,(),sim.observe(h).auditory)
        social=SocialObservation(1,(),normal.auditory,(VisibleIndividual(99,0,0),))
        body=Physiology.capture(h)
        self.assertEqual(h.brain.state(body,normal),h.brain.state(body,social))
        original_reward=Path('agents/learning.py').read_text()
        self.assertNotIn('social_memory',original_reward);self.assertNotIn('individuals',original_reward)

    def test_observer_does_not_call_memory_or_rng_and_has_bounded_diagnostic_cache(self):
        rows=[];sim=simulation(quiet(world_width=1),sink=rows.append)
        h=sim.agents[0];observation=sim.observe(h)
        update=h.social_memory.observe(tuple(i.individual_id for i in observation.individuals),1,capture=True)
        before=memory_state(h.social_memory);rng=random.getstate()
        with patch.object(SocialMemory,'observe',side_effect=AssertionError),patch.object(random.Random,'random',side_effect=AssertionError):
            sim.social_observer.observation(1,h,update,observation)
        self.assertEqual(before,memory_state(h.social_memory));self.assertEqual(rng,random.getstate())
        self.assertFalse(any(isinstance(v,Human) for v in vars(sim.social_observer).values()))

    def test_recording_off_allocates_no_detailed_contact_payloads(self):
        sim=simulation(quiet(world_width=1))
        with patch('agents.social_memory.SocialContact',side_effect=AssertionError),patch('agents.social_memory.SocialUpdate',side_effect=AssertionError):
            sim.step();sim.step()
        self.assertIsNone(sim.social_observer)
        self.assertEqual(len(sim.agents[0].social_memory),1)

    def test_recording_requires_feature_and_finish_rejects_future_events(self):
        with self.assertRaises(ValueError):Simulation(quiet(),social_sink=lambda r:None)
        sim=simulation(sink=lambda r:None);sim.social_observer.finish(sim.statistics)
        with self.assertRaises(RuntimeError):sim.social_observer.tick_end(sim.statistics)

    def test_finish_does_not_change_private_memory_or_rng(self):
        sim=simulation(quiet(world_width=1),sink=lambda r:None)
        sim.step();sim.step();before=state(sim,include_social=True);rng=random.getstate()
        sim.social_observer.finish(sim.statistics)
        self.assertEqual(before,state(sim,include_social=True));self.assertEqual(rng,random.getstate())
        self.assertEqual(sim.social_observer._evicted,{})

    def test_all_historical_fixture_hashes(self):
        expected={'v02_seed0_tick100.json':'5c3bf17c0ab4014d1ec98e823b239d860b74f92cc1394ab89c3bad73c5594e3b',
        'v03_auditory_control.json':'ba85a1314e1c68a3588f4ba7b6d698eca16d22b706de307850ff625ebf17260b',
        'v04_listening_control.json':'4b4b9e82362698b52a71e08e766cf16795575d4479cf023c8331eacfc624316a',
        'v05_listening_seed7_tick80.json':'5538900edb7d38eb74ca06b28602f213e45b3efbdbf8528130eb804cd7459b44',
        'v06_vocal_seed7_tick80.json':'0098529d6812003f97d718b3e1f4927b88fd371775047f82b640df96ec048dcd',
        'v061_communication_toys.json':'d397488e3f476b26d3e7b20c8714fd16a1d452b9c27b57b68fe03aeb955130da'}
        for name,digest in expected.items():self.assertEqual(hashlib.sha256((Path('tests/fixtures')/name).read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main()
