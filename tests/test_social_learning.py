"""v0.7.2 decision-time information, independent heads and passive measurement."""
from dataclasses import asdict,fields,FrozenInstanceError
import io,json,random,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.brain import Action
from agents.learning import LearningBrain,Physiology
from agents.human import Human
from agents.social_memory import SocialMemory
from agents.social_memory_policy import make_memory
from agents.social_learning import (Direction as D,SocialFeatures,SocialLearningObservation,SocialLearningBrain,
                                    direction,nearest,extract,learning_observation)
from agents.observation import Observation,SocialObservation,VisibleIndividual as V
from config import Config,SocialConfig,SocialPolicyConfig,SocialLearningConfig
from simulation.simulation import Simulation
from simulation.social_learning_observer import SocialLearningObserver,probe
from experiments.social_verification import state
from experiments.auditory_policy import frozen


def quiet(**kw):
    args=dict(brain='learning',world_width=1,world_height=1,initial_population=2,initial_food=0,initial_water=0,
              hunger_per_tick=0,thirst_per_tick=0,max_age=None,learning_controls_vocalization=False)
    args.update(kw);return Config(**args)


def sim(mode='memory',cfg=None,policy=None,observer=None,capacity=32):
    return Simulation(cfg or quiet(),social_config=SocialConfig(True,capacity),social_policy=policy,
                      social_learning=SocialLearningConfig(mode),social_learning_observer=observer)


def holds(s,action=Action.WAIT):
    for h in s.agents:h.brain.actions=(action,)


class FeatureTests(unittest.TestCase):
    def test_empty_and_same_cell_and_count_compression(self):
        self.assertEqual(extract((),include_memory=True),SocialFeatures(0,0))
        self.assertEqual(extract((V(4,0,0),)),SocialFeatures(1))
        self.assertEqual(extract((V(4,0,0),V(9,3,2))),SocialFeatures(6))

    def test_all_directions_and_diagonal_vertical_ties(self):
        for offset,expected in (((0,0),D.HERE),((0,-2),D.NORTH),((0,2),D.SOUTH),((3,1),D.EAST),((-3,-1),D.WEST),((2,-2),D.NORTH),((-2,2),D.SOUTH)):
            self.assertEqual(direction(*offset),expected)

    def test_nearest_chebyshev_then_id_not_input_order(self):
        values=(V(9,1,0),V(3,-1,1),V(0,2,0))
        self.assertEqual(nearest(values),values[1])
        self.assertEqual(extract(values),extract(values[::-1]))
        self.assertEqual(extract(values).visible_direction,D.SOUTH)

    def test_remembered_subset_has_own_nearest_and_candidates_are_unknown(self):
        m=make_memory(SocialConfig(True),SocialPolicyConfig(5))
        for t in range(1,5):m.observe((1,),t)
        self.assertEqual(extract((V(1,1,0),),m,include_memory=True),SocialFeatures(4,0))
        m.observe((1,),5)
        self.assertEqual(extract((V(1,1,0),),m,include_memory=True),SocialFeatures(4,4))
        self.assertEqual(extract((V(0,0,0),V(1,2,0)),m,include_memory=True),SocialFeatures(6,4))

    def test_frozen_no_identity_or_coordinates_or_memory_timestamps(self):
        o=learning_observation(SocialObservation(1,(),individuals=(V(777,3,0),)),SocialFeatures(4,0))
        self.assertEqual({f.name for f in fields(o)},{'tick','tiles','auditory','social'})
        self.assertEqual({f.name for f in fields(o.social)},{'visible','remembered_direction'})
        self.assertNotIn('777',json.dumps(asdict(o)))
        with self.assertRaises(FrozenInstanceError):o.social.visible=2

    def test_exact_structural_state_multipliers(self):
        valid=[]
        for token in range(11):
            SocialFeatures(token)
            for rd in range(6):
                try:valid.append(SocialFeatures(token,rd))
                except ValueError:pass
        self.assertEqual(len(valid),37)

    def test_invalid_configuration_and_impossible_features(self):
        for value in ('other',None,False):
            with self.assertRaises(ValueError):SocialLearningConfig(value)
        for args in ((True,None),(11,None),(0,1),(2,1),(2,3),(7,1)):
            with self.assertRaises(ValueError):SocialFeatures(*args)
        with self.assertRaises(ValueError):Simulation(quiet(),social_learning=SocialLearningConfig('memory'))
        with self.assertRaises(ValueError):Simulation(Config(),social_learning=SocialLearningConfig('perception'))
        with self.assertRaises(ValueError):Simulation(quiet(),social_learning=False)

    def test_feature_extraction_is_read_only_and_rng_free(self):
        m=SocialMemory();m.observe((1,),1);before=frozen(vars(m));r=random.getstate()
        with patch.object(random.Random,'random',side_effect=AssertionError):
            for _ in range(3):extract((V(1,1,1),),m,include_memory=True)
        self.assertEqual(before,frozen(vars(m)));self.assertEqual(r,random.getstate())


class TimingTests(unittest.TestCase):
    def test_first_sighting_uses_precommit_snapshot_and_public_preview_is_pure(self):
        s=sim();holds(s)
        self.assertEqual(s.observe(s.agents[0]).social.remembered_direction,0)
        self.assertEqual(len(s.agents[0].social_memory),0)
        s.step()
        for h in s.agents:
            self.assertEqual(h.brain.decision_state[-1],0);self.assertEqual(len(h.social_memory),1)
        s.step()
        for h in s.agents:self.assertEqual(h.brain.decision_state[-1],D.HERE)

    def test_fifth_sighting_still_unknown_then_sixth_remembered(self):
        s=sim(policy=SocialPolicyConfig(5));holds(s)
        for tick in range(1,7):
            s.step()
            self.assertEqual(s.agents[0].brain.decision_state[-1],0 if tick<=5 else D.HERE)

    def test_snapshot_survives_same_batch_eviction_and_forgotten_reappearance(self):
        s=sim(cfg=quiet(world_width=3,initial_population=3),capacity=1);holds(s)
        for h,x in zip(s.agents,(1,0,2)):h.x=x
        s.step();h=s.agents[0];self.assertIsNotNone(h.social_memory.get(2))
        s.step();self.assertEqual(h.brain.decision_state[-1],D.EAST)
        self.assertIsNone(h.social_memory.get(2));self.assertIsNotNone(h.social_memory.get(1))
        s.step();self.assertEqual(h.brain.decision_state[-1],D.WEST)

    def test_perception_requires_no_memory_and_zero_radius_same_cell(self):
        s=Simulation(quiet(perception_radius=0),social_learning=SocialLearningConfig('perception'))
        holds(s);s.step()
        self.assertEqual(s.agents[0].brain.decision_state[-1],1)
        self.assertFalse(hasattr(s.agents[0],'social_memory'))
        self.assertIsNone(s.observe(s.agents[0]).social.remembered_direction)

    def test_sequential_death_and_movement_apply_to_later_observation(self):
        s=sim(cfg=quiet(max_age=1));holds(s)
        original=list(s.agents)
        with patch.object(s._order_rng,'shuffle',lambda x:None):s.step()
        self.assertEqual(original[0].brain._values.keys().__iter__().__next__()[-2:],(1,0))
        self.assertEqual(original[1].brain._values.keys().__iter__().__next__()[-2:],(0,0))
        self.assertEqual(s.population,0)
        for reproduction in (False,True):
            s=sim('perception',quiet(world_width=4,perception_radius=1,reproduction_enabled=reproduction));holds(s)
            s.agents[0].x=0;s.agents[1].x=2;s.agents[0].brain.actions=(Action.MOVE_EAST,)
            with patch.object(s._order_rng,'shuffle',lambda x:None):s.step()
            self.assertEqual(s.agents[1].brain.decision_state[-1],int(D.WEST))

    def test_newborn_has_empty_q_and_memory_and_waits_until_next_tick(self):
        s=sim(cfg=quiet(reproduction_enabled=True,min_reproductive_age=0));holds(s,Action.REPRODUCE)
        s.agents[0].reproductive_type,s.agents[1].reproductive_type=0,1
        s.step();self.assertEqual(s.population,3);child=s.agents[-1]
        self.assertEqual(child.brain.memory_size,0);self.assertEqual(len(child.social_memory),0)
        self.assertIsNone(child.brain.decision_state)
        s.step();self.assertIsNotNone(child.brain.decision_state);self.assertEqual(child.brain.decision_state[-1],0)

    def test_sound_does_not_provide_identity_or_create_memory(self):
        s=sim(cfg=quiet(initial_population=1,perception_radius=0));h=s.agents[0]
        s.communication.emit(7,999,0,0);s.communication.advance()
        self.assertEqual(s.observe(h).social,SocialFeatures(0,0));s.step();self.assertEqual(len(h.social_memory),0)


class IntegrationTests(unittest.TestCase):
    def test_detailed_events_serialize_actual_brain_facing_snapshot(self):
        rows=[]
        s=Simulation(quiet(),social_config=SocialConfig(True),social_learning=SocialLearningConfig('memory'),event_sink=rows.append)
        holds(s);s.step()
        observations=[e.details['observation'] for e in rows if e.kind=='agent_step']
        self.assertEqual(len(observations),2)
        for o in observations:
            self.assertIsInstance(o,SocialLearningObservation)
            self.assertEqual(o.social,SocialFeatures(1,0));self.assertFalse(hasattr(o,'individuals'))

    def test_vocal_keys_q_rewards_and_rng_are_identical_on_matched_input_outcomes(self):
        cfg=quiet(learning_controls_vocalization=True,learning_epsilon=.2)
        brains=[LearningBrain(random.Random(4),cfg),SocialLearningBrain(random.Random(4),cfg,'perception'),SocialLearningBrain(random.Random(4),cfg,'memory')]
        humans=[Human(0,0,0,b,hunger=30,thirst=40,energy=50) for b in brains]
        for tick in range(1,16):
            token=(0,1,4)[tick%3];rd=0 if token==0 else token
            for b,h in zip(brains,humans):
                o=Observation(tick,()) if not hasattr(b,'social_mode') else SocialLearningObservation(tick,(),social=SocialFeatures(token,rd if b.social_mode=='memory' else None))
                b.choose_action(h,o);b.observe_outcome(Physiology(29,39,51,True))
            for b in brains[1:]:
                self.assertEqual(frozen(b._vocal_values),frozen(brains[0]._vocal_values))
                self.assertEqual(frozen(b._vocal_pending),frozen(brains[0]._vocal_pending))
                self.assertEqual(b.vocal_rng.getstate(),brains[0].vocal_rng.getstate())
                self.assertEqual(b.last_reward,brains[0].last_reward)
                self.assertEqual(b.actions,brains[0].actions)
        self.assertTrue(all(len(k)==8 for k in brains[1]._vocal_values))
        self.assertTrue(all(len(k)==9 for k in brains[1]._values));self.assertTrue(all(len(k)==10 for k in brains[2]._values))

    def test_modes_replay_and_recording_preserve_complete_state(self):
        global_rng=random.getstate()
        for mode in ('none','perception','memory'):
            cfg=Config.evolution(brain='learning',initial_population=8,random_seed=7)
            rows=[];observer=SocialLearningObserver(lambda k,r:rows.append((k,r)))
            a,b=sim(mode,cfg),sim(mode,cfg,observer=observer)
            for _ in range(25):
                a.step();b.step();self.assertEqual(state(a,include_social=True),state(b,include_social=True))
                self.assertFalse(any(hasattr(h.brain,'_physical_observer') for h in b.agents))
            snapshot=state(b,include_social=True);observer.probes(b);observer.finish(b)
            self.assertEqual(snapshot,state(b,include_social=True));self.assertEqual(global_rng,random.getstate())
            self.assertEqual(observer.total['updates'],b.statistics.total_learning_updates)
            self.assertEqual(observer.total['decisions'],b.statistics.exploratory_actions+b.statistics.exploitative_actions)
            self.assertEqual(observer.total['signals_emitted'],b.statistics.signals_emitted)
            self.assertFalse(observer.agents);self.assertFalse(observer.partners)

    def test_none_mode_preserves_legacy_pending_key_identity(self):
        s=sim('none',quiet(learning_controls_vocalization=True));s.step()
        self.assertIs(s.agents[0].brain._pending.state,s.agents[0].brain._vocal_pending.state)
        self.assertEqual(set(vars(s.agents[0].brain)),set(vars(LearningBrain(random.Random(1),s.config))))

    def test_actual_allocation_eviction_hooks_capacity_one_and_terminal_update(self):
        o=SocialLearningObserver(lambda k,r:None)
        s=sim('memory',quiet(learning_memory_capacity=1,max_age=4),observer=o);holds(s)
        for _ in range(4):s.step()
        o.finish(s)
        self.assertEqual(o.total['updates'],8)
        self.assertGreater(o.total['evictions'],0)
        self.assertEqual(o.total['allocations']-o.total['evictions'],2)

    def test_cli_validation_metadata_and_exclusive_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'s.csv.gz'
            args=[sys.executable,'main.py','--brain','learning','--population','2','--ticks','3','--social-learning-mode','perception','--social-learning-csv',str(path)]
            subprocess.run(args,check=True,capture_output=True)
            m=json.loads(Path(str(path)+'.metadata.json').read_text())
            self.assertEqual(m['config']['social_learning'],{'mode':'perception'});self.assertFalse(m['config']['social']['social_memory_enabled'])
            self.assertEqual(m['totals']['decisions'],6)
            self.assertNotEqual(subprocess.run(args,capture_output=True).returncode,0)
            bad=[sys.executable,'main.py','--brain','learning','--social-learning-mode','memory','--ticks','1']
            self.assertNotEqual(subprocess.run(bad,capture_output=True).returncode,0)


class DiagnosticTests(unittest.TestCase):
    def test_distance_uses_frozen_target_actual_positions_and_chebyshev_plateau(self):
        move=SocialLearningObserver.movement
        self.assertEqual(move(V(1,2,0),(0,0),SimpleNamespace(x=1,y=0)),'toward')
        self.assertEqual(move(V(1,2,0),(0,0),SimpleNamespace(x=-1,y=0)),'away')
        self.assertEqual(move(V(1,2,2),(0,0),SimpleNamespace(x=1,y=0)),'equal')
        self.assertEqual(move(V(1,0,0),(0,0),SimpleNamespace(x=1,y=0)),'away')

    def test_blocked_moves_not_counted_as_toward_or_away(self):
        o=SocialLearningObserver(lambda k,r:None);s=sim(observer=o);holds(s,Action.MOVE_NORTH)
        s.step();self.assertEqual(o.total['moves'],0);self.assertEqual(o.total['target_moves'],0)
        self.assertEqual(o.total['blocked_moves'],2)

    def test_probe_exact_observed_rows_filters_priors_and_preserves_all_state(self):
        b=SocialLearningBrain(random.Random(1),quiet(learning_uses_auditory=False),'memory');h=Human(0,0,0,b)
        base=(0,0,0,2,2,2,2);a=(*base,9,0);c=(*base,9,4)
        b._values[a]=[1.,0.,0.,0.,0.,0.,0.];b._values[c]=[0.,1.,0.,0.,0.,0.,0.]
        before=frozen(vars(b));r=random.getstate()
        hist={a:[10,5,2],c:[10,5,2]}
        result=probe([(h,hist)],'memory',10,5,'memory_only')
        self.assertEqual(result['pairs'],1);self.assertEqual(result['changed_greedy_sets'],1)
        self.assertAlmostEqual(result['tv_sum'],1-b.epsilon)
        self.assertEqual(probe([(h,{a:[10,5,2],c:[10,5,0]})],'memory',10,5)['pairs'],0)
        self.assertEqual(before,frozen(vars(b)));self.assertEqual(r,random.getstate())


if __name__=='__main__':unittest.main()
