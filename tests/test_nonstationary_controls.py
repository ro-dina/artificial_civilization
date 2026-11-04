"""v074 scheduled nuisance, fixed-experience semantics and frozen diagnostics."""
from contextlib import ExitStack
from dataclasses import asdict,replace,FrozenInstanceError
import inspect,json,random,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd

from agents.nuisance import (NuisanceModel,NuisanceEpoch,NonstationaryNuisanceModel,
    NuisanceConfig,NuisanceSource,NonstationaryNuisanceChannel,nuisance_model_from_dict)
from agents.learning import Physiology
from config import Config,SocialConfig,SocialPolicyConfig
from simulation.simulation import Simulation
from simulation.controls_observer import ControlsWriter,ControlsObserver,write_metadata
from experiments.information_controls import fingerprint,initial_fingerprint
from experiments.nonstationary_metrics import profiles,run_segments
from experiments.nonstationary_controls import gate_passes,conditional_tv
from experiments.controls_prediction import replay


def delta(k,index):return tuple(float(i==index) for i in range(k))


def model(k=11,hold=0):
    return NonstationaryNuisanceModel(k,(NuisanceEpoch(0,NuisanceModel(delta(k,0),0)),NuisanceEpoch(100,NuisanceModel(delta(k,k-1),hold))))


class ScheduleTests(unittest.TestCase):
    def test_schedule_boundaries_and_last_epoch(self):
        m=model();self.assertEqual(m.at(0),m.at(99));self.assertEqual(m.at(100),m.at(10000));self.assertNotEqual(m.at(99),m.at(100))

    def test_invalid_configuration(self):
        p=NuisanceModel((1/11,)*11)
        for args in [(3,(NuisanceEpoch(0,p),)),(11,()),(11,[NuisanceEpoch(0,p)]),(11,(NuisanceEpoch(1,p),)),(11,(NuisanceEpoch(0,p),NuisanceEpoch(0,p))),(37,(NuisanceEpoch(0,p),))]:
            with self.subTest(args=args),self.assertRaises(ValueError):NonstationaryNuisanceModel(*args)
        for clock in (-1,True,1.5,None):
            with self.assertRaises(ValueError):model().at(clock)

    def test_model_file_supports_both_versions(self):
        p={'probabilities':[.5,.5],'hold_probability':.4}
        self.assertEqual(nuisance_model_from_dict(p),NuisanceModel((.5,.5),.4))
        value={'kind':'nonstationary','category_count':11,'schedule':[{'start_tick':e.start_tick,**asdict(e.model)} for e in model().schedule]}
        self.assertEqual(nuisance_model_from_dict(value),model())
        for v in ({'kind':'unknown'},[],{'kind':'nonstationary'}, {'probabilities':[1]}):
            with self.assertRaises(ValueError):nuisance_model_from_dict(v)

    def test_boundary_has_no_forced_reset(self):
        channel=NuisanceSource(NuisanceConfig(model(hold=1))).new_channel()
        self.assertEqual(channel.advance(99),0);self.assertEqual(channel.advance(100),0)
        c=NuisanceSource(NuisanceConfig(model())).new_channel()
        self.assertEqual(c.advance(99),0);self.assertEqual(c.advance(100),10)

    def test_first_turn_uses_current_calendar_distribution(self):
        c=NuisanceSource(NuisanceConfig(model())).new_channel();before=c.rng.getstate()
        self.assertEqual(c.preview(100),10);self.assertEqual(c.rng.getstate(),before);self.assertEqual(c.opportunities,0)
        self.assertEqual(c.advance(101),10)
        with self.assertRaises(ValueError):c.advance(101)
        with self.assertRaises(ValueError):c.advance(99)

    def test_reproducibility_category_bounds_and_only_clock_api(self):
        for k in (11,37):
            m=NonstationaryNuisanceModel(k,(NuisanceEpoch(0,NuisanceModel((1/k,)*k,.5)),NuisanceEpoch(100,NuisanceModel((1/k,)*k,.2))))
            streams=[NuisanceSource(NuisanceConfig(m,7)).new_channel() for _ in range(2)]
            seq=[[s.advance(t) for t in range(1,201)] for s in streams]
            self.assertEqual(*seq);self.assertTrue(all(0<=i<k for i in seq[0]))
        self.assertEqual(list(inspect.signature(NonstationaryNuisanceChannel.advance).parameters),['self','tick'])

    def test_frozen_config(self):
        with self.assertRaises(FrozenInstanceError):model().category_count=37

    def test_simulation_rng_and_preview_isolation(self):
        cfg=Config.evolution(brain='learning',initial_population=5,random_seed=3)
        a=Simulation(cfg);b=Simulation(cfg,nuisance=NuisanceConfig(model(),31));before=random.getstate()
        self.assertEqual(initial_fingerprint(a),initial_fingerprint(b))
        old=fingerprint(b)
        for h in b.agents:b.observe(h)
        self.assertEqual(fingerprint(b),old)
        for h in b.agents:
            state=h.brain.rng.getstate();h.brain.nuisance_channel.advance(100);self.assertEqual(h.brain.rng.getstate(),state)
        self.assertEqual(before,random.getstate())

    def test_optional_diagnostics_leave_scheduled_trajectory_exact(self):
        cfg=Config.evolution(brain='learning',initial_population=5)
        for light in (False,True):
            o=ControlsObserver(lambda k,r:None,'D',light)
            a=Simulation(cfg,social_config=SocialConfig(True),nuisance=NuisanceConfig(model()))
            b=Simulation(cfg,social_config=SocialConfig(True),nuisance=NuisanceConfig(model()),social_learning_observer=o)
            for _ in range(12):a.step();b.step();self.assertEqual(fingerprint(a),fingerprint(b))
            o.finish(b);self.assertEqual(fingerprint(a),fingerprint(b))

    def test_vocal_state_and_reward_ignore_token(self):
        cfg=Config.evolution(brain='learning',initial_population=3)
        s=Simulation(cfg,nuisance=NuisanceConfig(model()));h=s.agents[0];b=h.brain;o=s.observe(h);body=Physiology.capture(h)
        self.assertEqual(b.state(body,o),b.state(body,replace(o,token=10)))
        self.assertEqual(b.physical_state(body,o)[:-1],b.state(body,o))
        h.brain.choose_action(h,o);h.hunger+=.5;h.thirst+=.3;h.energy-=.7;h.brain.observe_outcome(Physiology.capture(h))
        self.assertAlmostEqual(b.last_reward,-.5/cfg.max_hunger-.3/cfg.max_thirst-.7/cfg.max_energy)

    def test_scheduled_tokens_do_not_change_vocal_updates_on_fixed_actions(self):
        from agents.learning import LearningBrain
        from agents.nuisance import NuisanceBrain,NuisanceObservation
        from agents.human import Human
        from agents.observation import Observation
        cfg=Config.evolution(brain='learning',learning_epsilon=1.)
        m=NonstationaryNuisanceModel(11,(NuisanceEpoch(0,NuisanceModel((1/11,)*11,.2)),NuisanceEpoch(5,NuisanceModel((1/11,)*11,.8))))
        a=LearningBrain(random.Random(17),cfg);b=NuisanceBrain(random.Random(17),cfg,NuisanceSource(NuisanceConfig(m)).new_channel())
        humans=[Human(0,0,0,brain) for brain in (a,b)]
        for tick in range(1,15):
            u=b.nuisance_channel.advance(tick);observations=[Observation(tick,()),NuisanceObservation(tick,(),token=u)]
            self.assertEqual(*[h.brain.choose_action(h,o) for h,o in zip(humans,observations)])
            for h in humans:
                h.hunger+=.2;h.thirst+=.3;h.energy-=.1;h.alive=tick<14;h.brain.observe_outcome(Physiology.capture(h))
            self.assertEqual(a.last_reward,b.last_reward);self.assertEqual(a._vocal_values,b._vocal_values)
            self.assertEqual(a.vocal_rng.getstate(),b.vocal_rng.getstate());self.assertEqual(a.rng.getstate(),b.rng.getstate())

    def test_derived_age_equals_opportunities_including_newborns(self):
        from agents.brain import Action
        counts={};ages=[]
        class Observer(ControlsObserver):
            def begin(self,h,o,t):
                expected=counts.get(h.id,0);ages.append((h.id,h.age,expected));counts[h.id]=expected+1
                super().begin(h,o,t)
        cfg=Config.evolution(brain='learning',world_width=1,world_height=1,initial_food=0,initial_water=0,
            initial_population=2,hunger_per_tick=0,thirst_per_tick=0,max_age=None,min_reproductive_age=1,reproduction_cooldown=1)
        observer=Observer(lambda k,r:None,'A',True);s=Simulation(cfg,social_learning_observer=observer)
        s.agents[0].reproductive_type=0;s.agents[1].reproductive_type=1
        for _ in range(6):
            for h in s.agents:h.brain.actions=(Action.REPRODUCE,)
            s.step()
        self.assertGreater(s.statistics.births,0);self.assertTrue(any(identity>=2 for identity,_,_ in ages))
        self.assertTrue(all(age==expected for _,age,expected in ages));observer.finish(s)


class WindowAndGateTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'tick':[1,99,100,101,499,500],'agent_id':[1]*6,'base_id':[0]*6,'terminal':[0]*6,
            'opportunity_index':range(6),'derived_pre_age':range(6),'first_tick':[1]*6})

    def test_windows_lifetime_novelty_and_final_tick(self):
        rows,*_=profiles(self.frame(),[0]*6);rows=rows.set_index('window')
        self.assertEqual(rows.loc['0','decisions'],2);self.assertEqual(rows.loc['100','decisions'],2)
        self.assertEqual(rows.loc['400','decisions'],1);self.assertEqual(rows.loc['full','decisions'],6)
        self.assertEqual(rows.loc['100','first_state_fraction'],0);self.assertEqual(rows.loc['100','window_first_state_fraction'],.5)
        self.assertAlmostEqual(rows.loc['full','first_state_fraction'],1/6)

    def test_run_censoring(self):
        f=self.frame().iloc[:2];hist=run_segments(f,[0,1],{1:True})
        self.assertEqual(hist[(1,True,'observed_change')],1);self.assertEqual(hist[(1,False,'window_end_censored')],1)

    def test_gate_does_not_trust_override_flags_or_zero_support(self):
        good=pd.DataFrame({'difference':[.01],'target':[.02],'supported':[True],'within_target':[True],'gross_failure':[False]})
        self.assertTrue(gate_passes(good));self.assertFalse(gate_passes(good.iloc[:0]))
        for field,value in [('difference',.03),('supported',False),('difference',np.nan),('gross_failure',True)]:
            f=good.copy();f[field]=value;self.assertFalse(gate_passes(f))

    def test_conditional_support_matches_original_keys_not_local_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.csv';q=Path(tmp)/'b.csv';pd.DataFrame({'base_id':[0],'key_json':['[1,2]']}).to_csv(p,index=False)
            pd.DataFrame({'base_id':[9],'key_json':['[1,2]']}).to_csv(q,index=False)
            a=pd.DataFrame({'base_id':[0]*40});b=pd.DataFrame({'base_id':[9]*40})
            tv,s=conditional_tv(a,[0]*40,b,[1]*40,{'states':p},{'states':q});self.assertEqual(tv,1.);self.assertEqual(s['common_X_cells'],1)
            tv,s=conditional_tv(a.iloc[:10],[0]*10,b.iloc[:10],[1]*10,{'states':p},{'states':q});self.assertTrue(np.isnan(tv));self.assertEqual(s['common_X_cells'],0)

    def test_terminal_replay_does_not_invent_pending_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.csv.gz'
            with ExitStack() as stack:
                w=ControlsWriter.open(stack,p,True);o=ControlsObserver(w,'A',True)
                s=Simulation(Config(brain='learning',initial_population=1,max_hunger=1,hunger_per_tick=1),social_learning_observer=o)
                s.step();o.finish(s)
            write_metadata(p,s,o,w);r=replay(Path(str(p)+'.metadata.json'),'A',diagnostics=True)
            self.assertEqual(r['pending_cutoff'],0);self.assertEqual(r['updates'],1);self.assertEqual(r['retained_rows'],0)
            self.assertEqual(r['diagnostics']['state_updates'],[{'updates':1,'states':1}])

    def test_scheduled_source_replay_pending_and_churn_are_exact(self):
        for k,c in [(11,'D'),(37,'E')]:
            m=NonstationaryNuisanceModel(k,(NuisanceEpoch(0,NuisanceModel(delta(k,0),0)),NuisanceEpoch(2,NuisanceModel(delta(k,k-1),0))))
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'a.csv.gz'
                with ExitStack() as stack:
                    w=ControlsWriter.open(stack,path,True);o=ControlsObserver(w,c,True)
                    s=Simulation(Config.evolution(brain='learning',initial_population=3,learning_memory_capacity=1),nuisance=NuisanceConfig(m),social_learning_observer=o)
                    for _ in range(4):s.step()
                    o.finish(s)
                write_metadata(path,s,o,w);r=replay(Path(str(path)+'.metadata.json'),c,m,diagnostics=True)
                self.assertTrue(r['exact_source_reconstruction']);self.assertEqual(r['pending_cutoff'],3);self.assertGreater(r['evictions'],0)
                self.assertEqual(sum(v['updates']*v['states'] for v in r['diagnostics']['state_updates']),r['updates'])

    def test_frozen_calibration_cannot_be_changed(self):
        from experiments.nonstationary_controls import verify_frozen
        from simulation.vocal_learning_observer import hash_file
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'models').mkdir();(root/'preregistration.json').write_text('{}');(root/'models/D.json').write_text('{}')
            m={'inputs':{},'models':{'D':hash_file(root/'models/D.json')},'preregistration_sha256':hash_file(root/'preregistration.json')}
            (root/'models/frozen.json').write_text(json.dumps(m));verify_frozen(root)
            (root/'models/D.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):verify_frozen(root)


class PredictionTests(unittest.TestCase):
    def test_hierarchical_sparse_cells_back_off_to_the_same_parent(self):
        from experiments.nonstationary_prediction import Hierarchy,features
        p=Hierarchy(('X_A','X_B_A','X_C_A'),3);keys=features((0,),0,0,0,0,0)
        for _ in range(25):p.add(keys,0,1)
        p.freeze();unseen=features((0,),0,1,1,0,0)
        result,support=p.predict(unseen,0)
        self.assertEqual(result['X_A'],result['X_B_A']);self.assertEqual(result['X_B_A'],result['X_C_A'])
        self.assertFalse(support['X_B_A'])
        with self.assertRaises(RuntimeError):p.add(keys,0,1)

    def test_test_targets_do_not_change_trained_predictor(self):
        import copy
        from experiments.nonstationary_prediction import Hierarchy,features
        p=Hierarchy(('X_A','X_B_A'),2);keys=features((0,),0,0,0,0,0)
        p.add(keys,0,1);p.freeze();before=copy.deepcopy(p.cells);actions=copy.deepcopy(p.actions)
        for _ in range(10):p.predict(features((9,),7,5,2,3,4),7)
        self.assertEqual(before,p.cells);self.assertEqual(actions,p.actions)

    def test_future_reward_and_survival_censoring(self):
        from experiments.nonstationary_prediction import outcome_targets
        f=pd.DataFrame({'agent_id':[1]*7+[2]*2,'tick':list(range(1,8))+[1,2],
            'reward':[.1]*9,'terminal':[0]*6+[1,0,0]})
        values,reasons=outcome_targets(f)
        self.assertEqual(values['future_reward_5'][0],2);self.assertEqual(values['future_reward_5'][1],2)
        self.assertEqual(values['future_reward_5'][2],-1);self.assertEqual(reasons['future_reward_5'][2],'death_censored')
        self.assertEqual(values['survival_5'][0],1);self.assertEqual(values['survival_5'][1],0)
        self.assertEqual(values['survival_5'][6],0);self.assertEqual(values['survival_5'][7],-1)
        self.assertEqual(reasons['future_reward_5'][7],'run_end_censored')

    def test_missing_future_row_is_not_fabricated(self):
        from experiments.nonstationary_prediction import outcome_targets
        f=pd.DataFrame({'agent_id':[1]*6,'tick':[1,2,4,5,6,7],'reward':[.1]*6,'terminal':[0]*6})
        values,reasons=outcome_targets(f);self.assertEqual(values['future_reward_5'][0],-1)
        self.assertEqual(reasons['future_reward_5'][0],'gap_censored');self.assertEqual(values['survival_5'][0],1)

    def test_memory_specific_features_hold_perception_fixed(self):
        from experiments.nonstationary_prediction import features
        a=features((1,),0,6,12,0,12);b=features((1,),0,6,13,0,13)
        self.assertEqual(a['X_B_A'],b['X_B_A']);self.assertNotEqual(a['X_C_A'],b['X_C_A'])
        self.assertEqual(a['X_C_A'][:3],b['X_C_A'][:3])

    def test_no_outcome_or_identity_enters_attached_nuisance(self):
        from experiments.controls_metrics import attach_nuisance
        f=pd.DataFrame({'agent_id':[1,9,1,9],'tick':[1,1,2,2],'terminal':[0]*4,'reward':[9]*4})
        m=NonstationaryNuisanceModel(11,(NuisanceEpoch(0,NuisanceModel((1/11,)*11,.5)),))
        a=attach_nuisance(f,m,31);f.agent_id=f.agent_id.map({1:888,9:12});f.reward=-999
        self.assertEqual(a,attach_nuisance(f,m,31))

if __name__=='__main__':unittest.main()
