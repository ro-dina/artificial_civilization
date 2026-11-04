"""v073 controls are exogenous physical-key inputs, not alternative ecology."""
from contextlib import ExitStack
from dataclasses import replace
import inspect,json,random,tempfile,unittest
from pathlib import Path

from agents.nuisance import NuisanceModel,NuisanceConfig,NuisanceSource,NuisanceObservation,NuisanceBrain
from agents.learning import Physiology
from config import Config,SocialConfig,SocialPolicyConfig,SocialLearningConfig
from simulation.simulation import Simulation
from simulation.controls_observer import ControlsObserver,ControlsWriter,write_metadata,C_PAIRS
from experiments.social_learning_experiments import fingerprints
from experiments.information_controls import initial_fingerprint,fingerprint
from experiments.controls_metrics import sequence_metrics,attach_nuisance


class NuisanceTests(unittest.TestCase):
    def model(self,k=11,alpha=.6):return NuisanceModel((1/k,)*k,alpha)
    def sim(self,condition='D',observer=None,nseed=9,capacity=256):
        return Simulation(Config.evolution(brain='learning',initial_population=8,random_seed=3,learning_memory_capacity=capacity),
            social_config=SocialConfig(True),social_policy=SocialPolicyConfig(),
            social_learning=SocialLearningConfig({'A':'none','B':'perception','C':'memory','D':'none','E':'none'}[condition]),
            nuisance=NuisanceConfig(self.model(37 if condition=='E' else 11),nseed) if condition in 'DE' else None,
            social_learning_observer=observer)

    def test_model_validation(self):
        for args in [([],0),((.2,.2),0),((float('nan'),1),0),((-.1,1.1),0),((1.,),-1),((1.,),1.01)]:
            with self.subTest(args=args),self.assertRaises(ValueError):NuisanceModel(*args)
        for seed in ('a',True,None):
            with self.assertRaises(ValueError):NuisanceConfig(self.model(),seed)

    def test_repeatability_and_bounds(self):
        for k in (11,37):
            sources=[NuisanceSource(NuisanceConfig(self.model(k),5)) for _ in range(2)]
            streams=[[s.new_channel() for _ in range(4)] for s in sources]
            seq=[[[c.advance() for _ in range(100)] for c in stream] for stream in streams]
            self.assertEqual(seq[0],seq[1]);self.assertTrue(all(0<=x<k for line in seq[0] for x in line))
            self.assertNotEqual(seq[0][0],seq[0][1])

    def test_different_nuisance_seeds(self):
        channels=[NuisanceSource(NuisanceConfig(self.model(),s)).new_channel() for s in (5,6)]
        self.assertNotEqual([channels[0].advance() for _ in range(100)],[channels[1].advance() for _ in range(100)])

    def test_temporal_persistence(self):
        m=self.model(11,.8);c=NuisanceSource(NuisanceConfig(m)).new_channel();seq=[c.advance() for _ in range(20000)]
        observed=sum(a==b for a,b in zip(seq,seq[1:]))/(len(seq)-1)
        self.assertAlmostEqual(observed,m.same_token_probability,delta=.015)
        c=NuisanceSource(NuisanceConfig(self.model(alpha=1))).new_channel()
        self.assertEqual(len({c.advance() for _ in range(100)}),1)

    def test_no_global_rng_draws(self):
        before=random.getstate();self.sim().step();self.assertEqual(before,random.getstate())

    def test_independent_initial_streams(self):
        a=self.sim('A');d=self.sim('D');e=self.sim('E')
        self.assertEqual(initial_fingerprint(a),initial_fingerprint(d));self.assertEqual(initial_fingerprint(a),initial_fingerprint(e))
        b=d.agents[0].brain;before=b.rng.getstate()
        for _ in range(100):b.nuisance_channel.advance()
        self.assertEqual(before,b.rng.getstate());self.assertEqual(initial_fingerprint(d),initial_fingerprint(e))

    def test_generator_no_external_inputs(self):
        self.assertEqual(list(inspect.signature(NuisanceSource.new_channel).parameters),['self'])
        self.assertEqual(list(inspect.signature(type(NuisanceSource(NuisanceConfig(self.model())).new_channel()).advance).parameters),['self'])

    def test_read_only_preview(self):
        s=self.sim();before=fingerprint(s)
        for h in s.agents:
            o=s.observe(h);self.assertIsInstance(o,NuisanceObservation)
            self.assertFalse(hasattr(o,'individuals'));self.assertFalse(hasattr(o,'social_features'))
        self.assertEqual(before,fingerprint(s))

    def test_original_and_vocal_key_unchanged(self):
        s=self.sim();h=s.agents[0];o=s.observe(h);b=h.brain
        body=Physiology.capture(h);original=b.state(body,o);key=b.physical_state(body,o)
        self.assertEqual(key[:-1],original)
        self.assertEqual(b.state(body,replace(o,token=(o.token+1)%11)),original)
        self.assertFalse(any('social' in k for k in vars(b)))

    def test_default_unchanged_with_light_and_full_diagnostics(self):
        for condition in 'ABC':
            a=self.sim(condition)
            for light in (False,True):
                o=ControlsObserver(lambda k,r:None,condition,light);b=self.sim(condition,o)
                control=self.sim(condition)
                for _ in range(12):control.step();b.step();self.assertEqual(fingerprints(control),fingerprints(b))
                o.finish(b);self.assertEqual(fingerprints(control),fingerprints(b))

    def test_nuisance_logging_invariance(self):
        for condition in 'DE':
            a=self.sim(condition);rows=[];o=ControlsObserver(lambda k,r:rows.append((k,r)),condition);b=self.sim(condition,o)
            for _ in range(12):a.step();b.step();self.assertEqual(fingerprint(a),fingerprint(b))
            o.finish(b);self.assertEqual(fingerprint(a),fingerprint(b))
            outcomes=[r for k,r in rows if k=='outcomes'];self.assertEqual(len(outcomes),o.total['decisions'])
            self.assertEqual([r['sequence'] for r in outcomes],list(range(len(outcomes))))
            self.assertTrue(all('reward' in r and 'terminal' in r for r in outcomes))

    def test_invalid_combinations(self):
        n=NuisanceConfig(self.model())
        with self.assertRaises(ValueError):Simulation(Config(),nuisance=n)
        with self.assertRaises(ValueError):Simulation(Config(brain='learning'),nuisance=n,social_learning=SocialLearningConfig('perception'))
        with self.assertRaises(ValueError):Simulation(Config(brain='learning'),nuisance='invalid')

    def test_experiment_bounds_and_historical_protection(self):
        from experiments.information_controls import run_one
        for kwargs in ({'ticks':501},{'ticks':0},{'condition':'AB'}):
            args={'folder':'/tmp/v073-should-not-be-created','condition':'D','seed':0,'ticks':1};args.update(kwargs)
            with self.assertRaises(ValueError):run_one(**args)
        with self.assertRaises(ValueError):run_one('data/social_learning_v072/do-not-write','A',0,1)

    def test_rewards_and_vocal_learning_are_unmodified_on_fixed_actions(self):
        from agents.learning import LearningBrain
        from agents.human import Human
        from agents.observation import Observation
        cfg=Config.evolution(brain='learning',learning_epsilon=1.)
        a=LearningBrain(random.Random(17),cfg);b=NuisanceBrain(random.Random(17),cfg,NuisanceSource(NuisanceConfig(self.model())).new_channel())
        humans=[Human(0,0,0,brain) for brain in (a,b)]
        for tick in range(1,20):
            token=b.nuisance_channel.advance();obs=[Observation(tick,()),NuisanceObservation(tick,(),token=token)]
            decisions=[h.brain.choose_action(h,o) for h,o in zip(humans,obs)];self.assertEqual(*decisions)
            for h in humans:
                h.hunger+=.5;h.thirst+=.7;h.energy-=.1;h.brain.observe_outcome(Physiology.capture(h))
            self.assertEqual(a.last_reward,b.last_reward);self.assertEqual(a._vocal_values,b._vocal_values)
            self.assertEqual(a.rng.getstate(),b.rng.getstate());self.assertEqual(a.vocal_rng.getstate(),b.vocal_rng.getstate())

    def test_c_categories(self):
        self.assertEqual(len(C_PAIRS),37);self.assertEqual(len(set(C_PAIRS)),37)

    def test_writer_exclusive_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=[]
            for i in (0,1):
                p=Path(tmp)/f'{i}.csv.gz'
                with ExitStack() as stack:
                    w=ControlsWriter.open(stack,p,True);o=ControlsObserver(w,'D',True);s=self.sim(observer=o)
                    for _ in range(3):s.step()
                    o.finish(s)
                m=write_metadata(p,s,o,w);data.append({k:(p.parent/r['file']).read_bytes() for k,r in m['files'].items()})
                with ExitStack() as stack,self.assertRaises(ValueError):ControlsWriter.open(stack,p,True)
            self.assertEqual(data[0],data[1])


class AnalysisTests(unittest.TestCase):
    def frame(self):
        import pandas as pd
        return pd.DataFrame({'agent_id':[4,4,9,4,9],'base_id':[0,0,0,1,0],'terminal':[0,0,0,1,0],
                             'b_token':[0,0,1,1,0],'c_token':[0,0,1,1,0]})

    def test_actual_sequence_metrics(self):
        f=self.frame();m,counts,joint,runs,visits=sequence_metrics(f,f.b_token)
        self.assertEqual(m['decisions'],5);self.assertEqual(m['distinct_agent_states'],4)
        self.assertEqual(m['same_transitions'],1);self.assertEqual(m['transitions'],3)
        self.assertAlmostEqual(m['first_state_fraction'],.8);self.assertAlmostEqual(m['token_entropy'],.9709505944546686)
        self.assertEqual(sum(visits.values()),4);self.assertEqual(sum(runs.values()),4)

    def test_offline_attachment_no_id_or_reward_leakage(self):
        f=self.frame();m=NuisanceModel((.5,.5),.3)
        a=attach_nuisance(f,m,7);f.agent_id=f.agent_id.map({4:123,9:88})
        f['reward']=[1,2,3,4,5];f['action']=0
        self.assertEqual(a,attach_nuisance(f,m,7))

    def test_calibration_determinism_and_reward_blindness(self):
        from experiments.calibrate_information_controls import fit
        a=self.frame();a['reward']=999;a['action']=0
        b=a.copy();b['reward']=-999;b['action']=7
        self.assertEqual(json.dumps(fit([a,a],'B'),sort_keys=True),json.dumps(fit([b,b],'B'),sort_keys=True))

    def test_prediction_is_action_conditioned(self):
        from experiments.controls_prediction import Predictor
        p=Predictor()
        for _ in range(25):p.add((1,),0,-.2);p.add((1,),1,.2)
        a,ok,n=p.predict((1,),0);b,_,_=p.predict((1,),1)
        self.assertTrue(ok);self.assertEqual(n,25);self.assertGreater(a[0],a[2]);self.assertGreater(b[2],b[0])
        prior,ok,n=p.predict((9,),0);self.assertFalse(ok);self.assertEqual(n,0);self.assertAlmostEqual(sum(prior),1.)

    def test_hierarchical_n_is_ecological_seeds(self):
        import pandas as pd
        from experiments.analyze_information_controls import grouped
        f=pd.DataFrame({'condition':['D']*4,'seed':[0,0,0,1],'value':[0.,3.,6.,9.]})
        r=grouped(f,['condition'],['value']).iloc[0]
        self.assertEqual(r['n'],2);self.assertEqual(r['mean'],6.)
        self.assertAlmostEqual(r['std'],18**.5)

    def test_replay_exact_q_lru_pending_all_source_modes(self):
        from experiments.controls_prediction import replay
        helper=NuisanceTests()
        with tempfile.TemporaryDirectory() as tmp:
            for condition in 'ABCDE':
                path=Path(tmp)/f'{condition}.csv.gz'
                with ExitStack() as stack:
                    w=ControlsWriter.open(stack,path,True);o=ControlsObserver(w,condition,True);s=helper.sim(condition,o,capacity=1)
                    for _ in range(15):s.step()
                    o.finish(s)
                write_metadata(path,s,o,w)
                result=replay(Path(str(path)+'.metadata.json'),condition,helper.model(37 if condition=='E' else 11) if condition in 'DE' else None,9)
                # Source reconstruction uses its actual recorded tokens. An
                # alternative D/E representation attaches independent tokens.
                self.assertTrue(result['exact_source_reconstruction'])

    def test_empty_sequence_is_na(self):
        f=self.frame().iloc[:0];m,*_=sequence_metrics(f,[])
        self.assertEqual(m['decisions'],0);self.assertTrue(m['first_state_fraction']!=m['first_state_fraction'])

if __name__=='__main__':unittest.main()
