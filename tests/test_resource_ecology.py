"""v080 ecology opt-in, supply conservation and observational diagnostics."""
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from random import Random
import json,random,tempfile,unittest

from config import Config,ResourceEcologyConfig,SocialConfig,SocialPolicyConfig,SocialLearningConfig
from agents.brain import Action
from simulation.simulation import Simulation
from simulation.resource_observer import ResourceObserver,ResourceWriter,distance_map
from world.world import World
from world.resource_ecology import PatchGeometry,spatial_weights,ResourceEcology
from experiments.resource_ecology_experiments import fingerprint,ecology_config,pilot_passes
from experiments.social_learning_experiments import fingerprints


class EcologyTests(unittest.TestCase):
    def test_config_validation_and_legacy_default(self):
        self.assertTrue(ResourceEcologyConfig().legacy)
        for kw in ({'resource_supply_multiplier':-.1},{'resource_supply_multiplier':1.1},{'localization_strength':1},{'localization_strength':float('nan')},{'patch_count':0},{'patch_radius':-1},{'patch_count':True}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):ResourceEcologyConfig(**kw)

    def test_e0_all_modes_exact_with_optional_observer(self):
        cfg=Config.evolution(brain='learning',initial_population=8,random_seed=7)
        for mode,label in [('none','A'),('perception','B'),('memory','C')]:
            obs=ResourceObserver(lambda k,r:None,label,'E0',PatchGeometry.generate(cfg,ResourceEcologyConfig()))
            settings=dict(social_config=SocialConfig(True),social_policy=SocialPolicyConfig(),social_learning=SocialLearningConfig(mode))
            a=Simulation(cfg,**settings);b=Simulation(cfg,**settings,resource_ecology=ResourceEcologyConfig(),resource_observer=obs,social_learning_observer=obs)
            for _ in range(12):a.step();b.step();self.assertEqual(fingerprints(a),fingerprints(b))
            obs.finish(b);self.assertEqual(fingerprints(a),fingerprints(b));self.assertIsNone(b.resource_system)

    def test_patch_generation_and_persistence(self):
        cfg=Config.evolution(random_seed=3);eco=ResourceEcologyConfig(.5,.75)
        self.assertEqual(PatchGeometry.generate(cfg,eco),PatchGeometry.generate(cfg,eco))
        self.assertNotEqual(PatchGeometry.generate(cfg,eco).centers,PatchGeometry.generate(Config.evolution(random_seed=4),eco).centers)
        system=ResourceEcology(cfg,eco);old=system.geometry
        w=World.generate(cfg,Random(3));system.initialize(w)
        for _ in range(20):system.regenerate(w,Random(1),Random(2))
        self.assertEqual(old,system.geometry)

    def test_weights_normalize_with_positive_background(self):
        cfg=Config.evolution();g=PatchGeometry.generate(cfg,ResourceEcologyConfig())
        weights=spatial_weights(g,.75);self.assertAlmostEqual(sum(weights),1.);self.assertGreater(min(weights),0)
        self.assertGreater(sum(v for v,m in zip(weights,g.mask) if m),sum(g.mask)/len(g.mask))

    def test_initial_totals_equal_and_localized(self):
        cfg=Config.evolution(random_seed=4);legacy=Simulation(cfg);local=Simulation(cfg,resource_ecology=ResourceEcologyConfig(1,.75))
        from simulation.resource_observer import concentration
        for r in ('food','water'):
            a=[getattr(t,r) for row in legacy.world.tiles for t in row];b=[getattr(t,r) for row in local.world.tiles for t in row]
            self.assertEqual(sum(a),sum(b));self.assertTrue(all(0<=v<=5 for v in b))
            self.assertLess(concentration(b,local.resource_system.geometry.mask)['effective_cells'],concentration(a,local.resource_system.geometry.mask)['effective_cells'])

    def test_scarcity_scales_only_regeneration(self):
        cfg=Config.evolution(random_seed=5);a=Simulation(cfg);b=Simulation(cfg,resource_ecology=ResourceEcologyConfig(.5))
        self.assertEqual(asdict(a.config),asdict(b.config));self.assertEqual([(t.food,t.water) for row in a.world.tiles for t in row],[(t.food,t.water) for row in b.world.tiles for t in row])
        self.assertEqual(b.resource_system.supply_config.food_regeneration_probability,.01)
        for field in ('food_hunger_relief','water_thirst_relief','food_energy_gain','energy_per_tick','learning_rate','learning_discount','learning_memory_capacity'):
            self.assertEqual(getattr(a.config,field),getattr(b.resource_system.supply_config,field))

    def test_conditional_supply_counts_and_rng_equal(self):
        cfg=Config.evolution(initial_food=0,initial_water=0);a=World.generate(cfg,Random(2));b=World.generate(cfg,Random(2))
        system=ResourceEcology(cfg,ResourceEcologyConfig(1,.75));ar=[Random(3),Random(4)];br=[Random(3),Random(4)]
        for _ in range(20):
            for world in (a,b):
                for row in world.tiles:
                    for t in row:t.food=t.water=0
            self.assertEqual(a.regenerate(cfg,*ar),system.regenerate(b,*br));self.assertEqual(ar[0].getstate(),br[0].getstate());self.assertEqual(ar[1].getstate(),br[1].getstate())
            for r in ('food','water'):self.assertEqual(system.last_supply[r+'_attempted'],system.last_supply[r+'_realized']);self.assertEqual(system.last_supply[r+'_rejected'],0)

    def test_zero_supply_and_food_water_separation(self):
        cfg=Config.evolution(initial_food=0,initial_water=0,food_regeneration_probability=1,water_regeneration_probability=0)
        w=World.generate(cfg,Random(1));s=ResourceEcology(cfg,ResourceEcologyConfig(.5,.75));f=Random(2);water=Random(3);before=water.getstate()
        gained=s.regenerate(w,f,water);self.assertGreater(gained[0],0);self.assertEqual(gained[1],0);self.assertEqual(before,water.getstate())
        zero=ResourceEcology(cfg,ResourceEcologyConfig(0,.75));before=f.getstate();self.assertEqual(zero.regenerate(w,f,water),(0,0));self.assertEqual(before,f.getstate())

    def test_caps_and_mass_balance(self):
        cfg=Config.evolution(initial_food=0,initial_water=0,food_regeneration_probability=1,water_regeneration_probability=1)
        w=World.generate(cfg,Random(1));s=ResourceEcology(cfg,ResourceEcologyConfig(1,.75));food=water=0;fr=Random(2);wr=Random(3)
        for _ in range(20):
            f,u=s.regenerate(w,fr,wr);food+=f;water+=u
        self.assertEqual(food,6000);self.assertEqual(water,6000);self.assertTrue(all(t.food==5 and t.water==5 for row in w.tiles for t in row))

    def test_private_rng_and_same_initial_agents(self):
        cfg=Config.evolution(brain='learning',initial_population=5);before=random.getstate()
        a=Simulation(cfg);b=Simulation(cfg,resource_ecology=ResourceEcologyConfig(.5,.75))
        from experiments.resource_ecology_experiments import initial_agents
        self.assertEqual(initial_agents(a),initial_agents(b));self.assertEqual(before,random.getstate())

    def test_logging_invariance_and_no_patch_features(self):
        cfg=Config.evolution(brain='learning',initial_population=5);eco=ResourceEcologyConfig(.5,.75)
        o=ResourceObserver(lambda k,r:None,'A','E3',PatchGeometry.generate(cfg,eco));args=dict(social_config=SocialConfig(True),social_policy=SocialPolicyConfig(),resource_ecology=eco)
        a=Simulation(cfg,**args);b=Simulation(cfg,**args,resource_observer=o,social_learning_observer=o)
        self.assertEqual({f.name for f in __import__('dataclasses').fields(b.observe(b.agents[0]))},{'tick','tiles','auditory','individuals'})
        for _ in range(12):a.step();b.step();self.assertEqual(fingerprint(a),fingerprint(b))
        o.finish(b);self.assertEqual(fingerprint(a),fingerprint(b));self.assertEqual(b.agents[0].brain.capacity,256)

    def test_exact_actual_and_failed_acquisition_not_reward_sign(self):
        cfg=Config.evolution(brain='learning',world_width=1,world_height=1,initial_population=1,initial_food=1,initial_water=0,hunger_per_tick=0,thirst_per_tick=0,energy_per_tick=0)
        eco=ResourceEcologyConfig(patch_count=1,patch_radius=0);rows=[];o=ResourceObserver(lambda k,r:rows.append((k,r)),'A','E0',PatchGeometry.generate(cfg,eco))
        s=Simulation(cfg,resource_ecology=eco,resource_observer=o,social_learning_observer=o)
        s.config=__import__('dataclasses').replace(cfg,food_regeneration_probability=0,water_regeneration_probability=0)
        s.agents[0].brain.actions=(Action.EAT,)
        s.step();s.step();acq=[r for k,r in rows if k=='acquisitions']
        self.assertEqual(acq[0]['food_consumed'],1);self.assertEqual(acq[0]['reward'],0);self.assertEqual(acq[0]['acquired'],1)
        self.assertEqual(acq[1]['food_consumed'],0);self.assertEqual(acq[1]['action_succeeded'],0)
        o.finish(s)

    def test_death_and_censoring_life_records(self):
        cfg=Config(brain='learning',initial_population=1,max_hunger=1,hunger_per_tick=1);rows=[]
        o=ResourceObserver(lambda k,r:rows.append((k,r)),'A','E0',PatchGeometry.generate(cfg,ResourceEcologyConfig()))
        s=Simulation(cfg,resource_observer=o,social_learning_observer=o);s.step();o.finish(s)
        life=[r for k,r in rows if k=='individuals'];self.assertEqual(len(life),1);self.assertEqual(life[0]['death_cause'],'starvation');self.assertEqual(life[0]['censored'],0)

    def test_distance_metric_and_absence(self):
        w=World(3,3);self.assertEqual(distance_map(w,'food'),[-1]*9);w.tile_at(0,0).food=1;self.assertEqual(distance_map(w,'food')[-1],2)

    def test_gate_default_closed(self):
        with tempfile.TemporaryDirectory() as tmp:self.assertFalse(pilot_passes(tmp))

    def test_gate_rejects_failed_or_modified_evidence(self):
        import pandas as pd
        from simulation.vocal_learning_observer import hash_file
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'frozen.json').write_text('{}');(root/'preregistration.json').write_text('{}')
            p=root/'checks.csv';pd.DataFrame({'valid':[True]*35+[False]}).to_csv(p,index=False)
            receipt={'decision':'primary_allowed','frozen_sha256':hash_file(root/'frozen.json'),'preregistration_sha256':hash_file(root/'preregistration.json'),'checks_file':str(p),'checks_sha256':hash_file(p),'flags':['valid'],'inputs':{}}
            (root/'pilot_review.json').write_text(json.dumps(receipt));self.assertFalse(pilot_passes(root))
            pd.DataFrame({'valid':[True]*36}).to_csv(p,index=False);self.assertFalse(pilot_passes(root))

    def test_factorial_interaction_exact(self):
        import pandas as pd
        from experiments.analyze_resource_ecology import factorial
        rows=[{'seed':seed,'ecology':e,'mode':m,'value':{'E0':1,'E1':2,'E2':4,'E3':10}[e]+i} for seed in (0,1,2) for e in ('E0','E1','E2','E3') for i,m in enumerate('ABC')]
        f=factorial(pd.DataFrame(rows),['value']);a=f[f.family.eq('interaction')]
        self.assertTrue(a.difference.eq(5).all());self.assertEqual(len(a),9)
        self.assertTrue(f[f.family.eq('moderation')].difference.eq(0).all())

    def test_survivors_are_censored_not_deaths(self):
        import pandas as pd
        from experiments.analyze_resource_ecology import founder_survival
        f=pd.DataFrame({'founder':[1,1,0],'duration':[30,100,3],'censored':[0,1,1]})
        r=founder_survival(f,100);self.assertEqual(r['founder_rmst'],65);self.assertEqual(r['founder_censored'],1);self.assertEqual(r['founder_n'],2)
        with self.assertRaises(ValueError):founder_survival(f,500)

    def test_resource_access_gini(self):
        from experiments.analyze_resource_ecology import gini
        self.assertIsNone(gini([]));self.assertEqual(gini([0,0]),0);self.assertEqual(gini([1,1]),0);self.assertEqual(gini([0,1]),.5)

    def test_initial_field_same_with_and_without_scarcity(self):
        cfg=Config.evolution(random_seed=9,initial_population=1)
        a=Simulation(cfg,resource_ecology=ResourceEcologyConfig(1,.75));b=Simulation(cfg,resource_ecology=ResourceEcologyConfig(.5,.75))
        self.assertEqual([(t.food,t.water) for row in a.world.tiles for t in row],[(t.food,t.water) for row in b.world.tiles for t in row])

    def test_future_acquisition_uses_actual_events_and_censors(self):
        import pandas as pd
        from experiments.resource_ecology_prediction import targets
        f=pd.DataFrame({'agent_id':[0]*7,'tick':range(1,8),'reward':[0]*7,'terminal':[0]*6+[1],
            'action':[4]*7,'acquired':[0,0,1,0,0,0,0],'food_near3_tickstart':[1]*7,'water_near3_tickstart':[0]*7})
        values,reasons=targets(f);self.assertEqual(values['future5_acquisition'][0],1)
        self.assertEqual(values['future5_acquisition'][1],1);self.assertEqual(values['future5_acquisition'][2],-1)
        self.assertEqual(reasons[2],'death_censored')

    def test_vocal_occupancy_is_observational(self):
        cfg=Config.evolution(brain='learning',initial_population=3);rows=[]
        o=ResourceObserver(lambda k,r:rows.append((k,r)),'A','E0',PatchGeometry.generate(cfg,ResourceEcologyConfig()))
        s=Simulation(cfg,resource_observer=o,social_learning_observer=o);s.step()
        row=[r for k,r in rows if k=='resource_ticks'][-1]
        self.assertEqual(row['vocal_q_rows'],sum(h.brain.vocal_memory_size for h in s.agents));o.finish(s)

    def test_recording_streams_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            streams=[]
            for i in (0,1):
                cfg=Config.evolution(brain='learning',initial_population=2,random_seed=2);eco=ResourceEcologyConfig(.5,.75)
                p=Path(tmp)/f'{i}.csv.gz'
                with ExitStack() as stack:
                    w=ResourceWriter.open(stack,p);o=ResourceObserver(w,'A','E3',PatchGeometry.generate(cfg,eco));s=Simulation(cfg,resource_ecology=eco,resource_observer=o,social_learning_observer=o)
                    for _ in range(3):s.step()
                    o.finish(s)
                from simulation.resource_observer import paths
                streams.append({k:v.read_bytes() for k,v in paths(p).items()})
            self.assertEqual(*streams)

if __name__=='__main__':unittest.main()
