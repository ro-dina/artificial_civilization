"""Exact legacy behavior, online isolation, shared histories and fair denominators."""
from collections import Counter
from contextlib import ExitStack,redirect_stdout
from dataclasses import asdict,replace
import csv,gzip,io,json
from pathlib import Path
import random,subprocess,sys,tempfile,unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from config import Config,SocialConfig,SocialPolicyConfig
from simulation.simulation import Simulation
from simulation.social_observer import CSVSocialWriter,write_metadata
from simulation.social_policy_observer import PolicyWriter,write_metadata as policy_metadata
from experiments.auditory_policy import frozen
from experiments.social_verification import state
from experiments.social_history import HistoryCollector,open_stream,write_history_metadata,convert_legacy,load_metadata,iter_history,Opportunity
from experiments.social_policy_metrics import Evaluation,Treatment,primary
from experiments.analyze_social_memory_policies import analyze,treatments,paired
from simulation.vocal_learning_observer import hash_file
from tests.auditory_control import ActionDigest
from tests.test_social_memory import quiet
from tests.test_social_analysis import recording


def private(sim):return tuple((h.id,frozen(vars(h.social_memory))) for h in sim.agents)


def evaluate_ids(sequence,policies=None,window=100):
    e=Evaluation(policies or primary(),window_size=window);seq=0
    all_ids=sorted({0,*[i for ids in sequence for i in ids]})
    for identity in all_ids:
        e.accept(Opportunity(seq,0,'founder',identity,0,0));seq+=1
    e.accept(Opportunity(seq,0,'tick_end',population=len(all_ids),births=0,deaths=0));seq+=1
    for tick,ids in enumerate(sequence,1):
        for identity in all_ids:
            e.accept(Opportunity(seq,tick,'observation',identity,tick-1,0,tuple(ids) if identity==0 else ()));seq+=1
        e.accept(Opportunity(seq,tick,'tick_end',population=len(all_ids),births=0,deaths=0));seq+=1
    return e


class PolicyOnlineTests(unittest.TestCase):
    def test_all_mechanisms_preserve_ecology_q_lru_actions_and_rng(self):
        global_rng=random.getstate()
        for brain,listening in (('random',True),('learning',True),('learning',False)):
            cfg=Config.evolution(brain=brain,learning_uses_auditory=listening,random_seed=7,initial_population=8)
            digest_off=ActionDigest();baseline=Simulation(cfg,event_sink=digest_off)
            digests=[ActionDigest() for _ in primary()]
            sims=[Simulation(cfg,social_config=t.social,social_policy=t.policy,event_sink=d) for t,d in zip(primary(),digests)]
            for tick in range(21):
                if tick:
                    baseline.step()
                    for sim in sims:sim.step()
                for sim,d in zip(sims,digests):
                    self.assertEqual(state(baseline),state(sim));self.assertEqual(digest_off.digest.digest(),d.digest.digest())
                self.assertEqual(global_rng,random.getstate())

    def test_recording_and_trace_do_not_change_candidate_or_private_state(self):
        cfg=quiet(brain='learning',world_width=1,initial_population=3)
        for t in primary():
            off=Simulation(cfg,social_config=t.social,social_policy=t.policy)
            rows=[];on=Simulation(cfg,social_config=t.social,social_policy=t.policy,social_policy_sink=lambda kind,row:rows.append((kind,row)))
            for _ in range(10):
                off.step();on.step();self.assertEqual(state(off),state(on));self.assertEqual(private(off),private(on))
            before=(state(on),private(on),random.getstate());on.social_observer.finish(on.statistics)
            self.assertEqual(before,(state(on),private(on),random.getstate()));self.assertTrue(rows)

    def test_legacy_new_default_keeps_old_csv_bytes(self):
        cfg=quiet(world_width=1,initial_population=3)
        a,b=io.StringIO(),io.StringIO()
        off=Simulation(cfg,social_config=SocialConfig(True),social_sink=CSVSocialWriter(a))
        on=Simulation(cfg,social_config=SocialConfig(True),social_policy=SocialPolicyConfig(),social_sink=CSVSocialWriter(b))
        for _ in range(8):off.step();on.step()
        off.social_observer.finish(off.statistics);on.social_observer.finish(on.statistics)
        self.assertEqual(a.getvalue(),b.getvalue());self.assertEqual(private(off),private(on))

    def test_old_real_100_tick_gzip_hash_and_statistics_match(self):
        raw=Path('data/social_memory/seed_0_social.csv.gz')
        if not raw.exists():self.skipTest('Local archived pilot unavailable')
        meta=json.loads(Path(str(raw)+'.metadata.json').read_text());cfg=Config(**meta['config']['simulation'])
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'same.csv.gz'
            with ExitStack() as stack:
                writer=CSVSocialWriter.open(stack,out)
                sim=Simulation(cfg,social_config=SocialConfig(**meta['config']['social']),social_sink=writer)
                for _ in range(meta['final_tick']):sim.step()
                sim.social_observer.finish(sim.statistics)
            self.assertEqual(hash_file(out),meta['file']['sha256']);self.assertEqual(asdict(sim.statistics),meta['statistics'])
            self.assertEqual(hash_file('agents/social_memory.py'),meta['source_sha256']['agents/social_memory.py'])

    def test_newborn_candidates_empty_and_no_inheritance(self):
        from agents.brain import Action
        from tests.test_statistics import FixedBrain
        cfg=quiet(world_width=1,reproduction_enabled=True,min_reproductive_age=0)
        sim=Simulation(cfg,social_config=SocialConfig(True),social_policy=SocialPolicyConfig(5),brain_factory=lambda r,c:FixedBrain(Action.REPRODUCE))
        sim.agents[0].reproductive_type,sim.agents[1].reproductive_type=0,1
        sim.step();child=sim.agents[-1]
        self.assertEqual(len(child.social_memory),0);self.assertEqual(child.social_memory.candidates(),())
        self.assertNotEqual(child.social_memory.candidates(),sim.agents[0].social_memory.candidates())
        self.assertIsNot(child.social_memory,sim.agents[0].social_memory)

    def test_policy_validation_prevents_mislabeled_legacy_format(self):
        with self.assertRaises(ValueError):Simulation(quiet(),social_policy=SocialPolicyConfig(5))
        with self.assertRaises(ValueError):Simulation(quiet(),social_config=SocialConfig(True),social_policy=SocialPolicyConfig(5),social_sink=lambda r:None)
        with self.assertRaises(ValueError):Simulation(quiet(),social_config=SocialConfig(True),social_policy=False)

    def test_policy_csv_metadata_and_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'new.csv.gz'
            args=[sys.executable,'main.py','--brain','learning','--population','2','--ticks','5','--social-memory',
                '--social-formation-threshold','5','--social-eviction-policy','contact_decay','--social-policy-csv',str(path),'--social-policy-trace']
            subprocess.run(args,capture_output=True,check=True)
            m=json.loads(Path(str(path)+'.metadata.json').read_text())
            self.assertEqual(m['version'],'0.7.1');self.assertTrue(m['completed']);self.assertEqual(m['config']['policy']['formation_threshold'],5)
            self.assertEqual(m['files']['summary']['sha256'],hash_file(path));self.assertIn('trace',m['files'])
            failed=subprocess.run(args,capture_output=True);self.assertNotEqual(failed.returncode,0)


class HistoryReplayTests(unittest.TestCase):
    def test_cli_compact_history_captures_exact_empty_and_nonempty_opportunities(self):
        with tempfile.TemporaryDirectory() as folder:
            h=Path(folder)/'h.csv.gz'
            subprocess.run([sys.executable,'main.py','--population','2','--ticks','3','--social-memory',
                            '--social-history-csv',str(h)],capture_output=True,check=True)
            rows=list(iter_history(h));self.assertEqual(sum(r.kind=='observation' for r in rows),6)
            self.assertEqual(load_metadata(h)['final_tick'],3)

    def test_convert_includes_empty_opportunities_and_all_unknown_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);raw,_=recording(p/'old',cfg=quiet(world_width=1,max_ticks=8),ticks=8)
            h=p/'new'/'h.csv.gz';meta,_=convert_legacy(raw,h)
            rows=list(iter_history(h));obs=[r for r in rows if r.kind=='observation']
            self.assertEqual(len(obs),16);self.assertTrue(all(len(r.visible_ids)==1 for r in obs))
            self.assertEqual(meta['final_tick'],8)
            raw,_=recording(p/'empty',cfg=quiet(initial_population=1,max_ticks=3),ticks=3)
            h=p/'empty-new.csv.gz';convert_legacy(raw,h)
            self.assertEqual([r.visible_ids for r in iter_history(h) if r.kind=='observation'],[(),(),()])

    def test_online_and_replay_memory_equal_at_every_actor_opportunity(self):
        from agents.brain import Action
        from tests.test_statistics import FixedBrain
        cfg=quiet(world_width=1,initial_population=3,max_ticks=10,reproduction_enabled=True,min_reproductive_age=0,max_age=6)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'h.csv.gz';snapshots=[{} for _ in primary()];sims=[]
            for index,t in enumerate(primary()):
                holder={}
                def event_sink(event,index=index,holder=holder):
                    if event.kind=='agent_step':
                        human=next(h for h in holder['sim'].agents if h.id==event.agent_id)
                        snapshots[index][(event.tick,event.agent_id)]=frozen(vars(human.social_memory))
                sim=Simulation(cfg,social_config=t.social,social_policy=t.policy,event_sink=event_sink,
                               brain_factory=lambda r,c:FixedBrain(Action.REPRODUCE))
                holder['sim']=sim;sims.append(sim)
            with ExitStack() as stack:
                collector=HistoryCollector(open_stream(stack,path))
                base=Simulation(cfg,social_config=SocialConfig(True),social_sink=collector,
                                brain_factory=lambda r,c:FixedBrain(Action.REPRODUCE))
                for sim in [base,*sims]:sim.agents[0].reproductive_type,sim.agents[1].reproductive_type=0,1
                for _ in range(10):
                    if not base.population:break
                    base.step()
                    for sim in sims:sim.step()
                base.social_observer.finish(base.statistics)
            write_history_metadata(path,collector,cfg,SocialConfig(True),10,{'kind':'controlled_test'})
            evaluation=Evaluation(primary());checked=0
            for row in iter_history(path):
                evaluation.accept(row)
                if row.kind=='observation':
                    for index in range(4):
                        self.assertEqual(snapshots[index][(row.tick,row.agent_id)],
                                         frozen(vars(evaluation.agents[row.agent_id].memories[index])))
                    checked+=1
            self.assertGreater(checked,5);self.assertGreater(base.statistics.births,0);self.assertGreater(base.statistics.deaths,0)
            evaluation.finish()

    def test_prefix_has_complete_last_tick_and_no_fabricated_extension(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);raw,_=recording(p/'old',cfg=quiet(world_width=1,max_ticks=10),ticks=10)
            h=p/'prefix.csv.gz';m,_=convert_legacy(raw,h,5)
            self.assertEqual(m['final_tick'],5);self.assertEqual(m['provenance']['original_final_tick'],10)
            self.assertEqual([r.tick for r in iter_history(h) if r.kind=='tick_end'],list(range(6)))
            with self.assertRaises(ValueError):convert_legacy(raw,p/'invalid.csv.gz',11)
            with self.assertRaises(ValueError):list(iter_history(h,horizon=6))

    def test_corrupt_history_and_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);raw,_=recording(p/'old');h=p/'h.csv.gz';convert_legacy(raw,h)
            m=load_metadata(h);h.write_bytes(h.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):load_metadata(h)

    def test_observer_death_releases_private_and_offline_histories(self):
        e=evaluate_ids([(1,)]*5)
        e.accept(Opportunity(999,5,'death',0,5,0,death_cause='old_age'))
        self.assertNotIn(0,e.agents)
        e.finish();self.assertEqual(e.agents,{})


class RecognitionTests(unittest.TestCase):
    def test_fifth_contact_not_eligible_before_update_or_recognized_on_commit(self):
        e=evaluate_ids([(1,)]*5)
        for t in e.total:self.assertEqual(t.c['eligible_contacts'],0)
        self.assertEqual(e.total[1].c['previously_committed_contacts'],0)
        self.assertIsNone(e.total[1].values()['recognition_eligible'])
        e=evaluate_ids([(1,)]*6)
        for t in e.total:self.assertEqual(t.c['eligible_contacts'],1);self.assertEqual(t.c['eligible_hits'],1)
        self.assertEqual(e.total[1].c['repeated_contacts'],5);self.assertEqual(e.total[1].c['repeated_hits'],1)

    def test_shared_eligibility_fairly_detects_formation_filter_benefit(self):
        ts=(Treatment('A',capacity=1),Treatment('B',capacity=1,threshold=5))
        e=evaluate_ids([(1,)]*5+[(2,),(1,)],ts)
        self.assertEqual([t.c['eligible_contacts'] for t in e.total],[1,1])
        self.assertEqual([t.c['eligible_hits'] for t in e.total],[0,1])
        self.assertEqual(e.total[0].c['eligible_forgotten_misses'],1)

    def test_never_committed_candidate_is_not_forgotten_long_term_memory(self):
        ts=(Treatment('B',capacity=2,threshold=5,candidate_capacity=1),)
        e=evaluate_ids([(1,2)]*6,ts);c=e.total[0].c
        self.assertEqual(c['eligible_contacts'],2);self.assertEqual(c['eligible_hits'],0)
        self.assertEqual(c['eligible_never_committed_misses'],2);self.assertEqual(c['eligible_forgotten_misses'],0)
        self.assertEqual(c['allocations'],0);self.assertGreater(c['candidate_eviction'],0)

    def test_qualifying_reunion_denominator_independent_of_memory_condition(self):
        e=evaluate_ids([(1,)]*5+[()]*19+[(1,)])
        for t in e.total:
            self.assertEqual(t.c['qualifying_reunion_contacts'],1);self.assertEqual(t.c['qualifying_reunion_hits'],1)

    def test_contact_episode_lengths_and_censoring(self):
        e=evaluate_ids([(1,)]*3+[()]+[(1,)]*2);e.finish()
        self.assertEqual(e.source_hist[('episode_length','observed_absence',3)],1)
        self.assertEqual(e.source_hist[('episode_length','run_end_censored',2)],1)

    def test_retention_duration_eviction_separate_from_run_end(self):
        e=evaluate_ids([(1,)]*5+[(2,)],(Treatment('A',capacity=1),));e.finish()
        h=e.total[0].hist
        self.assertEqual(h[('retention_duration','eviction')][5],1)
        self.assertEqual(h[('retention_duration','run_end_censored')][0],1)

    def test_window_cumulative_conversion_has_no_future_leakage(self):
        e=evaluate_ids([(1,)]*6,window=2)
        self.assertEqual(e.window_cumulative[(1,0)],(0,1,0,1))
        self.assertEqual(e.window_cumulative[(1,1)],(0,1,0,1))
        self.assertEqual(e.window_cumulative[(1,2)],(1,1,1,1))

    def test_candidate_and_living_capacity_exposure_distinct(self):
        e=evaluate_ids([(1,)]*4)
        self.assertEqual(e.total[1].c['occupancy_sum'],0)
        self.assertEqual(e.total[1].c['candidate_occupancy_sum'],4)
        self.assertEqual(e.total[0].c['living_agent_ticks'],8)


class PolicyAnalysisTests(unittest.TestCase):
    def test_no_contacts_is_unavailable_recognition_not_zero_effect(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            p=Path(folder);raw,_=recording(p/'old',cfg=quiet(initial_population=1,max_ticks=3),ticks=3)
            h=p/'h.csv.gz';convert_legacy(raw,h)
            analyze([h],p/'out',horizon=3)
            frame=pd.read_csv(p/'out/run_summary.csv')
            self.assertTrue(frame.recognition_eligible.isna().all());self.assertTrue(frame.eligible_contacts.eq(0).all())
            self.assertTrue(frame.mean_occupancy.eq(0).all())
            pairs=pd.read_csv(p/'out/paired_by_seed.csv');self.assertTrue(pairs[pairs.metric.eq('recognition_eligible')].difference.isna().all())

    def test_runner_is_bounded_and_requires_review(self):
        from experiments.social_memory_experiments import run
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            for kwargs in (dict(phase='pilot',ticks=101),dict(phase='primary',ticks=1000),dict(phase='primary')):
                with self.assertRaises(ValueError):run(data_dir=p/'data',output_root=p/'out',**kwargs)

    def test_treatment_parameter_grid_and_bounded_candidate_budget(self):
        ts=treatments(('A','D'),(8,16,32,64,128),(100.,),32)
        self.assertEqual(len(ts),10);self.assertTrue(all(t.policy.resolved_candidate_capacity(t.capacity)==32 for t in ts))
        self.assertEqual(len(treatments(('D',),(32,),(20.,50.,100.,200.))),4)
        for args in ((('X',),(32,),(100.,)),(('A',),(0,),(100.,)),(('D',),(32,),(0.,))):
            with self.assertRaises(ValueError):treatments(*args)

    def test_full_analysis_replay_csv_json_png_rng_and_no_simulation(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            p=Path(folder);hist=[]
            for seed in (0,1):
                raw,_=recording(p/f'old{seed}',cfg=quiet(world_width=1,random_seed=seed,max_ticks=8),ticks=8)
                h=p/'hist'/f'{seed}.csv.gz';convert_legacy(raw,h);hist.append(h)
            original={h.name:hash_file(h) for h in hist};rng=random.getstate();nrng=np.random.get_state()
            with patch.object(Simulation,'step',side_effect=AssertionError):
                analyze(hist,p/'a',horizon=8,window_size=4);analyze(hist[::-1],p/'b',horizon=8,window_size=4)
            files=sorted(f.name for f in (p/'a').iterdir())
            self.assertEqual(len([f for f in files if f.endswith('.png')]),10)
            for name in files:self.assertEqual(hash_file(p/'a'/name),hash_file(p/'b'/name),name)
            self.assertEqual(original,{h.name:hash_file(h) for h in hist});self.assertEqual(rng,random.getstate())
            np.testing.assert_array_equal(nrng[1],np.random.get_state()[1]);self.assertEqual(nrng[2:],np.random.get_state()[2:])
            f=pd.read_csv(p/'a/run_summary.csv');self.assertEqual(set(f.eligible_contacts),{6})
            summary=pd.read_csv(p/'a/seed_summary.csv');self.assertEqual(set(summary[summary.metric.eq('recognition_eligible')]['n']),{2})
            with self.assertRaises(ValueError):analyze(hist,p/'a',horizon=8)


if __name__=='__main__':unittest.main()
