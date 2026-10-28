"""Read-only instrumentation, exact choice/update semantics, and controlled replay."""
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, replace
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from agents.learning import LearningBrain, Physiology
from agents.observation import Observation, TileObservation
from config import Config
from experiments.analyze_vocal_learning import analyze, analyze_run, equal_seed, fixed_window_pairs, join_communication
from experiments.vocal_learning_data import discover
from simulation.simulation import Simulation
from simulation.vocal_learning_observer import (VocalLearningObserver, CSVVocalLearningWriter, SCHEMAS,
                                                q_diagnostics, hash_file, recording_paths)
from tests.auditory_control import encoded
from tests.test_learning import make_brain
from tests.test_listening import identified
from tests.test_vocalization import vocal_state, FIXTURES


def observer(config):
    rows = defaultdict(list)
    return VocalLearningObserver(config, lambda kind,row: rows[kind].append(dict(zip(SCHEMAS[kind],row)))), rows


def record(directory, name='run', ticks=7, seed=0, **settings):
    path = Path(directory)/(name+'.csv.gz')
    cmd=[sys.executable,'main.py','--brain','learning','--population','2','--ticks',str(ticks),
         '--seed',str(seed),'--vocal-learning-csv',str(path)]
    for name,value in settings.items():
        cmd.append(('--' if value else '--no-')+name.replace('_','-'))
    subprocess.run(cmd,check=True,capture_output=True)
    return path


class QDiagnosticTests(unittest.TestCase):
    def test_uniform_control(self):
        q=q_diagnostics([0.0]*17,.2)
        self.assertEqual(q[:7],(0,0,0,0,0,17,1))
        self.assertEqual(q[8:10],(1,1))
        self.assertAlmostEqual(q[-1],1/17)

    def test_unique_signal_control(self):
        values=[0.0]*17;values[12]=2
        q=q_diagnostics(values,.2)
        self.assertEqual((q[4],q[5],q[8],q[9]),(2,1,2,0))
        self.assertEqual(q[-1],.2/17)

    def test_unique_silence_control(self):
        q=q_diagnostics([2.0]+[0.0]*16,.2)
        self.assertEqual(q[4:7],(2,1,0))
        self.assertEqual(q[-1],.8+.2/17)

    def test_multiple_maxima_control(self):
        q=q_diagnostics([2,0,2,0],.2)
        self.assertEqual(q[4:7],(0,2,0))
        self.assertEqual(q[-1],.2/4+.8/2)

    def test_exact_ties_no_tolerance(self):
        q=q_diagnostics([1.0,1.0+1e-14],.2)
        self.assertEqual(q[5],1)
        self.assertGreater(q[4],0)
        self.assertEqual(q[-1],.1)

    def test_read_only_no_q_allocation_lru_or_rng(self):
        brain,human,cfg=make_brain()
        brain.choose_action(human,Observation(1,()))
        before=(tuple((k,tuple(v)) for k,v in brain._vocal_values.items()),brain.rng.getstate(),brain.vocal_rng.getstate())
        with patch.object(random.Random,'random',side_effect=AssertionError('RNG probe')):
            for _ in range(5):
                q_diagnostics(next(iter(brain._vocal_values.values())),brain.epsilon)
                brain.vocal_action_values(Physiology.capture(human),Observation(2,()))
        self.assertEqual(before,(tuple((k,tuple(v)) for k,v in brain._vocal_values.items()),brain.rng.getstate(),brain.vocal_rng.getstate()))


class HookTests(unittest.TestCase):
    def sequence(self, capacity=2, directions=(1,-1,2,1)):
        brain,human,cfg=make_brain(learning_uses_auditory=False,vocal_learning_memory_capacity=capacity)
        obs,rows=observer(cfg)
        for tick,dx in enumerate(directions,1):
            obs.begin(human,tick)
            brain.choose_action(human,Observation(tick,(TileObservation(dx,0,1,0),)))
            brain.observe_outcome(Physiology.capture(human))
        return brain,human,obs,rows

    def test_first_stored_and_post_eviction(self):
        _,_,_,rows=self.sequence(2,(1,-1,2,1))
        # dx=2 and dx=1 share the encoder's E key; revisit the truly evicted W below.
        self.assertEqual([r['novelty'] for r in rows['decisions']],['first','first','stored','stored'])
        _,_,_,rows=self.sequence(1,(1,-1,1))
        self.assertEqual([r['novelty'] for r in rows['decisions']],['first','first','recreated'])
        self.assertEqual([r['state_id'] for r in rows['decisions']],[0,1,0])

    def test_actual_evictions_and_allocation_epochs(self):
        brain,_,_,rows=self.sequence(1,(1,-1,1))
        self.assertEqual(len([r for r in rows['rows'] if r['event']=='evict']),2)
        self.assertEqual(len([r for r in rows['rows'] if r['event']=='allocate']),3)
        self.assertEqual(rows['decisions'][-1]['evictions'],2)
        self.assertEqual(rows['decisions'][-1]['allocations'],3)
        self.assertEqual(rows['decisions'][-1]['newly_allocated'],1)
        self.assertEqual(rows['rows'][-1]['epoch'],2)
        self.assertEqual(brain.vocal_memory_size,1)

    def test_single_state_repeated(self):
        _,_,_,rows=self.sequence(3,(1,)*10)
        self.assertEqual(sum(r['present_before_selection'] for r in rows['decisions']),9)
        self.assertEqual(rows['decisions'][-1]['unique_states'],1)
        self.assertEqual(sum(r['event']=='allocate' for r in rows['rows']),1)
        self.assertEqual(sum(r['event']=='evict' for r in rows['rows']),0)

    def test_eviction_tracks_row_update_and_visit_counts(self):
        _,_,_,rows=self.sequence(1,(1,1,-1))
        eviction=next(r for r in rows['rows'] if r['event']=='evict')
        self.assertEqual((eviction['visits'],eviction['updates']),(2,2))

    def test_terminal_update_and_death_are_not_evictions(self):
        cfg=Config(brain='learning',initial_population=2,max_age=1)
        obs,rows=observer(cfg)
        sim=Simulation(cfg,vocal_learning_observer=obs)
        sim.step();obs.finish(sim.agents,sim.tick)
        self.assertEqual(sim.population,0)
        self.assertEqual([r['terminal'] for r in rows['updates']],[1,1])
        self.assertEqual(sum(r['event']=='evict' for r in rows['rows']),0)
        self.assertEqual(sum(r['event']=='death' for r in rows['rows']),2)
        self.assertEqual(obs.agents,{})

    def test_run_end_is_not_eviction_and_releases_history(self):
        brain,human,obs,rows=self.sequence(3,(1,1))
        before=list(brain._vocal_values.items())
        obs.finish([human],2);obs.finish([human],2)
        self.assertEqual([r['event'] for r in rows['rows']],['allocate','run_end'])
        self.assertEqual(obs.agents,{})
        self.assertEqual(list(brain._vocal_values.items()),before)
        self.assertIsNone(brain._vocal_observer)

    def test_td_error_and_actual_q_change(self):
        brain,human,cfg=make_brain(learning_uses_auditory=False)
        obs,rows=observer(cfg)
        obs.begin(human,1);brain.choose_action(human,Observation(1,()))
        transition=brain._vocal_pending
        brain.observe_outcome(Physiology(30,40,70,True))
        reward=brain.last_reward
        obs.begin(human,2);brain.choose_action(human,Observation(2,()))
        update=rows['updates'][0]
        self.assertEqual(update['target'],reward)
        self.assertEqual(update['td_error'],reward)
        self.assertEqual(update['new_q'],brain._vocal_values[transition.state][transition.action_index])
        self.assertEqual(update['q_change'],abs(cfg.learning_rate*reward))
        self.assertEqual((update['visits'],update['updates'],update['terminal']),(1,1,0))
        self.assertEqual(rows['decisions'][-1]['q_min'],min(brain._vocal_values[transition.state]))

    def test_updated_evicted_recreated_row_is_marked(self):
        brain,human,cfg=make_brain(vocal_learning_memory_capacity=1,learning_uses_auditory=False)
        obs,rows=observer(cfg)
        obs.begin(human,1);brain.choose_action(human,Observation(1,()))
        brain.observe_outcome(Physiology.capture(human))
        # Controlled direct row turnover while preserving a pending transition.
        brain._remember_vocal((9,)*7)
        obs.begin(human,2);brain.choose_action(human,Observation(2,()))
        self.assertEqual(rows['updates'][-1]['recreated'],1)
        self.assertEqual(rows['updates'][-1]['epoch'],2)

    def test_distinct_auditory_keys_only_with_listening(self):
        for listening in (False,True):
            brain,human,cfg=make_brain(learning_uses_auditory=listening)
            obs,rows=observer(cfg)
            for t,sig in enumerate((1,2),1):
                obs.begin(human,t)
                brain.choose_action(human,Observation(t,(),identified(sig)))
                brain.observe_outcome(Physiology.capture(human))
            self.assertEqual(len(obs.keys),2 if listening else 1)
            self.assertEqual({len(k) for k in obs.keys},{8 if listening else 7})

    def test_a_b_random_not_applicable_c_d_record_q(self):
        for name,cfg in [('A',Config(brain='learning',learning_uses_auditory=False,learning_controls_vocalization=False)),
                         ('B',Config(brain='learning',learning_controls_vocalization=False)),
                         ('C',Config(brain='learning',learning_uses_auditory=False)),
                         ('D',Config(brain='learning')),('random',Config())]:
            obs,rows=observer(cfg)
            sim=Simulation(replace(cfg,initial_population=1),vocal_learning_observer=obs);sim.step()
            row=rows['decisions'][0]
            self.assertEqual(row['condition'],name)
            self.assertEqual(row['state_id'] is not None,name in 'CD')
            if name not in 'CD':
                self.assertTrue(all(row[k] is None for k in SCHEMAS['decisions'][5:]))
                self.assertEqual(dict(rows).get('updates',[]),[])


class IsolationTests(unittest.TestCase):
    def test_all_historical_fixtures_and_report_are_unchanged(self):
        expected = {
            'v02_seed0_tick100.json':'5c3bf17c0ab4014d1ec98e823b239d860b74f92cc1394ab89c3bad73c5594e3b',
            'v03_auditory_control.json':'ba85a1314e1c68a3588f4ba7b6d698eca16d22b706de307850ff625ebf17260b',
            'v04_listening_control.json':'4b4b9e82362698b52a71e08e766cf16795575d4479cf023c8331eacfc624316a',
            'v05_listening_seed7_tick80.json':'5538900edb7d38eb74ca06b28602f213e45b3efbdbf8528130eb804cd7459b44',
            'v06_vocal_seed7_tick80.json':'0098529d6812003f97d718b3e1f4927b88fd371775047f82b640df96ec048dcd',
            'v061_communication_toys.json':'d397488e3f476b26d3e7b20c8714fd16a1d452b9c27b57b68fe03aeb955130da'}
        for name,digest in expected.items():
            self.assertEqual(hash_file(FIXTURES/name),digest)
        self.assertEqual(hash_file('docs/communication_analysis_v061.md'),
                         'd8741c7eb72483a9e924db2bdf8060f450502058f9842dfb8b6f960727f276fd')

    def test_every_tick_all_rng_q_lru_pending_memory_events_both_observers(self):
        original=Config(**json.loads((FIXTURES/'v06_vocal_seed7_tick80.json').read_text())['config'])
        before=random.getstate()
        for listening in (False,True):
            for vocal in (False,True):
                cfg=replace(original,learning_uses_auditory=listening,learning_controls_vocalization=vocal)
                events=[[],[],[]];comm=[[],[]]
                off=Simulation(cfg,event_sink=events[0].append,communication_sink=comm[0].append)
                obs,_=observer(cfg);both=Simulation(cfg,event_sink=events[1].append,communication_sink=comm[1].append,vocal_learning_observer=obs)
                obs2,_=observer(cfg);only=Simulation(cfg,event_sink=events[2].append,vocal_learning_observer=obs2)
                for _ in range(25):
                    off.step();both.step();only.step()
                    self.assertEqual(vocal_state(off),vocal_state(both))
                    self.assertEqual(vocal_state(off),vocal_state(only))
                obs.finish(both.agents,both.tick)
                self.assertEqual(vocal_state(off),vocal_state(both))
                self.assertEqual(events[0],events[1]);self.assertEqual(events[0],events[2]);self.assertEqual(comm[0],comm[1])
        self.assertEqual(before,random.getstate())

    def test_existing_v06_fixture_with_new_observer(self):
        f=json.loads((FIXTURES/'v06_vocal_seed7_tick80.json').read_text())
        cfg=Config(**f['config']);obs,_=observer(cfg)
        trajectory,events=hashlib.sha256(),hashlib.sha256()
        sim=Simulation(cfg,vocal_learning_observer=obs,event_sink=lambda e: events.update(encoded(asdict(e))))
        for t in range(f['ticks']+1):
            if t:sim.step()
            trajectory.update(encoded(vocal_state(sim)))
        self.assertEqual(trajectory.hexdigest(),f['result']['trajectory_sha256'])
        self.assertEqual(events.hexdigest(),f['result']['events_sha256'])
        self.assertEqual(hashlib.sha256(encoded(vocal_state(sim))).hexdigest(),f['result']['final_state_sha256'])

    def test_random_diagnostic_on_off_replay(self):
        cfg=Config.evolution(initial_population=4,random_seed=3)
        obs,_=observer(cfg);a,b=Simulation(cfg),Simulation(cfg,vocal_learning_observer=obs)
        for _ in range(10):
            a.step();b.step();self.assertEqual(vocal_state(a),vocal_state(b))

    def test_default_does_not_create_observer_or_history(self):
        sim=Simulation(Config(brain='learning',initial_population=1))
        sim.step()
        self.assertIsNone(sim.vocal_learning_observer)
        self.assertIsNone(sim.agents[0].brain._vocal_observer)

    def test_cli_aggregate_and_communication_bytes_identical(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for mode in ('off','on'):
                cmd=[sys.executable,'main.py','--brain','learning','--population','3','--ticks','9','--seed','7',
                     '--csv',str(root/(mode+'.csv')),'--communication-csv',str(root/(mode+'_comm.csv'))]
                if mode=='on':cmd+=['--vocal-learning-csv',str(root/'vocal.csv.gz')]
                subprocess.run(cmd,check=True,capture_output=True)
            self.assertEqual((root/'off.csv').read_bytes(),(root/'on.csv').read_bytes())
            self.assertEqual((root/'off_comm.csv').read_bytes(),(root/'on_comm.csv').read_bytes())
            self.assertEqual(len(next(csv.reader(io.StringIO((root/'on_comm.csv').read_text())))),27)

    def test_path_collisions_rejected_including_companions(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)/'v.csv'
            r=subprocess.run([sys.executable,'main.py','--ticks','0','--vocal-learning-csv',str(base),
                              '--csv',str(Path(d)/'v.updates.csv')],capture_output=True)
            self.assertNotEqual(r.returncode,0)


class AnalysisTests(unittest.TestCase):
    def test_actual_extinct_recording_stops_at_death_tick(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'extinct.csv.gz'
            cfg=Config(brain='learning',initial_population=2,max_age=1)
            with ExitStack() as stack:
                writer=CSVVocalLearningWriter.open(stack,path)
                obs=VocalLearningObserver(cfg,writer)
                sim=Simulation(cfg,vocal_learning_observer=obs)
                sim.step();sim.step();obs.finish(sim.agents,sim.tick)
            run={'paths':recording_paths(path),'metadata':{'config':asdict(cfg),'final_tick':sim.tick,
                  'files':{k:{'records':n} for k,n in writer.counts.items()}},
                 'condition':'D','seed':42,'run':'extinct','applicable':True}
            result=analyze_run(run,100,25,1,1)
            self.assertEqual(result['windows'].actual_window_end.tolist(),[1])
            self.assertEqual(result['runs'].iloc[0].terminal_updates,2)
            self.assertEqual(result['runs'].iloc[0].deaths,2)
            self.assertEqual(result['runs'].iloc[0].evictions,0)

    def test_duplicate_decision_detected_even_with_correct_file_hash(self):
        import gzip
        with tempfile.TemporaryDirectory() as d:
            path=record(d,ticks=2)
            with gzip.open(path,'rt') as stream:rows=list(csv.reader(stream))
            rows[2][1]=rows[1][1]
            with gzip.open(path,'wt',newline='') as stream:csv.writer(stream).writerows(rows)
            side=Path(str(path)+'.metadata.json');meta=json.loads(side.read_text())
            meta['files']['decisions'].update(sha256=hash_file(path),bytes=path.stat().st_size)
            side.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'Duplicate decision'):
                analyze_run(discover(d)[0],100,25,1,1)

    def test_missing_windows_are_not_interpolated(self):
        with tempfile.TemporaryDirectory() as d:
            record(d,ticks=1)
            run=discover(d)[0]
            run['metadata']['final_tick']=3
            rows=analyze_run(run,1,25,1,1)['windows']
            self.assertEqual(rows.decisions.tolist(),[2,0,0])
            self.assertTrue(rows.hit_rate.iloc[1:].isna().all())
            self.assertEqual(rows.observed_ticks.tolist(),[1,0,0])

    def test_fixed_window_pairs_require_same_id_and_exposure(self):
        import pandas as pd
        rows=[]
        for aid,start in [(1,0),(1,100),(2,0),(3,100)]:
            rows.append({'agent_id':aid,'window_start':start,'decisions':30,
                         **{m:start/100 for m in ('hit_rate','mean_q_gap','mean_p_silence','nonuniform_fraction','unique_max_fraction','mean_age')}})
        pairs,summary=fixed_window_pairs(pd.DataFrame(rows),100,200,20)
        self.assertEqual(set(pairs.agent_id),{1})
        self.assertTrue(pairs.delta_hit_rate.eq(1).all())
        self.assertTrue(summary.paired_agents.eq(1).all())

    def test_novelty_partition_and_exposure_accounting(self):
        with tempfile.TemporaryDirectory() as d:
            record(d,ticks=10)
            run=discover(d)[0];tables=analyze_run(run,3,2,1,1)
            w=tables['windows']
            import numpy as np
            np.testing.assert_allclose(w[['first_fraction','stored_fraction','recreated_fraction']].sum(axis=1),1)
            self.assertEqual(w.decisions.sum(),20)
            self.assertEqual(w.allocations.sum(),run['metadata']['files']['rows']['records']-2)
            self.assertEqual(tables['runs'].iloc[0].run_ends,2)

    def test_join_requires_exact_trajectory_and_full_config(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=record(root,ticks=1)
            run=discover(root)[0]
            (root/'analysis_metadata.json').write_text(json.dumps({'inputs':[]}))
            pd.DataFrame(columns=['run','scope','conditioning']).to_csv(root/'sender_information.csv',index=False)
            # A claimed same-seed run is insufficient: it must identify an actual matching recording.
            fake=root/'comm.csv';fake.write_text('x\n')
            side=Path(str(fake)+'.metadata.json')
            side.write_text(json.dumps({'config':run['metadata']['config'],'final_tick':1,'statistics':run['metadata']['statistics']}))
            run['metadata']['communication']={'file':str(fake),'sha256':hash_file(fake),'metadata_sha256':hash_file(side)}
            with self.assertRaisesRegex(ValueError,'No unique identical'):
                join_communication([run],pd.DataFrame(),pd.DataFrame(),root,25)
            meta=json.loads(side.read_text());meta['config']['random_seed']=99;side.write_text(json.dumps(meta))
            run['metadata']['communication']['metadata_sha256']=hash_file(side)
            with self.assertRaisesRegex(ValueError,'trajectory/config'):
                join_communication([run],pd.DataFrame(),pd.DataFrame(),root,25)

    def test_filename_independent_identity_partial_window_and_equal_seed(self):
        with tempfile.TemporaryDirectory() as d:
            record(d,'z',ticks=7,seed=0);record(d,'a',ticks=3,seed=4)
            runs=discover(d)
            self.assertEqual([r['seed'] for r in runs],[0,4])
            data=[analyze_run(r,4,2,1,1) for r in runs]
            self.assertEqual(data[0]['windows'].actual_window_end.tolist(),[4,7])
            self.assertTrue(data[0]['windows'].iloc[-1].partial_window)
            import pandas as pd
            frame=pd.DataFrame({'condition':['C','C'],'seed':[0,1],'window':[0,0],'hit_rate':[.2,.8],'decisions':[1000,1]})
            row=equal_seed(frame,['condition','window']).query("metric == 'hit_rate'").iloc[0]
            self.assertAlmostEqual(row['mean'],.5)
            self.assertAlmostEqual(row['std'],.6/2**.5)
            self.assertEqual(row['n'],2)

    def test_extinction_and_empty_windows(self):
        with tempfile.TemporaryDirectory() as d:
            record(d,ticks=0)
            run=discover(d)[0]
            data=analyze_run(run,100,25,50,20)
            self.assertTrue(data['windows'].empty)
            self.assertEqual(data['runs'].iloc[0].decisions,0)
            self.assertTrue(data['runs'].hit_rate.isna().all())

    def test_schema_hash_and_completion_validation(self):
        for change in ('hash','schema','incomplete'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                path=record(d)
                side=Path(str(path)+'.metadata.json');meta=json.loads(side.read_text())
                if change=='hash':path.write_bytes(path.read_bytes()+b'x')
                if change=='schema':meta['schemas']['decisions'].append('extra')
                if change=='incomplete':meta['completed']=False
                side.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):discover(d)

    def test_duplicate_seed_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            record(d,'a');record(d,'b')
            with self.assertRaises(ValueError):discover(d)

    def test_a_b_metrics_remain_undefined_in_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            record(d,learning_controls_vocalization=False)
            data=analyze_run(discover(d)[0],100,25,1,1)
            self.assertTrue(data['windows'].hit_rate.isna().all())
            self.assertTrue(data['runs'].mean_q_gap.isna().all())
            self.assertEqual(data['runs'].iloc[0].actor_decisions,14)
            for field in ('updates','allocations','evictions','decisions','unique_keys_run'):
                self.assertTrue(data['runs'][field].isna().all(),field)

    def test_recording_gzip_and_analysis_csv_json_png_are_reproducible(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);one=root/'one';two=root/'two'
            record(one);record(two)
            for kind in SCHEMAS:
                self.assertEqual(recording_paths(one/'run.csv.gz')[kind].read_bytes(),recording_paths(two/'run.csv.gz')[kind].read_bytes())
            self.assertEqual((one/'run.csv.gz.metadata.json').read_bytes(),(two/'run.csv.gz.metadata.json').read_bytes())
            before={str(p):hash_file(p) for p in one.iterdir()}
            rng=random.getstate()
            import numpy as np
            numpy_rng=np.random.get_state()
            a,b=root/'analysis_a',root/'analysis_b'
            analyze(one,a,window_size=4,min_decisions=1,min_pair_decisions=1)
            analyze(one,b,window_size=4,min_decisions=1,min_pair_decisions=1)
            self.assertGreaterEqual(len(list(a.glob('*.png'))),10)
            self.assertEqual({p.name:hash_file(p) for p in a.iterdir()},{p.name:hash_file(p) for p in b.iterdir()})
            self.assertEqual(before,{str(p):hash_file(p) for p in one.iterdir()})
            self.assertEqual(rng,random.getstate())
            after=np.random.get_state()
            self.assertEqual(numpy_rng[0],after[0]);np.testing.assert_array_equal(numpy_rng[1],after[1])
            self.assertEqual(numpy_rng[2:],after[2:])
            with self.assertRaises(ValueError):analyze(one,one)


if __name__=='__main__':
    unittest.main()
