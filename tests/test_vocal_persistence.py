"""Synthetic longitudinal diagnostics only: no new ecological simulations."""
from collections import defaultdict
from dataclasses import asdict
import csv
import gzip
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from experiments.analyze_vocal_persistence import analyze,equal_seed,verified_communication
from experiments.vocal_persistence import Options,RunAnalysis,hist_quantile
from experiments.vocal_persistence_data import catalog,decode_state,discover
from experiments.vocal_persistence_metrics import infer_maxima
from simulation.vocal_learning_observer import q_diagnostics,hash_file,recording_paths
from tests.vocal_persistence_fixtures import values,write_run,historical_sources


class Sink:
    def __init__(self):self.rows=defaultdict(list)
    def add(self,name,row):self.rows[name].append(row)


def process(directory,options=None):
    run=discover(directory)[0];catalog(run);sink=Sink()
    result=RunAnalysis(run,options or Options(window_size=2),sink).process()
    return sink.rows,result


def probe(q,selected=4,explore=False):
    d=q_diagnostics(q,.2)
    return SimpleNamespace(n_actions=len(q),max_ties=d[5],all_equal=d[6],q_max=d[0],q_min=d[1],q_gap=d[4],
        silence_q=d[7],silence_is_max=d[9],selected_action=selected,selected_q=q[selected],exploratory=explore)


def mutate(path,kind,change):
    files=recording_paths(path);p=files[kind]
    frame=pd.read_csv(p,float_precision='round_trip');frame=change(frame)
    with gzip.open(p,'wt',newline='') as stream:frame.to_csv(stream,index=False)
    side=Path(str(path)+'.metadata.json');meta=json.loads(side.read_text())
    meta['files'][kind].update(sha256=hash_file(p),bytes=p.stat().st_size,records=len(frame))
    side.write_text(json.dumps(meta))


class IdentificationTests(unittest.TestCase):
    def test_unique_exploitation_identifies_action(self):
        self.assertEqual(infer_maxima(probe(values(4))),(4,1<<4))

    def test_exploring_nonmaximum_does_not_identify_signal(self):
        self.assertEqual(infer_maxima(probe(values(4),9,True)),(None,None))

    def test_exploring_maximum_is_identifiable_from_actual_selected_q(self):
        self.assertEqual(infer_maxima(probe(values(4),4,True)),(4,1<<4))

    def test_unique_silence_is_identified_even_when_other_action_explored(self):
        self.assertEqual(infer_maxima(probe(values(0),9,True)),(0,1))

    def test_tied_signals_do_not_reveal_full_set(self):
        self.assertEqual(infer_maxima(probe(values(tied=(4,9)),4)),(None,None))

    def test_all_ties_and_exhaustive_memberships_have_exact_sets(self):
        self.assertEqual(infer_maxima(probe(values(),4)),(None,(1<<17)-1))
        self.assertEqual(infer_maxima(probe(values(tied=(0,4)),4)),(None,(1<<4)|1))
        # This later choice alone would not reveal the other tied signal.
        self.assertEqual(infer_maxima(probe(values(tied=(0,4)),0)),(None,None))

    def test_n_minus_one_ties_silence_outside_reveals_exact_complement(self):
        q=[0.]+[1.]*16
        self.assertEqual(infer_maxima(probe(q,4)),(None,((1<<17)-1)^1))

    def test_invalid_exploitation_and_ties_are_rejected(self):
        with self.assertRaises(ValueError):infer_maxima(probe(values(4),9,False))
        d=probe(values());d.max_ties=1
        with self.assertRaises(ValueError):infer_maxima(d)


class EpisodeTests(unittest.TestCase):
    def test_A_stable_unique_preference(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d);rows,_=process(d);s=rows['persistence_run_summary'][0]
            self.assertEqual((s['pairs'],s['id_pairs'],s['id_same']),(5,5,5))
            self.assertEqual(s['preferred_action_match_rate'],1)
            self.assertEqual(s['mean_exact_policy_tv'],0)
            self.assertEqual(s['mean_max_set_jaccard'],1)

    def test_B_one_preference_switch_with_unchanged_gap(self):
        q1=values(4);q1[9]=.9;q2=q1.copy();q2[4]=.8
        with tempfile.TemporaryDirectory() as d:
            write_run(d,encounters=[(1,0,0,q1,4,False),(2,0,0,q2,9,False)])
            rows,_=process(d);s=rows['persistence_run_summary'][0]
            self.assertEqual((s['id_pairs'],s['id_same']),(1,0))
            self.assertAlmostEqual(s['mean_gap_change'],0)
            self.assertEqual(s['mean_exact_policy_tv'],.8)
            self.assertEqual(s['mean_max_set_jaccard'],0)
            self.assertEqual((rows['preference_transitions'][0]['from_action'],rows['preference_transitions'][0]['to_action']),(4,9))

    def test_C_exploration_changes_behavior_not_a_fabricated_policy_change(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d,encounters=[(1,0,0,values(4),4,False),(2,0,0,values(4),9,True)])
            rows,_=process(d);s=rows['persistence_run_summary'][0]
            self.assertEqual((s['choice_changed'],s['changed_unknown'],s['changed_policy']),(1,1,0))
            self.assertIsNone(s['preferred_action_match_rate'])
            self.assertEqual(s['identity_coverage'],0)
            self.assertEqual(s['explore_pairs'],1)
            self.assertEqual(s['exploit_pairs'],0)

    def test_exploration_with_identifiable_stable_silence_policy(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d,encounters=[(1,0,0,values(0),0,False),(2,0,0,values(0),9,True)])
            rows,_=process(d);s=rows['persistence_run_summary'][0]
            self.assertEqual((s['choice_changed'],s['changed_stable']),(1,1))
            self.assertEqual(s['preferred_action_match_rate'],1)

    def test_D_tied_maxima_no_invented_identity_jaccard_or_tv(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d,encounters=[(1,0,0,values(tied=(4,9)),4,False),(2,0,0,values(tied=(4,9)),9,False)])
            rows,_=process(d);s=rows['persistence_run_summary'][0]
            self.assertEqual(s['choice_match_rate'],0)
            self.assertIsNone(s['preferred_action_match_rate'])
            self.assertIsNone(s['mean_max_set_jaccard']);self.assertIsNone(s['mean_exact_policy_tv'])
            self.assertEqual(s['unknown_policy'],1)

    def test_tied_sampling_can_change_without_any_policy_change(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d,encounters=[(1,0,0,values(),4,False),(2,0,0,values(),9,False)])
            rows,_=process(d);s=rows['persistence_run_summary'][0]
            self.assertEqual((s['changed_stable'],s['exploit_pairs'],s['exploit_same']),(1,1,0))
            self.assertEqual(s['mean_exact_policy_tv'],0)

    def test_cardinality_proves_policy_change_without_inventing_action_sets(self):
        first=values();second=values();second[4]=-.1
        with tempfile.TemporaryDirectory() as d:
            write_run(d,encounters=[(1,0,0,first,4,False),(2,0,0,second,9,False)])
            rows,_=process(d);s=rows['persistence_run_summary'][0]
            self.assertEqual(s['known_policy_changed'],1)
            self.assertEqual(s['mean_gap_change'],0)
            self.assertIsNone(s['mean_exact_policy_tv'])  # Conservative complete-set estimator only.
            self.assertEqual(s['identity_coverage'],0)

    def test_E_eviction_recreation_creates_separate_episodes(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d,capacity=1,encounters=[(1,0,0,values(4),4,False),(2,0,1,values(9),9,False),(3,0,0,values(),3,False)])
            rows,_=process(d);episodes=[e for e in rows['persistence_by_episode'] if e['state_id']==0]
            self.assertEqual([e['episode'] for e in episodes],[1,2])
            self.assertEqual([e['end_reason'] for e in episodes],['evict','run_end'])
            s=rows['persistence_run_summary'][0]
            self.assertEqual((s['recreated'],s['pairs'],s['actual_evictions']),(1,0,2))
            self.assertIsNone(s['preferred_action_match_rate'])

    def test_F_death_is_not_lru_eviction(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d,deaths={0:2},encounters=[(1,0,0,values(4),4,False),(2,0,0,values(4),4,False)])
            rows,_=process(d)
            self.assertEqual(rows['persistence_by_episode'][0]['end_reason'],'death')
            s=rows['persistence_run_summary'][0]
            self.assertEqual((s['deaths'],s['actual_evictions']),(1,0))
            self.assertEqual(rows['persistence_by_episode'][0]['row_updates'],2)

    def test_run_end_censors_retention_without_eviction(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d);rows,_=process(d)
            self.assertEqual(rows['persistence_by_episode'][0]['end_reason'],'run_end')
            self.assertEqual(rows['persistence_run_summary'][0]['actual_evictions'],0)

    def test_storage_duration_and_encounter_span_are_different(self):
        e=[(1,0,0,values(4),4,False)]+[(t,0,1,values(9),9,False) for t in range(2,6)]
        with tempfile.TemporaryDirectory() as d:
            write_run(d,encounters=e);rows,_=process(d)
            first=next(e for e in rows['persistence_by_episode'] if e['state_id']==0)
            self.assertEqual((first['storage_duration'],first['encounter_span'],first['encounters']),(4,0,1))
            self.assertIsNone(first['mean_interval'])
            self.assertTrue(all(r['actual_deaths']==0 for r in rows['agent_turnover']))

    def test_state_and_agent_updates_between_visits_are_counted_exactly(self):
        e=[(1,0,0,values(4),4,False),(2,0,1,values(4),4,False),(3,0,1,values(4),4,False),(4,0,0,values(4),4,False)]
        with tempfile.TemporaryDirectory() as d:
            write_run(d,encounters=e);rows,_=process(d)
            a=next(e for e in rows['persistence_by_episode'] if e['state_id']==0)
            self.assertEqual(a['mean_interval'],3)
            # The target state was updated once while the individual had three updates.
            last=next(e for e in rows['persistence_by_window'] if e['window_start']==2)
            self.assertEqual(last['mean_agent_updates_between'],2)
            self.assertEqual(last['mean_state_updates_between'],1)

    def test_SILENCE_category_transitions(self):
        qs=[values(0),values(tied=(0,4)),values(4)]
        with tempfile.TemporaryDirectory() as d:
            write_run(d,encounters=[(i+1,0,0,q,0 if i<2 else 4,False) for i,q in enumerate(qs)])
            rows,_=process(d);trans=rows['silence_preference_transitions']
            positive=[x for x in trans if x['count']>0]
            self.assertEqual({(x['from_category'],x['to_category']) for x in positive},{('unique_max','tied_max'),('tied_max','not_max')})
            self.assertTrue(all(x['probability']==1 for x in positive))

    def test_zero_category_counts_are_not_missing_seed_observations(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d);rows,_=process(d)
            transition=next(x for x in rows['silence_preference_transitions'] if x['window_start']==0 and
                            x['from_category']=='not_max' and x['to_category']=='unique_max')
            self.assertEqual(transition['origin_pairs'],1)
            self.assertEqual(transition['count'],0);self.assertEqual(transition['probability'],0)
            undefined=next(x for x in rows['silence_preference_transitions'] if x['from_category']=='unique_max')
            self.assertEqual(undefined['origin_pairs'],0);self.assertIsNone(undefined['probability'])
            absent=next(x for x in rows['retained_vs_recreated'] if x['window_start']==0 and x['encounter_class']=='recreated')
            self.assertEqual(absent['decisions'],0);self.assertIsNone(absent['mean_gap'])

    def test_visit_indices_and_age(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d);rows,_=process(d)
            self.assertEqual([r['visit_index'] for r in rows['persistence_by_visit_index'] if r['visit_kind']=='episode'],list(range(1,7)))
            self.assertEqual(rows['persistence_by_agent'][0]['first_age'],0)
            self.assertEqual(rows['persistence_by_agent'][0]['last_age'],5)
            ep=rows['persistence_by_episode'][0]
            self.assertEqual((ep['early_encounters'],ep['late_encounters']),(3,3))
            self.assertEqual(ep['first_observed_unique_visit'],1)

    def test_G_turnover_and_window_cohorts(self):
        e=[(1,0,0,values(4),4,False),(2,0,0,values(4),4,False),(3,1,0,values(4),4,False),(4,1,0,values(4),4,False)]
        with tempfile.TemporaryDirectory() as d:
            write_run(d,encounters=e,deaths={0:2});rows,_=process(d)
            early,late=rows['agent_turnover']
            self.assertEqual((early['active_agents'],early['actual_deaths']),(1,1))
            self.assertEqual((late['first_time_learners'],late['shared_with_previous'],late['newly_appearing']),(1,0,1))
            self.assertEqual((late['previous_agents_absent'],late['absent_dead_by_window_end'],late['absent_alive_at_window_end']),(1,1,0))
            shared=next(r for r in rows['persistence_by_cohort'] if r['window_start']==2 and r['cohort']=='shared_with_previous')
            self.assertEqual(shared['agents'],0);self.assertIsNone(shared['preferred_action_match_rate'])

    def test_histogram_quantiles_match_expanded_observations(self):
        from collections import Counter
        x=[0,0,1,3,3,4,8]
        for q in (0,.1,.5,.9,1):self.assertAlmostEqual(hist_quantile(Counter(x),q),np.quantile(x,q))
        self.assertIsNone(hist_quantile(Counter(),.5))


class IntegrityTests(unittest.TestCase):
    def test_metadata_schema_completion_hash_and_seed(self):
        for field in ('completed','schemas','config_id','seed','final_tick'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                p=write_run(d);side=Path(str(p)+'.metadata.json');m=json.loads(side.read_text())
                m[field]=False if field=='completed' else {} if field=='schemas' else -3 if field in ('seed','final_tick') else 'bad'
                side.write_text(json.dumps(m))
                with self.assertRaises((ValueError,KeyError)):discover(d)

    def test_all_four_stream_hashes_are_checked(self):
        for kind in ('decisions','updates','rows','states'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                p=write_run(d);file=recording_paths(p)[kind];file.write_bytes(file.read_bytes()+b'x')
                with self.assertRaises(ValueError):discover(d)

    def test_exact_state_dictionary_decoding_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p=write_run(d);r=discover(d)[0];key=catalog(r)[0]
            self.assertEqual(json.loads(key),[0,0,2,-1,0,0,0])
            for bad in ('null','[0]','[true,0,2,0,0,0,0]','[0,0,2,2,0,0,0]'):
                with self.assertRaises(ValueError):decode_state(bad,r['metadata']['config'])
            mutate(p,'states',lambda f:pd.concat([f,f],ignore_index=True))
            with self.assertRaises(ValueError):catalog(discover(d)[0])

    def test_auditory_catalog_structure_is_verified(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d,listening=True);r=discover(d)[0]
            self.assertEqual(len(json.loads(catalog(r)[0])),8)
            with self.assertRaises(ValueError):decode_state('[0,0,2,0,0,0,0,["IDENTIFIED",99,"E",0]]',r['metadata']['config'])

    def test_invalid_actual_epoch_and_missing_end_are_rejected(self):
        for mode in ('epoch','end'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                p=write_run(d)
                if mode=='epoch':mutate(p,'rows',lambda f:f.assign(epoch=f.epoch.where(f.event.ne('allocate'),7)))
                else:mutate(p,'rows',lambda f:f.loc[f.event.ne('run_end')])
                with self.assertRaises(ValueError):process(d)

    def test_missing_encounters_rejected_not_interpolated(self):
        with tempfile.TemporaryDirectory() as d:
            p=write_run(d)
            mutate(p,'decisions',lambda f:f.loc[f.tick.ne(3)])
            # Completion counters independently expose the missing decision.
            with self.assertRaises(ValueError):discover(d)

    def test_partial_windows_and_no_post_horizon_fill(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d,encounters=[(t,0,0,values(4),4,False) for t in range(1,4)])
            rows,_=process(d);w=rows['persistence_by_window']
            self.assertEqual([x['actual_window_end'] for x in w],[2,3])
            self.assertEqual([x['partial_window'] for x in w],[False,True])
            self.assertEqual([x['observed_ticks'] for x in w],[2,1])

    def test_empty_completed_zero_tick_run_has_undefined_rates(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d,encounters=[]);rows,_=process(d)
            self.assertEqual(rows['persistence_by_window'],[])
            s=rows['persistence_run_summary'][0]
            self.assertEqual(s['decisions'],0);self.assertIsNone(s['preferred_action_match_rate'])

    def test_no_cross_run_state_id_merging(self):
        with tempfile.TemporaryDirectory() as d:
            write_run(d,'z',seed=1);write_run(d,'a',seed=0)
            runs=discover(d);self.assertEqual([r['seed'] for r in runs],[0,1])
            for r in runs:
                catalog(r);s=Sink();RunAnalysis(r,Options(),s).process()
                self.assertEqual(s.rows['persistence_run_summary'][0]['first'],1)

    def test_completed_pilot_manifest_detects_missing_runs(self):
        with tempfile.TemporaryDirectory() as d:
            p=write_run(d)
            Path(d,'pilot_manifest.json').write_text(json.dumps({'completed':True,'runs':[{'condition':'D','seed':2,'metadata_sha256':'x'}]}))
            with self.assertRaisesRegex(ValueError,'Missing/extra'):discover(d)

    def test_I_equal_seed_weight_despite_unequal_counts(self):
        frame=pd.DataFrame({'run':['x','y'],'condition':['C','C'],'seed':[0,1],
                            'pairs':[1,1000],'id_pairs':[1,1000],'preferred_action_match_rate':[1.,0.]})
        from experiments.analyze_vocal_persistence import SUMMARY_KEYS
        tables={name:pd.DataFrame() for name in SUMMARY_KEYS};tables['persistence_run_summary']=frame
        row=equal_seed(tables).query("metric == 'preferred_action_match_rate'").iloc[0]
        self.assertEqual(row['mean'],.5);self.assertAlmostEqual(row['std'],2**-.5);self.assertEqual(row['n'],2)

    def test_options_validation_and_missing_files(self):
        for option in ({'window_size':0},{'age_bin':True},{'min_episode_encounters':5},{'low_updates':200}):
            with self.assertRaises(ValueError):Options(**option)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):discover(d)


class ArtifactTests(unittest.TestCase):
    def test_verified_historical_join_requires_all_recordings_and_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_run(root/'data');r=discover(root/'data')[0]
            cdir,vdir,cv,vv=historical_sources(root,r)
            runs=discover(root/'data')
            mapping,receipts=verified_communication(runs,cdir,vdir,cv,vv)
            self.assertEqual(mapping,{'oldC':'C_seed_0'})
            self.assertTrue(receipts)
            # Same seed, signed output bytes, but incompatible full Config.
            path=cdir/'analysis_metadata.json';m=json.loads(path.read_text())
            m['inputs'][0]['simulation_metadata']['config']['mutation_rate']=.25
            path.write_text(json.dumps(m));saved=json.loads(cv.read_text())
            saved['output_hashes'][path.name]=hash_file(path);cv.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError,'exact archived'):
                verified_communication(runs,cdir,vdir,cv,vv)

    def test_modified_historical_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_run(root/'data');r=discover(root/'data')[0]
            cdir,vdir,cv,vv=historical_sources(root,r)
            (cdir/'sender_information.csv').write_text('modified')
            with self.assertRaisesRegex(ValueError,'historical analysis hash'):
                verified_communication(discover(root/'data'),cdir,vdir,cv,vv)

    def test_J_reproducible_outputs_and_immutable_inputs_no_simulator_or_rng(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';write_run(data);write_run(data,'D_seed_1',seed=1,listening=True)
            before={p.name:hash_file(p) for p in data.iterdir()};rstate=random.getstate();nstate=np.random.get_state()
            with patch('simulation.simulation.Simulation.step',side_effect=AssertionError('New simulation forbidden')):
                m1=analyze(data,root/'a',options=Options(window_size=2,min_identity_pairs=1),no_mi=True)
                m2=analyze(data,root/'b',options=Options(window_size=2,min_identity_pairs=1),no_mi=True)
            self.assertEqual({p.name:hash_file(p) for p in (root/'a').iterdir()},{p.name:hash_file(p) for p in (root/'b').iterdir()})
            self.assertEqual(before,{p.name:hash_file(p) for p in data.iterdir()})
            self.assertEqual(rstate,random.getstate());after=np.random.get_state()
            self.assertEqual(nstate[0],after[0]);np.testing.assert_array_equal(nstate[1],after[1]);self.assertEqual(nstate[2:],after[2:])
            self.assertEqual(len(list((root/'a').glob('*.png'))),10)
            self.assertTrue((root/'a'/'persistence_by_episode.csv.gz').exists())
            self.assertEqual(m1,m2)

    def test_empty_analysis_generates_clear_figures_and_headers(self):
        with tempfile.TemporaryDirectory() as d:
            data=Path(d)/'data';write_run(data,encounters=[])
            m=analyze(data,Path(d)/'out',no_mi=True)
            self.assertEqual(m['streamed_records']['persistence_run_summary'],1)
            self.assertEqual(len(list((Path(d)/'out').glob('*.png'))),10)

    def test_output_guards_preserve_history(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';write_run(data)
            for dest in (data,data/'new',root):
                with self.assertRaises(ValueError):analyze(data,dest,no_mi=True)
            old=root/'old';old.mkdir();(old/'historical.csv').write_text('unchanged')
            with self.assertRaises(ValueError):analyze(data,old,no_mi=True)
            self.assertEqual((old/'historical.csv').read_text(),'unchanged')

    def test_historical_fixture_hashes_unchanged(self):
        expected={'v02_seed0_tick100.json':'5c3bf17c0ab4014d1ec98e823b239d860b74f92cc1394ab89c3bad73c5594e3b',
                  'v03_auditory_control.json':'ba85a1314e1c68a3588f4ba7b6d698eca16d22b706de307850ff625ebf17260b',
                  'v04_listening_control.json':'4b4b9e82362698b52a71e08e766cf16795575d4479cf023c8331eacfc624316a',
                  'v05_listening_seed7_tick80.json':'5538900edb7d38eb74ca06b28602f213e45b3efbdbf8528130eb804cd7459b44',
                  'v06_vocal_seed7_tick80.json':'0098529d6812003f97d718b3e1f4927b88fd371775047f82b640df96ec048dcd',
                  'v061_communication_toys.json':'d397488e3f476b26d3e7b20c8714fd16a1d452b9c27b57b68fe03aeb955130da'}
        for name,digest in expected.items():self.assertEqual(hash_file(Path('tests/fixtures')/name),digest)

if __name__=='__main__':unittest.main()
