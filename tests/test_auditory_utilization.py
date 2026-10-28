"""v0.6.4: exact probabilities, full-state isolation and sparse matched controls."""
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import asdict, replace, FrozenInstanceError
import ast
import io
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agents.auditory_memory import AuditoryEvent
from agents.brain import Action
from agents.learning import LearningBrain, Physiology
from agents.observation import Observation, TileObservation, SILENCE, MASKED, SourceDirection
from config import Config
from simulation.simulation import Simulation
from experiments.auditory_policy import (epsilon_policy, distances, evaluate, compare, identified,
    interventions, memory_before, simulation_fingerprint, frozen)
from experiments.auditory_controlled import (Design, CONDITIONS, CUES, config_for, schedules, train,
    evaluate_run, definition)
from experiments.auditory_retrospective import (Options, SCOPES, BASE, MatchTotals, auditory_features,
    match_scope, exposure, contrast, run_analysis)
from experiments.analyze_auditory_utilization import analyze, seed_summary, benchmark
from experiments.communication_data import file_hash
from tests.test_communication_analysis import fixture_file, record, MemoryOutput
from tests.test_vocalization import vocal_state

BODY = Physiology(0, 60, 100, True)
TILES = (TileObservation(0, 0, 0, 0),)


def receiver(listening=True):
    return LearningBrain(random.Random(8), Config(brain='learning', learning_uses_auditory=listening))


def obs(percept=SILENCE, tick=10):
    return Observation(tick, TILES, percept)


def store(brain, percept, values=None):
    state=evaluate(brain, BODY, obs(percept), ()).state
    brain._remember(state)[:] = [0.]*len(brain.actions) if values is None else values
    return state


class PolicyMathematicsTests(unittest.TestCase):
    def test_uniform_zero_and_negative_ties(self):
        for q in ([0]*7, [-9]*7):
            p,m=epsilon_policy(q,.2)
            self.assertEqual(m,tuple(range(7)))
            for x in p:self.assertAlmostEqual(x,1/7)

    def test_unique_maximum_formula(self):
        p,m=epsilon_policy([1,0,-1],.3)
        self.assertEqual(m,(0,));np.testing.assert_allclose(p,[.8,.1,.1])

    def test_exact_ties_not_approximately_equal(self):
        p,m=epsilon_policy([1.,1.+1e-14,1.],.2)
        self.assertEqual(m,(1,));self.assertGreater(p[1],p[0])
        p,m=epsilon_policy([-2,-1,-1],.3)
        np.testing.assert_allclose(p,[.1,.45,.45]);self.assertEqual(m,(1,2))

    def test_normalization_and_epsilon_boundaries(self):
        for epsilon in (0,.2,1):
            for q in ([0]*7,[-2,4,4,1],[-7,-8]):
                p,_=epsilon_policy(q,epsilon)
                self.assertAlmostEqual(sum(p),1);self.assertTrue(all(np.isfinite(p)));self.assertGreaterEqual(min(p),0)
        self.assertEqual(epsilon_policy([0,2],0)[0],(0.,1.))
        self.assertEqual(epsilon_policy([0,2],1)[0],(.5,.5))

    def test_invalid_values(self):
        for q,e in (([],.2),([np.nan],.2),([np.inf],.2),([0],-1),([0],1.1),([0],np.nan)):
            with self.assertRaises(ValueError):epsilon_policy(q,e)
        for p,q in (([0],[1]),([-1,2],[1,0]),([1],[.5,.5]),([np.nan],[1])):
            with self.assertRaises(ValueError):distances(p,q)

    def test_tv_jsd_known_and_symmetric(self):
        self.assertEqual(distances([1,0],[0,1]),(1.,1.))
        self.assertEqual(distances([.3,.7],[.3,.7]),(0.,0.))
        a=distances([.25,.75],[.75,.25]);self.assertEqual(a,distances([.75,.25],[.25,.75]))
        self.assertAlmostEqual(a[0],.5);self.assertAlmostEqual(a[1],.18872187554086717)

    def test_tied_representative_greedy_can_stay_same_while_policy_changes(self):
        # First argmax remains 0, but {0} and {0,1} induce different policies.
        a,_=epsilon_policy([2,0,0],.2);b,_=epsilon_policy([2,2,0],.2)
        self.assertGreater(distances(a,b)[0],0)
        # With fixed epsilon/action set, the SAME UNIQUE maximum implies identical
        # epsilon-greedy policy regardless of nonmaximal Q magnitudes.
        c,_=epsilon_policy([9,8,7],.2);self.assertEqual(a,c)


class ReadOnlyPolicyTests(unittest.TestCase):
    def test_absent_and_existing_zero_are_distinct(self):
        b=receiver();missing=evaluate(b,BODY,obs(identified(7)),())
        self.assertFalse(missing.present);self.assertEqual(b.memory_size,0)
        store(b,identified(7));present=evaluate(b,BODY,obs(identified(7)),())
        self.assertTrue(present.present);self.assertEqual(missing.probabilities,present.probabilities)
        self.assertEqual(missing.uniform_reason,'absent_zero_prior');self.assertEqual(present.uniform_reason,'stored_zero')

    def test_all_four_presence_patterns(self):
        for present_a,present_b,pattern in ((0,0,'neither_present'),(1,0,'original_only'),(0,1,'counterfactual_only'),(1,1,'both_present')):
            b=receiver()
            if present_a:store(b,identified(7))
            if present_b:store(b,identified(12))
            a=evaluate(b,BODY,obs(identified(7)),());c=evaluate(b,BODY,obs(identified(12)),())
            d=compare(a,c);self.assertEqual(d['row_pattern'],pattern)
            self.assertEqual(d['both_uniform'],True)

    def test_changed_unique_actions_and_per_action_deltas(self):
        b=receiver();store(b,identified(7),[2,0,0,0,0,0,0]);store(b,identified(12),[0,2,0,0,0,0,0])
        a=evaluate(b,BODY,obs(identified(7)),());c=evaluate(b,BODY,obs(identified(12)),())
        d=compare(a,c);self.assertTrue(d['unique_greedy_changed']);self.assertTrue(d['max_set_changed'])
        self.assertAlmostEqual(d['tv'],.8);self.assertAlmostEqual(sum(d['delta_probabilities']),0)

    def test_different_non_auditory_context_rejected(self):
        b=receiver();a=evaluate(b,BODY,obs(),())
        c=evaluate(b,replace(BODY,hunger=80),obs(),())
        with self.assertRaises(ValueError):compare(a,c)

    def test_no_allocator_no_decision_no_update_called(self):
        b=receiver();before=frozen(b)
        with patch.object(b,'_remember',side_effect=AssertionError),patch.object(b,'choose_action',side_effect=AssertionError):
            for percept in (SILENCE,MASKED,identified(7)):
                evaluate(b,BODY,obs(percept),())
        self.assertEqual(before,frozen(b))

    def test_live_encoder_reused_without_changes(self):
        b=receiver();b.auditory_memory.update(identified(12),9);before=frozen(b)
        a=evaluate(b,BODY,obs());self.assertEqual(a.state,b.state(BODY,obs()))
        self.assertEqual(a.state[-1],('IDENTIFIED',12,'E',1));self.assertEqual(before,frozen(b))

    def test_current_removal_preserves_past(self):
        b=receiver();past=(AuditoryEvent(9,identified(12)),)
        current=evaluate(b,BODY,obs(identified(7)),past)
        removed=evaluate(b,BODY,obs(),past)
        self.assertEqual(current.state[-1],('IDENTIFIED',7,'E',0))
        self.assertEqual(removed.state[-1],('IDENTIFIED',12,'E',1))

    def test_current_removal_empty_really_silent(self):
        self.assertEqual(evaluate(receiver(),BODY,obs(),()).state[-1],('SILENCE',))

    def test_precurrent_excludes_already_inserted_current_event(self):
        b=receiver();b.auditory_memory.update(identified(12),9);b.auditory_memory.update(identified(7),10)
        events=memory_before(b,10);self.assertEqual(events,(AuditoryEvent(9,identified(12)),))
        self.assertEqual(evaluate(b,BODY,obs(),events).state[-1],('IDENTIFIED',12,'E',1))

    def test_memory_expiration_boundary_and_substitution(self):
        b=receiver();before=frozen(b)
        for age in (1,3,4,5):
            state=evaluate(b,BODY,obs(),(AuditoryEvent(10-age,identified(7)),)).state[-1]
            self.assertEqual(state,('IDENTIFIED',7,'E',age) if age<4 else ('SILENCE',))
        a=evaluate(b,BODY,obs(),(AuditoryEvent(9,identified(7)),))
        c=evaluate(b,BODY,obs(),(AuditoryEvent(9,identified(12)),))
        self.assertNotEqual(a.state,c.state);self.assertEqual(before,frozen(b))

    def test_signal_direction_and_kind_interventions(self):
        b=receiver();features=[evaluate(b,BODY,obs(p),()).state[-1] for p in
            (SILENCE,MASKED,identified(7),identified(12),identified(7,SourceDirection.W))]
        self.assertEqual(len(set(features)),5)
        probes=interventions(10);self.assertEqual(len(probes),9)
        for _,kind,a,ma,c,mc in probes:
            if kind in ('current_removal','signal_identity','direction'):self.assertEqual(ma,mc)
            if kind=='signal_identity':self.assertEqual(a.signal.source_direction,c.signal.source_direction)
            if kind=='direction':self.assertEqual(a.signal.signal_id,c.signal.signal_id)
            if kind=='memory_content':self.assertEqual(a,c)

    def test_listening_off_is_seven_dimensional_not_empty_feature(self):
        b=receiver(False);states=[evaluate(b,BODY,obs(p),()).state for p in (SILENCE,MASKED,identified(7),identified(12))]
        self.assertEqual(len(set(states)),1);self.assertEqual(len(states[0]),7)
        self.assertEqual(len(evaluate(receiver(),BODY,obs(),()).state),8)

    def test_invalid_ids_and_directions(self):
        b=receiver()
        for signal in (-1,True):
            with self.assertRaises(ValueError):identified(signal)
        with self.assertRaises(ValueError):identified(7,'E')
        with self.assertRaises(ValueError):evaluate(b,BODY,obs(identified(b.vocab_size)),())
        with self.assertRaises(ValueError):evaluate(b,BODY,obs(),(AuditoryEvent(9,identified(b.vocab_size)),))

    def test_invalid_explicit_memories(self):
        b=receiver()
        for memory in ([],(AuditoryEvent(-1,identified(7)),),(AuditoryEvent(11,identified(7)),),
                       (AuditoryEvent(9,SILENCE),),(AuditoryEvent(9,identified(7)),AuditoryEvent(9,identified(12)))):
            with self.assertRaises(ValueError):evaluate(b,BODY,obs(),memory)

    def test_no_identity_or_exact_location(self):
        p=identified(7);self.assertEqual(set(asdict(p.signal)),{'signal_id','source_direction'})
        with self.assertRaises(FrozenInstanceError):p.signal.signal_id=12

    def test_complete_state_all_probes_and_order_replay(self):
        cfg=Config(brain='learning',learning_uses_auditory=True,initial_population=3,initial_food=30,initial_water=30)
        control=Simulation(cfg);probed=Simulation(cfg)
        for _ in range(8):control.step();probed.step()
        before=vocal_state(probed);full=simulation_fingerprint(probed);global_before=random.getstate()
        h=probed.agents[0];body=Physiology.capture(h);tick=probed.tick+1
        queries=interventions(tick);answers=[]
        def checked_evaluate(percept,memory):
            observation=Observation(tick,TILES,percept)
            inputs=(asdict(body),asdict(observation))
            result=evaluate(h.brain,body,observation,memory)
            # Every individual query must preserve complete state, not merely
            # a batch that might accidentally restore a previous mutation.
            self.assertEqual(vocal_state(probed),before)
            self.assertEqual(simulation_fingerprint(probed),full)
            self.assertEqual(global_before,random.getstate())
            self.assertEqual(inputs,(asdict(body),asdict(observation)))
            return result
        for sequence in (queries,queries[::-1]):
            answers.append({name: (checked_evaluate(a,ma),checked_evaluate(b,mb))
                            for name,_,a,ma,b,mb in sequence})
        self.assertEqual(answers[0],answers[1]);self.assertEqual(global_before,random.getstate())
        for _ in range(15):
            control.step();probed.step();self.assertEqual(vocal_state(control),vocal_state(probed))

    def test_diagnostics_not_in_normal_call_graph(self):
        sim=Simulation(Config(brain='learning',initial_population=2))
        with patch.object(LearningBrain,'inspect_physical_q',side_effect=AssertionError):
            for _ in range(5):sim.step()


class MatchedRetrospectiveTests(unittest.TestCase):
    def frame(self, cells=(3,3), agents=None):
        rows=[]
        for cue,n in zip((7,12,3),cells):
            for _ in range(n):
                t=len(rows)+1
                rows.append(asdict(record(t,agent=0,listening=True,auditory_kind='IDENTIFIED',heard_signal_id=cue,
                    heard_direction='E',encoded_auditory_kind='IDENTIFIED',encoded_signal_id=cue,encoded_direction='E',encoded_age=0,
                    physical_action='eat' if cue==7 else 'drink')))
        f=pd.DataFrame(rows)
        for column in BASE[1:]:f[column]=2
        if agents is not None:f['agent_id']=agents
        return f

    def match(self,frame,thresholds=(2,),scope='encoded',listening=True,totals=None):
        current,encoded=auditory_features(frame,listening);sink=MemoryOutput()
        context=dict(evidence='observational',run='toy',condition='B',seed=0,window_start=0,window_end=100,actual_window_end=100)
        actions=tuple(a.value for a in Action if a is not Action.REPRODUCE)
        match_scope(frame,current,encoded,context,scope,thresholds,actions,sink,listening,run_totals=totals)
        return sink

    def summary(self,sink):return next(r for r in sink.rows['retrospective_matched_summary'] if r['contrast']=='all')

    def test_exact_known_distance_and_sample_distribution(self):
        s=self.match(self.frame((3,5)));r=self.summary(s)
        self.assertEqual(r['matched_contexts'],1);self.assertEqual(r['individuals'],1)
        self.assertEqual(r['decisions_used'],8);self.assertEqual(r['coverage'],1)
        self.assertEqual((r['cell_n_min'],r['cell_n_median'],r['cell_n_max']),(3,4,5))
        self.assertEqual((r['mean_context_tv'],r['mean_context_jsd_bits']),(1.,1.))

    def test_threshold_sensitivity_sparse_means_unavailable(self):
        s=self.match(self.frame((3,3)),thresholds=(2,5,10))
        rows=[r for r in s.rows['retrospective_matched_summary'] if r['contrast']=='all']
        self.assertTrue(rows[0]['available'])
        for r in rows[1:]:
            self.assertFalse(r['available']);self.assertIsNone(r['mean_context_tv']);self.assertEqual(r['coverage'],0)

    def test_never_pool_individuals_or_body_contexts(self):
        for column,values in (('agent_id',[0]*3+[1]*3),('hunger_bin',[0]*3+[1]*3)):
            f=self.frame();f[column]=values;r=self.summary(self.match(f))
            self.assertEqual(r['matched_contexts'],0);self.assertIsNone(r['mean_context_tv'])

    def test_conditioning_on_exact_memory_or_current_removes_unmatched_cells(self):
        for scope in ('current_given_encoded','encoded_given_current'):
            r=self.summary(self.match(self.frame(),scope=scope));self.assertFalse(r['available'])

    def test_decision_coverage_uses_union_not_pair_multiplicity(self):
        s=self.match(self.frame((3,3,3)));r=self.summary(s)
        self.assertEqual(r['pairs'],3);self.assertEqual(r['decisions_used'],9);self.assertEqual(r['coverage'],1)
        self.assertEqual(r['feature_cells'],3)

    def test_time_confounding_is_visible(self):
        row=self.match(self.frame()).rows['retrospective_action_distances'][0]
        self.assertFalse(row['time_ranges_overlap']);self.assertEqual(row['mean_time_separation'],3)

    def test_contrast_categories_hold_other_dimensions_fixed(self):
        self.assertEqual(contrast(('IDENTIFIED',7,'E',1),('IDENTIFIED',12,'E',1)),'different_id_same_direction_age')
        self.assertEqual(contrast(('IDENTIFIED',7,'E',1),('IDENTIFIED',7,'W',1)),'different_direction_same_id_age')
        self.assertEqual(contrast(('IDENTIFIED',7,'E',1),('IDENTIFIED',7,'E',2)),'different_retained_age')
        self.assertEqual(contrast(('IDENTIFIED',7,'E',1),('IDENTIFIED',12,'E',2)),'other_exact_feature_difference')

    def test_exposure_retained_memory_not_current_sound(self):
        f=self.frame((2,0));f.loc[1,['auditory_kind','heard_signal_id','heard_direction']]=['SILENCE',None,None]
        f.loc[1,'encoded_age']=1
        c,e=auditory_features(f,True);sink=MemoryOutput();from collections import Counter
        exposure({},c,e,True,sink,Counter())
        row=sink.rows['retrospective_exposure_by_window'][0]
        self.assertEqual(row['current_encoded_difference_rate'],.5);self.assertEqual(row['encoded_nonempty_rate'],1)
        self.assertEqual(row['silent_with_retained_event_rate'],.5)

    def test_off_not_encoded_is_not_silence_or_zero_effect(self):
        s=self.match(self.frame(),listening=False);r=self.summary(s)
        self.assertEqual(r['unavailable_reason'],'auditory_not_encoded');self.assertFalse(r['available'])
        self.assertNotIn('mean_context_tv',r)

    def test_run_aggregation_unique_individual_and_median(self):
        totals=MatchTotals()
        self.match(self.frame((2,4)),totals=totals);self.match(self.frame((3,5)),totals=totals)
        sink=MemoryOutput();totals.write({},sink)
        r=next(r for r in sink.rows['retrospective_matched_by_run'] if r['contrast']=='all')
        self.assertEqual(r['individuals'],1);self.assertEqual(r['matched_contexts'],2)
        self.assertEqual(r['cell_n_median'],3.5);self.assertEqual(r['coverage'],1)

    def test_equal_seed_summary_not_exposure_weighted(self):
        frame=pd.DataFrame([dict(condition='B',seed=0,tv=0,decisions=10000),dict(condition='B',seed=1,tv=1,decisions=1)])
        out=seed_summary(frame,['condition']);r=out[out.metric.eq('tv')].iloc[0]
        self.assertEqual(r['mean'],.5);self.assertAlmostEqual(r['std'],2**-.5);self.assertEqual(r['n'],2)
        with self.assertRaises(ValueError):seed_summary(pd.concat([frame,frame]),['condition'])

    def test_options_validation(self):
        for kwargs in (dict(window_size=0),dict(thresholds=(2,2)),dict(thresholds=()),dict(thresholds=(True,))):
            with self.assertRaises(ValueError):Options(**kwargs)

    def test_archive_hash_mismatch_fails_before_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);fixture_file(path)
            archive=path/'registry.json';archive.write_text('{}')
            proof=path/'proof.json';proof.write_text(json.dumps({'output_hashes':{'analysis_metadata.json':'invalid'}}))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):run_analysis(path,MemoryOutput(),Options(),archive,proof)


class ControlledExperimentTests(unittest.TestCase):
    def test_control_definitions_and_default_protocol_saved_before_training(self):
        d=Design()
        for c in CONDITIONS:
            cfg=config_for(d,0,c)
            self.assertEqual(cfg.learning_uses_auditory,c!='predictive_off')
            self.assertEqual(cfg.learning_rate,Config().learning_rate)
            self.assertEqual(cfg.learning_discount,Config().learning_discount)
            self.assertEqual(cfg.learning_epsilon,Config().learning_epsilon)
            self.assertEqual(cfg.perception_radius,0);self.assertFalse(cfg.reproduction_enabled)
        saved=Path('data/v064_design.json')
        if saved.exists():self.assertEqual(json.loads(saved.read_text()),json.loads(json.dumps(definition(d))))

    def test_schedules_balanced_independent_and_replay(self):
        d=Design(trials=16)
        for seed in d.seeds:
            contexts,predictive,independent=schedules(d,seed)
            self.assertEqual((contexts,predictive,independent),schedules(d,seed))
            self.assertEqual(predictive,tuple(CUES[c] for c in contexts))
            for i in range(0,d.trials,4):
                self.assertEqual(set(zip(contexts[i:i+4],independent[i:i+4])),{(0,7),(0,12),(1,7),(1,12)})
        self.assertNotEqual(schedules(d,0),schedules(d,1))

    def test_no_cue_in_disabled_state_even_with_same_external_percepts(self):
        b=receiver(False)
        self.assertEqual(evaluate(b,BODY,obs(identified(7)),()).state,evaluate(b,BODY,obs(identified(12)),()).state)
        self.assertEqual(config_for(Design(),0,'predictive_on').signal_vocab_size,Config().signal_vocab_size)

    def test_small_training_replay_separation_all_conditions(self):
        d=Design(trials=4,seeds=(0,),checkpoint_trials=2)
        for c in CONDITIONS:
            a,visits,updates,curves,checks=train(d,0,c);b,*_=train(d,0,c)
            self.assertEqual(simulation_fingerprint(a),simulation_fingerprint(b))
            before=simulation_fingerprint(a);global_before=random.getstate()
            with patch.object(a,'step',side_effect=AssertionError),patch.object(LearningBrain,'choose_action',side_effect=AssertionError):
                probes,result,proof=evaluate_run(a,visits,updates,d,0,c)
            self.assertEqual(before,simulation_fingerprint(a));self.assertEqual(global_before,random.getstate())
            self.assertEqual(proof['before'],proof['after']);self.assertEqual(len(probes),18)
            self.assertTrue(all(h['before']==h['after'] for h in checks))
            self.assertEqual(curves[0]['trials'],0)
            if c=='untrained_on':
                self.assertEqual(result['physical_updates'],0);self.assertEqual(result['learning_result'],'failed_or_untrained')
                self.assertTrue(all(p['coverage']=='neither_present' for p in probes))
            if c=='predictive_off':self.assertEqual(result['root_identity_tv'],0);self.assertFalse(result['distinct_root_states'])

    def test_training_does_not_edit_q_or_inject_updates(self):
        tree=ast.parse(Path('experiments/auditory_controlled.py').read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('_remember',calls);self.assertNotIn('_update',calls)
        self.assertNotIn('choose_action',calls);self.assertNotIn('observe_outcome',calls)
        self.assertNotIn('_values',Path('experiments/auditory_controlled.py').read_text())

    def test_failed_learning_explicit_not_hidden(self):
        d=Design(trials=4,seeds=(0,));a,v,u,*_=train(d,0,'untrained_on')
        _,result,_=evaluate_run(a,v,u,d,0,'untrained_on')
        self.assertEqual(result['correct_cues'],0);self.assertEqual(result['learning_result'],'failed_or_untrained')

    def test_design_validation(self):
        for kwargs in (dict(trials=3),dict(trials=516),dict(steps_per_trial=4),dict(seeds=(0,0)),dict(correct_probability=0)):
            with self.assertRaises(ValueError):Design(**kwargs)

    def test_benchmark_is_separate_and_state_preserving(self):
        d=Design(trials=4,seeds=(0,));a,*_=train(d,0,'predictive_on');before=simulation_fingerprint(a)
        report=benchmark(a,repetitions=2,calls=5)
        self.assertTrue(report['state_unchanged']);self.assertEqual(before,simulation_fingerprint(a))

    def test_complete_small_output_replay_csv_json_png(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            root=Path(folder);inputs=root/'inputs';inputs.mkdir()
            fixture_file(inputs,listening=True,horizon=4)
            before=file_hash(inputs/'toy_communication.csv');rng=random.getstate()
            kwargs=dict(data_dir=inputs,part='all',options=Options(2,(1,2)),design=Design(trials=4,seeds=(0,),checkpoint_trials=2),
                        archive_metadata=None,verification=None)
            analyze(output_dir=root/'a',**kwargs);analyze(output_dir=root/'b',**kwargs)
            files=sorted(p.name for p in (root/'a').iterdir())
            self.assertGreaterEqual(len([f for f in files if f.endswith('.png')]),8)
            for name in files:self.assertEqual(file_hash(root/'a'/name),file_hash(root/'b'/name),name)
            metadata=json.loads((root/'a'/'analysis_metadata.json').read_text())
            self.assertIn('agents/human.py',metadata['source_sha256'])
            self.assertIn('systems/communication.py',metadata['source_sha256'])
            self.assertEqual(before,file_hash(inputs/'toy_communication.csv'));self.assertEqual(rng,random.getstate())

    def test_retrospective_mode_never_runs_simulation(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            root=Path(folder);inputs=root/'inputs';inputs.mkdir();fixture_file(inputs,horizon=2)
            with patch.object(Simulation,'step',side_effect=AssertionError):
                result=analyze(data_dir=inputs,output_dir=root/'out',part='retrospective',archive_metadata=None,verification=None)
            self.assertTrue(result['completed']);self.assertEqual(result['controlled_configs'],[])

    def test_plot_legend_includes_group_missing_first_category(self):
        from experiments.auditory_utilization_plots import points
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        try:
            frame=pd.DataFrame([dict(condition='A',seed=0,category=0,value=.1),
                                dict(condition='B',seed=0,category=1,value=.2)])
            points(ax,frame,'category','value','condition')
            self.assertEqual(ax.get_legend_handles_labels()[1],['A','B'])
        finally:plt.close(fig)

    def test_historical_directory_is_protected(self):
        with self.assertRaises(ValueError):analyze(output_dir=Path('analysis/communication'),part='controlled')


if __name__ == '__main__':unittest.main()
