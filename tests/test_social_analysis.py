"""Streaming social recording/analysis validation and deterministic artifacts."""
from collections import Counter
from contextlib import ExitStack,redirect_stdout
from dataclasses import asdict,replace
import csv
import gzip
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import Config,SocialConfig
from simulation.simulation import Simulation
from simulation.social_observer import CSVSocialWriter,write_metadata,COLUMNS,CONTACT_KINDS
from simulation.vocal_learning_observer import hash_file
from experiments.social_data import discover,records,Validator
from experiments.analyze_social_memory import analyze,seed_summary,histogram_stats
from tests.test_social_memory import quiet,simulation


def recording(root,name='seed0.csv',cfg=None,settings=None,ticks=3,before=None):
    path=Path(root)/name
    with ExitStack() as stack:
        writer=CSVSocialWriter.open(stack,path)
        sim=simulation(cfg or quiet(world_width=1,initial_population=3),settings,sink=writer)
        for t in range(1,ticks+1):
            if not sim.population:break
            if before:before(sim,t)
            sim.step()
        sim.social_observer.finish(sim.statistics)
    meta=write_metadata(path,sim,writer)
    return path,meta


def validate(path):
    run=discover(path.parent,path.name)[0];validator=Validator(run)
    for row in records(run):validator.accept(row)
    validator.finish()
    return run


def rewrite(path,mutate):
    # Rebind the recording hash to test semantic validation independently from
    # the cheaper integrity check. This helper never edits historical artifacts.
    with path.open(newline='') as f:rows=list(csv.DictReader(f))
    mutate(rows)
    for i,row in enumerate(rows):row['sequence']=str(i)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS);writer.writeheader();writer.writerows(rows)
    side=Path(str(path)+'.metadata.json');m=json.loads(side.read_text())
    m['file']={'sha256':hash_file(path),'bytes':path.stat().st_size};m['records']=len(rows)
    m['event_counts']=dict(Counter(r['event_kind'] for r in rows));side.write_text(json.dumps(m))


class SocialRecordingTests(unittest.TestCase):
    def test_plain_csv_schema_immutable_and_complete_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path,meta=recording(folder)
            run=validate(path)
            self.assertEqual(tuple(meta['columns']),COLUMNS);self.assertTrue(meta['completed'])
            self.assertEqual(meta['config']['social'],asdict(SocialConfig(True)))
            self.assertEqual(meta['final_tick'],3);self.assertEqual(meta['statistics']['population'],3)
            self.assertEqual(meta['records'],sum(meta['event_counts'].values()))
            self.assertEqual(meta['file']['sha256'],hash_file(path));self.assertIn('agents/social_memory.py',meta['source_sha256'])

    def test_gzip_and_sidecar_replay_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            a,_=recording(root/'a','seed.csv.gz');b,_=recording(root/'b','seed.csv.gz')
            self.assertEqual(a.read_bytes(),b.read_bytes())
            self.assertEqual(Path(str(a)+'.metadata.json').read_bytes(),Path(str(b)+'.metadata.json').read_bytes())
            validate(a)

    def test_exclusive_outputs_and_sidecars(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_=recording(folder);original=path.read_bytes()
            with ExitStack() as stack,self.assertRaises(ValueError):CSVSocialWriter.open(stack,path)
            self.assertEqual(path.read_bytes(),original)
            path.unlink()
            with ExitStack() as stack,self.assertRaises(ValueError):CSVSocialWriter.open(stack,path)

    def test_missing_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_=recording(folder);Path(str(path)+'.metadata.json').unlink()
            with self.assertRaisesRegex(ValueError,'metadata'):discover(folder)

    def test_corrupt_hash_and_config_metadata_rejected(self):
        for field in ('hash','config','completed','records','schema'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as folder:
                path,m=recording(folder);side=Path(str(path)+'.metadata.json')
                if field=='hash':m['file']['sha256']='0'*64
                elif field=='config':m['config']['social']['social_memory_capacity']=7
                elif field=='completed':m['completed']=False
                elif field=='records':m['records']+=1
                else:m['columns']=[]
                side.write_text(json.dumps(m))
                with self.assertRaises(ValueError):validate(path)

    def test_duplicate_contacts_rejected_even_with_rebound_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_=recording(folder)
            def mutation(rows):
                index=next(i for i,r in enumerate(rows) if r['event_kind']=='first_encounter')
                rows.insert(index+1,rows[index].copy())
            rewrite(path,mutation)
            with self.assertRaises(ValueError):validate(path)

    def test_invalid_negative_self_and_dead_identity_rejected(self):
        for value in ('-1','SELF','9999'):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as folder:
                path,_=recording(folder)
                def mutation(rows):
                    row=next(r for r in rows if r['event_kind']=='first_encounter')
                    row['observed_id']=row['observer_id'] if value=='SELF' else value
                rewrite(path,mutation)
                with self.assertRaises(ValueError):validate(path)

    def test_unsorted_ticks_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_=recording(folder)
            def mutation(rows):
                row=next(r for r in rows if r['event_kind']=='observation' and r['tick']=='2')
                row['tick']='0'
            rewrite(path,mutation)
            with self.assertRaises(ValueError):validate(path)

    def test_invalid_extra_death_information_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_=recording(folder)
            def mutation(rows):next(r for r in rows if r['event_kind']=='first_encounter')['actual_death_tick']='1'
            rewrite(path,mutation)
            with self.assertRaisesRegex(ValueError,'fields outside'):validate(path)

    def test_absence_cannot_be_substituted_for_privileged_death(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_=recording(folder)
            def mutation(rows):
                row=next(r for r in rows if r['event_kind']=='continuous_contact')
                row.update(event_kind='death',observed_id='',dx='',dy='',elapsed_since_last_seen='',first_seen_tick='',last_confirmed_alive_tick='',encounter_count='',actual_death_tick=row['tick'],death_cause='missing')
            rewrite(path,mutation)
            with self.assertRaises(ValueError):validate(path)

    def test_wrong_reunion_and_memory_occupancy_rejected(self):
        for key,value in (('event_kind','reunion'),('memory_occupancy','32'),('encounter_count','17')):
            with tempfile.TemporaryDirectory() as folder:
                path,_=recording(folder)
                def mutation(rows):next(r for r in rows if r['event_kind']=='continuous_contact')[key]=value
                rewrite(path,mutation)
                with self.assertRaises(ValueError):validate(path)

    def test_false_privileged_reappearance_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_=recording(folder)
            def mutation(rows):next(r for r in rows if r['event_kind']=='first_encounter')['event_kind']='memory_miss_after_eviction'
            rewrite(path,mutation)
            with self.assertRaises(ValueError):validate(path)

    def test_newborn_recording_and_zero_opportunity_individual(self):
        from agents.brain import Action
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def before(sim,t):
                sim.agents[0].reproductive_type,sim.agents[1].reproductive_type=0,1
                for h in sim.agents:h.brain.action=Action.REPRODUCE
            path,m=recording(root/'raw',cfg=quiet(world_width=1,reproduction_enabled=True,min_reproductive_age=0),ticks=1,before=before)
            validate(path)
            with redirect_stdout(io.StringIO()):analyze(root/'raw',root/'out')
            self.assertEqual(m['statistics']['births'],1)
            people=pd.read_csv(root/'out/social_by_individual.csv');child=people[people.individual_id.eq(2)].iloc[0]
            self.assertEqual(child.observations,0);self.assertEqual(child.birth_tick,1)
            self.assertTrue(pd.isna(child.mean_memory_occupancy));self.assertTrue(pd.isna(child.actual_death_tick))

    def test_duplicate_seed_configuration_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            recording(folder,'a.csv');recording(folder,'b.csv')
            with self.assertRaisesRegex(ValueError,'Duplicate'):discover(folder)

    def test_incomplete_recording_not_certified_by_truncated_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_=recording(folder)
            rewrite(path,lambda rows:rows.pop())
            with self.assertRaisesRegex(ValueError,'incomplete'):validate(path)

    def test_lifecycle_end_extinction_is_not_filled(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_=recording(folder,cfg=quiet(world_width=1,max_age=1),ticks=5)
            run=validate(path)
            self.assertEqual(run['metadata']['final_tick'],1)
            self.assertEqual(run['metadata']['event_counts']['death'],2)
            self.assertEqual(run['metadata']['event_counts']['observation'],2)

    def test_cli_flags_and_legacy_aggregate_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=[sys.executable,'main.py','--mode','evolution','--brain','learning','--seed','0','--population','3','--ticks','3']
            for name,flags in [('off',[]),('on',['--social-memory']),('record',['--social-memory','--social-csv',str(root/'social.csv.gz')])]:
                subprocess.run([*base,'--csv',str(root/(name+'.csv')),*flags],capture_output=True,check=True)
            self.assertEqual((root/'off.csv').read_bytes(),(root/'on.csv').read_bytes())
            self.assertEqual((root/'off.csv').read_bytes(),(root/'record.csv').read_bytes())
            validate(root/'social.csv.gz')

    def test_combined_old_communication_and_vocal_streams_are_byte_identical(self):
        from simulation.vocal_learning_observer import recording_paths
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=[sys.executable,'main.py','--mode','evolution','--brain','learning','--population','3','--seed','7','--ticks','5']
            for mode,flags in [('off',[]),('on',['--social-memory','--social-csv',str(root/'social.csv.gz')])]:
                subprocess.run([*base,'--communication-csv',str(root/(mode+'_comm.csv')),
                    '--vocal-learning-csv',str(root/(mode+'_vocal.csv.gz')),*flags],capture_output=True,check=True)
            self.assertEqual((root/'off_comm.csv').read_bytes(),(root/'on_comm.csv').read_bytes())
            off=recording_paths(root/'off_vocal.csv.gz');on=recording_paths(root/'on_vocal.csv.gz')
            for key in off:self.assertEqual(off[key].read_bytes(),on[key].read_bytes(),key)
            validate(root/'social.csv.gz')

    def test_cli_output_collisions_and_disabled_recording_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=[sys.executable,'main.py','--ticks','0','--social-memory']
            for flags in (['--social-csv',str(root/'same.csv'),'--events',str(root/'same.csv')],
                          ['--social-csv',str(root/'new.csv'),'--no-social-memory']):
                r=subprocess.run([*base,*flags],capture_output=True);self.assertNotEqual(r.returncode,0)
            target=root/'existing.csv';target.write_text('preserve')
            r=subprocess.run([*base,'--social-csv',str(root/'new.csv'),'--csv',str(target)],capture_output=True)
            self.assertNotEqual(r.returncode,0);self.assertEqual(target.read_text(),'preserve');self.assertFalse((root/'new.csv').exists())


class SocialAnalysisTests(unittest.TestCase):
    def run_analysis(self,raw,output,window=100):
        with redirect_stdout(io.StringIO()):return analyze(raw,output,window_size=window)

    def test_continuous_contact_only_no_reunions_known_exact_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);recording(root/'raw')
            self.run_analysis(root/'raw',root/'out',2)
            r=pd.read_csv(root/'out/social_by_run.csv').iloc[0]
            self.assertEqual(r.observations,9);self.assertEqual(r.contacts,18);self.assertEqual(r.first_encounters,6)
            self.assertEqual(r.continuous_contacts,12);self.assertEqual(r.reunions,0)
            self.assertEqual(r.mean_memory_occupancy,2);self.assertEqual(r.mean_visible_count,2)
            self.assertAlmostEqual(r.memory_hit_rate,2/3)
            self.assertEqual(r.lifetime_first_pairs,6);self.assertEqual(r.lifetime_recontacts,12)

    def test_exactly_one_reunion_asymmetric_visual_radius(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def before(sim,t):
                sim.agents[0].x=0;sim.agents[1].x=1 if t in (1,4) else 4
                sim.agents[1].genome=replace(sim.agents[1].genome,perception_radius=0)
            recording(root/'raw',cfg=quiet(perception_radius=1),settings=SocialConfig(True,32,3),ticks=4,before=before)
            self.run_analysis(root/'raw',root/'out')
            row=pd.read_csv(root/'out/social_by_run.csv').iloc[0]
            self.assertEqual(row.reunions,1);self.assertEqual(row.mean_reunion_interval,3)
            intervals=pd.read_csv(root/'out/social_interval_distribution.csv')
            b=intervals[intervals.scope.eq('run')&intervals.interval_kind.eq('reunion')]
            self.assertEqual(b['count'].sum(),1);self.assertEqual(b.elapsed_ticks.iloc[0],3)

    def test_evicted_reappearance_reconstructed_beyond_bounded_recorder_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            def before(sim,t):
                for h in sim.agents:h.x=5
                sim.agents[0].x=0;sim.agents[(1,2,3,1)[t-1]].x=0
            path,m=recording(root/'raw',cfg=quiet(initial_population=4,perception_radius=0),settings=SocialConfig(True,1,2),ticks=4,before=before)
            self.run_analysis(root/'raw',root/'out')
            f=pd.read_csv(root/'out/social_by_individual.csv');r=f[f.individual_id.eq(0)].iloc[0]
            self.assertEqual(r.memory_reappearances,1);self.assertEqual(r.raw_identified_reappearances,0)
            self.assertEqual(r.first_encounters,4);self.assertEqual(r.lifetime_first_pairs,3);self.assertEqual(r.evictions,3)

    def test_partial_windows_and_no_post_extinction_filling(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);recording(root/'raw',cfg=quiet(world_width=1,max_age=3),ticks=9)
            self.run_analysis(root/'raw',root/'out',2)
            f=pd.read_csv(root/'out/social_by_window.csv')
            self.assertEqual(f.actual_window_end.tolist(),[2,3]);self.assertEqual(f.observed_ticks.tolist(),[2,1])
            self.assertEqual(f.partial_window.tolist(),[False,True]);self.assertEqual(f.population_end.tolist(),[2,0])
            self.assertEqual(f.deaths.sum(),2)

    def test_zero_tick_empty_population_recording_safe_missing_rates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);recording(root/'raw',cfg=quiet(initial_population=0,max_ticks=0),ticks=0)
            self.run_analysis(root/'raw',root/'out')
            r=pd.read_csv(root/'out/social_by_run.csv').iloc[0]
            self.assertEqual(r.population_end,0);self.assertEqual(r.observations,0);self.assertTrue(pd.isna(r.memory_hit_rate))
            self.assertTrue(pd.read_csv(root/'out/social_by_window.csv').empty)
            self.assertTrue(pd.read_csv(root/'out/social_by_individual.csv').empty)

    def test_never_observed_survivor_is_not_called_dead(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);recording(root/'raw',cfg=quiet(initial_population=1),ticks=2)
            self.run_analysis(root/'raw',root/'out')
            r=pd.read_csv(root/'out/social_by_run.csv').iloc[0]
            self.assertEqual(r.contacts,0);self.assertEqual(r.active_observing_agents,1);self.assertEqual(r.observed_individuals,0)
            self.assertEqual(r.deaths,0);self.assertEqual(r.population_end,1);self.assertTrue(pd.isna(r.known_visible_fraction))
            person=pd.read_csv(root/'out/social_by_individual.csv').iloc[0]
            self.assertEqual(person.end_reason,'run_end');self.assertTrue(pd.isna(person.actual_death_tick))

    def test_single_seed_sd_missing_and_multiple_seed_summary_equal_weight(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for seed in (0,1):recording(root/'raw',f'seed{seed}.csv',cfg=quiet(world_width=1,random_seed=seed))
            self.run_analysis(root/'raw',root/'out')
            summary=pd.read_csv(root/'out/social_seed_summary.csv');r=summary[summary.metric.eq('contacts')].iloc[0]
            self.assertEqual(r['n'],2);self.assertEqual(r['std'],0)
            runs=pd.read_csv(root/'out/social_by_run.csv');runs.loc[0,'contacts']=0;runs.loc[1,'contacts']=100
            r=seed_summary(runs);r=r[r.metric.eq('contacts')].iloc[0]
            self.assertEqual(r['mean'],50);self.assertAlmostEqual(r['std'],50*2**.5)
            r=seed_summary(runs.iloc[:1]);self.assertTrue(r[r.metric.eq('contacts')]['std'].isna().all())

    def test_analysis_replay_rng_hashes_figures_and_no_simulation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for seed in (1,0):recording(root/'raw',f'seed{seed}.csv.gz',cfg=quiet(world_width=1,random_seed=seed))
            originals={p.name:hash_file(p) for p in (root/'raw').iterdir()}
            rng=random.getstate();nrng=np.random.get_state()
            with patch.object(Simulation,'step',side_effect=AssertionError('Offline analysis started a simulation')):
                self.run_analysis(root/'raw',root/'a',2);self.run_analysis(root/'raw',root/'b',2)
            files=sorted(p.name for p in (root/'a').iterdir());self.assertEqual(len([p for p in files if p.endswith('.png')]),8)
            for name in files:self.assertEqual(hash_file(root/'a'/name),hash_file(root/'b'/name),name)
            self.assertEqual(originals,{p.name:hash_file(p) for p in (root/'raw').iterdir()})
            self.assertEqual(rng,random.getstate());np.testing.assert_array_equal(nrng[1],np.random.get_state()[1])
            self.assertEqual(nrng[0],np.random.get_state()[0]);self.assertEqual(nrng[2:],np.random.get_state()[2:])
            meta=json.loads((root/'a/analysis_metadata.json').read_text());self.assertTrue(meta['completed'])
            self.assertEqual(next(iter(meta['example_selection'].values())),{'seed':0,'individual_id':0})

    def test_analysis_destination_collisions_and_invalid_window_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);recording(root/'raw');self.run_analysis(root/'raw',root/'out')
            before={p.name:hash_file(p) for p in (root/'out').iterdir()}
            with self.assertRaises(ValueError):self.run_analysis(root/'raw',root/'out')
            with self.assertRaises(ValueError):self.run_analysis(root/'raw',root/'raw/sub')
            with self.assertRaises(ValueError):self.run_analysis(root/'raw',root/'new',0)
            self.assertEqual(before,{p.name:hash_file(p) for p in (root/'out').iterdir()})

    def test_pilot_bounds_refuse_large_runs_and_existing_outputs(self):
        from experiments.social_memory_pilot import run_pilot
        with tempfile.TemporaryDirectory() as folder:
            for kwargs in (dict(ticks=101),dict(ticks=0),dict(seeds=(0,1,2)),dict(seeds=(0,0))):
                with self.assertRaises(ValueError):run_pilot(Path(folder)/'unused',**kwargs)
            (Path(folder)/'preserve').write_text('preserve')
            with self.assertRaises(ValueError):run_pilot(Path(folder),ticks=1,seeds=(0,))

    def test_histogram_mean_median_count(self):
        self.assertEqual(histogram_stats(Counter()),(None,None,0))
        self.assertEqual(histogram_stats(Counter({1:1,3:1})),(2,2,2))
        self.assertEqual(histogram_stats(Counter({1:1,3:2})),(7/3,3,3))


if __name__=='__main__':unittest.main()
