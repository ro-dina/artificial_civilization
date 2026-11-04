"""Complete v0.7.2 pipeline, seed statistics, reproducibility and phase guards."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io,json,random,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from simulation.simulation import Simulation
from simulation.vocal_learning_observer import hash_file
from experiments.social_learning_experiments import run_one,run
from experiments.analyze_social_learning import analyze,load,frame,summary,utilization


class SocialLearningAnalysisTests(unittest.TestCase):
    def test_seed_mean_is_unweighted_and_std_is_sample(self):
        f=pd.DataFrame([{'condition':'C','seed':0,'rate':.2},{'condition':'C','seed':2,'rate':.8}])
        row=summary(f,['condition'],['rate']).iloc[0]
        self.assertAlmostEqual(row['mean'],.5);self.assertAlmostEqual(row['std'],np.sqrt(.18));self.assertEqual(row['n'],2)
        self.assertTrue(pd.isna(summary(f.iloc[:1],['condition'],['rate']).iloc[0]['std']))

    def test_utilization_requires_matching_agent_window_base_and_minimum_support(self):
        rows=[]
        for token,action in ((2,0),(4,1)):
            rows.append(dict(window=0,agent_id=1,base_id=1,visible=token,remembered_direction=0,action=action,exploratory=0,n=5))
        data=pd.DataFrame(rows)
        use=utilization(data)[0];self.assertEqual(use['pairs'],1);self.assertEqual(use['tv_mean'],1.)
        self.assertEqual(utilization(data.assign(agent_id=[1,2]))[0]['pairs'],0)
        self.assertEqual(utilization(data.assign(window=[0,1]))[0]['pairs'],0)
        self.assertEqual(utilization(data.assign(base_id=[1,2]))[0]['pairs'],0)
        self.assertEqual(utilization(data.assign(n=[5,4]))[0]['pairs'],0)
        self.assertEqual(utilization(data.assign(visible=[9,9],remembered_direction=[0,4]))[1]['pairs'],1)

    def test_early_phase_and_long_sweep_are_blocked_before_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for options in (dict(phase='primary'),dict(phase='memory_policy'),dict(phase='pilot',ticks=101),dict(phase='primary',ticks=1000)):
                with self.assertRaises(ValueError):run(data_dir=p/'raw',output_root=p/'out',**options)
            self.assertFalse((p/'raw').exists())

    def test_run_replay_initial_ecology_and_all_output_bytes_match(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);paths=[];original_rng=random.getstate()
            for condition in ('A','B','C'):
                p,m,_=run_one(root/'one',condition,0,6)
                q,n,_=run_one(root/'two',condition,0,6)
                self.assertEqual(m['fingerprints'],n['fingerprints']);self.assertEqual(m['action_sha256'],n['action_sha256'])
                self.assertEqual(p.read_bytes(),q.read_bytes())
                for file in m['files'].values():self.assertEqual(hash_file(p.parent/file['file']),hash_file(q.parent/file['file']))
                paths.append((p,m))
            self.assertEqual(len({m['fingerprints']['initial_ecology'] for _,m in paths}),1)
            self.assertEqual(original_rng,random.getstate())
            with self.assertRaises(ValueError):run_one(root/'one','A',0,6)

    def test_complete_analysis_reversed_order_csv_json_png_and_rng(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);paths=[]
            for seed in (0,2):
                for condition in ('A','B','C'):paths.append(run_one(root/'raw',condition,seed,6)[0])
            old={str(p):hash_file(p) for p in paths};rng=random.getstate();nr=np.random.get_state()
            with patch.object(Simulation,'step',side_effect=AssertionError('Analysis attempted simulation')):
                a,_=analyze(paths,root/'a');b,_=analyze(paths[::-1],root/'b')
            self.assertEqual(a,b)
            names=sorted(p.name for p in (root/'a').iterdir());self.assertEqual(len(names),27)
            for name in names:self.assertEqual(hash_file(root/'a'/name),hash_file(root/'b'/name),name)
            self.assertEqual(old,{str(p):hash_file(p) for p in paths});self.assertEqual(rng,random.getstate())
            current=np.random.get_state();np.testing.assert_array_equal(nr[1],current[1]);self.assertEqual(nr[0],current[0]);self.assertEqual(nr[2:],current[2:])
            with self.assertRaises(ValueError):analyze(paths,root/'a')
            f=pd.read_csv(root/'a/seed_summary.csv');self.assertTrue(f[f.metric.eq('population')]['n'].eq(2).all())

    def test_metadata_corruption_and_record_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            p,m,_=run_one(Path(d),'A',0,3);load(p)
            bad=json.loads(p.read_text());bad['completed']=False;p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):load(p)
            p.write_text(json.dumps(m));bad=m.copy();bad['files']=dict(m['files']);bad['files']['ticks']=dict(m['files']['ticks'],records=99)
            with self.assertRaises(ValueError):frame(p,bad,'ticks')
            raw=p.parent/m['files']['ticks']['file'];raw.write_bytes(raw.read_bytes()+b'bad')
            with self.assertRaises(ValueError):load(p)

    def test_empty_population_is_unavailable_not_zero_and_plots_are_safe(self):
        from config import Config,SocialConfig,SocialLearningConfig
        from contextlib import ExitStack
        from simulation.social_learning_observer import SocialLearningObserver,SocialLearningWriter,write_metadata
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            root=Path(d);p=root/'empty.csv.gz'
            with ExitStack() as stack:
                w=SocialLearningWriter.open(stack,p);o=SocialLearningObserver(w)
                s=Simulation(Config(brain='learning',initial_population=0,max_ticks=3),social_config=SocialConfig(True),social_learning_observer=o)
                o.finish(s)
            write_metadata(p,s,o,w,'A');analyze([Path(str(p)+'.metadata.json')],root/'out')
            row=pd.read_csv(root/'out/run_summary.csv').iloc[0]
            self.assertEqual(row['decisions'],0);self.assertTrue(pd.isna(row['toward_fraction']));self.assertTrue(pd.isna(row['new_state_fraction']))


if __name__=='__main__':unittest.main()
