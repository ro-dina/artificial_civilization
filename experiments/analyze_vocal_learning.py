"""Offline Vocal Q diagnostics; equal-seed summaries, no simulator execution."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.vocal_learning_data import discover, read_chunks
from simulation.vocal_learning_observer import hash_file

MEANS = {'present_before_selection': 'hit_rate', 'present_at_start': 'start_hit_rate',
         'rows': 'mean_occupancy', 'q_gap': 'mean_q_gap', 'q_max': 'mean_q_max', 'q_min': 'mean_q_min',
         'q_std': 'mean_q_std', 'p_silence': 'mean_p_silence', 'at_capacity': 'at_capacity_fraction',
         'all_equal': 'all_equal_fraction', 'silence_is_max': 'silence_is_max_fraction',
         'unique_max': 'unique_max_fraction', 'multiple_max': 'multiple_max_fraction',
         'nonuniform': 'nonuniform_fraction', 'silent': 'observed_silence_rate',
         'silence_unique': 'silence_unique_fraction', 'first': 'first_fraction', 'stored': 'stored_fraction',
         'recreated': 'recreated_fraction', 'age': 'mean_age', 'generation': 'mean_generation',
         'decision_evictions': 'decision_eviction_rate', 'exploratory': 'exploratory_fraction',
         'state_visits': 'mean_state_visits'}
MEANS.update(max_ties='mean_max_ties', negative_only='negative_only_fraction', positive_max='positive_max_fraction',
             policy_tv='mean_policy_tv', selected_q='mean_selected_q')
COUNTS = ['actor_decisions', 'decisions', 'updates', 'allocations', 'evictions', 'evicted_updates',
          'evicted_visits', 'learned_evictions', 'deaths', 'run_ends', 'terminal_updates',
          'recreated_updates', 'positive_td', 'negative_td', 'zero_td']
UPDATE_MEANS = {'abs_td_error': 'mean_abs_td_error', 'td_error': 'mean_td_error', 'q_change': 'mean_q_change'}


class Collector:
    """Chunk-local pandas reduction; retain scalar sums, never raw decision history."""
    def __init__(self):
        self.groups = defaultdict(lambda: defaultdict(float))

    def add(self, frame, keys, values):
        if frame.empty:
            return
        summed = frame.groupby(keys, sort=True, dropna=False)[values].sum()
        for key, row in summed.iterrows():
            key = key if isinstance(key, tuple) else (key,)
            for name, value in row.items():
                self.groups[key][name] += float(value)

    def table(self, keys):
        return pd.DataFrame([{**dict(zip(keys, key)), **metrics(value)} for key, value in sorted(self.groups.items())])


def metrics(sums):
    n, u = sums.get('decisions', 0), sums.get('updates', 0)
    row = {k: sums.get(k, 0) for k in COUNTS}
    row.update({metric: sums.get(source, 0) / n if n else np.nan for source, metric in MEANS.items()})
    row.update({metric: sums.get(source, 0) / u if u else np.nan for source, metric in UPDATE_MEANS.items()})
    for source, name in [('allocations', 'allocation_rate'), ('evictions', 'eviction_rate')]:
        row[name] = sums.get(source, 0) / n if n else np.nan
    row['miss_rate'] = 1 - row['hit_rate']
    row['mean_updates_at_eviction'] = sums.get('evicted_updates', 0) / sums['evictions'] if sums.get('evictions') else np.nan
    row['mean_visits_at_eviction'] = sums.get('evicted_visits', 0) / sums['evictions'] if sums.get('evictions') else np.nan
    row['row_visit_sum'] = sums.get('state_visits', 0)
    return row


def prepare_decisions(frame, applicable):
    f = frame.copy()
    f['actor_decisions'] = 1
    f['decisions'] = int(applicable)
    if applicable:
        f['unique_max'] = f.max_ties.eq(1).astype(int)
        f['multiple_max'] = f.max_ties.gt(1).astype(int)
        f['nonuniform'] = f.all_equal.eq(0).astype(int)
        f['silent'] = f.selected_action.eq(0).astype(int)
        f['silence_unique'] = (f.silence_is_max.eq(1) & f.max_ties.eq(1)).astype(int)
        f['negative_only'] = (f.q_max.le(0) & f.q_min.lt(f.q_max)).astype(int)
        f['positive_max'] = f.q_max.gt(0).astype(int)
        # Exact total variation of the existing epsilon-greedy policy from uniform.
        f['policy_tv'] = (1-f.epsilon)*(1-f.max_ties/f.n_actions)
        for k in ('first', 'stored', 'recreated'):
            f[k] = f.novelty.eq(k).astype(int)
    else:
        for k in MEANS:
            if k not in f:
                f[k] = np.nan
    return f


def equal_seed(frame, keys):
    if frame.empty:
        return pd.DataFrame(columns=[*keys, 'metric', 'mean', 'std', 'min', 'max', 'n'])
    skip = {*keys, 'seed', 'run', 'applicable', 'first_age', 'last_age', 'agent_id', 'unique_states'}
    fields = [c for c in frame.select_dtypes(include='number') if c not in skip]
    long = frame.melt(id_vars=[*keys, 'seed'], value_vars=fields, var_name='metric', value_name='value')
    if long.duplicated([*keys, 'metric', 'seed']).any():
        raise ValueError('More than one contribution per seed/metric')
    return long.groupby([*keys, 'metric'], dropna=False, sort=True).value.agg(
        ['mean', 'std', 'min', 'max', 'count']).reset_index().rename(columns={'count': 'n'})


def fixed_window_pairs(frame, window_size, final_tick, threshold):
    """Same IDs in explicitly fixed early/next and early/final windows."""
    records, summaries = [], []
    fields = ('hit_rate', 'mean_q_gap', 'mean_p_silence', 'nonuniform_fraction', 'unique_max_fraction', 'mean_age')
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    late = ((final_tick - 1) // window_size) * window_size if final_tick else 0
    eligible = frame.loc[frame.decisions.ge(threshold)]
    first = eligible.loc[eligible.window_start.eq(0)].set_index('agent_id')
    for name, end in [('early_next', window_size), ('early_final', late)]:
        if end <= 0:
            continue
        last = eligible.loc[eligible.window_start.eq(end)].set_index('agent_id')
        ids = sorted(first.index.intersection(last.index))
        for aid in ids:
            row = {'comparison':name, 'agent_id':aid, 'first_window':0, 'last_window':end,
                   'first_decisions':first.at[aid,'decisions'], 'last_decisions':last.at[aid,'decisions']}
            for field in fields:
                row['delta_'+field] = last.at[aid,field]-first.at[aid,field]
            records.append(row)
        for field in fields:
            values = [r['delta_'+field] for r in records if r['comparison'] == name]
            summaries.append({'comparison':name, 'metric':'delta_'+field, 'paired_agents':len(ids),
                              'mean':float(np.mean(values)) if values else np.nan,
                              'std':float(np.std(values,ddof=1)) if len(values)>1 else np.nan})
    return pd.DataFrame(records), pd.DataFrame(summaries)


def analyze_run(run, window_size, age_bin, min_decisions, min_pair_decisions):
    catalog, exact = {}, set()
    for frame in read_chunks(run, 'states'):
        for row in frame.itertuples(index=False):
            key = json.dumps(json.loads(row.key_json), separators=(',', ':'))
            if row.state_id in catalog or key in exact:
                raise ValueError('Catalog IDs/keys must be one-to-one')
            catalog[int(row.state_id)] = key
            exact.add(key)
    run['catalog'] = set(catalog)
    windows, individuals, ages, individual_ages, individual_windows, state_updates = (Collector() for _ in range(6))
    bounds, unique, ticks = {}, {}, defaultdict(set)
    for f in read_chunks(run, 'decisions'):
        f = prepare_decisions(f, run['applicable'])
        f['window_start'] = ((f.tick-1)//window_size)*window_size
        f['age_start'] = (f.age//age_bin)*age_bin
        vals = ['actor_decisions', 'decisions', *MEANS]
        for collector, keys in ((windows, ['window_start']), (individuals, ['agent_id']), (ages, ['age_start']),
                                (individual_ages, ['agent_id', 'age_start']), (individual_windows, ['agent_id', 'window_start'])):
            collector.add(f, keys, vals)
        for wid, block in f.groupby('window_start', sort=True):
            ticks[int(wid)].update(int(x) for x in block.tick.unique())
        for aid, block in f.groupby('agent_id', sort=True):
            first, last = int(block.age.min()), int(block.age.max())
            old = bounds.get(int(aid), (first, last))
            bounds[int(aid)] = (min(first, old[0]), max(last, old[1]))
            unique[int(aid)] = max(unique.get(int(aid), 0), int(block.unique_states.max()) if run['applicable'] else 0)
    for f in read_chunks(run, 'updates'):
        if not run['applicable']:
            raise ValueError('Vocal updates in an inapplicable control')
        if not np.allclose(f.td_error, f.target-f.old_q, rtol=0, atol=0):
            raise ValueError('Recorded TD error inconsistent with actual target')
        f['window_start'] = ((f.tick-1)//window_size)*window_size
        f['age_start'] = (f.age//age_bin)*age_bin
        f['updates'] = 1
        f['terminal_updates'] = f.terminal
        f['recreated_updates'] = f.recreated
        f['positive_td'], f['negative_td'], f['zero_td'] = (f.td_error.gt(0).astype(int), f.td_error.lt(0).astype(int), f.td_error.eq(0).astype(int))
        vals = ['updates', *UPDATE_MEANS, 'terminal_updates', 'recreated_updates', 'positive_td', 'negative_td', 'zero_td']
        for collector, keys in ((windows, ['window_start']), (individuals, ['agent_id']), (ages, ['age_start']),
                                (individual_ages, ['agent_id', 'age_start']), (individual_windows, ['agent_id', 'window_start']),
                                (state_updates, ['agent_id', 'state_id'])):
            collector.add(f, keys, vals)
    for f in read_chunks(run, 'rows'):
        if not f.event.isin(['allocate', 'evict', 'death', 'run_end']).all():
            raise ValueError('Unknown row event')
        f['window_start'] = ((f.tick-1)//window_size)*window_size
        f['allocations'], f['evictions'] = f.event.eq('allocate').astype(int), f.event.eq('evict').astype(int)
        f['evicted_updates'] = f.updates.where(f.event.eq('evict'), 0)
        f['evicted_visits'] = f.visits.where(f.event.eq('evict'), 0)
        f['learned_evictions'] = (f.event.eq('evict') & f.updates.gt(0)).astype(int)
        f['deaths'], f['run_ends'] = f.event.eq('death').astype(int), f.event.eq('run_end').astype(int)
        vals = ['allocations', 'evictions', 'evicted_updates', 'evicted_visits', 'learned_evictions', 'deaths', 'run_ends']
        windows.add(f, ['window_start'], vals)
        individuals.add(f, ['agent_id'], vals)
        individual_windows.add(f, ['agent_id', 'window_start'], vals)
    for start in range(0, run['metadata']['final_tick'], window_size):
        windows.groups[(start,)]
    w = windows.table(['window_start'])
    if w.empty:
        w = pd.DataFrame(columns=['window_start', *metrics({})])
    w['window_end'] = w.window_start + window_size
    w['actual_window_end'] = w.window_end.clip(upper=run['metadata']['final_tick'])
    w['observed_ticks'] = w.window_start.map(lambda x: len(ticks[x]))
    w['partial_window'] = w.window_end.gt(w.actual_window_end)
    w['complete_window'] = w.observed_ticks.eq(w.actual_window_end-w.window_start) & ~w.partial_window
    i = individuals.table(['agent_id'])
    if i.empty:
        i = pd.DataFrame(columns=['agent_id', *metrics({})])
    i['first_age'] = i.agent_id.map(lambda aid: bounds.get(aid, (np.nan, np.nan))[0])
    i['last_age'] = i.agent_id.map(lambda aid: bounds.get(aid, (np.nan, np.nan))[1])
    i['unique_states'] = i.agent_id.map(unique)
    i['eligible'] = i.decisions.ge(min_decisions)
    ia, iw, a, su = (individual_ages.table(['agent_id', 'age_start']), individual_windows.table(['agent_id', 'window_start']),
                     ages.table(['age_start']), state_updates.table(['agent_id', 'state_id']))
    paired = []
    if not iw.empty:
        for aid, block in iw.loc[iw.decisions.ge(min_pair_decisions)].groupby('agent_id', sort=True):
            block = block.sort_values('window_start')
            if len(block) < 2:
                continue
            first, last = block.iloc[0], block.iloc[-1]
            row = {'agent_id': aid, 'first_window': first.window_start, 'last_window': last.window_start,
                   'first_decisions': first.decisions, 'last_decisions': last.decisions}
            for metric in ('mean_q_gap', 'hit_rate', 'mean_p_silence', 'observed_silence_rate', 'mean_age', 'nonuniform_fraction', 'unique_max_fraction'):
                row['first_'+metric], row['last_'+metric] = first[metric], last[metric]
                row['delta_'+metric] = last[metric]-first[metric]
            paired.append(row)
    if not su.empty:
        su['mixed_td_signs'] = (su.positive_td.gt(0) & su.negative_td.gt(0)).astype(int)
        su = su[['agent_id','state_id','updates','positive_td','negative_td','zero_td',
                 'mean_abs_td_error','mean_td_error','mean_q_change','mixed_td_signs']]
    if not a.empty and not ia.empty:
        a = a.merge(ia.groupby('age_start',sort=True).agent_id.nunique().rename('n_agents'),on='age_start',validate='one_to_one')
    fixed_pairs, fixed_summary = fixed_window_pairs(iw, window_size, run['metadata']['final_tick'], min_pair_decisions)
    # Run statistics are recomputed from scalar totals, not unweighted window means.
    totals = defaultdict(float)
    for data in windows.groups.values():
        for key, value in data.items():
            totals[key] += value
    r = {**metrics(totals), 'final_tick': run['metadata']['final_tick'], 'unique_keys_run': len(catalog),
         'unique_agent_keys': sum(unique.values()), 'eligible_agents': int(i.eligible.sum()),
         'paired_agents': len(paired), 'applicable': run['applicable']}
    if not run['applicable']:
        for frame in (w, i, ia, iw, a):
            for col in set(MEANS.values()) | set(UPDATE_MEANS.values()) | {'miss_rate','eviction_rate','allocation_rate'}:
                if col in frame:
                    frame[col] = np.nan
    context = {'run': run['run'], 'condition': run['condition'], 'seed': run['seed']}
    result = {'windows': w, 'individuals': i, 'ages': a, 'individual_ages': ia, 'individual_windows': iw,
              'paired_individuals': pd.DataFrame(paired), 'state_learning': su, 'runs': pd.DataFrame([r])}
    result.update(paired_early_late=fixed_pairs, paired_window_summary=fixed_summary)
    if not run['applicable']:
        # Absence of a Vocal Q head is not an all-zero learner. Actor exposure
        # and timing/identity remain defined; every Vocal Q measurement is N/A.
        unavailable = ((set(COUNTS) - {'actor_decisions'}) | set(MEANS.values()) | set(UPDATE_MEANS.values()) |
                       {'unique_states','unique_keys_run','unique_agent_keys','paired_agents','eligible_agents',
                        'miss_rate','allocation_rate','eviction_rate','mean_updates_at_eviction',
                        'mean_visits_at_eviction','row_visit_sum'})
        for frame in result.values():
            for col in unavailable & set(frame):
                frame[col] = np.nan
    return {k: frame.assign(**context) for k, frame in result.items()}


def join_communication(runs, windows, individual_windows, directory, age_bin, verification=None):
    """Exact simultaneous-recording hash + full Config + archived analysis-input hashes."""
    directory = Path(directory)
    meta_path = directory/'analysis_metadata.json'
    if not meta_path.exists():
        raise ValueError('Communication analysis metadata required for safe join')
    meta = json.loads(meta_path.read_text())
    accepted, receipts = [], []
    for run in runs:
        certificate = run['metadata'].get('communication')
        if certificate is None:
            continue
        path = Path(certificate['file'])
        sidecar = Path(str(path)+'.metadata.json')
        if hash_file(path) != certificate['sha256'] or hash_file(sidecar) != certificate['metadata_sha256']:
            raise ValueError('Simultaneous communication file changed')
        cm = json.loads(sidecar.read_text())
        if cm['config'] != run['metadata']['config'] or cm['final_tick'] != run['metadata']['final_tick'] or cm['statistics'] != run['metadata']['statistics']:
            raise ValueError('Communication trajectory/config differs from diagnostics')
        matches = [m for m in meta['inputs'] if m['sha256'] == certificate['sha256'] and
                   m['simulation_metadata']['config'] == cm['config'] and m['simulation_metadata']['statistics'] == cm['statistics'] and
                   m['simulation_metadata']['final_tick'] == cm['final_tick']]
        if len(matches) != 1:
            raise ValueError('No unique identical archived communication trajectory; seed alone is insufficient')
        old = matches[0]
        accepted.append((run, old))
        receipts.append({'run': run['run'], 'archived_file': old['file'], 'communication_sha256': old['sha256'],
                         'diagnostic_metadata_sha256': hash_file(run['metadata_path']), 'config_id': cm['config_id']})
    source = directory/'sender_information.csv'
    individual_source = directory/'sender_information_by_agent.csv'
    if verification:
        saved = json.loads(Path(verification).read_text())['output_hashes']
        for path in (meta_path,source,individual_source):
            if hash_file(path) != saved[path.name]:
                raise ValueError('Archived communication analysis differs from its verification hashes')
    # The v0.6.1 analysis metadata binds inputs; bind the reused output explicitly too.
    mi = pd.read_csv(source)
    joined = []
    for run, old in accepted:
        m = mi.loc[mi.run.eq(Path(old['file']).stem) & mi.scope.eq('vocal_including_silence') &
                   mi.conditioning.eq('within_agent_eligible')]
        w = windows.loc[windows.run.eq(run['run'])]
        part = w.merge(m[['window_start','actual_window_end','feature','n','eligible_agents','corrected_mi_bits']],
                       on=['window_start','actual_window_end'], how='inner', validate='one_to_many')
        joined.append(part)
    individuals = []
    if individual_source.exists() and accepted:
        mapping = {Path(old['file']).stem:run['run'] for run,old in accepted}
        cols = ['run','agent_id','window_start','actual_window_end','feature','scope','n','min_samples','corrected_mi_bits']
        base = individual_windows.merge(windows[['run','window_start','actual_window_end']],
                    on=['run','window_start'],validate='many_to_one')
        for chunk in pd.read_csv(individual_source,usecols=cols,chunksize=50000):
            chunk = chunk.loc[chunk.run.isin(mapping)&chunk.scope.eq('vocal_including_silence')&
                              chunk.feature.isin(['food_direction','water_direction'])].copy()
            if chunk.empty:
                continue
            chunk['run'] = chunk.run.map(mapping)
            chunk = chunk.rename(columns={'n':'mi_n'})
            part = base.merge(chunk,on=['run','agent_id','window_start','actual_window_end'],validate='one_to_many')
            part['mean_age_bin'] = (part.mean_age//age_bin)*age_bin
            individuals.append(part)
    return (pd.concat(joined, ignore_index=True) if joined else pd.DataFrame(),
            pd.concat(individuals,ignore_index=True) if individuals else pd.DataFrame(),
            {'matches': receipts, 'analysis_metadata_sha256': hash_file(meta_path), 'sender_information_sha256': hash_file(source),
             'individual_sender_information_sha256':hash_file(individual_source) if individual_source.exists() else None,
             'verification_sha256':hash_file(verification) if verification else None})


def paired_information(frame, window_size):
    """Within-ID MI/Q changes on the same eligible recorded windows, no new nulls."""
    records, seeds = [], []
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (run,c,seed,feature),block in frame.groupby(['run','condition','seed','feature'],sort=True):
        context={'run':run,'condition':c,'seed':seed,'feature':feature}
        early=block.loc[block.window_start.eq(0)].set_index('agent_id')
        for comparison,late_start in [('early_next',window_size),('early_final',int(block.window_start.max()))]:
            late=block.loc[block.window_start.eq(late_start)].set_index('agent_id')
            ids=sorted(early.index.intersection(late.index)) if late_start>0 else []
            deltas=[]
            for aid in ids:
                row={**context,'comparison':comparison,'agent_id':aid,'first_window':0,'last_window':late_start,
                     'first_mi_n':early.at[aid,'mi_n'],'last_mi_n':late.at[aid,'mi_n']}
                for metric in ('corrected_mi_bits','mean_q_gap','hit_rate','mean_age'):
                    row['delta_'+metric]=late.at[aid,metric]-early.at[aid,metric]
                records.append(row);deltas.append(row)
            seeds.append({**context,'comparison':comparison,'paired_agents':len(ids),
                          **{'mean_delta_'+m:float(np.mean([r['delta_'+m] for r in deltas])) if deltas else np.nan
                             for m in ('corrected_mi_bits','mean_q_gap','hit_rate','mean_age')}})
    return pd.DataFrame(records),pd.DataFrame(seeds)


def analyze(data_dir=Path('data/vocal_learning'), output_dir=Path('analysis/vocal_learning'), window_size=100,
            age_bin=25, min_decisions=50, min_pair_decisions=20, communication_analysis=None, communication_verification=None):
    if any(type(x) is not int or x < 1 for x in (window_size, age_bin, min_decisions, min_pair_decisions)):
        raise ValueError('Window, age bin and thresholds must be positive integers')
    runs = discover(data_dir)
    output_dir = Path(output_dir).resolve()
    if any(p.resolve().is_relative_to(output_dir) for r in runs for p in [r['metadata_path'], *r['paths'].values()]):
        raise ValueError('Output directory cannot contain/overwrite input recordings')
    output_dir.mkdir(parents=True, exist_ok=True)
    results = defaultdict(list)
    for run in runs:
        print(f"Analyzing {run['condition']} seed={run['seed']}", flush=True)
        for name, table in analyze_run(run, window_size, age_bin, min_decisions, min_pair_decisions).items():
            results[name].append(table)
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in results.items()}
    # Explicit schemas even for no qualifying pairs/states/age samples.
    for name, cols in {'ages':['age_start'], 'individual_ages':['agent_id','age_start'],
                       'individual_windows':['agent_id','window_start'], 'paired_individuals':['agent_id','first_window','last_window'],
                       'state_learning':['agent_id','state_id']}.items():
        if tables[name].empty:
            tables[name] = pd.DataFrame(columns=['run','condition','seed',*cols])
    summary = equal_seed(tables['windows'], ['condition','window_start','window_end','actual_window_end'])
    age_summary = equal_seed(tables['ages'], ['condition','age_start'])
    run_summary = equal_seed(tables['runs'], ['condition'])
    tables.update(time_series_summary=summary, age_summary=age_summary, run_summary=run_summary)
    joined, individual_join, receipt = (join_communication(runs, tables['windows'],tables['individual_windows'],
                                        communication_analysis,age_bin,communication_verification)
                       if communication_analysis else (pd.DataFrame(),pd.DataFrame(), None))
    if communication_analysis:
        tables['communication_comparison'] = joined
        tables['individual_communication_comparison'] = individual_join
        if not individual_join.empty:
            grouped = individual_join.groupby(['run','condition','seed','mean_age_bin','feature'],sort=True)
            tables['age_sender_mi'] = grouped.agg(mean_corrected_mi_bits=('corrected_mi_bits','mean'),
                n_agent_windows=('agent_id','size'),n_agents=('agent_id','nunique'),mi_decisions=('mi_n','sum')).reset_index()
            tables['age_sender_mi_summary'] = equal_seed(tables['age_sender_mi'],['condition','mean_age_bin','feature'])
            tables['paired_sender_mi'],tables['paired_mi_seed_summary'] = paired_information(individual_join,window_size)
            tables['paired_mi_summary'] = equal_seed(tables['paired_mi_seed_summary'],['condition','comparison','feature'])
    for name, frame in tables.items():
        frame.to_csv(output_dir/(name+'.csv'), index=False, float_format='%.12g')
    from experiments.vocal_learning_plots import plots
    images, selection = plots(tables, runs, output_dir, min_decisions)
    import matplotlib
    metadata = {'version':'0.6.2', 'window_size':window_size, 'age_bin':age_bin, 'min_decisions':min_decisions,
                'min_pair_decisions':min_pair_decisions, 'aggregation':'Per seed first, equal seed mean, sample SD ddof=1, finite n; no interpolation.',
                'selection':selection, 'inputs':[{'run':r['run'],'condition':r['condition'],'seed':r['seed'],
                'metadata_sha256':hash_file(r['metadata_path']), 'files':r['metadata']['files'], 'config':r['metadata']['config']} for r in runs],
                'communication_join':receipt, 'png_files':images, 'csv_files':[name+'.csv' for name in tables],
                'versions':{'pandas':pd.__version__, 'numpy':np.__version__, 'matplotlib':matplotlib.__version__},
                'source_sha256':{p.name:hash_file(p) for p in (Path(__file__), Path(__file__).with_name('vocal_learning_data.py'),
                                                            Path(__file__).with_name('vocal_learning_plots.py'))}}
    (output_dir/'analysis_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(tables['runs'][['condition','seed','decisions','hit_rate','eviction_rate','mean_q_gap','mean_p_silence','observed_silence_rate']].to_string(index=False))
    print(f'Outputs: {output_dir}')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/vocal_learning'))
    parser.add_argument('--output-dir',type=Path,default=Path('analysis/vocal_learning'))
    parser.add_argument('--window-size',type=int,default=100)
    parser.add_argument('--age-bin',type=int,default=25)
    parser.add_argument('--min-decisions',type=int,default=50)
    parser.add_argument('--min-pair-decisions',type=int,default=20)
    parser.add_argument('--communication-analysis',type=Path)
    parser.add_argument('--communication-verification',type=Path,help='Optional archived output-hash verification JSON')
    args = parser.parse_args()
    try:
        analyze(**vars(args))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))

if __name__ == '__main__':
    main()
