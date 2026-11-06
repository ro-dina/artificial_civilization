"""Frozen pilot/training gates, same-mode models, untouched new-seed scoring."""
from pathlib import Path
from time import perf_counter
from hashlib import sha256
import argparse
import json

import numpy as np
import pandas as pd

from experiments.future_resource_io import digest, save_csv, save_json, lookup_counts
from experiments.future_resource_prediction import score, summarize, keys as h_keys
from experiments.dense_resource_data import (ROOT, OUTPUT, plan, cases, load_case, prepare, patterns, combine_patterns, exact_counts, assess, behavior_reference, scientific_sources)
from experiments.dense_resource_models import Encoding, fit_logistic, hierarchy_from_patterns, feature_counts, feature_support, sigmoid
from experiments.resource_ecology_experiments import peak


def tag(meta):
    return {key: meta[key] for key in ('ecology', 'mode', 'seed')}


def input_hashes(path, meta):
    path = Path(path)
    values = {str(path): digest(path)}
    for r in meta['files'].values():
        p = path.parent / r['file']; values[str(p)] = r['sha256']
    dense = path.parent / meta['dense']['file']
    values[str(dense)] = meta['dense']['sha256']; values[str(dense) + '.json'] = digest(str(dense) + '.json')
    return values


def marginal(frame, labels, reasons):
    valid = labels >= 0; reference = reasons != 6; n = int(valid.sum()); positive = int(labels[valid].sum())
    return {'source_decisions': len(frame), 'reference_observations': int(reference.sum()), 'observed': n,
        'positive': positive, 'negative': n - positive, 'prevalence': positive / n if n else None,
        'label_coverage': n / int(reference.sum()) if reference.any() else 0., 'agents': int(frame.loc[valid, 'agent_id'].nunique()),
        'administrative_end': int((reasons == 2).sum()), 'not_at_risk': int((reasons == 6).sum())}


def marginal_pass(value, thresholds):
    return bool(value['observed'] and thresholds['prevalence_min'] <= value['prevalence'] <= thresholds['prevalence_max']
        and min(value['positive'], value['negative']) >= thresholds['minimum_each_class'] and value['agents'] >= thresholds['minimum_agents']
        and value['label_coverage'] >= thresholds['minimum_label_coverage'])


def review_pilot(root=ROOT, output=OUTPUT):
    root, output = Path(root), Path(output); rules = plan(root); rows = []; inputs = {}; start = perf_counter()
    for path in cases(root, 'pilot', rules['pilot']['seeds']):
        meta, files, frame, labels = prepare(path, audit=True); inputs.update(input_hashes(path, meta))
        for target, (values, reasons) in labels.items():
            value = marginal(frame, values, reasons)
            diagnostic, cells = assess(frame, values, reasons, patterns(frame, values), rules)
            rows.append({**tag(meta), 'target': target, **value, 'within_pilot_exact_coverage': diagnostic['M01_common_coverage'],
                'within_pilot_mixed_strata': diagnostic['adequately_mixed_baseline_strata'], 'conditional_support_is_not_independent_training': True,
                'marginal_pass': marginal_pass(value, rules['thresholds']), 'complete_ticks': meta['tick'] == 100,
                'snapshot_complete': meta['dense']['records'] == meta['tick'] + 1, 'population_viable': meta['statistics']['population'] >= 10,
                'decisions_viable': len(frame) >= 5000, 'consumption_viable': min(meta['statistics'][r + '_consumed'] for r in ('food', 'water')) >= 20})
    f = pd.DataFrame(rows); primary = f[f.target.eq('T1')]
    required = ['marginal_pass', 'complete_ticks', 'snapshot_complete', 'population_viable', 'decisions_viable', 'consumption_viable']
    allowed = len(primary) == 12 and bool(primary[required].all().all())
    save_csv(output / 'pilot/support.csv', f)
    save_json(root / 'pilot_review.json', {'allowed': allowed, 'decision': 'training_allowed' if allowed else 'blocked', 'inputs': inputs,
        'preregistration_sha256': digest(root / 'preregistration.json'), 'source_sha256': scientific_sources(),
        'wall_seconds': perf_counter() - start, 'peak_memory_bytes': peak(), 'no_predictive_gains_used': True,
        'note': 'Technical and marginal feasibility gate; within-run pilot cell support is diagnostic only, no independent predictive comparison.'})
    print('PILOT:', allowed, flush=True)


def training_review(root=ROOT, output=OUTPUT):
    root, output = Path(root), Path(output); rules = plan(root); start = perf_counter()
    rows, fits, inputs, strata = [], [], {}, []
    for ecology in ('E0', 'E1', 'E2', 'E3'):
        for mode in 'ABC':
            paths = [p for p in cases(root, 'training', rules['train_seeds']) if p.name.startswith(ecology + '_' + mode + '_')]
            by_seed = {}
            for path in paths:
                meta, files, frame, labels = prepare(path, audit=True); inputs.update(input_hashes(path, meta))
                if meta['tick'] != 500: raise ValueError('Incomplete500 training case; stop and report, do not pad')
                by_seed[meta['seed']] = {target: patterns(frame, values) for target, (values, reasons) in labels.items()}
            local_rows = []
            for path in paths:
                meta, files, frame, labels = prepare(path)
                for target, (values, reasons) in labels.items():
                    training = combine_patterns([tables[target] for seed, tables in by_seed.items() if seed != meta['seed']])
                    value, cells = assess(frame, values, reasons, training, rules)
                    row = {**tag(meta), 'target': target, **value, 'stage': 'training_leave_one_seed_out'}
                    rows.append(row); local_rows.append(row)
            local = pd.DataFrame(local_rows)
            for target in ('T1', 'T2'):
                support = local[local.target.eq(target)]
                qualification = {'H1': bool(support.eligible.all()), 'H2': bool(support.memory_eligible.all()),
                                 'R1': bool(support.R1_eligible.all()), 'R2': bool(support.R2_eligible.all())}
                full = combine_patterns([v[target] for v in by_seed.values()])
                stem = f'{ecology}_{mode}_{target}'
                pattern_path = root / 'models' / (stem + '.patterns.csv.gz'); save_csv(pattern_path, full)
                encoder = Encoding(2); indices = encoder.transform(full)
                counts = feature_counts(indices, encoder.dimension, full['count'].to_numpy()).tolist()
                regression = {}
                for level in range(3):
                    name = 'R' + str(level); e = Encoding(level)
                    if not qualification['R1'] or level == 2 and not qualification['R2']:
                        regression[name] = None; continue
                    before = perf_counter()
                    fitted = fit_logistic(indices[:, :13 + level], full.positive.to_numpy(), full['count'].to_numpy(), e.dimension)
                    fitted['wall_seconds'] = perf_counter() - before
                    regression[name] = fitted
                    fits.append({'ecology': ecology, 'mode': mode, 'target': target, 'model': name, **{k: v for k, v in fitted.items() if k != 'coefficients'}})
                if not all(regression.get(name) and regression[name]['converged'] for name in ('R0', 'R1')): qualification['R1'] = qualification['R2'] = False
                if not regression.get('R2') or not regression['R2']['converged']: qualification['R2'] = False
                model_path = root / 'models' / (stem + '.json')
                save_json(model_path, {'schema_version': 1, 'ecology': ecology, 'mode': mode, 'target': target,
                    'training_seeds': rules['train_seeds'], 'same_mode_training': True, 'training_qualification': qualification,
                    'pattern_file': str(pattern_path), 'pattern_sha256': digest(pattern_path), 'feature_level_counts': counts,
                    'regularized': regression, 'preregistration_sha256': digest(root / 'preregistration.json'), 'source_sha256': scientific_sources()})
                inputs[str(pattern_path)] = digest(pattern_path); inputs[str(model_path)] = digest(model_path)
                strata.append({'ecology': ecology, 'mode': mode, 'target': target, **qualification})
                print('TRAINING GATE', ecology, mode, target, qualification, flush=True)
    f = pd.DataFrame(rows); q = pd.DataFrame(strata)
    save_csv(output / 'training/support.csv', f); save_csv(output / 'training/qualification.csv', q); save_csv(output / 'training/convergence.csv', pd.DataFrame(fits))
    allowed = bool(q[['H1', 'R1']].any().any())
    save_json(root / 'training_review.json', {'allowed': allowed, 'decision': 'evaluation_allowed' if allowed else 'blocked_all_training_support',
        'inputs': inputs, 'source_sha256': scientific_sources(), 'preregistration_sha256': digest(root / 'preregistration.json'),
        'qualified_strata': strata, 'wall_seconds': perf_counter() - start, 'peak_memory_bytes': peak(),
        'no_evaluation_outcomes_seen': True, 'no_gain_selection': True})
    print('EVALUATION ALLOWED:', allowed, flush=True)


def evaluate(root=ROOT, output=OUTPUT):
    root, output = Path(root), Path(output); rules = plan(root); start = perf_counter()
    from experiments.dense_resource_study import allowed
    if not allowed(root, 'evaluation'): raise ValueError('Evaluation blocked by training review')
    rows, paired, bins, support_rows, diagnostics, probes, vocal, inputs = [], [], [], [], [], [], [], {}
    for ecology in ('E0', 'E1', 'E2', 'E3'):
        for mode in 'ABC':
            models = {}
            for target in ('T1', 'T2'):
                path = root / 'models' / f'{ecology}_{mode}_{target}.json'; model = json.loads(path.read_text())
                if digest(model['pattern_file']) != model['pattern_sha256']: raise ValueError('Predictor counts changed')
                pattern = pd.read_csv(model['pattern_file']); models[target] = (model, pattern, hierarchy_from_patterns(pattern))
                inputs[str(path)] = digest(path); inputs[model['pattern_file']] = model['pattern_sha256']
            for path in [p for p in cases(root, 'evaluation', rules['evaluation_seeds']) if p.name.startswith(ecology + '_' + mode + '_')]:
                meta, files, frame, labels = prepare(path, audit=True); inputs.update(input_hashes(path, meta))
                den = meta['totals']['decisions']; hist = pd.read_csv(files['distributions']); visits = hist[hist.metric.eq('physical_state_visits')]
                diagnostics.append({**tag(meta), **meta['statistics'], 'decisions': den, 'first_state_fraction': meta['totals']['first_agent_states'] / den,
                    'mean_visits_per_agent_state': float((visits.value * visits['count']).sum() / visits['count'].sum()),
                    'q_occupancy': meta['totals']['occupancy_sum'] / den, 'evictions': meta['totals'].get('evictions', 0),
                    'emission_rate': meta['statistics']['signals_emitted'] / den, 'identified_fraction': meta['totals'].get('identified', 0) / den,
                    'masked_fraction': meta['totals'].get('masked', 0) / den,
                    'vocal_q_rows_final': float(pd.read_csv(files['resource_ticks']).mean_vocal_q_rows.iloc[-1])})
                probes.append(pd.read_csv(files['probes']).assign(**tag(meta))); vocal.append(pd.read_csv(files['vocal']).assign(**tag(meta)))
                behavior, reasons = behavior_reference(frame, meta['tick'])
                support_rows.append({**tag(meta), 'target': 'B10', 'kind': 'descriptive_behavior_reference', **marginal(frame, behavior, reasons),
                    'death_competing': int((reasons == 4).sum()), 'history_missing': int((reasons == 3).sum())})
                for target, (values, reasons) in labels.items():
                    model, pattern, hierarchy = models[target]; assessed, cells = assess(frame, values, reasons, pattern, rules)
                    support_rows.append({**tag(meta), 'target': target, 'kind': 'environmental', **assessed})
                    valid = values >= 0; observed = frame.loc[valid]; y = values[valid]
                    feature_columns = ['x_key', 'action', 'b_token', 'rd_token']
                    codes, unique = pd.factorize(pd.MultiIndex.from_frame(observed[feature_columns]), sort=False)
                    unique_frame = unique.to_frame(index=False); unique_frame.columns = feature_columns
                    h_probability = np.zeros((len(unique_frame), 3)); h_supported = np.zeros((len(unique_frame), 3), bool)
                    for i, r in enumerate(unique_frame.itertuples(index=False)):
                        p, s = hierarchy.predict(h_keys(r.x_key, r.action, r.b_token, r.rd_token), r.action)
                        h_probability[i] = [p[n][1] for n in ('X_A', 'X_B_A', 'X_C_A')]; h_supported[i] = [s[n] for n in ('X_A', 'X_B_A', 'X_C_A')]
                    encoded = Encoding(2).transform(unique_frame); feature_count = np.array(model['feature_level_counts'])
                    common01 = h_supported[codes, :2].all(axis=1); common012 = h_supported[codes].all(axis=1)
                    r01 = feature_support(encoded[:, :14], feature_count)[codes]; r012 = feature_support(encoded, feature_count)[codes]
                    masks = {'all': np.ones(len(y), bool), 'H01': common01, 'H012': common012, 'R01': r01, 'R012': r012,
                             'R01_outside_H01': r01 & ~common01, 'R012_outside_H012': r012 & ~common012}
                    probabilities = {f'H{i}': h_probability[codes, i] for i in range(3)}
                    for i in range(3):
                        fitted = model['regularized'].get('R' + str(i))
                        if fitted and fitted['converged']:
                            probabilities['R' + str(i)] = sigmoid(np.array(fitted['coefficients'])[encoded[:, :13 + i]].sum(axis=1))[codes]
                    qualification = {'H1': model['training_qualification']['H1'] and assessed['eligible'],
                        'H2': model['training_qualification']['H2'] and assessed['memory_eligible'],
                        'R1': model['training_qualification']['R1'] and assessed['R1_eligible'],
                        'R2': model['training_qualification']['R2'] and assessed['R2_eligible']}
                    for subset, mask in masks.items():
                        rowset = sha256(observed.sequence.to_numpy(dtype='<i8')[mask].tobytes()).hexdigest()
                        result = {}
                        for name, probability in probabilities.items():
                            if name.startswith('H') and subset.startswith('R'): continue
                            supported = h_supported[codes, int(name[-1])] if name[0] == 'H' else feature_support(encoded[:, :13 + int(name[-1])], feature_count)[codes]
                            result[name], calibration = score(probability[mask], y[mask], observed.agent_id.to_numpy()[mask], supported[mask])
                            rows.append({**tag(meta), 'target': target, 'model': name, 'subset': subset, **result[name], 'same_decisions_sha256': rowset,
                                'exact_cell_qualified': bool(assessed['eligible'] if name[-1] != '2' else assessed['memory_eligible']),
                                'feature_qualified': bool(assessed['R1_eligible'] if name[-1] != '2' else assessed['R2_eligible']),
                                'H01_common_coverage': assessed['M01_common_coverage'], 'H012_common_coverage': assessed['M012_common_coverage'],
                                'training_same_policy': True, 'evaluation_seed_previously_untouched': True})
                            bins.extend({**tag(meta), 'target': target, 'model': name, 'subset': subset, **b} for b in calibration)
                        for a, b in [('H1', 'H0'), ('H2', 'H1'), ('R1', 'R0'), ('R2', 'R1')]:
                            if a not in result or b not in result: continue
                            subset_ok = min(int(y[mask].sum()), int(mask.sum() - y[mask].sum())) >= 100
                            permitted = qualification[a] and subset_ok
                            for metric in ('log_loss', 'brier', 'equal_agent_log_loss', 'equal_agent_brier'):
                                aa, bb = result[a][metric], result[b][metric]
                                raw = aa - bb if aa is not None and bb is not None else None
                                paired.append({**tag(meta), 'target': target, 'subset': subset, 'contrast': a + '-' + b, 'metric': metric,
                                    'raw_difference': raw, 'qualified_difference': raw if permitted else None, 'qualified': bool(permitted),
                                    'interpretation': 'model-dependent generalization' if a[0] == 'R' else 'hierarchical exact-cell qualification',
                                    'same_decisions_sha256': rowset, 'decisions': int(mask.sum()), 'positive': int(y[mask].sum()), 'negative': int(mask.sum() - y[mask].sum())})
                    print('SCORED', ecology, mode, meta['seed'], target, qualification, flush=True)
    out = output / 'evaluation'; f, p = pd.DataFrame(rows), pd.DataFrame(paired)
    save_csv(out / 'support.csv', pd.DataFrame(support_rows)); save_csv(out / 'scores.csv', f); save_csv(out / 'paired_by_seed.csv', p); save_csv(out / 'calibration.csv', pd.DataFrame(bins))
    save_csv(out / 'scores_summary.csv', summarize(f, ['ecology', 'mode', 'target', 'model', 'subset'], ['log_loss', 'brier', 'equal_agent_log_loss', 'equal_agent_brier', 'prevalence', 'coverage']))
    save_csv(out / 'paired_summary.csv', summarize(p, ['ecology', 'mode', 'target', 'subset', 'contrast', 'metric'], ['raw_difference', 'qualified_difference']))
    save_csv(out / 'diagnostics.csv', pd.DataFrame(diagnostics)); save_csv(out / 'policy_probes.csv', pd.concat(probes, ignore_index=True)); save_csv(out / 'vocal_distribution.csv', pd.concat(vocal, ignore_index=True))
    save_json(out / 'receipt.json', {'version': '0.8.2', 'inputs': inputs, 'source_sha256': scientific_sources(), 'wall_seconds': perf_counter() - start,
        'peak_memory_bytes': peak(), 'new_seed_evaluation': True, 'preregistration_sha256': digest(root / 'preregistration.json'),
        'replication': 'Equal-seed mean/sampleSD,min,max,n; three independent ecological seeds per condition. No decision-level significance.',
        'stopping': 'Final fixed resource-prediction methodology evaluation; no retuning or extra seeds.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase', choices=('pilot_review', 'training_review', 'evaluate'), required=True)
    parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--output', type=Path, default=OUTPUT)
    args = vars(parser.parse_args()); phase = args.pop('phase')
    try: globals()[phase](**args) if phase != 'pilot_review' else review_pilot(**args)
    except (ValueError, OSError, KeyError) as error: parser.error(str(error))


if __name__ == '__main__': main()
