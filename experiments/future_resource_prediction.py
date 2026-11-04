"""Two frozen retrospective target comparisons, never an ecological experiment."""
from pathlib import Path
from time import perf_counter
import argparse
import json
from hashlib import sha256

import numpy as np
import pandas as pd

from experiments.future_resource_io import (ROOT, OUTPUT, ARCHIVE, case_paths, digest, receipt, save_csv, save_json, sources, training_paths)
from experiments.future_resource_analysis import load_labels, labels_for, tags
from experiments.future_resource_targets import definitions
from experiments.nonstationary_prediction import Hierarchy
from experiments.resource_ecology_experiments import peak

HEADS = {'M0': 'X_A', 'M1': 'X_B_A', 'M2': 'X_C_A'}


def split_seeds(training, evaluation):
    if set(training) & set(evaluation):
        raise ValueError('Training and evaluation ecological seeds overlap')


def keys(x, action, visible, remembered):
    return {'X_A': (x, action), 'X_B_A': (x, action, visible), 'X_C_A': (x, action, visible, remembered)}


class Predictor:
    """Unchanged v074 Hierarchy and an independent X-only reference."""
    def __init__(self, environmental=False):
        self.main = Hierarchy(tuple(HEADS.values()), 2, minimum=20, strength=5)
        self.x_only = Hierarchy(('X_A',), 2, minimum=20, strength=5) if environmental else None

    def add(self, frame, labels):
        columns = ['x_key', 'action', 'b_token', 'rd_token']
        for (x, action, visible, remembered), label in zip(frame[columns].itertuples(index=False, name=None), labels):
            if label < 0:
                continue
            self.main.add(keys(x, action, visible, remembered), action, int(label))
            if self.x_only:
                self.x_only.add({'X_A': (x,)}, -1, int(label))

    def freeze(self):
        self.main.freeze()
        if self.x_only:
            self.x_only.freeze()

    def predict(self, frame):
        names = [*HEADS] + (['X_only'] if self.x_only else [])
        probabilities = {name: np.zeros(len(frame)) for name in names}
        supported = {name: np.zeros(len(frame), dtype=bool) for name in names}
        columns = ['x_key', 'action', 'b_token', 'rd_token']
        for i, (x, action, visible, remembered) in enumerate(frame[columns].itertuples(index=False, name=None)):
            result, support = self.main.predict(keys(x, action, visible, remembered), action)
            for name, head in HEADS.items():
                probabilities[name][i] = result[head][1]
                supported[name][i] = support[head]
            if self.x_only:
                result, support = self.x_only.predict({'X_A': (x,)}, -1)
                probabilities['X_only'][i] = result['X_A'][1]
                supported['X_only'][i] = support['X_A']
        return probabilities, supported


def score(probability, labels, agents, supported):
    n = len(labels)
    if not n:
        return {'decisions': 0, 'agents': 0, 'log_loss': None, 'brier': None, 'equal_agent_log_loss': None,
                'equal_agent_brier': None, 'prevalence': None, 'coverage': None}, []
    if (labels < 0).any() or not ((probability > 0) & (probability < 1)).all():
        raise ValueError('Observed binary labels and smoothed probabilities required')
    loss = -(labels * np.log(probability) + (1 - labels) * np.log1p(-probability))
    brier = (probability - labels) ** 2
    values = pd.DataFrame({'agent': agents, 'loss': loss, 'brier': brier})
    equal = values.groupby('agent')[['loss', 'brier']].mean().mean()
    bins = []
    indices = np.minimum(9, (10 * probability).astype(int))
    for index in sorted(np.unique(indices)):
        mask = indices == index
        bins.append({'bin': int(index), 'n': int(mask.sum()), 'mean_probability': float(probability[mask].mean()),
                     'positive_fraction': float(labels[mask].mean())})
    return {'decisions': n, 'agents': values.agent.nunique(), 'log_loss': float(loss.mean()), 'brier': float(brier.mean()),
        'equal_agent_log_loss': float(equal['loss']), 'equal_agent_brier': float(equal.brier),
        'prevalence': float(labels.mean()), 'coverage': float(supported.mean())}, bins


def summarize(frame, keys, metrics):
    measure = 'measure' if 'metric' in keys else 'metric'
    tidy = frame.melt(id_vars=[*keys, 'seed'], value_vars=metrics, var_name=measure, value_name='value')
    if tidy.duplicated([*keys, 'seed', measure]).any():
        raise ValueError('More than one contribution per ecological seed')
    return tidy.groupby([*keys, measure], dropna=False).value.agg(['mean', 'std', 'min', 'max', 'count']).reset_index().rename(columns={'count': 'n'})


def run(archive=ARCHIVE, root=ROOT, output=OUTPUT):
    root, output = Path(root), Path(output)
    plan = receipt(root / 'frozen_evaluation.json', root)
    split_seeds(plan['train_seeds'], plan['evaluation_seeds'])
    out = output / 'prediction'
    if out.exists():
        raise ValueError('Refusing prediction overwrite')
    start = perf_counter()
    catalog = {t.name: t for t in definitions()}
    selected = [catalog[plan['retrospective_environment']], catalog[plan['retrospective_behavior']]]
    feasibility = pd.read_csv(output / 'feasibility/target_support.csv').set_index(['ecology', 'mode', 'seed', 'target'])
    results, calibration, paired = [], [], []
    inputs = {str(root / 'frozen_evaluation.json'): digest(root / 'frozen_evaluation.json')}
    for ecology in ('E0', 'E1', 'E2', 'E3'):
        predictors = {target.name: Predictor(target.kind == 'environmental') for target in selected}
        for path in training_paths(archive, ecology):
            meta, files, frame, grids, events = load_labels(path)
            inputs[str(path)] = digest(path)
            for target in selected:
                labels, reasons = labels_for(frame, grids, events, target, meta['tick'])
                predictors[target.name].add(frame, labels)
        for predictor in predictors.values():
            predictor.freeze()
        for path in [p for p in case_paths(archive) if p.name.startswith(ecology + '_')]:
            meta, files, frame, grids, events = load_labels(path)
            inputs[str(path)] = digest(path)
            for target in selected:
                labels, reasons = labels_for(frame, grids, events, target, meta['tick'])
                observed = labels >= 0
                f, values = frame.loc[observed], labels[observed]
                probabilities, supported = predictors[target.name].predict(f)
                common01 = supported['M0'] & supported['M1']
                common012 = common01 & supported['M2']
                eligible = feasibility.loc[(ecology, meta['mode'], meta['seed'], target.name)]
                if len(f) != eligible.observed or int(common01.sum()) != eligible.M01_common_n or int(common012.sum()) != eligible.M012_common_n:
                    raise ValueError('Prediction observations/support differ from frozen feasibility')
                tag = {**tags(meta), 'target': target.name, 'kind': target.kind, 'retrospective': True,
                       'policy_transfer': 'same A policy' if meta['mode'] == 'A' else 'A-trained, cross-policy ' + meta['mode']}
                for subset, mask in [('all', np.ones(len(f), dtype=bool)), ('common01', common01), ('common012', common012)]:
                    rowset = sha256(f.sequence.to_numpy(dtype='<i8')[mask].tobytes()).hexdigest()
                    scores = {}
                    for name, probability in probabilities.items():
                        scores[name], bins = score(probability[mask], values[mask], f.agent_id.to_numpy()[mask], supported[name][mask])
                        results.append({**tag, 'model': name, 'subset': subset, **scores[name], 'same_decisions_sha256': rowset,
                            'target_eligible': bool(eligible.eligible), 'memory_eligible': bool(eligible.memory_eligible), 'eligible_labels': len(f),
                            'source_decisions': len(frame), 'M01_common_coverage': float(common01.mean()) if len(f) else 0.,
                            'M012_common_coverage': float(common012.mean()) if len(f) else 0.})
                        calibration.extend({**tag, 'model': name, 'subset': subset, **b} for b in bins)
                    for a, b in [('M1', 'M0'), ('M2', 'M1')]:
                        permitted = bool(eligible.memory_eligible if a == 'M2' else eligible.eligible)
                        for metric in ('log_loss', 'brier', 'equal_agent_log_loss', 'equal_agent_brier'):
                            first, second = scores[a][metric], scores[b][metric]
                            raw = first - second if first is not None and second is not None else None
                            paired.append({**tag, 'subset': subset, 'contrast': a + '-' + b, 'metric': metric,
                                'raw_difference': raw, 'eligible_difference': raw if permitted else None,
                                'support_eligible': permitted, 'paired_decisions': scores[a]['decisions'], 'same_decisions_sha256': rowset,
                                'NA_reason': '' if permitted else 'Frozen target/common-support/feature-variation criteria failed; not zero effect'})
                print('Retrospective', ecology, meta['mode'], meta['seed'], target.name, 'observed=', len(f), flush=True)
    f, p = pd.DataFrame(results), pd.DataFrame(paired)
    save_csv(out / 'model_comparison.csv', f)
    save_csv(out / 'paired_by_seed.csv', p)
    save_csv(out / 'calibration.csv', pd.DataFrame(calibration))
    save_csv(out / 'common_support.csv', f[['ecology', 'mode', 'seed', 'target', 'kind', 'model', 'subset', 'decisions', 'eligible_labels', 'source_decisions', 'coverage', 'M01_common_coverage', 'M012_common_coverage', 'target_eligible', 'memory_eligible']])
    save_csv(out / 'summary.csv', summarize(f, ['ecology', 'mode', 'target', 'kind', 'model', 'subset'], ['log_loss', 'brier', 'equal_agent_log_loss', 'equal_agent_brier', 'prevalence']))
    save_csv(out / 'paired_summary.csv', summarize(p, ['ecology', 'mode', 'target', 'kind', 'subset', 'contrast', 'metric'], ['raw_difference', 'eligible_difference']))
    save_json(out / 'receipt.json', {'version': '0.8.1', 'stage': 'retrospective exploratory, not untouched holdout confirmation',
        'inputs': inputs, 'source_sha256': sources(), 'wall_seconds': perf_counter() - start, 'peak_memory_bytes': peak(),
        'targets': [t.name for t in selected], 'no_simulations': True,
        'limits': 'Exact same decisions within target/case/subset. Binary Brier=(p-y)^2. Positive-probability calibration. Deaths competing, never padded as negatives. NA eligible effects when frozen support fails; raw backoff score differences are diagnostic only.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    try:
        run(**vars(parser.parse_args()))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
