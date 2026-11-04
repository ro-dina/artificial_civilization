"""Compact exact event/selected-target joins; references, not copied features."""
from pathlib import Path
from time import perf_counter
import argparse

import numpy as np
import pandas as pd

from experiments.future_resource_io import ROOT, OUTPUT, ARCHIVE, case_paths, digest, receipt, save_csv, save_json, sources
from experiments.future_resource_analysis import load_labels, labels_for, tags
from experiments.future_resource_targets import definitions, REASONS
from experiments.resource_ecology_experiments import peak


def next_sequences(frame, elapsed, event, resource):
    result = np.full(len(frame), -1, dtype=np.int64)
    for indices in frame.groupby('agent_id', sort=False).indices.values():
        observed = event[indices] == 1
        origins = indices[observed]
        offsets = np.arange(len(indices))[observed] + elapsed[origins]
        targets = indices[offsets]
        if not np.array_equal(frame.own_index.to_numpy()[targets], frame.own_index.to_numpy()[origins] + elapsed[origins]):
            raise ValueError('Invalid future acquisition own-index join')
        if not (frame.tick.to_numpy()[targets] > frame.tick.to_numpy()[origins]).all():
            raise ValueError('Future event must exclude origin')
        units = frame.acquired if resource == 'either' else frame[resource + '_consumed']
        if not (units.to_numpy()[targets] > 0).all():
            raise ValueError('Linked future event has no actual acquisition')
        result[origins] = frame.sequence.to_numpy()[targets]
    return result


def run(archive=ARCHIVE, root=ROOT, output=OUTPUT):
    root, output = Path(root), Path(output)
    plan = receipt(root / 'frozen_evaluation.json', root)
    folder = root / 'labels'
    if folder.exists():
        raise ValueError('Refusing derived-label overwrite')
    catalog = {t.name: t for t in definitions()}
    chosen = {'environment': catalog[plan['retrospective_environment']], 'behavior': catalog[plan['retrospective_behavior']]}
    support = pd.read_csv(output / 'feasibility/target_support.csv').set_index(['ecology', 'mode', 'seed', 'target'])
    start = perf_counter(); inputs, files = {}, {}
    for path in case_paths(archive):
        meta, source, frame, grids, events = load_labels(path)
        inputs[str(path)] = digest(path)
        derived = pd.DataFrame({'sequence': frame.sequence})
        for resource, (elapsed, event) in events.items():
            derived[resource + '_elapsed'] = elapsed
            derived[resource + '_event'] = event
            linked = next_sequences(frame, elapsed, event, resource)
            derived[resource + '_next_sequence'] = pd.Series(linked).where(linked >= 0).astype('Int64')
        for name, target in chosen.items():
            labels, reasons = labels_for(frame, grids, events, target, meta['tick'])
            derived[name + '_label'] = pd.Series(labels).where(labels >= 0).astype('Int8')
            derived[name + '_reason'] = reasons
            expected = support.loc[(meta['ecology'], meta['mode'], meta['seed'], target.name)]
            if int((labels >= 0).sum()) != expected.observed or int(labels[labels >= 0].sum()) != expected.positive:
                raise ValueError('Derived labels differ from feasibility')
        filename = folder / f'{meta["ecology"]}_{meta["mode"]}_seed_{meta["seed"]}.csv.gz'
        save_csv(filename, derived)
        files[str(filename)] = {**tags(meta), 'records': len(derived), 'bytes': filename.stat().st_size, 'sha256': digest(filename),
                                'source_metadata': str(path), 'source_acquisitions_sha256': digest(source['acquisitions'])}
    save_json(folder / 'receipt.json', {'version': '0.8.1', 'schema': 1, 'inputs': inputs, 'source_sha256': sources(),
        'files': files, 'wall_seconds': perf_counter() - start, 'peak_memory_bytes': peak(),
        'selected_targets': {name: target.name for name, target in chosen.items()}, 'reason_codes': REASONS,
        'event_codes': {'1': 'actual acquisition', '2': 'competing death', '3': 'administrative censor', '4': 'missing-history censor'},
        'join': 'case + sequence links immutable acquisitions (agent_id,tick,own_index,X,social tokens). *_next_sequence links actual next consumption only for event1; otherwise empty. *_elapsed is own decisions to event/censor, not fabricated event time.',
        'no_simulations': True, 'no_features_duplicated': True, 'all_labels_match_frozen_feasibility': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE); parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--output', type=Path, default=OUTPUT)
    try:
        run(**vars(parser.parse_args()))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
