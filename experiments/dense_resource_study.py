"""Frozen bounded v082 study: validation, pilot, training, then gated new seeds."""
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from random import Random
from time import perf_counter
import argparse
import hashlib
import json
from datetime import datetime, timezone

from config import Config, SocialConfig, SocialPolicyConfig, SocialLearningConfig
from simulation.simulation import Simulation
from simulation.resource_observer import ResourceObserver, ResourceWriter, paths, SCHEMAS
from simulation.dense_resources import DenseResourceObserver, DenseResourceWriter, file_hash
from world.resource_ecology import PatchGeometry
from experiments.resource_ecology_experiments import ecology_config, fingerprint, initial_agents, peak
from experiments.social_learning_experiments import fingerprints
from experiments.social_verification import digest as state_digest
from experiments.future_resource_io import digest, save_json
from experiments.dense_resource_data import ROOT, OUTPUT, DESIGN, plan, case_path, scientific_sources


def settings(mode):
    return {'social_config': SocialConfig(True, 32, 20), 'social_policy': SocialPolicyConfig(),
            'social_learning': SocialLearningConfig({'A': 'none', 'B': 'perception', 'C': 'memory'}[mode])}


def preregister(root=ROOT):
    root = Path(root)
    save_json(root / 'preregistration.json', {'version': '0.8.2', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'chronology': 'Frozen before instrumentation validation/new training/evaluation ecological runs; targets follow v081 retrospective evidence.',
        'design': json.loads(DESIGN.read_text()), 'design_sha256': digest(DESIGN), 'source_sha256': scientific_sources()})


def run_one(root, phase, ecology, mode, seed, ticks, dense_enabled=True):
    root = Path(root); rules = plan(root)
    cfg = Config.evolution(brain='learning', random_seed=seed, max_ticks=ticks)
    eco = ecology_config(ecology)
    geometry = PatchGeometry.generate(cfg, eco)
    meta_path = case_path(root, phase, ecology, mode, seed)
    csv_path = meta_path.with_name(meta_path.name[:-len('.metadata.json')])
    resource_path = csv_path.with_name(csv_path.name[:-7] + '.resources.u8.gz')
    start = perf_counter(); checkpoints = {}; action_checkpoints = {0: hashlib.sha256().hexdigest()}
    with ExitStack() as stack:
        writer = ResourceWriter.open(stack, csv_path)
        dense = stack.enter_context(DenseResourceWriter(resource_path, cfg)) if dense_enabled else None
        observer = DenseResourceObserver(writer, mode, ecology, geometry, dense) if dense else ResourceObserver(writer, mode, ecology, geometry)
        sim = Simulation(cfg, **settings(mode), resource_ecology=eco, resource_observer=observer, social_learning_observer=observer)
        initial = fingerprints(sim, True); initial_agent = initial_agents(sim); checkpoints[0] = fingerprint(sim)
        for _ in range(ticks):
            if not sim.population:
                break
            sim.step()
            if sim.tick % 100 == 0:
                checkpoints[sim.tick] = fingerprint(sim); action_checkpoints[sim.tick] = observer.action_digest.hexdigest()
                print(f'{phase} {ecology}/{mode} seed={seed} tick={sim.tick} population={sim.population}', flush=True)
        checkpoints[sim.tick] = fingerprint(sim)
        final = state_digest(sim)
        observer.finish(sim)
    proof = {'initial_ecology': initial, 'checkpoints': checkpoints, 'action_checkpoints': action_checkpoints, 'ecological_final': final}
    legacy = None
    if ecology == 'E0' and seed == 0 and ticks == 100:
        old = Path('data/resource_ecology_v080/pilot') / f'E0_{mode}_seed_0.csv.gz.metadata.json'
        previous = json.loads(old.read_text())
        if json.loads(json.dumps(proof)) != previous['fingerprints'] or asdict(sim.statistics) != previous['statistics'] or observer.action_digest.hexdigest() != previous['action_sha256']:
            raise ValueError('Complete archived E0 regression failure')
        legacy = {'file': str(old), 'sha256': digest(old), 'complete_state_rng_actions_statistics_exact': True}
    if observer.resource_totals['food'] != sim.statistics.food_consumed or observer.resource_totals['water'] != sim.statistics.water_consumed:
        raise ValueError('Consumption diagnostic mismatch')
    if writer.counts['acquisitions'] != observer.total['decisions'] or writer.counts['individuals'] != cfg.initial_population + sim.statistics.births:
        raise ValueError('Decision/life diagnostic incompleteness')
    files = paths(csv_path)
    payload = {'version': '0.8.2', 'completed': True, 'schema_version': 2, 'ecology': ecology, 'mode': mode, 'seed': seed, 'phase': phase, 'tick': sim.tick,
        'config': {'simulation': asdict(cfg), 'resource_ecology': asdict(eco), 'social': asdict(sim.social_config), 'social_policy': asdict(sim.social_policy), 'social_learning': asdict(sim.social_learning)},
        'geometry': asdict(geometry), 'statistics': asdict(sim.statistics), 'totals': dict(observer.total), 'resource_totals': dict(observer.resource_totals),
        'fingerprints': proof, 'initial_agents_rng_sha256': initial_agent, 'action_sha256': observer.action_digest.hexdigest(),
        'legacy_comparison': legacy, 'schemas': SCHEMAS,
        'files': {k: {'file': p.name, 'sha256': digest(p), 'bytes': p.stat().st_size, 'records': writer.counts[k]} for k, p in files.items()},
        'dense': json.loads(Path(str(resource_path) + '.json').read_text()) if dense else None,
        'runtime': {'wall_seconds': perf_counter() - start, 'peak_memory_bytes': peak()},
        'preregistration_sha256': digest(root / 'preregistration.json'), 'source_sha256': scientific_sources()}
    save_json(meta_path, payload)
    return meta_path


def validate(root=ROOT):
    root = Path(root); rules = plan(root); start = perf_counter(); records = []; inputs = {}
    for ecology, mode, seed, ticks in rules['instrumentation']['validation']:
        a = run_one(root, 'validation_off', ecology, mode, seed, ticks, False)
        b = run_one(root, 'validation_on', ecology, mode, seed, ticks, True)
        aa, bb = json.loads(a.read_text()), json.loads(b.read_text())
        if any(aa[k] != bb[k] for k in ('fingerprints', 'action_sha256', 'statistics')):
            raise ValueError('Dense instrumentation changed whole state/RNG/actions')
        if any(aa['files'][k]['sha256'] != bb['files'][k]['sha256'] for k in aa['files']):
            raise ValueError('Dense instrumentation altered old diagnostic streams')
        # Repeated dense file is also required byte-identical.
        c = run_one(root, 'validation_repeat', ecology, mode, seed, ticks, True)
        cc = json.loads(c.read_text())
        if bb['dense']['sha256'] != cc['dense']['sha256'] or bb['fingerprints'] != cc['fingerprints']:
            raise ValueError('Dense deterministic replay failed')
        records.append({'ecology': ecology, 'mode': mode, 'ticks': ticks, 'on_off_exact': True, 'repeated_binary_exact': True})
        inputs.update({str(p): digest(p) for p in (a, b, c)})
    for mode in rules['instrumentation']['archived_e0']:
        p = run_one(root, 'legacy', 'E0', mode, 0, 100)
        records.append({'ecology': 'E0', 'mode': mode, 'ticks': 100, 'archived_complete_exact': True, 'metadata': str(p)})
        inputs[str(p)] = digest(p)
    save_json(root / 'validation.json', {'passed': True, 'cases': records, 'wall_seconds': perf_counter() - start,
        'preregistration_sha256': digest(root / 'preregistration.json'), 'source_sha256': scientific_sources(), 'inputs': inputs})


def allowed(root, phase):
    root = Path(root); plan(root)
    review = root / ('validation.json' if phase == 'pilot' else 'pilot_review.json' if phase == 'training' else 'training_review.json')
    value = json.loads(review.read_text())
    if value['preregistration_sha256'] != digest(root / 'preregistration.json'):
        raise ValueError('Gate design changed')
    for p, sha in value['inputs'].items():
        if digest(p) != sha:
            raise ValueError('Gate input changed')
    return value['passed'] if phase == 'pilot' else value['allowed']


def run_phase(root, phase, seeds):
    rules = plan(root)
    expected = rules['pilot']['seeds'] if phase == 'pilot' else rules['train_seeds'] if phase == 'training' else rules['evaluation_seeds']
    if len(seeds) != len(set(seeds)) or any(s not in expected for s in seeds) or not allowed(root, phase):
        raise ValueError('Frozen seed set or stage gate violation')
    ticks = 100 if phase == 'pilot' else 500
    for seed in sorted(seeds):
        for ecology in ('E0', 'E1', 'E2', 'E3'):
            for mode in 'ABC':
                run_one(root, phase, ecology, mode, seed, ticks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--phase', choices=('preregister', 'validate', 'pilot', 'training', 'evaluation'), required=True)
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    try:
        if args.phase == 'preregister': preregister(args.root)
        elif args.phase == 'validate': validate(args.root)
        else: run_phase(args.root, args.phase, args.seeds or ([10] if args.phase == 'pilot' else [10, 11, 12] if args.phase == 'training' else [20, 21, 22]))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__': main()
