"""Optional streaming privileged measurements; no memory writes and no RNG."""
from collections import Counter, OrderedDict
from contextlib import ExitStack
import csv
from dataclasses import asdict, dataclass, fields
import gzip
import io
import json
from pathlib import Path
import platform

from simulation.vocal_learning_observer import hash_file

SCHEMA_VERSION = 1
CONTACT_KINDS = ('first_encounter', 'memory_miss_after_eviction', 'continuous_contact', 'return_contact', 'reunion')
DEFINITIONS = {
    'recognition': 'Living others within current genome visual Chebyshev radius, inclusive; relative coordinates, IDs sorted. Perfect imposed visual identity, never auditory identity.',
    'timing': 'One observation per existing actor at its ordinary sequential pre-action decision point. Earlier movement/deaths apply. Newborns join after all actions and first observe next tick.',
    'memory': 'Bounded private LRU. Snapshot-known hits first, then misses; each phase ascending ID. Reads do not refresh. Contact count counts sightings, not contact episodes.',
    'first_encounter': 'Unknown in current bounded agent memory, not necessarily lifetime-first. Evicted identities are unknown to the agent.',
    'memory_miss_after_eviction': 'Unknown contact identifiable in the privileged bounded recent-eviction cache. Offline complete history refines all such reappearances.',
    'memory_allocation': 'Each first_encounter or memory_miss_after_eviction row allocates exactly one entry; no redundant allocation event.',
    'memory_hit': 'Each continuous_contact, return_contact or reunion row is one hit; no redundant hit event.',
    'continuous_contact': 'Seen at immediately preceding observation opportunity, never a reunion.',
    'return_contact': 'Remembered, missed at an intervening observation opportunity, elapsed below threshold.',
    'reunion': 'Remembered, missed at an intervening observation opportunity and current_tick-last_seen_tick >= threshold (inclusive elapsed-sighting gap).',
    'knowledge': 'last_confirmed_alive_tick is last direct sighting. actual_death_tick appears only in separate privileged death rows. Absence never implies death; remote death never deletes memory.',
    'observation': 'Post-update occupancy; visible and known/unknown counts classify membership before the batch. Following eviction/contact rows reconstruct the update.',
    'eviction': 'One old entry removed before an unknown allocation; trigger_id identifies the currently visible incoming identity.',
    'tick_end': 'Completed-tick population and cumulative births/deaths; tick 0 is initial state. Includes newborns, excludes deaths.',
    'observer_memory': 'Recent evicted IDs only, at most social capacity per living observer, released on observer death or finish. No Human, Brain or lifetime history retained.',
}


@dataclass(frozen=True, slots=True)
class SocialRecord:
    sequence: int
    tick: int
    event_kind: str
    observer_id: int | None = None
    observed_id: int | None = None
    observer_age: int | None = None
    observer_generation: int | None = None
    memory_occupancy: int | None = None
    perception_radius: int | None = None
    dx: int | None = None
    dy: int | None = None
    visible_count: int | None = None
    known_visible: int | None = None
    unknown_visible: int | None = None
    elapsed_since_last_seen: int | None = None
    first_seen_tick: int | None = None
    last_confirmed_alive_tick: int | None = None
    encounter_count: int | None = None
    trigger_id: int | None = None
    population: int | None = None
    births: int | None = None
    deaths: int | None = None
    actual_death_tick: int | None = None
    death_cause: str | None = None


COLUMNS = tuple(f.name for f in fields(SocialRecord))


def config_payload(config, social_config):
    return {'simulation': asdict(config), 'social': asdict(social_config)}


def config_hash(payload):
    import hashlib
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class SocialObserver:
    def __init__(self, social_config, sink):
        self.config, self.sink = social_config, sink
        self.counts = Counter()
        self.sequence = 0
        self.finished = False
        self._evicted = {}  # Bounded per living observer, diagnostic only.

    def _write(self, tick, event_kind, **fields):
        if self.finished:
            raise RuntimeError('Social recording already finished')
        self.sink(SocialRecord(self.sequence, tick, event_kind, **fields))
        self.sequence += 1
        self.counts[event_kind] += 1

    def life(self, tick, human, event_kind, cause=None):
        self._write(tick, event_kind, observer_id=human.id, observer_age=human.age,
                    observer_generation=human.generation, memory_occupancy=len(human.social_memory),
                    actual_death_tick=tick if event_kind == 'death' else None, death_cause=cause)
        if event_kind == 'death':
            self._evicted.pop(human.id, None)

    def observation(self, tick, human, update, observation):
        base = dict(observer_id=human.id, observer_age=human.age, observer_generation=human.generation)
        offsets = {i.individual_id: (i.dx, i.dy) for i in observation.individuals}
        self._write(tick, 'observation', **base, memory_occupancy=update.occupancy,
                    perception_radius=human.genome.perception_radius, visible_count=len(update.contacts), known_visible=update.known_visible, unknown_visible=update.unknown_visible)
        for contact in update.contacts:
            identity = contact.entry.individual_id
            kind = contact.kind
            cache = self._evicted.get(human.id)
            if kind == 'first_encounter' and cache is not None and identity in cache:
                kind = 'memory_miss_after_eviction'
                del cache[identity]
            if contact.evicted is not None:
                old = contact.evicted
                self._write(tick, 'eviction', **base, observed_id=old.individual_id, trigger_id=identity,
                            memory_occupancy=contact.occupancy-1, first_seen_tick=old.first_seen_tick,
                            last_confirmed_alive_tick=old.last_seen_tick, encounter_count=old.encounter_count)
                cache = self._evicted.setdefault(human.id, OrderedDict())
                cache[old.individual_id] = None
                cache.move_to_end(old.individual_id)
                if len(cache) > self.config.social_memory_capacity:
                    cache.popitem(last=False)
            entry = contact.entry
            self._write(tick, kind, **base, observed_id=identity, dx=offsets[identity][0], dy=offsets[identity][1], memory_occupancy=contact.occupancy,
                        elapsed_since_last_seen=contact.elapsed_since_last_seen,
                        first_seen_tick=entry.first_seen_tick, last_confirmed_alive_tick=entry.last_seen_tick,
                        encounter_count=entry.encounter_count)

    def tick_end(self, statistics):
        self._write(statistics.tick, 'tick_end', population=statistics.population,
                    births=statistics.births, deaths=statistics.deaths)

    def finish(self, statistics):
        self._write(statistics.tick, 'run_end', population=statistics.population,
                    births=statistics.births, deaths=statistics.deaths)
        self._evicted.clear()
        self.finished = True


class CSVSocialWriter:
    """Caller owns stream; open() creates exclusive deterministic gzip/plain CSV."""
    def __init__(self, stream):
        self.writer = csv.writer(stream)
        self.writer.writerow(COLUMNS)
        self.records = 0

    def __call__(self, record):
        self.writer.writerow(tuple(getattr(record, name) for name in COLUMNS))
        self.records += 1

    @classmethod
    def open(cls, stack: ExitStack, path):
        path = Path(path)
        if path.exists() or Path(str(path)+'.metadata.json').exists():
            raise ValueError(f'Refusing existing social output: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.gz':
            raw = stack.enter_context(path.open('xb'))
            zipped = stack.enter_context(gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=1))
            stream = stack.enter_context(io.TextIOWrapper(zipped, encoding='utf-8', newline=''))
        else:
            stream = stack.enter_context(path.open('x', encoding='utf-8', newline=''))
        return cls(stream)


def write_metadata(path, sim, writer):
    """After close/flush only; certify a completed stream, never overwrite a sidecar."""
    path = Path(path)
    observer = sim.social_observer
    if observer is None or not observer.finished or writer.records != observer.sequence:
        raise ValueError('Social recording must be finished before metadata')
    payload = config_payload(sim.config, sim.social_config)
    root = Path(__file__).resolve().parents[1]
    sources = [root/'main.py', root/'config.py']
    for directory in ('agents', 'simulation', 'systems', 'world'):
        sources.extend(sorted((root/directory).glob('*.py')))
    meta = {'version': '0.7.0', 'schema_version': SCHEMA_VERSION, 'completed': True, 'recording_enabled': True,
            'seed': sim.config.random_seed, 'config': payload, 'config_id': config_hash(payload),
            'columns': COLUMNS, 'final_tick': sim.tick, 'statistics': asdict(sim.statistics),
            'records': writer.records, 'event_counts': dict(sorted(observer.counts.items())),
            'file': {'sha256': hash_file(path), 'bytes': path.stat().st_size}, 'definitions': DEFINITIONS,
            'memory_capacity': sim.social_config.social_memory_capacity,
            'reunion_threshold': sim.social_config.social_reunion_min_absence_ticks,
            'source_sha256': {str(p.relative_to(root)): hash_file(p) for p in sources},
            'python': platform.python_version()}
    with Path(str(path)+'.metadata.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(meta, indent=2, allow_nan=False)+'\n')
    return meta
