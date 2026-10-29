"""Strict streaming validation of completed v0.7.0 social recordings."""
from collections import Counter, OrderedDict
from dataclasses import asdict, fields
import csv
import gzip
import json
from pathlib import Path
import re

from agents.social_memory import SocialMemory
from config import Config, SocialConfig, GENOME_BOUNDS
from simulation.social_observer import COLUMNS, CONTACT_KINDS, SocialRecord, config_hash
from simulation.vocal_learning_observer import hash_file

STRING_FIELDS = {'event_kind', 'death_cause'}
KINDS = {*CONTACT_KINDS, 'founder', 'birth', 'death', 'observation', 'eviction', 'tick_end', 'run_end'}


def discover(directory, pattern='*.csv*'):
    runs = []
    for path in sorted(Path(directory).glob(pattern)):
        if not path.is_file() or not (path.name.endswith('.csv') or path.name.endswith('.csv.gz')):
            continue
        sidecar = Path(str(path)+'.metadata.json')
        try:
            meta = json.loads(sidecar.read_text())
            if (meta['version'] != '0.7.0' or meta['schema_version'] != 1 or meta['completed'] is not True
                    or meta['recording_enabled'] is not True or tuple(meta['columns']) != COLUMNS):
                raise ValueError('incomplete or unsupported schema')
            digest = hash_file(path)
            payload = meta['config']
            if set(payload) != {'simulation', 'social'}:
                raise ValueError('full simulation/social configuration required')
            if set(payload['simulation']) != {f.name for f in fields(Config)} or set(payload['social']) != {f.name for f in fields(SocialConfig)}:
                raise ValueError('incomplete configuration fields')
            cfg, social = Config(**payload['simulation']), SocialConfig(**payload['social'])
            if (not social.social_memory_enabled or type(meta['seed']) is not int or meta['seed'] != cfg.random_seed
                    or meta['config_id'] != config_hash(payload) or type(meta['final_tick']) is not int
                    or not 0 <= meta['final_tick'] <= cfg.max_ticks or type(meta['records']) is not int or meta['records'] < 2
                    or meta['memory_capacity'] != social.social_memory_capacity
                    or meta['reunion_threshold'] != social.social_reunion_min_absence_ticks
                    or meta['file']['bytes'] != path.stat().st_size or meta['file']['sha256'] != digest
                    or meta['statistics']['tick'] != meta['final_tick']):
                raise ValueError('configuration/hash/completion mismatch')
            if not meta['source_sha256'] or any(not re.fullmatch('[0-9a-f]{64}', h) for h in meta['source_sha256'].values()):
                raise ValueError('source hashes missing or invalid')
            grouping = {'simulation': {k:v for k,v in payload['simulation'].items() if k != 'random_seed'}, 'social':payload['social']}
            condition = f'{cfg.brain}_{config_hash(grouping)[:10]}'
            runs.append({'path':path, 'run':path.name, 'metadata':meta, 'config':cfg, 'social':social,
                         'condition':condition, 'seed':cfg.random_seed,
                         'sha256':digest, 'metadata_sha256':hash_file(sidecar)})
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f'{path}: invalid/missing social metadata: {error}') from error
    if not runs:
        raise ValueError(f'No social CSV recordings found in {directory}')
    keys = [(r['condition'],r['seed']) for r in runs]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate condition/seed recordings; independent replicates must be explicit')
    return sorted(runs, key=lambda r:(r['condition'],r['seed'],r['run']))


def records(run):
    path = run['path']
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f'{path}: social CSV columns do not match schema')
        for index, row in enumerate(reader):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'{path}: malformed social CSV row')
            parsed = {}
            for name, value in row.items():
                if not value:
                    parsed[name] = None
                elif name in STRING_FIELDS:
                    parsed[name] = value
                elif re.fullmatch(r'-?\d+', value):
                    parsed[name] = int(value)
                else:
                    raise ValueError(f'{path}: invalid integer {name}={value!r}')
            record = SocialRecord(**parsed)
            base={'sequence','tick','event_kind'}
            actor={'observer_id','observer_age','observer_generation','memory_occupancy'}
            allowed=(base|actor if record.event_kind in ('founder','birth') else
                base|actor|{'actual_death_tick','death_cause'} if record.event_kind=='death' else
                base|actor|{'perception_radius','visible_count','known_visible','unknown_visible'} if record.event_kind=='observation' else
                base|actor|{'observed_id','dx','dy','elapsed_since_last_seen','first_seen_tick','last_confirmed_alive_tick','encounter_count'} if record.event_kind in CONTACT_KINDS else
                base|actor|{'observed_id','trigger_id','first_seen_tick','last_confirmed_alive_tick','encounter_count'} if record.event_kind=='eviction' else
                base|{'population','births','deaths'})
            if any(value is not None and name not in allowed for name,value in parsed.items()):
                raise ValueError(f'{path}: fields outside event schema')
            if (record.sequence != index or record.event_kind not in KINDS or type(record.tick) is not int
                    or not 0 <= record.tick <= run['metadata']['final_tick']):
                raise ValueError(f'{path}: invalid sequence/kind/tick at row {index}')
            for name in ('observer_id','observed_id','trigger_id','observer_age','observer_generation','memory_occupancy',
                         'perception_radius','visible_count','known_visible','unknown_visible','elapsed_since_last_seen',
                         'first_seen_tick','last_confirmed_alive_tick','encounter_count','population','births','deaths','actual_death_tick'):
                value = getattr(record,name)
                if value is not None and value < 0:
                    raise ValueError(f'{path}: negative {name}')
            yield record


class Validator:
    """Bounded live cognitive replay plus scalar demography; no Human objects.

    A turn is buffered only until its declared number of visible contacts arrives.
    Contact lifetime distinctness is measured by the separate offline aggregator.
    """
    def __init__(self, run):
        self.run = run
        self.social = run['social']
        self.recent_evictions = {}
        self.alive = {}  # id -> [birth tick, generation, bounded mechanical memory]
        self.started = set()
        self.acted = set()
        self.next_id = 0
        self.last_tick_end = -1
        self.last_tick = 0
        self.births = self.deaths = 0
        self.finished = False
        self.turn = None
        self.turn_rows = []
        self.turn_contacts = 0
        self.counts = Counter()
        self.last_population = None

    def error(self, message):
        raise ValueError(f"{self.run['path']}: {message}")

    def _life(self, row):
        identity = row.observer_id
        if type(identity) is not int:
            self.error('Lifecycle event needs valid observer identity')
        if row.event_kind in ('founder','birth'):
            if identity != self.next_id or row.observer_age != 0 or row.memory_occupancy != 0:
                self.error('Invalid/reused lifecycle identity or nonempty initial memory')
            if row.event_kind == 'founder':
                if row.tick != 0 or row.observer_generation != 0 or identity >= self.run['config'].initial_population:
                    self.error('Invalid founder')
            else:
                if row.tick == 0 or row.observer_generation is None or row.observer_generation < 1 or self.acted != self.started:
                    self.error('Birth before existing acting phase completes')
                self.births += 1
            self.next_id += 1
            self.alive[identity] = [row.tick,row.observer_generation,SocialMemory(self.social.social_memory_capacity,
                                                                        self.social.social_reunion_min_absence_ticks)]
        else:
            if (identity not in self.alive or identity not in self.acted or row.actual_death_tick != row.tick
                    or row.death_cause not in ('starvation','dehydration','old_age','energy')):
                self.error('Invalid privileged death event')
            birth,generation,memory = self.alive[identity]
            if (row.observer_age != row.tick-birth or row.observer_generation != generation
                    or row.memory_occupancy != len(memory)):
                self.error('Death age/generation/occupancy mismatch')
            del self.alive[identity]
            self.recent_evictions.pop(identity,None)
            self.deaths += 1

    def _complete_turn(self):
        header = self.turn
        contacts = [r for r in self.turn_rows if r.event_kind in CONTACT_KINDS]
        if len(contacts) != header.visible_count:
            self.error('Missing/duplicate contacts in observation')
        identities = [r.observed_id for r in contacts]
        if (any(type(i) is not int for i in identities) or len(set(identities)) != len(identities)
                or any(i == header.observer_id or i not in self.alive or self.alive[i][0] >= header.tick for i in identities)):
            self.error('Invalid/self/dead/newborn contact identity')
        if (type(header.perception_radius) is not int or not 0 <= header.perception_radius <= GENOME_BOUNDS['perception_radius'][1]
                or any(type(r.dx) is not int or type(r.dy) is not int or max(abs(r.dx),abs(r.dy)) > header.perception_radius for r in contacts)):
            self.error('Contact outside recorded visual radius')
        memory = self.alive[header.observer_id][2]
        update = memory.observe(tuple(sorted(identities)),header.tick,capture=True)
        if ((header.known_visible,header.unknown_visible,header.memory_occupancy) !=
                (update.known_visible,update.unknown_visible,update.occupancy)):
            self.error('Observation membership/occupancy differs from bounded memory replay')
        expected = []
        for contact in update.contacts:
            kind=contact.kind
            cache=self.recent_evictions.get(header.observer_id)
            if kind=='first_encounter' and cache is not None and contact.entry.individual_id in cache:
                kind='memory_miss_after_eviction'
                del cache[contact.entry.individual_id]
            if contact.evicted is not None:
                cache=self.recent_evictions.setdefault(header.observer_id,OrderedDict())
                cache[contact.evicted.individual_id]=None
                cache.move_to_end(contact.evicted.individual_id)
                if len(cache)>self.social.social_memory_capacity:cache.popitem(last=False)
                expected.append(('eviction',contact.evicted,contact.entry.individual_id,contact.occupancy-1,None))
            expected.append((kind,contact.entry,None,contact.occupancy,contact.elapsed_since_last_seen))
        if len(expected) != len(self.turn_rows):
            self.error('Missing/extra eviction/contact event')
        for (kind,entry,trigger,occupancy,elapsed), row in zip(expected,self.turn_rows):
            actual_kind = row.event_kind
            if (actual_kind != kind or row.observed_id != entry.individual_id or row.trigger_id != trigger
                    or row.memory_occupancy != occupancy or row.elapsed_since_last_seen != elapsed
                    or row.first_seen_tick != entry.first_seen_tick or row.last_confirmed_alive_tick != entry.last_seen_tick
                    or row.encounter_count != entry.encounter_count):
                self.error('Contact/eviction differs from bounded memory replay')
        self.turn = None
        self.turn_rows = []

    def accept(self, row):
        if self.finished or row.tick < self.last_tick:
            self.error('Events after finish or unsorted ticks')
        if row.tick > self.last_tick_end+1:
            self.error('Missing completed-tick observation')
        if self.turn is not None and row.event_kind not in (*CONTACT_KINDS,'eviction'):
            self._complete_turn()
        self.last_tick = row.tick
        self.counts[row.event_kind] += 1
        if row.event_kind in ('founder','birth','death'):
            self._life(row)
        elif row.event_kind == 'observation':
            if row.tick != self.last_tick_end+1 or row.observer_id not in self.started or row.observer_id in self.acted:
                self.error('Invalid/duplicate observer opportunity')
            birth,generation,_ = self.alive[row.observer_id]
            if (row.observer_age != row.tick-birth-1 or row.observer_generation != generation
                    or type(row.visible_count) is not int or row.visible_count < 0):
                self.error('Invalid observer age/generation/count')
            self.acted.add(row.observer_id)
            self.turn = row
            self.turn_contacts = 0
            if row.visible_count == 0:
                self._complete_turn()
        elif row.event_kind in (*CONTACT_KINDS,'eviction'):
            if (self.turn is None or row.tick != self.turn.tick or row.observer_id != self.turn.observer_id
                    or row.observer_age != self.turn.observer_age or row.observer_generation != self.turn.observer_generation):
                self.error('Contact outside matching observation')
            self.turn_rows.append(row)
            self.turn_contacts += int(row.event_kind in CONTACT_KINDS)
            if len(self.turn_rows)>2*self.turn.visible_count:self.error('Excessive turn records')
            if self.turn_contacts == self.turn.visible_count:
                self._complete_turn()
        elif row.event_kind == 'tick_end':
            if (row.tick != self.last_tick_end+1 or row.tick == 0 and self.next_id != self.run['config'].initial_population
                    or row.tick > 0 and self.acted != self.started or row.population != len(self.alive)
                    or row.births != self.births or row.deaths != self.deaths
                    or row.population != self.run['config'].initial_population+self.births-self.deaths):
                self.error('Tick completion or population accounting mismatch')
            self.last_tick_end = row.tick
            self.started = set(self.alive)
            self.acted.clear()
            self.last_population = row.population
        elif row.event_kind == 'run_end':
            if (row.tick != self.last_tick_end or (row.population,row.births,row.deaths) != (len(self.alive),self.births,self.deaths)):
                self.error('Invalid final record')
            self.finished = True

    def finish(self):
        meta = self.run['metadata']
        if (not self.finished or self.turn is not None or self.last_tick_end != meta['final_tick']
                or dict(self.counts) != meta['event_counts'] or sum(self.counts.values()) != meta['records']
                or (len(self.alive),self.births,self.deaths) != tuple(meta['statistics'][k] for k in ('population','births','deaths'))):
            self.error('Recording incomplete or does not match completion metadata')
