"""Read-only v0.6.2 inputs, exact local key dictionaries and tick-sized streams."""
from itertools import groupby
from dataclasses import fields
import json
from pathlib import Path
import math

from experiments.vocal_learning_data import discover as discover_v062, read_chunks
from simulation.vocal_learning_observer import hash_file
from config import Config

DIRECTIONS = {'N','NE','E','SE','S','SW','W','NW','SAME_CELL'}


def discover(directory):
    runs = discover_v062(directory)
    for run in runs:
        meta, cfg = run['metadata'], run['metadata']['config']
        if set(cfg)!={f.name for f in fields(Config)}:
            raise ValueError('Complete explicit Config required; missing values are not defaulted for analysis')
        if not run['applicable'] or run['condition'].split(':')[0] not in {'C','D'}:
            raise ValueError('Persistence requires actual Vocal Q recordings (C/D), not A/B/random')
        stats = meta['statistics']
        if stats['tick'] != meta['final_tick'] or (meta['final_tick'] < cfg['max_ticks'] and stats['population']):
            raise ValueError('Incomplete horizon: recording must finish its requested ticks or go extinct')
        if stats['population'] != cfg['initial_population'] + stats['births'] - stats['deaths']:
            raise ValueError('Inconsistent population accounting in metadata')
        if meta['files']['decisions']['records'] != stats['exploratory_actions'] + stats['exploitative_actions']:
            raise ValueError('Incomplete actor decision stream')
        if meta['files']['updates']['records'] != stats['total_learning_updates']:
            raise ValueError('Incomplete Vocal Q update stream')
        run['metadata_sha256'] = hash_file(run['metadata_path'])
    manifest = Path(directory)/'pilot_manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if not saved.get('completed'):
            raise ValueError('Pilot manifest is incomplete')
        expected = {(r['condition'],r['seed']):r for r in saved['runs']}
        actual = {(r['condition'],r['seed']):r for r in runs}
        if set(expected) != set(actual):
            raise ValueError('Missing/extra pilot runs relative to completed manifest')
        for key, run in actual.items():
            if expected[key]['metadata_sha256'] != run['metadata_sha256']:
                raise ValueError('Pilot metadata hash differs from manifest')
    return runs


def integer(value, name, minimum=0):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or int(value)!=value or value < minimum:
        raise ValueError(f'{name}: expected integer >= {minimum}')
    return int(value)


def decode_state(text, config):
    value = json.loads(text)
    if not isinstance(value,list) or len(value)!=(8 if config['learning_uses_auditory'] else 7):
        raise ValueError('Encoded state has wrong body/visual/auditory structure')
    for number in value[:3]:
        if type(number) is not int or not 0 <= number < config['learning_need_bins']:
            raise ValueError('Invalid body bin in exact state dictionary')
    for dx,dy in (value[3:5],value[5:7]):
        if type(dx) is not int or type(dy) is not int or not ((dx in (-1,0,1) and dy in (-1,0,1)) or (dx,dy)==(2,2)):
            raise ValueError('Invalid visual direction in exact state dictionary')
    if len(value)==8:
        a=value[7]
        if not isinstance(a,list) or not a:
            raise ValueError('Invalid auditory feature')
        if a==['SILENCE']:
            pass
        elif len(a)==2 and a[0]=='MASKED':
            if type(a[1]) is not int or not 0 <= a[1] < config['auditory_memory_ticks']:
                raise ValueError('Invalid auditory memory age')
        elif len(a)==4 and a[0]=='IDENTIFIED':
            if (type(a[1]) is not int or not 0 <= a[1] < config['signal_vocab_size'] or a[2] not in DIRECTIONS
                or type(a[3]) is not int or not 0 <= a[3] < config['auditory_memory_ticks']):
                raise ValueError('Invalid identified auditory feature')
        else:
            raise ValueError('Unknown auditory feature')
    return json.dumps(value,separators=(',',':'))


def catalog(run):
    keys, exact = {}, set()
    for frame in read_chunks(run,'states'):
        for row in frame.itertuples(index=False):
            sid=integer(row.state_id,'state_id')
            key=decode_state(row.key_json,run['metadata']['config'])
            if sid in keys or key in exact:
                raise ValueError('State dictionary must be exact and one-to-one')
            keys[sid]=key;exact.add(key)
    if len(keys)!=run['metadata']['unique_keys']:
        raise ValueError('State dictionary size differs from completion metadata')
    run['catalog']=set(keys)
    return keys


def tick_records(run, kind):
    """One tick at a time, preserving recorded decision/row/update order."""
    def records():
        for frame in read_chunks(run,kind):
            for row in frame.itertuples(index=False):
                integer(row.tick,'tick',1);integer(row.agent_id,'agent_id')
                if not math.isnan(row.state_id):
                    integer(row.state_id,'state_id')
                yield row
    for tick, rows in groupby(records(),lambda row:int(row.tick)):
        yield tick,list(rows)


class TickCursor:
    def __init__(self, source):
        self.source=iter(source)
        self.next=next(self.source,None)

    def pop(self,tick):
        if self.next is not None and self.next[0]<tick:
            raise ValueError('Orphan events without an actor decision at their tick')
        if self.next is None or self.next[0]!=tick:
            return []
        result=self.next[1]
        self.next=next(self.source,None)
        return result

    def exhausted(self):
        if self.next is not None:
            raise ValueError('Unconsumed events after final decision')
