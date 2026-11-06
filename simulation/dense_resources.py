"""Opt-in exact resource snapshots after statistics. Standard library only."""
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
import csv
import gzip
import json
import struct

from simulation.resource_observer import ResourceObserver

MAGIC = b'ACRD082\n'
PHASE = 'end_tick_after_statistics'


def file_hash(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class DenseResourceWriter:
    """One uint64 tick and two row-major uint8 grids per gzip record."""
    def __init__(self, path, config):
        self.path = Path(path)
        self.metadata_path = Path(str(path) + '.json')
        self.stock_path = Path(str(path) + '.stocks.csv')
        if any(p.exists() for p in (self.path, self.metadata_path, self.stock_path)):
            raise ValueError('Refusing dense snapshot overwrite')
        capacities = {r: getattr(config, r + '_capacity') for r in ('food', 'water')}
        if any(type(c) is not int or not 0 <= c <= 255 for c in capacities.values()):
            raise ValueError('uint8 serialization requires finite integer capacities in [0,255]; use a new schema for larger/unbounded quantities')
        self.header = {'schema_version': 1, 'phase': PHASE, 'shape': [2, config.world_height, config.world_width],
            'resource_types': ['food', 'water'], 'dtype': 'uint8', 'order': 'resource,y,x', 'endianness': 'not applicable to uint8',
            'tick_encoding': 'uint64 little-endian', 'capacities': capacities, 'compression': 'gzip level6',
            'initial_stock': {r: getattr(config, 'initial_' + r) for r in capacities},
            'source_sha256': file_hash(__file__)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        raw = self.stack.enter_context(self.path.open('xb'))
        self.stream = self.stack.enter_context(gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=6))
        stock = self.stack.enter_context(self.stock_path.open('x', newline=''))
        self.stock = csv.writer(stock, lineterminator='\n')
        self.stock.writerow(['tick', 'food_stock', 'water_stock', 'food_expected', 'water_expected'])
        self.hash = sha256(); self.count = 0; self.last_tick = -1; self.closed = False
        payload = json.dumps(self.header, sort_keys=True, separators=(',', ':')).encode()
        self._write(MAGIC + struct.pack('<I', len(payload)) + payload)

    def _write(self, data):
        self.stream.write(data); self.hash.update(data)

    def snapshot(self, sim):
        if self.closed or sim.tick != self.last_tick + 1:
            raise ValueError('Dense snapshots must start at0 and be consecutive')
        if [2, sim.world.height, sim.world.width] != self.header['shape']:
            raise ValueError('Dense snapshot grid shape changed')
        if sim.statistics.tick != sim.tick:
            raise ValueError('Snapshot must follow statistics finalization')
        payload = bytearray(struct.pack('<Q', sim.tick)); totals = []; expected = []
        for resource in self.header['resource_types']:
            values = [getattr(tile, resource) for row in sim.world.tiles for tile in row]
            if any(type(v) is not int or not 0 <= v <= self.header['capacities'][resource] for v in values):
                raise ValueError('Resource quantity outside declared integer capacity')
            actual = sum(values)
            balance = self.header['initial_stock'][resource] + getattr(sim.statistics, resource + '_regenerated') - getattr(sim.statistics, resource + '_consumed')
            if actual != balance:
                raise ValueError('Dense resource mass balance failed: ' + resource)
            payload.extend(values); totals.append(actual); expected.append(balance)
        self._write(bytes(payload)); self.stock.writerow([sim.tick, *totals, *expected])
        self.last_tick = sim.tick; self.count += 1

    def close(self, completed=True):
        if self.closed:
            return
        self.stack.close(); self.closed = True
        if completed:
            if not self.count:
                raise ValueError('Missing initial snapshot')
            value = {**self.header, 'completed': True, 'tick_first': 0, 'tick_last': self.last_tick, 'records': self.count,
                'file': self.path.name, 'bytes': self.path.stat().st_size, 'sha256': file_hash(self.path),
                'uncompressed_sha256': self.hash.hexdigest(), 'stocks_file': self.stock_path.name,
                'stocks_sha256': file_hash(self.stock_path), 'all_mass_balance_checks_passed': True}
            with self.metadata_path.open('x') as stream:
                json.dump(value, stream, indent=2); stream.write('\n')

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(completed=kind is None)


class DenseResourceObserver(ResourceObserver):
    """Uses the existing tick_end hook; no Simulation or Brain changes."""
    def __init__(self, sink, mode, ecology, geometry, dense):
        super().__init__(sink, mode, ecology, geometry)
        self.dense = dense

    def tick_end(self, sim):
        super().tick_end(sim)
        self.dense.snapshot(sim)


def read_dense(path):
    """Verify sidecar/CRC/checksums; yield consecutive (tick, food bytes, water bytes)."""
    path = Path(path)
    meta = json.loads(Path(str(path) + '.json').read_text())
    if not meta.get('completed') or meta.get('schema_version') != 1 or meta.get('phase') != PHASE or meta.get('dtype') != 'uint8':
        raise ValueError('Unsupported or incomplete dense snapshot format/phase')
    if file_hash(path) != meta['sha256'] or file_hash(path.parent / meta['stocks_file']) != meta['stocks_sha256']:
        raise ValueError('Dense snapshot checksum mismatch')
    with gzip.open(path, 'rb') as stream:
        prefix = stream.read(len(MAGIC) + 4)
        if len(prefix) != len(MAGIC) + 4 or prefix[:len(MAGIC)] != MAGIC:
            raise ValueError('Invalid dense magic/header')
        length = struct.unpack('<I', prefix[-4:])[0]
        if length > 100000:
            raise ValueError('Invalid dense header length')
        raw = stream.read(length); header = json.loads(raw)
        if any(header.get(k) != meta.get(k) for k in header) or header['shape'][0] != 2:
            raise ValueError('Dense header/sidecar disagreement')
        if len(header['shape']) != 3 or any(type(n) is not int or n <= 0 for n in header['shape']) or header['resource_types'] != ['food', 'water']:
            raise ValueError('Invalid dense grid shape/resource order')
        size = header['shape'][1] * header['shape'][2]
        check = sha256(prefix + raw); count = 0
        while True:
            data = stream.read(8 + 2 * size)
            if not data:
                break
            if len(data) != 8 + 2 * size or struct.unpack('<Q', data[:8])[0] != count:
                raise ValueError('Truncated, missing or unordered dense snapshot')
            check.update(data)
            food, water = data[8:8 + size], data[8 + size:]
            if max(food, default=0) > header['capacities']['food'] or max(water, default=0) > header['capacities']['water']:
                raise ValueError('Serialized quantity exceeds capacity')
            yield count, food, water
            count += 1
        if count != meta['records'] or meta['tick_first'] != 0 or meta['tick_last'] != count - 1 or check.hexdigest() != meta['uncompressed_sha256']:
            raise ValueError('Dense record count/raw checksum mismatch')
