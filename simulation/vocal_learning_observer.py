"""Opt-in exact Vocal Q instrumentation. No RNG, Q writes, or agent retention.

Streams decisions, every TD update, actual row events and an interned key catalog.
Historical keys/counters live here, never in an agent's bounded cognitive memory.
"""
from contextlib import ExitStack
import csv
from dataclasses import dataclass, field
import gzip
import hashlib
import io
import json
from math import sqrt
from pathlib import Path

SCHEMA_VERSION = 1
DECISION_COLUMNS = (
    'tick', 'agent_id', 'age', 'generation', 'condition', 'state_id', 'rows',
    'present_at_start', 'present_before_selection', 'newly_allocated', 'novelty',
    'decision_allocations', 'decision_evictions', 'allocations', 'evictions', 'updates',
    'unique_states', 'state_visits', 'q_max', 'q_min', 'q_mean', 'q_std', 'q_gap',
    'max_ties', 'all_equal', 'silence_q', 'silence_rank', 'silence_is_max', 'selected_q',
    'selected_action', 'exploratory', 'n_actions', 'epsilon', 'p_silence', 'at_capacity')
UPDATE_COLUMNS = ('tick', 'agent_id', 'age', 'state_id', 'action', 'reward', 'next_value', 'target',
                  'old_q', 'new_q', 'td_error', 'abs_td_error', 'q_change', 'visits', 'updates',
                  'row_updates', 'epoch', 'recreated', 'terminal')
ROW_COLUMNS = ('event', 'tick', 'agent_id', 'state_id', 'visits', 'updates', 'epoch', 'rows')
STATE_COLUMNS = ('state_id', 'key_json')
SCHEMAS = {'decisions': DECISION_COLUMNS, 'updates': UPDATE_COLUMNS, 'rows': ROW_COLUMNS, 'states': STATE_COLUMNS}


def condition(config):
    return ('ABCD'[int(config.learning_uses_auditory) + 2 * int(config.learning_controls_vocalization)]
            if config.brain == 'learning' else config.brain)


def q_diagnostics(values, epsilon):
    """Population SD across actions; exact == ties as in the actual selector."""
    maximum, minimum = max(values), min(values)
    mean = sum(values) / len(values)
    ties = sum(v == maximum for v in values)
    second = sorted(values, reverse=True)[1] if len(values) > 1 else maximum
    silence_max = values[0] == maximum
    return (maximum, minimum, mean, sqrt(sum((v - mean) ** 2 for v in values) / len(values)),
            maximum - second, ties, int(maximum == minimum), values[0],
            1 + sum(v > values[0] for v in values), int(silence_max),
            epsilon / len(values) + (1 - epsilon) * int(silence_max) / ties)


def recording_paths(path):
    path = Path(path)
    stem = path.name[:-7] if path.name.endswith('.csv.gz') else path.stem
    suffix = '.csv.gz' if path.name.endswith('.gz') else '.csv'
    return {'decisions': path, **{k: path.with_name(stem + '.' + k + suffix)
                                  for k in ('updates', 'rows', 'states')}}


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class CSVVocalLearningWriter:
    """Caller owns four streams; use open() for deterministic gzip (mtime=0)."""
    def __init__(self, streams):
        self.writers = {k: csv.writer(stream) for k, stream in streams.items()}
        for key, writer in self.writers.items():
            writer.writerow(SCHEMAS[key])
        self.counts = dict.fromkeys(SCHEMAS, 0)

    @classmethod
    def open(cls, stack: ExitStack, path):
        streams = {}
        for key, dest in recording_paths(path).items():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.name.endswith('.gz'):
                raw = stack.enter_context(dest.open('wb'))
                compressed = stack.enter_context(gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=1))
                streams[key] = stack.enter_context(io.TextIOWrapper(compressed, encoding='utf-8', newline=''))
            else:
                streams[key] = stack.enter_context(dest.open('w', encoding='utf-8', newline=''))
        return cls(streams)

    def __call__(self, kind, row):
        self.writers[kind].writerow(row)
        self.counts[kind] += 1


@dataclass(slots=True)
class History:
    visits: int = 0
    updates: int = 0
    epoch: int = 0
    row_visits: int = 0
    row_updates: int = 0


@dataclass(slots=True)
class AgentHistory:
    states: dict = field(default_factory=dict)
    allocations: int = 0
    evictions: int = 0
    unique_decision_states: int = 0


class VocalLearningObserver:
    def __init__(self, config, sink):
        self.config, self.condition, self.sink = config, condition(config), sink
        self.keys = {}  # Exact tuple -> encounter-order integer; never Python hash() as an ID.
        self.agents = {}  # Alive actors' scalar/key histories only. Released on death/end.
        self.finished = False
        self.peak_histories = 0
        self.current_histories = 0

    def begin(self, human, tick):
        if self.finished:
            raise RuntimeError('Diagnostic observer already finished')
        self.tick, self.agent_id, self.age, self.generation = tick, human.id, human.age, human.generation
        from agents.learning import LearningBrain
        self.applicable = isinstance(human.brain, LearningBrain) and human.brain.learning_controls_vocalization
        if self.applicable:
            human.brain._vocal_observer = self
            self.agent = self.agents.setdefault(human.id, AgentHistory())
            self.start_allocations, self.start_evictions = self.agent.allocations, self.agent.evictions

    def identify(self, state):
        if state not in self.keys:
            sid = len(self.keys)
            self.keys[state] = sid
            self.sink('states', (sid, json.dumps(state, separators=(',', ':'))))
        return self.keys[state]

    def history(self, state):
        sid = self.identify(state)
        if sid not in self.agent.states:
            self.agent.states[sid] = History()
            self.current_histories += 1
            self.peak_histories = max(self.peak_histories, self.current_histories)
        return sid, self.agent.states[sid]

    def start_state(self, state, present):
        _, history = self.history(state)
        self.present_at_start = present
        self.novelty = 'first' if history.visits == 0 else 'stored' if present else 'recreated'

    def selecting(self, present):
        self.present_before_selection = present

    def allocated(self, state):
        sid, history = self.history(state)
        history.epoch += 1
        history.row_visits = history.row_updates = 0
        self.agent.allocations += 1
        self.sink('rows', ('allocate', self.tick, self.agent_id, sid, history.visits, history.updates, history.epoch, None))

    def evicted(self, state):
        sid, history = self.history(state)
        self.agent.evictions += 1
        self.sink('rows', ('evict', self.tick, self.agent_id, sid, history.row_visits, history.row_updates, history.epoch, None))

    def selected(self, brain, state, values, index):
        sid, history = self.history(state)
        if history.visits == 0:
            self.agent.unique_decision_states += 1
        history.visits += 1
        history.row_visits += 1
        q = q_diagnostics(values, brain.epsilon)
        self.sink('decisions', (
            self.tick, self.agent_id, self.age, self.generation, self.condition, sid, brain.vocal_memory_size,
            int(self.present_at_start), int(self.present_before_selection), int(not self.present_before_selection), self.novelty,
            self.agent.allocations - self.start_allocations, self.agent.evictions - self.start_evictions,
            self.agent.allocations, self.agent.evictions, brain.vocal_learning_updates,
            self.agent.unique_decision_states, history.visits, *q[:-1], values[index], index,
            int(brain.last_vocal_exploratory), len(values), brain.epsilon, q[-1], int(brain.vocal_memory_size == brain.vocal_capacity)))

    def updated(self, state, index, old, new, target, reward, next_value, terminal):
        sid, history = self.history(state)
        history.updates += 1
        history.row_updates += 1
        td = target - old
        self.sink('updates', (self.tick, self.agent_id, self.age, sid, index, reward, next_value, target,
                             old, new, td, abs(td), abs(new - old), history.visits, history.updates,
                             history.row_updates, history.epoch, int(history.epoch > 1), int(terminal)))

    def not_applicable(self):
        self.sink('decisions', (self.tick, self.agent_id, self.age, self.generation, self.condition,
                               *([None] * (len(DECISION_COLUMNS) - 5))))

    def end_agent(self, human, tick, event):
        history = self.agents.pop(human.id, None)
        if history is not None:
            self.current_histories -= len(history.states)
            self.sink('rows', (event, tick, human.id, None, sum(s.visits for s in history.states.values()),
                               sum(s.updates for s in history.states.values()), None, human.brain.vocal_memory_size))
            human.brain._vocal_observer = None

    def finish(self, agents, tick):
        if not self.finished:
            for human in agents:
                self.end_agent(human, tick, 'run_end')
            self.finished = True
