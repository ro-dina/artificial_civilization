"""Exact archive labels. No simulator, RNG or predictive-model state is touched."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

OBSERVED, NO_SNAPSHOT, ADMIN_END, HISTORY_GAP, DEATH, NO_REFERENCE, NOT_AT_RISK = range(7)
REASONS = dict(enumerate(('observed', 'missing_snapshot', 'administrative_end',
                          'missing_history', 'death_competing', 'no_reference', 'not_at_risk')))


@dataclass(frozen=True)
class Target:
    name: str
    family: str
    kind: str
    resource: str
    horizon: int = 0
    radius: int = 0
    location: str = 'current'
    transition: str = ''


def definitions():
    result = []
    for resource in ('food', 'water'):
        for location in ('current', 'neighbor'):
            result.append(Target(f'T1_{resource}_{location}_h5', 'T1', 'environmental', resource, 5, location=location))
        for radius in (0, 1, 2):
            for horizon in (1, 5, 10):
                if radius == 0 and horizon == 5:
                    continue  # Exact alias of T1 current, not an independent target.
                result.append(Target(f'T2_{resource}_r{radius}_h{horizon}', 'T2', 'environmental', resource, horizon, radius))
        for transition in ('depletion', 'recovery', 'increase', 'decrease'):
            result.append(Target(f'T3_{resource}_{transition}_h5', 'T3', 'environmental', resource, 5, transition=transition))
    for resource in ('food', 'water', 'either'):
        result.append(Target(f'T4_{resource}', 'T4', 'time_to_event', resource))
        for k in (5, 10):
            result.append(Target(f'T5_{resource}_k{k}', 'T5', 'behavioral', resource, k))
    return result


def target_table():
    rows = []
    for target in definitions():
        rows.append({**asdict(target), 'future_clock': 'own decisions' if target.family in ('T4', 'T5') else 'end tick t+h',
                     'reference': 'frozen pre-action location' if target.kind == 'environmental' else 'same agent actual future actions',
                     'reconstruction': 'exact only at saved endpoint snapshots' if target.kind == 'environmental' else 'exact linked contiguous decision histories'})
    return pd.DataFrame(rows)


def snapshots(frame, width, height):
    """Require complete unique grids. Never interpolate between recorded ticks."""
    result = {}
    for tick, group in frame.groupby('tick', sort=True):
        if len(group) != width * height or group.duplicated(['x', 'y']).any():
            raise ValueError('Incomplete or duplicate spatial grid')
        if not (group.x.between(0, width - 1) & group.y.between(0, height - 1)).all():
            raise ValueError('Out-of-bounds spatial record')
        ordered = group.sort_values(['y', 'x'])
        result[int(tick)] = {r: ordered[r].to_numpy(dtype=int).reshape(height, width) for r in ('food', 'water')}
    return result


def local_presence(array, radius):
    height, width = array.shape
    padded = np.pad(array > 0, radius)
    result = np.zeros_like(array, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy:dy + height, dx:dx + width]
    return result


def environmental(frame, grids, target, end_tick):
    n = len(frame)
    labels = np.full(n, -1, dtype=np.int8)
    reasons = np.full(n, NO_SNAPSHOT, dtype=np.uint8)
    # Coordinates are diagnostics for labels only; post_x/post_y never consulted.
    x = frame.x.to_numpy().copy()
    y = frame.y.to_numpy().copy()
    reference = np.ones(n, dtype=bool)
    if target.location == 'neighbor':
        reference = frame.nearest_visible_dx.notna().to_numpy() & frame.nearest_visible_dy.notna().to_numpy()
        x += frame.nearest_visible_dx.fillna(0).to_numpy(dtype=int)
        y += frame.nearest_visible_dy.fillna(0).to_numpy(dtype=int)
    current = frame[target.resource + '_here'].to_numpy()
    if target.transition == 'depletion':
        reference &= current > 0
    elif target.transition == 'recovery':
        reference &= current == 0
    reasons[~reference] = NO_REFERENCE if target.location == 'neighbor' else NOT_AT_RISK
    future_tick = frame.tick.to_numpy() + target.horizon
    reasons[reference & (future_tick > end_tick)] = ADMIN_END
    for tick, resources in grids.items():
        indices = np.flatnonzero(reference & (future_tick == tick))
        if not len(indices):
            continue
        array = resources[target.resource]
        if not ((x[indices] >= 0) & (x[indices] < array.shape[1]) & (y[indices] >= 0) & (y[indices] < array.shape[0])).all():
            raise ValueError('Invalid fixed reference coordinates')
        quantity = array[y[indices], x[indices]]
        if target.transition == 'depletion':
            values = quantity == 0
        elif target.transition == 'recovery':
            values = quantity > 0
        elif target.transition == 'increase':
            values = quantity > current[indices]
        elif target.transition == 'decrease':
            values = quantity < current[indices]
        else:
            values = local_presence(array, target.radius)[y[indices], x[indices]]
        labels[indices] = values
        reasons[indices] = OBSERVED
    return labels, reasons


def future_events(frame, end_tick):
    """Next acquisition competes with death; gaps are never bridged.

    Event: 1 acquisition, 2 death, 3 administrative censor, 4 missing history.
    Consumption precedes death on a future terminal row. Origin consumption is
    excluded; origin death is an immediate competing event at elapsed zero.
    """
    n = len(frame)
    result = {r: (np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.uint8)) for r in ('food', 'water', 'either')}
    for group in frame.groupby('agent_id', sort=False).indices.values():
        ticks = frame.tick.to_numpy()[group]
        own = frame.own_index.to_numpy()[group]
        cuts = np.r_[0, np.flatnonzero((np.diff(ticks) != 1) | (np.diff(own) != 1)) + 1, len(group)]
        for start, stop in zip(cuts[:-1], cuts[1:]):
            indices = group[start:stop]
            m = len(indices)
            positions = np.arange(m)
            terminal = frame.terminal.to_numpy()[indices].astype(bool)
            if terminal[:-1].any():
                raise ValueError('Decision after a terminal transition')
            for resource in result:
                acquired = (frame.acquired if resource == 'either' else frame[resource + '_consumed']).to_numpy()[indices] > 0
                future = np.minimum.accumulate(np.where(acquired, positions, m)[::-1])[::-1]
                future = np.r_[future[1:], m]  # Strictly AFTER the current decision.
                event = future < m
                elapsed = np.where(event, future - positions, m - 1 - positions)
                cause = np.where(event, 1, 2 if terminal[-1] else 3 if frame.tick.iloc[indices[-1]] == end_tick else 4).astype(np.uint8)
                cause[terminal] = 2
                elapsed[terminal] = 0
                result[resource][0][indices] = elapsed
                result[resource][1][indices] = cause
    return result


def behavioral(frame, events, target):
    elapsed, event = events[target.resource]
    labels = np.full(len(frame), -1, dtype=np.int8)
    reasons = np.full(len(frame), HISTORY_GAP, dtype=np.uint8)
    acquired = (event == 1) & (elapsed <= target.horizon)
    complete = (elapsed >= target.horizon) & ~((event == 2) & (elapsed == target.horizon))
    labels[complete] = 0
    reasons[complete] = OBSERVED
    labels[acquired] = 1
    reasons[acquired] = OBSERVED
    reasons[(event == 2) & (elapsed <= target.horizon)] = DEATH
    reasons[(event == 3) & (elapsed < target.horizon)] = ADMIN_END
    # Death on a future acquisition row loses the tie to consumption, already
    # encoded as acquisition by future_events(). Never turn death into negative.
    return labels, reasons


def competing_curve(elapsed, event, horizon=10):
    """Discrete Aalen–Johansen risks; descriptive overlapping prediction origins."""
    survival = 1.0
    acquisition = death = 0.0
    rows = []
    for time in range(horizon + 1):
        risk = int((elapsed >= time).sum())
        acq = int(((elapsed == time) & (event == 1)).sum())
        dead = int(((elapsed == time) & (event == 2)).sum())
        censored = int(((elapsed == time) & np.isin(event, [3, 4])).sum())
        if risk:
            acquisition += survival * acq / risk
            death += survival * dead / risk
            survival *= 1 - (acq + dead) / risk
        rows.append({'elapsed': time, 'at_risk': risk, 'acquisitions': acq, 'deaths': dead, 'censored': censored,
                     'acquisition_cif': acquisition, 'death_cif': death, 'event_free': survival})
    return rows
