"""Mechanical memory of recent non-silent percepts, independent of reward."""

from collections import deque
from dataclasses import dataclass

from agents.observation import AuditoryKind, AuditoryPercept, SourceDirection

AuditoryFeature = (tuple[AuditoryKind] | tuple[AuditoryKind, int]
                   | tuple[AuditoryKind, int, SourceDirection, int])


@dataclass(frozen=True, slots=True)
class AuditoryEvent:
    received_tick: int
    percept: AuditoryPercept


class AuditoryMemory:
    """At most one event per tick; age 0..horizon-1 is retained.

    Repeated non-silent observations at the same tick replace that tick's event. Gaps in
    tick numbers count as elapsed time. Read-only previews never store events.
    """

    def __init__(self, horizon: int) -> None:
        if type(horizon) is not int or horizon < 1:
            raise ValueError("Auditory memory horizon must be a positive integer")
        self.horizon = horizon
        self._events: deque[AuditoryEvent] = deque(maxlen=horizon)
        self._last_tick: int | None = None

    def update(self, percept: AuditoryPercept, tick: int) -> None:
        if type(tick) is not int or tick < 0 or (self._last_tick is not None and tick < self._last_tick):
            raise ValueError("Auditory memory ticks must be nonnegative integers in nondecreasing order")
        while self._events and tick - self._events[0].received_tick >= self.horizon:
            self._events.popleft()
        if percept.kind is not AuditoryKind.SILENCE:
            if self._events and self._events[-1].received_tick == tick:
                self._events.pop()
            self._events.append(AuditoryEvent(tick, percept))
        self._last_tick = tick

    def events(self, tick: int) -> tuple[AuditoryEvent, ...]:
        """Read-only snapshot of retained events, oldest first, excluding future events."""
        return tuple(event for event in self._events if 0 <= tick - event.received_tick < self.horizon)

    def feature(self, percept: AuditoryPercept, tick: int) -> AuditoryFeature:
        """Preview the current percept plus stored past events without updating memory."""
        age = 0
        if percept.kind is AuditoryKind.SILENCE:
            events = self.events(tick)
            if not events:
                return (AuditoryKind.SILENCE,)
            event = events[-1]
            percept, age = event.percept, tick - event.received_tick
        if percept.kind is AuditoryKind.MASKED:
            return (AuditoryKind.MASKED, age)
        return (AuditoryKind.IDENTIFIED, percept.signal.signal_id, percept.signal.source_direction, age)
