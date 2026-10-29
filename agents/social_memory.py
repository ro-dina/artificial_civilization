"""Bounded individual knowledge: sightings confirm life; absence conveys no death."""
from collections import OrderedDict
from dataclasses import dataclass, replace
from config import SOCIAL_MEMORY_CAPACITY, SOCIAL_REUNION_MIN_ABSENCE_TICKS


@dataclass(frozen=True, slots=True)
class SocialEntry:
    individual_id: int
    first_seen_tick: int
    last_seen_tick: int
    encounter_count: int

    @property
    def last_confirmed_alive_tick(self) -> int:
        return self.last_seen_tick


@dataclass(frozen=True, slots=True)
class SocialContact:
    kind: str
    entry: SocialEntry
    elapsed_since_last_seen: int | None
    evicted: SocialEntry | None
    occupancy: int


@dataclass(frozen=True, slots=True)
class SocialUpdate:
    contacts: tuple[SocialContact, ...]
    known_visible: int
    unknown_visible: int
    occupancy: int


class SocialMemory:
    """LRU of recognized identities, with no lifetime/evicted/death registry.

    Batch membership is classified before updates. Known sightings refresh first,
    followed by unknown allocations; each phase uses ascending identity. This
    prevents an allocation from changing another sighting's hit classification
    within a simultaneous visual observation. Crowds exceeding capacity can still
    cause deterministic churn. Diagnostic reads never refresh entries.
    """
    def __init__(self, capacity: int = SOCIAL_MEMORY_CAPACITY, reunion_min_absence_ticks: int = SOCIAL_REUNION_MIN_ABSENCE_TICKS):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Social memory capacity must be positive')
        if type(reunion_min_absence_ticks) is not int or reunion_min_absence_ticks < 1:
            raise ValueError('Social reunion threshold must be positive')
        self.capacity = capacity
        self.reunion_min_absence_ticks = reunion_min_absence_ticks
        self._entries: OrderedDict[int, SocialEntry] = OrderedDict()
        self._last_observation_tick: int | None = None

    def __len__(self):
        return len(self._entries)

    def entries(self) -> tuple[SocialEntry, ...]:
        """Immutable snapshot, least to most recent; no recency update."""
        return tuple(self._entries.values())

    def get(self, individual_id: int) -> SocialEntry | None:
        return self._entries.get(individual_id)

    def observe(self, individual_ids: tuple[int, ...], tick: int, *, capture: bool = False) -> SocialUpdate | None:
        """Exactly one observation opportunity per tick, including empty sightings."""
        if (type(tick) is not int or tick < 0 or
                self._last_observation_tick is not None and tick <= self._last_observation_tick):
            raise ValueError('Social observation ticks must be strictly increasing nonnegative integers')
        if (not isinstance(individual_ids, tuple) or any(type(i) is not int or i < 0 for i in individual_ids)
                or any(a >= b for a, b in zip(individual_ids, individual_ids[1:]))):
            raise ValueError('Visible identities must be a strictly increasing immutable tuple')
        known = [(i, self._entries[i]) for i in individual_ids if i in self._entries]
        unknown = [i for i in individual_ids if i not in self._entries]
        contacts = [] if capture else None
        for identity, previous in known:
            elapsed = tick - previous.last_seen_tick
            kind = ('continuous_contact' if previous.last_seen_tick == self._last_observation_tick
                    else 'reunion' if elapsed >= self.reunion_min_absence_ticks else 'return_contact')
            entry = replace(previous, last_seen_tick=tick, encounter_count=previous.encounter_count + 1)
            self._entries[identity] = entry
            self._entries.move_to_end(identity)
            if capture:
                contacts.append(SocialContact(kind, entry, elapsed, None, len(self)))
        for identity in unknown:
            evicted = self._entries.popitem(last=False)[1] if len(self) == self.capacity else None
            entry = SocialEntry(identity, tick, tick, 1)
            self._entries[identity] = entry
            if capture:
                contacts.append(SocialContact('first_encounter', entry, None, evicted, len(self)))
        self._last_observation_tick = tick
        return SocialUpdate(tuple(contacts), len(known), len(unknown), len(self)) if capture else None
