from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite

from agents.observation import AuditoryKind, AuditoryPercept, HeardSignal, MASKED, SILENCE, SourceDirection
from config import AUDITORY_MASKING_RATIO


def apparent_loudness(distance: int) -> float:
    """Equal-intensity abstract sound: 1/(Chebyshev distance + 1)^2."""
    return 1 / (distance + 1) ** 2


@dataclass(frozen=True, slots=True)
class Signal:
    """Internal transmission provenance; never given to a Brain."""

    signal_id: int
    sender_id: int
    x: int
    y: int


class Communication:
    """Meaning-free signals audible for the following tick only."""

    def __init__(self, vocab_size: int, hearing_radius: int,
                 masking_ratio: float = AUDITORY_MASKING_RATIO) -> None:
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("Vocabulary size must be a positive integer")
        if type(hearing_radius) is not int or hearing_radius < 0:
            raise ValueError("Hearing radius must be a nonnegative integer")
        if type(masking_ratio) not in (int, float) or not isfinite(masking_ratio) or masking_ratio <= 1:
            raise ValueError("Masking ratio must be finite and greater than 1")
        self.vocab_size = vocab_size
        self.hearing_radius = hearing_radius
        self.masking_ratio = masking_ratio
        self._cell_size = max(1, hearing_radius)
        self._audible: dict[tuple[int, int], list[Signal]] = {}
        self._pending: dict[tuple[int, int], list[Signal]] = {}

    def validate_signal(self, signal_id: int) -> None:
        if type(signal_id) is not int or not 0 <= signal_id < self.vocab_size:
            raise ValueError(f"Signal must be an integer in [0, {self.vocab_size})")

    def emit(self, signal_id: int, sender_id: int, x: int, y: int) -> None:
        self.validate_signal(signal_id)
        cell = (x // self._cell_size, y // self._cell_size)
        self._pending.setdefault(cell, []).append(Signal(signal_id, sender_id, x, y))

    def advance(self) -> None:
        self._audible, self._pending = self._pending, {}

    def hear(self, receiver_id: int, x: int, y: int) -> tuple[HeardSignal, ...]:
        """Legacy physical-candidate diagnostic view. Brains receive perceive() instead."""
        return tuple(HeardSignal(signal.signal_id, SourceDirection.from_offset(signal.x - x, signal.y - y))
                     for signal in self._received(receiver_id, x, y))

    def _strongest(self, receiver_id: int, x: int, y: int) -> tuple[Signal | None, int | None, int | None]:
        """Find the two nearest candidates in linear time and constant extra space.

        An equal-distance runner-up forces MASKED, so traversal order cannot
        give any signal an identification advantage.
        """
        strongest = None
        first_distance = second_distance = None
        for signal in self._received(receiver_id, x, y):
            distance = max(abs(signal.x - x), abs(signal.y - y))
            if first_distance is None or distance < first_distance:
                strongest, first_distance, second_distance = signal, distance, first_distance
            elif second_distance is None or distance < second_distance:
                second_distance = distance
        return strongest, first_distance, second_distance

    def perceive(self, receiver_id: int, x: int, y: int) -> AuditoryPercept:
        strongest, first_distance, second_distance = self._strongest(receiver_id, x, y)
        if strongest is None:
            return SILENCE
        if second_distance is not None:
            # Algebraically L1/L2; integer squares avoid intermediate rounding.
            ratio = (second_distance + 1) ** 2 / (first_distance + 1) ** 2
            if ratio < self.masking_ratio:
                return MASKED
        return AuditoryPercept(AuditoryKind.IDENTIFIED,
                              HeardSignal(strongest.signal_id, SourceDirection.from_offset(strongest.x - x, strongest.y - y)))

    def _received(self, receiver_id: int, x: int, y: int) -> Iterator[Signal]:
        """Inclusive Chebyshev range; deterministic lookup with no random draws."""
        cx, cy = x // self._cell_size, y // self._cell_size
        # Nine spatial buckets avoid scanning every sender for every receiver.
        for by in range(cy - 1, cy + 2):
            for bx in range(cx - 1, cx + 2):
                for signal in self._audible.get((bx, by), ()):
                    dx, dy = signal.x - x, signal.y - y
                    if signal.sender_id != receiver_id and max(abs(dx), abs(dy)) <= self.hearing_radius:
                        yield signal

    def diagnostics(self, receiver_id: int, x: int, y: int, receiving_tick: int) -> tuple[dict, ...]:
        """Privileged event data, separate from Observation. Call only if logging.

        Like hear(), this reads the buffer for the current acting phase. It logs
        delivered signals only; absent/out-of-range signals create no record.
        """
        return tuple({
            "signal_id": signal.signal_id,
            "sender_id": signal.sender_id,
            "sender_position": {"x": signal.x, "y": signal.y},
            "receiver_id": receiver_id,
            "receiver_position": {"x": x, "y": y},
            "source_direction": SourceDirection.from_offset(signal.x - x, signal.y - y),
            "chebyshev_distance": max(abs(signal.x - x), abs(signal.y - y)),
            "apparent_loudness": apparent_loudness(max(abs(signal.x - x), abs(signal.y - y))),
            "hearing_radius": self.hearing_radius,
            "inside_hearing_range": True,
            "emitted_tick": receiving_tick - 1,
            "receiving_tick": receiving_tick,
        } for signal in self._received(receiver_id, x, y))

    def resolution_diagnostics(self, receiver_id: int, x: int, y: int) -> dict:
        """Optional privileged masking calculation, never Brain-facing."""
        _, first, second = self._strongest(receiver_id, x, y)
        return {"strongest_loudness": apparent_loudness(first) if first is not None else None,
                "second_strongest_loudness": apparent_loudness(second) if second is not None else None,
                "dominance_ratio": (second + 1) ** 2 / (first + 1) ** 2 if second is not None else None,
                "masking_ratio": self.masking_ratio,
                "resolved_percept": self.perceive(receiver_id, x, y)}
