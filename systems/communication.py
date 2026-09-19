from dataclasses import dataclass

from agents.observation import HeardSignal


@dataclass(frozen=True, slots=True)
class Signal:
    signal_id: int
    sender_id: int
    x: int
    y: int


class Communication:
    """Meaning-free signals audible for the following tick only."""

    def __init__(self, vocab_size: int, hearing_radius: int) -> None:
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("Vocabulary size must be a positive integer")
        if type(hearing_radius) is not int or hearing_radius < 0:
            raise ValueError("Hearing radius must be a nonnegative integer")
        self.vocab_size = vocab_size
        self.hearing_radius = hearing_radius
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
        heard = []
        cx, cy = x // self._cell_size, y // self._cell_size
        # Nine spatial buckets avoid scanning every sender for every receiver.
        for by in range(cy - 1, cy + 2):
            for bx in range(cx - 1, cx + 2):
                for signal in self._audible.get((bx, by), ()):
                    dx, dy = signal.x - x, signal.y - y
                    if signal.sender_id != receiver_id and max(abs(dx), abs(dy)) <= self.hearing_radius:
                        heard.append(HeardSignal(signal.signal_id, signal.sender_id, dx, dy))
        return tuple(heard)
