"""Optional detailed events, separate from aggregate statistics."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TextIO


@dataclass(frozen=True, slots=True)
class Event:
    tick: int
    kind: str
    agent_id: int | None = None
    x: int | None = None
    y: int | None = None
    details: dict[str, object] = field(default_factory=dict)


EventSink = Callable[[Event], None]


class JSONLEventWriter:
    """Callable event sink. The caller owns and closes the output stream."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream

    def __call__(self, event: Event) -> None:
        self.stream.write(json.dumps(asdict(event), separators=(",", ":")) + "\n")
