"""Local pairing and biological eligibility; no family or social model."""

from random import Random

from agents.human import Human
from config import Config


class Reproduction:
    def __init__(self, config: Config, rng: Random) -> None:
        self.config = config
        self.rng = rng
        self._cell_size = max(1, config.reproduction_radius)
        self._buckets: dict[tuple[int, int], dict[int, Human]] = {}

    def _cell(self, x: int, y: int) -> tuple[int, int]:
        return x // self._cell_size, y // self._cell_size

    def begin_tick(self, agents: list[Human]) -> None:
        self._buckets.clear()
        for human in agents:
            self._buckets.setdefault(self._cell(human.x, human.y), {})[human.id] = human

    def moved(self, human: Human, old_x: int, old_y: int) -> None:
        old, new = self._cell(old_x, old_y), self._cell(human.x, human.y)
        if old != new:
            del self._buckets[old][human.id]
            if not self._buckets[old]:
                del self._buckets[old]
            self._buckets.setdefault(new, {})[human.id] = human

    def clear(self) -> None:
        # Never retain dead Human objects between ticks.
        self._buckets.clear()

    def eligible(self, human: Human, tick: int) -> bool:
        cfg = self.config
        return (
            cfg.reproduction_enabled and human.alive
            and human.age >= cfg.min_reproductive_age
            and (cfg.max_reproductive_age is None or human.age <= cfg.max_reproductive_age)
            and tick >= human.next_reproduction_tick
            and human.hunger < cfg.max_hunger and human.thirst < cfg.max_thirst
            and human.hunger <= cfg.max_hunger * cfg.reproduction_need_fraction
            and human.thirst <= cfg.max_thirst * cfg.reproduction_need_fraction
            and human.energy >= max(cfg.min_reproduction_energy, cfg.reproduction_energy_cost)
            # If energy is lethal, payment must leave a positive reserve.
            and (not cfg.energy_depletion_lethal or human.energy > cfg.reproduction_energy_cost)
        )

    def partner_for(self, human: Human, tick: int) -> Human | None:
        if not self.eligible(human, tick):
            return None
        cx, cy = self._cell(human.x, human.y)
        candidates = []
        for by in range(cy - 1, cy + 2):
            for bx in range(cx - 1, cx + 2):
                # Python dictionaries preserve insertion order. They are populated
                # from the seeded acting order and updated in that same order.
                for other in self._buckets.get((bx, by), {}).values():
                    if (other.id != human.id and other.reproductive_type != human.reproductive_type
                            and max(abs(other.x - human.x), abs(other.y - human.y)) <= self.config.reproduction_radius
                            and self.eligible(other, tick)):
                        candidates.append(other)
        return self.rng.choice(candidates) if candidates else None

    def charge(self, a: Human, b: Human, tick: int) -> None:
        for parent in (a, b):
            parent.energy -= self.config.reproduction_energy_cost
            parent.next_reproduction_tick = tick + max(1, self.config.reproduction_cooldown)
