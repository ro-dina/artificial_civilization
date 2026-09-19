from __future__ import annotations

from collections.abc import Iterator
from random import Random
from typing import TYPE_CHECKING

from config import Config
from world.tile import Tile

if TYPE_CHECKING:
    from agents.human import Human


class World:
    def __init__(self, width: int, height: int) -> None:
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("World dimensions must be positive integers")
        self.width = width
        self.height = height
        self.tiles = [[Tile() for _ in range(width)] for _ in range(height)]

    @classmethod
    def generate(cls, config: Config, rng: Random) -> World:
        world = cls(config.world_width, config.world_height)
        # Placement with replacement preserves exact totals, even on small grids.
        for resource, count in (("food", config.initial_food), ("water", config.initial_water)):
            capacity = getattr(config, f"{resource}_capacity")
            if capacity is not None:
                # Uniform among non-full cells; swap removal avoids rejection loops.
                available = [tile for row in world.tiles for tile in row]
                for _ in range(count):
                    index = rng.randrange(len(available))
                    tile = available[index]
                    setattr(tile, resource, getattr(tile, resource) + 1)
                    if getattr(tile, resource) == capacity:
                        available[index] = available[-1]
                        available.pop()
                continue
            for _ in range(count):
                tile = world.tile_at(rng.randrange(world.width), rng.randrange(world.height))
                setattr(tile, resource, getattr(tile, resource) + 1)
        return world

    def regenerate(self, config: Config, food_rng: Random, water_rng: Random) -> tuple[int, int]:
        """At tick start, each non-full tile may gain one unit of each resource."""
        totals = []
        for resource, rng in (("food", food_rng), ("water", water_rng)):
            probability = getattr(config, f"{resource}_regeneration_probability")
            capacity = getattr(config, f"{resource}_capacity")
            gained = 0
            if probability > 0:
                for row in self.tiles:
                    for tile in row:
                        if getattr(tile, resource) < capacity and rng.random() < probability:
                            setattr(tile, resource, getattr(tile, resource) + 1)
                            gained += 1
            totals.append(gained)
        return totals[0], totals[1]

    def is_valid(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def tile_at(self, x: int, y: int) -> Tile:
        if not self.is_valid(x, y):
            raise IndexError(f"Coordinates outside world: ({x}, {y})")
        return self.tiles[y][x]

    def nearby_tiles(self, x: int, y: int, radius: int) -> Iterator[tuple[int, int, Tile]]:
        """Inclusive square (Chebyshev distance), clipped at world boundaries."""
        if type(radius) is not int or radius < 0:
            raise ValueError("Radius must be a nonnegative integer")
        if not self.is_valid(x, y):
            raise IndexError(f"Coordinates outside world: ({x}, {y})")
        for ty in range(max(0, y - radius), min(self.height, y + radius + 1)):
            for tx in range(max(0, x - radius), min(self.width, x + radius + 1)):
                yield tx, ty, self.tiles[ty][tx]

    def move(self, human: Human, dx: int, dy: int) -> bool:
        """Attempt one cardinal step. Overlapping agents are allowed."""
        if (dx, dy) not in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            raise ValueError("Movement must be one cardinal step")
        x, y = human.x + dx, human.y + dy
        if not human.alive or not self.is_valid(x, y):
            return False
        human.x, human.y = x, y
        return True

    def consume_food(self, x: int, y: int) -> int:
        tile = self.tile_at(x, y)
        if tile.food <= 0:
            return 0
        tile.food -= 1
        return 1

    def consume_water(self, x: int, y: int) -> int:
        tile = self.tile_at(x, y)
        if tile.water <= 0:
            return 0
        tile.water -= 1
        return 1
