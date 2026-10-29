"""Read-only local query shared with the existing live reproduction buckets."""
from agents.observation import VisibleIndividual


def nearby_individuals(buckets, cell_size, observer, radius):
    """Inclusive Chebyshev square, excluding self/dead; independent ID ordering."""
    found = []
    for by in range((observer.y-radius)//cell_size, (observer.y+radius)//cell_size+1):
        for bx in range((observer.x-radius)//cell_size, (observer.x+radius)//cell_size+1):
            for other in buckets.get((bx, by), {}).values():
                dx, dy = other.x-observer.x, other.y-observer.y
                if other.alive and other.id != observer.id and max(abs(dx), abs(dy)) <= radius:
                    found.append(VisibleIndividual(other.id, dx, dy))
    return tuple(sorted(found, key=lambda record: record.individual_id))


class SocialIndex:
    """Only needed when reproduction's existing spatial index is inactive."""
    def __init__(self):
        self._buckets = {}

    def begin_tick(self, agents):
        self._buckets.clear()
        for human in agents:
            self._buckets.setdefault((human.x, human.y), {})[human.id] = human

    def moved(self, human, old_x, old_y):
        del self._buckets[(old_x, old_y)][human.id]
        if not self._buckets[(old_x, old_y)]:
            del self._buckets[(old_x, old_y)]
        self._buckets.setdefault((human.x, human.y), {})[human.id] = human

    def visible(self, human):
        return nearby_individuals(self._buckets, 1, human, human.genome.perception_radius)

    def clear(self):
        self._buckets.clear()
