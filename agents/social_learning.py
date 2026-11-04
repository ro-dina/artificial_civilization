"""Coarse visual features; identity is used for selection, never in Q keys."""
from dataclasses import dataclass
from enum import IntEnum

from agents.learning import LearningBrain
from agents.observation import Observation


class Direction(IntEnum):
    NONE = 0
    HERE = 1
    NORTH = 2
    SOUTH = 3
    EAST = 4
    WEST = 5


def direction(dx, dy):
    """Dominant axis, vertical on ties; north decreases y."""
    if dx == 0 and dy == 0:
        return Direction.HERE
    if abs(dy) >= abs(dx):
        return Direction.NORTH if dy < 0 else Direction.SOUTH
    return Direction.EAST if dx > 0 else Direction.WEST


def nearest(individuals):
    return min(individuals, key=lambda i: (max(abs(i.dx), abs(i.dy)), i.individual_id), default=None)


@dataclass(frozen=True, slots=True)
class SocialFeatures:
    visible: int = 0
    remembered_direction: int | None = None

    def __post_init__(self):
        if type(self.visible) is not int or not 0 <= self.visible <= 10:
            raise ValueError('visible must be a coarse integer token in [0,10]')
        r = self.remembered_direction
        if r is not None:
            if type(r) is not int or not 0 <= r <= 5:
                raise ValueError('remembered_direction must be None or an integer in [0,5]')
            d = self.visible_direction
            if (not self.visible and r != 0 or self.count_bin == 1 and r not in (0,d)
                    or d != Direction.HERE and r == Direction.HERE):
                raise ValueError('Impossible visible/remembered feature combination')

    @property
    def count_bin(self):
        return 0 if not self.visible else 1 if self.visible <= 5 else 2

    @property
    def visible_direction(self):
        return 0 if not self.visible else (self.visible - 1) % 5 + 1

    @property
    def suffix(self):
        return (self.visible,) if self.remembered_direction is None else (self.visible,self.remembered_direction)


@dataclass(frozen=True, slots=True)
class SocialLearningObservation(Observation):
    """No individual IDs, offsets, memory counts, timestamps or candidates."""
    social: SocialFeatures = SocialFeatures()

    def __post_init__(self):
        Observation.__post_init__(self)
        if not isinstance(self.social,SocialFeatures):
            raise ValueError('social must be an immutable SocialFeatures')


def extract(individuals, memory=None, *, include_memory=False):
    """Pure read of beginning-of-opportunity LTM; must run before observe()."""
    target = nearest(individuals)
    token = 0 if target is None else int(direction(target.dx,target.dy)) + (5 if len(individuals) > 1 else 0)
    remembered = tuple(i for i in individuals if memory is not None and memory.get(i.individual_id) is not None)
    known_target = nearest(remembered)
    rd = int(direction(known_target.dx,known_target.dy)) if known_target is not None else 0
    return SocialFeatures(token,rd if include_memory else None)


def learning_observation(observation, features):
    return SocialLearningObservation(observation.tick,observation.tiles,observation.auditory,features)


class SocialLearningBrain(LearningBrain):
    def __init__(self,rng,config,mode):
        if mode not in ('perception','memory'):
            raise ValueError('SocialLearningBrain requires perception or memory mode')
        super().__init__(rng,config)
        self.social_mode = mode

    def physical_state(self,body,observation,original=None):
        if not isinstance(observation,SocialLearningObservation):
            raise ValueError('Social learning requires the authoritative coarse decision snapshot')
        features = observation.social
        if (features.remembered_direction is not None) != (self.social_mode == 'memory'):
            raise ValueError('Social features do not match brain mode')
        return (*(self.state(body,observation) if original is None else original),*features.suffix)
