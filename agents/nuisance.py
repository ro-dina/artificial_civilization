"""Exogenous stationary nuisance inputs. No body, world, identity or Q access."""
from dataclasses import dataclass,field
from math import isfinite
from random import Random

from agents.learning import LearningBrain
from agents.observation import Observation


@dataclass(frozen=True,slots=True)
class NuisanceModel:
    probabilities: tuple[float,...]
    hold_probability: float = 0.0

    def __post_init__(self):
        if (not isinstance(self.probabilities,tuple) or not self.probabilities
                or any(type(p) not in (int,float) or not isfinite(p) or p<0 for p in self.probabilities)
                or abs(sum(self.probabilities)-1)>1e-10):
            raise ValueError('Nuisance probabilities must be an immutable finite probability vector summing to one')
        if type(self.hold_probability) not in (int,float) or not isfinite(self.hold_probability) or not 0<=self.hold_probability<=1:
            raise ValueError('hold_probability must be in [0,1]')

    @property
    def categories(self):return len(self.probabilities)

    @property
    def same_token_probability(self):
        return self.hold_probability+(1-self.hold_probability)*sum(p*p for p in self.probabilities)


@dataclass(frozen=True,slots=True)
class NuisanceConfig:
    model: NuisanceModel
    seed: int = 0

    def __post_init__(self):
        if not isinstance(self.model,NuisanceModel) or type(self.seed) is not int:
            raise ValueError('NuisanceConfig requires NuisanceModel and an integer nuisance seed')


@dataclass(slots=True)
class NuisanceChannel:
    model: NuisanceModel
    rng: Random = field(repr=False)
    token: int = field(init=False)
    opportunities: int = field(init=False,default=0)

    def __post_init__(self):self.token=self._draw()

    def _draw(self):
        u=self.rng.random();total=0.;last_positive=0
        for i,p in enumerate(self.model.probabilities):
            if p:last_positive=i
            total+=p
            if u<total:return i
        return last_positive  # Floating summation remainder, never a zero-probability tail.

    def advance(self):
        # Initial stationary draw is already available for read-only previews.
        if self.opportunities:
            if self.rng.random()>=self.model.hold_probability:self.token=self._draw()
        self.opportunities+=1
        return self.token


class NuisanceSource:
    """Exchangeable substreams assigned at brain construction, not seeded by IDs.

    Creation order indexes independent draws only; the generator never receives
    a birth tick, age, individual ID, ecological seed, observation or outcome.
    """
    def __init__(self,config):
        if not isinstance(config,NuisanceConfig):raise ValueError('Expected NuisanceConfig')
        self.config=config;self.rng=Random(f'v073:nuisance:{config.seed}:{config.model.categories}');self.created=0
    def new_channel(self):
        self.created+=1
        return NuisanceChannel(self.config.model,Random(self.rng.getrandbits(128)))


@dataclass(frozen=True,slots=True)
class NuisanceObservation(Observation):
    token: int = 0
    def __post_init__(self):
        Observation.__post_init__(self)
        if type(self.token) is not int or self.token<0:raise ValueError('Nuisance token must be a nonnegative integer')


class NuisanceBrain(LearningBrain):
    def __init__(self,rng,config,channel):
        super().__init__(rng,config)
        if not isinstance(channel,NuisanceChannel):raise ValueError('Expected NuisanceChannel')
        self.nuisance_channel=channel
    def physical_state(self,body,observation,original=None):
        if not isinstance(observation,NuisanceObservation) or observation.token>=self.nuisance_channel.model.categories:
            raise ValueError('Nuisance brain requires an in-range actual nuisance observation')
        return (*(self.state(body,observation) if original is None else original),observation.token)
