"""Exogenous stationary/scheduled nuisance. Clock selects frozen epochs only."""
from dataclasses import dataclass,field
from math import isfinite
from random import Random
from bisect import bisect_right

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
class NuisanceEpoch:
    start_tick: int
    model: NuisanceModel

    def __post_init__(self):
        if type(self.start_tick) is not int or self.start_tick<0 or not isinstance(self.model,NuisanceModel):
            raise ValueError('Epoch requires a nonnegative integer start and a stationary parameter model')


@dataclass(frozen=True,slots=True)
class NonstationaryNuisanceModel:
    category_count: int
    schedule: tuple[NuisanceEpoch,...]

    def __post_init__(self):
        if type(self.category_count) is not int or self.category_count not in (11,37):raise ValueError('Scheduled controls support 11 or 37 categories')
        if not isinstance(self.schedule,tuple) or not self.schedule or any(not isinstance(e,NuisanceEpoch) for e in self.schedule):raise ValueError('Immutable nonempty epoch tuple required')
        starts=tuple(e.start_tick for e in self.schedule)
        if starts[0]!=0 or any(a>=b for a,b in zip(starts,starts[1:])):raise ValueError('Schedule must start at zero and increase strictly')
        if any(e.model.categories!=self.category_count for e in self.schedule):raise ValueError('Epoch category counts must agree')

    @property
    def categories(self):return self.category_count

    def at(self,tick):
        if type(tick) is not int or tick<0:raise ValueError('Clock must be a nonnegative integer')
        return self.schedule[bisect_right(tuple(e.start_tick for e in self.schedule),tick)-1].model


def nuisance_model_from_dict(value):
    """Explicit versioned schedule, with unchanged v073 stationary file support."""
    try:
        if value.get('kind')=='nonstationary':
            return NonstationaryNuisanceModel(value['category_count'],tuple(NuisanceEpoch(e['start_tick'],NuisanceModel(tuple(e['probabilities']),e['hold_probability'])) for e in value['schedule']))
        if value.get('kind') not in (None,'stationary'):raise ValueError('Unknown nuisance model kind')
        return NuisanceModel(tuple(value['probabilities']),value['hold_probability'])
    except (KeyError,TypeError,AttributeError) as error:raise ValueError('Invalid nuisance model file structure') from error


@dataclass(frozen=True,slots=True)
class NuisanceConfig:
    model: NuisanceModel | NonstationaryNuisanceModel
    seed: int = 0

    def __post_init__(self):
        if not isinstance(self.model,(NuisanceModel,NonstationaryNuisanceModel)) or type(self.seed) is not int:
            raise ValueError('NuisanceConfig requires a frozen nuisance model and an integer seed')


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


@dataclass(slots=True)
class NonstationaryNuisanceChannel(NuisanceChannel):
    """Only an exogenous clock selects frozen parameters; no world inputs."""
    model: NonstationaryNuisanceModel
    last_tick: int | None = field(init=False,default=None)
    initial_uniform: float = field(init=False,repr=False)

    def __post_init__(self):
        self.initial_uniform=self.rng.random()
        self.token=self._categorical(self.model.at(0),self.initial_uniform)

    @staticmethod
    def _categorical(parameters,uniform):
        total=0.;last=0
        for i,p in enumerate(parameters.probabilities):
            if p:last=i
            total+=p
            if uniform<total:return i
        return last

    def preview(self,tick):
        parameters=self.model.at(tick)
        return self._categorical(parameters,self.initial_uniform) if self.last_tick is None else self.token

    def advance(self,tick):
        parameters=self.model.at(tick)
        if self.last_tick is not None and tick<=self.last_tick:raise ValueError('Actual scheduled opportunities must advance the clock')
        if self.last_tick is None:self.token=self._categorical(parameters,self.initial_uniform)
        elif self.rng.random()>=parameters.hold_probability:self.token=self._categorical(parameters,self.rng.random())
        self.opportunities+=1;self.last_tick=tick
        return self.token


class NuisanceSource:
    """Exchangeable substreams assigned at brain construction, not seeded by IDs.

    Creation order indexes independent draws only; the generator never receives
    a birth tick, age, individual ID, ecological seed, observation or outcome.
    """
    def __init__(self,config):
        if not isinstance(config,NuisanceConfig):raise ValueError('Expected NuisanceConfig')
        version='v074' if isinstance(config.model,NonstationaryNuisanceModel) else 'v073'
        self.config=config;self.rng=Random(f'{version}:nuisance:{config.seed}:{config.model.categories}');self.created=0
    def new_channel(self):
        self.created+=1
        cls=NonstationaryNuisanceChannel if isinstance(self.config.model,NonstationaryNuisanceModel) else NuisanceChannel
        return cls(self.config.model,Random(self.rng.getrandbits(128)))


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
