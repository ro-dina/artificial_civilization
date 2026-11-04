"""Fixed shared patches and capacity-aware relocation of legacy supply offers."""
from bisect import bisect_right
from dataclasses import dataclass,replace
from random import Random
from config import ResourceEcologyConfig


@dataclass(frozen=True,slots=True)
class PatchGeometry:
    centers: tuple[tuple[int,int],...]
    radius: int
    mask: tuple[bool,...]
    seed: str

    @classmethod
    def generate(cls,config,ecology):
        n=config.world_width*config.world_height
        if ecology.patch_count>n:raise ValueError('More patch centers than grid cells')
        seed=f'{config.random_seed}:resources:patch_geometry'
        centers=tuple((i%config.world_width,i//config.world_width) for i in Random(seed).sample(range(n),ecology.patch_count))
        mask=tuple(any(max(abs(x-cx),abs(y-cy))<=ecology.patch_radius for cx,cy in centers)
                   for y in range(config.world_height) for x in range(config.world_width))
        return cls(centers,ecology.patch_radius,mask,seed)


def spatial_weights(geometry,strength):
    n=len(geometry.mask);inside=sum(geometry.mask)
    return tuple((1-strength)/n+strength*bool(value)/inside for value in geometry.mask)


class WeightedPool:
    """Positive weights among non-full cells; rebuild CDF only at saturation."""
    def __init__(self,tiles,weights,resource,capacity):
        self.tiles=tiles;self.resource=resource;self.capacity=capacity
        self.available=[i for i,t in enumerate(tiles) if capacity is None or getattr(t,resource)<capacity]
        self.weights=weights;self._cdf()
    def _cdf(self):
        total=0.;self.cumulative=[]
        for i in self.available:total+=self.weights[i];self.cumulative.append(total)
        self.total=total
    def add(self,rng):
        if not self.available or self.total<=0:return False
        pos=min(bisect_right(self.cumulative,rng.random()*self.total),len(self.available)-1)
        tile=self.tiles[self.available[pos]];setattr(tile,self.resource,getattr(tile,self.resource)+1)
        if self.capacity is not None and getattr(tile,self.resource)==self.capacity:
            self.available[pos]=self.available[-1];self.available.pop();self._cdf()
        return True


class ResourceEcology:
    """No agent, population, reward, ownership or social information is consulted."""
    def __init__(self,config,ecology):
        if not isinstance(ecology,ResourceEcologyConfig):raise ValueError('ResourceEcologyConfig required')
        self.config=config;self.ecology=ecology
        self.supply_config=replace(config,food_regeneration_probability=config.food_regeneration_probability*ecology.resource_supply_multiplier,
                                   water_regeneration_probability=config.water_regeneration_probability*ecology.resource_supply_multiplier)
        self.geometry=PatchGeometry.generate(config,ecology) if ecology.localization_strength else None
        self.weights=spatial_weights(self.geometry,ecology.localization_strength) if self.geometry else None
        self.placement_rng={r:Random(f'{config.random_seed}:resources:placement:{r}') for r in ('food','water')} if self.geometry else {}
        self.last_supply={}

    def initialize(self,world):
        if self.geometry is None:return
        tiles=[t for row in world.tiles for t in row]
        for resource in ('food','water'):
            count=sum(getattr(t,resource) for t in tiles)
            for tile in tiles:setattr(tile,resource,0)
            pool=WeightedPool(tiles,self.weights,resource,getattr(self.config,resource+'_capacity'))
            for _ in range(count):
                if not pool.add(self.placement_rng[resource]):raise ValueError('Insufficient initial resource capacity')

    def regenerate(self,world,food_rng,water_rng):
        tiles=[t for row in world.tiles for t in row];info={};counts=[]
        if self.geometry is None:
            for r in ('food','water'):
                cap=getattr(self.config,r+'_capacity');info[r+'_eligible']=sum(cap is None or getattr(t,r)<cap for t in tiles)
            gains=world.regenerate(self.supply_config,food_rng,water_rng)
            for r,n in zip(('food','water'),gains):info[r+'_attempted']=info[r+'_realized']=n;info[r+'_rejected']=0
        else:
            for resource,rng in (('food',food_rng),('water',water_rng)):
                p=getattr(self.supply_config,resource+'_regeneration_probability');cap=getattr(self.config,resource+'_capacity')
                eligible=[t for t in tiles if cap is None or getattr(t,resource)<cap]
                offers=sum(rng.random()<p for t in eligible) if p>0 else 0
                pool=WeightedPool(tiles,self.weights,resource,cap)
                gained=sum(pool.add(self.placement_rng[resource]) for _ in range(offers))
                info.update({resource+'_eligible':len(eligible),resource+'_attempted':offers,resource+'_realized':gained,resource+'_rejected':offers-gained})
                counts.append(gained)
            gains=tuple(counts)
        for r in ('food','water'):
            p=getattr(self.supply_config,r+'_regeneration_probability')
            info[r+'_potential_expected']=len(tiles)*p
            info[r+'_eligible_expected']=info[r+'_eligible']*p
            info[r+'_capacity_censored_expected']=(len(tiles)-info[r+'_eligible'])*p
        self.last_supply=info
        return gains
