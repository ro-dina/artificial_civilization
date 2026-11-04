"""Optional actual consumption/survival outcomes and privileged ecology diagnostics."""
from collections import Counter,deque
from dataclasses import asdict
import csv,gzip,hashlib,io,json,math
from pathlib import Path

from simulation.controls_observer import ControlsObserver,SCHEMAS as BASE,paths as base_paths

ACQUISITIONS=('ecology','mode','sequence','tick','agent_id','own_index','base_id','b_token','c_token','action','exploratory',
 'x','y','post_x','post_y','birth_tick','generation','age','pre_hunger','pre_thirst','pre_energy','post_hunger','post_thirst','post_energy',
 'food_here','water_here','visible_individuals','nearest_visible_distance','nearest_visible_dx','nearest_visible_dy','remembered_visible_count',
 'food_at_visible_neighbor','water_at_visible_neighbor','food_distance_tickstart','water_distance_tickstart','food_near3_tickstart','water_near3_tickstart',
 'eligible_reproduction','food_consumed','water_consumed','acquired','action_succeeded','reward','terminal','death_cause')
PEOPLE=('agent_id','birth_tick','generation','founder','duration','censored','death_cause','opportunities','food','water','eat_attempts','drink_attempts',
        'acquisitions','reward','eligible_opportunities','reproduction_choices','successful_reproduction_choices')
RESOURCE_FIELDS=('stock','entropy','effective_cells','HHI','patch_share','eligible','attempted','realized','rejected','potential_expected','eligible_expected','capacity_censored_expected','mean_distance_tickstart','distance_observations','no_resource_opportunities')
SCHEMA_VERSION=2
SCHEMAS={**BASE,'acquisitions':ACQUISITIONS,'individuals':PEOPLE,
 'resource_ticks':('tick','population','patch_cells','agents_in_patches','patch_agent_share','vocal_q_rows','vocal_q_agents','mean_vocal_q_rows',*(r+'_'+f for r in ('food','water') for f in RESOURCE_FIELDS)),
 'spatial':('tick','x','y','patch','food','water','agents')}


def paths(path):
    p=Path(path);files=base_paths(p);stem=p.name[:-7] if p.name.endswith('.csv.gz') else p.stem
    suffix='.csv.gz' if p.suffix=='.gz' else '.csv'
    return {**files,**{k:p.with_name(stem+'.'+k+suffix) for k in SCHEMAS if k not in files}}


class ResourceWriter:
    def __init__(self,streams):
        self.writers={k:csv.DictWriter(s,fieldnames=SCHEMAS[k],extrasaction='ignore') for k,s in streams.items()};self.counts=Counter()
        for writer in self.writers.values():writer.writeheader()
    @classmethod
    def open(cls,stack,path):
        outputs=paths(path)
        if any(p.exists() for p in (*outputs.values(),Path(str(path)+'.metadata.json'))):raise ValueError('Refusing existing resource recording')
        streams={}
        for k,p in outputs.items():
            p.parent.mkdir(parents=True,exist_ok=True);raw=stack.enter_context(p.open('xb'))
            stream=stack.enter_context(gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0,compresslevel=1)) if p.suffix=='.gz' else raw
            streams[k]=stack.enter_context(io.TextIOWrapper(stream,encoding='utf8',newline=''))
        return cls(streams)
    def __call__(self,kind,row):self.writers[kind].writerow(row);self.counts[kind]+=1


def distance_map(world,resource):
    """Exact Chebyshev nearest available resource at this snapshot; None if absent."""
    w,h=world.width,world.height;values=[-1]*(w*h);queue=deque()
    for y,row in enumerate(world.tiles):
        for x,tile in enumerate(row):
            if getattr(tile,resource)>0:values[y*w+x]=0;queue.append((x,y))
    while queue:
        x,y=queue.popleft();d=values[y*w+x]+1
        for ty in range(max(0,y-1),min(h,y+2)):
            for tx in range(max(0,x-1),min(w,x+2)):
                i=ty*w+tx
                if values[i]<0:values[i]=d;queue.append((tx,ty))
    return values


def concentration(values,mask):
    total=sum(values)
    if not total:return {'stock':0,'entropy':None,'effective_cells':None,'HHI':None,'patch_share':None}
    probabilities=[v/total for v in values if v]
    h=-sum(p*math.log2(p) for p in probabilities)
    return {'stock':total,'entropy':h,'effective_cells':2**h,'HHI':sum(p*p for p in probabilities),
            'patch_share':sum(v for v,m in zip(values,mask) if m)/total}


class ResourceObserver(ControlsObserver):
    def __init__(self,sink,mode,ecology,geometry):
        super().__init__(sink,mode,False);self.ecology_label=ecology;self.geometry=geometry
        self.people={};self.supply_tick={};self.distances={};self.resource_totals=Counter();self.distance_totals=Counter();self.sim=None;self.last_spatial=None
    def bind(self,sim):self.sim=sim
    def _person(self,h):
        return self.people.setdefault(h.id,{'agent_id':h.id,'birth_tick':h.birth_tick,'generation':h.generation,'founder':int(h.parent_a_id is None),
            'opportunities':0,'food':0,'water':0,'eat_attempts':0,'drink_attempts':0,'acquisitions':0,'reward':0.,'eligible_opportunities':0,'reproduction_choices':0,'successful_reproduction_choices':0})
    def before_regeneration(self,sim):
        self.supply_tick={};self.distance_totals.clear();tiles=[t for row in sim.world.tiles for t in row]
        for r in ('food','water'):
            cap=getattr(sim.config,r+'_capacity');p=getattr(sim.config,r+'_regeneration_probability')*sim.resource_ecology.resource_supply_multiplier
            eligible=sum(cap is None or getattr(t,r)<cap for t in tiles)
            self.supply_tick.update({r+'_eligible':eligible,r+'_potential_expected':len(tiles)*p,r+'_eligible_expected':eligible*p,r+'_capacity_censored_expected':(len(tiles)-eligible)*p})
    def after_regeneration(self,sim,food,water):
        for r,n in [('food',food),('water',water)]:
            self.supply_tick.update({r+'_attempted':n,r+'_realized':n,r+'_rejected':0})
            self.distances[r]=distance_map(sim.world,r)
        if sim.resource_system is not None:self.supply_tick.update(sim.resource_system.last_supply)
    def begin(self,human,observation,tick):
        super().begin(human,observation,tick);sim=self.sim;person=self._person(human)
        tile=sim.world.tile_at(human.x,human.y);ids=getattr(observation,'individuals',())
        memory=getattr(human,'social_memory',None);target=self.target
        neighbor=sim.world.tile_at(human.x+target.dx,human.y+target.dy) if target else None
        self.pre={'own_index':person['opportunities'],'x':human.x,'y':human.y,'birth_tick':human.birth_tick,'generation':human.generation,'age':human.age,
            'pre_hunger':human.hunger,'pre_thirst':human.thirst,'pre_energy':human.energy,'food_here':tile.food,'water_here':tile.water,
            'visible_individuals':len(ids),'nearest_visible_distance':max(abs(target.dx),abs(target.dy)) if target else None,
            'nearest_visible_dx':target.dx if target else None,'nearest_visible_dy':target.dy if target else None,
            'remembered_visible_count':sum(memory.get(i.individual_id) is not None for i in ids) if memory else 0,
            'food_at_visible_neighbor':neighbor.food if neighbor else None,'water_at_visible_neighbor':neighbor.water if neighbor else None,
            'eligible_reproduction':int(sim.reproduction.eligible(human,tick))}
        for r in ('food','water'):
            d=self.distances[r][human.y*sim.world.width+human.x];self.pre[r+'_distance_tickstart']=d if d>=0 else None
            self.pre[r+'_near3_tickstart']=int(0<=d<=3)
            self.distance_totals[r+'_distance_observations']+=d>=0
            self.distance_totals[r+'_distance_sum']+=max(0,d);self.distance_totals[r+'_no_resource_opportunities']+=d<0
    def resource_outcome(self,human,decision,cause,food,water,success):
        if not math.isfinite(human.brain.last_reward):raise RuntimeError('Nonfinite physiological reward')
        p=self._person(human);p['opportunities']+=1;p['food']+=food;p['water']+=water;p['acquisitions']+=bool(food or water)
        p['eat_attempts']+=decision.action.value=='eat';p['drink_attempts']+=decision.action.value=='drink';p['reward']+=human.brain.last_reward
        p['eligible_opportunities']+=self.pre['eligible_reproduction'];p['reproduction_choices']+=decision.action.value=='reproduce'
        p['successful_reproduction_choices']+=decision.action.value=='reproduce' and success
        self.resource_totals.update(food=food,water=water,acquisitions=int(bool(food or water)),eat_attempts=int(decision.action.value=='eat'),drink_attempts=int(decision.action.value=='drink'))
        self.sink('acquisitions',{'ecology':self.ecology_label,'mode':self.condition,**self.turn,**self.pre,
            'post_x':human.x,'post_y':human.y,'post_hunger':human.hunger,'post_thirst':human.thirst,'post_energy':human.energy,
            'food_consumed':food,'water_consumed':water,'acquired':int(bool(food or water)),'action_succeeded':int(success),
            'reward':human.brain.last_reward,'terminal':int(cause is not None),'death_cause':cause})
        if cause is not None:self._close_person(human,cause,False)
    def _close_person(self,human,cause,censored):
        person=self._person(human);self.sink('individuals',{**person,'duration':human.age,'censored':int(censored),'death_cause':cause});self.people.pop(human.id,None)
    def snapshot(self,sim):
        counts=Counter((h.x,h.y) for h in sim.agents)
        for y,row in enumerate(sim.world.tiles):
            for x,t in enumerate(row):self.sink('spatial',{'tick':sim.tick,'x':x,'y':y,'patch':int(self.geometry.mask[y*sim.world.width+x]),'food':t.food,'water':t.water,'agents':counts[x,y]})
        self.last_spatial=sim.tick
    def tick_end(self,sim):
        super().tick_end(sim)
        tiles=[t for row in sim.world.tiles for t in row];npatch=sum(self.geometry.mask);inside=sum(self.geometry.mask[h.y*sim.world.width+h.x] for h in sim.agents)
        row={'tick':sim.tick,'population':sim.population,'patch_cells':npatch,'agents_in_patches':inside,'patch_agent_share':inside/sim.population if sim.population else None,**self.supply_tick}
        vocal=[h.brain.vocal_memory_size for h in sim.agents if hasattr(h.brain,'vocal_memory_size')]
        row.update(vocal_q_rows=sum(vocal),vocal_q_agents=len(vocal),mean_vocal_q_rows=sum(vocal)/len(vocal) if vocal else None)
        for r in ('food','water'):
            values=[getattr(t,r) for t in tiles];metrics=concentration(values,self.geometry.mask)
            expected=getattr(sim.config,'initial_'+r)+getattr(sim.statistics,r+'_regenerated')-getattr(sim.statistics,r+'_consumed')
            if sum(values)!=expected:raise RuntimeError('Resource conservation failed: '+r)
            cap=getattr(sim.config,r+'_capacity')
            if any(v<0 or cap is not None and v>cap for v in values):raise RuntimeError('Resource capacity failed')
            row.update({r+'_'+k:v for k,v in metrics.items()})
            n=self.distance_totals[r+'_distance_observations'];row[r+'_mean_distance_tickstart']=self.distance_totals[r+'_distance_sum']/n if n else None
            row[r+'_distance_observations']=n;row[r+'_no_resource_opportunities']=self.distance_totals[r+'_no_resource_opportunities']
        self.sink('resource_ticks',row)
        if sim.tick%100==0:self.snapshot(sim)
    def finish(self,sim):
        for h in sim.agents:self._close_person(h,None,True)
        if self.last_spatial!=sim.tick:self.snapshot(sim)
        super().finish(sim);self.sim=None
