# Artificial Civilization Simulation — v0.2

A headless, standard-library-only artificial-life simulation for experiments in
emergent behavior. v0.2 extends the v0.1 grid, local perception, random brains,
survival, and meaningless signals with reproduction, inherited traits, mutation,
renewable resources, and lightweight lineage tracking.

**Agents still do not learn. Signals have no predefined meaning, and RandomBrain
does not intentionally communicate. Agriculture, economics, emotions, religion,
technology, crafting, combat, culture, and civilization are not implemented.**
Renewable food is an environmental process, not farming. Higher-level phenomena
remain research questions; there are no scripted civilization stages or outcomes.

## Run and test

Use Python **3.11+** from the repository root. No installation or network access is
required. The existing `.venv/bin/python` can replace `python3` in these commands.

```bash
# Legacy defaults preserve v0.1 dynamics, including unchanged energy.
python3 main.py --mode legacy --seed 42 --ticks 1000 --csv data/legacy.csv

# Evolution preset: reproduction, energy metabolism, mutation, regeneration.
python3 main.py --mode evolution --seed 42 --ticks 1000 --csv data/evolution.csv

# Detailed logs are opt-in; start with a short run.
python3 main.py --mode evolution --seed 42 --ticks 50 --events data/events.jsonl
python3 main.py --help
python3 -m unittest discover -s tests -v
```

`python3 main.py` and `Config()` retain **legacy defaults**. Select evolution
explicitly with `--mode evolution` or `Config.evolution()`. Other CLI options are
`--population` and `--status-every`. Both modes stop at extinction or the requested
tick limit. Output destinations are overwritten only when explicitly supplied;
CSV and events must use different paths. Generated files under `data/` are ignored
by Git. No GUI, rendering, multiprocessing, or automatic detailed history exists.

The 23 original tests are unchanged. Additional tests cover pairing, eligibility,
costs, cooldown, deferred births, fresh brains, lineage, mutation, capped resources,
safeguards, death causes, full evolutionary replay, and logging independence.

## Architecture

```text
main.py                    CLI, modes, output ownership, stopping conditions
config.py                  Validated immutable Config, presets, trait limits
world/tile.py              Integer food and water units
world/world.py             Grid, generation, movement, consumption, regeneration
agents/human.py            Body, physiology, genome reference, lineage metadata
agents/genome.py           Immutable Genome, inheritance, bounded mutation
agents/brain.py            Brain protocol, Action, Decision, RandomBrain
agents/observation.py      Immutable local sensory records
systems/communication.py   Meaning-free signals with spatial lookup
systems/reproduction.py    Eligibility, local partner lookup, costs, cooldown
simulation/simulation.py   Seeded setup, tick loop, births, aggregate accounting
simulation/statistics.py   Latest aggregate snapshot and streaming CSV writer
simulation/events.py       Optional generic event sink and JSONL writer
tests/test_basic.py         Original v0.1 tests (unchanged)
tests/test_evolution.py     v0.2 tests
```

The existing architecture is extended rather than replaced. A brain implements
`choose_action(human, observation) -> Decision`. It reads its own body and local
observation and returns a decision without mutating the body. Observations contain
`tick`, nearby `(dx, dy, food, water)` records, and heard
`(signal_id, sender_id, dx, dy)` records. They contain no mutable world references
or distant resource information. This remains a programming contract, not a
sandbox for untrusted custom brains.

RandomBrain uniformly chooses among the original seven actions in legacy mode,
or those actions plus `REPRODUCE` when reproduction is enabled. It does not pursue
resources or mates intelligently. Signalling is an independent random draw using
the agent's inherited probability. An optional `brain_factory(rng, config)` passed
to `Simulation`/`create_default` constructs a fresh controller for every founder
and child. Brain memory, Python object state, and parents' RNG states are never
inherited. Replacing only one agent's brain does not change the simulation's
factory for later children.

## Physical rules and tick timing

Coordinates are integer cells. North decreases `y`; movement is one cardinal
step with hard boundaries, no wrapping, and allowed overlap. Perception, hearing,
and reproduction use inclusive square neighborhoods (Chebyshev distance). A zero
radius includes only the current cell.

Each nonempty tick proceeds as follows:

1. Regenerate resources before any agent observes.
2. Shuffle the existing agents using the dedicated ordering stream. Build a local
   partner index in that order.
3. For each existing agent: observe, choose, execute, optionally emit a signal,
   increase hunger/thirst, subtract the energy tick cost, age, and check death.
   A reproduction action may charge two parents and queue one newborn.
4. Remove dead agents, append queued newborns, and discard the partner index.
   Emit optional birth records and advance the one-tick signal buffer.
5. Update aggregate statistics and check population accounting.

Newborns have age zero and **do not act, age, pay metabolism, or reproduce during
their birth tick**. They become active next tick. Queued births survive a parent's
later death during the same tick. No dead Human objects are retained for genealogy.
An extinct simulation's `step()` is a no-op, including environmental updates.

Actions are sequential. Later agents see earlier consumption and movement.
Eligibility uses a partner's current state, including any aging or eating earlier
in the phase. This is a modeling choice, not simultaneous action resolution.

Eating or drinking consumes one available unit on the current tile. It reduces
the corresponding need by its configured relief, clamped at zero; food also
restores energy. A blocked move, failed reproduction, or absent resource wastes
the action, and the usual tick costs still apply. Random agents may waste food
or water while their needs are already zero.

Signals remain integers in `[0, signal_vocab_size)`. They originate at the
sender's post-action position, are heard only on the next tick, then expire.
Receivers cannot hear themselves. The signal persists for that tick if its sender
dies. There is no signal cost or built-in semantic mapping; vocabulary sizes such
as 4, 16, 256, and 4000 need no architectural change.

## Reproduction and energy

Types `0` and `1` are compatibility categories only. There is no pregnancy,
gestation, relationship, family, gender, kin-recognition, or incest restriction.
One agent initiates `REPRODUCE`; its partner need not independently select that
action. Both must currently satisfy all these conditions:

- Alive, at least `min_reproductive_age` (20 by default), and at most the optional
  `max_reproductive_age` (disabled by default).
- Out of cooldown, of different reproductive types, and within
  `reproduction_radius` (default 1). Self-pairing is excluded.
- Hunger and thirst below their death limits and each at most
  `reproduction_need_fraction` (0.75) of that limit.
- Energy at least both `min_reproduction_energy` (35) and
  `reproduction_energy_cost` (30). When zero energy is lethal, payment must leave
  a strictly positive reserve in both parents.

A seeded uniform choice selects among eligible local partners. Spatial buckets
are visited in coordinate order; their dictionaries use Python's guaranteed
insertion order, built from the shuffled acting order and updated on movement.
There is no unordered set traversal or permanent preference for low agent IDs.

Success costs **each parent** 30 energy by default, queues one child, and sets
both parents' next eligible tick to
`current_tick + max(1, reproduction_cooldown)`. Thus even cooldown zero permits
only one successful participation per parent per tick. Failed attempts charge no
reproduction energy and set no cooldown, but still pay ordinary metabolism.

In the evolution preset, food restores 20 energy, capped at `max_energy=100`;
each tick costs 0.2 energy, clamped at zero; and zero energy is lethal. Reproduction
can therefore shorten survival as well as limit fertility. Hunger, thirst, and
energy are separate: energy does not erase thirst or hunger, and drinking does
not supply energy. These values are configurable independently. Legacy defaults
use no food energy gain, no tick energy cost, and nonlethal zero energy.

Death priority is deterministic: **starvation, dehydration, old_age, energy**.
Hunger/thirst kill at `>=` their limit; optional maximum age kills at `>= max_age`.
`max_age=None` disables age deaths; `energy_depletion_lethal=False` disables
energy deaths. These checks occur after the agent's action and tick costs.

A child appears on the initiating parent's cell, receives the configured
`offspring_hunger=0`, `offspring_thirst=0`, and `offspring_energy=40`, and gets a
seeded random reproductive type. Birth energy is an abstract configured endowment,
not a physically conserved transfer formula; defaults charge parents a combined
60 energy for a 40-energy child. Changing these parameters changes that balance.

IDs increase monotonically and are never reused, including after death. Founders
have `parent_a_id=None`, `parent_b_id=None`, `generation=0`, and `birth_tick=0`.
Children record both parent IDs, the current birth tick, and
`generation = max(parent_a.generation, parent_b.generation) + 1`.

## Genome, inheritance, and mutation

All founders start with an identical immutable genome. There is no initial
variation option in v0.2. The four traits are:

| Trait | Founder value | Permitted values | Effect |
| --- | --- | --- | --- |
| `perception_radius` | `Config.perception_radius` (2) | Integers 0–8 | Local resource observation radius |
| `signal_probability` | `Config.signal_probability` (0.1) | 0–1 | Random emission probability |
| `hunger_multiplier` | 1 | 0.5–1.5 | Multiplies the environmental hunger cost |
| `thirst_multiplier` | 1 | 0.5–1.5 | Multiplies the environmental thirst cost |

`GENOME_BOUNDS` in `config.py` defines experiment constraints, not inherited
state. Baseline configuration and genomes are validated against these bounds.
Base hunger/thirst rates may be zero for controlled experiments; inherited
multipliers themselves are strictly positive. Death limits, maximum age, resource
capacities, and reproductive eligibility limits remain environmental settings.

For **each trait independently**, inheritance chooses one parent's value with
probability 1/2. Then, with `mutation_rate=0.05` by default, it applies mutation:

- Continuous traits receive a uniform additive change in
  `[-mutation_strength * trait_span, +mutation_strength * trait_span]`.
- Integer perception receives a uniform integer change between
  `-ceil(mutation_strength * trait_span)` and the positive equivalent.
- The result is clamped to the trait's bounds. A selected mutation may have no
  effect because of a zero integer step or clamping.

Default `mutation_strength=0.05` is a fraction of each trait's allowed span, not
of its parental value. Both rate and strength must be in `[0, 1]`. Any positive
strength allows an integer step of at least
one when the perception mutation is selected. Setting mutation rate **or**
strength to zero leaves pure inheritance and consumes no mutation-stream draws.
There are no chromosomes, dominance rules, learned weights, or inherited brains.

## Renewable resources and population

Every tile is eligible for environmental renewal; there are no plants, rainfall,
seasons, geography, or farming. At tick start, each non-full tile independently
has the configured probability of gaining **one** food unit and similarly one
water unit. Food and water use separate random streams. Full tiles consume no
regeneration draws. Added units are immediately available that tick.

The evolution preset uses probability 0.02 per resource per tile per tick, with
capacity 5 units of each resource per tile. Initial capped placement chooses
uniformly among non-full cells until the configured total is placed. Impossible
initial totals are rejected instead of silently dropping resources. Regeneration
requires a finite capacity; it never exceeds it.

Set both regeneration probabilities to zero to disable renewal completely.
Legacy defaults also leave capacities `None`, preserving the original placement
with replacement and exact resource totals, even when totals exceed cell count.
Finite stored units are preserved; `None` is not an infinite resource supply.

There is **no population cap by default**, in either mode. Availability, costs,
reproduction, and mortality control population. Optional `max_population` is a
computational safeguard only. A viable attempt blocked by it increments
`population_limit_blocks` and pays no reproductive cost. Its check counts current
survivors and pending births; deaths already processed free space, but future
deaths in the same tick are not anticipated. Initial population above a configured
safeguard is rejected.

## Configuration and deterministic streams

```python
from config import Config
from simulation.simulation import Simulation

settings = Config.evolution(random_seed=42, max_ticks=1000, mutation_rate=0.05)
sim = Simulation(settings)
for _ in range(settings.max_ticks):
    if sim.population == 0:
        break
    sim.step()
sim.print_status()
```

Use `Config(...)` for legacy defaults, `Config.evolution(**overrides)` for the
preset, or `dataclasses.replace` to derive independent configurations. The mode
is a preset, not a hidden switch: reproduction, metabolism, and renewal can each
be configured independently. Disabling reproduction and renewal in an evolution
config does **not** automatically disable its energy metabolism; use `Config()`
for exact legacy dynamics. `create_default(seed=..., config=...)` still accepts
seed zero and overrides the supplied config's seed. `max_ticks` and
`status_interval` belong to the runner; `step()` imposes no run-length limit.

All randomness derives from the simulation seed through named `random.Random`
instances. Original streams remain `world`, `spawn`, `order`, and `brain:<id>`.
New independent streams are `partners`, `offspring`, `inheritance`, `mutation`,
`regeneration:food`, and `regeneration:water`. Births never consume founder-placement
randomness. Each child receives a fresh `brain:<new_id>` stream. There is no use
of global random state, wall clocks, or Python hash randomization in dynamics.
Logging and statistics consume no randomness.

Identical configuration, seed, source code, and Python version reproduce the
same outcomes. Record all four for experiments; replay across future code or
Python versions is not promised. Legacy seed 42 was compared against a v0.1
snapshot at **every tick**, including original body fields, tile resources, and
statistics, through extinction at tick 309. Original dynamics match exactly.
CSV has additional columns, events have additional fields, and status output has
additional counters; their textual formats intentionally extend v0.1.

## Statistics and events

The simulation retains only its latest immutable `Statistics` snapshot. CSV
streams tick zero and each completed tick. All columns remain scalar integers:

| Fields | Semantics |
| --- | --- |
| `tick`, `population` | Current completed tick and living population |
| `births`, `deaths` | Cumulative counts; founders are not births |
| `highest_generation` | Highest generation ever born, retained after death/extinction |
| `reproduction_attempts` | All selected REPRODUCE actions, including failures |
| `successful_reproductions` | Cumulative successful pairings; equals births in v0.2 |
| `population_limit_blocks` | Otherwise viable attempts blocked by the safeguard |
| `food_consumed`, `water_consumed`, `signals_emitted` | Cumulative actual units/emissions |
| `food_regenerated`, `water_regenerated` | Cumulative actual additions, excluding initial placement |
| `starvation_deaths`, `dehydration_deaths`, `old_age_deaths`, `energy_deaths` | Cumulative deaths by the priority-selected cause |

Subtract adjacent CSV rows for per-tick counts. An accounting assertion checks
`initial_population + births - deaths == population`. Tests also check unique IDs,
nonnegative energy/resources, bounds, and resource conservation. Mean trait values
are not computed every tick; experiments can sample living `human.genome` values
without adding a permanent population-wide analysis pass to the loop.

An optional callable `event_sink` receives records; `--events` uses the existing
streaming JSONL writer. Payloads are built only when logging is enabled:

- `run_started`: effective config and the identical founder genome.
- `agent_step`: existing start position, before/after body state, local observation,
  action/success, emissions, consumption, death cause, plus a reproduction failure
  reason when applicable. Partner costs may change a body after its own step
  record; birth records identify both charged parents.
- `birth`: emitted after the acting phase, with `agent_id`/`details.child_id`, tick,
  `x`/`y`, both parent IDs, generation, complete `child_genome`, reproductive type,
  and initial needs/energy. Lineage can be reconstructed without retaining dead
  bodies. The inherited trait chosen before mutation and zero-effect mutation
  attempts are not separately logged.

Custom sinks can retain only births or sample observations. The default
simulation stores neither events nor past bodies/genomes. CSV and JSONL writers
use caller-owned streams and do not retain rows.

## Baselines, performance, and scientific limitations

Seed 42, default dimensions/population, and 1,000 requested ticks:

| Mode | Final tick | Population | Births | Deaths | Highest generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy | 309 (extinct) | 0 | 0 | 100 | 0 |
| Evolution | 1,000 | 869 | 6,114 | 5,345 | 31 |

The legacy run consumed 361 food and 389 water units and emitted 1,425 signals,
matching v0.1. Evolution used the preset as defined above, with no safety cap or
parameter tuning to force survival. These are single-seed smoke tests, not claims
of ecological stability or guaranteed indefinite survival. Extinction remains a
valid outcome and is explicitly tested even with reproduction and renewal enabled.

The two 1,000-tick evolution runs matched every captured tick (bodies/genomes,
resources, signals, IDs, and statistics), all final RNG states, and CSV bytes.
The second run enabled an event sink that retained only run metadata and births;
3,935 newborns had genomes differing from the founder baseline. No events were
enabled in the first run. The peak population was 1,470. Population, death-cause,
resource-conservation, and trait-bound checks passed throughout both runs.

On the local Python 3.11 environment, simulation steps took approximately 12.0
seconds without events and 13.4 seconds with the birth-filtering sink. Full-state
hashing and invariant checks increased total validation time to 18.2 and 19.7
seconds respectively. These are indicative single-machine measurements, not
portable throughput guarantees. Local artifacts are `data/v02_legacy.csv`,
`data/v02_evolution_1.csv`, `data/v02_evolution_2.csv`, `data/v02_births.jsonl`, and
`data/v02_verification.json` (ignored generated files, not required to run).

Memory scales with the world, living population, pending births, and recent
signals, rather than elapsed ticks. Perception allocates local snapshots, resource
renewal scans the grid when enabled, and partner lookup searches nearby buckets.
Very dense local clusters can still make pairing expensive. Detailed serialization
and large inherited perception radii add cost. No claim of million-tick or
thousand-run throughput is made without further measurement.

The ecology is deliberately abstract. All tiles can regenerate, related agents
can pair, births are immediate, and reproductive types are compatibility bits.
There is no penalty balancing lower metabolic multipliers, so selection can favor
the configured lower bounds; this is not a validated biological fitness model.
RandomBrain ignores observed resources and signals, so perception/signalling
variation need not improve fitness. Multiple generations and genetic change do
not demonstrate intelligence, adaptation to every trait, or emergent language.

A sensible v0.3 direction is a reproducible experiment runner with replicated
seeds, sampled trait distributions, extinction/survival measures, and measured
performance. Use it to assess ecological sensitivity and explicit trait tradeoffs
before introducing learning or claiming adaptive communication. Reproduction,
objects, sensory systems, and controllers remain separate extension points; none
should directly script agriculture, institutions, or other civilization outcomes.
