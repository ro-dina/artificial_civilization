# Artificial Civilization Simulation — v0.3

A headless, standard-library-only artificial-life simulation for experiments in
emergent behavior. v0.3 adds optional individual memory and experience-based
learning to the existing survival, reproduction, inheritance, mutation, renewable
resources, and lineage systems. The experiment is whether local sensory
experience and physiological consequences can produce useful acquired behavior.
The default RandomBrain remains the unchanged v0.2 control.

**v0.3 does not implement language learning or meaningful communication.** Only
LearningBrain learns; signals have no predefined meaning and neither controller
intentionally communicates. Agriculture, economics, emotions, religion,
technology, crafting, combat, culture, and civilization are not implemented.
Renewable food is an environmental process, not farming. Higher-level phenomena
remain research questions; there are no scripted civilization stages or outcomes.

## Run and test

Use Python **3.11+** from the repository root. The simulation needs no installation
or network access. CSV analysis has separate optional dependencies, described
below. The existing `.venv/bin/python` can replace `python3` in these commands.

```bash
# Legacy defaults preserve v0.1 dynamics, including unchanged energy.
python3 main.py --mode legacy --seed 42 --ticks 1000 --csv data/legacy.csv

# Evolution preset: reproduction, energy metabolism, mutation, regeneration.
python3 main.py --mode evolution --seed 42 --ticks 1000 --csv data/evolution.csv

# Same ecology, different controllers. Random remains the default.
python3 main.py --mode evolution --brain random --seed 0 --ticks 1000 --csv data/v03_random_seed_0.csv
python3 main.py --mode evolution --brain learning --seed 0 --ticks 1000 --csv data/v03_learning_seed_0.csv
python3 main.py --mode evolution --brain learning --seed 0 --ticks 5000 --csv data/v03_learning_seed_0_5000.csv

# Hold perception fixed in every generation while the other traits evolve.
python3 main.py --mode evolution --brain learning --fixed-perception-radius 2 --seed 0 --ticks 5000 --csv data/learning_fixed_p2_seed_0.csv

# Controlled repeated experience, independent of evolutionary claims.
python3 -m experiments.learning_demo

# Detailed logs are opt-in; start with a short run.
python3 main.py --mode evolution --seed 42 --ticks 50 --events data/events.jsonl
python3 main.py --help
python3 -m unittest discover -s tests -v
```

`python3 main.py` and `Config()` retain **legacy defaults**. Select evolution
explicitly with `--mode evolution` or `Config.evolution()`. Other CLI options are
`--population` and `--status-every`. Both modes stop at extinction or the requested
tick limit. Output destinations are overwritten only when explicitly supplied;
CSV, its generated `.metadata.json` sidecar, and events must use different paths.
Generated files under `data/` are ignored
by Git. No GUI, rendering, multiprocessing, or automatic detailed history exists.

All 72 pre-v0.3 tests remain unchanged and pass. The 25 new tests cover exact
Q updates and reward, real observation timing, bounded independent memory,
fresh offspring, local sensing, exploration, signal isolation, complete replay,
logging/statistics independence, accounting, extinction, CSV metadata, a stored
pre-v0.3 control fixture, and a controlled demonstration of learned preference.
The fixed-perception extension adds eight tests for a total of 105 simulation tests, including
multi-generation constraints, unchanged other-trait RNG draws, and CLI validation.

## Architecture

```text
main.py                    CLI, modes, output ownership, stopping conditions
config.py                  Validated immutable Config, presets, trait limits
world/tile.py              Integer food and water units
world/world.py             Grid, generation, movement, consumption, regeneration
agents/human.py            Body, physiology, genome reference, lineage metadata
agents/genome.py           Immutable Genome, inheritance, bounded mutation
agents/brain.py            Brain protocol, Action, Decision, RandomBrain
agents/learning.py         Bounded individual Q-learning and outcome snapshots
agents/observation.py      Immutable local sensory records
systems/communication.py   Meaning-free signals with spatial lookup
systems/reproduction.py    Eligibility, local partner lookup, costs, cooldown
simulation/simulation.py   Seeded setup, tick loop, births, aggregate accounting
simulation/statistics.py   Latest aggregate snapshot and streaming CSV writer
simulation/events.py       Optional generic event sink and JSONL writer
tests/test_basic.py         Original v0.1 tests, with extended CSV schema support
tests/test_evolution.py     v0.2 tests
tests/test_statistics.py    Living-genome summaries and observation-only verification
tests/test_learning.py      v0.3 learning, isolation, replay, and control regression
tests/test_fixed_perception.py Fixed-radius experiments, inheritance, and CLI validation
tests/fixtures/             Pre-v0.3 deterministic control digest
experiments/learning_demo.py Controlled repeated-resource learning validation
experiments/analyze_fixed_perception.py Offline CSV aggregation and PNG plots
tests/test_analyze_fixed_perception.py Analysis fixtures, missing data, and reproducibility
requirements-analysis.txt   Optional pandas/matplotlib dependencies
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

LearningBrain uses the same decision interface. After the existing action and
physiology/death check, the controller calls an optional
`observe_outcome(Physiology)` hook with immutable hunger, thirst, energy, and
liveness values. It supplies no World or extra observation. A living learner
updates the previous experience when its next ordinary observation arrives;
terminal experiences update immediately. Custom brains without this hook retain
their existing contract. The only ordering addition is this private learning
feedback; biological action, reproduction, resource, and signal ordering is
unchanged.

## Individual memory and learning

`LearningBrain` uses tabular **Q-learning**, with no pretrained values or
resource-seeking action rules. All unseen state/action values start at zero.
Its state consists of:

- Hunger, thirst, and energy, each normalized by its configured limit and clipped
  into `learning_need_bins=3` bins.
- The direction of the nearest visible food tile and nearest visible water tile.
  Nearest means Manhattan distance within the existing square perception window;
  ties use `(dy, dx)`. Each direction is `(sign(dx), sign(dy))`, including `(0, 0)`
  for the current cell. `(2, 2)` denotes no visible resource of that type.

The encoder reads only the immutable Observation and the individual's body.
Resources outside the inherited perception radius cannot affect these features.
It ignores exact resource quantities beyond presence, exact distances after
nearest selection, absolute position, tick, IDs, lineage, reproductive partners,
and all heard signals. It has no world reference, map, pathfinding, or predefined
direction-to-action preferences. Larger perception can supply additional local
information, but receives no reward bonus or extra metabolic cost.

Memory is an individual `OrderedDict` from state tuples to action-value lists,
bounded to `learning_memory_capacity=256` states. Each row has seven action values,
or eight when reproduction is enabled. Access during learning/action selection
updates recency; the least recently used row is evicted at capacity. One pending
transition stores the preceding state, action, body snapshot, and reward. No
lifetime history is retained. Diagnostic reads do not allocate rows or consume
randomness. **Every newborn starts with empty learned memory and zero updates.**
Only the unchanged four genome traits are inherited; learning rate, exploration,
memory capacity, and learned values are not heritable.

For each action, let `before` be the body just before choosing and `after` be the
body after that action and its ordinary physiology/death check. The exact reward
is:

```text
r = (hunger_before - hunger_after) / max_hunger
  + (thirst_before - thirst_after) / max_thirst
  + (energy_after - energy_before) / max_energy
```

This uses actual physiological changes, including ordinary costs. There is no
reward for seeing a resource, a specific action label, birth, perception radius,
or signal, and no additional survival/death bonus. A partner's passive reproduction
charge outside this individual's action interval is not retroactively included
in that reward; it is reflected in the next observed body state. The model does
not provide perfect causal attribution for interactions between turns.

At the next real decision, update the preceding transition using:

```text
Q(s, a) += learning_rate * (r + learning_discount * max_a' Q(s_next, a') - Q(s, a))
```

Default `learning_rate=0.2`, `learning_discount=0.9`. Unseen next states have value
zero; death bootstraps with zero. The simulator never obtains an extra future
observation for learning. A surviving individual's last transition remains
pending at an arbitrary run cutoff, so it is not counted as an update yet.

Action selection is fixed epsilon-greedy: with `learning_epsilon=0.2`, choose
uniformly among all available actions; otherwise choose uniformly among the
highest-valued tied actions. Epsilon and learning rate must be in `(0, 1]`;
discount is in `[0, 1]`. Exploration remains enabled throughout life. Reproduction
is an ordinary available action, with its unchanged physiological costs and no
fitness reward. Signalling continues as independent random emission according
to the inherited probability, and heard signals do not enter the learning state.

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
variation option. The four traits are unchanged from v0.2:

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
There are no chromosomes, dominance rules, or inherited learned weights/brains.

### Fixed-perception experiments

`--fixed-perception-radius N` sets the **actual genome trait** to `N` for every
founder and every newborn, in all generations. Valid values are integers **0–8**;
zero means only the current tile. The option works with either brain and either
ecological preset. Programmatically, use
`Config.evolution(brain="learning", fixed_perception_radius=2)`.

The default is `None` (CLI option omitted): perception retains its existing
inheritance/mutation behavior, with unchanged seeded simulation trajectories.
`Config.perception_radius` alone sets the founder baseline; the new fixed option
overrides that baseline and constrains all offspring too. Other genome traits
continue to inherit and mutate normally; learning, reproduction, physiology,
resource dynamics, and signal rules are unchanged.

For each birth, the normal inheritance and mutation draws are still performed,
then the child's stored perception radius is replaced by the fixed value. This
preserves RNG consumption for the other traits instead of shifting their draws
by skipping a trait. Fixed and evolving experiments can still diverge as changed
perception changes learning, behavior, births, and deaths.

CSV `mean_perception_radius`, `min_perception_radius`, and
`max_perception_radius` all equal `N` while the population is nonempty; extinction
still produces empty cells. Birth events record the fixed child trait. The CSV's
`.metadata.json` sidecar includes `config.fixed_perception_radius`, and `config_id`
distinguishes these experimental settings. Adding this config field changes
configuration IDs even for the default `None`; it does not change default dynamics.

The example above was verified through tick 5,000: population 899, 29,812 births,
29,013 deaths, and highest generation 146. All 5,001 CSV snapshots (including tick
zero) have perception mean/min/max equal to 2, while the other three traits vary.
This single-seed check validates the constraint, not its ecological superiority.

### Analyze fixed-perception CSVs

This is offline post-processing only: it reads existing CSVs without importing or
running simulation code, changing Config/RNG, or modifying the input files.
Install pandas and matplotlib separately; seaborn is not used:

```bash
python3 -m venv .venv  # Only needed if a virtual environment does not exist yet.
source .venv/bin/activate
python3 -m pip install -r requirements-analysis.txt
python3 -m experiments.analyze_fixed_perception

# Direct invocation and custom locations also work:
python3 experiments/analyze_fixed_perception.py --data-dir data --output-dir analysis/fixed_perception
python3 -m unittest discover -s tests -v
```

Inputs are `data/learning_fixed_p*_seed_*.csv`. Radius and seed come from the
filename; new radii and seed numbers need no code changes. Runs are sorted
numerically and each condition's seed count/list is printed. Duplicate run IDs
or duplicate ticks are errors. The three required metric columns are exactly
`population`, `mean_memory_size`, and `mean_learning_updates`, plus `tick`.
Missing optional final metrics produce warnings and unavailable values (`n=0`
if absent from every run); invalid numeric measurements warn and become NaN.

Results in `analysis/fixed_perception/` are overwritten on each execution:

- `population_over_time.png`, `memory_size_over_time.png`, and
  `learning_updates_over_time.png`: seed means with ±1 sample SD bands.
- `final_population.png`: each seed's final value and mean ±1 SD, with deterministic
  horizontal offsets. `death_causes_final.png` compares three selected cumulative
  death causes; `traits_over_time.png` shows three evolving traits in separate panels.
- `final_metrics.csv`: tidy `perception,metric,mean,std,min,max,n` from each run's
  **maximum recorded tick**. NaNs in that final row are preserved.
- `run_summary.csv`: one row per run, including filename, seed, final tick,
  population, counters, and available living-population means.
- `time_series_metrics.csv`: `perception,tick,metric,mean,std,min,max,n` for all
  six plotted time-series metrics. `analysis_metadata.json` records input hashes,
  software versions, and aggregation/missing-data rules.

Every run has equal weight, including when its population differs. SD uses
`ddof=1`; `n` counts nonmissing values **per metric**, not agents. Missing ticks
are neither interpolated nor carried forward, even after extinction: later
averages may therefore represent fewer runs. Inspect `n` when grids differ.
Population zero remains a valid observation; living-genome and living-learner
means are undefined at extinction and excluded. Empty CSV cells represent NaN;
`n<2` gives undefined SD and no SD band/error bar. Different final ticks warn
because endpoints then cover different durations.

Plots use a headless backend, 180 dpi, and no random jitter. Identical inputs and
software environment reproduce identical CSV/PNG bytes. The optional analysis
tests cover synthetic unequal tick grids, NaNs/extinction, sample SD, file order,
both entrypoints, and reproducible output; they skip if dependencies are absent.
Memory size measures currently retained Q-table states in living learners, not
all distinct states ever experienced. The script performs no significance tests
and does not label any perception radius optimal.

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

settings = Config.evolution(random_seed=42, max_ticks=1000, brain="learning")
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

LearningBrain uses its private `brain:<id>` stream for exploration and tied
choices. At construction it derives a separate emission RNG from one 64-bit draw
of that stream. Changing signal probability or vocabulary therefore cannot shift
its future action-choice draws. RandomBrain's original RNG consumption is
untouched. The two controllers consume randomness differently: their individual
trajectories are expected to differ even under the same seed. Changes in behavior
can also change later population, resource, and partner draws; separate streams
do not imply matched outcomes across experimental conditions.

Identical configuration, seed, source code, and Python version reproduce the
same outcomes. Record all four for experiments; replay across future code or
Python versions is not promised. Legacy seed 42 was compared against a v0.1
snapshot at **every tick**, including original body fields, tile resources, and
statistics, through extinction at tick 309. Original dynamics match exactly.
CSV has additional columns, events have additional fields, and status output has
additional counters; their textual formats intentionally extend v0.1.

## Statistics and events

The simulation retains only its latest immutable `Statistics` snapshot. CSV
streams tick zero and each completed tick. Existing ecological columns remain
integers; genome and learning means contain numeric values or empty cells:

CLI CSV files prefix every row with `mode`, `brain`, `seed`, and `config_id`.
`<output>.csv.metadata.json` records the full effective Config, Python version,
release version, and the same ID. The ID is SHA-256 of sorted JSON for the full
configuration, including runner settings; it is not a source-code hash. Retain
the code revision alongside results. Programmatic `CSVStatisticsWriter(stream)`
still emits statistics alone; optional `metadata=...` adds identifying columns
and cannot overwrite metrics. Existing metric names and meanings are preserved,
but consumers should read named columns rather than fixed positions.

| Fields | Semantics |
| --- | --- |
| `tick`, `population` | Current completed tick and living population |
| `births`, `deaths` | Cumulative counts; founders are not births |
| `highest_generation` | Highest generation ever born, retained after death/extinction |
| `reproduction_attempts` | All selected REPRODUCE actions, including failures |
| `successful_reproductions` | Cumulative successful pairings; equals births |
| `population_limit_blocks` | Otherwise viable attempts blocked by the safeguard |
| `food_consumed`, `water_consumed`, `signals_emitted` | Cumulative actual units/emissions |
| `food_regenerated`, `water_regenerated` | Cumulative actual additions, excluding initial placement |
| `starvation_deaths`, `dehydration_deaths`, `old_age_deaths`, `energy_deaths` | Cumulative deaths by the priority-selected cause |
| `mean_perception_radius`, `min_perception_radius`, `max_perception_radius` | Current living agents' perception radii |
| `mean_signal_probability`, `min_signal_probability`, `max_signal_probability` | Current living agents' random signal probabilities |
| `mean_hunger_multiplier`, `min_hunger_multiplier`, `max_hunger_multiplier` | Current living agents' hunger multipliers |
| `mean_thirst_multiplier`, `min_thirst_multiplier`, `max_thirst_multiplier` | Current living agents' thirst multipliers |
| `learning_agents` | Currently living LearningBrain individuals, including newborns |
| `total_learning_updates` | Cumulative Q updates, including those by agents that later died |
| `mean_learning_updates` | Mean lifetime update count among living LearningBrain agents only |
| `mean_memory_size` | Mean stored state rows among living LearningBrain agents only |
| `exploratory_actions`, `exploitative_actions` | Cumulative branch counts for LearningBrain decisions, including deceased agents |

Learning means are `None`/empty CSV cells when there are no living learners;
RandomBrain runs have zero learning counts and empty learning means. Newborns
contribute zero updates and zero memory rows. An exploitative action can still be
random among ties; the labels record the epsilon branch, not whether an action
was objectively useful. Living means require one read-only linear pass.

Subtract adjacent CSV rows for per-tick counts. An accounting assertion checks
`initial_population + births - deaths == population`. Tests also check unique IDs,
nonnegative energy/resources, bounds, and resource conservation.

Genome summaries are population-weighted snapshots, not cumulative measures:
every living agent counts once. Tick zero includes founders; later snapshots are
calculated after deaths are removed and newborns are added, including those
newborns immediately. Means use `math.fsum` for accurate floating-point summation.
When no agents remain, all twelve trait values are `None` in Python and empty
cells in CSV (not zero or NaN). A repeated step after extinction retains that
empty snapshot. Genome columns remain the final twelve CSV columns.

The summary pass reads only body liveness and immutable genome values. It neither
draws randomness nor changes bodies, brains, genomes, ordering, or simulation
rules. It takes linear time and temporary storage in living population size for
four traits; no historical genomes or samples are retained. Writing a snapshot
to CSV does not recompute it or affect the simulation.

An optional callable `event_sink` receives records; `--events` uses the existing
streaming JSONL writer. Payloads are built only when logging is enabled:

- `run_started`: effective config and the identical founder genome.
- `agent_step`: existing start position, before/after body state, local observation,
  action/success, emissions, consumption, death cause, plus a reproduction failure
  reason when applicable. Partner costs may change a body after its own step
  record; birth records identify both charged parents. LearningBrain records also
  contain `learning.reward`, `exploratory`, cumulative `updates`, and `memory_size`;
  full learned tables are not serialized in each event.
- `birth`: emitted after the acting phase, with `agent_id`/`details.child_id`, tick,
  `x`/`y`, both parent IDs, generation, complete `child_genome`, reproductive type,
  and initial needs/energy. Lineage can be reconstructed without retaining dead
  bodies. The inherited trait chosen before mutation and zero-effect mutation
  attempts are not separately logged.

Custom sinks can retain only births or sample observations. The default
simulation stores neither events nor past bodies/genomes. CSV and JSONL writers
use caller-owned streams and do not retain rows.

## Baselines, performance, and scientific limitations

Verification used Python 3.11.9 and the unchanged evolution preset, seed 0,
100 founders, and no population safeguard or ecology tuning:

| Brain | Tick | Population | Births | Deaths | Highest generation | Food / water consumed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 1,000 | 927 | 6,129 | 5,302 | 31 | 23,831 / 23,680 |
| Learning | 1,000 | 810 | 5,246 | 4,536 | 29 | 24,114 / 23,931 |
| Learning | 5,000 | 830 | 28,465 | 27,735 | 143 | 119,904 / 120,628 |

The RandomBrain run matches the actual pre-v0.3 control (commit `1e53e9e`) at
every captured tick through 1,000, including bodies/genomes, resources, signals,
IDs, and original ecological counters, with identical final RNG states:

```text
trajectory SHA-256: ff10294e2141a4684fcabad9bf759048ad045b5862c80a7865e7f1e8d6c86dc0
final state:       117738e2663e321026d72d23eea63212b61446e187d693355d168d9f92e9d6c8
final RNG states:  2e8b13531fef4da4ae409dfbe2fb353a663f33b94ae19a07915398e728b219ee
```

The regression test also compares a tick-100 body/world/signal/RNG digest
generated by running the original code from that Git commit. Output metadata
and new statistics columns intentionally differ. The retained legacy seed-42
baseline is extinction at tick 309, with no births, 100 deaths, 361 food units,
389 water units, and 1,425 signals.

Two separate 1,000-tick LearningBrain verification runs compare every tick,
including Q tables and their recency order, pending transitions, all bodies and
genomes, resources, signals, statistics, and all RNG states. One enables a sink
that writes metadata/birth events plus extra summary/CSV reads. They match each
other and the CLI CSV metrics; global random state remains unchanged. Population,
death-cause, unique-ID, learning-update, memory-bound, and resource-conservation
checks are applied throughout. Full JSONL serialization independence is also
covered by the automated test suite.

At tick 1,000, learners have made **693,705 updates**, with 139,302 exploratory
and 555,207 exploitative decisions. Living learners average 96.08 updates and
48.58 stored states. Peak population is 864, versus 1,484 for the random control.
At tick 5,000, learning has 3,916,313 cumulative updates and a mean memory size of
14.54 states. No population safeguard blocks occur.

The controlled `experiments.learning_demo` uses a one-cell world with one naive
agent, no reproduction, and no age death. Before each of 400 trials the harness
sets thirst to 60 and the tile's water to one unit; actions and physiology run
through the ordinary simulation. This intervention supplies repeated comparable
experience, not an instruction to the brain. All seven initial Q values are
zero. After training, `drink` is the unique highest-valued action (3.6196 versus
2.6334–3.0751 for the others), and was chosen 323/400 times. There are 399 updates
and one stored state. Other actions can have positive values through discounted
future returns. The changed learned values and greedy preference establish
learning in this controlled task; they do not establish ecological superiority.

Local CLI wall times, including CSV, were approximately **13.1 s** for random
1,000 ticks, **17.2 s** for learning 1,000 ticks, and **76.5 s** for learning 5,000
ticks. The first control shared the machine with verification work, and agent
counts differ, so these are indicative observations, not a controlled speed
benchmark or portable throughput guarantees. Full-state hashing, especially all
per-agent RNG states, adds substantial validation cost outside normal simulation.
A 200-tick learning profile attributes most cumulative time to local observation
construction (1.27 s) and action selection (1.22 s, including 0.83 s in the feature
encoder). These are profiled timings with instrumentation overhead, not normal
run benchmarks; `data/v03_profile.txt` records the breakdown. No multiprocessing
or ecological parameter changes were introduced in response.
Local generated artifacts (ignored by Git) are:

- `data/v03_random_seed_0.csv`, `data/v03_learning_seed_0.csv`, and
  `data/v03_learning_seed_0_5000.csv`, each with a `.metadata.json` sidecar.
- `data/v03_verification.json`, `data/v03_runtime.json`,
  `data/v03_learning_demo.json`, and `data/v03_learning_births.jsonl`.

Memory scales with the world, living population, pending births, and recent
signals, plus at most 256 state rows per learner by default, rather than elapsed
ticks. Perception allocates local snapshots, resource
renewal scans the grid when enabled, and partner lookup searches nearby buckets.
Very dense local clusters can still make pairing expensive. Detailed serialization
and large inherited perception radii add cost. No claim of million-tick or
thousand-run throughput is made without further measurement.

The ecology is deliberately abstract. All tiles can regenerate, related agents
can pair, births are immediate, and reproductive types are compatibility bits.
There is no penalty balancing lower metabolic multipliers, so selection can favor
the configured lower bounds; this is not a validated biological fitness model.
RandomBrain ignores observations; LearningBrain uses a coarse, partially observed
state that can merge quite different situations. Local nearest-resource features
omit richer spatial structure. Short lives, changing neighbors, sparse repeated
states, optimistic zero values, exploration, and eviction can impede learning.
Reward can discourage costly reproduction even when reproduction benefits the
lineage. The controller has no reproductive-fitness objective or learned knowledge
transfer between generations.

The learning run has fewer births and fewer deaths than the control at tick 1,000;
population alone does not measure learning quality or individual survival benefit.
Mean perception radius is 2.0992 for random and 2.0148 for learning at tick 1,000.
By tick 5,000 it is **0.0675** in the learning run, down from founder value 2.
This does not support a simple claim that larger perception is favored here.
Fewer distinct sensory states might make value reuse easier, but that is a
hypothesis requiring controlled experiments, not an established explanation.
One seed cannot distinguish selection from drift or establish long-term stability.
Extinction remains legitimate and is tested. Multiple generations and acquired
behavior do not establish intelligence, culture, or emergent language.

The next experiment should compare RandomBrain and LearningBrain across the same
replicated seeds and unchanged ecological configurations, reporting survival,
births/deaths, agent-time exposure, resource use, and trait distributions. Follow
with explicit frozen-learning and perception ablations to separate learning from
feature/exploration effects. Keep hyperparameters fixed before interpreting the
comparison; do not tune the ecology until a controller wins. Signal-learning
mechanisms remain future work, and no extension should directly script farming,
institutions, or other civilization outcomes.
