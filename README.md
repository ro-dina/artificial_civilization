# Artificial Civilization Simulation — v0.4

A headless, standard-library-only artificial-life simulation for experiments in
emergent behavior. v0.4 separates direct visual/resource perception from
longer-range auditory perception of meaningless signals. It retains v0.3's
optional individual learning and the existing survival, reproduction,
inheritance, mutation, renewable resources, and lineage systems. Both controllers'
decision rules are unchanged; auditory input is intentionally unused.

**v0.4 does not implement language learning or meaningful communication.** Only
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

# Three-tick sensory mechanism demonstration; no learned signal meanings.
python3 -m experiments.auditory_demo

# Short auditory diagnostics. Vocabulary 16 retains the v0.3 emission baseline.
python3 main.py --mode evolution --brain learning --seed 0 --ticks 10 \
  --hearing-radius 8 --events data/v04_auditory.jsonl
# An explicit alternative vocabulary for future experiments (not a semantic map).
python3 main.py --seed 0 --ticks 10 --signal-vocab-size 40

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

The complete suite includes simulation, learning, observational analysis, and
historical controls. v0.4 adds 18 auditory tests to the existing 151 tests.
Existing auditory assertions are migrated to the intentionally narrower sensory
record; historical state digests are retained, not regenerated to fit the change.
See [v0.4 verification](#v04-verification-and-compatibility) for the scope of
the regression checks and the deliberate observation/configuration migration.

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
tests/test_auditory.py       Hearing, privacy, timing, RNG and historical controls
tests/auditory_control.py    Full ecological state/action digest helpers
tests/fixtures/             Pre-v0.3 and pre-v0.4 deterministic control digests
experiments/learning_demo.py Controlled repeated-resource learning validation
experiments/auditory_demo.py Three-tick sensory mechanism check
experiments/analyze_fixed_perception.py Offline CSV aggregation and PNG plots
tests/test_analyze_fixed_perception.py Analysis fixtures, missing data, and reproducibility
requirements-analysis.txt   Optional pandas/matplotlib dependencies
```

The existing architecture is extended rather than replaced. A brain implements
`choose_action(human, observation) -> Decision`. It reads its own body and local
observation and returns a decision without mutating the body. Observations contain
`tick`, nearby `(dx, dy, food, water)` records, and heard
`HeardSignal(signal_id, source_direction)` records. They contain no sender
identity, exact sender displacement/distance, mutable world references,
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

## Visual and auditory perception (v0.4)

The long-term question is: **How do environmental conditions,
perceptual/cognitive traits, and social conditions affect the emergence and
development of communication and language?** This release establishes the
sensory mechanism only. Later versions will test whether agents can learn to use
otherwise meaningless signals and whether environmental/perceptual conditions
affect the emergence of communication.

`Genome.perception_radius` still controls exactly the same local food/water tile
observations. `Config.hearing_radius` is a separate **nonheritable integer,
default 8**, validated as nonnegative. It is larger than the founder visual
radius of 2 in the default 40×30 world, while remaining local. It need not exceed
every evolved individual's visual range. Hearing has no mutation, inheritance,
metabolic cost, or dependency on visual perception. Radius zero permits hearing
other agents' signals from the receiver's current cell only.

Delivery retains the existing inclusive **Chebyshev distance** rule:
`max(abs(sender_x - receiver_x), abs(sender_y - receiver_y)) <= hearing_radius`.
Visual and reproduction distance rules are unchanged. Within auditory range,
each signal is received perfectly; outside it, the signal is absent. There are
no obstacles, attenuation, transmission failures, confusion, phonetic similarity,
sound intensity, approximate-distance feature, or masking. Multiple emissions
are retained as separate records, including identical signal IDs.

The immutable, slotted Brain-facing record has exactly two fields:

```python
HeardSignal(signal_id: int, source_direction: SourceDirection)
```

`SourceDirection` is a string enum, serialized as one of the following values.
These are **sign sectors**, not geometric angular octants. Directions describe
the acoustic source; they do not give meanings to signal IDs.

| Sign of dx | Sign of dy | Direction |
| --- | --- | --- |
| 0 | − | N |
| + | − | NE |
| + | 0 | E |
| + | + | SE |
| 0 | + | S |
| − | + | SW |
| − | 0 | W |
| − | − | NW |
| 0 | 0 | SAME_CELL |

North decreases y. Exact sender coordinates, dx/dy, distance, strength, and
sender ID are **not** properties of this record. Internal immutable `Signal`
records still hold emission position and sender ID for filtering and self
exclusion; no reference to them is given to the Brain.

During acting tick **t**, an agent observes signals emitted in **t−1**, relative
to its current pre-action position. It then chooses and executes its action,
and any new emission uses its **post-action** position. That emission cannot be
heard during t, even by an agent acting later in the same tick. The end-of-tick
buffer swap makes it available throughout the acting phase of t+1, then it
expires. Self-signals are excluded. Death or subsequent movement of a sender
does not erase or relocate the already emitted signal. Newborn action timing
and every biological/learning/statistics phase remain unchanged. The public
`sim.observe()` helper called between steps reads the prepared next-phase buffer;
its `tick` is still the last completed tick. Event logs and the demo use actual
in-step observations to identify receiving ticks unambiguously.

`LearningBrain.state()` is unchanged: hunger, thirst, and energy bins plus
nearest visible food and water directions. It ignores **all auditory features**,
including direction. Q updates, rewards, LRU recency, and pending transitions
are unchanged. RandomBrain still ignores observations. Emission is not an
action: the same independent random mechanism uses inherited
`signal_probability`, then selects a meaningless integer in
`[0, signal_vocab_size)`. There is no communication reward, penalty, intent,
identity recognition, signal-to-action mapping, or learned signal selection.

The default vocabulary remains **16** to preserve v0.3 trajectories, especially
RandomBrain's shared action/emission stream. Use `--signal-vocab-size 40` or
`Config(signal_vocab_size=40)` for an explicitly different experimental setting.
Integer neighbors have no built-in similarity: 17 is no more similar to 18 than
to 39. Hearing/filtering/direction conversion use no randomness and create no
new RNG stream. Neither brain implementation, the genome, nor the ecology is
modified by this release.

Full `--events` JSONL includes `auditory_schema_version: 2` in `run_started`.
Each `agent_step.details.observation.signals` contains only the two sensory
fields. A separate `details.auditory_diagnostics` array contains received signal
ID, sender ID, **emission** position, receiver ID and pre-action position, coarse
direction, Chebyshev distance, effective hearing radius, `inside_hearing_range`,
`emitted_tick`, and `receiving_tick`. These are privileged analysis data, not
inputs to `choose_action`. Only delivered signals are logged; all have
`inside_hearing_range=true`. Out-of-range emissions are absent, not additional
sensory records. Detailed payloads are constructed only for an enabled full
event sink, streamed, and discarded; lineage-only logging does not build them.

### v0.4 verification and compatibility

Before product changes, short controls were captured from v0.3 commit
`9dc56bed749a2d3e318ae2e7cbe52f6c1173b4a5` (Python 3.11.9), seed 0, 100 founders,
ticks 0–100, in all four legacy/evolution × random/learning combinations.
`tests/fixtures/v03_auditory_control.json` retains those results. The tests
compare every completed state and the full action/outcome stream against them
at both hearing radius 3 and 8. Coverage includes bodies, living IDs/genomes,
resources, births/deaths, every aggregate, emitted signals, Q values and LRU
order, pending transitions, all retained simulation/brain RNG states, and
unchanged global random state. Signals are canonicalized by emission provenance
for this comparison because bucket layout and unused auditory observations
intentionally differ. The older v0.2 fixture still checks its original **exact**
bucket layout at radius 3 with its unchanged stored digest.

The archived `data/v03_random_seed_0.csv` and `data/v03_learning_seed_0.csv` were
also compared directly: **all aggregate values at all 101 snapshots through
tick 100 match exactly** at the new hearing default. Final population/births/
deaths are 232/136/4 (Random) and 207/109/2 (Learning). Full-event logging,
removing all heard inputs, extra observation reads, and serialization are tested
for behavioral/RNG isolation. The full suite passes **169 tests**. No new long
evolutionary or 5000-tick multi-seed study was run for v0.4.

Intentional API/output changes:

- Old `HeardSignal(signal_id, sender_id, dx, dy)` consumers must migrate to the
  two-field auditory record. No legacy identity/coordinate properties remain on
  a Brain-facing object. Tests of original delivery boundaries and self exclusion
  are preserved using distinct test vocalizations; exact provenance is checked
  separately in diagnostics.
- New Config field `hearing_radius=8` replaces the old default hearing range 3.
  The old constant `SIGNAL_RANGE=3` is retained for historical callers.
  `signal_range` remains a deprecated optional constructor alias, now default
  `None`. If supplied (including zero), it replaces the default `hearing_radius`
  and the resolved radius is stored in Config. Conflicting nondefault values
  are rejected. Saved v0.3 configurations containing `signal_range=3` still load
  with radius 3. When changing a loaded legacy config, clear the alias explicitly:
  `replace(old_config, signal_range=None, hearing_radius=8)`.
- Aggregate columns and values are unchanged. Config hashes/metadata differ
  because Config now includes hearing settings and the CLI version is 0.4.
  Full auditory event payloads intentionally differ; they are not byte-compatible
  with v0.3. The existing analysis code/data and preserved perception baseline
  study are unchanged.

`python3 -m experiments.auditory_demo` uses two stationary agents separated by
5 cells, visual radius 2, hearing radius 8. The receiver cannot see the sender's
cell. It receives no signal at tick 1, only
`{"signal_id": 7, "source_direction": "E"}` at tick 2, and none at tick 3.
The harness emits one known **meaningless** ID; this establishes a sensory
mechanism, not learned listening, intentional communication, or language.

A separate 10-tick CLI smoke run (evolution, LearningBrain, seed 0) ended with
100 living agents, 0 births/deaths, and 99 emissions. Its 1,785 delivered records
included 1,615 sources outside the receiver's visible tiles. All records were
checked against the preceding tick's actual emissions for timing, range,
self exclusion, and sensory-field privacy. Local diagnostic artifacts are
`data/v04_auditory_smoke.csv` (plus metadata), `data/v04_auditory_smoke.jsonl`,
`data/v04_auditory_smoke_validation.json`, and `data/v04_short_regression.json`.
These short-run outputs are ignored by Git; the regression fixture and tests
are included in the source tree.

Hearing still uses nine nearby spatial buckets. A larger range entails more
candidate signals and observation allocations; dense populations and full
diagnostic serialization can be costly. No historical signal/agent storage or
new performance guarantee is introduced.

Possible later experiments, **not implemented**: distance-dependent reception
probability; noisy direction; signal confusion; acoustic/phonological feature
representations and greater confusion between similar sounds; intensity and
approximate distance; obstacles/terrain attenuation; simultaneous-speaker
masking; learned listening and signal selection; signal costs; repeated or
sequential vocalizations; compositional signals; social learning,
intergenerational transmission, and dialect formation. Any future auditory noise
needs a separate named RNG stream. v0.5 learned-listening design should be
reviewed before implementation; this release claims no emergence of semantics,
culture, or language.

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

### Free-evolution observations and cohort analysis

The replicated LearningBrain runs, fixed p0/p1/p2/p4 experiments, and ten-seed
cohort follow-up are complete and preserved as the
[perception baseline study](#perception-baseline-study). Why freely inherited
perception tends toward zero despite larger fixed p1/p2 populations remains
unresolved; causal investigation is deferred until needed. A **fixed
condition's population is an ecological outcome**; reproductive participation
and death rates **within a mixed population** measure different quantities.
Neither establishes the cause of trait change on its own.

All additions here are observational. Config, aggregate CSV columns, brains,
learning/Q-table recency, inheritance/mutation, partner selection, update order,
signals, resources, and named RNG draws are unchanged. Simulation remains
standard-library-only. Optional offline analyses use the same pandas/matplotlib
dependencies as fixed-perception analysis, without seaborn or random jitter.

Analyze the **existing** ten runs first (no simulation execution):

```bash
python3 -m experiments.analyze_evolving_perception
# Defaults: data/v03_learning_seed_*_5000.csv → analysis/evolving_perception/
# New instrumentation runs instead:
python3 -m experiments.analyze_evolving_perception --pattern 'evolving_seed_[0-9].csv'
```

The second pattern selects seeds 0–9 without also matching cohort files. For
larger seed sets, put aggregate CSVs in a separate directory and use a broad
`--pattern 'evolving_seed_*.csv'`. Multiple files for one seed are rejected rather than mixed.
All three new analyzers accept `--data-dir`, `--output-dir`, and `--pattern`, and
also support direct invocation as `python3 experiments/<script>.py`.

Required aggregate columns are `tick`, `population`, `mean_perception_radius`.
Optional columns are min/max perception, mean hunger/thirst/signal, mean memory
size/learning updates, births/deaths, all four death causes, highest generation.
Missing optional columns warn and have `n=0`; required columns must exist.
The actual filenames are preserved. Recorded brain/config metadata, if present,
rejects RandomBrain or fixed-perception inputs. Each run has equal weight;
sample SD uses `ddof=1`, missing ticks are not filled, and extinct living means
are NaN. Final metrics use each seed's own last recorded row (warn if horizons
differ). No earlier value replaces a NaN in that row.

Outputs in `analysis/evolving_perception/`:

- `perception_over_time.png`, `population_over_time.png`, `learning_over_time.png`,
  `traits_over_time.png`: seed means ±1 sample SD.
- `perception_by_seed.png`: separate panels to inspect differences in timing.
- `perception_population_relationship.png`: matching-tick seed means colored by
  tick, a descriptive association only.
- `evolving_time_series.csv`, `evolving_final_metrics.csv`: tidy mean/std/min/max/n.
- `evolving_run_summary.csv`: maximum-tick row per seed.
- `analysis_metadata.json`: input hashes, versions, and aggregation rules.

#### Record cohorts and compact lineages

```bash
python3 main.py --mode evolution --brain learning --seed 0 --ticks 5000 \
  --csv data/evolving_seed_0.csv \
  --cohort-csv data/evolving_seed_0_cohorts.csv \
  --lineage-events data/evolving_seed_0_lineage.jsonl
```

Both new flags default OFF and work independently of `--csv` and `--events`.
Destinations are overwritten, like existing output flags; all output and sidecar
paths must be distinct. Each cohort CSV has a `.metadata.json` sidecar containing
the same config ID and explicit definitions. Observational flags are outside
Config, so toggling them does not change config IDs. Programmatic callers can
pass `cohort_sink` and/or `lineage_sink` to `Simulation`/`create_default`.

Cohort CSV is tidy: nine rows (perception 0–8) for tick 0 and every completed
tick, even for empty cohorts. Counters are **per tick, not cumulative**. Tick 0
records the initial population/means, with zero events and zero exposures.

| Field | Exact definition |
| --- | --- |
| `living_population` | Alive individuals at completed tick, after deaths are removed and newborns added. |
| `living_agent_ticks` | One per individual in that completed snapshot, except tick 0 contributes zero. Includes newborns; excludes that tick's deaths. |
| `action_opportunities` | One per individual actually entering the action phase. Excludes newborns; includes individuals that die after their action. All ages/actions count, not just reproductively eligible individuals. |
| `births_as_child` | Births attributed to the child's genome perception. No founder births are counted. |
| `successful_parent_participations` | Both parents at each successful birth, attributed to each parent's perception: exactly two participations per birth, including two in the same cohort if applicable. Counts remain even if a parent dies that tick. |
| `reproduction_initiations` | Only the individual choosing REPRODUCE, including unsuccessful attempts. The compatible partner is not an initiator unless it independently chooses that action. |
| `failed_reproduction_attempts` | Initiations without a birth, including biological/partner and computational-population-limit failures. |
| `deaths` and `*_deaths` | Deaths attributed to genome perception at death; causes are starvation, dehydration, old_age, energy in the existing priority order. Genomes normally change only at birth. |
| `mean_signal_probability`, `mean_hunger_multiplier`, `mean_thirst_multiplier` | Means over living members of that completed cohort only. |
| `mean_generation`, `mean_age` | Living members' means; newborn age is zero. |

Empty-cohort means are `None`, written as blank cells. A newborn has one living
agent-tick on its birth tick and **no action opportunity until the next tick**.
A dying agent has the converse. Summed over cohorts, parent participations equal
twice births; initiations minus failures equal births; causes sum to deaths.
The observer retains only nine rows of counters and an O(N) temporary scalar
aggregation, never dead Human objects, genome history, or per-tick history.
No Q-table/recency lookup is needed. Extinction's final tick is emitted once;
subsequent no-op steps do not emit duplicate rows.

`--lineage-events` reuses existing `Event`/`JSONLEventWriter` and the same birth
event used by detailed logging. It streams `run_started` (config), `founder`
(ID, genome, generation=0, tick=0), `birth` (child, **both** parents, genome,
generation, position), and `death` (agent ID, tick, cause, genome at death).
The CLI appends `run_finished` with final tick/population, including if no event
happened at that tick. A programmatic caller may append the same Event when
closing a run. Without that marker, the lineage analyzer warns and uses the last
event tick as its known horizon. Deaths are emitted during action processing;
birth events follow the phase, so a parent's death may precede its child's birth
record **within the same tick**. This preserves existing birth timing.
There are no observation/action/Q-table payloads per individual per tick in this
compact log. Full `--events` logging remains a separate, potentially large option.

#### Analyze recorded cohorts and lineages

```bash
python3 -m experiments.analyze_perception_cohorts --window-size 1000
python3 -m experiments.analyze_lineages --ticks 0 1000 2000 3000 4000 5000
```

Cohort input defaults to `data/evolving_seed_*_cohorts.csv`; results go to
`analysis/perception_cohorts/`:

- `perception_frequency_over_time.png` (living shares, NaN at extinction),
  `perception_population_over_time.png` (absolute counts).
- `perception_reproductive_rate.png`, `perception_death_rate.png`:
  windowed parent-participation and death rates per action opportunity.
- `perception_exposure.png`: per-window action exposure to reveal sparse cohorts.
- `perception_traits_over_time.png`: p0/p1/p2; `perception_traits_all_cohorts.png`:
  all nine cohorts, with thinner dashed lines for p3–p8.
- `cohort_summary.csv`: separate **seed × perception** rows with total recorded
  exposures/events, death causes, rates, tick range and observed tick count.
- `cohort_window_rates.csv`: separate seed/perception/window counts and rates,
  first/last observed tick, observed tick count, and `complete_window`.
- `cohort_window_summary.csv`: equal-seed mean/std/min/max/n for those window
  values; `cohort_time_series.csv`: equal-seed count/share/trait summaries.
- `analysis_metadata.json`.

Rate windows are **(0,1000], (1000,2000], ...** by default. Within each seed and
window, divide summed successful parent participations, deaths, or initiations
by summed `action_opportunities`. A zero denominator gives NaN, never zero.
Then average these per-seed rates with equal seed weights; do not pool exposure
across seeds. These are crude participation/death rates, not age-, eligibility-,
location-, or generation-adjusted fitness estimates. Offspring need not inherit
the parents' perception, so participation minus death is not a cohort growth rate.

Missing whole ticks warn; only observed counts contribute and partial windows
are flagged. There is no extrapolation or survival carry-forward. Every recorded
tick must contain all nine cohorts; missing cohort rows are errors rather than
assumed zero. Counts, cause totals, and birth/participation identities are
validated. Small cohorts can have volatile trait means/rates: always inspect
exposure, population, and each metric's finite-seed `n`. With just seed 0, SD
is undefined. All plots use 180 dpi, with deterministic CSV/PNG output in the
same software environment; no significance or optimality tests are performed.

Lineage input defaults to `data/evolving_seed_*_lineage.jsonl`. The optional
offline analyzer retains both parents in a DAG. A founder's descendants are
individuals reachable through **either** parental path, excluding that founder
itself. Memberships overlap; they are not independent lineages and must not be
summed into population totals. Bitsets store founder reachability, not genetic
contribution. Outputs in `analysis/lineages/`:

- `parent_child_edges.csv`: both parent IDs for every child.
- `founder_descendants.csv`: descendants ever born and still alive by each selected
  tick (plus the final known tick), per seed/founder.
- `founder_trait_distributions.csv`: living descendant counts per perception and
  mean/min/max hunger, thirst, signal within each founder/perception membership.
- `ancestry_mixing.csv`: living population and mean/min/max reachable founder
  count per individual; `analysis_metadata.json`.

This modest offline analysis costs roughly founders × recorded individuals ×
selected ticks; it is not a scalable inference of independent genetic lineages.
Extensive ancestry overlap can make founder membership uninformative about
hitchhiking. Recent-ancestor or trait-transmission analyses remain future work.

#### Verified observation results

All **143 tests** pass with analysis dependencies (118 existing + 25 added).
The new suite covers attribution, exposures, empty populations, counts/bounds,
both-parent event reuse, per-seed rates, missing ticks, NaNs, deterministic
analysis bytes, and four observation modes: OFF, cohort only, lineage only,
both. Each mode is compared every tick for 120 ticks under all four combinations
of legacy/evolution and RandomBrain/LearningBrain. Checks include all bodies,
genomes, ordering, resources, signals, statistics, Q values/LRU order, pending
transitions, all simulation/brain RNG states, and unchanged global random state.

Before simulation edits, seed 0 × 5000 ticks was recorded at commit `1dbda1e`.
Afterward, OFF and both-ON runs matched all **5001 aggregate snapshots**, plus
complete-state hashes at ticks 0, 100, 1000, 2000, 3000, 4000, 5000. The actual
new CLI aggregate also matches every pre-existing
`v03_learning_seed_0_5000.csv` statistics cell. No existing tests or ecological
parameters were changed. `data/perception_observation_verification.json` stores
hashes/method and `data/perception_observation_runtime.json` the CLI measurement.

Local step-plus-write times, excluding validation hashing: before **76.78 s**,
after OFF **78.02 s**, both ON **80.51 s** (about 3.2% above after-OFF).
Separate actual CLI wall time was **78.76 s**. These are single local measurements,
not performance guarantees; the OFF run partly overlapped tests. Seed 0 output
sizes excluding small sidecars: aggregate **2,089,165 bytes**, cohort
**7,673,406 bytes** (45,009 data rows), compact lineage **17,259,256 bytes**.
At 5000 ticks this is 28,465 birth and 27,735 death records, plus 100 founders
and start/end markers, rather than millions of agent-step records.

Existing ten-seed free evolution shows mean perception **2.0435 ±0.0744 at
tick 1000**, **0.5146 ±0.3656 at 3000**, and **0.06116 ±0.02080 at 5000**.
The first recorded mean below 1 varies from tick **1874 to 3085** across seeds.
Final population is **819.8 ±31.65** and living memory **14.827 ±0.751 states**;
at tick 1000 memory was **47.556 ±1.482**. These simultaneous changes do not
identify causality, cohort exposure, or ancestry from old aggregate CSVs alone.

New seed 0 ends with **830** living agents: **p0=779 (93.86%), p1=46, p2=5**.
Births=28,465, deaths=27,735, highest generation=143. In the (1000,2000] window:

| Cohort | Action opportunities | Successful parent participations / opportunity | Deaths / opportunity |
| --- | ---: | ---: | ---: |
| p0 | 13,542 | 0.01684 | 0.00524 |
| p1 | 156,685 | 0.01445 | 0.00674 |
| p2 | 501,547 | 0.01427 | 0.00755 |

The preceding p0 window has only **172** opportunities and one successful parent
participation; its rate is especially unstable. p7/p8 have no exposure in this
run. Final p0/p1 mean thirst multipliers are 0.92437/0.93012; p2's 1.01050 comes
from only five survivors. These associations are from **one seed**, confounded by
timing, age, location and other traits, and do not establish a perception effect.
By tick 3000, **every living individual reaches all 100 founders** through its
two-parent ancestry; the same holds at tick 5000. Thus founder reachability is
already saturated here and cannot distinguish a small successful genetic lineage
or prove hitchhiking. The birth/death DAG is available for finer offline analysis.

The following commands document how the ten observed replicates were collected.
They are retained for reproducibility, not as a pending experiment; the baseline
study is complete. Reusing these destinations would overwrite those runs:

```bash
for seed in {0..9}; do
  python3 main.py --mode evolution --brain learning --seed "$seed" --ticks 5000 \
    --csv "data/cohort_replicates/evolving_seed_${seed}.csv" \
    --cohort-csv "data/cohort_replicates/evolving_seed_${seed}_cohorts.csv" \
    --lineage-events "data/cohort_replicates/evolving_seed_${seed}_lineage.jsonl"
done
python3 -m experiments.analyze_perception_cohorts --data-dir data/cohort_replicates
python3 -m experiments.analyze_lineages --data-dir data/cohort_replicates
python3 -m experiments.analyze_evolving_perception --data-dir data/cohort_replicates \
  --pattern 'evolving_seed_[0-9].csv'
```

### Perception baseline study

**Status (2026-10-03): observational baseline complete and preserved.** Further
work to identify the causal effect of perception itself is deferred. Reopen that
question only when a future research or design decision needs it; there is no
scheduled additional simulation or causal-analysis task for this study.

The study used the existing `data/cohort_replicates` LearningBrain free-evolution
seeds 0–9, each covering ticks 0–5000. It compared p0/p1/p2 in five 1000-tick
windows. Rates were calculated per seed from event counts / action opportunities;
living traits, age, and generation used living-exposure weights within a seed.
Seed-level values were then summarized with equal weights and sample SD
(`ddof=1`), preserving paired p0−p2 differences, finite n, and exposure.
No new simulation or changes to simulation dynamics were used for this analysis.

Recorded findings:

- In each of (1000,2000], (2000,3000], and (3000,4000], p0 had higher successful
  parent participation per action opportunity and lower death per opportunity
  than p2 in **10/10 seeds**. These are descriptive cohort comparisons.
- The first window was seed-dependent and p0 exposure was sparse (172–9467
  action opportunities). In the last window, each rate direction held in 8/9
  evaluable seeds; seed 8 had no p2 exposure. The two exceptions were different
  seeds, so both directions held simultaneously in 7/9.
- Hunger/thirst differences were not consistent across seeds. In (1000,2000],
  seeds 4 and 9 had both p0 metabolic means at least as high as p2, yet the same
  rate directions. A simple uniformly-lower-metabolism explanation is unsupported;
  this does not rule out other-trait or demographic explanations.
- Temporal composition matters: the (1000,2000] p0−p2 generation difference was
  **+7.847** with separate cohort exposure weights, but **−0.223** when comparing
  both cohorts at the same ticks with equal time weights. Growing and shrinking
  cohorts weight different parts of a window differently.

Individual trait combinations, age distributions, reproductive eligibility,
location/resource conditions, and cohort-specific learning state were not
controlled. Cohort learning metrics were not recorded. **The causal effect of
perception=0 has not been identified.** This uncertainty is part of the retained
baseline, not a reason to continue this investigation immediately.

Preserved deliverables:

- [Detailed report and output index](analysis/perception_transition/README.md).
- **33 unchanged analysis outputs** in `analysis/perception_transition/`:
  14 CSVs, 18 PNGs, and `analysis_metadata.json` (input hashes and methods).
- [Validation record](analysis/perception_transition/validation.json): eight
  arithmetic tests, 1050 independent raw-CSV checks, byte-identical regeneration
  of all 33 outputs, and unchanged input hashes/Python/NumPy RNG states.
- [Baseline archive](analysis/perception_baseline_study.zip): the 33 outputs,
  report, validation record, and
  [SHA-256 manifest](analysis/perception_transition/baseline_manifest.json).

The archive preserves this snapshot if the working analysis directory is reused.
It contains derived results, not the raw `data/cohort_replicates` inputs, which
remain local and are ignored by Git. Keep those inputs for future recomputation.
If this question is reopened, use the existing analyzer with a different output
directory (`--output-dir`) to keep follow-up results separate from this baseline.

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

Replicated LearningBrain evolution, fixed p0/p1/p2/p4 × ten seeds, and the
ten-seed cohort transition analysis are retained as the
[perception baseline study](#perception-baseline-study). This observational
study is complete. Perception's causal role remains unresolved and is deferred
until a future question requires it; additional perception experiments are not
the current next task. Reuse the preserved results as a reference for subsequent
development. Signal learning remains future work, and no extension should
directly script farming, institutions, or other civilization outcomes.
