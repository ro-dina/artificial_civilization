# v0.7.0 — Visual Social Recognition and Finite Social Memory

## Objective and scope

Can individuals retain bounded knowledge of visually encountered individuals,
distinguish continuous contact from reunions, and support reliable measurements
without altering existing decisions, learning, reward, ecology or acoustics?

**v0.7.0 introduces visual identity recognition and bounded social memory but does
not introduce social motivation, social rewards, communication incentives,
learned auditory identity, cooperation, friendship, or language.** Perfect visual
identity recognition is an imposed cognitive capability, not an emergent result.
The longer-term question remains which environmental, cognitive, perceptual and
social conditions could support spontaneous communication without scripted
meanings or conventions. This release does not answer that question.

## Preliminary inspection and compatibility design

The starting commit was `d5a80c6` (v0.6.4). Before implementation, README, Config,
Human, Observation, both brains, auditory memory, tick ordering, reproduction,
communication/vocal observers, historical tests/fixtures and v0.6.1–v0.6.4
reports were inspected. `data/v070_before.json` snapshots 358 source/fixture/
historical result files. All **409 baseline tests passed in 68.08 s**.
`data/v070_design.json` records the design and bounded pilot before implementation.

Important findings, documented before edits:

- Actual resource vision uses the genome's inclusive Chebyshev square at each
  actor's sequential decision point. Nearest-resource encoding later uses its
  existing distance/tie rules; neither is changed.
- Communication diagnostics' radius-2 density is a **tick-start** population
  prefix grid. It cannot be reused as a current visual neighborhood.
- Reproduction buckets already follow earlier movement. Dead references remain
  until tick-end but eligibility filters them. A read-only visual query can use
  these buckets while independently filtering actual liveness and visual range.
- Newborns are queued until after existing actors finish; IDs are monotonic.
  Private memories must be separately constructed for every child.
- Frozen hashes include serialized Config, Human fields and Observation. Adding
  empty fields to these base dataclasses would change historical outputs despite
  identical ecology. Optional data-only subclasses and an independent settings
  object preserve the original schemas exactly.
- The old CLI/sidecars retained v0.6.1/0.6.2 recording-format versions; this was an
  intentional compatibility convention, not a biological discrepancy. CLI help
  now identifies v0.7.0; historical recording versions remain unchanged.

No behavioral discrepancy with the documented model was found. The visual/
diagnostic timing distinction and serialization constraints determined the design.

## Implementation and public interfaces

Modified files:

- `config.py`: independent frozen `SocialConfig` and three defaults; the existing
  `Config` class, defaults, fields and serialization remain unchanged.
- `agents/human.py`: optional data-only `SocialHuman(Human)` extension.
- `agents/observation.py`: immutable `VisibleIndividual` and optional
  `SocialObservation(Observation)` extension; legacy auditory classes unchanged.
- `simulation/simulation.py`: optional recognition/memory/recording hooks,
  independent memory creation for founders/children and sequential index use.
- `systems/reproduction.py`: one pure read-only `visible` query; all existing
  pairing, cost, index update and RNG methods are unchanged.
- `main.py`: opt-in flags, exclusive social paths and completed social sidecar.
- `README.md`, `.gitignore`: usage/documentation and generated result exclusion.

Added files:

- `agents/social_memory.py`
- `world/social_index.py`
- `simulation/social_observer.py`
- `experiments/social_data.py`
- `experiments/analyze_social_memory.py`
- `experiments/social_plots.py`
- `experiments/social_verification.py`
- `experiments/social_memory_pilot.py`
- `tests/test_social_memory.py`
- `tests/test_social_analysis.py`
- `docs/social_memory_v070.md`

Programmatic entry points:

```python
from config import Config, SocialConfig
from simulation.simulation import Simulation

settings = SocialConfig(
    social_memory_enabled=True,
    social_memory_capacity=32,
    social_reunion_min_absence_ticks=20,
)
sim = Simulation(Config.evolution(brain="learning", random_seed=0),
                 social_config=settings, social_sink=None)
sim.step()
# Also accepted by Simulation.create_default(...).

# Read-only diagnostics, no recency refresh:
entries = sim.agents[0].social_memory.entries()  # immutable, LRU to MRU
entry = sim.agents[0].social_memory.get(other_id)
```

`SocialMemory.observe(sorted_ids_tuple, tick, capture=False)` is the sole normal
memory-update operation and permits exactly one opportunity per tick. Capture
returns immutable `SocialUpdate`/`SocialContact` records only when needed by the
optional recorder. Recording OFF allocates no detailed contact payloads.
`CSVSocialWriter.open(stack, path)` creates an exclusive plain/gzip stream.
Programmatic callers finish `sim.social_observer.finish(sim.statistics)`, close
streams, then call `write_metadata(path, sim, writer)` to certify completion.
CLI and pilot handle this sequence automatically.

The simulator and enabled social mechanism remain standard-library-only.
Offline numpy/pandas/matplotlib dependencies are unchanged; seaborn is not used.
No physical or vocal brain, reward, genome, acoustic system, resource rule or
aggregate-statistics implementation is modified.

## Settings and visual assumption

| Independent SocialConfig parameter | Default | Validation / role |
|---|---|---|
| social_memory_enabled | False | bool; historical runs remain disabled |
| social_memory_capacity | 32 | positive integer; engineering capacity choice |
| social_reunion_min_absence_ticks | 20 | positive integer; classification only |

These settings do not inherit, mutate, alter perception radius or add costs.
They are not new genome traits. The social sidecar binds both the unchanged
simulation Config and SocialConfig with a canonical joint SHA-256. Ordinary
aggregate CSV/config IDs remain unchanged, including in matched enabled runs;
use the separate social sidecar to reproduce social measurements.

Brain-visible visual record:

```text
VisibleIndividual(individual_id, dx, dy)
```

Only living others with `max(abs(dx), abs(dy)) <= observer.genome.perception_radius`
are included, sorted by numeric ID. Boundary is inclusive and clipped by the
existing world. Radius zero sees others on the same tile. Self is excluded.
There is no access to others' needs, energy, brains, memories, reproductive type,
parentage, genealogy or later death state. Relative offsets are visual data;
absolute target coordinates are not stored. Perfect distinguishable IDs are
an explicit simplifying assumption. This mechanism is not learned recognition.

Disabled runs retain exactly `Human` and `Observation`, without extra fields.
Enabled runs use their data-only subclasses. Both existing brains ignore
`SocialObservation.individuals` and `Human.social_memory`. Q keys retain their
original seven body/visual integers and optional eighth auditory feature.

## Timing, indexing, newborns and death

At each ordinary existing actor's decision point:

1. Construct existing visual resources and auditory reception unchanged.
2. Query current living visual neighbors using the same visual radius.
3. Update that actor's independent social memory once, before choose_action.
4. Optionally stream the observation/contact update.
5. Run the unchanged action, reproduction, emission, physiology and Q feedback.
6. Record an actual death separately if physiology caused it.

Earlier actors may already have moved or died. They are not replaced by a
population snapshot. Reproduction's live buckets are queried read-only when
active; otherwise a separate unit-cell index is created only when the social
feature is enabled. Movement updates that index and liveness is filtered at query
time. Indices release all references at tick-end. The added reproduction query
never changes bucket insertion order, partner candidates or random selection.

A diagnostic `sim.observe` outside the acting phase builds a temporary index to
respect current caller-supplied positions without mutating persistent state or
memory. That diagnostic fallback costs O(population); ordinary per-agent decision
queries reuse the live index and do not scan the entire population.

Births retain their original timing and IDs. Each founder/newborn gets a fresh
empty component with the run's capacity/threshold, never parental memory. Queued
newborns are not visible to existing actors during that acting phase. They join
afterward, and first act/update memory on the next tick. Diagnostic observation
outside the phase can of course see the now-added living population, without
updating memory.

A remembered target may die remotely. No memory entry is deleted or updated
because of that death. Dead identities can remain until LRU eviction or the
observer's own death. No dead Human objects are retained for social history.
Hearing does not establish identity or update social memory.

## Memory, recency and event definitions

An immutable resident entry contains exactly:

```text
individual_id, first_seen_tick, last_seen_tick, encounter_count
```

`last_confirmed_alive_tick` is a read-only alias of `last_seen_tick`. There is no
actual-death field in agent memory. `first_seen_tick` and encounter_count refer
to the current residence episode: after forgetting, they restart. Encounter
count counts sightings, not separated contact episodes. One additional scalar
last-observation tick distinguishes consecutive opportunities from missed ones.
No position, lifetime edge history, eviction registry or full trajectory is kept
inside the agent.

Membership is determined at observation-batch start. Known sightings refresh
first, then unknown sightings allocate; IDs ascend within each phase. Hits
increase encounter_count and move to MRU. A miss allocates once and, if full,
evicts exactly the LRU entry. This prevents an earlier unknown allocation from
retroactively changing another simultaneous known sighting into a miss.
Diagnostics never refresh. Over-capacity crowds can still produce deterministic
churn and ID-order artifacts; no social preference is implied.

Let `L` be the target's last sighting, `P` the actor's preceding observation
opportunity, `t` the current tick and `T=20` the configured threshold:

| Condition | Agent-local classification |
|---|---|
| target absent from resident memory | first encounter / allocation |
| remembered, `L == P` | continuous contact |
| remembered, `L != P`, `t-L < T` | short return contact |
| remembered, `L != P`, `t-L >= T` | reunion |

The threshold is inclusive and measures elapsed time between sightings. Seeing
at tick 1, missing at opportunities 2..20, then seeing at 21 qualifies (gap 20,
19 intervening ticks). Continuous contact takes precedence even at threshold 1.
An unobserved time gap between consecutive supplied opportunities does not by
itself prove absence. Normal living agents have one opportunity each tick;
complete recordings validate that invariant. Multiple targets in one observation
are supported; duplicate opportunities/identities in one tick are rejected.

The agent treats an evicted identity as unknown. The privileged recorder has a
**separate bounded recent-eviction cache**, at most capacity entries per living
observer, to label some unknown contacts `memory_miss_after_eviction`. Its cache
is not available to the body/brain and is released on observer death/finish.
Unknown contacts outside that cache retain the local `first_encounter` label;
this is not a claim of lifetime novelty. Offline complete history distinguishes
all true lifetime first pairs from reappearances after forgetting. Unit cases
explicitly exercise a reappearance too old for the recorder cache.

`last_confirmed_alive_tick` is a direct sighting time. `actual_death_tick` occurs
only in a separate privileged death row. Both can have the same tick because
sighting can precede death within a sequential tick; record sequence resolves
that order. Absence, silence, memory eviction and disappearance never imply death.

## Streaming schema and integrity

`SocialRecord` is immutable. The CSV schema, version 1, is:

```text
sequence,tick,event_kind,observer_id,observed_id,observer_age,
observer_generation,memory_occupancy,perception_radius,dx,dy,
visible_count,known_visible,unknown_visible,elapsed_since_last_seen,
first_seen_tick,last_confirmed_alive_tick,encounter_count,trigger_id,
population,births,deaths,actual_death_tick,death_cause
```

Kinds are founder, birth, death, observation, first_encounter,
memory_miss_after_eviction, continuous_contact, return_contact, reunion,
eviction, tick_end and run_end. Blank columns are inapplicable, not zero.
Lifecycle rows use observer_id for the named individual and no observed_id.

- An observation row gives post-update occupancy and pre-batch known/unknown
  counts, including a row when no neighbors are visible.
- Its following contact rows each represent one hit or allocation; separate
  redundant hit/allocation rows are avoided.
- An eviction precedes the corresponding allocation contact in the stream;
  trigger_id links the incoming identity. Eviction preserves the removed entry's
  last direct sighting, without claiming the target's fate.
- Contact rows include relative dx/dy; observation rows include actual visual
  radius, allowing range checks. No auditory provenance is joined to identity.
- Founders and initial tick_end occur at 0 without observations. Each completed
  tick has population and cumulative births/deaths. Newborn lifecycle rows follow
  all existing actors; run_end terminates the stream.

Plain CSV and deterministic gzip are supported (empty stored filename, mtime=0,
compression level 1). Existing raw/sidecar paths are refused. When a social CSV is requested, CLI
also refuses collisions with any requested aggregate/event/communication/vocal
output, preserving old files. The completed sidecar is written exclusively only
after finish/flush/close. Interrupted files without a valid completion sidecar
are rejected by analysis.

Sidecars contain version/schema, seed, both effective configs and joint hash,
final tick/statistics, columns, record/event counts, definitions, capacity,
threshold, recording status, full file size/SHA-256, core source hashes and Python
version. Existing communication/vocal formats are unmodified. A combined CLI
control verifies identical communication and all four Vocal Q stream bytes with
social memory/recording ON versus OFF.

## Offline analysis and plots

```bash
python3 -m experiments.analyze_social_memory \
  --data-dir data/social_memory --output-dir analysis/social_memory
python3 experiments/analyze_social_memory.py --help
```

Analysis never instantiates or steps Simulation. Nonempty destinations are
refused; a replay requires a fresh output directory. Inputs are discovered by
CSV/gzip extension; metadata, not filename conventions, supplies seeds/configs.
Files are ordered deterministically. Conditions group all effective parameters
except random_seed; duplicate condition/seed inputs are rejected. Missing seeds
are acceptable; contributing finite n is always explicit.

Validation binds config/file/source hash declarations, full schema, monotonic
sequence/ticks, valid IDs, unique opportunities/contacts, liveness/birth eligibility,
relative visual range, age/generation, bounded LRU replay including cached
reappearance labels, record totals, contiguous completed ticks and population
accounting. Actual lifecycle events, not disappearance, establish deaths.
Partial/corrupt files are not silently filled or repaired.

Windows default to `(0,100]`, `(100,200]`, etc. Actual end, observed ticks and
partial-window flag are explicit. Tick 0 contributes initial population/lifecycle,
not an observation. Zero-tick completed runs have no windows. There is no
interpolation or post-extinction filling.

Each run is summarized before equal-seed means, sample SD (`ddof=1`), min/max and
finite n. No per-decision or per-individual significance inference is made. With
one seed SD is missing; with zero exposure rates are missing. Within a run:

- Occupancy is post-observation and opportunity weighted; it is not the final
  living-population mean (birth-tick newborns have no opportunity).
- Known/unknown/hit rates divide by directed visible contacts.
- Evictions divide by observation opportunities; full fraction uses opportunities.
- First encounters count bounded-memory misses, including known forgotten
  reappearances. Lifetime first pairs and all forgotten reappearances are separately
  reconstructed offline from complete observer-target history.
- Continuous persistence is continuous contacts divided by remembered contacts.
  Recontact intervals include forgotten pairs; reunion gaps require the defined
  resident-memory/absence criterion.
- Active observers, distinct visible targets, globally first-visible targets and
  explicitly dead IDs are separate sets. Population comes from completed tick rows.

Analysis streams raw records and buffers only a current turn plus bounded
mechanical memory per living observer. Separate offline last-contact maps support
exact distinctness/intervals and are released on explicit observer death/end.
The run-wide observed-ID set and scalar individual/window summaries can grow with
recording size; these are privileged offline data, never agent knowledge.

`analysis/social_memory/` contains seven CSVs:

- `social_by_run.csv`
- `social_by_window.csv`
- `social_by_individual.csv`
- `social_individual_windows.csv`
- `social_interval_distribution.csv`
- `social_seed_summary.csv`
- `social_window_seed_summary.csv`

Individual lifecycle summaries include zero-opportunity newborns with missing
rates and explicit birth/death/end reasons. Individual-window rows represent
actual observing agents only. Finalized dead individuals are streamed without
retaining Human objects. All exact column schemas and output hashes are in
`analysis_metadata.json`, along with inputs, full configs, source hashes,
software, definitions, window settings and completion.

Eight headless 180-dpi PNGs are generated:

| Figure | Interpretation |
|---|---|
| encounters_reunions_over_time.png | local memory misses and qualifying reunions, separated |
| memory_occupancy_over_time.png | post-opportunity occupancy/full fraction |
| memory_hits_evictions.png | explicit contact/opportunity denominators |
| known_unknown_visible.png | pre-batch resident knowledge, not sensing accuracy |
| reunion_interval_distribution.png | directed reunion-gap counts separately per seed |
| distinct_social_contacts.png | per-seed partner-count distributions, including zero-opportunity newborns |
| social_demography.png | completed population and explicit birth/death events |
| individual_social_trajectories.png | lowest seed, then lowest ID with an observation, independent of outcomes |

Mean lines/error bars/shading use equal-seed means and sample SD; points show
seeds, including when a short pilot has only one window. Distribution curves are
per-seed descriptions, not independent-agent replicates. A no-reunion case gets
an explicit empty-data message, not a fabricated distribution. These plots do not
imply friendship, cooperation or communication.

## Tests and regression verification

The final complete suite passes **477 tests**, including **68 new tests**; all
409 prior tests remain unmodified. The final suite took approximately 69.6 s.
No historical fixture values were regenerated.
The saved baseline, interim legacy regression and final logs/results are under
`data/v070_*tests*` and `data/v070_legacy_regression.log`.

New tests cover capacity/LRU, read-only diagnostics and RNG, pre-batch membership,
first/continuous/short-return/reunion distinctions, one missed tick, just below/
exactly/above the threshold, forgotten identities, multiple simultaneous targets,
zero/different radii, same-cell targets, immutable/private records, sequential
movement/death on the live indices, empty/newborn memory isolation and remote
unknown deaths. Capture/recording OFF creates no detailed contact payloads.
Finish also leaves private memory and RNG unchanged.

Matched feature OFF / ON / ON+recording runs check every tick for RandomBrain,
LearningBrain listening ON/OFF and its random-emission control. Comparisons cover
physical/vocal actions (including dying actors), body state/position/IDs/genomes,
parentage/generation/cooldowns, births/deaths, resources, signals, visual resources,
auditory reception/memory, both ordered Q tables, pending transitions and all
simulation/brain RNG states. Only the explicitly new social state/observation is
excluded from the ecological projection. Recording ON/OFF additionally compares
every social entry, recency order and last-observation tick. Global random state
is unchanged. Existing learning, random brain, genome and auditory implementation
files retain their original bytes.

Recording/analysis controls include valid completed zero-population/zero-tick
streams, continuous-only contacts, one asymmetric reunion, eviction/reappearance,
bounded-cache misses recovered offline, multiple observers/seeds, zero-opportunity
newborns, partial windows, explicit extinction, missing/corrupt metadata, config/
file hashes, duplicate contacts/IDs, self or invalid identities, false privileged
reappearance labels, illegal death information, unsorted ticks, incomplete
termination and output collisions. Single-seed SD stays missing; seed means are
not exposure weighted. Python and NumPy global RNG and inputs remain unchanged.
Offline tests patch Simulation.step to fail if called. Plain/gzip streams and
sidecars replay byte-identically. Combined legacy communication/Vocal Q recording
streams also retain exact data bytes with social recording enabled.

The bounded pilot checks complete ecological/RNG hashes and full private social
state at tick 0 and every tick through 100. All three modes agree ecologically;
enabled unrecorded/recorded modes also agree socially. Each seed's three ordinary
statistics CSVs are byte-identical.

For an additional pre-change control, the actual v0.6.4 commit was exported
read-only to a temporary directory. The same two seeds/configs ran for 100 ticks,
using the identical pure state projection. **Entire ecological/RNG trajectory
hashes match v0.7.0 disabled runs exactly**, not only final population. This starts
no old communication pilot or permutation analysis and rewrites no historical
file. `data/v070_prechange_benchmark.json` records hashes and timings.

The two completed pilot streams were analyzed again into a separate temporary
output. All **16 CSV/JSON/PNG outputs are byte-identical** in the same dependency/
plotting environment. A figure-only label/layout refinement is regenerated on
both copies without new simulation. Figure PNGs are visually inspected at 180 dpi.
The six historical fixture files and all snapshotted historical analysis/result
files retain their hashes: 281 historical/fixture/report files were checked.
Of all 358 snapshotted files, 350 are unchanged and eight have the intentional
changes listed above. `data/v070_verification.json` records per-file checks.

## Bounded pilot: observed results

The design was fixed before outcomes: unchanged `Config.evolution`, LearningBrain,
seeds 0/1, 100 ticks, default founder visual radius 2, social capacity 32, reunion
threshold 20. No resource, reward, behavior or demographic parameter was tuned.
The pilot helper refuses horizons above 100, more than two seeds and nonempty
destinations. Matched controls use the same seeds/configuration.

| Measurement | seed 0 | seed 1 |
|---|---:|---:|
| Initial population | 100 | 100 |
| Completed ticks | 100 | 100 |
| Observation opportunities | 13,768 | 14,102 |
| Directed visible contacts | 43,709 | 47,584 |
| Bounded-memory first encounters / allocations | 3,341 | 3,324 |
| Continuous contacts | 36,679 | 40,178 |
| Short return contacts | 3,294 | 3,664 |
| Qualifying reunions | 395 | 418 |
| Memory hit rate | .923563 | .930145 |
| Mean post-opportunity occupancy | 10.087812 | 10.085449 |
| Median occupancy | 9 | 9 |
| Maximum occupancy of any individual | 32 | 32 |
| Evictions | 32 | 0 |
| Forgotten reappearances (offline exact) | 0 | 0 |
| Mean reunion gap (ticks) | 34.493671 | 33.913876 |
| Median reunion gap | 29 | 29 |
| Births | 110 | 109 |
| Actual deaths | 3 | 1 |
| Final living population | 207 | 208 |
| Distinct active observers during run | 208 | 206 |
| Distinct visually observed targets during run | 208 | 206 |
| Lifecycle individuals recorded | 210 | 209 |
| Birth-tick/end-of-run newborns with no opportunities | 2 | 3 |

Seed 0 has one starvation and two dehydration deaths; seed 1 has one dehydration
death. No energy or old-age death occurs in this short pilot. Accounting is exact:
`initial_population + births - deaths == final_population` (207 and 208).
Active observers/targets are not the final population: they can include earlier
deaths and exclude newborns at the final tick. Their numerical equality here is
incidental; the analyzer maintains separate sets.

Equal-seed mean ± sample SD, **n=2**, without pooling contacts:

| Metric | Mean ± sample SD |
|---|---:|
| First encounters / allocations | 3332.5 ± 12.020815 |
| Reunions | 406.5 ± 16.263456 |
| Mean occupancy | 10.086631 ± .001671 |
| Hit rate | .926854 ± .004654 |
| Evictions | 16 ± 22.627417 |
| Mean reunion gap | 34.203773 ± .409977 |
| Births | 109.5 ± .707107 |
| Deaths | 2 ± 1.414214 |
| Final population | 207.5 ± .707107 |

Both seeds naturally produced qualifying reunions. Forgotten reappearances did
not occur during this limited pilot, despite 32 evictions in seed 0; deterministic
unit scenarios verify that mechanism, including beyond the recorder's cache.
Capacity was respected without tuning the ecology to create events. The default
100-tick windows yield one aggregate window per seed; finer descriptive windows
can be obtained from the **same CSVs**, for example `--window-size 10` with a fresh
output directory, without another simulation.

## Performance and storage

The same two seeds were compared in three modes. Numbers below are summed wall
time inside ordinary Simulation.step, excluding full-state/RNG verification and
ordinary statistics CSV writes. Recorded mode includes synchronous social CSV/
gzip work during steps. Construction and final gzip close times are separately
available in the pilot JSON. Mode order rotates each tick. These are local short
measurements, not repeated performance trials or universal guarantees.

| Implementation / mode | seed 0 step seconds | seed 1 step seconds |
|---|---:|---:|
| Actual v0.6.4 source control | .492110 | .506845 |
| v0.7.0 feature disabled | .499182 | .529865 |
| v0.7.0 enabled, recording OFF | .671085 | .719395 |
| v0.7.0 enabled, recording ON | 1.030221 | 1.086658 |

Relative to the current disabled path, enabled unrecorded cost 1.344×/1.358×;
recording cost 2.064×/2.051×. The disabled path is approximately 1.4%/4.5% above
the separately run old source control here. The small pre-change difference can
reflect branch/dispatch costs and timing variability; the processes/schedules
are not a repeated controlled microbenchmark. No zero-overhead claim is made.

Feature OFF allocates no per-agent social memory, visual social records, spatial
index or recorder. Only small per-simulation optional settings/references and
conditional hooks remain. Enabled memory is O(living population × capacity),
plus the existing/optional spatial index and temporary local contacts. Logical
peak private entries were 3,245/3,313; the privileged recent-eviction cache peaked
at 30/0 IDs. Both are bounded per living observer, with no dead Human retention.
Very dense local contacts can still require quadratic output in the number of
co-located individuals; the index cannot eliminate actual visible pair volume.

The complete three-mode verification harness took **203.32 s**, mostly whole-Q/
RNG serialization and hashing outside the timed steps; all six step totals sum
to about **4.54 s**. Whole-process peak RSS was **103.84 MiB**, including three
concurrent simulations and validation snapshots, not a per-mode memory estimate.
It must not be attributed to the social component alone.

Offline analysis took approximately **2.65 s** with whole-process peak RSS
**150.70 MiB** including pandas/matplotlib. The raw gzip recordings are 895,732
and 963,913 bytes; with their sidecars, about **1.88 MB**. Six ordinary statistics
CSVs and design/verification receipts bring the pilot directory to about
**2.08 MB**. The 16 analysis outputs occupy about **0.80 MB** (PNG layout can
slightly affect size). No full Q snapshots or complete agent histories are saved.
Timings are separate from deterministic analysis metadata.

Saved artifacts:

- `data/social_memory/seed_0_social.csv.gz` and `seed_1_social.csv.gz`, each with
  `.metadata.json` completion sidecar.
- `data/social_memory/statistics/`: six matched ordinary CSVs.
- `data/social_memory/pilot_design.json`, `pilot_summary.json`.
- `data/v070_design.json`, `v070_before.json`, baseline/final test JSON/logs,
  pilot/analysis/replay logs and timing receipts, old-source comparison and
  `v070_verification.json`.
- `analysis/social_memory/`: seven CSVs, eight PNGs and complete metadata.

Generated data/analysis artifacts remain saved locally under the established
Git-ignore policy. Historical results are not regenerated or overwritten.

## Scientific interpretation and limitations

**Established:** perfect local visual ID delivery at the correct sequential
point, bounded independent knowledge, deterministic forgetting, correctly
classified reunions, separation of sighting/absence/death, strict recording and
reproducible analysis. Enabling memory or recording leaves action/Q/RNG/ecological
trajectories unchanged, including comparison to actual v0.6.4 source.

**Not established:** social motivation, social reward usefulness, learned
recognition, cooperation, friendship, partner preference, shared meanings,
communicative intent or language. There is no new reward or behavioral input,
so these observations cannot demonstrate a causal benefit of social memory.
Even many recurrent contacts are not social relationships in this model.

The two-seed, 100-tick pilot is a mechanism check, not a hypothesis test or a
long-term demographic study. Spatial overlap, local resources, inherited visual
radii, birth on a parental tile, population growth and selective survival all
shape contacts. High hit rates describe resident memory membership, **not**
measured recognition accuracy (recognition is assumed perfect). Similar mean
occupancy across these two seeds does not establish an optimal cognitive limit.

Directed contacts/reunions need not be mutual. Occupancy averages weight observation
opportunities and mix ages/generations; final-tick newborns have no opportunity.
Reunion intervals are conditioned on an observed remembered return. Long gaps
without return are censored, not evidence of death. Finite capacity and deterministic
within-batch ordering select which identities remain available for reunion
classification. The bounded recorder cache recognizes only some forgotten
reappearances; offline lifetime reconstruction is separate privileged analysis.
No absent acquaintance gains a simulator-provided death label.

## Future directions — documentation only

The implementation is ready for **v0.7.1 design review**, not automatic reward
activation. A later explicit experiment could compare physiological reward only
against physiological reward plus a small configurable reunion reward, using
predefined seeds/horizon/outcomes and continuous-contact suppression. Reward must
use permitted resident knowledge, never the observer's lifetime/eviction/death
registries. Define the elapsed-absence convention, potential cycling/farming,
capacity effects and physiological reward scale before implementation; measure
resource access, survival and reproduction separately from reunion counts.

For v0.7.2, test whether otherwise arbitrary signals can help reunion without
rewarding emission/reception directly, and whether both sender and receiver can
obtain downstream benefits through changed behavior. Do not assume that memory
or shared physiological reward already provides this mechanism.

A separate future reproductive-type acoustic manipulation could investigate
partner finding/recognition/population persistence. It is not the historical
v0.6.1 ablation C and does not equate reproductive type with individual identity;
no type-specific or unique individual voices are added here. Familiar-voice
recognition would require explicit acoustic similarity, familiarity learning and
perceptual uncertainty. Future self-masking could create a speaking opportunity
cost, but remains unimplemented. No new acoustic features, social rewards,
identity-to-signal mappings or v0.7.1 behavior are introduced in this release.
