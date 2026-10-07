# Artificial Civilization Simulation — v0.6

A headless, standard-library-only artificial-life simulation for experiments in
emergent behavior. v0.6 adds learned vocalization selection through a second,
independent Vocal Q head in LearningBrain. Physical actions and vocal choices
occur in the same turn and receive the same existing physiological reward.
The v0.5 masking, auditory memory and learned listening remain intact, as do
survival, reproduction, inheritance, mutation, resources and lineage systems.
RandomBrain remains the ecological control; a sender ablation preserves v0.5
random emissions exactly.

**v0.6 implements learnable vocal choices, not language or shared communication.**
Signals have no predefined meaning. Agents cannot select receivers or observe
whether they were heard. There are no communication rewards. Agriculture, economics, emotions, religion,
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

# Short runs in the same ecology. Random remains the default.
python3 main.py --mode evolution --brain random --seed 0 --ticks 100 --csv data/v06_random_smoke.csv
python3 main.py --mode evolution --brain learning --seed 0 --ticks 100 --csv data/v06_vocal_smoke.csv
# v0.5-equivalent listening with random production.
python3 main.py --mode evolution --brain learning --no-learning-controls-vocalization --seed 0 --ticks 100 --csv data/v06_sender_control.csv
# Original visual encoder and random production.
python3 main.py --mode evolution --brain learning --no-learning-uses-auditory --no-learning-controls-vocalization --seed 0 --ticks 100 --csv data/v06_both_controls.csv

# Hold perception fixed in every generation while the other traits evolve.
python3 main.py --mode evolution --brain learning --fixed-perception-radius 2 --seed 0 --ticks 100 --csv data/v06_fixed_p2_smoke.csv

# Controlled repeated experience, independent of evolutionary claims.
python3 -m experiments.learning_demo

# Three-tick sensory mechanism demonstration; no learned signal meanings.
python3 -m experiments.auditory_demo
python3 -m experiments.masking_demo
python3 -m experiments.listening_demo
python3 -m experiments.vocal_learning_demo
# Initial zero-reward tied policy + one short seed for the four ablations.
python3 -m experiments.vocal_verification

# Short auditory diagnostics. Vocabulary 16 retains the historical repertoire.
python3 main.py --mode evolution --brain learning --seed 0 --ticks 10 \
  --hearing-radius 8 --events data/v06_auditory.jsonl
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
historical controls. The pre-v0.6 suite of 194 tests was run before changes.
Historical state digests are retained; historical learned-brain controls now
explicitly disable learned production. See [v0.6 verification](#v06-verification-and-compatibility),
[v0.5 verification](#v05-verification-and-compatibility)
and the retained [v0.4 verification](#v04-verification-and-compatibility).

## Architecture

```text
main.py                    CLI, modes, output ownership, stopping conditions
config.py                  Validated immutable Config, presets, trait limits
world/tile.py              Integer food and water units
world/world.py             Grid, generation, movement, consumption, regeneration
agents/human.py            Body, physiology, genome reference, lineage metadata
agents/genome.py           Immutable Genome, inheritance, bounded mutation
agents/brain.py            Brain protocol, Action, Decision, RandomBrain
agents/learning.py         Independent bounded Physical Q and Vocal Q heads
agents/auditory_memory.py  Time-bounded non-silent auditory events and read-only encoder
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
tests/test_listening.py     Masking, memory, listening, ablations and v0.5 fixture
tests/test_vocalization.py  Dual heads, shared reward, RNG isolation, controls and v0.6 fixture
tests/auditory_control.py    Full ecological state/action digest helpers
tests/fixtures/             Pre-v0.3 and pre-v0.4 deterministic control digests
experiments/learning_demo.py Controlled repeated-resource learning validation
experiments/auditory_demo.py Three-tick sensory mechanism check
experiments/masking_demo.py Five deterministic sensory masking examples
experiments/listening_demo.py Controlled delayed sound/action association and ablation
experiments/vocal_learning_demo.py Repeated physiological reward / arbitrary vocal preferences
experiments/vocal_verification.py Initial tied policy and short 2x2 descriptive checks
experiments/analyze_fixed_perception.py Offline CSV aggregation and PNG plots
tests/test_analyze_fixed_perception.py Analysis fixtures, missing data, and reproducibility
requirements-analysis.txt   Optional pandas/matplotlib dependencies
```

The existing architecture is extended rather than replaced. A brain implements
`choose_action(human, observation) -> Decision`. It reads its own body and local
observation and returns a decision without mutating the body. Observations contain
`tick`, nearby `(dx, dy, food, water)` records, and one resolved
`AuditoryPercept`: SILENCE, MASKED, or IDENTIFIED with
`HeardSignal(signal_id, source_direction)`. They contain no sender
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
- When `learning_uses_auditory=True` (the v0.5 default), one nested feature for
  the latest retained non-silent auditory event: ID, coarse direction, and age,
  or MASKED and age, or SILENCE/NONE. See the exact encoding below. The disabled
  condition keeps the original seven-integer state exactly.

The encoder reads only the immutable Observation and the individual's body.
Resources outside the inherited perception radius cannot affect these features.
It ignores exact resource quantities beyond presence, exact distances after
nearest selection, absolute position, absolute tick, IDs, lineage, and reproductive
partners. Auditory age is relative elapsed time. It has no world reference, map, pathfinding, or predefined
direction-to-action preferences. Larger perception can supply additional local
information, but receives no reward bonus or extra metabolic cost.

Physical Q memory is an individual `OrderedDict` from state tuples to action-value lists,
bounded to `learning_memory_capacity=256` states. Each row has seven action values,
or eight when reproduction is enabled. Access during learning/action selection
updates recency; the least recently used row is evicted at capacity. Each enabled
head has its own pending transition containing state, action, body snapshot and
reward. Vocal Q has a separate bounded table, detailed below. No
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
fitness reward. RandomBrain and the sender-ablation LearningBrain signal using
the inherited probability. By default LearningBrain instead selects SILENCE or
a signal with Vocal Q; it never gates that choice with `signal_probability`.
Auditory input changes both heads' state only in the listening-enabled condition.

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

## Visual and auditory perception (v0.4 baseline)

The following section records the v0.4 mechanism and verification. v0.5 preserves
its physical channel, timing, geometry, and privacy while adding perceptual
masking and optional learned listening, documented in the next section. Statements
about ignoring auditory input and identifying all candidates describe v0.4.

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
needs a separate named RNG stream. The v0.5 extension below implements masking
and learned listening; these baseline measurements remain historical records.

## Learned listening, masking, and finite auditory memory (v0.5)

This section preserves the v0.5 receiver baseline and its historical results.
In v0.6, use `learning_controls_vocalization=False` to retain this sender protocol;
the receiver mechanism below is unchanged.

The receiver can now associate an arbitrary sound with future experience through
ordinary individual Q-learning. Senders still make independent random emissions
using inherited `signal_probability` and random integer IDs. There is no learned
production, receiver selection, communication reward, meaning dictionary, shared
convention, teaching, imitation, identity recognition, language, or culture.

The layers are deliberately separate:

```text
physical candidates within hearing_radius
    -> distance-based apparent loudness
    -> perceptual masking
    -> one immutable resolved auditory percept
    -> finite non-silent-event memory
    -> one auditory Q-state feature
    -> existing Q-learning and action selection
```

Future experiments can independently vary physical hearing, masking, memory,
encoding, and cognitive access. Masking is neutral sensory physics before learning;
thirst, hunger, reward, usefulness, familiarity, and Q values do not change which
sound is identified or how long it is retained.

### Configuration and masking

| Setting | Default | Validation / role |
| --- | ---: | --- |
| `hearing_radius` | 8 | Existing nonnegative integer physical range; nonheritable |
| `auditory_masking_ratio` | 3.0 | Finite number strictly greater than 1 |
| `auditory_memory_ticks` | 4 | Integer at least 1; time horizon, not lifetime history |
| `learning_uses_auditory` | True | Boolean; False preserves the v0.4 Q encoder |
| `signal_vocab_size` | 16 | Existing positive integer; explicitly selectable as 40 or larger |

The three new settings are Config parameters, not genome traits; no mutation,
inheritance, or new biological cost is introduced. The loudness exponent is fixed
at 2. All emitters have the same intensity. For Chebyshev distance d:

```text
L(d) = 1 / (d + 1)^2
d:     0      1       2       3       4
L:     1     1/4     1/9     1/16    1/25
```

This is abstract apparent loudness for competition, not accurate acoustic
physics. Every external signal inside the existing inclusive hearing radius
still physically arrives. Self and out-of-range signals cannot compete. The
resolver finds the two strongest candidates in one scan:

- No candidates: SILENCE.
- One candidate: IDENTIFIED with its arbitrary signal ID and coarse direction.
- At least two candidates: IDENTIFIED with the strongest only when
  `L1 / L2 >= auditory_masking_ratio`; otherwise MASKED.

Equal-distance candidates have ratio 1 and therefore mask each other, even with
identical IDs. Sender IDs and traversal order cannot break a perceptual tie. Three
or more speakers use exactly the same top-two rule; the resolver does not sum
loudness or impose a special speaker-count threshold. Internally the ratio is
computed as `(d2 + 1)^2 / (d1 + 1)^2`, algebraically identical to L1/L2 and avoiding
intermediate reciprocal rounding. There are no stochastic draws, noise,
obstacles, attenuation, variable volume, or intensity costs.

`python3 -m experiments.masking_demo` verifies:

| Example | Candidate distances | Dominance ratio | Percept |
| --- | --- | ---: | --- |
| A | 1 and 3 | 4 | Nearer ID identified |
| B | 1 and 2 | 2.25 | MASKED |
| C | 2 and 2 | 1 | MASKED |
| D | 2 only | — | IDENTIFIED |
| E | None | — | SILENCE |

### Sensory record, memory, and exact Q feature

`Observation.auditory` is an immutable/slotted tagged record:

```python
AuditoryPercept(kind=AuditoryKind.SILENCE, signal=None)
AuditoryPercept(kind=AuditoryKind.MASKED, signal=None)
AuditoryPercept(kind=AuditoryKind.IDENTIFIED,
                signal=HeardSignal(signal_id, source_direction))
```

Tags are string enums, not reserved signal IDs. SILENCE means no external physical
candidate arrived. MASKED means candidates arrived but identification failed;
its direction is unknown and no ID/direction payload is permitted. IDENTIFIED
uses the same v0.4 sign-sector compass directions, including SAME_CELL. The
record has no speaker identity, position, dx/dy, distance, loudness, intensity,
lineage, relationship, or semantic label. `Observation.signals` remains a
read-only zero-or-one identified-sound view; use the tag to distinguish MASKED
from SILENCE. The legacy `Communication.hear()` is a physical-candidate diagnostic
view only; `Simulation.observe()` gives the Brain `Communication.perceive()`'s
resolved result, never the complete candidate list.

Each LearningBrain owns an `AuditoryMemory`. Raw entries are immutable
`AuditoryEvent(received_tick, percept)` objects in a deque bounded by the horizon.
It stores only IDENTIFIED and MASKED events, at most one per tick; silence does
not add an entry or erase a still-recent event. Every ordinary decision expires
entries whose `current_tick - received_tick >= auditory_memory_ticks`. Skipped
tick numbers count as elapsed time. Age 0 is the current receiving tick; the
default retains ages 0, 1, 2, 3 and forgets at age 4. Repeated non-silent observations
at the same tick replace that tick's event rather than growing the deque;
backward memory updates are rejected. No raw lifetime sequence is retained.

For example, an identified ID 12 from E received at tick 100 survives SILENCE
at ticks 101, 102, and 103 with ages 1, 2, and 3. At tick 104 it is forgotten.
MASKED follows the same rule and can replace an older identified event as the
latest event used by learning. Every newborn has empty auditory memory, empty
Q rows, and no pending transition. None of these are inherited. RandomBrain
allocates no auditory memory.

The baseline encoder uses only the latest retained non-silent event:

```text
identified: (AuditoryKind.IDENTIFIED, signal_id, SourceDirection, age)
masked:     (AuditoryKind.MASKED, age)
empty:      (AuditoryKind.SILENCE,)

original = (hunger_bin, thirst_bin, energy_bin,
            food_dx_sign, food_dy_sign, water_dx_sign, water_dy_sign)
enabled_state = (*original, auditory_feature)
disabled_state = original
```

The original components are neither removed nor redefined. Raw past events
remain available for later encoder experiments, but the complete sequence is
not a Q key. With vocabulary 16, nine directions, and horizon 4 there are at
most `1 + 4 + 16*9*4 = 581` auditory feature values. The existing 256-row Q-table
limit still applies. This larger state space can fragment experience and cause
LRU eviction; richer sensory input need not improve outcomes.

Memory updates once at the ordinary `choose_action` call before deriving the
next Q state. The preceding transition then uses that state in the existing
discounted update; action selection, random emission, and outcome feedback retain
their order. `state()` and `action_values()` are read-only previews of the supplied
percept plus retained memory: no event insertion/expiration mutation, Q/LRU
update, or RNG draws. Inspecting a future preview does not teach the brain.

`learning_uses_auditory=False` excludes the entire auditory feature and returns
the exact seven-integer v0.4 key. Physical delivery, masking, and raw short-term
memory still operate. This is a cognitive-input ablation in the same sensory
world, not a switch that makes the world silent. The CLI supports
`--learning-uses-auditory` and `--no-learning-uses-auditory`; the horizon and
masking ratio are programmatic Config settings.

The reward formula remains exactly:

```text
r = (hunger_before - hunger_after) / max_hunger
  + (thirst_before - thirst_after) / max_thirst
  + (energy_after - energy_before) / max_energy
```

No reward is given for hearing, responding, approaching a speaker, identifying
sounds, avoiding masking, or emitting. The same Q-learning discount may propagate
later physiological reward through preceding actions; there is no sound-specific
credit assignment or supervised label. Emission code and its independent RNG
are unchanged and consult neither Q values nor auditory memory.

### Controlled learned-listening result

`python3 -m experiments.listening_demo` runs 300 six-action trials per condition
in a 5×1 world with one learner, no reproduction and no age death. Between trials
the harness places the receiver at x=2 with thirst 60 and one water unit at x=4
or x=0 in alternating contexts. It presents arbitrary ID 7 or 12 from the same
direction E using an external test emitter, only on the trial's first tick.
Visual radius zero makes the two initial resource observations identical. The
cue is remembered while later movement/drinking receives ordinary physiological
feedback. The matched ablation uses the same physical cues and context schedule.

The harness changes the environment/body between trials, as the original
learning demo does. It never edits Q values, forces actions, supplies meaning
labels to the brain, or changes rewards. Pending transitions persist across
trial resets; this is a continuously learning intervention, not isolated episodic
training or a natural ecological experiment.

With seed 0, all initial Q values were zero. At the final starting-state probe,
the enabled receiver's ID-7 MOVE_EAST value was **0.142586**, higher than every
other action for that cue. In the last quarter it first moved east in **29/37**
ID-7 trials, versus **0/38** ID-12 trials. ID-7 and ID-12 action values differ.
The ablation has exactly identical probe values for both IDs because its encoder
cannot distinguish them. Each condition completed **1,799 Q updates**.

The enabled receiver consumed **69** water units (63 in ID-7 contexts, 6 in
ID-12 contexts), versus **111** in the ablation (111 and 0). It did **not** learn
the expected westward initial preference for ID 12; its final greedy action
there was a blocked MOVE_NORTH. Thus the demo establishes a learned association
with an arbitrary sound and a useful future action for one condition. It does
not establish that listening is globally superior, that an ID means water,
that production is intentional, or that a convention, language, or ecological
fitness advantage emerged.

### v0.5 verification and compatibility

The unchanged v0.4 suite passed **169 tests before behavior changes**. The final
suite passes **194 tests** (Python 3.11.9), including new masking, strict privacy,
time/age expiration, masked memory, silence, boundedness, fresh offspring,
unchanged reward/emission, cognitive ablation, logging/RNG isolation, and demos.
The original v0.2 and v0.3 fixtures are unchanged. Tests that asserted auditory
ignorance explicitly select the ablation; serialization assertions use the new
tagged record. Unrelated biological tests retain their original assertions.

Before implementation, additional seed-7, 80-tick, 12-founder renewable controls
were frozen from v0.4 commit `ea53e96ed7d1eb00beea759002ca8f5c1551bf5d` in
`tests/fixtures/v04_listening_control.json`. RandomBrain and disabled listening
match every captured state and action/outcome digest: bodies/genomes, resources,
signals, aggregate counters, Q values/LRU, pending transitions, and all retained
RNG states. The existing seed-0 100-tick historical controls still match at
hearing radii 3 and 8. No historical digest was regenerated to accommodate v0.5.
Enabled listening has its own `v05_listening_seed7_tick80.json`, including raw
auditory memory and complete resolved/diagnostic event hashes.

| Seed-7 short control at tick 80 | Population | Births | Deaths |
| --- | ---: | ---: | ---: |
| RandomBrain (matches v0.4) | 39 | 32 | 5 |
| LearningBrain, auditory disabled (matches v0.4) | 41 | 35 | 6 |
| LearningBrain, auditory enabled (new v0.5 fixture) | 37 | 30 | 5 |

Actual seed-0 100-tick CLI smoke runs also compared all **101 snapshots** of
every aggregate against the archived v0.3 CSVs: Random and the learning ablation
match exactly. The enabled run intentionally differs:

| Seed-0 evolution smoke at tick 100 | Population | Births | Deaths | CLI seconds |
| --- | ---: | ---: | ---: | ---: |
| Random | 232 | 136 | 4 | 0.267 |
| Learning ablation | 207 | 109 | 2 | 0.366 |
| Learned listening | 206 | 108 | 2 | 0.386 |

These are short mechanism checks, not comparative ecological evidence or large
benchmarks. Local timings include startup/CSV and are not portable throughput
guarantees. Masking scans only candidates in the existing spatial buckets, in
linear candidate time with constant extra space. Raw memory uses O(horizon)
space per learner; the Q-table cap is unchanged. Full diagnostic serialization
does additional candidate scans and can be expensive when many sounds overlap.

Detailed JSONL remains opt-in/streaming; the default stores no event history.
`run_started.auditory_schema_version` is now **3**. The canonical observation
serializes `auditory.kind` and its optional two-field `signal` only. Privileged
`auditory_diagnostics` additionally records candidate loudness, alongside the
existing identity/position/distance/timing fields. `auditory_resolution` records
the strongest/runner-up loudness, dominance ratio, masking ratio, and resolved
percept. Learning records add `auditory_learning` with the bounded memory,
relative ages, and supplied feature (None for the ablation). All diagnostic
payloads are built only for an enabled full event sink. Lineage-only logging
does not construct them. No aggregate columns or inferred semantic counters
were added; `mean_memory_size` still means Q-table rows, not auditory events.

Intentional incompatibilities: LearningBrain's default now uses auditory input,
so its Q keys and ecological trajectory may differ from v0.4. Loading an old
Config alone now enables the new default; use
`Config(**old_settings, learning_uses_auditory=False)` to replay old cognitive
behavior. The third Observation constructor argument is now an AuditoryPercept;
raw candidate tuples are rejected. The `.signals` read view can expose only one
identified sound. Observation/event JSON and Config hashes/metadata change;
CSV metric names remain compatible. Existing analysis inputs and the preserved
perception baseline study have not been modified. Historical perception commands
below explicitly disable auditory learning to keep their original protocol.

Local verification artifacts (ignored by Git) are `data/v05_listening_demo.json`,
`data/v05_masking_demo.json`, `data/v05_short_runs.json`,
`data/v05_verification.json` (source hashes, test/control results, demos), the three
`data/v05_*_smoke.csv` files and metadata, and short full auditory JSONL.
The fixtures, demos, tests, and this documentation are retained in source.

Possible later work, **documented only**: learned signal production or receiver
selection; whispering/shouting or variable intensity; learned attention and
need-dependent hearing; semantic memory; voice/speaker/kin recognition and
names; signal sequences, compositionality, grammar; phonetic feature vectors
and acoustic similarity/confusion; stochastic hearing; obstacles and terrain
attenuation; cultural transmission, inherited learned knowledge, imitation,
and teaching. The next design decision should follow review of this receiver
mechanism. No v0.6 learned signaling or new 5000-tick multi-seed experiments
were run.

## Learned vocalization selection (v0.6)

LearningBrain now selects two concurrent components from one encoded state:

```text
                         state
                       /       \
              Physical Q       Vocal Q
                  |               |
           physical action     vocal action
                  +-------+-------+
                          |
              ordinary action + physiology
                          |
                 one existing body reward
                  +-------+-------+
             Physical update   Vocal update
```

A physical action is never replaced by speaking. Moving, eating, drinking,
waiting and reproducing can each accompany a vocal action. There is no speaking
cost, delay, receiver targeting, knowledge of delivery, or new biological rule.

### Configuration, representation and bounded heads

| New setting | Default | Validation / role |
| --- | ---: | --- |
| `learning_controls_vocalization` | True | Boolean; nonheritable; LearningBrain only |
| `vocal_learning_memory_capacity` | 256 | Positive integer; independent LRU state-row cap |

Physical action order and indices remain N, S, E, W, EAT, DRINK, WAIT, followed
by REPRODUCE only when enabled. Physical Q remains the original
`_values: OrderedDict[State, list[float]]`, with 7 or 8 values per row and the
unchanged `learning_memory_capacity=256`. Vocal Q is separately
`_vocal_values: OrderedDict[State, list[float]]`, with vocabulary+1 values per row.
There is no physical×vocal Cartesian product and no shared eviction pool.
Both tables move rows to the most-recent end during action selection/updates,
evicting the oldest row at their own capacity. Diagnostic probes do neither.
Newborns receive two empty tables, no pending transitions, fresh private RNGs,
and zero updates. Neither table nor auditory memory is inherited.

The frozen/slotted production record is:

```python
VocalAction(kind=VocalKind.SILENCE, signal_id=None)
VocalAction(kind=VocalKind.SIGNAL, signal_id=arbitrary_integer)
```

It is distinct from an incoming `AuditoryPercept`. SILENCE has no fake integer
ID. Enabled brains enumerate index 0 as SILENCE and index i+1 as signal i.
Vocabulary sizes 4, 16, 40, 256 therefore give 5, 17, 41, 257 choices; integer
neighbors have no acoustic similarity. `Decision(action, signal_id=None)`
keeps its existing constructor and fields. Its read-only `vocal_action` property
provides the explicit tagged view of the concurrent production component.

Both heads receive the **same state object**, encoded once per turn: the
original seven body/visual integers, optionally plus the v0.5 latest retained
auditory feature. No sender identity, intended receiver, communication-success
flag, or additional social feature enters either state. Hearing radius 8,
Chebyshev geometry, loudness `1/(d+1)^2`, masking ratio 3, coarse directions,
privacy, and the four-tick non-silent memory are unchanged.

All unseen Vocal Q values are zero, including SILENCE. Vocal choice reuses
`learning_epsilon=0.2`: one private random draw selects exploration, which is
uniform over all choices; otherwise a uniform random choice among maximum-valued
indices breaks ties. There is no preferred ID or SILENCE prior. Physical epsilon
selection is unchanged. The vocal head reuses alpha 0.2 and discount 0.9.

### Independent sender/receiver controls and RNG

| Condition | `learning_uses_auditory` | `learning_controls_vocalization` | Mechanism |
| --- | --- | --- | --- |
| A | False | False | Visual/body learning, random production; historical control |
| B | True | False | v0.5 learned listening, random production |
| C | False | True | Learned production, auditory feature excluded |
| D | True | True | Learned listening and production; v0.6 LearningBrain default |

Neither flag is heritable. RandomBrain ignores both learning flags and retains
its original draws/action ordering. Sender OFF does not disable reception,
masking, memory or receiver learning. Example ablation flags:

```bash
# B: preserve v0.5 sender behavior
python3 main.py --mode evolution --brain learning --no-learning-controls-vocalization --seed 0 --ticks 100
# C: learn production while excluding auditory features
python3 main.py --mode evolution --brain learning --no-learning-uses-auditory --seed 0 --ticks 100
```

Learned production is the only emission path when enabled: its chosen signal
emits, and SILENCE emits nothing. **It does not consult inherited
`signal_probability`.** The trait is retained unchanged for RandomBrain and
sender-OFF controls, where the exact v0.5 independent emission gate and uniform
signal-ID choice still apply.

Simulation still supplies each brain `Random(f"{seed}:brain:{id}")`. LearningBrain
preserves the original single `getrandbits(64)` construction draw as `signal_seed`.
The legacy `signal_rng = Random(signal_seed)` is unchanged. Enabled brains also
construct `vocal_rng = Random(f"{signal_seed}:vocal-choice")`, without drawing
from either historical stream. Disabled brains allocate no vocal RNG/action rows.
The original brain RNG continues to select physical actions; the dedicated vocal
RNG performs only vocal epsilon/choice draws. Enabling Vocal Q therefore cannot
shift physical draws merely through its extra random choices. Later auditory
states/actions/ecology may differ when listening is ON. No global random state
or hearing-delivery RNG is used.

### Shared reward, transitions and timing

The simulator's tick order is unchanged. At the next ordinary observation,
memory is updated and the current state encoded. The previous physical
transition is updated and a physical choice made; the enabled vocal head
independently updates its previous transition and makes its choice from that
same state. These private head operations finish before physical execution.
Physical execution is followed by post-action emission, ordinary physiology,
death evaluation, and immutable body feedback. Newborn timing and regeneration
are unchanged. A signal from tick t is available only during tick t+1, expires
after that receiving phase, excludes its sender, and survives that sender's death.

`_pending` and `_vocal_pending` are separate `Transition` objects sharing the
same immutable `before` snapshot/state. Outcome feedback computes exactly once:

```text
r = (hunger_before - hunger_after) / max_hunger
  + (thirst_before - thirst_after) / max_thirst
  + (energy_after - energy_before) / max_energy
Qv(s,v) += alpha * (r + gamma * max_v' Qv(s_next,v') - Qv(s,v))
```

The identical scalar is attached to both transitions. Living updates wait for
the next actual observation; death immediately updates both with bootstrap zero.
The last living turn remains pending at a finite cutoff. No additional future
observation or sound-specific temporal mechanism is introduced.

**This is coarse shared credit, not causal attribution.** DRINK+SIGNAL_12 can
give both selected choices positive reward because the body benefited from
drinking. It does not show that SIGNAL_12 caused that benefit. There is no reward
for being heard, matching an ID, receiver behavior, social proximity, reduced
masking, conventions, or communication. Feedback contains only body values and
liveness; the sender receives neither delivery status nor receiver outcomes.
A state-dependent vocal preference is an arbitrary policy association. Signal
semantics and a shared communication convention would require separate evidence.

### Diagnostics, initial activity and controlled result

CSV columns are unchanged: existing learning metrics refer to **Physical Q**,
and `signals_emitted` counts actual non-silent emissions only. Opt-in events
already suffice to validate the new mechanism without expanding aggregate CSVs.
Each learner's `agent_step.details.vocal_learning` records learned/random-control
path, tagged selected action, epsilon branch (None in random control), cumulative
vocal updates, updates during this turn, vocal memory size, branch counts, and
selected pre-outcome Q value (None in random control). Existing `action` and
`learning` record physical choice, branch, reward, updates and table size.
Full Q tables are never serialized per turn. Payloads are built only for a full
event sink; diagnostic reads draw no RNG and change no Q/LRU/memory/body state.

Zero tied values mean the initial policy is uniform over SILENCE plus all IDs.
With vocabulary 16, expected initial emission probability is **16/17≈94.12%**,
much higher than the historical 10% random gate. A controlled seed-0 check of
1,000 identical-state, zero-reward decisions measured **954 emitted / 46 silent
(95.4%)**. Q values stay zero throughout this measurement; it is not a 1,000-tick
evolution run. No silence bias, probability gate or activity tuning was added.

`python3 -m experiments.vocal_learning_demo` runs 600 ordinary one-agent,
one-cell decisions, alternating two distinguishable body/resource contexts:
(hunger 60, thirst 0, local food 1) and (hunger 0, thirst 60, local water 1),
energy 100. The harness resets the body/resources between trials but never
edits Q, forces actions or changes rewards. Pending transitions persist across
resets. There is **no receiver**; listening is OFF to isolate production.

Seed 0 started with all values zero. Final probe maxima were arbitrary
**SIGNAL_15** in context A and **SIGNAL_5** in context B, with different vocal
value rows, 599 updates in each head and two rows per table. The body consumed
243 food and 50 water units. This demonstrates state-dependent Qv associations
under shared physiological reward. It does not establish signal meaning,
causality, receiver benefit, a convention or stable behavior in free ecology.
No preferred ID is coded; other seeds are free to produce different results.

### v0.6 verification and compatibility

Before edits, all **194 tests passed in 58.544 s** on Python 3.11.9 at source
commit `02aaf3169e879c9582f40ced83720d27a411dd4c`. Their fixture hashes and the
complete old listening-demo result were saved before implementation. After changes,
all **236 tests pass** (194 existing + 42 new, 71.409 s in the recorded run). New tests
cover concurrent actions, reproduction/newborns, timing/silence, state/privacy,
independent LRU/epsilon, bootstrap/terminal updates, shared reward, sender/receiver
ablations, full dual-head replay, RNG/logging/CSV isolation and diagnostics.

All four historical JSON fixtures remain byte-identical. RandomBrain and
sender-OFF controls match the old v0.2/v0.3/v0.4 state/action digests. Listening-ON,
sender-OFF matches the full v0.5 80-tick seed-7 state/memory/Q/LRU/pending/RNG and
event digests: population 37, births 30, deaths 5. The event comparison projects
out only the two new Config fields and the new `vocal_learning` diagnostic; it
retains every old event field and every original state assertion. Historical
tests explicitly select sender OFF rather than replacing their fixture values.
The updated CLI version assertion is intentionally 0.6.

The v0.5 listening demo explicitly keeps sender learning OFF. Its entire result
matches the pre-change result, apart from the two new Config fields: ID-7 E value
0.14258632186131284, water 69 versus 111 in the auditory ablation, and 1,799 updates
per condition. `v06_vocal_seed7_tick80.json` is a separate enabled-vocal fixture
with complete dual-head/memory/RNG/event hashes. Logging/extra probes/CSV are
checked against a nonlogging replay at every tick; global random state is intact.
The v0.6 seed-7 80-tick fixture ends with population 32, births 30 and deaths 10;
this enabled-sender divergence from v0.5 is intentional. Actual CLI RandomBrain,
both-OFF learning, and sender-OFF/listening-ON runs each match **all 101 aggregate
snapshots** of their archived v0.5 100-tick CSVs, comparing every statistics
column and excluding only identifying metadata/config hashes.

One seed-0, 100-tick evolution check per condition, with unchanged ecological
parameters and streaming diagnostics:

| Condition | Population | Births | Deaths | Signals | MASKED / all observations | Physical updates | Vocal updates | Mean physical / vocal rows (living) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A: both OFF | 207 | 109 | 2 | 1,405 | 51.82% | 13,718 | 0 | 19.04 / — |
| B: listening ON | 206 | 108 | 2 | 1,406 | 52.80% | 13,726 | 0 | 44.60 / — |
| C: production ON | 207 | 109 | 2 | 13,074 | 80.65% | 13,718 | 13,718 | 19.04 / 19.04 |
| D: both ON | 207 | 110 | 3 | 13,004 | 82.09% | 13,563 | 13,563 | 27.78 / 27.78 |

Denominators are actual agent observations, including silence and dying actors,
excluding newborns during their birth tick. Update counts include deceased
learners; mean rows refer only to current living learners. This checks mechanism
accounting and acoustic saturation, not fitness, significance or communication.
The higher masking in this short run was recorded without ecological tuning.
Local times with diagnostics were approximately 0.56–1.32 s per condition,
partly concurrent with tests; these are not controlled throughput benchmarks.

Intentional compatibility changes: LearningBrain now defaults to learned
production; its acoustic/learning trajectory may change. Loading old settings
without an explicit sender flag enables the new default. Use sender OFF for
v0.5 replay and both flags OFF for older visual-learning controls. Config hashes,
version metadata and detailed events gain fields; aggregate column names/meanings,
Decision's constructor, RandomBrain draws and biological rules remain unchanged.
The v0.4/v0.5 sections above and perception results below are historical records,
not claims about the new default.

Space per enabled learner scales with physical-cap×physical-actions plus
vocal-cap×(vocab+1) plus auditory-horizon, two pending experiences and a private RNG. Vocal max/tie
scans cost O(vocab) per decision; large vocabularies can be expensive despite
bounded state rows. More auditory states may fragment both tables, and high vocal
activity can increase candidate scans/masking. No dead heads or lifetime history
are retained. No capacity/cost mechanism or optimization changes were added.

Local verification artifacts, ignored by Git: `data/v06_verification.json`,
`data/v06_short_checks.json`, `data/v06_vocal_learning_demo.json`,
`data/v06_listening_demo.json`, short smoke CSV/metadata and diagnostic JSONL.
No new 5,000-tick or multi-seed ecology experiment was run. Absence of meaningful
or stable vocal structure remains a valid baseline result.

Future work, **document only**: a finite shared processing budget could make
physical action, vocal production, auditory processing and memory/attention
compete, perhaps affecting difficult movement, simultaneous speakers, stopping
or approaching to communicate. This version assumes concurrency without extra
cost. No budget, signal cost, intensity choice, communication/social reward,
identity/targeting, imitation, teaching, inherited learning, acoustic features,
sequences, composition, semantics, grammar or culture is implemented. Further
scientific design should follow review of these mechanism results.

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
| `signal_probability` | `Config.signal_probability` (0.1) | 0–1 | Random emission probability; no gate in learned-production mode |
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

The historical v0.3 seed-0 fixed-radius-2 run (auditory input ignored) was
verified through tick 5,000: population 899, 29,812 births,
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
python3 main.py --mode evolution --brain learning --no-learning-uses-auditory --no-learning-controls-vocalization --seed 0 --ticks 5000 \
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
  python3 main.py --mode evolution --brain learning --no-learning-uses-auditory --no-learning-controls-vocalization --seed "$seed" --ticks 5000 \
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

LearningBrain uses its private `brain:<id>` stream for physical exploration and
tied choices. It preserves the original one 64-bit construction draw and the
derived legacy emission RNG. The enabled Vocal Q uses a separately labeled
`vocal-choice` RNG derived from that same scalar, without extra historical draws.
Changing probability or vocabulary cannot shift physical draws through emission
or vocal exploration, though later heard states can change behavior. RandomBrain's original RNG consumption is
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
| `total_learning_updates` | Cumulative Physical Q updates, including those by agents that later died |
| `mean_learning_updates` | Mean lifetime Physical Q update count among living LearningBrain agents only |
| `mean_memory_size` | Mean Physical Q state rows among living LearningBrain agents only |
| `exploratory_actions`, `exploitative_actions` | Cumulative physical-choice branch counts, including deceased learners |

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

## Recorded v0.3 baselines, performance, and scientific limitations

These are preserved v0.3 results with auditory input ignored, not new v0.5 runs.
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
development. v0.6 adds learnable production without a communication objective;
learned preferences do not establish semantics or conventions. Further changes
must not directly script farming, institutions, or other civilization outcomes.
