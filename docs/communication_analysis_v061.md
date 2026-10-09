# Communication analysis v0.6.1 — verification record

This is an observational release. No v0.7 mechanism, communication reward,
speaking cost, cognitive-capacity change, semantic label or ecology tuning was
introduced. The longer-term question remains how environmental, perceptual,
cognitive and social conditions affect emergent communication.

## Implementation and compatibility

Changed: `.gitignore`, `README.md`, `agents/learning.py`, `main.py`,
`simulation/simulation.py`, and the CLI version assertions in
`tests/test_auditory.py` / `tests/test_vocalization.py`.

Added: `simulation/communication_observer.py`,
`experiments/analyze_communication.py`, `experiments/communication_data.py`,
`experiments/communication_metrics.py`, `experiments/communication_plots.py`,
`experiments/communication_pilot.py`, `tests/test_communication_observer.py`,
`tests/test_communication_analysis.py`,
`tests/fixtures/v061_communication_toys.json`, and this report.

There are **no new Config fields** or altered defaults. A shared pure primitive
encoder replaces duplicated definitions; the learner's chosen state is exposed
through an immutable read-only `decision_state` property. Simulation adds two
conditional observer calls, without changing actor order or the tick phases.
OFF creates no observer, density grid or decision records. ON streams immutable
records without retaining agent objects, Q tables or history.

The new simulation CLI option is `--communication-csv PATH`. Its schema has 27
columns, listed in the README. It distinguishes actual resolved sound from
retained auditory-memory features, and learned selection from random emission.
An actor's row at t pairs receiving t−1 sound with that same turn's physical and
vocal choices at t. Tick 0 has no choices; birth-tick newborns do not act.
Agent IDs and radius-2 tick-start density are privileged measurements, not new
Brain inputs. No sender ID, exact location, distance or loudness is exposed.

The recording sidecar includes schema, full Config, seed, config hash, final
tick/statistics, source hashes, timing, density and sampling definitions. The
analyser's CLI supports input/output directories, glob, window size, permutation
count, analysis seed, individual/receiver thresholds, pair limit and display
signal limit. The pilot runner is explicitly separate from offline analysis and
refuses existing destinations or horizons above 1000.

## Tests and historical replay

- Before edits: **236 tests, OK, 69.968 seconds**.
- After implementation: **282 tests, OK, 49.830 seconds** (Python 3.11.9).
- All four learning ablations compare every tick ON/OFF, including bodies,
  genomes, births/deaths, resources, full aggregate statistics, signal provenance,
  Q values, both LRU orders, pending transitions, memory and every simulation / brain
  RNG. Full detailed events and actor ordering also match; dying actors are covered.
- RandomBrain and legacy controls match ON/OFF. Recording never calls an RNG or
  modifies LRU/state. The ordinary aggregate CSV and config ID match ON/OFF.
- The existing v0.6 seed-7 tick-80 trajectory, final state and complete-event
  hashes match with recording enabled. Its final population remains 32, births
  30, deaths 10 and emitted signals 2106. All five historical fixture hashes are
  unchanged. The only existing test edits reflect sidecar version `0.6.1`.
- Two analyses of synthetic multi-run files produce identical bytes for all
  15 CSVs, metadata JSON and 10 PNGs. Input hashes and Python/NumPy global RNG
  states remain unchanged. Sparse/missing/partial/empty/extinct inputs and
  100/1000/5000 horizons are covered; no interpolation or carry-forward occurs.

Machine-readable checks are saved locally in `data/v061_verification.json`.

## Ecological pilot and instrumentation cost

The requested pilot was run in full: **1000 ticks × seeds 0/1/2 × all four
conditions = 12 runs**. All reached tick 1000. The unmodified evolution preset
was used, including initial population 100, world 40×30, vocabulary 16, hearing
radius 8 and free inherited perception (founder value 2). Only the two existing
learning flags vary by condition; no parameters were tuned after seeing results.

| Condition | Listening | Production | Final population mean ± sample SD (n=3) |
|---|---|---|---:|
| A | OFF | OFF | 831.667 ± 22.008 |
| B | ON | OFF | 858.667 ± 18.230 |
| C | OFF | ON | 831.667 ± 22.008 |
| D | ON | ON | 880.000 ± 22.539 |

These population differences are descriptive, not an advantage or communication
claim. A/C match all **1001 aggregate snapshots** except config ID and emitted-signal
count, for all three seeds, consistent with listening-OFF behavior ignoring production.
The recorder captured **8,424,234 decisions**. Compact communication CSVs total
**860,737,923 bytes (820.864 MiB)**; the complete pilot directory including ordinary
statistics, sidecars, logs and manifest totals **865,785,667 bytes (825.678 MiB)**.
Simulation CLI times sum to **481.109 s**; individual runs took 22.8–61.2 s.
These are local timings, not general performance guarantees.

An independent seed-0, 100-tick benchmark used one warm-up and five timed repeats
per variant, each in a separate process, without concurrent simulations. Medians:
archived v0.6 **0.472655 s**, v0.6.1 recording OFF **0.473484 s**, streaming compact
CSV ON **0.570636 s**. OFF differs by about 0.18%, within timing noise; ON is about
20.5% slower here. All three final full-state hashes are identical. The compact
100-tick recording is 1,367,852 characters. `data/v061_performance.json` retains
individual measurements, method and state digests. Long densely populated records
can be large; no unreported sampling was used to reduce this pilot.

## Estimators and validation fixtures

V includes SILENCE; S conditions on emission. Empirical entropy/MI use log2
bits, and NMI divides by the smaller entropy only when positive. Undefined
values are blank/null. Observed-minus-permutation MI may be negative.

Pooled shuffles preserve window marginals, without controlling identity.
Within-agent shuffles preserve individual marginals; per-agent MI and empirical
I(V;X|agent) remain distinct from pooled MI. Conditional estimates weight exposure
inside a run, whereas aggregation across seeds always gives each run equal weight.
Permutations use isolated labelled analysis RNGs, never simulation/global RNGs.
Equal-length strata are shuffled as independent matrix rows, preserving the
same within-stratum null while avoiding one Python RNG call per individual.

JSD uses base-2 entropy, 0–1 bits. The emitted-ID frequency-matched null preserves
each eligible individual's emission count and the population's ID distribution.
Observed and null JSD use the identical exhaustive/seeded sampled pair subset.
Receiver MI uses the same receiving tick's action and resolved percept, with
need and need+visual conditioning where sample thresholds permit. SILENCE and
masking have separate decision/observation denominators. IDENTIFIED/emitted is
a descriptive ratio; it can exceed one and crosses window boundaries.

| Synthetic fixture | Measured result |
|---|---:|
| A: exact independent sender choices/state | MI = 0 bits |
| B: perfectly dependent arbitrary categorical choices/state | MI = 1 bit |
| C: identity mixture | Pooled MI = 1 bit; within-agent MI = 0 bits |
| D: identical emitted-ID distributions | JSD = 0 bits |
| E: disjoint distributions | JSD = 1 bit |
| F: action independent of resolved auditory category | MI = 0 bits |
| G: action dependent on arbitrary auditory category | MI = 1 bit |
| Uniform four-category counts | Entropy = 2 bits |

These validate the estimators and do not represent ecological communication.

## Pilot measurements

All averages below give each seed equal weight; SD is sample SD, n=3. Full
per-seed values, all five sender features, both V/S scopes, exposures and nulls
remain in the CSVs. The recording corpus is unchanged after analysis.

| Condition | Whole-run emission % ± SD | Whole-run MASKED % ± SD | IDENTIFIED % ± SD |
|---|---:|---:|---:|
| A | 9.923 ± 0.124 | 80.332 ± 0.159 | 19.396 ± 0.128 |
| B | 10.020 ± 0.045 | 80.642 ± 0.031 | 19.124 ± 0.042 |
| C | 94.140 ± 0.005 | 67.151 ± 0.200 | 32.835 ± 0.200 |
| D | 94.120 ± 0.009 | 66.782 ± 0.081 | 33.204 ± 0.081 |

SILENCE selection is the complement of emission (about 90% in A/B and 5.9% in
C/D). It did not show a large sustained rise in this pilot. More emissions did
not correspond to a larger MASKED fraction across these conditions. This is a
description of the existing proximity/dominance resolution and changed trajectories,
not a causal monotonicity claim or a recommendation to increase emission.

**Early structure differs from the final window.** Identity-conditioned corrected
I(V;visible food direction) in (0,100] was **0.132478 ± 0.014394 bits for C** and
**0.095352 ± 0.010949 bits for D**; all three seeds in each condition were positive.
Visible-water counterparts were **0.136122 ± 0.003936** and **0.111744 ± 0.006760**.
These associations were transient: in (900,1000] food-direction values were
**−0.022982 ± 0.001725** (C) and **−0.010216 ± 0.002621** (D). Final pooled corrected
values for all five features were close to zero (absolute seed-mean values under
0.00015 bits for V). The negative corrections mean below the chosen shuffled
baseline, not negative MI or a proof of independence. Early individual structure
does not establish population-wide mappings or meanings; time and repeated
experience remain possible confounds.

Need+visual-conditioned receiver I(A;resolved percept), minus its permutation null,
was **0.032800 ± 0.001914 bits for B** and **0.032410 ± 0.008480 bits for D** in
(0,100]; both were positive in all three seeds. B uses random emission, so this
alone cannot indicate informative learned production or interpreted meaning.
The same final-window estimates were **−0.000330 ± 0.000788** (B) and
**−0.002136 ± 0.002041** (D). Current-percept conditioning cannot separate retained
memory, state fragmentation, shared environments, temporal trends or identity.

| Final (900,1000] | Mean pairwise JSD ± seed SD | Frequency-matched null ± seed SD | Observed − null ± seed SD |
|---|---:|---:|---:|
| A | 0.496811 ± 0.002822 | 0.497739 ± 0.003551 | −0.000929 ± 0.001312 |
| B | 0.492038 ± 0.009251 | 0.493664 ± 0.006927 | −0.001626 ± 0.002673 |
| C | 0.150691 ± 0.000644 | 0.157102 ± 0.001010 | −0.006410 ± 0.001102 |
| D | 0.157790 ± 0.003054 | 0.160806 ± 0.003185 | −0.003016 ± 0.000342 |

All use 1000 sampled pairs per run. Eligible emitting agents range 217–294 (A),
268–293 (B), 1293–1344 (C), 1361–1374 (D), with emission exposures 2633–3568,
3241–3563, 76238–78776 and 80427–80630 respectively. Lower raw JSD in C/D must be
read with those much larger per-agent samples and their low null JSD; it is not
evidence for conventions. Final sender within-agent V analysis uses 820–867
eligible individuals and 66491–71123 decisions across conditions; receiver
need+visual estimates use 73540–77975 decisions. These are window observations
including actors that die, rather than final living-population counts.

Offline analysis took **1438.33 s** locally with 100 permutations/window (about
24 minutes). The 15 CSVs, 10 PNGs and metadata total **358,724,895 bytes
(342.107 MiB)**. The largest per-agent tables stream to disk; raw records load
one window at a time, with summary rows retained for plotting. This is still a
costly descriptive analysis at millions of observations; large-vocabulary dense
contingencies, repeated nulls and output storage are scale limitations.
`data/v061_pilot_summary.json`, `data/v061_analysis.log`, and output hashes in
`data/v061_verification.json` preserve these checks. Graphs were visually reviewed;
all 120 windows are complete, proportions sum correctly, all MI bounds hold,
all input/sidecar hashes match and every PNG is 180 dpi.

**Observed:** transient individual state/choice associations early in learned
conditions; final estimates near/below null; largely uniform population signal
frequencies; no major increase in silence. **Hypotheses:** changing experience,
state fragmentation, temporal structure and finite-sample effects may contribute.
**Causal conclusion:** none about message content, semantics, cooperation or
language follows. The next experiment should distinguish transient individual
associations from information that receivers use, with time and memory controls.

## Outputs and interpretation

`analysis/communication/` contains the README-listed **15 tidy CSVs, 10 PNGs
at 180 dpi, and analysis_metadata.json**. The latter records options, source/input
hashes, all effective configs, seed lists, software versions, window/missing-data
rules and exact display selection. No simulation input is overwritten.

Sender state association, receiver association, distributional similarity,
individual preferences, silence and masking are separate measurements. Neither
positive MI, low JSD, changed silence nor population differences establishes
meaning, a convention, cooperation, ecological advantage or language. Permuting
observations does not control time autocorrelation, shared environmental trends
or selective survival. Conditioning remains sparse and cannot remove all context;
current-percept receiver analyses do not model retained memory's delayed effects.

Recommended next experimental question: does state-related vocal variation
remain after stronger individual/time controls, and do listening-ON receivers
actually use that variation beyond body, visual and retained-memory context?
Any causal signal intervention needs separate scientific design review.
