# v0.8.0 — Resource ecology and social information

## Research question and executed design

Does scarcity and spatial localization change the predictive, behavioral or
ecological value of social information? This release manipulates environmental
supply, adds optional measurements and executes one bounded exploratory study.
It introduces no attraction, avoidance, cooperation, ownership, communication
reward or signal meaning.

The preregistration was written before calibration at
`data/resource_ecology_v080/preregistration.json`; an identical tracked copy is
`experiments/resource_ecology_design.json`. Baseline commit:
`43f954bb162513844463b955211b70e826f16c97` (v0.7.4).

* H1: scarcity may increase resource search difficulty or resource-related death;
  a monotone final-population effect is not assumed.
* H2: localization may concentrate resources and encounters.
* H3: individuals may indicate resources or competition; benefit is not assumed.
* H4: remembered status may add information, but existing memory is no resource map.
* H5: scarcity and localization may interact.

All hypotheses remain exploratory with three independent ecological seeds.

| Ecology | Supply multiplier | Localization strength |
|---|---:|---:|
| E0 | 1.0 | 0.0 |
| E1 | 0.5 | 0.0 |
| E2 | 1.0 | 0.75 |
| E3 | 0.5 | 0.75 |

Each ecology contains A (original physical input), B (+visible social token) and
C (+remembered-visible direction). These are the original v0.7.2–v0.7.4 modes.
All modes retain passive social memory for measurement. No D/E nuisance modes
were introduced. Calibration used seeds3/4; evaluation used seeds0/1/2.

## Actual legacy implementation and preserved ordering

The inspected evolution preset has a40×30 bounded grid,600 initial integer food
units and600 water units, capacity5 of each per tile, and regeneration probability
0.02 per resource per **non-full** tile per tick. A successful trial adds one
unit. A full tile consumes no regeneration draw. Food and water have separate
named RNG streams. Eating/drinking consumes actual0/1 units; the existing hunger
and thirst relief is30, and food restores20 energy in evolution mode. Resource
nutrition, physiology, death priority, reproduction and mutation are unchanged.

Tick ordering remains: increment clock; regenerate resources; shuffle existing
living actors; perceive and choose physical/vocal decisions; execute physical
action/reproduction; emit from the post-action position; update physiology,
reward and mortality; finish learning/diagnostic callbacks; append queued births;
advance auditory buffers; refresh statistics. Newborns do not act on their birth
tick. Social observation is still before that individual's memory commit; memory
is read before update. Diagnostics do not reorder any of these operations.

The original physical X includes hunger/thirst/energy bins, nearest visible
food/water direction and the existing auditory feature. Nearest-resource choice
uses the existing Manhattan ordering and direction signs; absent resource uses
the existing sentinel. B adds its11-category visible count/direction token; C
uses37 structurally valid B/memory combinations. Vocal Q retains its original
input and learning rules. Both LRU capacities remain256; reward, exploration,
TD learning, auditory behavior and social encoders are unchanged. Raw individual
IDs and patch metadata never enter Q states.

## Opt-in environmental API

`config.ResourceEcologyConfig` is a separate frozen, slotted dataclass. The
original `Config` fields, genome schema and default CLI remain unchanged.

| Field | Default | Validation |
|---|---:|---|
| `resource_supply_multiplier` | 1.0 | finite number in[0,1] |
| `localization_strength` | 0.0 | finite number in[0,1) |
| `patch_count` | 4 | positive integer, no more than grid cells when used |
| `patch_radius` | 4 | nonnegative integer |

Pass `resource_ecology=ResourceEcologyConfig(...)` to `Simulation` or
`Simulation.create_default`. `None` or the default configuration takes the
untouched legacy `World.generate` / `World.regenerate` path and creates no new
environmental RNG. The core remains standard-library-only and headless.

Scarcity changes only food/water regeneration probabilities, multiplying each by
the same scalar. E1 delegates to the original regeneration method with those two
probabilities replaced. Initial totals, per-tile capacity, consumption, action
cost and reward scale remain unchanged. A multiplier of0 disables renewal.

Localization uses four fixed Chebyshev squares of radius4, clipped at grid
boundaries. Centers are sampled uniquely once; squares can overlap. Food and
water **share the patch union**, but use separate placement streams. For N tiles,
K tiles in the patch union and localization strength s, normalized weights are:

```text
weight[i] = (1-s)/N + s*inside_patch[i]/K
```

There is positive support on every tile, including outside patches. Patch centers
do not move. Geometry, radius, mask and seed are saved per run. E0/E1 use the same
geometry only as a diagnostic reference region; they have no resource patches.

Initialization first executes original world generation, then E2/E3 relocate the
exact resulting totals using the weights and capacities. Thus E0/E1 initial
fields match, E2/E3 initial fields match, and all four have600 units of each
resource. Initial agent bodies, brains and RNG states match across factorial
conditions within seed. No agent position is changed to fit patches.

Private RNG names are `"{seed}:resources:patch_geometry"` and
`"{seed}:resources:placement:food"` / `water`. The original food/water RNGs still
generate supply offers. New streams never perturb spawn, ordering, agents,
offspring or mutation streams. Optional observation consumes no draws.

## Supply budgets and the capacity qualification

For localized renewal, scan non-full tiles in original raster order and draw the
original Bernoulli supply offers at scaled p. Relocate each offered unit using
weighted sampling among non-full recipient tiles. Full recipients are removed
from the sampler. There is no rejection sampling loop. Offers cannot exceed the
initial eligible-cell count; each such cell has at least one vacancy. Therefore
positive background support guarantees that every offer can be realized.

Record separately, for food and water: eligible cells, potential expectation
N×p, eligible expectation eligible×p, capacity-censored expectation
(N−eligible)×p, actual attempted offers, realized units and rejected units.
Initial total + cumulative regeneration − actual consumption equals current stock.

**Budget equality is conditional, not unconditional.** With equal eligibility
counts, identical rate/seed yields the same offers and legacy RNG state. Spatial
saturation can change those counts and later draw order. Localized and uniform
worlds can consequently have different realized supplies even without agents.
This is a capacity-mediated part of the intervention, not evidence of social
behavior and not a claim of pure quantity-independent localization.

The resource-only audit used seeds3/4 for200 ticks, with absent consumption,
scheduled withdrawal (one unit from each nonempty tile every10 ticks; actual
withdrawals are recorded), and a reset-empty control before every regeneration.
Mean realized units/tick over the two seeds:

| Regime | Ecology | Food | Water |
|---|---|---:|---:|
| No consumption | E0 | 20.578 | 20.318 |
| | E1 | 11.862 | 11.832 |
| | E2 | 19.068 | 18.942 |
| | E3 | 10.552 | 10.432 |
| Scheduled withdrawal | E0 | 24.378 | 24.268 |
| | E1 | 12.250 | 12.232 |
| | E2 | 24.100 | 23.942 |
| | E3 | 12.238 | 12.212 |
| Reset empty | E0 = E2 | 24.378 | 24.268 |
| | E1 = E3 | 12.250 | 12.232 |

Every offered unit was placed; rejection was0. Under reset-empty control the
offer counts, gains and budget RNG fingerprints match **each tick**, not merely
in the mean. Without consumption, localization lowers realized supply by roughly
7% at full supply and11% at half supply through capacity. The scheduled rule
fixes the withdrawal instruction, not an identical actual consumption trajectory.
Raw9600 rows are in `data/resource_ecology_v080/supply_audit.csv.gz`.

## Calibration, freeze and pilot decision

The bounded candidate lists were supply[0.5,0.25] and localization[0.75,0.5],
with four patches/radius4 fixed. Selection was first feasible, **A only**, on
200-tick calibration seeds3/4. Social-mode benefit never entered selection.
Each candidate required population100≥20, population200≥10,≥5000 decisions
and≥20 actual consumed units of each resource. Scarcity required mean realized
renewal≤80% of baseline for both resources and≥10% stock difference in at least
one resource over ticks100–200. Localization required initial effective cells
≤80% of baseline for both resources, positive support, equal initial totals and
the reset-empty supply check.

Both first candidates passed. No candidate was rejected; the fallback candidates
were **not run**. A combined E3 check passed without retuning. Eight calibration
runs were executed (E0/E1/E2/E3 × seeds3/4, A only):

| Ecology | Population200 seeds3,4 | Food consumed seeds3,4 | Water consumed seeds3,4 |
|---|---|---|---|
| E0 | 456,491 | 4582,4429 | 4450,4396 |
| E1 | 356,324 | 2515,2402 | 2498,2383 |
| E2 | 352,369 | 4344,4096 | 4255,3850 |
| E3 | 192,134 | 2366,1881 | 2393,1865 |

Initial effective occupied food cells decreased from426.4/449.7 to290.2/256.0;
water from426.3/447.1 to261.7/262.2. Scarcity mean stock over ticks100–200
fell from1193/1255 to707/728 food and1195/1224 to713/697 water. Full evidence,
selected parameters and input hashes are frozen in `data/.../frozen.json`.

All36 100-tick pilot cases passed completeness, population/exposure, consumption,
initial-state equality, actual-outcome accounting and E0 compatibility checks.
Nonfinite rewards, stock/capacity failures and Q-update inconsistencies fail the
run. A hashed pilot review permits primary only while frozen inputs and gate
results remain unchanged. The review is `primary_allowed`; there is no
social-benefit gate. All36 primary cases completed500 ticks without extinction.
There were no1000-tick runs, new evaluation seeds or post-primary parameter edits.

E0 archived outcomes lacked actual-acquisition/life diagnostics, so instrumented
E0 reruns were necessary. They are exact controls, not independent new replicates.

## Measurements and statistical unit

Optional `ResourceObserver` extends the existing physical learning diagnostics.
It records actual0/1 consumption, failed eat/drink outcomes, pre/post physiology,
existing immediate reward, terminal/death cause, identifiers for linkage,
original-state catalog reference, social tokens, position, local resources,
current visible-neighbor resources and reproductive eligibility. These values
are diagnostic only and never modify observations, brains or RNG.

World nearest-resource distances are Chebyshev distance maps computed **after
tick-start regeneration**, frozen for that tick and explicitly named
`*_tickstart`. They can become stale after preceding actors consume. In contrast,
`food_here`, `water_here` and resources at the nearest visible individual's cell
are current sequential pre-action world values. They are not interchangeable.
Resource concentration uses unit-weighted entropy,2**entropy effective cells,
HHI, patch resource share and agent patch share. Snapshots retain actual grids
at tick0/every100/final; no average over unrelated geographic seeds is plotted.

Per-individual scalar records are streamed at death/end; full dead Human objects
and Q snapshots are not retained. Primary acquisition is the within-run mean of
each actor's actual acquisitions / own opportunities, excluding never-acting
newborns and preserving actor n. Aggregate successes/attempts and
successes/opportunities, their numerators and exposures are separate fields.
Founder-only acquisition rates provide a secondary cohort check. Age/cohort
composition still differs across runs; these are not intrinsic fitness scores.

Founder restricted mean survival time (RMST) is measured through H=500. All100
founders are observed from0; survivors are right-censored at H, not recorded as
deaths. Hence mean min(observed death time,H) identifies founder RMST without
inventing post-censor lifespan. Newborn follow-up and censor counts are reported
separately; mean observed newborn follow-up is **not** mean lifespan. Resource
death rate is starvation+dehydration / actual agent opportunities. Reproduction
choices are normalized by eligible opportunities, distinct from parent
participations. Gini describes actor acquisition-rate inequality, not ownership.

Each run contributes one observation to seed means, sample SD(ddof=1), min/max/n.
Paired differences use the same ecological seed. Millions of decisions do not
increase independent experimental n beyond3. No significance test or
decision-based confidence interval is used. Raw rows and denominators remain
available in `primary/runs.csv`, `individuals.csv` and `summary.csv`.

## Primary ecological and individual outcomes

Final population raw values (seeds0,1,2):

| Ecology | A | B | C |
|---|---|---|---|
| E0 | 832,823,799 | 831,810,786 | 846,770,799 |
| E1 | 372,363,366 | 357,346,369 | 383,365,380 |
| E2 | 773,798,798 | 864,780,888 | 897,857,819 |
| E3 | 354,343,423 | 430,384,400 | 408,382,426 |

Means±sample SD across ecological seeds (all n=3). Resource-death rate is shown
per1000 actual opportunities for readability; saved CSVs use unscaled rates.

| Ecology/mode | Final population | Individual acquisitions/opportunity | Founder RMST ticks | Resource deaths/1000 opportunities |
|---|---:|---:|---:|---:|
| E0/A | 818.0±17.1 | .06274±.00023 | 318.32±11.58 | 4.375±.033 |
| E0/B | 809.0±22.5 | .05922±.00057 | 317.11±4.10 | 4.607±.102 |
| E0/C | 805.0±38.4 | .05955±.00123 | 314.65±5.16 | 4.622±.058 |
| E1/A | 367.0±4.6 | .06343±.00166 | 266.41±2.20 | 4.472±.203 |
| E1/B | 357.3±11.5 | .05891±.00139 | 256.81±2.51 | 4.822±.122 |
| E1/C | 376.0±9.6 | .05953±.00076 | 260.36±3.86 | 4.685±.045 |
| E2/A | 789.7±14.4 | .07014±.00094 | 267.93±9.60 | 4.338±.186 |
| E2/B | 844.0±56.7 | .06361±.00147 | 260.36±7.10 | 4.610±.167 |
| E2/C | 857.7±39.0 | .06458±.00141 | 261.77±10.95 | 4.463±.054 |
| E3/A | 373.3±43.4 | .07702±.00547 | 195.71±14.01 | 4.052±.079 |
| E3/B | 404.7±23.4 | .07005±.00365 | 191.02±8.87 | 4.311±.205 |
| E3/C | 405.3±22.1 | .07374±.00751 | 188.55±10.60 | 4.322±.214 |

Scarcity reduced population and founder RMST. Absolute resource-death counts
also fell because the number of actors/opportunities fell; they cannot be read
as improved survival. Dehydration dominates starvation in every condition.
For example E0/A has mean335.7 starvation and951.3 dehydration deaths, whereas
E3/A has114.0 and316.7; exposure-normalized rates and RMST give the necessary
context. Energy deaths remain separate (E0/A338.0; E3/A72.0).

Localized B/C final population exceeds A in the seed means, but **both individual
acquisition and founder RMST are lower than A in every ecology's seed mean**.
Founder acquisition rates also remain lower for B/C. Mean births are1956.7,
2304.0,2216.3 in E2/A/B/C and776.0,959.7,913.3 in E3. Population therefore
mixes births, mortality and cohort composition; it is not individual survival or
proof that a social cue is beneficial. Paired seed differences are not uniformly
positive; raw seed results are retained.

Localization increases actor access-rate Gini (A: E0 .403→E2 .489;
E1 .373→E3 .501). Acquisition per attempted consumption in A is.303/.298/.351/
.387 in E0/E1/E2/E3, respectively. These different denominators and cohort
compositions explain why some acquisition summaries improve while RMST declines.
The report does not identify a unique causal mechanism from these associations.

## Social cues, held-out prediction and target support

Visible-individual exposure is already95.1–99.5% of decisions. Contact,
re-encounter, movement relative to the pre-action nearest visible target and
resource-at-neighbor diagnostics are preserved. Coarse presence can consequently
have little remaining variation. E3 lowers visible exposure relative to E0;
localization did not universally increase encounters.

Prediction reuses v0.7.4's hierarchical count/backoff helpers, min20 training
observations and Dirichlet strength5. It trains on selected A-only calibration
seeds3/4 for200 ticks in each ecology; tests actual A/B/C primary experience on
seeds0/1/2 restricted to the first200 ticks. B/C testing has a behavior-policy
distribution shift from A training. The three heads are X,action;
X,B,action; and X,B,remembered-direction,action. Memory comparisons hold B fixed.
All/common-support scores, backoff coverage, calibration bins, equal-agent loss,
per-seed differences and censor reasons are saved. No hyperparameters were tuned.

Targets are the preregistered immediate physiological reward bins, actual
acquisition on eating/drinking attempts, tick-start food/water within Chebyshev3,
and any acquisition in the next five **actual own decisions**, excluding the
current decision. Death/end/gap windows with fewer than five are censored;
there is no absorbing padding. Short-horizon cumulative reward and survival
classifiers were not added; primary founder survival is measured separately.

Two important support checks qualify every predictive result:

1. **Immediate acquisition is already determined by X and action.** Eating or
   drinking succeeds exactly when its original visible-resource direction is
   HERE(0,0). The analysis checks this oracle against every scored attempted
   consumption. Thus apparent nested-model log-loss improvements here cannot
   be extra social information: the target is fully specified by existing input
   and action. They are finite-sample/backoff/smoothing effects.
2. **Nearby-resource presence is nearly saturated.** Across evaluated runs the
   food-near3 positive fraction is≥.99849; water≥.99570, and E0 food is exactly1.
   Extremely small predictor differences on these near-constant targets do not
   establish a useful ecological resource cue. Full prevalence/counts are in
   `prediction/target_prevalence.csv`.

Next-five actual acquisition log loss on the all-decision held-out set worsens
on average with each social extension in **all12 ecology/source-mode combinations**.
For C test experience,
equal-seed mean log-loss differences (positive means worse):

| Ecology | X+B − X | X+B+memory − X+B |
|---|---:|---:|
| E0 | +.001522 | +.000901 |
| E1 | +.001485 | +.000507 |
| E2 | +.001930 | +.000613 |
| E3 | +.001448 | +.000414 |

Immediate physiological reward has tiny/mixed B improvements (C test:
−.000165/−.000220/−.000219/+.000274 for E0–E3); the extra memory head worsens
the mean in all four. Reward includes the unchanged physiology calculation,
not just acquisition. Sparse support, early follow-up, selected actions and
policy shift prevent interpreting predictor failure as proof that information
is absent. The optional analysis does not estimate another policy's value.

## Learned policy, fragmentation and vocal observations

Read-only Q probes compare retained trained rows with the same original state
and different social tokens, requiring≥2 visits and≥1 actual TD update per row.
Final epsilon-greedy policy total variation, equal-seed means:

| Ecology | B | C |
|---|---:|---:|
| E0 | .3548 | .3599 |
| E1 | .3604 | .3637 |
| E2 | .4244 | .4283 |
| E3 | .4387 | .4541 |

Each cell has three supported ecological seeds, but these are selected retained
pairs, not a randomized social intervention. Pair support at≥2 visits ranges
845–1998 per final run; at≥5 visits it ranges3–268. Qualified agent/pair counts,
nonflat rows and changed greedy sets remain in `primary/probes.csv`. These are
all-social probes; no memory-only Q probe was recorded in this release.

First-agent-state fractions for A/B/C are.682/.802/.807 in E0 and
.569/.722/.722 in E3. Q evictions/opportunity in E3 are.00007/.00181/.00328.
B/C remain much more fragmented; occupancy is an exposure-weighted within-run
measure, not a claim that all256 rows are full. Social-token policy differences
and altered physical action distributions are measured, but learning-history
and fragmentation effects still prevent isolating causal information value.

Vocal Q is unchanged. Emission is about.940–.942 per opportunity. Masked-decision
fractions for A/B/C are.681/.669/.671 in E0 and.761/.742/.751 in E3. Final mean
living Vocal Q row counts are65.75/62.97/62.51 in E0 and58.81/57.94/54.08 in E3.
Raw vocal action histograms and hearing exposure are saved. Ecological changes
in masking or distributions do not establish intentional signaling, shared
semantics, cooperation or language.

## Paired factorial effects

`contrasts_by_seed.csv` preserves all E1−E0, E2−E0, E3−E1, E3−E2 and B−A,
C−B,C−A differences. Interaction is (E3−E2)−(E1−E0), separately by mode;
moderation subtracts those interactions between modes. Absolute effect sizes,
sample SD,min,max,n are in `contrasts_summary.csv`, with no significance claim.

| Mode | Population interaction | Founder RMST interaction | Individual acquisition-rate interaction |
|---|---:|---:|---:|
| A | +34.67±27.06 | −20.32±21.16 | +.00619±.00583 |
| B | +12.33±73.51 | −9.05±15.18 | +.00676±.00459 |
| C | −23.33±48.06 | −18.93±13.14 | +.00916±.00901 |

For example, population-interaction moderation is−22.33 for B−A and−35.67
for C−B. These contrasts include population/supply feedback and state-space
differences. They are not causal social-information benefits.

## Verification, runtime and saved outputs

Before edits,602 tests passed in100.520s. The final full suite passed625 tests
in92.007s (602 existing +23 new). No historical fixture or old test was changed.
The18 E0 pilot/primary controls match archived v0.7.2 source runs compatible
with v0.7.4: full bodies/genomes/resources, Physical/Vocal Q, LRU/pending,
named RNG states, actions and aggregate statistics. Old files remain unchanged:
1308 protected files match the pre-edit1312-file inventory; only four existing
files were intentionally edited.

Repeated E3/A seed0 100-tick diagnostic runs have identical complete state/RNG
fingerprints and byte-identical streams. Tests also compare instrumentation
on/off. Calibration/pilot used diagnostic schema1; before primary, schema2 added
only living Vocal Q occupancy columns. Repeating a pilot case with schema2
preserved every body/RNG/statistic and every old stream/shared resource field;
only those new observational columns differ. No pilot occupancy was fabricated.
The schema distinction and source hashes are saved per run.

Runtime measurements include full diagnostics and fingerprint construction:

| Phase | Runs | Sum of logged case wall seconds | Per-case seconds | Largest process high-water RSS |
|---|---:|---:|---:|---:|
| Calibration | 8 | 64.65 | see per-run metadata | 240.5 MB |
| Pilot | 36 | 121.25 | 2.84–3.70 | 100.3 MB |
| Primary | 36 | 2147.97 | 21.95–101.11 | 535.1 MB |

Primary seeds ran in three concurrent independent CLI processes; summed case
time is not elapsed end-to-end time. RSS is a cumulative process high-water
mark, not an incremental per-run allocation. Analysis took8.24s, prediction49.29s,
figures3.84s. Primary compressed streams total602.6 MB; the complete new data
directory is about662 MB and analysis about30.3 MB. Detailed measurements are in
`runtime_by_case.csv`, metadata and the final verification receipt. Diagnostic
world maps are O(grid cells) per tick; optional detailed traces dominate storage.
This is not a long-run performance benchmark, and no multiprocessing was added.

Data are saved under `data/resource_ecology_v080/`: baseline inventory/test log,
preregistration, frozen calibration, supply-only audit, pilot review, calibration,
pilot, primary, compatibility and deterministic repeats. Each run saves config,
geometry, source/input hashes, fingerprints, runtime, RSS, bytes and15 optional
compressed stream roles. Existing global statistics CSV schema is unchanged.

Analysis is under `analysis/resource_ecology_v080/`: raw runs, tidy seed
mean/SD/min/max/n, resource/population time series, individual follow-up,
spatial grids, social resource cues, Q probes/visits, vocal distributions,
paired effects, predictions/support/calibration, runtime and verification.
Sixteen primary PNGs are generated at180dpi with no random jitter:

```text
resource_availability.png       resource_concentration.png
spatial_food.png                spatial_water.png
population.png                  births_deaths.png
acquisition_success.png         individual_survival.png
resource_deaths.png             patch_agent_density.png
social_exposure.png             physical_actions.png
q_fragmentation.png             factorial_contrasts.png
vocal_exposure.png              physical_policy_dependence.png
```

They show seed means/sample SD or explicitly labelled seed0/A maps; raw seed
points are included on outcome comparisons. Three representative figures were
visually inspected for axes, titles, legends and margins. Generated artifacts
use the repository's existing git-ignore convention and remain saved locally.

Changed existing files: `config.py`, `simulation/simulation.py`, `.gitignore`,
`README.md`. Added: `world/resource_ecology.py`,
`simulation/resource_observer.py`, `tests/test_resource_ecology.py`,
`experiments/resource_ecology_design.json`, `experiments/resource_ecology_experiments.py`,
`experiments/analyze_resource_ecology.py`, `experiments/resource_ecology_prediction.py`,
`experiments/resource_ecology_plots.py`, `experiments/resource_ecology_audit.py`
and this report. World, genome, reproduction, learning and statistics modules
were not modified. Dependencies remain optional analysis-only.

## Reproduction and audit commands

Run from the repository root with Python3.11+; use `.venv/bin/python` here for the
existing optional numpy/pandas/matplotlib environment. All completed destinations
are exclusive; a fresh root prevents accidental overwrite. Historical v0.7.2
metadata is required for automatic exact E0 comparison.

```bash
python3 -m unittest discover -s tests -q
mkdir -p data/resource_ecology_v080_replay
cp experiments/resource_ecology_design.json data/resource_ecology_v080_replay/preregistration.json
python3 -m experiments.resource_ecology_experiments --root data/resource_ecology_v080_replay --phase calibrate
python3 -m experiments.resource_ecology_experiments --root data/resource_ecology_v080_replay --phase pilot
python3 -m experiments.resource_ecology_experiments --root data/resource_ecology_v080_replay --output-root analysis/resource_ecology_v080_replay --phase review
python3 -m experiments.resource_ecology_experiments --root data/resource_ecology_v080_replay --phase primary
python3 -m experiments.analyze_resource_ecology --root data/resource_ecology_v080_replay --phase primary --output analysis/resource_ecology_v080_replay/primary
python3 -m experiments.resource_ecology_prediction --root data/resource_ecology_v080_replay --output analysis/resource_ecology_v080_replay/prediction
python3 -m experiments.resource_ecology_plots --data-dir analysis/resource_ecology_v080_replay/primary --output analysis/resource_ecology_v080_replay/primary/figures
```

`--seeds 0` (or1/2) can split pilot/primary work into independent CLI processes;
review requires all36 pilot cases. These commands document reproduction; they
are not an instruction to retune the saved study. The final audit for the actual
saved study uses its pre-edit inventory, frozen inputs and test log:

```bash
python3 -m experiments.resource_ecology_audit \
  --test-log data/resource_ecology_v080/release_tests.log \
  --output analysis/resource_ecology_v080/verification_rerun.json
```

The saved final receipt is `analysis/resource_ecology_v080/verification_final.json`;
use another fresh receipt destination to repeat it. Its audit is specific to the
recorded study paths and histories, not a general-purpose arbitrary-study audit.

## Scientific conclusion and next experiment

Ecological manipulations worked: scarcity reduced renewal and founder survival;
localization concentrated initial supply, altered access inequality and changed
population trajectories. Social-token-conditioned learned preferences differ,
with larger retained-pair differences in localized ecologies. This does not
isolate useful information from representation and learning-history effects.

No tested ecology establishes added resource-predictive value: immediate
acquisition already follows from X+action, nearby-resource presence is nearly
constant, and the specified future-acquisition predictor did not improve. B/C
increase final population under localization but do not improve mean individual
acquisition or founder survival. Additional memory has no consistent predictive
benefit here. Three seeds, capacity-mediated supply changes, cohort composition,
sparse support and A-trained early prediction leave the causal question open.

The recommended v0.8.1 is a preregistered **target-support and temporal resource
information study**. First define a nondegenerate, genuinely future resource
target beyond current X+action, inspect its prevalence/support using existing
traces, and freeze evaluation choices. A later separate ecology could introduce
bounded temporal patch availability only if needed to create uncertainty; that
would be a new intervention with its own supply audit, not a retuning of this
study. Do not yet add attraction, cooperation or semantic signals. Proximity,
reproduction and population changes alone remain insufficient evidence of
cooperation, communication or language emergence.
