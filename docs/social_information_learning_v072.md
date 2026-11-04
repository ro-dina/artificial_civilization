# v0.7.2 — Social Information in Action Learning

## Objective and inspected control flow

Does coarse information about visible individuals and their pre-existing
remembered status change learned physical behavior under unchanged physiology,
rewards, learning parameters and ecology? Availability of information, learned
Q dependence, actual behavior, ecology and vocal activity are separate questions.
None implies friendship, cooperation or language.

The baseline is commit `272a847ffd9e420badaa8950ab2a2eebb879ebf6` (v0.7.1).
README, Config, both state encoders, action spaces, Q update/LRU routines,
Simulation, immutable observations, spatial/communication systems, social
memory/formation/decay, tests, historical hashes and experimental/analysis code
were inspected before editing. `data/v072_design.json` records the actual
pre-change flow and experimental design; `data/v072_before.json` binds 295
baseline files and historical artifacts.

The actual v0.7.1 tick order is:

1. Increment tick, regenerate resources, shuffle existing actors, initialize live
   spatial buckets; newborns are not in this acting set.
2. At each sequential actor, construct one local resource/auditory/visual batch.
   Earlier movement/death is already reflected. Update social memory from it.
3. LearningBrain updates auditory memory, captures physiology and encodes the
   old seven primitive components (three need bins and two signed offsets for
   each nearest food/water tile), plus an optional auditory feature.
4. Bootstrap the previous physical transition using that actual new state,
   update/possibly recreate the previous row, then allocate/refresh/select the
   current physical row. The bounded OrderedDict is LRU. Initial Q values are zero.
5. Independently update/select Vocal Q using the same old key, separate RNG and
   table; return one physical action and vocal choice.
6. Execute action or reproduction, emit at post-action position, update
   physiology/death, then assign the existing physiological reward to both
   pending transitions. Terminal transitions update immediately without bootstrap;
   surviving transitions wait until the next real observation.
7. Remove deaths, append queued newborns, advance signals to next-tick hearing,
   construct aggregate statistics. No artificial final nonterminal update occurs.

v0.7.2 inserts an immutable coarse snapshot **between steps 2's observation and
memory update**, before any new commitment/refresh. It reuses the same raw batch
for diagnostics, memory and feature extraction. Only the Physical Q key is then
extended; Vocal Q keeps the old key and exact update algorithm. No action, reward,
physiology, acoustic, genome, reproduction, resource or RNG rule changes.

## Encoding and configuration

The unchanged three configuration schemas are accompanied by frozen
`SocialLearningConfig(mode="none")` with modes none/perception/memory. Memory
mode requires explicitly enabled social memory. Perception mode can operate
without social memory. New inputs require the built-in LearningBrain. Invalid
combinations fail instead of silently enabling other mechanisms.

The original LearningBrain remains the none path. A small subclass overrides
only `physical_state`; the unchanged `state` method is the vocal/legacy encoder.
In none mode both pending heads even retain the original shared state object.

| Condition | Additional physical key | Information |
|---|---|---|
| A / none | none | Original physical key |
| B / perception | `(visible_token,)` | Token 0 empty; 1–5 one HERE/N/S/E/W; 6–10 ≥2 HERE/N/S/E/W |
| C / memory | `(visible_token, remembered_direction)` | Identical B token plus 0 NONE / 1 HERE / 2 N / 3 S / 4 E / 5 W |

Nearest selection minimizes Chebyshev distance then individual ID. Coarse
cardinal projection chooses the dominant axis and the vertical axis on diagonal
ties; north decreases y. IDs are internal tie-breakers, never Q components.
The new Brain-facing observation has only tick, the original immutable resource
tiles, original auditory percept and coarse features. No individual IDs/offsets,
encounter counts, memory strengths/timestamps, reunion labels or death knowledge
enter the new input. Candidates never count as remembered. The fifth committing
sighting remains unremembered for that decision; the sixth may be remembered.

B has **11** structural combinations. C has **37**: empty 1; single visible
5×(unremembered or same direction)=10; multiple with nearest HERE 6; multiple
with nearest cardinal 4×5=20. A remembered HERE is impossible when the nearest
visible target is not HERE. Its rectangular storage bound is 11×6=66, but
impossible combinations are rejected. There is no redundant remembered-presence
bit. These are relative multipliers of the original key space, not predictions
of observed state coverage. Physiological/auditory semantics are unchanged.

The physical action sequence stays N/S/E/W/eat/drink/wait/reproduce in evolution
mode. Both table capacities remain 256, epsilon, alpha, gamma and zero priors are
unchanged. The shared reward remains normalized changes in hunger, thirst and
energy, without any social term. The vocal vocabulary, state, action set,
selection/update rules and acoustic behavior are unchanged.

## Measurement design

New diagnostics are optional, streaming and versioned, with deterministic gzip,
source/input/output hashes and full four-part configuration. Existing aggregate
CSV schema and legacy none-mode outputs are preserved. New modes cannot use
legacy social recording formats whose metadata cannot describe the new inputs.

The eight streams are tick-level original statistics plus Q occupancy/diversity;
window counts; grouped agent/window/original-key/social-state/action cells;
original-key catalog; movement strata; distributions; Q probes; vocal choices.
No full Q dumps or per-update CSVs are generated. Actual Q hooks count all
allocations/evictions, including prior-row recreation and terminal updates.
External histories are released on observer death/end and never enter bounded
agent memory. Per-agent distinct states and global distinct keys are separate.

Movement rates use **actual moved actors with a selected target** as denominator.
Targets are frozen at their pre-action positions from that actor's authoritative
batch; only the actor's before/after displacement changes distance. Earlier
actors' movement is already reflected, later movement is not inferred. Chebyshev
distance reduction/increase/plateau are toward/away/equal. Stationary actions and
blocked move attempts are reported separately. Remembered-target measures use
the same pre-update memory membership as the learned feature.

Social exposure uses actor opportunities, including dying actors and excluding
birth-tick newborns. Contact lengths count consecutive actual opportunities;
absence closes an episode, death/end are separate censoring reasons. Repeated
contacts mean previously observed identity. Long-gap encounters require an
intervening missed opportunity and elapsed sighting gap ≥20 ticks. A remembered
reunion is a long-gap contact with pre-update LTM membership; remote target
death is never supplied to memory. Distinct partners are measured per observer.

Movement strata retain pre-action local/visible resource presence, hunger/thirst
bins, age//25 and visible-count bin. The last is a sensory exposure, not a fixed
physical-density estimate when visual radius evolves. Death-age distributions
condition on death; surviving ages are right-censored, not completed lifetimes.
Current IDENTIFIED/MASKED/SILENCE hearing is distinct from emitted vocal actions.

### State utilization

Observed physical action associations match the same agent, 100-tick window and
exact original key (including auditory features), then compare supported social
cells with ≥5 decisions. All decisions and exploitative-only subsets are separate.
These are observational associations, not counterfactual preferences.

Read-only Q probes at completed 100-tick checkpoints use retained observed rows
in the **same current living brain**. Both rows require ≥2 or ≥5 visits in their
current residence and ≥1 TD update. They share the exact original key.
`memory_only` additionally fixes B's visible token. The probe compares greedy
maximal-action sets, epsilon-policy total variation and maximum Q difference.
It allocates no rows, changes no recency, executes no actions and consumes no RNG.
Absent/synthetic states are excluded. Individual action values can still retain
optimistic zero initialization; support does not establish optimality, deliberate
social use or preference. Checkpoints condition on survival and current retention.
Pairs are diagnostic support; seeds remain independent replication units.

## Preregistered execution and reproduction

Pilot: A/B/C × seeds 0/1/2, 100 ticks. After correctness/runtime/storage review,
primary: the same conditions/seeds, 500 ticks. All start from identical seeded
ecology, passive immediate-LRU visual memory K=32, physical/vocal Q capacity 256,
and default evolution parameters. Unlike v0.7.1, action and ecology trajectories
may diverge and cannot be replaced by common-history offline replay.

Optional later phases support C immediate-LRU versus five-sighting/decay K=32,
tau=100, candidate budget 32; and B/C physical Q capacity 1024 versus primary
256, holding vocal capacity fixed. Candidate storage is additional working
storage, not equal total cognitive storage. Neither optional phase automatically
runs before primary correctness checks. The runner refuses overwrites and
horizons beyond 500; no 1000-tick factorial sweep is launched.

```bash
python3 main.py --mode evolution --brain learning --seed 0 --ticks 100 \
  --social-learning-mode perception \
  --social-learning-csv data/social_learning_v072/example_B.csv.gz
python3 main.py --mode evolution --brain learning --seed 0 --ticks 100 \
  --social-memory --social-learning-mode memory \
  --social-learning-csv data/social_learning_v072/example_C.csv.gz
python3 -m experiments.social_learning_experiments --phase pilot
python3 -m experiments.social_learning_experiments --phase primary
python3 -m experiments.social_learning_experiments --phase memory_policy
python3 -m experiments.social_learning_experiments --phase q_capacity
python3 -m experiments.analyze_social_learning \
  --data-dir data/social_learning_v072/primary \
  --output-dir analysis/social_learning_v072/reanalysis
python3 -m unittest discover -s tests
```

Use `.venv/bin/python` with the repository's existing analysis dependencies if
system Python lacks them. Simulation remains standard-library-only. New raw data
and outputs are under `data/social_learning_v072/` and
`analysis/social_learning_v072/`. Earlier v0.7.0/v0.7.1 datasets are preserved.

<!-- RESULTS -->

## Implementation verification and compatibility

The unchanged baseline suite passed **523 tests** before extension. The complete
suite now passes **554 tests**: all 523 existing tests and 31 new tests, with
zero failures/errors/skips. No historical tests or fixtures were weakened.
The seven analysis tests were rerun successfully after the final display-only
plot refinement. Tests cover immutable encoding, structural bounds, invalid
combinations, ties/HERE/zero radius, candidate/fifth-sighting timing, same-batch
eviction, reappearance, newborn/death/sequential movement, no auditory identity,
vocal isolation, reward/recording/RNG invariance, real allocation hooks, actual
movement, exact probes, empty populations, metadata integrity and output replay.

An untouched archive of v0.7.1 was executed independently. Three 40-tick complete
trajectories/action streams match (RandomBrain, listening LearningBrain and
non-listening LearningBrain; seed 7, eight founders). Baseline A also matches
three 100-tick archived full states and the saved historical social gzip fixture.
At **500 ticks, seeds 0/1/2 all match the archived v0.7.1 complete final state**:
bodies/genomes, resources, both Q heads/row order/pending transitions, auditory
memory, demography, private social LRU and every RNG state. These are exact
hash comparisons, not population-only checks. Saved v0.7.1 500-tick aggregate
statistics agree where available. See data/v072_legacy_500_control.json.

With identical supplied observations and physiological outcomes, the new
physical features leave Vocal Q keys, values, pending transitions and RNG states
identical to the legacy learner. In actual ecologies, changed physical behavior
can legitimately change future hearing and shared physiological rewards, and
therefore indirectly change vocal learning.

Recording ON/OFF is tested against complete states after every tick in all three
modes. Actual pilot reruns reproduce **81 raw/sidecar files plus 27 analysis
files** byte-for-byte, including JSON/PNG; reversed input order and Python/NumPy
RNG isolation are verified. After final plot QA, all 27 revised analysis outputs
were independently reproduced and every analysis CSV remained unchanged.

Diagnostic inspection found that detailed Event.observation still showed the
internal visual batch in new modes. It was corrected to the actual coarse
Brain-facing snapshot and tested before the final pilot/primary. The initial
pilot and its exact core-source snapshot remain in data/v072_initial_pilot_archive.
All 72 raw gzip streams and nine complete action/state fingerprints are identical
before/after that logging correction. No biological or learning fix was needed.

Existing Config/SocialConfig/SocialPolicyConfig class schemas and all historical
v0.7.0/v0.7.1 files are preserved. The final verification binds the baseline
snapshot, current sources, raw files, scientific outputs and test receipts.

## Executed conditions and replication

Final pilot: A/B/C × seeds 0/1/2 × 100 ticks, independently rerun. Primary:
A/B/C × the same seeds × 500 ticks. Primary correctness, runtime/storage and
support were reviewed before the optional three new **C_decay** 500-tick runs;
that analysis reuses the original three primary C controls without rerunning
them. C_decay means **N=5 plus contact decay**, K=32, tau=100, candidate budget
32, with exactly C's input definitions. It does not isolate threshold from
retention and has extra candidate storage. Q-capacity sensitivity was supported
but deferred: primary churn was below 0.43% of decisions and sparse repetition
was the larger concern. No 1000-tick sweep or parameter tuning was performed.

All primary conditions share normalized initial ecological/RNG fingerprints.
The memory variant shares C's initial ecology. New modes have different action
and final-state hashes from A; this is expected, not a compatibility failure.
Pilot and primary first-100-tick action checkpoints and aggregate rows agree for
each condition/seed. Paired seeds do not imply identical realized encounter,
resource or demographic histories. **n=3 seeds**, not millions of contacts.
All rates below are calculated per seed before equal-seed means and sample SD
(ddof=1). Denominators remain in CSVs; zero exposure is NA, not a measured zero.

## Ecological outcomes: primary per-seed results

| Condition | Seed | Final population | Births | Deaths | Food units | Water units | Highest generation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 832 | 2,373 | 1,641 | 12,184 | 12,183 | 16 |
| A | 1 | 823 | 2,342 | 1,619 | 12,092 | 12,017 | 16 |
| A | 2 | 799 | 2,314 | 1,615 | 12,025 | 12,035 | 17 |
| B | 0 | 831 | 2,627 | 1,896 | 12,050 | 12,016 | 16 |
| B | 1 | 810 | 2,606 | 1,896 | 11,857 | 11,858 | 16 |
| B | 2 | 786 | 2,497 | 1,811 | 11,903 | 11,896 | 16 |
| C | 0 | 846 | 2,563 | 1,817 | 12,104 | 12,000 | 16 |
| C | 1 | 770 | 2,603 | 1,933 | 11,921 | 11,850 | 17 |
| C | 2 | 799 | 2,518 | 1,819 | 11,840 | 11,882 | 17 |

| Metric | A mean ± sample SD | B | C |
| --- | --- | --- | --- |
| population | 818.000 ± 17.059 | 809.000 ± 22.517 | 805.000 ± 38.354 |
| births | 2,343.000 ± 29.513 | 2,576.667 ± 69.788 | 2,561.333 ± 42.525 |
| deaths | 1,625.000 ± 14.000 | 1,867.667 ± 49.075 | 1,856.333 ± 66.403 |
| food_consumed | 12,100.333 ± 79.827 | 11,936.667 ± 100.808 | 11,955.000 ± 135.244 |
| water_consumed | 12,078.333 ± 91.090 | 11,923.333 ± 82.470 | 11,910.667 ± 79.002 |
| highest_generation | 16.333 ± 0.577 | 16.000 ± 0.000 | 16.667 ± 0.577 |
| mean_death_age | 133.961 ± 0.968 | 133.761 ± 1.324 | 133.534 ± 1.415 |
| mean_survivor_age | 93.530 ± 0.642 | 89.277 ± 1.995 | 88.908 ± 2.325 |

B and C have more births and more deaths than A in every seed. B's final
population is slightly lower in all three seeds; C−A has mixed signs. Population
alone does not measure learned social value. Death ages condition on having
died; survivor ages are right-censored. Different birth timing and age structure
prevent interpreting these simple means as matched survival probabilities.
Death causes, reproduction attempts, resource regeneration, evolving genome means
and the full original aggregate statistics are preserved in tick streams.

| Condition | Seed | Starvation | Dehydration | Energy | Age deaths |
| --- | --- | --- | --- | --- | --- |
| A | 0 | 326 | 976 | 339 | 0 |
| A | 1 | 336 | 941 | 342 | 0 |
| A | 2 | 345 | 937 | 333 | 0 |
| B | 0 | 381 | 1,103 | 412 | 0 |
| B | 1 | 390 | 1,149 | 357 | 0 |
| B | 2 | 366 | 1,062 | 383 | 0 |
| C | 0 | 404 | 1,045 | 368 | 0 |
| C | 1 | 401 | 1,111 | 421 | 0 |
| C | 2 | 415 | 1,052 | 352 | 0 |

## Learning coverage and Q capacity

| Condition | Seed | Decisions | Distinct global keys | Distinct agent-state pairs | First-state decisions (%) | Q evictions | Evictions / decisions (%) | Mean rows / decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 295,455 | 22,823 | 201,664 | 68.255 | 80 | 0.0271 | 54.226 |
| A | 1 | 294,189 | 22,737 | 200,844 | 68.270 | 199 | 0.0676 | 54.383 |
| A | 2 | 292,898 | 23,094 | 199,385 | 68.073 | 38 | 0.0130 | 54.404 |
| B | 0 | 325,958 | 42,841 | 261,759 | 80.305 | 1,398 | 0.4289 | 64.660 |
| B | 1 | 325,742 | 43,091 | 262,110 | 80.466 | 1,022 | 0.3137 | 64.270 |
| B | 2 | 314,387 | 42,494 | 251,021 | 79.845 | 1,000 | 0.3181 | 64.635 |
| C | 0 | 317,990 | 46,423 | 256,401 | 80.632 | 1,039 | 0.3267 | 64.359 |
| C | 1 | 323,971 | 48,037 | 262,643 | 81.070 | 1,068 | 0.3297 | 63.978 |
| C | 2 | 316,090 | 46,354 | 254,021 | 80.364 | 1,256 | 0.3974 | 65.004 |

| Metric | A mean ± SD | B | C |
| --- | --- | --- | --- |
| new_state_fraction | 68.200 ± 0.110 | 80.205 ± 0.322 | 80.688 ± 0.357 |
| q_evictions_per_decision | 0.036 ± 0.028 | 0.354 ± 0.065 | 0.351 ± 0.040 |
| q_full_fraction | 0.057 ± 0.041 | 0.448 ± 0.086 | 0.440 ± 0.050 |
| mean_physical_occupancy | 54.338 ± 0.097 | 64.522 ± 0.218 | 64.447 ± 0.518 |
| exploratory_fraction | 20.000 ± 0.137 | 19.957 ± 0.153 | 19.909 ± 0.171 |
| updates | 293,368.333 ± 1,263.501 | 321,226.000 ± 6,597.236 | 318,553.333 ± 4,138.411 |

| Condition | Seed | Mean visits per agent-state so far | States seen once (%) |
| --- | --- | --- | --- |
| A | 0 | 1.465 | 76.365 |
| A | 1 | 1.465 | 76.403 |
| A | 2 | 1.469 | 76.191 |
| B | 0 | 1.245 | 83.743 |
| B | 1 | 1.243 | 83.883 |
| B | 2 | 1.252 | 83.469 |
| C | 0 | 1.240 | 84.241 |
| C | 1 | 1.234 | 84.554 |
| C | 2 | 1.244 | 83.971 |

The extra features increase observed novelty and split already limited
experience. The table's 11/37 multiplier is structural; observed coverage is much
smaller and differs with endogenous population/exposure. First lifetime decision
keys begin with zero Q values, so the novelty rate is a lower bound on decisions
without a learned row-specific preference. The exploitative branch can still
choose uniformly among zero ties; it is not proof of informed exploitation.

More physical row churn occurs under B/C, but absolute eviction rates remain
small and occupancy rarely reaches capacity. This does not remove the **sample
fragmentation** confound, which exists even without eviction. More births or
different movement cannot be attributed solely to predictive social information.
The original capacity was not raised. Global distinct keys and per-agent lifetime
distinct keys count different things; visits/death/end censoring are explicit.

## Learned policy utilization versus observed behavior

### Read-only final physical-Q probes, ≥5 current-row visits

| Condition | Seed | Qualified rows | Matched groups | Supported brains | Pairs | Mean policy TV | Changed greedy sets / pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 1,032 | 0 | 0 | 0 | NA | NA |
| A | 1 | 1,076 | 0 | 0 | 0 | NA | NA |
| A | 2 | 976 | 0 | 0 | 0 | NA | NA |
| B | 0 | 195 | 12 | 6 | 20 | 0.4973 | 0.8500 |
| B | 1 | 179 | 6 | 5 | 13 | 0.5641 | 0.8462 |
| B | 2 | 206 | 10 | 7 | 12 | 0.4222 | 0.6667 |
| C | 0 | 166 | 3 | 2 | 7 | 0.4571 | 0.5714 |
| C | 1 | 165 | 12 | 7 | 38 | 0.5211 | 0.7105 |
| C | 2 | 249 | 15 | 8 | 27 | 0.4533 | 0.6296 |

B/C have trained retained rows with the same exact original key but
different social suffixes and different learned Q preferences. This demonstrates
implementation-level input dependence on the **supported subset**, not optimal
social decisions. A has no social key and therefore no such counterfactual pairs;
its NA is not a measured zero effect. At tick 500, only a small number of living
brains support these comparisons; no claim is made for the unobserved majority.

For primary C, **memory-only comparisons with ≥5 visits have zero pairs at all
five checkpoints in all three seeds**. Thus additional learned use of remembered
status beyond B's visible features is not established in the primary experiment.
The prespecified ≥2-visit sensitivity has sparse support, shown below; it does
not replace the ≥5 criterion or turn missing evidence into a negative effect.

| C seed | Tick | Pairs at ≥2 visits | Supported brains | Mean TV |
| --- | --- | --- | --- | --- |
| 0 | 500 | 2 | 2 | 0.1238 |
| 1 | 500 | 0 | 0 | NA |
| 2 | 500 | 3 | 2 | 0.5333 |

### Observed action associations, same actor/window/original key

| Condition | Seed | Qualified cells | Matched groups | Cell pairs | Supported decisions | Mean observed action TV |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 569 | 42 | 57 | 602 | 0.1305 |
| A | 1 | 538 | 40 | 46 | 535 | 0.1573 |
| A | 2 | 569 | 50 | 64 | 748 | 0.1970 |
| B | 0 | 765 | 69 | 99 | 1,052 | 0.6966 |
| B | 1 | 773 | 68 | 94 | 1,073 | 0.6260 |
| B | 2 | 785 | 72 | 108 | 1,128 | 0.6159 |
| C | 0 | 787 | 83 | 110 | 1,292 | 0.6429 |
| C | 1 | 755 | 80 | 123 | 1,227 | 0.6475 |
| C | 2 | 825 | 71 | 98 | 1,096 | 0.6391 |

These empirical action comparisons are not frozen-Q counterfactuals.
Even A can show context associations despite having no social key: compressed
original states, changing Q values within a window, resource quantities, geometry
and finite sample variation remain. Action-association TV is not proof of social
preference. Individual unvisited action values may still be optimistic zeros
even in otherwise trained Q rows; no language/convention is inferred.

## Movement, contact and observable confounds

| Condition | Seed | Toward / actual target moves | Rate (%) | Remembered toward / target moves | Rate (%) | Historical long-gap contacts | Remembered reunion rate (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 9,941 / 142,115 | 6.9950 | 10,047 / 140,809 | 7.1352 | 51,947 | 10.8668 |
| A | 1 | 9,834 / 141,851 | 6.9326 | 10,001 / 140,541 | 7.1161 | 52,405 | 10.8291 |
| A | 2 | 9,825 / 140,463 | 6.9947 | 9,957 / 139,184 | 7.1538 | 53,528 | 10.2283 |
| B | 0 | 10,084 / 157,217 | 6.4141 | 10,239 / 155,790 | 6.5723 | 64,512 | 7.6466 |
| B | 1 | 10,300 / 157,668 | 6.5327 | 10,460 / 156,224 | 6.6955 | 66,423 | 7.1662 |
| B | 2 | 10,170 / 151,400 | 6.7173 | 10,270 / 150,084 | 6.8428 | 63,693 | 8.0794 |
| C | 0 | 10,038 / 153,304 | 6.5478 | 10,188 / 151,906 | 6.7068 | 63,577 | 7.6569 |
| C | 1 | 10,202 / 156,402 | 6.5229 | 10,386 / 154,916 | 6.7043 | 65,036 | 7.6419 |
| C | 2 | 10,110 / 151,682 | 6.6653 | 10,222 / 150,256 | 6.8031 | 62,665 | 8.7848 |

| Metric | A mean ± SD (%) | B | C |
| --- | --- | --- | --- |
| visible_fraction | 99.4483 ± 0.0702 | 99.4974 ± 0.0505 | 99.3717 ± 0.0482 |
| remembered_fraction | 98.5373 ± 0.0796 | 98.6000 ± 0.0564 | 98.4570 ± 0.0715 |
| toward_fraction | 6.9741 ± 0.0359 | 6.5547 ± 0.1528 | 6.5787 ± 0.0760 |
| away_fraction | 66.8493 ± 0.1652 | 69.0546 ± 0.4552 | 68.9768 ± 0.3604 |
| equal_fraction | 26.1766 ± 0.1924 | 24.3907 ± 0.3355 | 24.4445 ± 0.3032 |
| remembered_toward_fraction | 7.1350 ± 0.0189 | 6.7036 ± 0.1354 | 6.7380 ± 0.0563 |
| reunion_recognition | 10.6414 ± 0.3583 | 7.6307 ± 0.4568 | 8.0279 ± 0.6556 |

Both B/C have a lower unadjusted toward fraction than A in every seed.
This is measurable behavior under the stated frozen-target metric, **not evidence
of avoidance**. Nearest-target geometry and exposure change with the ecology.
The long-gap denominator here includes *any previously observed* target with an
intervening absence and gap≥20; it is not v0.7.1's five-sighting-restricted primary
eligibility denominator, and cross-version percentages are not interchangeable.

| Condition | Nearest target HERE: seed 0 (%) | seed 1 | seed 2 | Mean ± SD (%) |
| --- | --- | --- | --- | --- |
| A | 47.396 | 47.044 | 47.460 | 47.300 ± 0.224 |
| B | 51.149 | 51.397 | 50.163 | 50.903 ± 0.653 |
| C | 50.550 | 51.575 | 50.357 | 50.827 ± 0.655 |
| C_decay | 51.438 | 51.071 | 51.530 | 51.346 ± 0.242 |

Co-location is directly observed from the packed visual token, not
inferred from an action label. A HERE target has zero initial distance, which no
movement can reduce. Its increased prevalence is compatible with lower apparent
approach fractions. These are decision exposures, not the exact geometry
distribution among movers, so they do not quantitatively explain the effect.
They demonstrate why an attraction/avoidance conclusion would be premature.

| Condition | Any food/water on actor tile | Toward mean ± SD (%) | Min target-move exposure per seed | Finite n |
| --- | --- | --- | --- | --- |
| A | False | 6.4820 ± 0.0067 | 93,534 | 3 |
| A | True | 7.9409 ± 0.1035 | 46,929 | 3 |
| B | False | 5.7892 ± 0.1898 | 93,508 | 3 |
| B | True | 7.8094 ± 0.0877 | 57,892 | 3 |
| C | False | 5.7876 ± 0.1461 | 93,632 | 3 |
| C | True | 7.8693 ± 0.0528 | 58,050 | 3 |

This coarse resource stratification does not fully remove the movement
difference, but leaves geometry, age, needs, food/water amounts and endogenous
population composition as confounds. Complete pre-action hunger/thirst-bin,
age-bin, visible-count and local/visible-resource strata with actual movement
denominators are saved in movement_strata.csv. No missing cohort information is
inferred and no preference model is fitted.

## Vocal and auditory observations

| Condition | Seed | Emitted signals | Signals / decisions (%) | IDENTIFIED / decisions (%) | MASKED / decisions (%) |
| --- | --- | --- | --- | --- | --- |
| A | 0 | 278,174 | 94.1511 | 31.8570 | 68.1092 |
| A | 1 | 276,964 | 94.1449 | 31.9886 | 67.9774 |
| A | 2 | 275,785 | 94.1574 | 31.6810 | 68.2849 |
| B | 0 | 307,077 | 94.2075 | 33.1911 | 66.7782 |
| B | 1 | 306,653 | 94.1398 | 33.1895 | 66.7798 |
| B | 2 | 295,876 | 94.1120 | 32.6951 | 67.2731 |
| C | 0 | 299,336 | 94.1338 | 32.9010 | 67.0675 |
| C | 1 | 304,807 | 94.0847 | 33.0042 | 66.9649 |
| C | 2 | 297,430 | 94.0966 | 32.7609 | 67.2074 |

Signal production stays near 94.1%, close to the 16/17 nonsilent actions
of the unchanged default vocabulary. The familiar genotype signal-probability
rule still applies in the existing random-emission ablation; it does not control
the learned vocal head. Current resolved hearing changes slightly across social
conditions, consistent with different proximity and ecology. The vocal action
IDs remain categorical and meaningless; neighboring IDs have no intrinsic
similarity and are not connected as a phonetic continuum in the figure.
Changes in vocal exposure or the shared-reward vocal learner do not demonstrate
intentional communication, common semantics or language emergence.

## Optional memory-policy comparison

| Policy | Seed | Population | Births | Deaths | Remembered exposure (%) | First-state decisions (%) | Q evictions | Toward (%) | Reunion recognition (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | 0 | 846 | 2,563 | 1,817 | 98.524 | 80.632 | 1,039 | 6.5478 | 7.6569 |
| C | 1 | 770 | 2,603 | 1,933 | 98.466 | 81.070 | 1,068 | 6.5229 | 7.6419 |
| C | 2 | 799 | 2,518 | 1,819 | 98.381 | 80.364 | 1,256 | 6.6653 | 8.7848 |
| C_decay | 0 | 841 | 2,621 | 1,880 | 94.468 | 83.792 | 1,795 | 6.4797 | 25.2241 |
| C_decay | 1 | 783 | 2,576 | 1,893 | 94.530 | 83.866 | 1,805 | 6.4555 | 25.1645 |
| C_decay | 2 | 785 | 2,586 | 1,901 | 94.450 | 83.804 | 1,468 | 6.4478 | 26.2159 |

| Metric | C-LRU mean ± SD | C N5+decay mean ± SD |
| --- | --- | --- |
| population | 805.000 ± 38.354 | 803.000 ± 32.924 |
| births | 2,561.333 ± 42.525 | 2,594.333 ± 23.629 |
| deaths | 1,856.333 ± 66.403 | 1,891.333 ± 10.599 |
| remembered_fraction | 98.457 ± 0.071 | 94.483 ± 0.042 |
| new_state_fraction | 80.688 ± 0.357 | 83.821 ± 0.040 |
| mean_physical_occupancy | 64.447 ± 0.518 | 67.363 ± 0.225 |
| q_evictions_per_decision | 0.351 ± 0.040 | 0.520 ± 0.061 |
| toward_fraction | 6.579 ± 0.076 | 6.461 ± 0.017 |
| reunion_recognition | 8.028 ± 0.656 | 25.535 ± 0.591 |

| C N5+decay seed | Tick | Qualified memory-only pairs | Supported brains | Mean TV |
| --- | --- | --- | --- | --- |
| 0 | 100 | 0 | 0 | NA |
| 0 | 200 | 6 | 6 | 0.3587 |
| 0 | 300 | 6 | 6 | 0.4000 |
| 0 | 400 | 1 | 1 | 0.8000 |
| 0 | 500 | 0 | 0 | NA |
| 1 | 100 | 0 | 0 | NA |
| 1 | 200 | 4 | 4 | 0.2000 |
| 1 | 300 | 5 | 5 | 0.3200 |
| 1 | 400 | 0 | 0 | NA |
| 1 | 500 | 0 | 0 | NA |
| 2 | 100 | 0 | 0 | NA |
| 2 | 200 | 6 | 6 | 0.4000 |
| 2 | 300 | 5 | 5 | 0.4800 |
| 2 | 400 | 2 | 2 | 0.4000 |
| 2 | 500 | 0 | 0 | NA |

The predeclared combined policy produces fewer remembered-visible
decisions, higher novelty/churn and different actions/ecology. Its long-gap
recognition rate is higher, but these are **different realized histories**;
this is not the pure fixed-history memory comparison performed in v0.7.1.
It does not improve final population consistently across seeds.

At ticks 200/300, every seed supplies a small number of ≥5-visit memory-only
trained Q-row comparisons, with nonzero policy differences. Thus remembered
direction can affect learned physical preferences in this supported subset.
Support is only a handful of brains and disappears at the final checkpoint;
it cannot establish population-wide predictive benefit or social motivation.
Candidate budget 32 is additional to LTM K=32; equal long-term capacity is not
equal total storage. Threshold and decay changed together, not independently.

## Paired-seed descriptive differences

| Contrast | Metric | Mean difference ± sample SD | Min | Max | n |
| --- | --- | --- | --- | --- | --- |
| B-A | births | 233.6667 ± 44.1626 | 183.0000 | 264.0000 | 3 |
| B-A | deaths | 242.6667 ± 41.8848 | 196.0000 | 277.0000 | 3 |
| B-A | identified_fraction (pp) | 1.1830 ± 0.1608 | 1.0141 | 1.3341 | 3 |
| B-A | new_state_fraction (pp) | 12.0052 ± 0.2152 | 11.7714 | 12.1951 | 3 |
| B-A | population | -9.0000 ± 6.9282 | -13.0000 | -1.0000 | 3 |
| B-A | q_evictions_per_decision (pp) | 0.3177 ± 0.0786 | 0.2461 | 0.4018 | 3 |
| B-A | toward_fraction (pp) | -0.4194 ± 0.1527 | -0.5810 | -0.2774 | 3 |
| C-A | births | 218.3333 ± 37.6076 | 190.0000 | 261.0000 | 3 |
| C-A | deaths | 231.3333 ± 72.9475 | 176.0000 | 314.0000 | 3 |
| C-A | identified_fraction (pp) | 1.0465 ± 0.0322 | 1.0156 | 1.0799 | 3 |
| C-A | new_state_fraction (pp) | 12.4887 ± 0.2726 | 12.2903 | 12.7995 | 3 |
| C-A | population | -13.0000 ± 35.3412 | -53.0000 | 14.0000 | 3 |
| C-A | q_evictions_per_decision (pp) | 0.3154 ± 0.0627 | 0.2620 | 0.3844 | 3 |
| C-A | toward_fraction (pp) | -0.3955 ± 0.0602 | -0.4473 | -0.3295 | 3 |
| C-B | births | -15.3333 ± 43.8216 | -64.0000 | 21.0000 | 3 |
| C-B | deaths | -11.3333 ± 60.3683 | -79.0000 | 37.0000 | 3 |
| C-B | identified_fraction (pp) | -0.1365 ± 0.1829 | -0.2901 | 0.0659 | 3 |
| C-B | new_state_fraction (pp) | 0.4835 ± 0.1419 | 0.3273 | 0.6044 | 3 |
| C-B | population | -4.0000 ± 31.1929 | -40.0000 | 15.0000 | 3 |
| C-B | q_evictions_per_decision (pp) | -0.0023 ± 0.0921 | -0.1021 | 0.0793 | 3 |
| C-B | toward_fraction (pp) | 0.0240 ± 0.0974 | -0.0520 | 0.1337 | 3 |

These differences are descriptive three-seed replications. Common
initial seeds are not common observed social histories after actions diverge.
No significance test, independent-contact n or claim of ecological superiority
is made. C−B isolates availability of additional memory features at configuration
level, but not their predictive value independently of representation sparsity
and subsequent ecological feedback.

## Performance, storage and reproduction receipts

| Phase | Ticks | New online runs | Wall s | Analysis s | Raw MB | Raw stream rows | Analysis MB | Peak process RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot | 100 | 9 | 24.588 | 1.523 | 0.953 | 136896 | 1.657 | 181.83 |
| primary | 500 | 9 | 685.653 | 3.944 | 18.051 | 3013437 | 4.435 | 469.84 |
| memory_policy | 500 | 3 | 273.049 | 3.421 | 6.495 | 1046077 | 3.370 | 531.08 |

Per-run timing, output sizes, seeds, exact configurations, hashes and
record counts are saved in phase runtime/design receipts. Simulation-loop timing
includes diagnostics and checkpoint probes, but excludes separate complete-state
hashing in the runner. Phase wall time includes hashing, output and analysis.
Process RSS includes tables, privileged analysis histories, hash buffers and
plotting; it is not per-agent cognitive storage. Timings are local measurements,
not portable benchmarks. Contact/state histories grow with live exposure and
global catalogs; recording OFF creates none of them. Window action/stratum cells
are flushed rather than retaining every decision row or Q-value.

The pilot review and primary review precede subsequent stages. A complete
500-tick repeat of every new condition was not run: real 100-tick A/B/C replay,
forced-capacity/terminal unit tests, recording isolation, prefix equivalence and
exact 500-tick archived A controls provide the stated verification scope. The
optional Q-capacity runner remains available without changing default capacity.
No new 1000-tick ecological experiment was executed.

## Files and saved outputs

Modified: config.py (companion config only), agents/learning.py (physical-key
hook and optional observational hooks; vocal key explicitly separate),
simulation/simulation.py (one pre-update snapshot, feature/observer wiring),
main.py (validated CLI/config metadata), README and .gitignore.

Added: agents/social_learning.py; simulation/social_learning_observer.py;
experiments/social_learning_experiments.py, analyze_social_learning.py and
social_learning_plots.py; tests/test_social_learning.py and
test_social_learning_analysis.py; this report.

Each of pilot/primary/memory_policy under analysis/social_learning_v072 contains
**13 CSVs + 13 PNGs + analysis_metadata.json (27 files)**. CSVs include per-seed
run summary, equal-seed means/sample SD/min/max/n, tick/window series,
distributions, probes, matched state utilization, action/vocal probabilities,
movement strata and paired differences. Memory-policy analysis reuses the three
primary C controls explicitly. Raw data are eight compressed streams plus a
completion sidecar per online run, with stage designs/runtime receipts.

Key figures: [population](../analysis/social_learning_v072/primary/population_over_time.png),
[Q occupancy/churn](../analysis/social_learning_v072/primary/q_occupancy_and_evictions.png),
[actual relative movement](../analysis/social_learning_v072/primary/movement_relative.png),
[Q-policy sensitivity](../analysis/social_learning_v072/primary/counterfactual_policy_sensitivity.png).

The new diagnostics contain complete original aggregate statistics. Historic
recording formats and research outputs were not overwritten. Verification,
archived controls, source snapshots, test logs and real replay receipts are
data/v072_*. The initial diagnostic-format pilot is archived separately with
its exact source; it is not pooled with final experiments.

## Answer, limitations and next experiment

**Implementation:** verified. Social features reach only Physical Q, at the
correct pre-commit decision time. None mode remains exactly v0.7.1-equivalent.

**Learning:** general social-key dependence is measurable in supported trained
rows. Primary immediate-LRU memory-specific use is unresolved at the declared
five-visit support. The combined memory policy yields a few supported
memory-specific comparisons at intermediate checkpoints, not broad coverage.

**Behavior:** action trajectories and unadjusted movement/contact statistics
change reproducibly. More co-location and different resource/exposure histories
confound attraction/avoidance interpretations. Frozen targets are a clearly
defined measurement convention, not counterfactual social intentions.

**Ecology:** births/deaths change; final population does not show a consistent
benefit. State sparsity and zero-prior decisions are major alternative
explanations. Small eviction rates do not eliminate fragmentation. Only three
seeds and one default ecology are studied; no significance/general optimality
claim is made.

**Communication:** vocal algorithms/states are unchanged; production/hearing
statistics vary indirectly. No signal meaning, intentional receiver selection,
social reward, friendship, cooperation or language is implemented or established.

The next experiment should distinguish predictive social information from
state-partition/sample-size effects: predeclare supported-context requirements
and consider a matched-cardinality non-social information control or more targeted
exposure replication under unchanged rewards. Merely enlarging Q tables will not
solve infrequent revisits. Retain the current results as baseline; do not add a
social reward to force visible behavior. No v0.7.3/v0.8 mechanism is included.
