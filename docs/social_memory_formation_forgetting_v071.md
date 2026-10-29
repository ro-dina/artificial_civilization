# v0.7.1 — Social Memory Formation, Retention and Forgetting

## Research objective and scope

Does selective formation and experience-dependent retention improve recognition
of historically eligible individuals relative to immediate encoding and LRU,
on the same ecological encounter histories and bounded long-term memory?

This release changes memory mechanics and measurement only. Visual individual
recognition is an imposed capability, not learned recognition. Identity is not
inferred from vocal signals. Contact strength is not friendship. The retention
score is an experimental hypothesis, not a validated model of human memory.
There is no social reward and no social feature enters either Q-learning head.
No evidence of cooperation, communication or language emergence is claimed.

The unmodified ecology is `Config.evolution(brain="learning")`. Physical/vocal
actions, both Q heads, rewards, audition, vocabulary, masking, genome rules,
reproduction, resource regeneration, mortality and RNG streams are preserved.
No v0.7.2 mechanisms, voices, self-masking or new ecological interactions were
implemented. Historical social data, fixtures and analysis outputs are retained.

## Implementation and exact semantics

The original `SocialMemory` and `SocialConfig` are unchanged. The immediate+LRU
factory path returns that original class, rather than emulating it with a new
algorithm. `Config` retains its original dataclass schema. A companion frozen
`SocialPolicyConfig` supplies `formation_threshold=1`, `eviction_policy="lru"`,
`decay_tau=100.0`, `strength_mode="log_contact_decay"` and
`candidate_capacity=None`. The last value resolves explicitly to long-term K.
Threshold/candidate capacity must be positive integers, policy/mode must be
supported, and tau must be finite and positive. Invalid input is rejected.

### Formation and bounded working storage

For threshold N, identity commitment requires N consecutive **actual observer
opportunities**, including empty observations as possible interruptions. Elapsed
global ticks are not substituted for opportunities. A missing candidate is
removed at the next opportunity; reappearance starts at progress one. Candidates
are unknown, consume no long-term slots, and trigger no remembered reunions.

Each update refreshes committed hits, then continuing candidates, then new
candidates, with ascending IDs within each phase. This preserves the original
known-before-new ordering and protects a progressing candidate from an unrelated
newcomer before its scheduled progress update. The bounded candidate buffer
always admits a newcomer when necessary by discarding the least recently seen
resident, lowest ID on a tie. Dense batches can therefore cause deterministic
candidate churn. Candidate reset and overflow eviction are separate events.

Commitment occurs on sighting N, with initial contact strength N. In particular,
all five forming sightings count at N=5. First forming sighting and commitment
tick are separate. Subsequent actual sightings increment strength by one;
current streak resets to one after an intervening missed opportunity. Strength
is the count in the **current retained residence**, not a hidden lifetime counter.
Eviction removes this experience; relearning must again satisfy formation.

Each founder/newborn owns empty private memory. Newborns first observe on the
following tick. A dead observer's memory/candidates are released. A remote
target's death never updates another agent's memory or establishes private
knowledge of death; last directly seen alive and simulator-known death remain
different facts.

### Retention and deterministic ordering

LRU uses the existing recency ordering. The alternative score is

`M_i(t) = log(1 + C_i) * exp(-(t - last_i) / tau)`.

When a commitment needs a slot, the weakest **existing resident** is evicted,
then the newcomer is admitted. This is not a top-K admission filter including
the newcomer. Known contacts refresh before allocations. Multiple allocations
can evict a just-refreshed or just-admitted entry in a dense batch.

Comparison uses cached `log(log1p(C))` and cancels current t in pairwise score
differences. Same-time entries compare integer counts; opposing time/strength
terms use integer cross-products of the evaluated floating coefficients' exact
ratios. No exponential or RNG draw occurs. Tiny tau, large ticks and counts do
not underflow into arbitrary ties. Equal evaluated scores evict the lowest ID.
No tolerance introduces random or iteration-dependent near-ties. Diagnostic log
scores that cannot be represented are blank; exact count/time components remain.
There is no per-tick full-memory decay pass: the equivalent relative comparison
is evaluated only when a slot is needed.

## Recording, replay and metrics

New summary/optional transition-trace CSVs are versioned separately from the
unchanged v0.7.0 format. Summary has one row per observation plus lifecycle and
tick checkpoints. Trace records candidate creation/progress/reset/overflow,
commitment and LTM eviction, including strength/last-seen/residence information.
It is opt-in, not an additional row for every recognized contact. No policy
diagnostic payloads are constructed when recording is disabled.

Complete replay uses a separate compact **visual opportunity history**: sorted
simultaneous visible IDs for every actor opportunity, including empty ones,
exact lifecycle order, and complete tick checkpoints. This is sufficient because
the original social CSV records all visible identities, including unknowns.
Conversion validates its original metadata, SHA-256, schema and LRU replay;
missing opportunities are never guessed. Summary-only policy diagnostics are
not falsely treated as complete replay histories.

Alternative policies share one ordered encounter stream per seed. Offline
privileged lifetime/contact histories distinguish never seen, seen but never
committed, currently remembered, and previously committed but forgotten.
Those histories never enter agents. They are retained only for living observers;
private LTM/candidate structures remain bounded. CSVs stream rather than storing
all contact rows in RAM. Small histograms and per-window/age aggregates are kept.

### Recognition denominators

* **Primary:** contacts whose observer-target pair achieved at least five
  consecutive opportunities strictly **before** this contact, regardless of
  the tested policy. This denominator is identical across A–D for each seed.
* Recognition is membership before the current update. The fifth sighting that
  first commits an identity is neither a primary eligible contact nor a hit.
* Repeated-contact rate includes every contact after the first historical sighting.
* Previously-committed rate uses that policy's own prior commitment history;
  its denominator varies by policy and is secondary.
* Primary misses split into previously committed/evicted and never committed.
  A never-committed candidate is not a forgotten long-term memory.
* Qualifying reunion uses the shared historical eligibility, absence at an
  intervening opportunity, and `current_tick - last_seen_tick >= 20`.
  Recognition is still measured before refresh/recommitment.

Every rate retains numerator/denominator. Zero denominator is blank/NA, not
zero. Each seed contributes one rate to equal-seed summaries (sample SD,
ddof=1, finite n, min/max). Contacts are not independent replications. Paired
differences use matching seeds and identical primary exposure.

Occupancy is measured after updates, opportunity-weighted. Completed-tick living
agent exposure separately includes empty newborn memories. `active_observers`
counts distinct agents with an actual opportunity in the relevant group, not
final survivors. Mean age is opportunity-weighted pre-action age; age bins are
25 ticks. Population/birth/death counts come from actual source checkpoints.
Contact episode lengths count consecutive opportunities; observed absence,
observer death and run-end censoring are separate. Retention duration uses
commitment-to-eviction, with death/end-censored retained entries separated.
Unique observed pairs and unique visual target IDs are both reported.

## Configuration and execution

```bash
# Actual mechanism, separate new diagnostics, optional transition trace:
python3 main.py --mode evolution --brain learning --seed 0 --ticks 100 \
  --social-memory --social-formation-threshold 5 \
  --social-eviction-policy contact_decay --social-decay-tau 100 \
  --social-candidate-capacity 32 \
  --social-policy-csv data/social_memory_v071/example.csv.gz \
  --social-policy-trace

# Explicit stages; each refuses existing destinations:
python3 -m experiments.social_memory_experiments --phase pilot
python3 -m experiments.social_memory_experiments --phase primary
python3 -m experiments.social_memory_experiments --phase capacity
python3 -m experiments.social_memory_experiments --phase decay

# Pure replay; no Simulation.step calls, fresh destination required:
python3 -m experiments.analyze_social_memory_policies \
  --histories data/social_memory_v071/histories/seed_{0,1,2}_500.csv.gz \
  --horizon 500 --conditions A B C D --candidate-capacity 32 \
  --output-dir analysis/social_memory_v071/replay_check

python3 -m unittest discover -s tests
```

Use the repository `.venv/bin/python` if system Python lacks the existing
analysis dependencies. Simulation remains standard-library-only; analysis uses
the existing numpy/pandas/matplotlib environment, with no new dependency.
New output/sidecar collisions are refused. Nonlegacy policies cannot be labeled
with the legacy `--social-csv` schema. The runner defaults to 100 ticks and
requires a completed pilot receipt before a separately invoked primary phase.
Later explicit execution up to 1000 ticks is supported with `--allow-long`;
no large sweep is automatically launched.

## Design fixed before the experiments

`data/v071_design.json` records the design before the new ecological runs.

| Condition | Formation | Eviction | LTM K | Candidate budget | tau |
|---|---|---|---:|---:|---:|
| A | 1 | original LRU | 32 | unused | unused |
| B | 5 consecutive | LRU | 32 | 32 | unused |
| C | 1 | contact decay | 32 | unused | 100 |
| D | 5 consecutive | contact decay | 32 | 32 | 100 |

All stages use seeds 0, 1, 2. Pilot: 100 ticks, four policies. Primary: 500 ticks,
four policies. Capacity sensitivity: A/D, K=8/16/32/64/128, 100 ticks. Decay
sensitivity: D, tau=20/50/100/200, K=32, 100 ticks, with A as control. The
candidate budget stays **32** across capacity sensitivity: it does not grow with
K. The default API can resolve candidate capacity to K, but this experiment
explicitly holds it fixed.

The comparison controls long-term K, not total identity storage: B/D additionally
have up to 32 candidate identities. This must be considered when interpreting
formation effects. No policy parameter or ecology was tuned to improve an
observed result. Full 500-tick capacity-by-tau and new 1000-tick sweeps are deferred.

Existing seeds 0/1 supply complete validated 100-tick histories; seed 2 is a new
100-tick unchanged baseline. Primary seeds 0/2 are new 500-tick unchanged
baselines. Seed 1 reuses the complete ordered **prefix through tick 500** of the
existing 1000-tick recording. The entire original file hash and completion
metadata are bound, and original LRU replay is validated through the selected
prefix; this does not claim a fresh dynamic validation of ticks 501–1000.

<!-- RESULTS -->

## Compatibility verification

All **523 tests pass** (477 unchanged existing tests plus 46 new tests; no tests
removed or weakened). Before implementation, the existing 477 tests passed.
New tests cover formation/reset/overflow/batch ordering, strength and numerical
ties, relearning/reunions, malformed parameters, newborns/deaths, shared historical
eligibility and pre-commit timing, NA rates, censored episodes/durations, no
future-window leakage, recording isolation and reproducible replay/plots.

An untouched archive of baseline commit
`5e485f913dd2f95b4015688a2426d0af982f840e` was executed separately. Three 40-tick
complete trajectory/action digests match exactly: RandomBrain, LearningBrain
with listening, and LearningBrain without listening (seed 7, eight founders).
These comparisons include bodies/genomes, resources, demography, both Q tables
and their LRU order, pending transitions, reward/counters, communication buffers,
all RNG states, and the original social LRU. Three 100-tick final complete-state
digests (learning, seeds 0/1/2) also match the archive and all 12 A–D online
benchmark ecologies. The actual archived seed-0 100-tick social gzip SHA-256 and
aggregate statistics are reproduced exactly. Its social memory source hash is
unchanged. Existing metadata schema/config IDs and meanings remain compatible;
newly generated source hashes correctly describe changed implementation files.

Online/replay equality is tested after **every actor opportunity**, including
birth/death, in controlled fixtures. The real 100-tick three-seed A–D online
benchmark also matches each replay's complete final private-memory hash,
including candidate progress, entry order, strength, clocks and policy fields.
Recording/trace ON/OFF cannot alter private or ecological state. Python and
NumPy global RNG states remain untouched.

The real three-seed pilot was independently replayed with reversed input order
and Simulation.step forbidden: all **20 CSV/JSON/PNG files are byte-identical**.
An automated two-seed fixture repeats the same complete-output comparison.
The 100-tick pilot histories also match the first 100 ticks of all three primary
histories row-for-row, including empty opportunities and lifecycle ordering.
Primary 500-tick comparisons use common histories by construction; four new
500-tick online ecologies and an additional full 500-tick repeat were deliberately
not run. Original 1000-tick data are hashed/preserved; compatibility at that
horizon is not claimed from a newly executed 1000-tick Q-state comparison.

`data/v071_before.json` captures 240 baseline file hashes. Verification confirms
no missing/unexpected changes and preserves all historical artifacts. Only
README, .gitignore, config.py, main.py and simulation/simulation.py intentionally
change among that snapshot. Both original Config/SocialConfig class ASTs are
identical. Brain, reward, audition, genome, reproduction, resource, statistics,
original social memory/observer and historical analysis code remain unchanged.
See `data/v071_verification.json` and the referenced detailed receipts.

## Primary 500-tick results

Rates below use the common historical eligibility denominator. The fifth first
qualifying sighting is excluded until a later contact. Counts are descriptive;
millions of contacts do not increase independent replication beyond **n=3 seeds**.
Percentages use 0–100; SD and paired differences are percentage points (pp).

| Seed | Condition | Recognized | Eligible contacts | Recognition (%) | LTM evictions | LTM allocations |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | A | 2,784,326 | 2,892,046 | 96.2753 | 398,426 | 476,517 |
| 0 | B | 2,757,157 | 2,892,046 | 95.3359 | 139,274 | 214,428 |
| 0 | C | 1,574,095 | 2,892,046 | 54.4284 | 2,282,650 | 2,360,741 |
| 0 | D | 2,244,813 | 2,892,046 | 77.6202 | 210,775 | 285,929 |
| 1 | A | 2,812,160 | 2,944,407 | 95.5085 | 425,652 | 502,903 |
| 1 | B | 2,788,587 | 2,944,407 | 94.7079 | 141,562 | 216,131 |
| 1 | C | 1,595,769 | 2,944,407 | 54.1966 | 2,306,136 | 2,383,387 |
| 1 | D | 2,258,871 | 2,944,407 | 76.7173 | 216,049 | 290,618 |
| 2 | A | 2,834,001 | 2,958,601 | 95.7886 | 412,297 | 488,680 |
| 2 | B | 2,803,616 | 2,958,601 | 94.7615 | 141,563 | 215,111 |
| 2 | C | 1,592,729 | 2,958,601 | 53.8339 | 2,316,886 | 2,393,269 |
| 2 | D | 2,271,535 | 2,958,601 | 76.7773 | 216,413 | 289,961 |

| Condition | Recognition mean ± sample SD (%) | Min (%) | Max (%) | n | Evictions mean ± SD | Occupancy mean ± SD |
| --- | --- | --- | --- | --- | --- | --- |
| A | 95.8575 ± 0.3880 | 95.5085 | 96.2753 | 3 | 412,125.0 ± 13,613.8 | 29.2262 ± 0.0465 |
| B | 94.9351 ± 0.3481 | 94.7079 | 95.3359 | 3 | 140,799.7 ± 1,321.3 | 25.7267 ± 0.0627 |
| C | 54.1530 ± 0.2997 | 53.8339 | 54.4284 | 3 | 2,301,890.7 ± 17,508.4 | 29.2262 ± 0.0465 |
| D | 77.0383 ± 0.5049 | 76.7173 | 77.6202 | 3 | 214,412.3 ± 3,155.3 | 25.7267 ± 0.0627 |

| Paired contrast | Mean difference ± sample SD (pp) | Min (pp) | Max (pp) | n |
| --- | --- | --- | --- | --- |
| B-A | -0.9223 ± 0.1142 | -1.0270 | -0.8006 | 3 |
| C-A | -41.7045 ± 0.3442 | -41.9547 | -41.3119 | 3 |
| D-A | -18.8192 ± 0.1797 | -19.0112 | -18.6551 | 3 |
| D-B | -17.8968 ± 0.1569 | -17.9906 | -17.7156 | 3 |
| D-C | 22.8853 ± 0.3393 | 22.5207 | 23.1918 | 3 |

All three seeds have A > B > D > C on the primary metric. Under these
specific 500-tick histories, K=32 and tau=100, neither selective formation nor
contact-decay retention improves **overall historically eligible recognition**
over the original immediate LRU. This is a descriptive controlled memory
mechanics result, not evidence about survival, social ability or human cognition.
It is not a universal ordering across ecologies, time horizons or task weighting.

### Reunion tradeoff and secondary recognition

| Seed | Condition | Reunion hits | Common qualifying-reunion contacts | Reunion recognition (%) |
| --- | --- | --- | --- | --- |
| 0 | A | 3,504 | 30,252 | 11.5827 |
| 0 | B | 9,761 | 30,252 | 32.2656 |
| 0 | C | 16,788 | 30,252 | 55.4939 |
| 0 | D | 15,160 | 30,252 | 50.1124 |
| 1 | A | 3,534 | 31,099 | 11.3637 |
| 1 | B | 10,030 | 31,099 | 32.2518 |
| 1 | C | 17,182 | 31,099 | 55.2494 |
| 1 | D | 15,400 | 31,099 | 49.5193 |
| 2 | A | 3,422 | 32,187 | 10.6316 |
| 2 | B | 9,782 | 32,187 | 30.3912 |
| 2 | C | 17,490 | 32,187 | 54.3387 |
| 2 | D | 15,612 | 32,187 | 48.5041 |

| Condition | recognition_repeated mean ± SD (%) | recognition_previously_committed mean ± SD (%) | recognition_qualifying_reunion mean ± SD (%) | n |
| --- | --- | --- | --- | --- |
| A | 95.554 ± 0.332 | 95.554 ± 0.332 | 11.193 ± 0.498 | 3 |
| B | 66.501 ± 0.193 | 94.988 ± 0.330 | 31.636 ± 1.078 | 3 |
| C | 50.399 ± 0.190 | 50.399 ± 0.190 | 55.027 ± 0.609 | 3 |
| D | 53.964 ± 0.131 | 77.094 ± 0.487 | 49.379 ± 0.813 | 3 |

Decay retains more long-gap familiar contacts: **C > D > B > A** for the
shared qualifying-reunion rate in every seed. Thus the overall-recognition loss
coexists with improved long-gap recognition. Reunion exposures are much smaller
than all eligible-contact exposures; these are distinct targets, not contradictory
measurements. Previously-committed denominators are policy-specific and must not
be substituted for the common primary denominator. Raw repeated/committed/reunion
numerators and denominators are retained in run/window/age CSVs.

### Formation, failure causes and retention

| Seed | Condition | Candidate creates | Commits | Resets | Candidate evictions | Unique-pair ever-committed fraction | Mean candidate occupancy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | B | 523,412 | 214,428 | 294,731 | 1,133 | 0.6474 | 5.0399 |
| 0 | D | 665,201 | 285,929 | 358,124 | 1,923 | 0.6473 | 6.5293 |
| 1 | B | 524,657 | 216,131 | 293,335 | 2,420 | 0.6510 | 5.0773 |
| 1 | D | 670,633 | 290,618 | 356,979 | 4,250 | 0.6508 | 6.6185 |
| 2 | B | 519,618 | 215,111 | 289,982 | 1,955 | 0.6542 | 5.0586 |
| 2 | D | 666,167 | 289,961 | 354,172 | 3,395 | 0.6541 | 6.6147 |

| Metric | A mean ± SD | B mean ± SD | C mean ± SD | D mean ± SD |
| --- | --- | --- | --- | --- |
| cumulative_ever_committed_fraction | 1.0000 ± 0.0000 | 0.6508 ± 0.0034 | 1.0000 ± 0.0000 | 0.6507 ± 0.0034 |
| cumulative_candidate_conversion | NA ± NA | 0.4119 ± 0.0022 | NA ± NA | 0.4328 ± 0.0028 |
| fraction_observations_full | 0.7957 ± 0.0023 | 0.6171 ± 0.0040 | 0.7957 ± 0.0023 | 0.6171 ± 0.0040 |
| fraction_agents_full | 0.7883 ± 0.0023 | 0.6103 ± 0.0040 | 0.7883 ± 0.0023 | 0.6103 ± 0.0040 |
| mean_strength_at_eviction | 8.6850 ± 0.2355 | 17.2958 ± 0.0807 | 1.2217 ± 0.0010 | 8.6952 ± 0.0300 |
| mean_elapsed_at_eviction | 7.5975 ± 0.2996 | 20.6775 ± 0.3721 | 1.1548 ± 0.0154 | 13.0615 ± 0.2499 |
| mean_retention_at_eviction | 17.4319 ± 0.6203 | 38.3502 ± 0.4340 | 1.4958 ± 0.0177 | 18.7985 ± 0.2953 |
| evictions_per_allocation | 0.8421 ± 0.0053 | 0.6542 ± 0.0043 | 0.9675 ± 0.0006 | 0.7423 ± 0.0047 |

| Seed | Condition | Eligible misses: previously committed/forgotten | Eligible misses: never committed | Relearned pairs | Repeatedly forgotten pairs |
| --- | --- | --- | --- | --- | --- |
| 0 | A | 107,720 | 0 | 100,563 | 83,017 |
| 0 | B | 133,886 | 1,003 | 14,953 | 8,897 |
| 0 | C | 1,317,951 | 0 | 191,259 | 184,753 |
| 0 | D | 645,909 | 1,324 | 46,738 | 39,044 |
| 1 | A | 132,247 | 0 | 103,709 | 86,875 |
| 1 | B | 153,515 | 2,305 | 16,897 | 10,507 |
| 1 | C | 1,348,638 | 0 | 189,455 | 183,030 |
| 1 | D | 682,519 | 3,017 | 47,896 | 40,181 |
| 2 | A | 124,600 | 0 | 102,876 | 85,224 |
| 2 | B | 153,396 | 1,589 | 16,957 | 10,434 |
| 2 | C | 1,365,872 | 0 | 188,287 | 181,867 |
| 2 | D | 685,001 | 2,065 | 48,153 | 40,331 |

Selective formation encodes about 65% of unique observed observer-target
pairs, versus all pairs under immediate encoding. Candidate conversion counts
commitments divided by created candidate attempts; pending attempts at end remain
in its denominator, so this is not an uncensored success probability. Overflow
can prevent commitment even for historically eligible contacts, but most primary
misses here concern identities previously committed and evicted, rather than
never-committed candidates.

B reduces allocations/evictions relative to A, yet slightly lowers primary
recognition: relearning requires another five consecutive sightings. D improves
recognition relative to C while reducing churn, but remains below A/B overall.
C evicts entries with mean strength near 1.22 sightings and short residence,
consistent with repeatedly discarding newly encoded low-strength contacts while
retaining experienced residents. Its improved reunion rate is consistent with
that retention tradeoff. These observed patterns do not establish a universally
optimal tau or a validated cognitive explanation. Fewer evictions alone are not
a quality measure. Strength/residence distributions are conditional on eviction;
death/end-censored retained entries are separate in distributions.csv.

### Time windows and demography

| Window | A recognition mean ± SD (%) | B | C | D |
| --- | --- | --- | --- | --- |
| (0,100] | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 | 100.000 ± 0.000 |
| (100,200] | 99.670 ± 0.109 | 99.744 ± 0.123 | 94.094 ± 2.169 | 99.222 ± 0.506 |
| (200,300] | 96.954 ± 0.519 | 96.731 ± 0.715 | 62.981 ± 1.990 | 85.371 ± 1.919 |
| (300,400] | 94.680 ± 0.575 | 93.451 ± 0.592 | 45.965 ± 0.197 | 70.501 ± 0.881 |
| (400,500] | 95.699 ± 0.302 | 94.437 ± 0.025 | 49.386 ± 0.175 | 74.350 ± 0.158 |

| Seed | Final population | Births | Deaths | Actor opportunities | Directed contacts | Unique observer-target pairs |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 832 | 2373 | 1641 | 295,455 | 4,461,785 | 306,320 |
| 1 | 823 | 2342 | 1619 | 294,189 | 4,500,831 | 303,263 |
| 2 | 799 | 2314 | 1615 | 292,898 | 4,502,328 | 300,183 |

These demographic/exposure values are shared across all four algorithms.
The population invariant is 100 + births − deaths = final population. The first
100-tick window saturates at 100% eligible recognition and cannot separate the
mechanisms. Later crowding/contact histories put memory under pressure. Age CSVs
separate younger observers with little formation exposure from older observers;
their observation metrics are not age-specific birth/death incidence estimates.
No interpolation or artificial post-extinction observations are added.

## Short capacity and decay sensitivity

All following runs use the **100-tick** histories, common denominators
27,356 / 30,828 / 29,827 for seeds 0/1/2, candidate budget 32 and tau=100 unless
varied explicitly. They do not establish effects at 500 ticks.

| Condition | K | Seed 0 (%) | Seed 1 (%) | Seed 2 (%) | Equal-seed mean ± SD (%) | n |
| --- | --- | --- | --- | --- | --- | --- |
| A | 8 | 97.0829 | 96.2664 | 94.8201 | 96.0565 ± 1.1459 | 3 |
| A | 16 | 99.8063 | 99.8540 | 99.7083 | 99.7895 ± 0.0743 | 3 |
| A | 32 | 100.0000 | 100.0000 | 100.0000 | 100.0000 ± 0.0000 | 3 |
| A | 64 | 100.0000 | 100.0000 | 100.0000 | 100.0000 ± 0.0000 | 3 |
| A | 128 | 100.0000 | 100.0000 | 100.0000 | 100.0000 ± 0.0000 | 3 |
| D | 8 | 91.5631 | 91.1509 | 89.5665 | 90.7602 ± 1.0541 | 3 |
| D | 16 | 99.6929 | 99.7762 | 99.6211 | 99.6968 ± 0.0776 | 3 |
| D | 32 | 100.0000 | 100.0000 | 100.0000 | 100.0000 ± 0.0000 | 3 |
| D | 64 | 100.0000 | 100.0000 | 100.0000 | 100.0000 ± 0.0000 | 3 |
| D | 128 | 100.0000 | 100.0000 | 100.0000 | 100.0000 ± 0.0000 | 3 |

Within these short histories, larger tested K alleviates eligible
recognition losses at K=8/16. At K≥32, both A/D saturate at 1.0; additional
capacity benefit is unmeasurable on this metric. A exceeds D at K=8/16 in all
three seeds. The combined policy is not superior here despite its lower churn.
Candidate budget is held fixed, so changing K does not silently add candidate
storage; A/D still differ in total available working identity storage.

| D tau | Seed 0 (%) | Seed 1 (%) | Seed 2 (%) | Mean ± SD (%) | LTM evictions across seeds | n |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 100.000 | 100.000 | 100.000 | 100.000 ± 0.000 | 0/0/0 | 3 |
| 50 | 100.000 | 100.000 | 100.000 | 100.000 ± 0.000 | 0/0/0 | 3 |
| 100 | 100.000 | 100.000 | 100.000 | 100.000 ± 0.000 | 0/0/0 | 3 |
| 200 | 100.000 | 100.000 | 100.000 | 100.000 ± 0.000 | 0/0/0 | 3 |

All four tau values are indistinguishable for D in this pilot: no
long-term evictions occur at K=32, so the eviction decision has no opportunity
to separate them. This is a lack of informative pressure, not robustness across
decay parameters. A 500-tick tau/capacity grid is supported but deliberately
deferred after the measured primary runtime; parameters were not retuned.

## Runtime, memory, storage and reproducibility

| Phase | Ticks | Policies per seed | Wall seconds | Replay seconds, seeds 0/1/2 | Process peak RSS MiB | Files | Current output MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pilot | 100 | 4 | 8.952 | 1.686/1.813/1.839 | 137.66 | 20 | 1.040 |
| primary | 500 | 4 | 727.365 | 175.172/180.952/180.089 | 252.94 | 20 | 3.014 |
| capacity | 100 | 10 | 13.933 | 4.003/4.375/4.345 | 149.81 | 20 | 1.914 |
| decay | 100 | 5 | 7.725 | 2.307/2.335/2.249 | 145.34 | 20 | 1.143 |

| Unrecorded online condition | Seed 0 seconds | Seed 1 | Seed 2 | Mean ± sample SD seconds | n |
| --- | --- | --- | --- | --- | --- |
| A | 0.6616 | 0.6928 | 0.6688 | 0.6744 ± 0.0164 | 3 |
| B | 0.7330 | 0.7410 | 0.7463 | 0.7401 ± 0.0067 | 3 |
| C | 0.7228 | 0.7347 | 0.7291 | 0.7289 ± 0.0059 | 3 |
| D | 0.7160 | 0.7496 | 0.7324 | 0.7327 ± 0.0168 | 3 |

Online benchmark is 100 ticks with recording OFF; simulation-loop timing
excludes initialization and full-state hashing. All four policies share identical
final ecological hashes for each seed and match the untouched legacy source.
These short timings are not a dense 500-tick speed estimate. Decay victim search
is O(K) per allocation; private storage is O(K + candidate_capacity).

The primary's new legacy-history generation takes 53.56/53.83 seconds for seeds
0/2, including original per-contact observation construction and compact recording;
validating/converting seed 1's existing prefix takes 82.44 seconds. Four-policy
replay plus aggregation takes about 175–181 seconds per seed. That is substantially
slower than the tiny pilot and is why no long sensitivity grid was launched.
Replay timing includes validation, exact-memory hashing and incremental
scientific metrics, not just the raw memory update kernel. Plot/import/allocation
costs are included in process RSS. RSS is a process high-water mark, not an
isolated per-policy memory allocation. Timings are local observations, not
portable throughput guarantees or controlled microbenchmarks.

The first pilot took 8.95 seconds, ~0.91 MB analysis output and ~0.55 MB compact
histories. Its runtime/storage review is saved before the bounded primary.
Display-only caption/legend refreshes were performed later, preserving every
CSV byte; current sizes above include those refreshed plots. Separate receipts
retain both original phase timing and refresh timing.

| Compact history | Saved rows | Original stream records folded | MB | Provenance |
| --- | --- | --- | --- | --- |
| seed_0_100.csv.gz | 14,083 | 57,824 | 0.177 | validated_legacy_full |
| seed_0_500.csv.gz | 300,071 | 5,160,282 | 9.954 | online_unchanged_legacy |
| seed_1_100.csv.gz | 14,414 | 61,998 | 0.185 | validated_legacy_full |
| seed_1_500.csv.gz | 298,752 | 5,225,234 | 9.990 | validated_legacy_prefix |
| seed_2_100.csv.gz | 14,247 | 61,827 | 0.183 | online_unchanged_legacy |
| seed_2_500.csv.gz | 297,429 | 5,212,054 | 9.913 | online_unchanged_legacy |

All artifacts are saved locally. Each phase has 9 CSVs, 10 headless
180-dpi PNGs and analysis_metadata.json (20 files; 80 across four phases).
No full per-contact rows or Q histories are duplicated by policy. Compact sources
are about 30.40 MB total (100- and 500-tick files together). Output metadata bind
input SHA-256, original/effective configs, seed/horizon, policy and resolved
candidate bounds, software, schemas, source hashes and output hashes. Performance
receipts are separate so wall times cannot spoil scientific-output reproducibility.
The independent real pilot replay checks all 20 outputs byte-for-byte, including
PNG and JSON. The full suite also checks input-order independence and RNG isolation.

## Saved outputs and file map

* `agents/social_memory_policy.py`: bounded formation, immutable entries,
  deterministic score comparison, original-class compatibility factory.
* `config.py`, `main.py`, `simulation/simulation.py`: companion policy config,
  opt-in CLI and minimal body/observer factory wiring.
* `simulation/social_policy_observer.py`: separate optional summary/trace and
  completion sidecars; legacy social format is unchanged.
* `experiments/social_history.py`: complete compact histories and validated
  conversion, explicit prefix censoring and ordered streaming reader.
* `experiments/social_policy_metrics.py`: common historical eligibility,
  formation/retention/recognition counts, distributions, window/age exposure.
* `experiments/analyze_social_memory_policies.py`, `social_policy_plots.py`:
  equal-seed summaries, paired comparisons, deterministic discrete plots.
* `experiments/social_memory_experiments.py`: staged bounded runner and grids.
* `experiments/benchmark_social_memory_policies.py`: online timing and exact
  ecological/private-state replay comparison.
* `tests/test_social_memory_policy.py`, `test_social_policy_replay.py`: 46 tests.
* README, this report and .gitignore: usage, results and versioned outputs.

Each phase directory under `analysis/social_memory_v071/` contains:

* `run_summary.csv`, `window_summary.csv`, `age_summary.csv`, `distributions.csv`;
* `seed_summary.csv`, `window_seed_summary.csv`, `age_seed_summary.csv`;
* `paired_by_seed.csv`, `paired_seed_summary.csv`, `analysis_metadata.json`;
* `memory_occupancy.png`, `memory_at_capacity.png`, `long_term_evictions.png`,
  `eligible_recognition.png`, `candidate_formation.png`,
  `recognition_failure_causes.png`, `capacity_sensitivity.png`,
  `recognition_by_age.png`, `strength_at_eviction.png`, `paired_recognition.png`.

Key figures: [primary eligible recognition](../analysis/social_memory_v071/primary/eligible_recognition.png),
[paired effects](../analysis/social_memory_v071/primary/paired_recognition.png),
[failure causes](../analysis/social_memory_v071/primary/recognition_failure_causes.png),
[capacity](../analysis/social_memory_v071/capacity/capacity_sensitivity.png).

New raw histories, experiment designs, phase runtime receipts and online benchmark
are under `data/social_memory_v071/`. Baseline snapshots, archived controls, real
replay receipt, full-test logs and final verification are `data/v071_*`.

## Limitations and next step

Only three seeds and one default evolving ecology are evaluated; no significance
claim is made. Encounter-type weighting changes the answer: overall eligible
recognition favors A here, while qualifying reunions favor experienced retention.
The short tau sweep cannot discriminate policies without evictions. Candidate
storage adds an extra finite budget, and deterministic batch admission/ID ties
remain explicit modeling choices. Tau and the score are hypotheses, not biological
constants. Floating-log evaluation is deterministic within the recorded software
environment; cross-platform bit identity is not promised.

Offline histories are privileged and can grow with live observer-target exposure;
aggregate identifier sets/histograms also grow with analysis scope. They are not
agent knowledge. Normal bodies retain only bounded private memory and do not
remember complete contact/death histories. Streaming removes contact-row RAM
retention but does not make millions of contacts computationally free.

Keep the unchanged immediate LRU as the reference. Before changing cognitive or
social incentives, predeclare whether the target is overall recognition or
long-gap reunion recognition, then consider matched total-storage controls and
additional seeds/horizons only if that question warrants the measured cost.
If a longer tau/capacity study is needed, first profile/streamline evaluator
aggregation without changing its tested semantics. No need to add social rewards,
voices, learned identity or v0.7.2 mechanisms to interpret this release.

**Conclusion:** the tested selective/decay mechanisms do not improve the primary
overall recognition metric over original LRU on these matched 500-tick histories.
They change which identities are retained, and improve the separately measured
long-gap reunion rate. This is a controlled memory tradeoff, not evidence of
behavioral adaptation, cooperation, communication or language.
