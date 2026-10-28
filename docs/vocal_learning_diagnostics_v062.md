# Vocal Q learning diagnostics v0.6.2

This release measures the existing learner. It does not introduce a new
communication mechanism, meaning, reward, speaking cost, capacity limit, or
ecological rule. Nonuniform values or state-dependent choices are not evidence
of communication, language, shared semantics or cooperation.

## Architecture and isolation

`VocalLearningObserver` is opt-in through `Simulation(vocal_learning_observer=...)`
or `main.py --vocal-learning-csv PATH`. The learner has a nullable observer
reference and conditional hooks on its existing operations. OFF allocates no
observer/history/diagnostic records. The physical head, reward expression,
vocal update arithmetic, epsilon-greedy draws, memory, capacities, actor ordering,
physiology, reproduction, mutation and regeneration are unchanged. No Config
fields or defaults change. The 27-column communication recorder is unchanged.

The observer binds only scalar agent identity/age/generation at the start of a
turn; it does not retain a Human, Brain, Observation or Q table. It records:

1. The actual key's presence **before** the previous Vocal Q transition update.
2. Actual row allocations/evictions, including during that update.
3. Presence immediately before selecting/allocating the current row.
4. Actual choice-time Q diagnostics after updating the previous transition.
5. Each actual subsequent TD update, including immediate terminal updates.

Choice diagnostics are calculated after the random selection while the same
row is unchanged, before action execution or outcome feedback. They never
select an action. Reads do not call `_remember*`, move LRU entries, obtain
another observation, recompute reward, or call any RNG. TD targets and rewards
are captured from the existing update path; the original update expression
remains intact. Terminal/nonterminal distinction changes logging only.

The optional observer interns exact tuple keys into encounter-order integers,
starting at zero. IDs do not derive from Python's randomized `hash()` and are
not Brain inputs. A run key dictionary and per-living-individual per-key scalar
histories cost O(run unique keys + living individual/key pairs). These histories
can grow beyond cognitive capacity; they are privileged diagnostic memory,
never extra agent memory. Historical per-individual counts are streamed and
removed on death/end. There is no sampling or full Q-table serialization.

## Exact recording schema (version 1)

`PATH` is the decision CSV. The three companion CSVs use the same basename and
`.updates`, `.rows`, `.states` suffixes. `.csv.gz` uses streaming gzip level 1,
empty embedded filename and mtime=0. Plain `.csv` is supported too. CSV uses
empty fields for not-applicable/undefined measurements, and JSON uses null.
A/B and RandomBrain have one identity-only decision record per actor with all
Vocal Q fields blank. They do not have zero-valued Vocal Q tables.

**decisions (35 columns)**

```text
tick,agent_id,age,generation,condition,state_id,rows,present_at_start,present_before_selection,newly_allocated,novelty,decision_allocations,decision_evictions,allocations,evictions,updates,unique_states,state_visits,q_max,q_min,q_mean,q_std,q_gap,max_ties,all_equal,silence_q,silence_rank,silence_is_max,selected_q,selected_action,exploratory,n_actions,epsilon,p_silence,at_capacity
```

**updates (19 columns)**

```text
tick,agent_id,age,state_id,action,reward,next_value,target,old_q,new_q,td_error,abs_td_error,q_change,visits,updates,row_updates,epoch,recreated,terminal
```

**rows (8 columns)**

```text
event,tick,agent_id,state_id,visits,updates,epoch,rows
```

**states (2 columns)**

```text
state_id,key_json
```


Decision definitions:

- `rows` is stored Vocal Q occupancy after remembering the current row.
  `present_at_start` precedes the previous transition update;
  `present_before_selection` follows it. `newly_allocated` refers to the current
  selection, while `decision_allocations/evictions` count actual events from
  turn start through selection. Terminal updates occur afterward and remain in
  update/row streams. `allocations`, `evictions`, `updates` are lifetime counts
  through selection, not run-level population totals.
- `novelty` is per individual: `first` means never chosen before; `stored` means
  previously chosen and present at turn start; `recreated` means previously
  chosen but absent at turn start. These categories partition decisions. They
  are deliberately separate from post-update presence and allocation flags.
- `unique_states` counts lifetime chosen keys per individual; `state_visits`
  counts current key's lifetime decisions, including this choice.
- Q SD is population SD over available actions (`ddof=0`). Top-two gap counts
  tied maxima separately, hence is zero with multiple maxima. Ties use exact
  `==`, matching selection. SILENCE rank is competition rank: one plus the
  number of actions strictly above SILENCE. `all_equal` compares extrema.
- Vocal action indices match the existing head: 0 = SILENCE, i+1 = signal i.
  Integer signal IDs have no meaning or numerical similarity. `exploratory`
  records the branch taken, even if exploration selects a maximum.
- With N available actions, M exact maxima and effective epsilon:
  `p_silence = epsilon/N + (1-epsilon)*silence_is_max/M`.
  Default uniform = 1/17; unique SILENCE max = 0.811764706;
  SILENCE not max = 0.011764706. This is a policy probability, separate from
  observed SILENCE frequency. Nonuniform values need not give a unique maximum.

Update definitions: `td_error = target-old_q` is measured before the actual
update; `abs_td_error` and `q_change = abs(new_q-old_q)` retain magnitude.
`target` is the actual reward + discount × actual bootstrap maximum. `visits`
and `updates` count lifetime decisions/updates of that individual's key;
`row_updates` resets on allocation, and `epoch` counts allocations of that key.
`recreated` means epoch > 1, including a row recreated on an earlier choice;
it does not assert that this particular update caused reallocation.
`terminal` means the existing death update with bootstrap zero.

Row definitions: `allocate` and `evict` are hooks on actual table operations,
not a second inferred LRU. Eviction visits/updates describe the departing row's
current epoch, allowing learned-row eviction exposure to be counted.
An allocation's visits/updates are pre-allocation lifetime counts.
`death` and `run_end` have blank state/epoch, retained-row occupancy and lifetime
individual visits/updates. They never increment eviction counts. A row may
be lost with an agent without ever being evicted.

The completion sidecar is written after streams close. It contains full Config,
seed/config SHA-256, final tick/statistics, exact schemas, all file sizes/hashes/
counts, source hashes, unique key count and peak living agent/key history count.
If simultaneous communication recording is enabled it also certifies that
complete recording and its metadata. Ordinary statistics and communication
sidecars keep their previous v0.6.1 version/schema for compatibility.

## Offline methods and individual comparisons

```bash
python3 -m experiments.analyze_vocal_learning \
  --data-dir data/vocal_learning --output-dir analysis/vocal_learning \
  --window-size 100 --communication-analysis analysis/communication \
  --communication-verification data/v061_verification.json
```

The analyzer validates completed sidecars, config hashes, file hashes, schemas,
record counts, identity/order, key catalogs and diagnostic payloads. Duplicate
condition/seed recordings are rejected. Different ecologies are separated;
filenames/discovery order do not establish identity. Raw files stream in chunks;
scalar individual/key/window summaries are retained. This is not constant-memory
analysis: summary storage grows with observed individual/key pairs, although
raw decisions and Q tables are never retained.

Window rates use actual exposures in `(k*100,(k+1)*100]`; empty rates remain
undefined. Nominal/actual endpoints, observed ticks and partial/complete flags
remain explicit. No interpolation, missing-value filling or carry-forward after
extinction. Per-run results precede equal-seed mean/sample SD/min/max/finite n.
Population windows/age bins weight that run's decisions, never cross-seed agent
counts. Age plots have per-bin agent counts and decision/update exposures.
Repeated agent observations are not independent experimental replicates.

Individual eligibility is at least 50 Vocal Q decisions. Every individual still
has a summary row. Same-ID first/last qualifying windows require 20 decisions
per window; separate fixed early/next and early/final tables preserve matched
IDs, exposures and within-agent differences. Population/age profiles cannot
identify an age effect independently of selective survival and cohort turnover.
The individual figure selects the lowest seed per condition, then the four
largest eligible decision exposures with ID tie-breaking, without inspecting
the plotted pattern. The selection is saved in metadata.

The compact state-learning table retains every updated individual/key pair,
update counts, positive/negative/zero TD counts, mean TD/magnitude/Q-change and
whether both positive and negative updates occurred. This diagnoses inconsistent
signs descriptively; changing bootstrap values and reward context are possible
contributors. No explanation is forced into an exclusive category.

### Safe reuse of v0.6.1 analysis

The pilot simultaneously records both observers. Before reusing the archive it
checks the full Config, final statistics/tick, **every communication CSV byte**
and **every ordinary statistics CSV byte** against the corresponding old run.
Only a verified redundant new communication copy is removed; diagnostics then
reference the unchanged original communication file/sidecar. The analyzer
revalidates these hashes against the archived analysis's exact input identities,
full configs and final states. An optional historical verification JSON also
checks the reused MI output/metadata hashes. Matching seed alone is insufficient.
No 100-permutation analysis is rerun.

Window comparisons join exact start/actual-end points and retain all five
within-agent corrected V/state MI features. Individual comparisons stream the
existing per-agent MI table, join exact run/ID/window endpoints and retain food/
water directions, threshold and MI sample count. Age/MI summaries bin the
individual's **mean age in that window**, not every decision's age; they average
eligible agent-window estimates within seed, then equally across seeds.
Different windows of one agent are not new independent replicates. These
conditional descriptive estimates still do not control ecology, autocorrelation,
bootstrap dynamics or selective survival, and are not causal tests.

## Verification and pilot results

- Before changes: **282 tests, OK, 58.472 s**.
- After changes: **317 tests, OK, 54.178 s** (35 new tests; Python 3.11.9).
- All six historical fixtures and the v0.6.1 report retain their exact hashes.
  Historical v0.6 seed-7 tick-80 trajectory/final-state/full-event hashes match
  with diagnostics ON. The four learning ablations, RandomBrain and both-observer
  controls compare every tick, including both Q/LRU/pending heads, auditory
  memory, all Brain/simulation RNGs, actions, bodies, genomes, births/deaths,
  resources, full statistics and events.
- Controlled tests cover all-tied, unique-signal, unique-SILENCE and multiple-max
  probabilities; exact equality; actual eviction/recreation; one repeated state;
  identical body/visual versus differing auditory keys; and actual TD updates.
- Empty/extinct/missing/partial data, duplicate decisions/seeds, invalid hashes,
  N/A controls, equal-seed aggregation and unsafe joins are tested. Small complete
  recordings and all analyzer CSV/JSON/PNGs repeat byte-identically, leaving
  inputs and Python/NumPy global RNG unchanged. All **12 full-pilot PNGs** also
  match across two analyses. Simulator diagnostics work with `python -S`, without
  analysis dependencies. No historical fixture was regenerated.

The six requested runs used exactly the previous evolution preset, seeds 0/1/2,
1000 ticks, C/D only. World 40×30, initial population 100, free founder perception
2, hearing 8, vocabulary 16, alpha 0.2, gamma 0.9, epsilon 0.2, both capacities
256, and existing mutation/regeneration/reproduction remain unchanged. All runs
reached tick 1000. Every complete simultaneous communication CSV and ordinary
statistics CSV matches its v0.6.1 archive byte-for-byte. No ecology was retuned.

### Per-seed whole-run measurements

Rates use each run's recorded decisions. They include actors that die and
exclude newborns on birth ticks; occupancy is measured at selection, not only
among final survivors. C listening OFF; D listening ON; both Vocal Q ON.

| Condition | Seed | Decisions | Hit % | Evictions / 1000 decisions | Mean rows | Mean Q gap | Implied SILENCE % | Observed SILENCE % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | 0 | 694509 | 48.9467 | 0.56731 | 45.3982 | 0.0015342 | 5.88360 | 5.86558 |
| C | 1 | 688515 | 49.2239 | 0.36601 | 44.4496 | 0.0015486 | 5.87759 | 5.85623 |
| C | 2 | 679214 | 49.2578 | 0.45494 | 45.6713 | 0.0016061 | 5.87085 | 5.85927 |
| D | 0 | 714512 | 29.2912 | 4.85506 | 60.9088 | 0.0010792 | 5.87310 | 5.88989 |
| D | 1 | 719294 | 29.2619 | 5.86964 | 61.6571 | 0.0010816 | 5.87262 | 5.87534 |
| D | 2 | 713626 | 29.3790 | 4.71536 | 61.1681 | 0.0011183 | 5.86931 | 5.87352 |

### Equal-seed summaries

Mean ± sample SD; n=3 for every entry. No cross-seed pooling.

| Metric | C | D |
| --- | --- | --- |
| State hit % | 49.1428 ± 0.170699 | 29.3107 ± 0.0609306 |
| Actual evictions / 1000 decisions | 0.46275 ± 0.100878 | 5.14669 ± 0.629984 |
| Mean occupancy | 45.173 ± 0.641209 | 61.2447 ± 0.379971 |
| At capacity % | 0.114924 ± 0.0256351 | 0.790345 ± 0.0965218 |
| Unique run keys | 2490 ± 31.1929 | 29965.7 ± 205.291 |
| Nonuniform Q rows % of decisions | 49.1425 ± 0.170545 | 29.3105 ± 0.060897 |
| Unique maximum % of decisions | 3.88304 ± 0.187459 | 2.65384 ± 0.0571778 |
| Mean top-two Q gap | 0.00156296 ± 3.80399e-05 | 0.00109302 ± 2.1909e-05 |
| Mean policy TV from uniform | 0.0705213 ± 0.00168238 | 0.0417934 ± 0.000525367 |
| Mean maximum tie count (of 17) | 15.5014 ± 0.0357506 | 16.1119 ± 0.0111641 |
| Negative-only nonuniform rows % | 45.2801 ± 0.0829957 | 26.6621 ± 0.0188227 |
| Implied SILENCE % | 5.87735 ± 0.00637774 | 5.87168 ± 0.0020666 |
| Observed SILENCE % | 5.86036 ± 0.00477193 | 5.87959 ± 0.00897178 |
| Unique SILENCE max % | 0.221122 ± 0.00738185 | 0.1439 ± 0.00259472 |
| Mean absolute TD / update | 0.0417537 ± 3.03585e-05 | 0.0418952 ± 6.81074e-05 |
| Post-eviction key revisits % | 0.00125646 ± 0.000786254 | 0.00748788 ± 0.00347233 |

Additional purely diagnostic definitions: policy total variation from uniform
is `(1-epsilon)*(1-M/N)`, not a policy change or a semantic score.
`negative_only_fraction` counts nonuniform rows with no positive action value;
`positive_max_fraction` counts rows with a positive maximum. These descriptive
categories are not forced explanations. Other retained statistics include mean
maximum/minimum/selected Q, mean tie count and mean action-wise Q SD.

Nonuniform Q values do occur, but most such rows retain multiple untried/zero
maxima after negative updates. C has about 45.28% negative-only nonuniform rows
out of 49.14% nonuniform decisions; D about 26.66% out of 29.31%. The mean policy
TV is about 0.0705 (C) and 0.0418 (D): policies are not all uniform, while their
label-averaged SILENCE probability remains near 1/17. Unique maxima are uncommon,
not absent. These are numerical preferences under shared body reward, not
learned meanings or proof that vocal acts improve reward.

### Retention and TD evidence

| Run | Actual evicted rows | Updates on evicted rows | Rows lost with death | Rows retained at end | First eviction tick |
| --- | --- | --- | --- | --- | --- |
| C_seed_0 | 394 | 717 | 314825 | 39351 | 772 |
| C_seed_1 | 252 | 462 | 309227 | 40122 | 631 |
| C_seed_2 | 309 | 625 | 301927 | 42412 | 661 |
| D_seed_0 | 3469 | 4768 | 446776 | 54978 | 435 |
| D_seed_1 | 4222 | 5722 | 450069 | 54524 | 400 |
| D_seed_2 | 3365 | 4593 | 443298 | 57307 | 449 |

For each run, allocations − actual evictions equals rows discarded on death plus
rows retained at run end. No death/end event is counted as an eviction. The
number of learned updates associated with evicted rows is a retention proxy,
**not information measured in bits**. It is 462–717 (C) and 4593–5722 (D), out
of roughly 0.68–0.72 million updates/run. Memory discarded with deaths is much
larger by row count. Only 5–15 (C) and 30–80 (D) decisions/run recreate an
already-chosen evicted key; rare recreation does not prove evictions harmless.

| Run | Updated individual/key pairs | Only one update | Multiple updates | Both TD signs | Mixed signs / repeated % |
| --- | --- | --- | --- | --- | --- |
| C_seed_0 | 354128 | 184499 | 169629 | 21053 | 12.411 |
| C_seed_1 | 349188 | 182036 | 167152 | 20808 | 12.449 |
| C_seed_2 | 344202 | 179485 | 164717 | 20673 | 12.551 |
| D_seed_0 | 504538 | 388246 | 116292 | 14525 | 12.490 |
| D_seed_1 | 508138 | 390724 | 117414 | 14604 | 12.438 |
| D_seed_2 | 503285 | 386875 | 116410 | 14607 | 12.548 |

Most updated keys get very few updates, especially D. About 12.4–12.6% of keys
updated more than once receive both positive and negative TD errors. This can
reflect changing rewards/bootstrap values; it is not a direct measure of noisy
communication. Mean absolute TD remains near 0.042, so the learner is not simply
receiving zero updates. Over 92% of updates are negative in these runs; zero TD
count is zero. Recreated-row flags are recorded separately in the update stream.

### Time, age, individual turnover and previous MI

Seed means for (0,100] → (900,1000]: C hit rate 71.80% → 47.75%, mean Q gap
0.007268 → 0.001060; D hit 57.93% → 27.57%, gap 0.005918 → 0.000667.
Identity-conditioned, permutation-corrected V/food MI falls from 0.132478 to
−0.022982 bits (C), and 0.095352 to −0.010216 (D). The reused corrections may
be negative; this does not mean negative empirical MI.

By (200,300], corrected food MI is already near zero (C 0.004441, D 0.005900),
before any Vocal Q eviction in any of the six runs (first eviction ≥400).
**LRU eviction cannot account for the initial MI collapse in these runs.**
Later retention effects remain possible. Increasing state novelty, different
experience and turnover accompany the population-level decline; their causal
contributions have not been separated.

Older-age profiles generally have greater hit rates and Q gaps through moderately
supported ages, but mix survivors/cohorts/ecology. At age 800–824, D has only one
agent per seed and 1/25/25 decisions: those tail estimates are sparse, not strong
age evidence. Same-ID comparisons are therefore reported separately:

| Run | Comparison | Paired agents (≥20 decisions/window) | Mean within-ID Q-gap change |
| --- | --- | --- | --- |
| C_seed_0 | early_next | 173 | 0.0029248 |
| C_seed_0 | early_final | 0 | undefined |
| C_seed_1 | early_next | 160 | 0.0028771 |
| C_seed_1 | early_final | 0 | undefined |
| C_seed_2 | early_next | 165 | 0.0033372 |
| C_seed_2 | early_final | 4 | 0.0040211 |
| D_seed_0 | early_next | 168 | 0.0036559 |
| D_seed_0 | early_final | 0 | undefined |
| D_seed_1 | early_next | 167 | 0.0023286 |
| D_seed_1 | early_final | 0 | undefined |
| D_seed_2 | early_next | 170 | 0.0028659 |
| D_seed_2 | early_final | 1 | -0.0119236 |

The fixed early/next comparisons increase mean within-ID Q gap in all six runs,
while the population mean starts falling. Using the archived MI threshold of
50 decisions/window, the same-ID early/next pairs number 120–127 (C) and 121–130
(D). Their mean food-MI changes are **positive in all six runs**: +0.0439 to
+0.0905 bits (C), +0.0424 to +0.0518 (D); water changes are positive too.
Thus an early population decline must not be equated with immediate within-ID
loss. These are descriptive matched survivors, not randomized age effects.

Early/final overlap is 0/0/4 agents in C and 0/0/1 in D at both relevant
thresholds. The four C agents give food-MI change +0.0270 and water −0.1023 bits;
the single D agent gives −0.2577 and −0.0283. These extremely sparse pairs do not
resolve long-term within-individual loss. Almost all early individuals are absent
from the final eligible-window sample (including dying actors, not a final-survivor
sample); turnover is an important observed difference,
not a quantitatively isolated causal explanation of the MI change.

### Direct early-cohort survival check

| Run | Actors in (0,100] | Dead by tick 1000 | Alive at tick 1000 | Seen in final window |
| --- | --- | --- | --- | --- |
| C_seed_0 | 208 | 208 | 0 | 0 |
| C_seed_1 | 194 | 194 | 0 | 0 |
| C_seed_2 | 190 | 189 | 1 | 4 |
| D_seed_0 | 208 | 208 | 0 | 0 |
| D_seed_1 | 206 | 206 | 0 | 0 |
| D_seed_2 | 207 | 207 | 0 | 1 |

These counts use every early actor, without MI eligibility filtering. Death and
run-end records account for all of them. Only one early actor survives to the
end across the six runs, while more may act and die in the final window. This
confirms extensive replacement, but does not isolate its contribution to MI.

### Figures and files

`analysis/vocal_learning/` contains 20 CSVs, 12 PNGs and analysis_metadata.json
(33 files). CSVs retain windows/runs, individuals, individual-window and age
profiles, first/last and fixed-window pairs, state-learning counts, equal-seed
summaries, safe window/individual MI joins, age/MI profiles and paired MI changes.
The metadata binds input/source hashes, full configs, exact sample thresholds,
selection, software versions, archived verification and reused MI output hashes.

The ten required figures show hit rate, actual eviction rate, occupancy with
capacity 256, Q gap, implied/observed SILENCE with 1/17 reference, ties, age
profiles, novelty categories summing to 100%, absolute TD error and deterministically
selected individual trajectories. Two additional figures show matched-window
Q/retention versus food MI and matched individual-window age versus food/water MI.
All were visually checked and saved at 180 dpi; captions separate C/D controls,
seed SD and descriptive interpretation. CSV exposures are authoritative for
sparse age/individual estimates. The MI scatter is descriptive and not a causal
model or independent-window significance test.

### Performance and storage

| Variant, 100 ticks | Median seconds (5 timed repeats) | Output bytes |
| --- | --- | --- |
| baseline_v061 | 0.469890 | 0 |
| v062_off | 0.473242 | 0 |
| v062_vocal_on | 0.734305 | 887363 |
| v062_both_on | 0.839931 | 2255215 |

One warm-up precedes five separately timed repetitions/variant. Benchmarks were
sequential without concurrent analysis/simulations and include stream/gzip closure,
excluding metadata hashing. OFF differs by 0.71% from archived v0.6.1
(within local timing noise); diagnostics ON costs about 55.2% over OFF here,
and both observers about 77.5%. Every benchmark full-state SHA-256 is identical:
`24472d133ad6a3a3b9094163d6b622196de1b4158921427034f5bf4faa0aa31c`.

Six-run recording time sums to 422.819 s;
individual runs take 63.6–78.3 s. These pilot wall times include normal CLI,
metadata hashing and some concurrent test/analysis activity, not a controlled
overhead estimate. Controlled overhead comes from the separate benchmark above.
The 4,209,670 decisions and every TD/row event use
282,773,547 compressed bytes
(269.674 MiB), excluding small
sidecars/ordinary statistics/logs. Peak living individual/key histories range
41,785–44,528 (C), 58,320–60,330 (D); run catalogs contain 2454–2509 and
29,734–30,125 keys. These are object counts, not measured resident memory.

Final offline analysis takes 112.950 s locally and writes
304,964,812 bytes (290.837 MiB).
The per-state table is the largest output; summary storage still scales with
observed individual/key pairs. Nothing silently samples updates or events.
The old 24-minute permutation analysis and duplicate permanent communication
CSVs were avoided. Timings depend on this machine/cache and are not scale guarantees.

Machine-readable checks: `data/v062_verification.json`, `data/v062_pilot_summary.json`,
`data/v062_performance.json`, `data/v062_analysis_timing.json`,
`data/v062_row_retention.csv`, `data/v062_state_revisits.csv`,
`data/v062_turnover.csv`, and the pilot/analysis
metadata. Raw outputs use existing data/analysis ignore rules and remain saved locally.

## Scientific answers and remaining limits

1. **Nonuniform Q values?** Yes, at 49.14% (C) and 29.31% (D) of decisions;
   most differences leave several equal maxima. They do not imply communication.
2. **Nonuniform policies?** Yes. Mean TV from uniform is 0.0705/0.0418, while
   unique maxima occur in only 3.88%/2.65% of decisions. Aggregate label frequency
   and individual state policy must be distinguished.
3. **Revisitation?** Hits are 49.14%/29.31%; many updated individual/key pairs
   are updated once, especially D. Recreated chosen keys are rare and separate
   from first visits.
4. **Auditory fragmentation?** D has more run keys and lower hit rate in all
   three seeds. The controlled identical-body/visual test proves that different
   auditory features create distinct keys only when listening is enabled.
   Ecological C/D differences also include changed physical experience and
   population composition; their aggregate difference is not an isolated cause.
5. **LRU loss?** Actual evictions are measured: mean 318/3685 rows per run,
   about 601/5028 associated updates. Death discards far more rows. No information
   bit count or counterfactual performance loss is identified. Initial MI collapse
   precedes every eviction.
6. **SILENCE preferred?** It is uniquely maximal at about 0.221%/0.144% of
   decisions; merely tying for max is much more frequent. This is not cooperation.
7. **Mean SILENCE near 1/17?** Yes: implied 5.877%/5.872%, observed 5.860%/5.880%.
   This is compatible with state-level policy differences and arbitrary-label
   averaging, rather than proof of no learning or intentional restraint.
8. **Stronger with age?** Moderate-age profiles and fixed early/next same-ID
   changes support stronger preferences among observed survivors. Very old bins
   are sparse; survival, cohort and ecological effects prevent a causal age claim.
9. **Early MI explained?** Not uniquely. Declining population Q gaps/hits and
   extensive turnover accompany the fall. Same-ID early/next MI actually grows;
   early/final pairs are too sparse. LRU cannot explain the initial decline.
   A decomposition into changing context, replacement and within-ID change is
   unresolved; correlations do not establish a causal mechanism or semantics.
10. **Unresolved?** Stability of particular preferences in matched retained
    states, counterfactual retention costs, time/age/ecology confounding, and
    receiver use of vocal variation. TD sign variation is not an explanation by
    itself. A shared convention or communication benefit has not been shown.

Recommended next question: using existing recordings first, compare retained
individual/state/action trajectories across matched short time windows and
founder/birth cohorts, with explicit exposure/age/context matching. Only after
that observational review should any causal intervention be separately designed.
No reward, cost, capacity, ecology, or v0.7 change is proposed as part of this release.

## Files changed

Modified: `.gitignore`, `README.md`, `agents/learning.py`, `main.py`,
`simulation/simulation.py`.
Added: `simulation/vocal_learning_observer.py`,
`experiments/vocal_learning_data.py`, `experiments/analyze_vocal_learning.py`,
`experiments/vocal_learning_plots.py`, `experiments/vocal_learning_pilot.py`,
`tests/test_vocal_learning_diagnostics.py`, and this report.
The existing Config, RandomBrain, sensory/biology/resource code, aggregate CSV
schema, communication schema, historical reports and fixtures remain unchanged.
