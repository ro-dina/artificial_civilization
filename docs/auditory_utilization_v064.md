# v0.6.4 — Counterfactual Auditory Utilization

## Research question and evidence levels

Does changing only auditory input change the exact physical policy of the same
trained receiver? Does a difference reflect acquired predictive discrimination,
and does it improve an ecological outcome? These are separate questions.

1. **Observational association:** empirical actions differ between recorded
   auditory conditions. Unobserved Q, learning, time, sub-bin physiology,
   environment and selective survival can explain differences.
2. **Frozen-policy sensitivity:** an auditory intervention changes the exact
   epsilon-greedy distribution with the same stored Q table and nonauditory
   input. This establishes dependence of that frozen lookup policy on the
   intervention, subject to the specified memory baseline.
3. **Learned discrimination:** distinct, familiar, updated auditory states have
   different policies, and predictive learning exceeds appropriate controls.
   A learned row versus an unseen zero prior is insufficient.
4. **Ecological benefit:** a separate controlled outcome comparison demonstrates
   improved resource acquisition or survival. This release does not perform
   matched frozen ecological rollouts.

None of these levels establishes intentional signals, shared meaning, speaker
adaptation, conventions, culture or language. An experimenter's predictive cue
is not a spontaneously informative agent signal. Mutual information between
observed sounds and sampled actions is not a causal effect estimator.

## Architecture and compatibility

Only one normal-simulator file changes: `agents/learning.py` gains
`inspect_physical_q(encoded_state) -> (present, immutable_values)`. It uses the
existing OrderedDict's nonmutating `get`, returns one row, and uses the actual
zero prior without allocating for an absent row. A stored zero row remains
identifiably present. Existing methods and normal call paths are unchanged.

The other code is isolated under `experiments/`:

- `auditory_policy.py`: exact probabilities, interventions, row coverage,
  categorical distances and full-state verification.
- `auditory_retrospective.py`: streaming, window-bounded observational matching.
- `auditory_controlled.py`: predefined training schedule and frozen probes.
- `auditory_utilization_plots.py`: headless 180-dpi figures without random jitter.
- `analyze_auditory_utilization.py`: CLI, streaming outputs, seed summaries,
  metadata, hashes and separate evaluator benchmark.

No changes to Config, biology, resources, rewards, Q equations, action selection,
exploration, auditory delivery/masking, emission/timing, vocabulary, memory
semantics, schemas or RNG streams. No observer is inserted into normal decisions.
The ordinary simulator remains standard-library-only. Analysis uses the existing
optional numpy/pandas/matplotlib dependencies; no seaborn is used.

## Exact frozen policy and interventions

For N physical actions and M exactly tied maxima:

```text
P(a | s) = epsilon/N + (1-epsilon) * I[Q(a) == max(Q)]/M
```

The selector's exact equality is preserved. The evaluator checks finite,
nonnegative probabilities summing to one; tolerance is used only to validate the
sum, never to decide Q ties. It returns the whole physical probability vector,
maximal set and unique maximum when defined. TV is half the L1 distance; JSD is
in bits with the usual zero-probability convention.

With a fixed action set and epsilon, an identical **unique** maximum gives an
identical epsilon-greedy policy even if Q magnitudes differ. Conversely, an
unchanged representative argmax can conceal a changed maximal set and policy.
Unique-greedy changes are unavailable if either side is tied; maximal-set and
full-distribution changes remain defined. No sampled actions estimate a frozen
policy.

The bounded probe set is defined before training. It uses thirst 60, hunger 0,
energy 100, the same evaluation tick and two separate visual contexts: current
tile empty, or containing one water unit. The two visual contexts are never
compared as an auditory-only intervention. Each has nine paired probes:

| Probe | Original → alternative | Past-memory baseline |
|---|---|---|
| remove_current_empty | identified 7/E → SILENCE | empty on both sides |
| remove_current_keep_past | identified 7/E → SILENCE | retained 12/E, age 1 on both sides |
| substitute_id_a_b | 7/E → 12/E | empty |
| substitute_id_b_a | 12/E → 7/E | empty |
| substitute_direction | 7/E → 7/W | empty |
| substitute_memory_id | current SILENCE on both sides | retained 7/E → 12/E, age 1 |
| empty_encoded_auditory | current SILENCE on both sides | retained 7/E, age 1 → empty |
| silence_masked | SILENCE → MASKED | empty |
| substitute_untrained_id | 7/E → lowest valid nontraining ID (0) | empty |

IDs 7, 12 and 0 are categorical units, not meanings or similar sounds. No exact
sender position or identity enters an observation.

The default live preview reuses `LearningBrain.state` with memory as currently
held. An explicit immutable memory tuple constructs only an isolated mechanical
`AuditoryMemory`; no live memory is cleared, updated or cloned with its Q tables.
All controlled interventions explicitly use pre-current-event history. The
`memory_before` helper excludes an already inserted same-tick event, preventing
"remove current" from accidentally recalling the very cue being removed.
Current removal can recall older sound; it is not equivalent to erasing memory.
An empty eighth auditory feature is not a listening-disabled seven-element key.

Each pair holds the same brain, body, visual features, action set, epsilon, tick
and past memory fixed except for the declared intervention. Pending physical and
vocal transitions are held unchanged: evaluation does not complete the previous
transition as a future ordinary decision would. Policy results describe the
stored snapshot, not that possible later updated table.

Row presence is split into both present, original only, alternative only and
neither; summaries combine the two one-present directions only after preserving
their individual probe labels. Stored-zero, stored-all-tied, absent-zero and
pure-exploration uniformity are distinguished. A present row can be merely
allocated: independent training visit/update counters additionally establish
actual update exposure. Primary familiar-state comparisons require **two
distinct present keys with at least one update each**. The task is checked to
remain below physical memory capacity, so lifetime update counters refer to
retained rows without intervening evictions.

## Retrospective design and identifiability

The twelve existing v0.6.1 recordings, A/B/C/D × seeds 0/1/2, are reused.
B/D encode auditory input; A/C are current-reception observational controls.
Each input and metadata sidecar is checked against the original archived input
hash, full effective Config and final horizon. The archived analysis metadata
itself is pinned by the existing v0.6.1 verification manifest. A seed alone never
authorizes a join. No historical pilot or permutation analysis is rerun.

Receiver exposure is reported for each run and `(0,100]`, `(100,200]`, … window.
Current SILENCE/MASKED/IDENTIFIED, identified ID/direction, and exact encoded
memory feature are separate measures. Encoded age is retained. A current
IDENTIFIED event has age 0; SILENCE can retain an older identified or masked
event. Listening-OFF encoded rates are unavailable, not silent/zero estimates.
Feature frequencies use sparse observed categories; omitted categories have no
observed occurrences, not an estimated extra category. Encoded frequencies for
OFF use NOT_ENCODED with unavailable probabilities.

Matching is within run, individual, window and the seven recorded primitive
integers: hunger/thirst/energy bin, nearest-food dx/dy and nearest-water dx/dy.
There is no pooling across individuals or physical contexts to enlarge samples.
Four scopes distinguish reception from encoding:

- `current`: compare exact current auditory percepts.
- `encoded`: compare exact encoded auditory features.
- `current_given_encoded`: hold the exact encoded feature fixed as well.
- `encoded_given_current`: hold the exact current percept fixed as well.

For each scope, an auditory cell requires 2, 5 or 10 decisions independently.
At least two eligible cells are required per matched context. Every unordered
eligible exact-feature pair yields empirical physical action TV/JSD, action
entropies, sample sizes, first/last ticks, mean tick separation and temporal-range
overlap. Signal ID contrasts fix direction and age; direction contrasts fix ID
and age. SILENCE/IDENTIFIED, SILENCE/MASKED and retained-age contrasts are also
separated. Other simultaneous feature differences are labeled explicitly.

Within a context, pair distances have equal weight. Context means then have
equal weight within a run. Run-level unique individuals are unioned across
windows; window-context counts are summed. Exposure coverage counts the union
of supported decisions once, not their repeated membership in multiple pairs.
Cell sample-size minima/median/maxima and complete pair/context tables are saved.
Seed summaries give equal weight to each run with finite measurements and use
sample SD (`ddof=1`). Counts and exposure accompany every result. Unavailable
distances are blank; zero matched coverage is a real measured zero.

These matches do not identify historical policies. Different auditory events
can occur at different times, with different Q values, learning histories,
sub-bin body needs, environments and survival/reproduction selection. Current
percepts and remembered earlier events are not interchangeable. Existing
communication records contain sampled physical actions, not full Physical Q;
Vocal Q diagnostic summaries cannot supply the missing Physical Q either.

## Predefined controlled experiment

The complete design was written to `data/v064_design.json` before default
training. Each output also contains `controlled_design.json` and its canonical
protocol SHA-256 in metadata. No outcome-based horizon/seed/cue tuning occurred.
The earlier v0.5/v0.6 listening demonstration is not a bidirectional success:
its saved outputs prefer east for cue 7 but **north**, not west, for cue 12.

The new test is one receiver in a 3×1 grid, no reproduction or regeneration,
visual radius 0, hearing radius 8, vocabulary 16, auditory memory 4. Default
alpha .2, gamma .9, epsilon .2 and physical capacity 256 are unchanged. Vocal Q
remains enabled and uses its original separate RNG and shared reward. Self
emissions remain excluded from hearing. Hunger increases by 0, thirst by 1;
maximum age is disabled. These task-specific Config choices are recorded in full.

Seeds 0/1/2 each run 256 trials of 8 ordinary simulation steps. Before each trial,
the external harness resets position to x=1 and hunger/thirst/energy to 0/60/100;
age, tables and learning histories persist. One water unit is placed at x=2
(context 0) or x=0 (context 1). The external cue is emitted from x=2 in both
contexts, always coarse direction E. Ordinary local perception and physiology
supply all learning inputs and rewards. Eight ticks exceed the four-tick memory
horizon; the next trial's root has no retained previous cue.

No Q rows/values, selected actions, TD updates, rewards or terminal flags are
injected. Pending transitions persist across environmental/body resets as in
the earlier demonstrations; cross-trial bootstrapping is a limitation, not an
artificial terminal-reset modification to the learner.

Four conditions:

- `predictive_on`: listening ON; external cue 7 accompanies right water,
  cue 12 accompanies left water.
- `predictive_off`: identical context schedule, external emissions and resets,
  but the physical state has no auditory feature.
- `independent_on`: same context schedule; each randomized four-trial block
  contains all four context/cue pairs once, eliminating the cue/context
  association. Independent named schedule RNGs are derived from each seed.
- `untrained_on`: listening ON, constructed but never stepped or updated.

Each block has both contexts twice. No trial number, absolute tick, earlier
context label or precise source coordinate is encoded. Cue source direction is
constant. The independent schedule is balanced within each context, not a
circular shift that accidentally preserves a temporal mapping. These controls
remove explicit cue/timing leakage into the root state; the ordinary learner
still has a changing Q history, so finite-sample and learning-order effects
remain possible and are measured rather than assumed absent.

Checkpoints at 0/32/64/…/256 trials compute exact root probabilities without
changing training. The preceding block's first-action correctness, water per
trial and physiological reward per action are separately recorded. Those are
training outcomes; trajectories differ between conditions as learning proceeds.
They are not matched frozen ecological rollout effects.

Predeclared per-cue criterion: the expected move (east for 7; west for 12) must
be the unique maximum with probability at least .5, in a present updated row.
Both cues meeting it is bidirectional; one is partial; neither is failed or
untrained. Aggregate support requires bidirectional predictive learning and
higher mean correct-action probability than OFF/independent/untrained in every
seed. Failures and control discrimination are retained. No ecological benefit
claim follows from this criterion.

## Observed retrospective results

The completed archive contains **8,424,234 receiver decisions** across
12 runs and 120 run-windows. Every run has seeds 0/1/2. Identified exposure
(equal-seed mean ± sample SD) is 19.1244% ± 0.0424 percentage points in B and
33.2036% ± 0.0810 points in D. Most current reception is MASKED: 80.6419% in B,
66.7824% in D. Encoded nonempty rates are 99.9765% and 99.9860%.
In B, current/encoded difference and current-SILENCE-with-memory rates are both
0.2102% ± 0.0121 points; D has zero such decisions in these recordings. These
are exposure descriptions, not evidence that receivers use the distinction.

Exact encoded-feature matching, minimum two decisions per cell:

| condition | seed | matched_contexts | individuals | decisions_used | receiver_decisions | coverage | mean_context_tv | mean_context_jsd_bits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | 0 | 370 | 286 | 2949 | 716567 | 0.004115 | 0.777547 | 0.712957 |
| B | 1 | 342 | 264 | 2795 | 717996 | 0.003893 | 0.743286 | 0.675920 |
| B | 2 | 360 | 277 | 3010 | 717763 | 0.004194 | 0.766917 | 0.695277 |
| D | 0 | 2103 | 1609 | 12387 | 714512 | 0.017336 | 0.748384 | 0.698672 |
| D | 1 | 2080 | 1614 | 11606 | 719294 | 0.016135 | 0.753672 | 0.704622 |
| D | 2 | 2110 | 1649 | 11957 | 713626 | 0.016755 | 0.739974 | 0.689121 |

Coverage is a fraction of that run's decisions. Equal-seed summaries (n=3) are:

| Condition | Matched window-contexts | Decision coverage | Observed action TV | Observed action JSD bits |
|---|---|---|---|---|
| B | 357.333 ± 14.189 | 0.4067% ± 0.0156 percentage points | 0.762583 ± 0.017537 | 0.694718 ± 0.018525 |
| D | 2097.667 ± 15.695 | 1.6742% ± 0.0601 percentage points | 0.747343 ± 0.006908 | 0.697472 ± 0.007820 |

These large empirical distances use very few decisions per exact cell (median
**2 in every B/D run**). They are especially vulnerable to sampling variability.
At threshold **5**, B seed 1 alone has one eligible encoded context, 26 decisions,
coverage 0.00362%, TV .7125, JSD .698690 (effect n=1, SD unavailable).
All other B/D runs have no comparison. At threshold **10**, none has an encoded
comparison. Unavailable values are not evidence of zero sensitivity.

Even listening-OFF controls show large matched **current-percept** action TV at
threshold 2: A seeds 0/1/2 give .735784/.710066/.713649; C gives
.875837/.881847/.862730. Their brains do not encode sound. This illustrates why
observed action differences must not be interpreted as causal auditory use.

With current percept held fixed, B has only 7/6/3 matched encoded-memory
contexts (33/35/20 decisions; coverage .00461%/.00487%/.00279%). D has none.
With encoded feature fixed, neither B nor D has matched current-percept
contrasts at threshold 2. Specific ID contrasts fixing direction and age have
11/13/9 contexts in B and 134/122/118 in D. Direction-only contrasts have
0/1/0 in B and 0/2/0 in D. Most general contrasts change more than one exact
auditory component. No direction-specific ecological conclusion is supported.

The mean pair tick separation inside encoded contexts is 9.46/10.92/10.44 ticks
in B and 6.04/6.13/6.28 in D. Full temporal ranges are retained in pair outputs;
recorded physical-state matching does not freeze historical Q or learning.

## Observed frozen-policy and controlled-learning results

The following table contains **every condition and seed**, not selected positive
runs. E/W/N/S denote physical moves, and the expected moves are E for cue 7 and
W for cue 12. Probabilities average the two externally specified first moves.
The exact action strings are retained below. Water is cumulative **training**
water, not a frozen-rollout treatment effect.

| condition | seed | cue_7_unique_action | cue_12_unique_action | mean_correct_probability | root_identity_tv | root_identity_jsd_bits | learning_result | training_water_consumed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| predictive_on | 0 | move_east | move_west | 0.828571 | 0.800000 | 0.676421 | bidirectional | 241 |
| predictive_off | 0 | move_west | move_west | 0.428571 | 0.000000 | 0.000000 | partial | 124 |
| independent_on | 0 | move_west | wait | 0.028571 | 0.800000 | 0.676421 | failed_or_untrained | 212 |
| untrained_on | 0 | — | — | 0.142857 | 0.000000 | 0.000000 | failed_or_untrained | 0 |
| predictive_on | 1 | move_south | move_west | 0.428571 | 0.800000 | 0.676421 | partial | 239 |
| predictive_off | 1 | move_west | move_west | 0.428571 | 0.000000 | 0.000000 | partial | 121 |
| independent_on | 1 | move_south | move_west | 0.428571 | 0.800000 | 0.676421 | partial | 200 |
| untrained_on | 1 | — | — | 0.142857 | 0.000000 | 0.000000 | failed_or_untrained | 0 |
| predictive_on | 2 | wait | drink | 0.028571 | 0.800000 | 0.676421 | failed_or_untrained | 222 |
| predictive_off | 2 | move_west | move_west | 0.428571 | 0.000000 | 0.000000 | partial | 126 |
| independent_on | 2 | move_west | move_east | 0.028571 | 0.800000 | 0.676421 | failed_or_untrained | 180 |
| untrained_on | 2 | — | — | 0.142857 | 0.000000 | 0.000000 | failed_or_untrained | 0 |

For both trained listening-ON conditions, both root cue keys were present and
updated **128 times each**. Substituting 7/E ↔ 12/E therefore compares two distinct
familiar learned rows, not a trained row versus an unseen prior. All three seeds
have TV **.8** and JSD **.676421 bits** in both predictive and independent controls.
OFF uses the same seven-element key for both cues and has exactly zero TV/JSD;
untrained ON has distinct absent keys and uniform policies, also zero TV/JSD.
This establishes Level 2 frozen-policy dependence in trained listening brains.
The independent control shows that sensitivity alone is not predictive usefulness.

Equal-seed summaries (n=3, sample SD):

| Condition | Root ID TV | Root ID JSD bits | Mean correct-action probability | Training water |
|---|---|---|---|---|
| predictive ON | .800000 ± .000000 | .676421 ± .000000 | .428571 ± .400000 | 234.000 ± 10.440 |
| predictive OFF | .000000 ± .000000 | .000000 ± .000000 | .428571 ± .000000 | 123.667 ± 2.517 |
| independent ON | .800000 ± .000000 | .676421 ± .000000 | .161905 ± .230940 | 197.333 ± 16.166 |
| untrained ON | .000000 ± .000000 | .000000 ± .000000 | .142857 ± .000000 | 0 (no training) |

Predictive seed 0 meets the bidirectional criterion, seed 1 is partial, seed 2
fails. All independent controls also acquire auditory-conditioned differences;
one meets the single-cue action criterion. **The predefined replicated Level 3
criterion is not met.** A single successful seed establishes a limited capability
example, not robust learned predictive use across seeds. No seeds, horizons,
rewards or probes were changed to obtain a stronger result. Training acquisition
can be high despite an unsuitable frozen first-action preference: the eight-step
trial permits later moves and drinking, and the evaluation intentionally targets
one narrow root state. Neither endpoint is a general intelligence/fitness score.

### Interventions and Q-state coverage

Predictive ON root-context distances (equal-seed mean ± sample SD, n=3):

| Intervention | TV | JSD bits | Coverage in every seed |
|---|---|---|---|
| current removal, empty past | .533333 ± .461880 | .450947 ± .390532 | both present, updated |
| current removal, retained past | .533333 ± .461880 | .450947 ± .390532 | both present, updated |
| ID substitution, either direction | .800000 ± .000000 | .676421 ± .000000 | both present, updated |
| retained-memory ID substitution | .800000 ± .000000 | .676421 ± .000000 | both present, updated |
| empty encoded auditory baseline | .533333 ± .461880 | .450947 ± .390532 | both present, updated |
| source E → W | .685714 ± .000000 | .373094 ± .000000 | original only |
| SILENCE → MASKED | .685714 ± .000000 | .373094 ± .000000 | original only |
| ID 7 → unseen ID 0 | .685714 ± .000000 | .373094 ± .000000 | original only |

The three .533333 mean distances do not mean the same seeds changed: current
removal with empty past is zero in seed 0, removal preserving cue 12 memory is
zero in seed 1, and memory erasure is zero in seed 1. Exact per-probe rows retain
that difference. Direction/MASKED/unseen-ID effects are learned-versus-zero-prior
contrasts and do **not** demonstrate familiar-state direction or masking learning.

Across the predefined **18 paired probes per receiver** (two distinct visual
contexts, not ecological exposure-weighted samples):

| Condition (all seeds identical coverage) | Both present | One present | Neither present | Distinct updated comparisons per seed |
|---|---|---|---|---|
| predictive ON | 8/18 | 6/18 | 4/18 | 8 |
| independent ON | 8/18 | 6/18 | 4/18 | 8 |
| predictive OFF | 18/18 | 0/18 | 0/18 | 0 (same key) |
| untrained ON | 0/18 | 0/18 | 18/18 | 0 |

Among distinct updated comparisons, nonzero TV counts are **5/6/8** for
predictive seeds 0/1/2 and **4/8/7** for independent seeds 0/1/2. Probes include
reversed ID substitutions; they are bounded diagnostics, not independent
replicates. The seed, not the probe, is the independent comparison unit.

Coverage is deliberately limited. Some visible-water/retained-memory alternatives
were never experienced at the exact body bins, so neither row exists. Absent
states give the real zero prior, not a reconstructed learned policy. Counts,
actual row presence, update/visit exposure and uniformity reasons accompany each
probe; no unavailable learned-state claim is converted into zero effect.

## Artifacts, schemas and reproducibility

Run from the repository root with Python 3.11+ and `requirements-analysis.txt`:

```bash
# Offline historical diagnostics only; never steps Simulation.
python3 -m experiments.analyze_auditory_utilization --part retrospective

# Explicit isolated small experiment only (no historical pilot).
python3 -m experiments.analyze_auditory_utilization --part controlled

# Both components; the benchmark receipt is separate because timings vary.
python3 -m experiments.analyze_auditory_utilization \
  --performance-path data/v064_performance.json
# Direct invocation is also supported:
python3 experiments/analyze_auditory_utilization.py --help

# Complete tests, including every historical regression.
python3 -m unittest discover -s tests -v
```

Default input is `data/communication/*_communication.csv`; default output is
`analysis/auditory_utilization/`. CLI options include `--data-dir`, `--output-dir`,
`--window-size`, `--thresholds`, bounded `--trials`, `--seeds`, and explicit archive
metadata/verification paths. Default archive checks are required for actual
historical results; programmatic synthetic tests explicitly pass no registry.
`--part retrospective` writes empty controlled tables rather than implying that
training occurred; `--part controlled` does not load historical recordings.
Existing unknown/historical destination files are rejected. Repeated v0.6.4
outputs replace only the prior manifested files; old component figures are not
left behind if the selected part changes.

The completed combined output has **26 files**, 15 data tables, one design JSON,
one metadata JSON and nine PNGs:

| File | Main unit / contents |
|---|---|
| retrospective_exposure_by_run.csv | 12 independent run exposure rows |
| retrospective_exposure_by_window.csv | 120 run/window rows |
| retrospective_feature_frequencies.csv | 9,932 sparse run/window feature-category rows |
| retrospective_matched_contexts.csv.gz | 21,020 exact window-context/threshold/scope rows |
| retrospective_action_distances.csv.gz | 24,213 empirical exact-feature-pair distances and temporal/sample support |
| retrospective_matched_summary.csv | 6,840 run/window/scope/threshold/contrast rows |
| retrospective_matched_by_run.csv | 684 run/scope/threshold/contrast rows, unioned individuals and supported decisions |
| retrospective_seed_summary.csv | equal-seed mean/sample SD/min/max/finite n by condition/scope/threshold/contrast/metric |
| counterfactual_probes.csv | 216 intervention pairs; current/history/key/probabilities/maxima/exposures/presence/deltas |
| counterfactual_summary_by_run.csv | condition/seed/visual-context/intervention/coverage summaries and defined greedy denominators |
| counterfactual_summary_by_intervention.csv | equal-seed summaries, keeping visual context and row-coverage class separate |
| counterfactual_state_coverage.csv | condition/seed presence counts and fractions over the 18 selected probes |
| training_learning_curves.csv | 84 checkpoints; frozen root policy and preceding training block outcomes |
| controlled_experiment_results.csv | all 12 condition/seed endpoints, explicit failed/partial/bidirectional status |
| controlled_experiment_seed_summary.csv | endpoint equal-seed mean/sample SD/min/max/n |
| controlled_design.json | prespecified complete protocol and effective configuration template |
| analysis_metadata.json | completion/schema, full input/config/source/output hashes, probes, software, read-only proofs |

The exact column lists for every table are stored in metadata. `evidence`
separates `observational`, `frozen_policy_intervention`, `frozen_policy` and
`training_and_frozen_policy`. No complete Q tables are exported. The two large
observational tables stream through deterministic gzip (empty stored filename,
mtime=0) and are not reloaded into a large dataframe for plotting.

Figures, all headless at 180 dpi:

- `retrospective_exposure_memory.png`
- `retrospective_matched_coverage.png`
- `retrospective_action_distances.png`
- `counterfactual_policy_tv.png`
- `counterfactual_policy_jsd.png`
- `counterfactual_state_coverage.png`
- `greedy_vs_policy_changes.png`
- `controlled_learning_curves.png`
- `controlled_condition_comparison.png`

Points show seeds with deterministic offsets; uncertainty is sample SD, not a
confidence interval or a significance test. Conditional coverage/greedy
fractions use only their stated denominators. No per-decision independence is
assumed. Empty comparisons remain gaps. These diagnostics do not select an
optimal auditory condition.

Metadata binds all 12 input/sidecar hashes, the pinned historical registry,
complete Configs, controlled seeds/horizon, source hashes of all simulation
modules and the new evaluators, exact intervention rules, thresholds, output
schemas/counts/hashes, software versions and completion. Per-run frozen proofs
and checkpoint proofs compare complete state fingerprints. Runtime and benchmark
measurements live in separate ignored `data/v064_*.json` receipts, keeping
scientific CSV/JSON metadata deterministic. Existing recordings and archived
analysis directories are never rewritten. Generated results remain saved locally
under the repository's established ignored-data policy.

## Verification and cost

Before implementation, all **360** existing tests passed (64.592 s). The final
suite contains **409 passing tests**, including 49 new tests and every unchanged
legacy test. Tests cover exact zero/unique/tied/negative-Q probabilities, exact
rather than approximate ties, all four row-presence patterns, TV/JSD, invalid
inputs, memory expiry, explicit past versus current sound, all interventions,
OFF dimensionality, sparse thresholds, within-individual matching, union-based
coverage, temporal separation, unique-agent run aggregation and equal-seed SD.

Every probe in the complete-state isolation test checks both Q tables and LRU
orders, both pending transitions, auditory memory and counters, all body/genome
fields, resources, simulation/brain/global RNG state and immutable inputs before
and after the individual query. Reversed probe order gives identical policies;
subsequent normal actions and complete state agree with an unprobed simulation.
Allocator/decision call guards reject accidental mutation. The controlled harness
is checked to call ordinary Simulation.step only, with no direct Q edits/updates.
All four small conditions replay identically; untrained failures and OFF's
collapsed keys are explicitly tested. Retrospective-only tests forbid stepping.

The full combined analysis was independently replayed. All **26 CSV/gzip/JSON/
PNG files** were byte-identical, including state hashes and metadata, in the
same Python/numpy/pandas/matplotlib environment. The final figure-only legend/
layout correction was also regenerated identically on both copies, without new
training. Synthetic end-to-end tests independently verify deterministic complete
outputs. No historical fixture was regenerated. The six frozen fixture files,
all 206 snapshotted historical data/result files and every pre-existing source/
report other than the single LearningBrain file retain their original hashes.
Removing only the added inspection method from its AST yields exactly the old
LearningBrain module AST; all old algorithmic code is identical.

The frozen harness checks complete pre/post state and global RNG at each of 84
training checkpoints and all 12 endpoint batches (216 paired probes). These
verification snapshots occur outside the evaluator and ordinary decision path;
normal simulations neither copy tables nor call the new method. Persisted
receipts are `data/v064_before.json`, `v064_design.json`, `v064_test_results.json`,
`v064_runtime.json`, `v064_performance.json`, `v064_replay.json` and
`v064_verification.json`; test/analysis/replay logs are also retained.

Local combined runtime was **44.61 s**, with peak process RSS approximately
**232.05 MiB**. This includes parsing/hash-checking 8.42 million old rows, the
nine trained single-agent experiments (18,432 total ordinary training ticks),
three untrained receivers, frozen checkpoints/probes, figures and evaluator
benchmark. The independent replay took 42.30 s. These are local indicative times,
not portable throughput guarantees or a simulator performance comparison.

A separate five-repeat × 5,000-call benchmark gave median **5.61 μs** per live
read-only evaluation and **6.03 μs** with isolated explicit empty memory. Complete
state checks confirmed no side effects. Cost scales with local observation size,
one physical action row and bounded auditory history; whole-Q fingerprints are
verification overhead outside timed calls. The normal hot path has no extra
calls, branches, Q copies or observer bookkeeping. The new result directory is
approximately **5.02 MB** of file contents (26 files), rather than full per-agent
histories. Streamed retrospective working memory is bounded by a time window;
run-wide contributing ID sets and cell-size counters are diagnostic aggregation
state, not agent history. Exceedingly dense feature combinations could still
produce many pair rows; the selected exact-cell thresholds and compression make
that cost explicit rather than silently subsampling.

## Scientific conclusion and next questions

**Implemented and verified:** reproducible observational matching, exact frozen
physical-policy interventions, row/update coverage, no-state-change evaluation
and a prespecified controlled acquisition test, with intact historical behavior.

**Observed:** narrow retrospective support with severe threshold sensitivity;
nonzero frozen auditory sensitivity between familiar updated rows in trained
listening receivers; identical root sensitivity in cue-independent controls;
seed-dependent task learning (one bidirectional, one partial, one failed).

**Not established:** replicated predictive learned discrimination beyond controls
under the predefined criterion, causal interpretation of historical associations,
any controlled ecological benefit, speaker usefulness, shared meaning or language.
No ecological-benefit rollout was attempted; training water is not that test.

Only three small controlled seeds and one task/horizon were predefined. The
lookup state is coarse and partially observed. Present rows have selected
training exposure; missing alternatives and biased state coverage restrict
generalization. Episodes reset body/environment while pending transitions remain,
which can carry cross-trial learning. Exploration, noisy incremental values,
optimistic zeros, simultaneous independent Vocal Q and the narrow first-action
criterion can affect endpoints. Seed-dependent failures are retained rather than
explained away or repaired through outcome-driven parameter tuning.

Both learning heads receive the same physiological reward, which tells a speaker
about its own outcome, not whether an emitted signal was heard or helped a
receiver. Receiver sensitivity therefore does not demonstrate sender adaptation
for receiver benefit. Experimenter-imposed cues cannot demonstrate spontaneous
information production or population conventions.

A later reviewed study could predefine a held-out acquisition/rollout design,
more independent schedules, matched frozen resource/survival outcomes and a
criterion separating stable predictive response from arbitrary control-table
fragmentation. Retained memory and current reception should remain separately
intervened; row coverage must accompany any new result. This release stops at
diagnostics. It does not implement v0.7, self-masking, speaking costs, new rewards,
phonetic semantics, social imitation or language algorithms.
