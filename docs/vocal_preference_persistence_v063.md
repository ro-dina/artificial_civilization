# Vocal preference persistence v0.6.3

This is an offline, longitudinal analysis of the six existing v0.6.2 diagnostic
runs. No new ecological simulation, permutation analysis, learning mechanism,
Config default, cost, reward, auditory rule or diagnostic schema was introduced.
Stable numerical preferences do not establish communication, meaning, conventions,
cooperation or language.

## Research question, inputs and integrity

When the same individual encounters the same exact encoded state again, does its
vocal preference persist? Formation, continuous retention, updates, eviction,
death/replacement and sampled behavior are separate observations.

Inputs: `data/vocal_learning/{C,D}_seed_{0,1,2}.csv.gz`, their `.updates`, `.rows`,
`.states` companion streams and completed sidecars/manifest. All runs finish at
1000 ticks with the unchanged evolution preset. C has learned listening OFF,
D ON; both have Vocal Q ON. Full Config, seed, condition, horizon, schema version 1,
completion, all four SHA-256 streams, record counts and the exact state dictionary
are validated. The pilot manifest detects missing complete runs. No partial-input
option is implemented: missing/inconsistent encounters, orphan events or unclosed
episodes are errors. A completed partial *final window* is supported separately.

The 30 original source/fixture/report files and 114 historical data/analysis files
captured before editing retain their exact hashes. `main.py`, Config, agents,
simulation/systems/world modules, diagnostic schemas and historical reports are
unchanged. Only new analysis/tests, README and the analysis ignore path change.
Old results cannot be overwritten; new outputs stage in a temporary directory and
are published only on successful completion.

## Units, timing and allocation episodes

The primary key is `(run, agent_id, exact encoded state, allocation episode)`.
Run-local state IDs are never compared across runs. The one-to-one canonical JSON
state dictionary is decoded and structurally validated; cross-run pooling of
identically numbered states is prohibited.

An encounter is an actual Vocal Q choice. Streams are read by tick and recorded
actor order. Previous nonterminal updates precede the current choice; terminal
updates and death follow it. Actual row events interleave at the learner's recorded
update/selection phases, using their state/epoch identity and counters. The
analyzer does not simulate a second LRU, reconstruct Q vectors or calculate rewards.

Each actual allocation begins an episode; actual eviction, death or run end
closes it. `run_end` is a censored storage endpoint, not preference failure.
The same key recreated after eviction gets a new episode. Encounters are first-ever,
retained revisit, or post-eviction revisit. Primary persistence pairs are consecutive
encounters **within the same actual episode**. Recreated encounters remain separate
and do not count as memory retention. Death/end are never counted as LRU evictions.

Storage duration is end tick − allocation tick; encounter span is last − first
encounter tick. Same-tick episodes can have duration zero; singleton encounter
span is zero and mean/max revisit interval is undefined. Long storage need not
mean reuse or reinforcement. Intervening update counts use the actual update stream
and recorded cumulative counters through the second choice, not a later snapshot.
Every retained pair in this corpus has exactly one intervening update to that row;
the individual's total intervening updates can be larger.

`first_observed_unique_tick/visit` locate the first **encounter** with a unique
maximum. Preference formation may precede its next encounter; its exact continuous
formation time is not observed from decision summaries alone.

## Identifiability and metric definitions

A unique maximum means recorded `max_ties == 1`, independently of using gap as
an identity proxy. The recorded top-two Q gap is numerical strength; zero means
tied maxima, not necessarily an unchanged policy. Ties use the learner's exact
floating-point equality, without new tolerances.

The streams contain summary Q values, selected action/Q, exploration branch,
SILENCE membership and tie count, **not complete Q vectors or an explicit list of
maximal actions**. Exact identity uses only these current-encounter facts:

- Unique SILENCE is identified by membership and cardinality, regardless of the
  sampled action. Otherwise a unique maximum is identified when selected Q equals
  maximum Q: exploitation always meets that equality; exploration only sometimes.
- An exploring nonmaximal signal choice generally leaves the unique preferred
  signal unknown. Its ID is blank, not copied from a preceding encounter.
- Tied sets are not generally known from one choice. Limited complete sets follow
  from exact cardinality/membership constraints: all actions tied; all except a
  known unique nonmaximum; or enough observed members/complement members to exhaust
  the set. For example, tied SILENCE plus a different selected maximum identifies
  both members only when the recorded tie count is exactly two.
- IDs and sets are not imputed across time, inferred from gap magnitude or
  propagated through TD logs. No missing action values are invented.

Preferred-action persistence divides identical, identifiable **unique** endpoint
pairs by identifiable unique endpoint pairs. Coverage divides that denominator
by **all retained pairs**; coverage given both maxima unique is separate.
Choice persistence compares sampled action indices, separately for all pairs,
both exploited and at least one explored. It is behavioral choice persistence,
not policy persistence. Tied exploitation can sample different actions.

Jaccard and policy TV are conservatively reported only when complete endpoint
maximum sets A/B are identified (epsilon=1 additionally proves uniform-policy
TV=0). With constant epsilon/action count:

```text
J(A,B) = |A intersect B| / |A union B|
TV = (1-epsilon) * (1 - |A intersect B| / max(|A|, |B|))
```

These are identifiable-subset estimates, not full-corpus policy stability.
Unknown sets generally prevent exact TV; the reported estimator does not attempt
additional special-case overlap deductions from incomplete sets. Changing tie
cardinality with epsilon<1 proves a **binary policy change**, even if its members
and exact TV are unavailable. Known same/changed/unknown policy partitions are
therefore different from the complete-set/TV coverage. A behavioral change with
unknown policy is not recorded as “unchanged policy.”

SILENCE has three exact categories: unique maximum, tied maximum, not maximal.
Transition rates condition on the previous category, within retained pairs.
Their coverage is complete for these summaries; stable SILENCE probability or
category does not establish a stable complete vocal policy or cooperation.

## Statistical methods and selection

Runs are summarized first. C/D summaries give every seed equal weight, sample SD
(ddof=1), finite n and encounter/pair/episode/agent exposures. Millions of repeated
observations are not independent experimental replicates. Every rate keeps its
own denominator; no division by all decisions for identity persistence. Undefined
estimates are blank CSV/null JSON, not zero. Known encounter/experience categories
and SILENCE destinations have explicit zero counts within observed support; a
zero origin exposure still gives an undefined transition rate. This prevents
positive-only seed averages for rare transitions. No missing windows, measurements or
post-extinction values are filled. Windows are `(0,100]`, …, `(900,1000]`, with
nominal/actual endpoints, observed ticks and complete/partial flags.

Window attribution uses the second encounter of a retained pair; the earlier
encounter may be in an earlier window. Age profiles use second-encounter age,
25-tick bins. Visit-index CSVs distinguish within-episode from lifetime state
visits; graphs show only the former, first 20 visits. Later visits select heavily
reused states; declining exposures remain explicit.

Paired development compares the first three and last three encounters of the
same episode, with at least six encounters (no overlap). It is separate from
cross-sectional age profiles and does not establish a causal age effect.
Experience bins use cumulative Vocal Q updates at choice: low <10, middle 10–99,
high >=100. Age/update quantiles weight actor decisions within a seed. Lifecycle
first appearances/deaths and window shared/new/all cohorts are separate counts.
A first-time learner means its first recorded decision, not a birth event; queued
newborns that have not acted are excluded. Active-window participants include
agents that die, so this is not a final living-population snapshot.

Identity graphs display only seed/windows with at least 20 identifiable pairs,
using markers/error bars and an explicit coverage panel. Individual trajectories
use the lowest eligible seed per condition and the two highest-encounter episodes,
breaking ties by agent/state/episode. No outcome-based selection or random jitter.

## Tests and independent verification

- Baseline: **317 tests, OK, 62.535 s**.
- Final: **360 tests, OK, 57.253 s** (43 new tests).
- Synthetic A–J cover stable identity, one switch with unchanged gap, exploration,
  tied maxima, actual eviction/recreation, death, turnover, unknown identity,
  unequal seed exposure and deterministic output. Additional tests cover exact
  membership deductions, cardinality-only change, update counts, age, visit
  indices, censored storage, empty/partial/missing inputs, manifests/hashes,
  unsafe historical joins and output protection.
- Two analyses of the same synthetic inputs produce byte-identical CSV/gzip,
  JSON and all ten PNGs. Python/NumPy global RNG and input hashes remain unchanged.
  The analyzer is tested with `Simulation.step` patched to fail if called.
- Independent pandas same-state shifts on exposure-selected, non-evicted real
  agents match all pair/identity/choice counts: C seed0 agent21 (608 pairs,
  117 identifiable, 115 matches); D seed0 agent31 (323/80/80).
- Every episode endpoint, decision partition, actual eviction, identity coverage,
  policy-status/behavior-status partition and per-window exposure reconciles.
  All historical fixtures and analyses remain unchanged. No fixture was regenerated.

## Per-seed results

There are **4,209,670 encounters**, **1,642,843 retained
pairs**, and **2,566,827 allocation episodes**. Only retained
same-individual/state/episode pairs enter primary persistence rates.

| Run | Retained pairs | Identifiable unique pairs | Same preferred | Identity coverage % | Preferred match % | Sampled choice match % |
| --- | --- | --- | --- | --- | --- | --- |
| C_seed_0 | 339939 | 10131 | 9892 | 2.980 | 97.641 | 6.634 |
| C_seed_1 | 338914 | 10475 | 10081 | 3.091 | 96.239 | 6.676 |
| C_seed_2 | 334566 | 11228 | 10889 | 3.356 | 96.981 | 6.989 |
| D_seed_0 | 209289 | 7499 | 7288 | 3.583 | 97.186 | 7.581 |
| D_seed_1 | 210479 | 7232 | 7046 | 3.436 | 97.428 | 7.460 |
| D_seed_2 | 209656 | 7551 | 7316 | 3.602 | 96.888 | 7.671 |

Equal-seed mean ± sample SD, n=3 throughout:

| Metric | C | D |
| --- | --- | --- |
| Preferred-action match %, conditional | 96.9534 ± 0.701519 | 97.1674 ± 0.270629 |
| Identity coverage % | 3.14233 ± 0.193111 | 3.54022 ± 0.0907583 |
| All sampled choices match % | 6.76621 ± 0.193852 | 7.57088 ± 0.105897 |
| Both-exploit sampled choices match % | 7.68088 ± 0.293487 | 8.79584 ± 0.153212 |
| At least one explored: choices match % | 5.1397 ± 0.0809403 | 5.39599 ± 0.0928836 |
| Complete-set/TV coverage % | 7.65554 ± 0.169458 | 8.83112 ± 0.122208 |
| Jaccard, complete-set subset | 0.825198 ± 0.00359461 | 0.807577 ± 0.00172729 |
| Exact policy TV, complete-set subset | 0.139841 ± 0.00287569 | 0.153938 ± 0.00138183 |
| Known same policy, all retained pairs % | 3.06532 ± 0.184297 | 3.46061 ± 0.0772947 |
| Provably changed policy % | 94.3565 ± 0.312329 | 93.9454 ± 0.168284 |
| Unknown policy status % | 2.5782 ± 0.156585 | 2.59396 ± 0.0915503 |
| Mean revisit interval, ticks | 8.44102 ± 0.321279 | 8.8113 ± 0.146298 |
| Paired same-episode late − early gap | 0.00734441 ± 3.96672e-05 | 0.0105971 ± 0.000182815 |

**Coverage is central.** About 97% of identifiable unique preferences persist,
but only 3–4% of all retained pairs identify both preferred actions. This is not
“97% of policies are stable.” The subset requires unique maxima and sufficient
observed membership information, preferentially selecting already strong states.
Most pairs involve ties. At the same time about 94% of retained pairs provably
change epsilon-greedy policy, often through changed maximum-set cardinality;
the exact size of those changes is mostly unreported by the conservative
complete-set estimator. Stable unique identity and shifting tied policies coexist.

Only about 0.11–0.12% of sampled action changes are certified to occur with an
unchanged policy in the recorded subset. Many policy magnitudes/identities remain
unknown. Synthetic controls prove exploration/tie sampling can change choices
without policy changes; the ecological data must not be explained entirely by
sampling. No causal mechanism or semantics follows from either statistic.

### Formation, reuse and storage

| Run | Episodes | One encounter | Ever observed unique maximum | Eligible paired episodes | Median storage ticks | Median encounter span |
| --- | --- | --- | --- | --- | --- | --- |
| C_seed_0 | 354570 | 184752 | 10344 | 10130 | 65.0 | 0.0 |
| C_seed_1 | 349601 | 182235 | 10422 | 10220 | 64.0 | 0.0 |
| C_seed_2 | 344648 | 179724 | 10363 | 10221 | 65.0 | 0.0 |
| D_seed_0 | 505223 | 388804 | 7465 | 4652 | 66.0 | 0.0 |
| D_seed_1 | 508815 | 391259 | 7571 | 4766 | 66.0 | 0.0 |
| D_seed_2 | 503970 | 387420 | 7622 | 4819 | 66.0 | 0.0 |

Over half of C episodes and roughly 77% of D episodes have only one encounter.
Median storage lasts 64–66 ticks while median encounter span is zero: storage
is not repeated reinforcement. First-visit gap is zero in all six runs; gaps grow
with observed reuse. At visit 20, mean gap is about 0.030–0.038 (C, 237–320
encounters/seed) and 0.031–0.043 (D, 178–191). Later-visit composition is selected.
Paired first/last-three comparisons are positive in all six runs: C +0.007344
±0.000040, D +0.010597 ±0.000183. Only about 10,130–10,221 C episodes and
4,652–4,819 D episodes qualify; this does not describe every allocated row.

### Eviction and turnover

Across all six runs there are 26 C and 161 D post-eviction revisits, versus
1,013,419 C and 629,424 D retained pairs. These recreated visits are separate
forgetting/relearning episodes, not uninterrupted persistence. Mean post-eviction
share of all revisits is about 0.00256% (C) and 0.02555% (D). Actual row evictions
remain 394/252/309 (C), 3469/4222/3365 (D), matching v0.6.2. Within-retained
identity switches cannot be attributed to an allocation break by construction;
unvisited evicted rows' counterfactual value is not measured.

| Run | Early actors (first observed by tick100) | Dead by end | Alive at end | Seen in final window |
| --- | --- | --- | --- | --- |
| C_seed_0 | 208 | 208 | 0 | 0 |
| C_seed_1 | 194 | 194 | 0 | 0 |
| C_seed_2 | 190 | 189 | 1 | 4 |
| D_seed_0 | 208 | 208 | 0 | 0 |
| D_seed_1 | 206 | 206 | 0 | 0 |
| D_seed_2 | 207 | 207 | 0 | 1 |

Only one early actor remains alive at tick1000; a few more act and die in the
final window. Replacement is extensive. Actual death dates, absent-but-alive
counts and all/shared/new window cohorts are retained in CSVs; missing a state
encounter is never interpreted as death.

The simple “more inexperienced agents cause the decline” account is not
supported by the chosen cumulative-update metric. Decisions with <10 updates
fall from about 13.5–14.3% early to 7.2–7.6% late. Mean decision age/updates rise
from roughly 43 to 90–101. Nevertheless newly allocated-row decision share rises
from 27.7–28.9% to 51.9–52.6% (C), and 41.6–42.7% to 72.1–72.6% (D).
Agent experience, fresh state exposure and population replacement are distinct.
Their causal contributions to MI have not been separated.

### SILENCE and source MI

SILENCE remains a unique maximum on the next retained encounter in 88.32% (C)
and 89.74% (D) of unique-origin pairs. Origin exposures are 915–1003 and 595–619
pairs/seed. Tied-maximum persistence is 90.83%/90.34%; not-maximal persistence
97.34%/97.50%. These are category transition probabilities, not a communication
or cooperation measure, and do not identify every other maximal action.

Corrected within-agent V/food MI is reused exactly from v0.6.1: mean 0.132478 →
−0.022982 bits in C and 0.095352 → −0.010216 in D, from first to final window.
Meanwhile identifiable preferred-action matching is high in both windows and
coverage falls from about 17% to about 1.1–1.3%. Raw within-agent MI rises rather
than following its corrected counterpart. Raw finite-sample association and
permutation-corrected association must not be conflated.

The join binds all four diagnostic hashes/sidecar, full Config/horizon/statistics,
simultaneous communication CSV/metadata hashes and the archived analysis input
identity. Historical result hashes are checked against v0.6.1/v0.6.2 verification
records. Five source features, raw and corrected MI, original eligible-agent and
decision counts, mean individual vocal/emission entropy, old hit/eviction rates,
and first/last observed emission-distribution JSD are preserved. No permutation
analysis is rerun. Matching seed alone is never sufficient.

MI uses actual frequency samples, including SILENCE for V; source within-agent
eligibility is 50 decisions/window and exposure-weighted conditioning within a
run. Mean individual entropy in the comparison instead averages eligible agents
within run; both counts remain explicit. Observed emitted-ID first/last JSD uses
the source's eligibility/windows and is not a Q-policy distance. The individual
comparison labels those source windows; its Q persistence summarizes that agent's
whole observed lifetime. Neither coincidence of trends nor this different-scope
comparison identifies an explanation for MI decline. Temporal/confounded exposure,
selective survival and nonunique policy changes remain unresolved.

## Output schemas and figures

All exact headers are also saved in `analysis_metadata.json.schemas`. Types are
integer IDs/ticks/counts, categorical strings, boolean flags and floating summaries;
undefined measurements are empty/null. Shared header groups, in order:

```text
C = run,condition,seed
W = window_start,window_end,actual_window_end,observed_ticks,partial_window,complete_window
M = decisions,first,retained,recreated,unique,identified_unique,pairs,choice_same,exploit_pairs,exploit_same,explore_pairs,explore_same,both_unique,id_pairs,id_same,set_pairs,set_same,tv_pairs,choice_changed,changed_stable,changed_policy,changed_unknown,unique_to_tied,tied_to_unique,known_policy_same,known_policy_changed,unknown_policy,unique_max_fraction,mean_gap,mean_p_silence,observed_silence_rate,choice_match_rate,exploit_choice_match_rate,explore_choice_match_rate,preferred_action_match_rate,identity_coverage,identity_coverage_given_unique,max_set_coverage,mean_max_set_jaccard,exact_policy_coverage,mean_exact_policy_tv,mean_gap_change,mean_abs_gap_change,mean_revisit_interval,mean_agent_updates_between,mean_state_updates_between,mean_p_silence_change,mean_abs_p_silence_change,known_policy_same_fraction,known_policy_changed_fraction,unknown_policy_fraction
```

M contains encounter/pair exposures and ratios. `first/retained/recreated`
partition encounters. `pairs` counts incoming retained comparisons, except grouped
recreated/first rows where it is zero. `id_pairs/id_same` are identifiable unique
endpoint counts. `both_unique` includes unidentified identities. `set_pairs/set_same`
use complete known maximum sets. `known_policy_same/changed/unknown` partition
all pairs using complete sets or cardinality proof. `changed_stable/changed_policy/
changed_unknown` partition only behavioral changes. `unique_to_tied/tied_to_unique`
record maximum-status loss/formation, not the unobserved identity of every tied action.

Every M rate is computed as follows (sums refer to the recorded endpoint quantities):

| Rate/mean | Numerator | Denominator |
| --- | --- | --- |
| unique_max_fraction | unique | decisions |
| mean_gap | gap_sum | decisions |
| mean_p_silence | p_sum | decisions |
| observed_silence_rate | silent | decisions |
| choice_match_rate | choice_same | pairs |
| exploit_choice_match_rate | exploit_same | exploit_pairs |
| explore_choice_match_rate | explore_same | explore_pairs |
| preferred_action_match_rate | id_same | id_pairs |
| identity_coverage | id_pairs | pairs |
| identity_coverage_given_unique | id_pairs | both_unique |
| max_set_coverage | set_pairs | pairs |
| mean_max_set_jaccard | jaccard_sum | set_pairs |
| exact_policy_coverage | tv_pairs | pairs |
| mean_exact_policy_tv | tv_sum | tv_pairs |
| mean_gap_change | delta_gap_sum | pairs |
| mean_abs_gap_change | abs_delta_gap_sum | pairs |
| mean_revisit_interval | elapsed_sum | pairs |
| mean_agent_updates_between | agent_updates_sum | pairs |
| mean_state_updates_between | row_updates_sum | pairs |
| mean_p_silence_change | delta_p_sum | pairs |
| mean_abs_p_silence_change | abs_delta_p_sum | pairs |
| known_policy_same_fraction | known_policy_same | pairs |
| known_policy_changed_fraction | known_policy_changed | pairs |
| unknown_policy_fraction | unknown_policy | pairs |

`gap_sum`, `p_sum`, `silent` sum decision gap, implied probability and observed
SILENCE indicators. Pair sums use next−previous gaps/probabilities, absolute changes,
actual elapsed ticks, cumulative agent-update differences and actual episode-update
differences. Jaccard/TV sums use only their known-set subset. Within visit/age/window
rows pair metrics describe pairs whose second encounter falls in that group.

Exact per-table headers by composition (C/W/M expand as above):

| File | Columns in order |
| --- | --- |
| `persistence_run_summary.csv` | C + M, agents, episodes, actual_evictions, deaths, run_ends, eligible_paired_episodes, mean_paired_gap_change, mean_paired_p_silence_change |
| `persistence_by_window.csv` | C + W + M |
| `persistence_by_episode.csv.gz` | C + agent_id, state_id, episode, allocated_tick, end_tick, end_reason, storage_duration, first_encounter_tick, last_encounter_tick, encounter_span, first_age, encounters, row_updates, retained_pairs, mean_interval, max_interval, unique_max_fraction, mean_gap, mean_p_silence, observed_silence_rate, choice_match_rate, id_pairs, id_same, identity_coverage, preferred_action_match_rate, set_pairs, mean_max_set_jaccard, mean_exact_policy_tv, eligible_early_late, early_encounters, late_encounters, early_mean_gap, late_mean_gap, delta_gap, early_mean_p_silence, late_mean_p_silence, delta_p_silence, first_observed_unique_tick, first_observed_unique_visit |
| `persistence_by_agent.csv` | C + agent_id, generation, first_tick, last_tick, first_age, last_age, death_tick, end_reason, episodes, actual_evictions, M |
| `persistence_by_agent_window.csv` | C + W + agent_id, M |
| `persistence_by_visit_index.csv` | C + visit_kind, visit_index, M |
| `persistence_by_age.csv` | C + age_start, M |
| `persistence_by_experience.csv` | C + update_bin, M |
| `paired_episode_development.csv` | C + first_age_start, episodes, mean_gap_change, mean_p_silence_change |
| `preference_transitions.csv` | C + W + branch, from_action, to_action, count, branch_pairs, window_id_pairs, probability |
| `silence_preference_transitions.csv` | C + W + from_category, to_category, count, origin_pairs, probability, retained_pairs |
| `retained_vs_recreated.csv` | C + W + encounter_class, M |
| `state_revisit_intervals.csv` | C + elapsed_ticks, count, retained_pairs, probability |
| `agent_turnover.csv` | C + W + active_agents, first_time_learners, first_time_founders, newly_appearing, shared_with_previous, previous_agents_absent, absent_dead_by_window_end, absent_alive_at_window_end, actual_deaths, previous_cohort_deaths, mean_age, median_age, p10_age, p90_age, mean_updates, median_updates, p10_updates, p90_updates, low_experience_decision_share, high_experience_decision_share, newly_allocated_decision_share, retained_row_decision_share |
| `persistence_by_cohort.csv` | C + W + cohort, agents, M |
| `persistence_seed_summary.csv` | table, condition, window_start, window_end, actual_window_end, visit_kind, visit_index, age_start, update_bin, first_age_start, encounter_class, cohort, from_category, to_category, metric, mean, std, min, max, n |
| `state_dictionary.csv` | run, condition, seed, state_id, key_json |
| `persistence_mi_comparison.csv` | run, condition, seed, window_start, window_end, actual_window_end, observed_ticks, partial_window, complete_window, decisions, first, retained, recreated, unique, identified_unique, pairs, choice_same, exploit_pairs, exploit_same, explore_pairs, explore_same, both_unique, id_pairs, id_same, set_pairs, set_same, tv_pairs, choice_changed, changed_stable, changed_policy, changed_unknown, unique_to_tied, tied_to_unique, known_policy_same, known_policy_changed, unknown_policy, unique_max_fraction, mean_gap, mean_p_silence, observed_silence_rate, choice_match_rate, exploit_choice_match_rate, explore_choice_match_rate, preferred_action_match_rate, identity_coverage, identity_coverage_given_unique, max_set_coverage, mean_max_set_jaccard, exact_policy_coverage, mean_exact_policy_tv, mean_gap_change, mean_abs_gap_change, mean_revisit_interval, mean_agent_updates_between, mean_state_updates_between, mean_p_silence_change, mean_abs_p_silence_change, known_policy_same_fraction, known_policy_changed_fraction, unknown_policy_fraction, feature, mi_decisions, eligible_agents, raw_within_agent_mi_bits, corrected_within_agent_mi_bits, mean_individual_vocal_entropy, mean_individual_emitted_entropy, entropy_agents, entropy_decisions, v062_state_hit_rate, v062_eviction_rate |
| `persistence_individual_comparison.csv` | run, condition, seed, agent_id, generation, first_tick, last_tick, first_age, last_age, death_tick, end_reason, episodes, actual_evictions, decisions, first, retained, recreated, unique, identified_unique, pairs, choice_same, exploit_pairs, exploit_same, explore_pairs, explore_same, both_unique, id_pairs, id_same, set_pairs, set_same, tv_pairs, choice_changed, changed_stable, changed_policy, changed_unknown, unique_to_tied, tied_to_unique, known_policy_same, known_policy_changed, unknown_policy, unique_max_fraction, mean_gap, mean_p_silence, observed_silence_rate, choice_match_rate, exploit_choice_match_rate, explore_choice_match_rate, preferred_action_match_rate, identity_coverage, identity_coverage_given_unique, max_set_coverage, mean_max_set_jaccard, exact_policy_coverage, mean_exact_policy_tv, mean_gap_change, mean_abs_gap_change, mean_revisit_interval, mean_agent_updates_between, mean_state_updates_between, mean_p_silence_change, mean_abs_p_silence_change, known_policy_same_fraction, known_policy_changed_fraction, unknown_policy_fraction, first_window_start, first_window_end, last_window_start, last_window_end, first_emitted, last_emitted, eligible_windows, observed_emitted_first_last_jsd_bits |
| `individual_episode_trajectories.csv` | run, condition, seed, agent_id, state_id, episode, tick, age, gap, p_silence, selected_action, exploratory, preferred_action |

`persistence_seed_summary` is tidy: table/scope keys, metric, equal-seed mean,
sample SD/min/max and finite n. Some scope columns are blank for other tables.
Episode detail streams directly to deterministic gzip (mtime=0); no full Q vectors
or per-pair intermediate table is stored. Whole episodes use actual termination
counters; no LRU inference from occupancy. The state dictionary is canonical,
run-local and one-to-one. Input files are never modified.

Ten 180-dpi PNGs (C blue, D orange) were visually checked:

| Figure | Meaning |
| --- | --- |
| [preference_persistence_over_time.png](../analysis/vocal_persistence/preference_persistence_over_time.png) | Conditional unique-identity matching and coverage; no misleading continuous identity line. |
| [choice_consistency_over_time.png](../analysis/vocal_persistence/choice_consistency_over_time.png) | All/both-exploit/at-least-one-explored sampled matches. |
| [preference_gap_by_visit.png](../analysis/vocal_persistence/preference_gap_by_visit.png) | Episode visit index, gap and shrinking encounter exposure. |
| [state_revisit_intervals.png](../analysis/vocal_persistence/state_revisit_intervals.png) | Actual tick intervals, log2 bins, equal-seed frequencies. |
| [silence_preference_transitions.png](../analysis/vocal_persistence/silence_preference_transitions.png) | Three-category transitions with seed/origin-pair exposure labels. |
| [preference_stability_by_age.png](../analysis/vocal_persistence/preference_stability_by_age.png) | Cross-sectional conditional age profile beside paired episode development. |
| [retained_vs_recreated.png](../analysis/vocal_persistence/retained_vs_recreated.png) | Post-eviction revisit proportion and sparse absolute exposures. |
| [agent_turnover_over_time.png](../analysis/vocal_persistence/agent_turnover_over_time.png) | Active-window participants, first observations and actual deaths. |
| [persistence_vs_sender_mi.png](../analysis/vocal_persistence/persistence_vs_sender_mi.png) | Verified raw/corrected MI and matching-window conditional persistence; no causal fit. |
| [individual_preference_trajectories.png](../analysis/vocal_persistence/individual_preference_trajectories.png) | Exposure-selected same-state episodes: gap, SILENCE probability, sampled and identifiable actions. |

## Storage, performance and commands

```bash
python3 -m experiments.analyze_vocal_persistence
# Equivalent direct invocation, with explicit paths:
python3 experiments/analyze_vocal_persistence.py \
  --data-dir data/vocal_learning --output-dir analysis/vocal_persistence
```

Default historical paths are `analysis/communication`, `analysis/vocal_learning`
and `data/v061_verification.json`, `data/v062_verification.json`; the CLI exposes
all four. `--no-mi` explicitly omits historical comparisons for synthetic/other
complete data. Missing required files cause an error; the analyzer never substitutes
new simulations. Options affect analysis only, not Config or original schemas.

The final analysis took **153.025 s** locally (the first completed
pass took 152.117 s). Total output is **108,146,401 bytes
(103.136 MiB), 20 CSVs (one gzipped), ten PNGs and
one metadata file. The episode detail is about 48.6 MiB compressed. Raw inputs
stream by tick/chunk; active episode summaries and living-state histories are
kept, plus small agent/window/visit aggregates. Peak active rows are recorded
per run in metadata. Memory does not retain full decision histories or Q tables;
summary memory still scales with agents/windows and active state histories.

These are local wall times, not scaling guarantees. No simulation-instrumentation
overhead is introduced because no simulator code changes. No new ecological runs
or expensive permutation analysis occurred. Reused results retain original hashes.
Machine-readable evidence is in `data/v063_verification.json`, `data/v063_summary.json`,
`data/v063_independent_checks.json`, `data/v063_episode_checks.json`, test logs,
analysis timing and the output metadata. Results use the existing local analysis/
data ignore policy and remain saved on disk.

## Scientific answers

1. **Same preferred action on revisit?** On the identifiable unique subset,
   C 96.95% ±0.70 percentage points and D 97.17% ±0.27. This is conditional
   persistence, not a 97% stability claim for all policies.
2. **Exact identity coverage?** C 3.142% ±0.193 points, D 3.540% ±0.091 of
   retained pairs; both-unique conditional coverage is only about two thirds.
   Unidentified IDs remain blank. Complete named-set coverage is 7.66%/8.83%.
3. **Behavior changes without identified policy changes?** Choice matching is
   6.77%/7.57%; both-exploit matching is 7.68%/8.80%. Known unchanged-policy
   behavioral changes exist but are rare (~0.11–0.12% of changed-choice pairs).
   Unknown is not unchanged. Cardinality changes certify many policy changes
   without establishing their exact TV or signal identities.
4. **Strength versus repeated visits?** Gaps increase in visit-index profiles
   and paired same-episode comparisons in all six runs, with sharply shrinking
   exposure. Most allocated episodes are never revisited; late reuse is selected.
5. **Is D less persistent?** Not consistently in the identifiable subset.
   D−C changes sign by seed; its mean is slightly higher. D has fewer retained
   encounters overall. Different coverage/context prevents a causal comparison.
6. **LRU contribution?** Recreated visits are rare and explicitly excluded from
   continuous persistence. Retained identity switches occur without eviction;
   no counterfactual cost of never-revisited evicted rows is identified.
7. **Turnover contribution?** Replacement is extensive; only one early actor
   survives to the endpoint. But low-update decision share decreases, while fresh
   row exposure rises. No numerical causal decomposition of MI/instability follows.
8. **SILENCE persistent?** Unique-maximum status stays unique about 88.32%/89.74%
   on retained encounters, with exposures shown. Tied/not-max categories also
   persist often. This is not an inferred communicative or cooperative meaning.
9. **MI decline associated with persistence changes?** Corrected MI declines
   while conditional identity matching remains high and coverage collapses.
   Stronger same-row preferences, turnover and new-state exposure coexist.
   The data do not establish generalized loss of preferred action as the cause.
10. **Unanswerable?** Most full preferred sets/policy distances, hidden Q vectors,
    precise unobserved formation/switch times, counterfactual retention effects,
    age/ecology/turnover causal shares and receiver use/semantics remain unknown.
    Observed emission entropy/JSD and stable gap/SILENCE are not policy identities.

**Observed:** conditional unique identity is persistent; ties/policy cardinality
change often; coverage and reuse are limited; replacement is extensive.
**Plausible explanations:** sparse state reuse, shared-reward updates, changing
state exposure and cohort composition may jointly explain population patterns.
**Causal conclusions:** none about semantic meaning, communication benefit,
cooperation or language; no mechanism was experimentally changed.

## Future directions, documentation only

First use the existing recordings for context/cohort-matched same-state episode
comparisons and conditional preference-switch analysis. If complete-set persistence
later becomes necessary, a minimal future optional recorder could intern exact
maximum-action sets and emit their set IDs under a new diagnostic schema. It would
need no full Q vector or RNG, but is **not implemented** here; no pilot was rerun
to fill missing information.

Future self-generated masking is a separate research direction. Currently an
agent's own signals are excluded, perception receives previous-tick emissions,
and emission occurs after physical action. A self-masking model must explicitly
define which speaking interval overlaps which listening interval, including
self-produced versus external acoustic competition. Sensory interference could
create a cost of speaking without a negative speaking reward, but no advantage
of SILENCE should be assumed. No self-masking, SILENCE bias, emission timing,
reward, cost or acoustic mechanism changes occur in v0.6.3.

## Files changed

Added five analysis modules: `experiments/analyze_vocal_persistence.py`,
`experiments/vocal_persistence.py`, `experiments/vocal_persistence_data.py`,
`experiments/vocal_persistence_metrics.py`, `experiments/vocal_persistence_plots.py`;
synthetic helpers/tests `tests/vocal_persistence_fixtures.py`,
`tests/test_vocal_persistence.py`; and this report. Updated README and .gitignore.
All simulator/learning/recording code, Config, historical reports/analyses and
fixtures remain unchanged. This release stops here; it does not implement v0.7.
