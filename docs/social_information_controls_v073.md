# v0.7.3 — Social Information Value and State Fragmentation Controls

## Objective and inspected implementation

Question: are the behavioral/ecological changes from social information
distinguishable from similarly fragmented independent non-social inputs?
Policy differences alone do not establish useful learning or survival benefit.
No social/vocal reward, preference, new ecology, semantics or language is added.

Before edits, the repository, README, tests, v0.7.2 report and experiments were
inspected at commit `cf64a6da956a98d59966307b31501b7fc6edfd69`. The existing 554
tests passed. A SHA-256 inventory of 579 existing source/historical files was
saved in `data/v073_before.json`; a separate untouched baseline code archive was
saved for compatibility checks. The implementation inspected was:

1. Regenerate resources, shuffle existing actors. One sequential raw visual
   observation is acquired for the actor. Coarse B/C information is frozen before
   this sighting updates long-term memory. Births do not act during their birth tick.
2. LearningBrain updates auditory memory and encodes the original physiological,
   visible resource and auditory state. B appends one social integer; C appends
   the existing `(visible, remembered_direction)` pair (37 valid combinations).
   The original key is used unchanged for Vocal Q.
3. Physical Q reads the next row's bootstrap **before** updating/recreating its
   previous row. It then allocates/refreshes the current row and makes the same
   epsilon-greedy choice with uniform ties and zero initial values. Both bounded
   Q LRUs retain capacity 256. Terminal updates happen immediately; surviving
   updates wait for that individual's next actual observation.
4. Actual actions, reproduction, emissions, physiology, reward and optional
   diagnostics follow. Death removal, newborn insertion, communication advance
   and statistics retain their existing order. No learning core was rewritten.
5. Existing v0.7.2 passive diagnostics observe actual movement against a frozen
   visible target, contacts, memory, auditory masking, vocal choices and retained
   trained Q rows. Their grouped action cells lack temporal token order and
   per-decision outcomes. Contact episode lengths are not token run lengths.

## Conditions and nuisance process

| Condition | Extra Physical Q input | Structural multiplier |
|---|---|---:|
| A | None, original key | 1 |
| B | Existing visible count/direction | 11 |
| C | Existing visible + remembered direction | 37 |
| D | Independent stationary nuisance token | 11 |
| E | Independent stationary nuisance token | 37 |

All conditions retain passive visual memory capacity 32 and legacy formation/LRU
policy. D/E never put that memory or live social features into either Q head.
The maximum category count is structural, not a claim about occupied categories.

`NuisanceModel(probabilities, hold_probability)` and
`NuisanceConfig(model, seed)` are separate frozen experiment configurations;
the original Config schemas are unchanged. A dedicated master Random seeded with
`v073:nuisance:{nuisance_seed}:{category_count}` supplies 128-bit seeds to private
channels at **brain construction**, including offspring. No identity, age, tick,
position, observation, resource, action, reward or future event enters this API.
Construction order assigns exchangeable draws; it does not encode an identity.
Each channel begins with a stationary categorical draw pi, then at each own
decision opportunity holds with probability alpha, else independently resamples
pi. Thus `P(same)=alpha+(1-alpha)*sum(pi**2)`. Per-category runs are geometric;
arbitrary heavy tails and X-conditioned social distributions cannot generally
be reproduced. The first actual opportunity uses the stationary initial draw.
Read-only observation/probes make no draws. Main streams are never consumed by
the nuisance generator. Policy divergence can naturally change later ordinary
tie-breaking draws; that is a representation effect, not RNG sharing.

Parameters never receive ecological feedback. Nevertheless, a policy responding
to an exogenous token can make token/body/survival distributions empirically
associated. Independent innovations alone are not an assurance of zero predictive
association in a selected online population.

## Calibration and frozen acceptance rules

The design and tolerances were recorded before the new online experiments in
`data/v073_design.json`. Archived v0.7.2 B/C seeds 0/1/2 provide marginal and
original-state conditional category frequencies and fragmentation. Missing token
order, persistence and actual decision rewards are explicitly NA in that audit.
Additional B/C calibration recordings use separate ecological seeds **3 and 4**,
500 ticks. Model selection reads only actual token/order/original-state labels;
not actions, rewards, population, births, deaths or survival performance.

pi is the equal-calibration-seed marginal distribution. Candidate alpha values
are the empirical same-token estimate plus/minus 0.1 and 0.2 within [0,1), and
zero. Fixed calibration nuisance seeds 11/29 generate independent tokens on the
same calibration original-state sequences. The predeclared feature-only score
combines first-state fraction, relative mean visits, entropy and persistence
errors scaled by their tolerances. Models and source hashes are frozen before
evaluation. No test seed ecology or reward is used to select them.

| Diagnostic target | Absolute tolerance |
|---|---:|
| First-state fraction | 0.02 |
| Mean visits per agent-state | 10% relative |
| Token entropy | 0.25 bits |
| Conditional token entropy given original X | 0.25 bits |
| Same-token transition probability | 0.05 |

Gross mismatch thresholds (first-state 0.05, token entropy 0.75 bits, persistence
0.15) block the primary experiment. Moderate mismatches still require an explicit
review and exploratory classification. Parameters are not retuned using ecological
performance. Conditional tables preserve support counts (min20 annotated), and
unconditional plug-in entropy is not an unbiased finite-sample information claim.

## Measurement, prediction and replay definitions

`ControlsObserver` streams original keys and one actual ordered outcome per
decision: sequence, tick, agent, original-state ID, passive B/C tokens, actual
used token, chosen action, exploration, existing physiological reward, terminal.
The stream is optional and never exposed to the learner. Hooks also preserve
lifetime state visits/updates and current-residence retained-row visit/update
histograms. Histories are external scalar diagnostics for living agents and are
released on death/end; no whole dead bodies/Q tables are kept.

First-state fraction = distinct lifetime agent-state pairs / actual decisions.
States-once fraction divides by distinct pairs. Q occupancy averages actual
decision-time table size. Evictions and updates count actual hooks, including
previous-row recreation. Token persistence uses consecutive **own observation
opportunities**; final/death segments are right-censored and separately marked.
Retained probes require >=2 or >=5 current-row visits and >=1 current-row update,
hold exact original state fixed and never synthesize absent rows. Unsupported
comparisons are NA, not zero dependence.

Predictive comparison trains on ecological seeds 3/4 and tests 0/1/2. On the SAME
actual B or C experience, it compares X,A; X,S,A; X,U,A. U is independently attached
in recorded decision order; individual IDs are ownership lookup keys only, never
seeds. Target = actual immediate physiological reward in bins <=-0.1, (-0.1,0),
>=0, conditioned on actual chosen action. At least 20 training observations per
cell are required; Dirichlet strength 5 smooths toward the action-specific prior.
Unseen/sparse cells back off to that prior. Held-out log loss/Brier, coverage,
decision/agent counts and equal-agent log loss are reported by ecological seed.
No repeated decision is treated as an independent experimental replicate.
Sparse finite-sample conditional mutual information is not forced into a number.

Fixed-experience replay, if primary review permits, reads the next actual row of
the same individual to supply the bootstrap. It preserves read-before-update,
zero priors, LRU allocation order, actual actions, actual reward and terminal
updates. At the run boundary pending survivor transitions remain unupdated.
Source-representation replay must reproduce the physical Q/LRU/pending digest
exactly. Alternative encodings demonstrate representation effects on the same
experience; they cannot establish ecological counterfactual outcomes or unbiased
policy improvement.

## Execution and results

### Scope, compatibility and gate

The numerical record is in `analysis/social_learning_v073/`. The reviewed pilot
is `pilot_reviewed/` (33 files: 17 CSVs, 14 PNGs, review and manifest).
The initial `pilot/` analysis is preserved as an earlier review artifact; its
matching-quality panel showed only three metrics. The reviewed panel includes
entropy failures explicitly. `prediction/` preserves the initial all-decision
prediction; `prediction_support/` adds a clearly labelled exploratory subset
diagnostic and tidy summaries. Both retain actual per-seed values/denominators.

Executed:

* Four calibration recordings: B/C, ecological seeds 3/4, 500 ticks.
* Nine necessary instrumentation-only references: A/B/C, seeds 0/1/2, 500 ticks.
  All nine matched archived v0.7.2 full fingerprints at 0/100/200/300/400/500,
  action checkpoints, aggregate statistics and final ecological digests exactly.
  These fingerprints include bodies, genomes, world, physical/vocal Q tables,
  LRU order, pending transitions, memory and main/brain RNG states. These are
  **not nine new independent replications**; archived ecological results are reused.
* The 100-tick prefixes also matched archived pilot action digests and every
  aggregate field; the only configuration difference is the inert run horizon.
* Six 100-tick D/E pilots, ecological seeds 0/1/2, nuisance seeds 10000/10100/10200.
* Six additional **100-tick** D/E realizations with nuisance seeds
  10001/10101/10201. D and E have independent streams despite matching seed labels
  because their category counts are included in the stream name.
* D/E seed 0 realization 0 each rerun at 100 ticks: complete state/RNG/action
  fingerprints, statistics and every compressed stream were byte-identical.
* Two-tick main.py CLI smoke with four agents, frozen D model and nuisance seed
  123: population 4, zero deaths/births, one water consumed, eight emissions.

The **500-tick D/E primary and 500-tick nuisance robustness were not run**.
The pilot crossed the predeclared gross entropy-mismatch boundary. The explicit
review receipt `data/social_learning_v073/pilot_review.json` blocks the primary,
and invoking the primary command failed safely without creating a primary output
directory. The only extra online nuisance evaluation allowed was the reviewed
100-tick sensitivity check. No 1000-tick run or capacity sweep was executed.
The numerical fixed-experience replay is implemented and tested, including exact
Q/LRU/pending reconstruction under forced capacity-one churn, but its 500-tick
comparison was deferred because the primary matching prerequisite failed.

`verification.json` checked all 579 baseline inventory files: 575 unchanged,
four explicitly edited existing files (README, gitignore, main, Simulation).
All v0.7.2 experimental outputs and earlier studies remain unchanged. Config,
LearningBrain, original social encoder, genomes, reproduction and resource code
retain their baseline hashes. Diagnostics on/off equality and forced-action
reward/Vocal-Q equality are tested separately from ecological comparisons.

### Frozen calibration and transport failure

| Model | alpha | Calibration social first-state fraction | Nuisance first-state fraction | Social H | Nuisance H | Social P(same) | Nuisance P(same) |
|---|---:|---:|---:|---:|---:|---:|---:|
| D vs B | 0.313897538 | 0.799253 | 0.814183 | 2.029425 | 2.033255 | 0.536498 | 0.536324 |
| E vs C | 0.302890187 | 0.807849 | 0.820955 | 2.229420 | 2.225851 | 0.520326 | 0.520380 |

Calibration social values average seeds 3/4; nuisance values additionally average
the two fixed calibration realizations per seed. Full pi vectors, candidate
scores, target metrics, supported conditional metrics and calibration source
hashes are saved in `models/D.json`, `models/E.json`, `models/frozen.json`.
The archived 500-tick B entropies were 2.000/2.003/2.023 bits and C entropies
2.231/2.247/2.237, reasonably near the separate calibration seeds. This apparent
whole-run agreement concealed substantial **within-run nonstationarity**.

The early 100 ticks have B entropy 2.959/2.988/3.040 and C entropy
3.279/3.359/3.278. Frozen stationary nuisance parameters calibrated over 500 ticks
cannot track that changing occupancy distribution without a different model.
There was no retuning on the evaluation seeds, no age/tick/position hash, and no
live social/X feedback added to repair this failure.

Across all 12 evaluation nuisance realizations:

* First-state fractions, relative mean visits and same-token probability passed
  their provisional tolerances in every case.
* All 12 marginal entropies failed, absolute differences **0.938–1.142 bits**;
  all 12 conditional entropies failed, differences **0.494–0.610 bits**.
* All 12 marginal-entropy failures exceeded the 0.75-bit gross boundary. Thus
  24 of 60 metric checks failed; there are still only **three ecological seeds**.
* Similar first-state fractions and 11/37 structural categories did **not** make
  the controls empirically equivalent. Run-length distributions and conditional
  category tables are retained for inspection; censored segments are explicit.

### Pilot results by ecological seed (100 ticks)

| Ecological seed | A population | B | C | D r0 / r1 | E r0 / r1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 207 | 217 | 198 | 217 / 216 | 214 / 212 |
| 1 | 208 | 209 | 200 | 211 / 208 | 203 / 205 |
| 2 | 208 | 211 | 210 | 216 / 208 | 214 / 206 |

The following summaries average realizations inside seed, then average three
seeds; ± is between-seed **sample SD**:

| Condition | Population | Births | Deaths | First-state % | States visited once % | Mean visits | Mean decision-time Q rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 207.67 ± 0.58 | 109.00 ± 1.00 | 1.33 ± 1.53 | 42.07 ± 0.56 | 60.94 ± 0.89 | 2.377 | 19.665 |
| B | 212.33 ± 4.16 | 114.33 ± 5.77 | 2.00 ± 2.00 | 62.15 ± 0.65 | 69.07 ± 0.27 | 1.609 | 28.141 |
| C | 202.67 ± 6.43 | 105.00 ± 5.20 | 2.33 ± 1.53 | 62.85 ± 0.36 | 70.53 ± 0.16 | 1.591 | 28.852 |
| D | 212.67 ± 3.55 | 115.00 ± 2.78 | 2.33 ± 1.26 | 61.25 ± 0.36 | 70.79 ± 0.41 | 1.633 | 27.928 |
| E | 209.00 ± 4.58 | 110.33 ± 4.86 | 1.33 ± 0.29 | 62.27 ± 0.71 | 71.75 ± 0.38 | 1.606 | 28.717 |

No Q eviction occurred in any of these 100-tick cases; that is not a statement
about the archived 500-tick cases. Actual Q update counts and retained-row
visit/TD-update histograms are in `runs.csv`/`retained_rows.csv` for A–E. The
additional lifetime update histogram for 100 ticks is available for D/E; A/B/C
reference lifetime histograms end at 500 and are not relabelled as 100-tick data.

Pilot mean effective token counts were B 7.976, C 9.887, D 4.081, E 4.691
(the A baseline has one constant context). Mean conditional entropies were
B 2.227, C 2.389, D 1.687, E 1.811 bits. This exposes the mismatch that is hidden
by the closely matched first-state fractions.

### Learned dependence, behavior, ecology and vocals

At tick 100, fixed-original-state retained-row policy TV (>=2 visits and >=1
actual TD update) was B 0.47415 ± 0.00754, C 0.47265 ± 0.00323,
D 0.48798 ± 0.01089, E 0.48249 ± 0.00600. At >=5 visits it was
B 0.51949 ± 0.02234, C 0.42344 ± 0.02727, D 0.49958 ± 0.00670,
E 0.49972 ± 0.05095. Raw pair/agent/qualified-row counts accompany every probe;
pair counts are not replications. A has no alternative-token pair, hence NA.
Independent nuisance **also produced Q dependence**, so Q dependence alone does
not establish information value.

Actual toward-visible-target movement fractions were A 0.12625, B 0.12541,
C 0.13230, D 0.12419, E 0.12612. They exclude stationary/blocked cases and refer
to the frozen pre-action nearest target, not inferred movement intentions.
Mean food/water consumptions were respectively A 1557/1605, B 1258/1281.67,
C 1220.67/1271, D 1280.17/1301, E 1268/1304.50. Reproduction-action fractions
were A 0.10064, B 0.11063, C 0.11072, D 0.11197, E 0.11069. Full action,
contact, death-cause, generation, age, need-stratum and denominator tables remain
available; short-run counts do not establish a survival advantage.

Signals per decision remained about 0.939–0.942, with identified-hearing fractions
about 0.169–0.182 across A–E. Vocal Q code/state were unchanged; different physical
actions/positions/populations can still change learned vocals and reception.
This is passive observation, not intentional communication or shared meaning.

Changing nuisance realization changed final D populations by 1/3/8 agents for
seeds 0/1/2, and E by 2/2/8. Even 100 ticks shows nuisance-seed variability.
All summaries average the two realizations inside seed; n remains three.

### Held-out physiological reward prediction (500-tick B/C experience)

This separate observational analysis does not depend on claiming D/E online
matching. Train seeds 3/4, test seeds 0/1/2; actual chosen physical action is
conditioned on in every model. All-decision held-out log loss (lower is better):

| Fixed source | X,A | X,S,A | X,U,A | Supported X / S / U test decisions |
|---|---:|---:|---:|---:|
| B | 0.096571 | 0.124112 | 0.123280 | 58.08% / 36.25% / 36.61% |
| C | 0.097462 | 0.126794 | 0.125215 | 58.27% / 34.65% / 35.57% |

The predeclared estimator backs sparse cells off to the action-specific prior.
Extra dimensions reduce support; this comparison alone cannot isolate feature
information from estimation error. Original all-decision results are preserved.

A separately labelled **post-hoc descriptive support diagnostic** restricts all
three predictors to the exact same test decisions with >=20 training observations
in every head. Eligibility uses feature/action counts, never reward, and no
threshold/model was retuned. It retains 29.39% of B test decisions and 27.59% of C
(per seed about 87–96 thousand decisions, 2576–2713 actual agents). Mean log loss:

| Fixed source, common support | X,A | X,S,A | X,U,A |
|---|---:|---:|---:|
| B | 0.084228 | 0.088721 | 0.088440 |
| C | 0.088049 | 0.092432 | 0.092236 |

All three seeds had slightly worse social than independent-nuisance log loss on
this supported subset. This estimator therefore **did not demonstrate incremental
predictive benefit**. It does not show that social information is absent: coarse
reward bins, finite training data, smoothing, auditory-state sparsity, selected
support and endogenous behavior all limit that inference. Brier loss, sample SD,
equal-agent losses, per-seed paired contrasts and denominators are saved. No
conditional-MI/significance/survival-benefit claim is made.

### Runtime, storage and verification

Measured per-case wall times: calibration 67.9–71.9 seconds (4 cases, sum 276.8),
references 59.2–77.8 seconds (9 cases, sum 629.5), initial nuisance pilots
3.12–3.24 seconds (6 cases, sum 19.11), extra realizations 3.12–3.25 seconds
(6 cases, sum 19.04), deterministic reruns 3.27–3.32 seconds (2 cases, sum 6.59).
These sums are **not study elapsed wall time**: independent reference CLI
processes ran concurrently. No multiprocessing was added to the simulator.
Compressed diagnostic streams total about 66.7 MB across those cases; raw data
plus manifests/models/CLI smoke are about 67.4 MB. The detailed recorded byte
counts are in `verification.json`; review figures add a few MB of analysis data.
Ordered outcomes are optional and considerably larger than default aggregate
statistics. A default simulation acquires no new retained history.

The final complete suite passed **577 tests in 89.007 seconds** (554 existing +
23 new). The pre-edit baseline passed 554 tests; the first implementation stage
passed 571 before the additional prediction/replay/grouping/validation tests
were added. No existing test was weakened or removed. Final log:
`data/v073_final_tests.log`. Focused tests cover deterministic nuisance generation,
persistence/bounds, independent/global RNG, preview purity and feature privacy,
unchanged original/vocal states, diagnostic on/off equality in all A–E modes,
fixed-action reward/Vocal-Q equality, malformed configs, exclusive byte-identical
writers, calibration determinism/reward blindness, hierarchical n, actual
sequence accounting, action-conditioned prediction and exact replay under churn.

`python3 -m experiments.verify_information_controls` passed: all nine full
references, all nine pilot prefixes, both full-state/all-stream deterministic
reruns, the frozen model hashes, identical initial ecology/main streams for all
D/E cases, and all 575 protected existing files. The default core remains
standard-library-only; pandas/matplotlib are used only in analysis. Reviewed
figures were visually checked, including the entropy failure panels.

### Conclusion and limits

Independent representations can reproduce increased fragmentation and sizeable
trained Q dependence, with changed actions and ecology, without social input.
But the nuisance controls were **not matched in entropy/conditional distribution**,
and the primary experiment was correctly blocked. The held-out estimator did not
demonstrate useful extra prediction; it is not a proof of information absence.
The central question therefore remains **unresolved**: current evidence does not
separate social information value from all representation/visitation effects.
There is no established social preference, cooperation or survival benefit.

Prioritize a fresh preregistered matching study that treats nonstationarity and
state-conditional support explicitly, while retaining independent live nuisance
innovations. First validate fixed-experience coverage/update comparisons and
held-out predictors, then consider a new reviewed online experiment. Do not
repair matching by reading live social features, ecological variables, tick/age
hashes, raising Q capacity or adding a social reward. A new estimator or model
would be a new exploratory design, not a retroactive success for this pilot.

## Reproduction

Use the commands in the README v0.7.3 section, in their listed order. All new
destinations are exclusive; choose a fresh `--root` and `--output` for reruns.
`main.py --mode evolution --brain learning --social-memory --nuisance-model
data/social_learning_v073/models/D.json --nuisance-seed 10000 --ticks 100
--controls-csv data/new_D.csv.gz` runs an individually inspectable D control.
All analyses use ecological seeds as units, average nuisance realizations inside
seed first, then report equal-seed mean/sample SD/min/max/n. No significance claim
is made from n=3. Figures identify horizons, capacity, replication and mismatch.

For this blocked pilot, the follow-up commands actually applicable are:

```bash
python3 -m experiments.information_controls --phase pilot_robustness
python3 -m experiments.analyze_information_controls \
  --output analysis/social_learning_v073/pilot_reviewed
python3 -m experiments.controls_followup
python3 -m experiments.controls_followup --common-support \
  --output analysis/social_learning_v073/prediction_support
python3 -m experiments.verify_information_controls
```

The robustness command requires the saved review allowing only a 100-tick
exploratory sensitivity check. The 500-tick primary/robustness and
`--include-replay` commands remain blocked. No receipt was changed to permit them.

## Files

Existing files edited: `main.py`, `simulation/simulation.py` (optional nuisance
wiring only), `README.md`, `.gitignore`. Added:

* `agents/nuisance.py`: frozen experiment config, independent stationary channels,
  immutable token observation and physical-head-only subclass.
* `simulation/controls_observer.py`: versioned optional outcomes/row diagnostics.
* `experiments/information_controls.py`: bounded staged runner and legacy proof.
* `experiments/calibrate_information_controls.py`: feature-only frozen calibration.
* `experiments/controls_metrics.py`: checked readers and empirical sequence metrics.
* `experiments/analyze_information_controls.py`, `controls_plots.py`: grouped
  summaries, paired results, matching diagnostics and figures.
* `experiments/controls_prediction.py`, `controls_followup.py`: held-out prediction
  and optional fixed-experience learning reconstruction.
* `experiments/verify_information_controls.py`: file/trajectory/seed/model audit.
* `tests/test_information_controls.py` and this report.

## Interpretation and next experiments

Adding dimensions, changing visitation, predictive information, Q dependence,
changed actions and improved survival are distinct claims. A social feature can
predict reproductive eligibility or consequences of action selection without
improving a policy. Movement near others is not a learned social preference.
Changed vocal/hearing frequencies are not evidence of intentional communication.
The first next step is to improve/diagnose matching and held-out support where
needed, not to add social rewards, extra Q capacity or semantic signal labels.
