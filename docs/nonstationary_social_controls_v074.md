# v0.7.4 — Nonstationary social controls and fixed experience

## Objective and stopping point

Can social-information effects on Physical Q be distinguished from fragmentation
and learning history? This release tests **one** prespecified nonstationary
nuisance family. No positive result is required. The pilot failed the frozen
gate, so no500-tick D/E online primary was run and no redesign followed.
Fixed-experience replay and held-out prediction were completed independently.
The methodological phase is now closed; the unresolved distinctions are explicit
below. No v0.8 ecology, social reward, action, signal meaning, sharing or cooperation
mechanism was implemented.

## Inspected implementation and reusable archives

Inspection began at v0.7.3 commit `3835353cf3d8ccdb3e8afb451fe23bda61de2785`.
README, v0.7.3 report, source, tests, calibration/gates, ordered diagnostics,
replay, prediction and compatibility tools were read before changes. The full
baseline passed577 tests in100.878s. A986-file pre-edit inventory was saved.

The actual unchanged sequence is: regenerate; shuffle existing actors; acquire
one sequential raw visual observation; freeze coarse social input before this
sighting updates LTM; auditory update and original X encoding; physical suffix;
read next-row bootstrap **before** previous-row recreation/TD update; current-row
allocation/LRU refresh and epsilon-greedy choice; Vocal Q using original X;
action/reproduction/emission/physiology/reward; immediate terminal update or
pending survivor transition; death removal/newborn insertion/communication/stats.
Newborns do not act at birth. Zero initialization and both Q capacities256 remain.

The v0.7.3 archive already contains actual ordered decisions, original-key catalogs,
passive B/C tokens, chosen actions, immediate reward, terminal flags, retained-row
histograms and physical-Q/pending digests. All B/C500 recordings are reused.
v0.7.2 grouped strata lack exact order/reward, so they are not substituted for
transitions. No exact per-decision resource quantities, acquisition-success field,
parentage or pending before-physiology exists in these compact records.

Age is derived as zero-based own-opportunity index only because complete records
start at birth and physiology increments age once per own turn. A test including
newborns validates this identity. Age and opportunities therefore are **collinear**,
not independent predictors. Original X distinguishes resource-visible/here/absent
(the absent sentinel is(2,2)); no amounts or successful eating/drinking are invented.

## Phase I: nonstationarity, lifetime and composition

Calibration-only seeds3/4 were analyzed first. Windows are exactly[0,100),
[100,200),...,[400,500). Tick0 has no decisions; tick500 is included only in the
full500-tick profile. Extra diagnostic-stratum rows labelled window5 describe
tick500 alone and are not complete-window comparisons or gate inputs.

Equal-calibration-seed marginal entropies:

| Calendar window | B bits | C bits |
|---|---:|---:|
| 0–99 | 3.0359 | 3.3159 |
| 100–199 | 2.3879 | 2.6025 |
| 200–299 | 1.8892 | 2.0988 |
| 300–399 | 1.7325 | 1.9363 |
| 400–499 | 1.9509 | 2.1085 |

`nonstationarity/` preserves per-seed frequencies, conditional/supported entropies,
effective categories, transitions, clipped runs, novelty, visits, distinct pairs,
decision/agent exposure, previous-tick population and age/opportunity composition.
`diagnostic_strata.csv` conditions descriptively on population bins, visible-count
0/1/multiple, age/opportunity bins, founder/newborn cohort and actual encoded local
resource availability. `cohort_selection.csv` supplements first observed agents,
newborn first opportunities, terminal observations and founder exposure share.
These diagnostics never become nuisance inputs.

Mean pre-action age rises from about42.5 in the first interval to about88 late
in the run, while population and founder/newborn exposure also change. Even
within age bins the token distribution changes over calendar windows. These are
associations: calendar effects, cohort composition, selection, genome changes,
population and resources are not independently manipulated or causally separated.
They explain why matching only a whole-run frequency is inadequate.

Strict statistical X-independence would require H(U|X,t)=H(U|t). Social conditional
distributions generally violate that restriction. Plug-in entropy/MI has sparse-X
bias, so the entropy gap is not an unbiased information estimate or an impossibility
proof. The actual independent-innovation family is audited on **fixed calibration
X**. Clock, inherited process history and birth timing can still induce recorded
associations; online policy/selection can add more. Independence of random draws
does not guarantee unconditional independence from outcomes or original X.

## Frozen design, fit and generator

The actual `preregistration.json` was written before model implementation and
calibration/evaluation. Its tracked copy is `experiments/nonstationary_design.json`;
the later `preregister` CLI reproduces it for a fresh study destination.
Models use only seeds3/4; seeds0/1/2 are reserved for evaluation. No ecological
performance, reward, movement or survival selects parameters. There is no search:
pi is the equal-seed marginal frequency in each epoch, and
`alpha=(empirical_same-sum(pi**2))/(1-sum(pi**2))`. Saturation to[0,1] would be
flagged; this fit did not require model redesign. Frozen JSON stores complete pi
vectors, raw moment estimates, source hashes, category counts and schedule.

| Epoch start | D alpha (11 categories) | E alpha (37 categories) |
|---:|---:|---:|
| 0 | 0.460122 | 0.423360 |
| 100 | 0.355455 | 0.361224 |
| 200 | 0.289654 | 0.286779 |
| 300 | 0.261451 | 0.246803 |
| 400 | 0.281144 | 0.269298 |

`NonstationaryNuisanceModel(category_count, schedule)` is frozen, with immutable
`NuisanceEpoch(start_tick, NuisanceModel(...))` entries starting at zero and strictly
increasing. The last entry continues after400. `NuisanceConfig` accepts stationary
or scheduled models; no original Config/Genome schema changes. A separate master
Random `v074:nuisance:{seed}:{K}` assigns128-bit private channel seeds by brain
construction order, never by identity. The stationary branch retains its exact
v073 namespace/state/draw order.

Each scheduled channel draws one private U(0,1) at construction. On its first
actual turn that same uniform is mapped through pi(current tick), including
late-born agents. A read-only pre-turn preview maps the same uniform without
drawing. Subsequent own turns hold with alpha(t), otherwise sample pi(t). There
is **no forced reset** at an epoch boundary; this is a time-inhomogeneous Markov
matrix `Pij(t)=alpha(t)*1[i=j]+(1-alpha(t))*pi_j(t)`. Marginals can transiently lag
the schedule. No per-elapsed-tick draws, large hidden model or live feedback exist.
Only an increasing actual calendar tick enters advance; token IDs remain arbitrary.
The clock itself may encode expected ecological stage, an explicit relaxation
from v0.7.3, not a hash of age/tick or an assertion of zero outcome information.

## Acceptance rules and pilot decision

Every supported metric in every **complete** window and full pilot must pass in
every seed/realization. Unsupported essential comparisons fail closed. Incomplete
tick100-only data are retained diagnostically, not treated as a whole second window.

| Metric | Target | Gross mismatch |
|---|---:|---:|
| Marginal entropy difference | 0.25 bits | 0.75 bits |
| Conditional entropy difference | 0.25 bits | 0.50 bits |
| Marginal TV | 0.10 | 0.25 |
| Conditional TV on common X | 0.25 | 0.50 |
| Same-token probability difference | 0.05 | 0.15 |
| Clipped run-CDF distance | 0.15 | 0.30 |
| Lifetime first-state fraction difference | 0.02 | 0.05 |
| Relative mean visits difference | 10% | 25% |
| States-once fraction difference | 0.05 | 0.10 |
| Relative effective-category difference | 20% | 50% |

Original v0.7.3 H/novelty/mean-visits/same-token targets were retained. Added
conditional/persistence/distribution/occupancy criteria were justified and frozen
before evaluation. Conditional TV uses exact original-key matches with>=20
decisions in EACH run and>=10% coverage in EACH. Its per-cell/exposure support
is retained. Clipped run-CDF cutpoints are1,2,3,4,5,8,16,32,infinity, with left
and death/window-end censoring explicit. These are finite-sample diagnostic
tolerances, not significance tests or guarantees of equivalence.

Twelve100-tick pilots used ecological seeds0/1/2 and two realizations each for D/E.
Nuisance seeds are20000+100*ecological_seed+realization (K separates D/E streams).
Frozen calibration already showed early novelty limitations; the bounded pilot
tested the one family without any post-evaluation tuning.

**Gate: BLOCKED.** Across240 complete-window/full-run checks,44 failed,20 gross,
zero unsupported. Marginal entropy differences were only−0.054 to+0.097 bits,
marginal TV0.025–0.053, persistence difference−0.018 to+0.029, run-CDF distance
0.035–0.063. But first-state fractions were **4.17–6.86 percentage points higher**
(all24 checks failed;20 gross), and20 conditional-entropy checks failed, with
all conditional differences ranging0.212–0.362 bits. Conditional TV0.177–0.226
passed its provisional threshold; that does not negate entropy/novelty failures.

Full-pilot equal-seed means (nuisance realizations averaged within seed):

| Condition | H | H(token|X) | First-state fraction | Mean visits | States once |
|---|---:|---:|---:|---:|---:|
| B | 2.995210 | 2.227122 | 0.621490 | 1.609153 | 0.690649 |
| D | 3.035932 | 2.522033 | 0.675420 | 1.480680 | 0.725111 |
| C | 3.305063 | 2.388839 | 0.628479 | 1.591178 | 0.705265 |
| E | 3.322328 | 2.682605 | 0.692274 | 1.444584 | 0.742871 |

The nonstationary family repairs the old marginal mismatch, but still assigns
independent tokens to original contexts differently, creating more new states.
Matching marginal entropy and persistence does not match learning opportunities.
The500-tick primary command was explicitly tested and refused. No500/1000-tick
D/E ecology run, threshold override, capacity change or second nuisance cycle ran.

## Fixed-experience Q replay (high priority, independent of online gate)

All nine A/B/C archived500-tick source representations reproduced their exact
physical-Q/LRU and pending(state,action,reward) digests. Pending before-physiology
is unavailable and unnecessary once actual reward is recorded; it is not invented
or claimed to be reconstructed. Full simulator state is not replayed.
Read-before-update, zero priors, capacity256, current-row residence visits/updates,
terminal updates and survivor pending cutoff are preserved. Unit tests force
capacity-one churn to exercise previous-row recreation ordering.

Then54 comparisons used fixed B/C experience: X; X+B; X+C; X+U11; X+U37;
X+B+independent remembered-direction projection of U37. Non-nuisance heads run
once; nuisance heads use two realizations (seed50000+100*eco+r). The final
diagnostic projection holds actual B fixed but has66 possible combinations,
versus37 valid actual C combinations. It is **not** an empirically matched online
condition and never enters an agent. C-B and memory-only probes distinguish
perception from additional remembered information.

Offline nuisance channels are allocated at the first actual recorded decision
because biological construction order is not recorded. IDs are ownership lookup
keys only, not seed material. These exchangeable private streams are deliberately
new realizations, not a fabricated reconstruction of original birth order. Source
D/E replay, when tested, uses its actual recorded tokens for exact reconstruction.

On fixed **C** experience, equal-seed means:

| Encoding | First-state fraction | Visits/state | Once fraction | Retained mean Q range | Retention among living observed keys |
|---|---:|---:|---:|---:|---:|
| X | 0.678873 | 1.473084 | 0.768322 | 0.010857 | 0.999065 |
| X+B | 0.800881 | 1.248638 | 0.836697 | 0.009807 | 0.992180 |
| X+C | 0.806884 | 1.239352 | 0.842553 | 0.009766 | 0.991668 |
| X+U11 | 0.817869 | 1.222704 | 0.853778 | 0.009786 | 0.991213 |
| X+U37 | 0.823456 | 1.214405 | 0.857749 | 0.009756 | 0.990452 |
| X+B+Urd | 0.898039 | 1.113541 | 0.909253 | 0.009296 | 0.981725 |

Visitation/update distributions, row retention, evictions, Q ranges, retained
action probabilities and read-only trained-row policy probes are retained.
At>=2 row visits and>=1 update, all-suffix policy TV on C experience was
B0.36324, C0.35993, U11 0.34756, U37 0.33934. Independent inputs also yield
learned dependence. Memory-only C has just2/0/3 qualifying pairs in seeds0/1/2
at>=2 visits, and **zero in every seed at>=5 visits**. The strict memory comparison
is NA, not evidence of no memory effect. Replay establishes representation and
learning-history differences, never unbiased policy or survival value.

## Held-out outcome prediction

One prespecified compact hierarchical count model trains on3/4 and tests0/1/2.
Min20 cell observations and Dirichlet strength5 remain. Unlike v0.7.3, sparse
token cells back off to their **same X,A parent**, and remembered extensions to
X,B,A. Training freezes before test scoring. All models score identical eligible
decisions; the common subset requires support in every compared head, with
per-seed/agent/exposure reporting. Two attached nuisance realizations use
60000+100*eco+r. Calibration bins/ECE, Brier, log loss, equal-agent losses,
common coverage, per-window scores, paired differences and censoring are saved.

Targets: actual immediate reward in bins<=−0.1, (−0.1,0), >=0; sum of next five
ACTUAL own-decision rewards excluding current, same bins; observed survival after
t+5. Fewer than five future reward observations are death/end/gap-censored,
without invented absorbing rewards. Observed death before/at the survival horizon
is known0; actual horizon-alive observation is1; otherwise censored. No resource
acquisition target is manufactured from an action label or reward sign.

All-decision immediate-reward log loss (lower is better):

| Fixed source | X,A | X,B,A | X,C,A | X,U11,A | X,U37,A |
|---|---:|---:|---:|---:|---:|
| B | 0.096571 | 0.095859 | NA | 0.095899 | NA |
| C | 0.097462 | 0.096686 | 0.096732 | 0.096840 | 0.096822 |

Social perception improves immediate-reward prediction slightly over X,A in all
three seeds. The improvement over U11 is small/mixed on all B decisions (two
seeds better, one worse), and consistent but small on C experience. On identical
common support, B-source losses are X0.083502, B0.081827, U11 0.082400;
C-source losses are X0.121679, B0.120199, C0.120195, U37 0.121136,
B+Urd0.121842. The common subsets differ between source policies and must not be
compared as the same population. Coverage is only29.40% for B and14.11% for C;
C uses the stricter intersection of six heads. Extra actual memory versus B has
mixed tiny common-support signs, not a consistent additional benefit.

For next-five reward, X losses are0.670454(B)/0.670630(C), B extensions
0.674650/0.674557 and actual C0.678090. For survival, X0.131755/0.132019,
B0.132401/0.132916 and C0.134140. These estimators do not improve those outcomes
over X. Better performance against a more fragmented independent-memory surrogate
does not establish memory value over the perception parent. This is observational
prediction, not an information-theoretic proof, causal effect or policy evaluation.
No significance is claimed from three ecological seeds, and no unsupported
memory-pair comparison is converted to zero.

## Pilot ecological observations and compatibility

| Ecological seed | D r0 / r1 population | E r0 / r1 population |
|---:|---:|---:|
| 0 | 217 / 202 | 218 / 211 |
| 1 | 222 / 227 | 212 / 213 |
| 2 | 219 / 222 | 214 / 212 |

Nuisance realization changes final population by up to15 agents even at100 ticks.
Raw births/deaths/resources, actions, actual moved-target directions, passive social
exposure, auditory/vocal counts and generation statistics are preserved in pilot
records/online_runs.csv. They are descriptive unmatched outcomes, not primary
survival evidence; no primary population plot is fabricated.

Compatibility: seed0 A/B/C100 complete state/action/RNG/statistics checks match
archived v0.7.2. Both v0.7.3 stationary D/E seed0 controls match full states,
RNGs and every compressed diagnostic stream byte-for-byte. Scheduled D/E seed0
reruns also match all states/RNGs/streams exactly; all12 pilots share the original
initial ecology/main streams. The original Config, LearningBrain, social encoder,
genome, reproduction, resource and old test files remain unchanged.
The final suite passed **602 tests in90.719s**: all577 original plus25 new.
Tests include schedule validation/boundaries/no-reset/late-born first turns,
independent RNG/previews/privacy, reward/Vocal isolation, exact source replay,
terminal/pending/churn semantics, censoring, hierarchical backoff/test isolation,
age/opportunity identity, conditional support and fail-closed frozen gates.

## Performance, outputs and reproduction

Calibration/audit:40.62s, peak392MB; replay including9 validations and54 comparisons:
132.03s, peak300MB; prediction:346.21s, peak658MB; figures:1.68s, peak213MB.
These are measured separate-process wall times/high-water RSS (decimal bytes),
not a fabricated total study elapsed time. Pilot/determinism cases also have
per-case wall/RSS/stream-byte receipts. Analysis loads bounded500-tick runs and
two training-seed frames, not an unbounded full-history simulator. Diagnostics
are optional/streamed; no per-tick Q snapshots or whole dead Human objects are saved.
`verification.json`/`study_manifest.json` contain final sizes, input/source hashes,
test log and compatibility receipts. Earlier v0.7.0–v0.7.3 artifacts and the
v0.7.3 failed gate remain intact.

The12 pilot case wall times sum to26.04s, recorded process peak164MB; two scheduled
determinism cases sum4.37s. Legacy A/B/C compatibility cases sum7.77s and
stationary D/E compatibility6.56s. Their older timing helper did not record RSS,
so those RSS fields are NA, not guessed. Raw study data are about5.46MB and
analysis about3.23MB before final receipts/manifests.977 protected existing files
passed the hash audit; only the nine explicitly listed existing source/docs files
changed. No historical experimental artifact was overwritten.

New outputs: `data/social_learning_v074/` for design/frozen models/pilots/proofs;
`analysis/social_learning_v074/` for nonstationarity, calibration, replay,
prediction, pilot tables and11 PNG figures. Ratios are computed within run;
nuisance realizations are averaged inside ecological seed, then equal-seed
means/sample SD/min/max/n. Calibration n=2, held-out n=3; paired seeds and
realization sensitivity are retained, not pooled as millions of independent decisions.

The README lists the staged CLI. Existing destinations are exclusive. For a
fresh reproduction with the same archived inputs, use `--root` and
`--output-root` on the controls runner, then matching `--root`/`--output` paths
on replay/prediction/figures. Diagnosis precedes `--phase preregister`; the latter
archives the tracked design with diagnostic hashes. Current study's preregistration
already exists and must not be overwritten. `--phase primary` remains blocked.
The study audit requires the saved pre-edit `before.json` and final test log;
these are current-release verification artifacts, not a command to rerun history.

## Files and conclusions

Existing files extended: agents/nuisance.py; simulation/simulation.py;
simulation/controls_observer.py; main.py; experiments/controls_metrics.py;
experiments/controls_prediction.py; experiments/information_controls.py;
README.md; .gitignore. Added six `experiments/nonstationary_*.py` modules,
the tracked design JSON,25 tests in test_nonstationary_controls.py, and this report.

The central question is **not fully resolved**. Fixed experience separates and
measures representation/learning-history effects. Small observational immediate
perception-prediction gains are present under this estimator. The independent
online controls still do not match lifetime novelty/conditional distributions,
so learned dependence and online behavior cannot be cleanly attributed to useful
social information rather than changed visitation/history. Memory-specific strict
Q support is inadequate, and policy/survival benefit remains unestablished.
No social preference, attraction, cooperation, communication or language is inferred.

Stop this nuisance-design cycle. A sensible v0.8 starting point is one small,
preregistered ecological relevance experiment: scarce spatially clustered renewable
resources with explicit individual resource/survival outcomes, retaining the old
ecology as control. Keep any new rule primitive and vary one factor; do not add
social bonuses, prescribed signals or cooperative roles. The present ecology need
not produce a positive social-information result before that next research phase.
