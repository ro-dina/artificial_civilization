# v0.8.1 — Future resource information and target support

## Research question and stopping decision

Can existing social perception or remembered-individual information predict
future resources beyond original physical state X? This release audits and
post-processes the v0.8.0 archive. It changes no simulation, Config, reward,
learning, sensory, biological or resource code. No scientific simulation runs,
new-seed confirmation or simulator instrumentation were added.

**There is measurable future single-tile variation, but no candidate met all
frozen feasibility rules.** Primary environmental target: null. Secondary
behavioral target: null. Two fixed fallback comparisons were executed only as
retrospective diagnostics. Qualified perception and memory effects are NA,
not zero. This is a measurement/support limitation, not evidence that social
information is intrinsically useless.

The baseline commit is `f902d840ae965425a97a0ef5993a73f3caa62fd1`. The existing
README, v0.8.0 report, observer, resource timing, encoders, ordered records,
life records, experiment runner, predictor, metadata and verification receipts
were inspected before implementation. Baseline625 tests passed in102.014s.

## Archive inventory and integrity

The audit covered all36 primary cases: E0–E3 × A/B/C × seeds0/1/2,500 ticks,
540 compressed streams,7,685,665 actual decisions. The resource parameters are
unchanged: E0=(1,0), E1=(.5,0), E2=(1,.75), E3=(.5,.75), where the pair is
supply multiplier/localization strength. Q capacities remain256.

`analysis/future_resource_v081/audit/field_inventory.csv` has one row per
stream: filename, schema2, record count, bytes, tick range, agent coverage,
columns, null counts, missing schema fields, identifiers, timestamps, units,
social representation and timing/linkage notes. All declared schemas and
checksums matched. No exact future world events were reconstructed from stocks.

| Stream/field | Available information | Timing and limitation |
|---|---|---|
| `acquisitions` | sequence, agent ID, own index, X catalog ID, action, B/C tokens; coordinates; actual0/1 consumed units; success/failure; reward; terminal/cause | One actual ordered decision. Before/after quantities and physiology are distinct. |
| `food_here`, `water_here` | Exact integer units at actor tile | Current sequential **pre-action** world, after earlier actors may have consumed. |
| `food_at_visible_neighbor`, water equivalent | Exact quantity at current nearest visible individual's cell | Same pre-action world; empty if no visible target. Diagnostic only. |
| `*_distance_tickstart`, `*_near3_tickstart` | Exact distance/binary result at actor's current coordinate on a frozen world map | Map taken **after regeneration, before actors**. Can be stale by the actor's turn. The map itself is not saved. |
| `spatial` | Full1200-cell integer food/water arrays and agent counts | **End ticks0/100/200/300/400/500 only**, not every tick and not tick-start. |
| `resource_ticks`, aggregate ticks | Stocks, supply/consumption budgets and summaries | Cannot identify exact tile-level depletion between snapshots. |
| `states` | Canonical original X JSON per local catalog ID | X includes old need/resource/auditory features. IDs are local; canonical keys are used across seeds. |
| B/C tokens | Existing visible token B and valid(B,RD) pair | RD decoded through unchanged `C_PAIRS`. Memory is read before current visual-batch commit. |
| `outcomes` | Same sequence, action, actual reward, terminal | Matches acquisition stream exactly; no unrecorded next observation. |
| `individuals` | Birth tick, generation, actual age/duration, opportunities, acquisition totals, death/censor flag | Every founder/child, including birth-tick newborns with zero decisions. |
| Other diagnostic roles | Grouped actions, visits/updates, Q probes, vocal counts, windows | Not replacements for exact resource event histories. |

Integrity checks passed for all36 cases: unique/consecutive sequence and
(agent,own-index) keys; birth/age/own-index/tick alignment; no missing own
histories; monotone global tick order; at most one terminal at history end;
correct death versus censor linkage; actual failed/successful consumption;
individual and aggregate resource totals; population/life counts; complete
unique snapshot grids; quantities in[0,5]; and stock mass balance. The independent
current-X/action acquisition oracle still matches every recorded decision.
This oracle applies to immediate acquisition, not to the new future targets.

## Exact time alignment and label populations

The unchanged event sequence is: increment tick; regenerate; shuffle existing
actors; observe pre-action resources/social state; choose action; execute and
consume or reproduce; emit vocalization; update physiology, mortality and
learning; stream outcomes; append newborns; advance auditory buffers; record
statistics and end-tick snapshots when scheduled. Newborns first act next tick.

For environmental prediction at an origin decision t, coordinates are frozen
from that origin's sequential pre-action record. The endpoint is the **actual
end-of-tick t+h** resource array, including all actions through that tick.
For h5, eligible saved endpoints are100/200/300/400/500, so origin ticks are
95/195/295/395/495. No later same-tick observation, future position, interpolation,
aggregate stock inversion or future location selection enters a current feature.
An environmental label can still exist if the originating individual later dies.

This is a sparse calendar-selected sample, not a random sample of all decisions.
Comparisons between h1/h5/h10 use different origins relative to common endpoints;
the horizon figure is descriptive, not an isolated causal effect of horizon.
Future world states also depend on subsequent population behavior: calling a
target environmental does not make it policy-independent or exogenous.

## Candidate definitions, fixed before support or model losses

`experiments/future_resource_design.json` and
`data/future_resource_v081/design_receipt.json` fix34 binary variants and3
time-to-event variants. They are prospective analysis rules for **previously
examined archives**, not a preregistration claiming untouched evaluation seeds.

| Family | Exact definition | Variants |
|---|---|---:|
| T1 environmental | Food/water present at current tile or current nearest visible individual's tile at end t+5 | 4 |
| T2 environmental | Food/water anywhere within Chebyshev radius0/1/2 around frozen current position, end t+1/5/10 | 16 new; radius0/h5 aliases T1 and is not duplicated |
| T3 environmental | Current tile at end t+5: depletion conditional present now; recovery conditional absent now; strict integer quantity increase/decrease | 8 |
| T4 time-to-event | Next actual food, water or either acquisition after origin; own-opportunity duration and competing/censor cause | 3 |
| T5 behavioral | Food, water or either acquisition within next5 or10 actual own decisions, excluding origin | 6 |

No-reference visible-neighbor cases and inapplicable T3 starting conditions are
reported separately. Unrecorded endpoint snapshots are missing observations.
No resource change is inferred from binary presence when quantity is required.
T1/T2/T3 can be exact on the saved subset; all-calendar environmental outcomes
are unavailable. T4/T5 are exactly linkable from actual contiguous histories.

T4 records acquisition, competing death, administrative censor or history gap.
Death at the origin is a competing event at elapsed0. Future consumption occurs
before death on the same terminal decision, so an acquisition wins that tie.
Nothing is linked across a missing history gap. The audit found no actual gaps.

For T5, an acquisition observed before the horizon is a known positive even if
later follow-up ends. A negative requires complete alive follow-up through k.
Death without acquisition on/before k is competing and excluded from the binary
label, not ordinary censoring or an invented negative. Administrative end/gaps
without an observed event remain missing. These binary comparisons consequently
have an outcome-dependent observed population and are not full cumulative-risk
estimates. T4 separately reports discrete Aalen–Johansen competing-risk curves;
overlapping origins, heterogeneous agents and potentially informative calendar
censoring limit their interpretation. No T4 binary accuracy model was fitted.

The v0.8.0 next-five negative analysis remains a historical reference, not a new
discovery. Its complete-window definition differs from v0.8.1's observed-event
and explicit competing-death handling; they must not be treated as independent
replications or identical estimands.

## Support thresholds and frozen decisions

Before inspecting gains, the rule receipt fixed prevalence[.10,.90], at least
100 positives and100 negatives,100 distinct agents,20 training observations per
supported feature/action cell,20% M0/M1 common coverage,50% observed-label
coverage, at least two X/action strata with≥20 of **each** class, and at least
two supported visible-social tokens. Memory additionally requires20% stricter
M0/M1/M2 common coverage and two supported remembered-direction values.

These are heuristics, not statistical guarantees. Environmental coverage is
measured against reference-eligible origins across the calendar; accepting a
sparse exact-snapshot subset does not remove that missingness gate. Conditional
support means training cells, not merely many total observations. An observed
pure outcome in a sparse X/action cell is not proof of structural determinism.

A candidate could be selected only if a whole ecology passed all nine A/B/C
seed cases. Fixed selection priority began with current-tile water at t+5 for
environment and either-resource acquisition within10 decisions for behavior.
The rule never used predictive gains. All1224 binary candidate/case rows failed
at least one gate:838 exploratory-only,386 degenerate,0 supported. None had a
qualified memory comparison. The after-feasibility/before-scoring record is
`data/future_resource_v081/frozen_evaluation.json`:

```text
primary_environment = null
secondary_behavior = null
retrospective_environment = T1_water_current_h5  # fixed diagnostic fallback
retrospective_behavior = T5_either_k10          # fixed diagnostic fallback
new-seed confirmation = not justified / not executed
```

There was no target/radius/horizon/threshold/model retuning and no attempt to
find a positive result by relaxing support. Selected fallbacks are not promoted
to qualified primary outcomes.

## Feasibility results

Future single-tile food/water is variable: T1 current-tile prevalence ranges
14.5–24.2% for food and14.8–26.7% for water across individual case estimates.
T3 depletion ranges26.1–44.4% food and27.2–45.5% water. Recovery is rarer
(about2.8–9.5% food,3.3–8.7% water), often failing the10% prevalence rule.
Radius2 presence remains near-saturated (roughly94–99.9%); radius1 has more
variation (about76–90%). These are observed snapshot-aligned populations only.

For diagnostic T1 water/h5,84,836 of7,685,665 origins have exact labels. Each run
has1075–3582 labeled origins,185–731 positives and890–2921 negatives. Marginal
class counts are adequate, but label coverage is only1.055–1.137%. No X/action
stratum reaches20 of each class for this target. M0/M1 training common support
is0 in E0/E1/E3 and only0.4–0.8% of observed labels in E2; it cannot estimate a
reliable social increment.

T5 either/k10 has7,106,574 observed binary labels. Prevalence is approximately
46–51% in condition means and coverage about92–93%, but common training support
is below20% in **all36 cases**. The problem is not merely a constant target.

Mode C examples below are mean±sample SD across three ecological seeds; complete
A/B/C raw and tidy summaries are saved separately. Percentages use each run's
own denominator, then equal-seed averaging.

| Ecology | T1 water/h5 positive % | T5 either/k10 positive % | T5 M0/M1 common % |
|---|---:|---:|---:|
| E0/C | 19.19±0.79 | 47.51±0.61 | 11.84±0.30 |
| E1/C | 19.31±0.75 | 47.49±0.40 | 9.44±0.21 |
| E2/C | 21.11±1.34 | 46.97±0.62 | 10.23±0.23 |
| E3/C | 23.23±3.07 | 49.49±1.96 | 8.83±2.06 |

T5 either/k10 competing-death fractions average about4.9–5.6% and administrative
end fractions1.6–2.5% across ecology/mode summaries. Exact seed-specific counts
and all eligible/missing reasons remain in `censoring.csv`; no missing follow-up
was silently filled. Many T4 origins from the same individual overlap, so their
event counts are not independent experimental sample sizes.

## Retrospective model comparison

The two frozen diagnostic fallbacks use the existing v0.7.4 `Hierarchy`:
minimum20, Dirichlet strength5, binary classes; M0=X,A; M1=X,A,B with M0 parent;
M2=X,A,B,RD with M1 parent. Memory thus holds actual B fixed. Environmental
prediction also has an independent X-only reference. No coordinates, patch
geometry, exact diagnostic quantities, IDs, age/generation, future actions or
future state enter the models. Future actions determine behavioral labels only.

Training uses the selected A calibration seeds3/4,200 ticks within each ecology;
testing uses actual A/B/C primary seeds0/1/2 through500 ticks. Seeds are disjoint,
but these primary seeds were examined in v0.8.0 and are **not untouched holdouts**
for newly developed questions. A evaluation is same-policy; B/C is cross-policy
transfer. Early-training versus later-evaluation time, population and cohort
shift is an additional limitation. There is no random decision split or agent-ID
feature through which identically numbered individuals across seeds can leak.

All heads score exactly the same eligible decisions for each target/case/subset.
All-label, M0/M1-common and stricter all-head-common populations are separate;
their sequence-set hashes and denominators are saved. Binary Brier is(p−y)^2,
log loss uses the true class probability, calibration bins use positive
probability, and equal-agent losses are also reported. Within-run metrics precede
equal-seed mean/sampleSD/min/max/n. No decision-level significance is claimed.

Raw all-label log-loss differences for C experience below are **unsupported
backoff diagnostics**, not qualified social effects. Positive means worse.

| Ecology | T1 M1−M0 | T1 M2−M1 | T5 M1−M0 | T5 M2−M1 |
|---|---:|---:|---:|---:|
| E0 | .000000 | .000000 | +.000619 | +.000858 |
| E1 | .000000 | .000000 | +.000844 | +.000424 |
| E2 | +.000140 | +.000049 | +.001213 | +.000927 |
| E3 | .000000 | .000000 | +.001529 | +.001047 |

Environmental equality in E0/E1/E3 occurs because unsupported social heads fall
back to their parents; **it is not a measured zero information effect**. E2/A
has a tiny raw perception improvement(−.000123), whereas E2/B/C worsen on
average. All are below support criteria and are not evidence of benefit.

T5 mean log loss worsens with each social extension in all12 ecology/mode
combinations. Brier differences are not uniformly in the same direction (some
E0 raw differences improve slightly), so this is not an across-metric universal
negative result. Even baseline log loss is poor in some late distributions;
model limitations and early-to-late policy/ecology shift remain substantial.
All qualified `eligible_difference` entries are NA. Raw numbers are retained
for transparency, without treating a weak model as proof of weak information.

## Implementation correction and reproducibility

The initial scoring pass completed all comparisons but failed while melting the
paired-summary table: `metric` was both a grouping ID and the measure-column
name. Original sources, the original frozen plan, failure log and five completed
score tables were preserved. The correction separates those column names; it
does not change labels, targets, counts, model probabilities, seeds or selection.

`data/future_resource_v081/reporting_correction.json` records exact original and
corrected source hashes. Receipt verification recognizes only these exact
recorded versions and still requires the original plan hash and all input
hashes. A repeated deterministic scoring pass produced **byte-identical**
model comparisons, paired raw results, calibration, common-support and summary
tables. `reporting_recovery_verification.json` records that check. This is a
transparent implementation repair after scoring, not a new preregistration or
an independent replicate. Fresh reproduction with corrected code runs once.

## Outputs, tests and compatibility

Only existing `README.md` and `.gitignore` were edited. New files:

```text
experiments/future_resource_design.json
experiments/future_resource_io.py
experiments/future_resource_targets.py
experiments/future_resource_analysis.py
experiments/future_resource_prediction.py
experiments/future_resource_tables.py
experiments/future_resource_labels.py
experiments/future_resource_plots.py
experiments/future_resource_verify.py
tests/test_future_resource.py
docs/future_resource_information_v081.md
```

The complete final suite passed658 tests(625 unchanged existing +33 new) in
101.153s. New tests cover fixed reference/end-tick alignment, no future feature
leakage, exact quantity changes, gap/agent linkage, competing death/ties,
administrative censoring, support/prevalence/degeneracy, seed separation,
hierarchical backoff, model features/scores, frozen receipts, original-file
immutability and deterministic exclusive CSV outputs. Python/global numpy RNG
states remain unchanged by analysis. No simulator observer was introduced, so
instrumentation-on/off behavior is not a new experimental condition.

Final verification compares1673 protected historical/source files with the
pre-edit1675-file inventory, excluding the two documented existing edits.
All simulator/observer/Config/brain/encoder/resource/reproduction/genome/statistics
sources, original tests and v0.8.0 artifacts are byte-identical. This preserves
the archived complete state/Q/LRU/pending/RNG fingerprints; the original
regression tests also reran. There was no need to simulate the36 cases again.
Receipts/manifests distinguish this source/archive proof from new-run evidence.

`data/future_resource_v081/` holds the pre-edit inventory, design/freeze,
test/phase logs, reporting repair evidence and compact derived labels. Each
case's label table contains sequence plus T4 elapsed/event/next-acquisition
sequence for food/water/either, and the two selected diagnostic label/reason
pairs. Sequence joins the original acquisition stream, so complete features and
histories are **not copied**. All7,685,665 origins are retained. Event1 points
to an actual recorded future acquisition of the same agent; other event types
have no invented acquisition pointer. This gives inspectable death/censor
causes without retaining any Human object or Q snapshot.

`analysis/future_resource_v081/` includes:

* `audit/`: machine-readable field inventory, integrity checks, definitions.
* `feasibility/`: reconstructability, raw support, conditional cell summaries,
  compressed class-count histograms, censoring, time-to-event/competing curves,
  eligibility and equal-seed mean/SD/min/max/n tables.
* `prediction/`: model scores, raw and support-qualified paired effects,
  common coverage, positive-probability calibration and seed summaries.
* `figures/`:9 PNGs at180dpi: prevalence, horizon variation, censoring,
  conditional outcome variation, feature support, raw prediction loss,
  calibration, distinct environmental/behavioral summaries and competing events.
* runtime/storage audit, verification receipt and study manifest.

The unsupported memory-increment and qualified paired-effect figures were
deliberately omitted; `figures/receipt.json` records NA reasons. No empty plot
or zero-effect substitute was fabricated. Three representative figures were
visually checked for readable labels and margins. Plotting uses no random jitter.

Runtime including hash verification/diagnostics, approximate decimal MB:

| Phase | Wall seconds | Process peak RSS MB |
|---|---:|---:|
| Archive inventory/integrity | 39.19 | 756.7 |
| Feasibility | 81.52 | 810.1 |
| Successful retrospective scoring | 38.24 | 899.4 |
| Figures | 2.27 | 220.1 |
| Compact derived event/label tables | 123.80 | 781.4 |

These are per-process high-water marks, not a sum of allocations; concurrent
test execution can affect wall time. The initial failed reporting pass is
preserved separately and is not included in successful-scoring time. New data
are about77 MB (mostly76.4 MB of compact derived joins), analysis about7.7 MB.
Only one case's detailed histories are loaded at a time; inventory uses chunked
readers. The approximately600 MB primary archive was not copied. Metadata records
sources, input sizes/hashes, missing counts, output bytes and timing. Generated
directories follow the existing git-ignore convention and remain saved locally.

## Reproduction

Use Python3.11+ and the existing optional numpy/pandas/matplotlib environment
(`.venv/bin/python` here). Run from repository root. These commands only read
archives and write fresh analysis destinations; they never start simulations.
Completed destinations refuse overwrite.

```bash
mkdir -p data/future_resource_v081_replay
cp data/future_resource_v081/design_receipt.json data/future_resource_v081_replay/
cp data/future_resource_v081/before.json data/future_resource_v081_replay/
python3 -m experiments.future_resource_analysis --phase audit --root data/future_resource_v081_replay --output analysis/future_resource_v081_replay
python3 -m experiments.future_resource_analysis --phase feasibility --root data/future_resource_v081_replay --output analysis/future_resource_v081_replay
python3 -m experiments.future_resource_analysis --phase freeze --root data/future_resource_v081_replay --output analysis/future_resource_v081_replay
python3 -m experiments.future_resource_tables --root data/future_resource_v081_replay --output analysis/future_resource_v081_replay
python3 -m experiments.future_resource_prediction --root data/future_resource_v081_replay --output analysis/future_resource_v081_replay
python3 -m experiments.future_resource_labels --root data/future_resource_v081_replay --output analysis/future_resource_v081_replay
python3 -m experiments.future_resource_plots --root data/future_resource_v081_replay --output analysis/future_resource_v081_replay
python3 -m unittest discover -s tests -q
# Example verification of the fresh repeat, using a saved complete test log:
python3 -m experiments.future_resource_verify --root data/future_resource_v081_replay --output analysis/future_resource_v081_replay --test-log data/future_resource_v081/release_tests.log
```

The saved study's verification is `analysis/future_resource_v081/verification.json`.
No new ecological seeds are run by any command above. The v0.8.0 archive path can
be supplied through `--archive`; its metadata must still point to intact selected
calibration inputs. Reproducing old scientific runs is outside this release.

## Interpretation and recommended next experiment

Level1(target variation) is present at fixed single tiles and for actual future
acquisitions, while wider-radius presence is often saturated. Levels2–3(extra
social information and predictive improvement) remain unresolved: missing
environmental observations and sparse conditional training support prevent a
qualified test. The retrospective comparisons provide no supported positive
result. Levels4–5(policy dependence and ecological benefit) are not new tests
here; v0.8.0's state-fragmentation and learning-history confounds remain.

The main limitations are previously examined seed reuse; only three independent
evaluation seeds; A-only early training/cross-policy and late-time shift; passive
agent/cohort selection; overlapping temporal origins; sparse X/action/social
cells; event-dependent binary inclusion and competing mortality; end-of-run
censoring; ecological consumption/population feedback; resource saturation; and
an intentionally compact predictor. Small raw improvements, backoff equality or
prediction failure cannot establish useful action selection or information
absence. Proximity/population changes establish no cooperation or language.

The smallest next measurement is **passive dense end-tick resource quantities**
or a compact fixed-coordinate future-label join, first validated in a bounded
unchanged-ecology run and frozen before new outcomes. The smallest missing field
is the exact resource array at non-hundred endpoint ticks. With current1200
cells, two resources capped5, a uint8 array representation would require1.2 MB
per500-tick run uncompressed (43.2 MB for36), plus metadata; actual compression
and any future larger capacity need validation. This is a storage estimate,
not a promise that denser data meet conditional support or yield social value.

Before claiming confirmation, freeze adequate training exposure, feature/action
support and genuinely new ecological seeds; do not call old seeds untouched.
If a separately preregistered measurement study still lacks a variable supported
target, a minimal **bounded temporal resource-availability intervention** may be
appropriate as a later independent ecology study. Do not implement or tune it
inside v0.8.1. No attraction, sharing, social reward, communication semantics,
cooperation or language mechanism was added. The frozen analysis stops here.
