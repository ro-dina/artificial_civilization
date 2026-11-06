# v0.8.2 — Dense future-resource observation and final predictive evaluation

## Research question and conclusion

With exact end-of-tick resource states, does existing social perception or
remembered-direction information improve prediction beyond original X and the
selected action? This release completes one frozen, prospective new-seed study.
It adds passive measurement and offline predictors, with no new agent knowledge
or ecological/biological/learning mechanisms.

**Social perception improves prediction in some supported settings.** For the
primary future-water target, both model families improve in E2 (localized,
normal supply) in all three modes and all three evaluation seeds. The regularized
family improves decision-weighted log loss and Brier in every evaluated primary
case. The hierarchical family worsens log loss in E0/E1; E3 primary hierarchical
comparisons remain NA because their training support gate failed. Benefits are
therefore model- and condition-dependent, not a universal information claim.

**Additional memory has no robust cross-family improvement.** Hierarchical
memory worsens log loss in every qualified primary and secondary case. The
regularized memory differences are much smaller, sometimes improve and sometimes
reverse across seeds, subsets or equal-agent metrics. Prediction is distinct
from useful learned action selection, survival benefit, cooperation or language.

The resource-prediction measurement cycle is closed here. No further target,
threshold, regularization, capacity or nuisance-control search was performed.

## Historical evidence and unchanged simulator

Baseline commit: `7463340919bd12ce5ebca22db1b035f3e980222d` (v0.8.1).
README, v0.8.0/v0.8.1 reports, resource/tick code, observers, encoders, streams,
predictors, compatibility fixtures and receipts were inspected first. Baseline
658 tests passed in102.324s before edits. A3704-file source/archive inventory was
saved; only README and `.gitignore` are permitted existing-file edits.

v0.8.0's36 primary cases and v0.8.1's7,685,665-decision audit motivated this
study. Prior single-tile targets varied, but environmental labels covered only
about1.1% of origins, and A-only200-tick training had sparse support. Those
previously examined seeds are not reused as confirmation seeds. Their artifacts
are preserved rather than overwritten or copied into the new training set.

The same40×30 grid, initial600 food/600 water, capacities5, primitive physiology,
food/water consumption and old regeneration/placement RNG streams remain.
The ecology pairs(supply multiplier, localization strength) are E0(1,0),
E1(.5,0), E2(1,.75), E3(.5,.75). Four fixed radius4 patches, their geometry
algorithm and existing capacity-mediated supply accounting are unchanged.
Equal supply remains conditional on eligibility, as documented in v0.8.0; no
new claim of unconditional equal realized supply is introduced.

A uses original Physical Q input; B adds its existing visible-social token;
C adds remembered direction as before. All have the same passive memory
configuration32/threshold1/LRU, read before current visual-batch commit. Original
Physical and Vocal Q, exploration, rewards, mutation, reproduction, mortality,
movement, senses and both capacities256 are untouched. RandomBrain is untouched.
No food sharing, toxins, temporal patches, signal meanings or social reward exist.

## Dense observer and exact storage

`simulation/dense_resources.py` contains a standard-library streaming writer,
reader and `DenseResourceObserver(ResourceObserver)` adapter. The adapter uses
the existing callback; `Simulation` and `Config` were not modified. Dense
measurement is disabled in ordinary/default runs. The reusable writer itself
only reads world quantities and finalized aggregate statistics.

For every actual tick, including0, the adapter calls the original observer,
then records the actual world after all actions/consumption, physiology/death,
learning, queued births, auditory-buffer advance **and statistics finalization**.
The world is not regenerated for measurement. Newborns still act next tick.
There is no separate tick-start file, and no conflation with old tick-start
distance-map diagnostics. A statistics/tick mismatch stops recording.

Format schema1 is `*.resources.u8.gz` with sidecar `.json` and a streaming
`.stocks.csv` ledger. The gzip stream has magic`ACRD082\n`, a little-endian
uint32 JSON-header length, the header, then consecutive records:

```text
uint64 little-endian tick
1200 uint8 food quantities, row-major y,x
1200 uint8 water quantities, row-major y,x
```

Metadata records per-frame shape[2,30,40], dtype, axis/resource order, quantity
limits, phase`end_tick_after_statistics`, tick range/count, compression(gzip6),
source hash, compressed and uncompressed SHA256 and ledger checksum. Byte order
does not apply to uint8; the tick encoding is explicit. Gzip filename/timestamp
are fixed for deterministic bytes. The writer rejects noninteger, negative,
out-of-capacity quantities and unbounded/capacity>255 configurations rather than
truncating future resource models. A different dtype requires a new schema.

Each frame verifies, separately for food and water:

```text
initial units + cumulative regeneration − actual consumption = world stock
```

The reader verifies metadata/phase, compressed checksum, CRC, shape, consecutive
ticks, quantity bounds, record count and raw-stream checksum. Analysis loads
only one case's small resource cube (about1.2 MB uncompressed for501 frames).
The writer retains no agent, dead body, Q table or resource-history array.

## Purity and historical compatibility

Four30-tick scenarios(E0/A,E0/C,E3/B,E3/C,seed0) were run with the old resource
observer, dense observer, and a repeated dense observer:12 short validation
runs. Complete world/bodies/genomes, Physical/Vocal Q values and LRU order,
pending transitions, named RNG states, actions and statistics matched. All15
old diagnostic streams also matched byte-for-byte on/off, and repeated dense
files were byte-identical. Unit tests additionally compare no observer versus
dense observation across A/B/C and E0/E3, at every step and after observer finish.

Three100-tick E0/A/B/C seed0 controls matched archived v0.8.0 fingerprints,
action digests and statistics exactly. This comparison includes whole state
and RNG; it is not inferred from population equality. There were no unexplained
compatibility failures and no historical fixture changes. These15 validation
runs are separate from the planned72 full scientific runs.

## Frozen plan, stages and new seeds

The tracked design is `experiments/dense_resource_design.json`. The actual
pre-run receipt `data/dense_resource_v082/preregistration.json` records its hash,
UTC creation time and scientific source hashes. It was written before validation
and any new training/evaluation ecological run. No scientific implementation
correction or plan rewrite was needed after freezing. Software versions and
numeric-thread settings are in `run_environment.json`; no packages were installed.

Stages executed:

| Stage | Seeds/cases | Horizon | Decision |
|---|---|---:|---|
| Instrumentation/historical validation | 15 short controls, seed0 | 30/100 | Exact checks passed |
| Pilot | seed10, E0–E3 × A/B/C,12 cases | 100 | Training allowed |
| Training | seeds10/11/12,36 cases | 500 | Full same-mode fits and support review completed |
| Evaluation | new seeds20/21/22,36 cases | 500 | Allowed by frozen training gate; all completed |

All72 full runs reached500 ticks, with no extinction during that horizon. There
were no additional seeds,1000-tick extensions or ecological parameter changes.
Pilot seed10 is the same trajectory prefix as training seed10, not another
independent replication and not duplicated in model training. Prefix action/
physiology hashes match. `max_ticks` differs between those configurations, so
their full configuration-containing fingerprints are not falsely called equal.

The pilot was technical/marginal feasibility only: all12 cases had enough
population/exposure/actual consumption, complete dense grids and a variable T1.
T1 prevalence was.174–.400, coverage.927–.945, with171–210 actors/case. Pilot
within-run cell coverage was measured but not treated as independent held-out
support or used to choose favorable ecological conditions.

Training review used leave-one-training-seed-out **same-mode** support: two
other500-tick seeds supplied counts for each held-out training seed. This removes
the previous A-only-to-B/C policy transfer. It did not compare predictive gains.
All72 logistic fits converged, in at most6 Newton steps, maximum final gradient
infinity norm8.4151e−9≤1e−8. All fitted model files and sufficient-count files
were hashed before evaluation, and the stage gate verifies those hashes.

## Targets, alignment and missingness

**Primary T1:** water>0 at origin pre-action(x,y), exact end tick t+5.
**Secondary T2:** endpoint water==0 at the same tile, conditional on origin
pre-action water>0. Recovery is not combined with depletion. Food was recorded
and checked for integrity, but no alternative food target replaced these targets.

Coordinates are fixed from the actual origin record. Future agent positions,
future actions and later same-tick observations never select a location or enter
features. The endpoint includes other agents' future consumption and therefore
is not exogenous or independent of population behavior. An originating agent's
death does not invalidate its environmental label. Origins after tick495 are
administratively unavailable; there is no interpolation, padding or fabricated
world after the horizon. No interior snapshot was missing.

Compact `*.labels.csv.gz` files retain only sequence, exact endpoint water,
T1/T2 labels and eligibility reasons. The original acquisition stream supplies
agent/tick/catalog/social linkage; complete feature histories are not copied
per target. Catalog IDs are local and are joined to canonical original X JSON.
Endpoint quantities are diagnostic labels, never predictive inputs.

The B10 descriptive reference retains v0.8.1 actual next-ten-own-decision
semantics: observed acquisition is positive, complete alive follow-up can be
negative, death is competing, and administrative/gap missingness is separate.
Consumption wins a same-future-decision death tie. Its prevalence ranged.452–.548,
observed-label coverage.918–.934, and no history gap occurred. No B10 predictor
was fitted or promoted to a primary outcome.

## Support gates and qualifications

Exact-cell gates remain: prevalence[.10,.90],≥100 positives and negatives,
≥100 actors,≥20 training observations/cell,≥20% H0/H1 common coverage,≥50%
exact-label coverage,≥2 X/action strata with≥20 of each class, and≥2 supported
B tokens. Memory also requires≥20% H0/H1/H2 common coverage and≥2 supported RD
values. Every scored subset also requires100 of each class for a qualified
effect. These heuristics are not significance guarantees.

R has a separately frozen category-level gate: each active feature level must
have≥20 training observations, supported rows≥95%, plus the target variation,
actor, coverage and B/RD variation requirements. This never substitutes for
strict exact-cell qualification. R estimates outside common exact cells are
explicitly **model-dependent generalization**.

Training OOF H0/H1 coverage ranged.163–.593 across target/case rows;
R feature coverage was≥.9998. E3's primary H1/H2 gate failed for all three modes;
those analyses stayed NA even though full-three-seed models subsequently had
more held-out support. No threshold was rounded or relaxed, and later support
did not revive a failed training gate. Every other H stratum passed; all R
strata passed and converged, justifying all36 planned evaluation cases.

Held-out T1 labels cover7,659,198 of7,765,759 origins, with1,499,099 positives
and6,160,099 negatives;106,561 origins exceed the horizon. T2 has1,807,109
reference-eligible origins,1,785,974 exact labels,649,668 positives,1,136,306
negatives and21,135 administrative endpoints.5,958,650 origins were not at risk.

| Ecology | T1 prevalence range | T1 exact coverage range | T1 H01 common range | T2 prevalence range |
|---|---:|---:|---:|---:|
| E0 | .170–.200 | .986–.988 | .396–.445 | .369–.443 |
| E1 | .151–.190 | .988–.990 | .305–.339 | .310–.387 |
| E2 | .190–.227 | .980–.986 | .381–.422 | .339–.421 |
| E3 | .171–.236 | .981–.988 | .218–.283 | .286–.404 |

These ranges describe nine cases/ecology, not nine independent seeds. T2
coverage was.985–.992, exact H01 coverage.398–.637, and R feature coverage≥.9998.
There are three independent evaluation seeds **per condition**. Raw/seed-summary
support tables preserve all counts, strata and denominators. `scores.csv`'s
`exact_cell_qualified` describes held-out support alone; use
`joint_qualification.csv` and `paired_by_seed.csv` for the required combined
training+evaluation qualification.

## Predictors and paired scoring

H0/H1/H2 reuse the existing hierarchical binary count model, minimum20,
Dirichlet strength5, parent backoff: X,A; X,A,B; X,A,B,RD. Integer sufficient
counts are exactly equivalent to repeated legacy training observations.
H2's parent is H1, so the memory increment does not include perception's gain.

R0/R1/R2 are one fixed additive categorical logistic family. One-hot categories
encode the original three need bins, four resource-direction components,
auditory kind/ID/coarse direction/short-memory age and selected action; B then
RD are appended. Dimensions including intercept are69/80/86. Signal IDs are
unordered categories with no intrinsic similarity. Auditory age is already in
X; it is not the agent's chronological age. No identity, absolute location,
patch geometry, chronological time, genome or diagnostic resource quantity is
used. X-only reference models were not added.

The objective is mean decision negative log likelihood plus.5×.001 times the
squared non-intercept coefficients. There are no class weights or random
initializations: zero slopes, training-prevalence intercept, deterministic
damped Newton, analytic gradient/Hessian, tolerance1e−8, maximum100 iterations,
at most30 line-search halvings. Exact repeated feature patterns are collapsed
with their actual counts, not sampled or reweighted equally. No hyperparameter
search was run. This offline family supplies no learning weights to agents.

Both families train separately within each ecology and learning mode, on only
the three designated training seeds through500 ticks. All eligible evaluation
origins are scored identically across nested heads. Common H01/H012,
feature-supported R01/R012 and feature-supported origins outside exact common
cells are separately recorded. Rowset hashes, counts and class prevalence
identify every comparison population. No random decision split or agent-ID
leakage occurs.

Metrics are log loss(nats), binary Brier(p−y)^2, positive-probability calibration,
and equal-agent loss. Per-run scores precede equal-seed mean/sampleSD/min/max/n.
No decision-level significance test or confidence interval is used. Counts of
directions of effects below are descriptive repeated-condition results, not
additional independent ecological replication.

## Primary perception results

All-label paired log-loss differences, mean±sample SD over seeds20/21/22;
negative is better. Every numerical cell has n=3; E3/H is NA from training.

| Ecology/mode | H1−H0 | R1−R0 |
|---|---:|---:|
| E0/A | +0.0012762±0.0002202 | -0.0016057±0.0001508 |
| E0/B | +0.0009786±0.0004425 | -0.0018187±0.0000784 |
| E0/C | +0.0009902±0.0002512 | -0.0019587±0.0000957 |
| E1/A | +0.0027534±0.0004017 | -0.0008506±0.0000905 |
| E1/B | +0.0025787±0.0001321 | -0.0011032±0.0001150 |
| E1/C | +0.0020166±0.0001489 | -0.0009743±0.0001042 |
| E2/A | -0.0018156±0.0004281 | -0.0004307±0.0001849 |
| E2/B | -0.0028402±0.0005967 | -0.0005251±0.0000488 |
| E2/C | -0.0025406±0.0006826 | -0.0005108±0.0001141 |
| E3/A | NA | -0.0002812±0.0001410 |
| E3/B | NA | -0.0005778±0.0001426 |
| E3/C | NA | -0.0005189±0.0001010 |

H1 improves T1 in E2/A,B,C for all three seeds and also improves Brier and
equal-agent log loss there. On the H01-common subset its mean changes are
−.004705/−.006893/−.006079 in A/B/C. E0/E1 H1 worsen log loss; some Brier
differences improve, so metrics should not be collapsed into one universal loss
claim. E3 primary H raw backoff scores are preserved as unsupported diagnostics.

R1 improves T1 decision-weighted log loss and Brier in every one of the36 cases.
Equal-agent log loss improves in35/36 individual seed cases; all12 condition
means improve, but one E3/A seed reverses. This is repeatable predictive evidence
under the specified R family, not an exact-cell or causal information guarantee.

Outside H01 common support, R1 has negative mean T1 changes in E0/E1/E3, but
E2/B and E2/C have positive mean changes(+.000056 and+.000133). Therefore its
overall gain must not be described as uniform improvement on unseen/sparse
joint contexts. Exact-cell support, feature support and OOS performance remain
three different measurements.

## Secondary depletion and memory results

T2 perception comparisons, same definitions/units and n=3:

| Ecology/mode | H1−H0 | R1−R0 |
|---|---:|---:|
| E0/A | -0.0051106±0.0009160 | -0.0116498±0.0005409 |
| E0/B | -0.0064463±0.0010485 | -0.0121930±0.0006705 |
| E0/C | -0.0067862±0.0000829 | -0.0128568±0.0004018 |
| E1/A | +0.0011471±0.0004786 | -0.0074301±0.0011138 |
| E1/B | +0.0003441±0.0005686 | -0.0076879±0.0012187 |
| E1/C | -0.0005391±0.0005406 | -0.0076307±0.0004643 |
| E2/A | -0.0126296±0.0007565 | -0.0224293±0.0027392 |
| E2/B | -0.0142427±0.0006385 | -0.0228336±0.0016655 |
| E2/C | -0.0134479±0.0009732 | -0.0218585±0.0015979 |
| E3/A | -0.0026642±0.0008329 | -0.0096324±0.0011272 |
| E3/B | -0.0046086±0.0010140 | -0.0115487±0.0022557 |
| E3/C | -0.0048138±0.0005111 | -0.0107938±0.0011248 |

R1 improves T2 log loss and Brier in every evaluation case and improves
equal-agent log loss in all36. H1 improves consistently in E0/E2/E3; E1/A worsens,
E1/B varies across seeds, and E1/C improves. R1 mean outside-exact-cell depletion
gains are negative in all12 conditions. These secondary findings do not replace
the primary target or prove a policy benefit.

H2 worsens log loss relative to H1 in every qualified T1 case and every T2 case,
including the strict H012-common subset. More exact conditioning can increase
finite-sample variance; this is not proof of zero underlying memory information.

R2 has very small mean improvements, with29/36 seed cases lower and7/36 higher
log loss for each target. For C-mode examples, additional memory changes are:

| Ecology/C | T1 H2−H1 | T1 R2−R1 | T2 H2−H1 | T2 R2−R1 |
|---|---:|---:|---:|---:|
| E0/C | +0.0026510±0.0001421 | -0.0000086±0.0000049 | +0.0008115±0.0001370 | -0.0000782±0.0000323 |
| E1/C | +0.0029286±0.0001732 | -0.0000011±0.0000092 | +0.0009223±0.0000607 | -0.0000251±0.0000801 |
| E2/C | +0.0016268±0.0001900 | -0.0000158±0.0000030 | +0.0010939±0.0002666 | -0.0000875±0.0000382 |
| E3/C | NA | -0.0000076±0.0000081 | +0.0009801±0.0000407 | -0.0000490±0.0001020 |

Equal-agent and outside-cell results can reverse these tiny differences. For
example T1/E0/C has mean decision-weighted R2−R1≈−8.6e−6 but equal-agent
log-loss change≈+3.1e−5; E1/C and E3/C also reverse under equal-agent averaging.
R2 is therefore weak/model-dependent memory evidence, not a robust cross-family
or individual-benefit result. No extra target/regularization experiment followed.

## Physical Q, vocal observations and interpretation

Physical Q still has capacity256. First-agent-state fractions in E0/A,B,C are
.683/.803/.807 and in E3 .586/.729/.724. Mean visits per agent state are only
1.24–1.71; Q occupancy is about49–67 rows averaged over actual exposures, not
an assertion that every agent fills256 rows. E0 mean eviction totals are
164/1285/1035 in A/B/C; raw exposure denominators are retained.

Read-only retained-pair probes(≥2 visits,≥1 actual TD update) give final policy
TV means B/C: E0 .342/.351, E1 .356/.358, E2 .423/.427, E3 .425/.436.
Supported final pairs range714–2293 per B/C run, with≥5-visit results also saved.
These are observed Q-row dependence diagnostics, not randomized social
interventions. Original fragmentation, selective visitation and learning-history
confounds remain. No memory-only Q probe was newly introduced.

Vocal Q is unchanged. Emission is about.941 per opportunity. Identified/masked
fractions are roughly.319/.681 in E0/A and.257/.742 in E3/C; final mean living
Vocal Q row counts are64.74 and58.38, respectively. Complete signal-choice
histograms and hearing/masking exposures are retained. Changes in arbitrary
signal frequency do not establish intended signaling, semantics or language.

Levels are distinct: target variation and qualified prediction gains now exist;
additional memory has weak/inconsistent evidence; Physical Q dependence was
already present; useful action selection and ecological benefit are **not
causally tested** by this offline study. R pools information across many agents
and seeds and generalizes across states, unlike each live exact-state Q learner.
It is not a capability newly installed in the simulated population.

## Verification, costs and saved artifacts

All658 original tests are preserved. The implementation suite passed681 tests
(658 existing+23 new). New tests cover timing/tick0, shape/dtype/capacity,
compression/checksums, mass balance, exact fixed-coordinate labels, death and
boundary handling, no feature/position leakage, H equivalence, logistic analytic
derivatives/convergence/determinism, feature support, seeds and stage gates.
The final full suite passed681 tests in89.325s. Its log and checksum are saved
in `data/dense_resource_v082/final_tests.log` and the verification receipt.

The completed final audit verified3702 protected-file hashes, complete on/off/repeat and
archived E0 proofs, all dense-frame checksums/balances, sparse-world-grid agreement,
pilot/training prefix hashes, frozen model/source inputs, support-mask counts,
same-decision score hashes and NA handling. No original core/config/test source
changed.95 dense files contain37,835 frames including validation/pilot;
the72 full runs alone contain36,072 frames, every run tick0..500. Each resource
stays in0..5, and every mass-balance error is zero.

Measured costs (decimal MB; process cumulative high-water RSS, not summed
allocation):

| Phase | Wall measurement | Largest process RSS |
|---|---|---:|
| Validation | 21.43s total for15 short controls | about146 MB |
| Pilot runs | 39.49s summed case time | 149 MB |
| Training runs | 2249.36s summed case time,27.92–106.77s/case | 580 MB |
| Training review/fits | 141.99s;72 logistic fits total38.05s | 811 MB |
| Evaluation runs | 2142.72s summed case time,19.40–100.56s/case | 558 MB |
| Evaluation labeling/scoring | 120.44s | 960 MB |

Training/evaluation each used three independent existing CLI processes, one per
seed. Summed case times are not elapsed end-to-end time. No multiprocessing
framework was added. Fingerprinting and original optional diagnostic streams
contribute substantial runtime; no complete Q history is retained.

The72 full-run dense gzip files occupy7,975,431 bytes (about8 MB), versus about
1235.5 MB of existing-schema decision/diagnostic streams for those runs.
The complete new data directory is about1.32 GB, including labels, sufficient
counts, models, validation and logs; analysis/figures are about13 MB. The original
archives were not duplicated. Actual final totals/hashes are in the manifest.

Saved paths:

* `data/dense_resource_v082/`: inventory/baseline tests, preregistration,
  environment, validation/legacy/pilot/training/evaluation cases, dense binary
  sidecars/ledgers, compact labels, training sufficient counts/coefficients,
  gate receipts and logs.
* `analysis/dense_resource_v082/`: pilot/training support, convergence and
  qualification, held-out scores/paired differences/common populations,
  calibration, ecological/Q/vocal diagnostics, per-seed signs, joint
  qualifications, runtime, integrity/verification receipts and final manifest.
* `figures/`:14 final PNGs at180dpi: resource integrity, primary/secondary
  prevalence, label coverage, feature support, H/R loss, perception/memory
  increments, per-seed pairs, calibration, Q fragmentation, outside-exact
  generalization and vocal observations.

Three representative figures were visually inspected. E3/NA margins were
corrected without changing scientific code, data or model scores. Original
layout drafts and their source copy are retained in `figures_initial_visual_review/`
and the data directory; final figure checksums identify the delivered version.
Generated directories use the repository's existing git-ignore convention.

Changed existing files: README and `.gitignore`. Added:

```text
simulation/dense_resources.py
experiments/dense_resource_design.json
experiments/dense_resource_models.py
experiments/dense_resource_data.py
experiments/dense_resource_study.py
experiments/dense_resource_analysis.py
experiments/dense_resource_reporting.py
experiments/dense_resource_verify.py
tests/test_dense_resources.py
docs/dense_resource_prediction_v082.md
```

## Reproduction and limitations

README contains the complete exclusive-destination workflow. Use Python3.11+
with the existing optional numpy/pandas/matplotlib environment; the simulator
and binary observer require only the standard library. Numeric-library threads
were limited to1 for recorded deterministic analyses. Copy the baseline inventory
to a fresh root, preregister, validate, pilot/review, train/review, then evaluate
only if allowed. Models are frozen before new evaluation outcomes. The command
`python3 -m experiments.dense_resource_verify` audits the actual saved study after
the final test log exists. Do not rerun into completed destinations.

Only three independent evaluation seeds per condition limit certainty. Same-mode
500-tick training reduces the prior policy/time mismatch but does not remove
within-run nonstationarity, evolutionary/cohort selection or differences in
resource/population feedback. Conditioning on selected action is a predictive
estimand, not a causal policy intervention. Targets depend on future behavior.

X is a compressed controller state: raw visual tile quantities contain more
information than its resource-direction encoding. A gain beyond X is not proof
of information absent from the entire raw visual observation. R further imposes
additive main effects rather than arbitrary X/action interactions; extra B/RD
can compensate for model misspecification/regularization as well as carrying
resource cues. Exact-cell heuristics and feature-level coverage are not formal
identifiability guarantees. Sparse joint contexts, model dependence, repeated
origins and agent heterogeneity remain; no decision-level significance or
survival/cooperation claim follows.

## Next experiment: separate v0.8.3 ecology

The current measurement cycle stops. No additional seeds, predictor tuning,
Q-capacity change, nuisance redesign or alternate horizon search is proposed.
The next study should be a separate preregistered **temporally variable resource
patch** intervention, with a bounded exogenous availability process and explicit
supply accounting. Keep local observations primitive and withhold the latent
environment schedule from agents. Test whether existing social observations are
useful under that new uncertainty without assigning social rewards, attraction,
cooperation or vocal meanings. Its design needs review before implementation;
no such intervention was implemented or run in v0.8.2.
