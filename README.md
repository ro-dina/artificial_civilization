# Artificial Civilization Simulation — v0.1

A small, headless artificial-life simulation for future experiments in emergent
behavior. The current world contains finite food and water, moving agents, local
perception, random decisions, aging, death, and meaningless integer signals.

**Language, agriculture, economics, money, writing, religion, technology, culture,
social classes, and civilization are not implemented.** Such phenomena should
ideally emerge from more primitive mechanisms rather than being directly scripted.
There are no jobs, emotions, predefined signal meanings, or crafting recipes.

## Run

Python **3.11+**, standard library only. Run commands from this directory; no
installation, GUI, or network access is required.

```bash
python3 main.py
python3 main.py --seed 42 --ticks 50 --csv data/smoke.csv
python3 main.py --seed 42 --ticks 20 --population 10 --events data/events.jsonl
python3 main.py --help
```

The runner stops at the requested tick limit or extinction, prints periodic and
final status, and creates no output files unless requested. Output paths are
overwritten when explicitly provided. Detailed event logging can be large and is
disabled by default. Generated files under `data/` are ignored by Git.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Tests cover deterministic world generation and full simulation replay, bounded
movement, limited immutable perception, consumption and conservation, survival
thresholds, signal range and lifetime, configurable vocabularies, output formats,
configuration validation, and independence from global random state and logging.

## Architecture

```text
main.py                    CLI, output files, stopping conditions
config.py                  Default constants and validated per-run Config
world/tile.py              Food and water counts
world/world.py             Grid generation, bounds, queries, movement, consumption
agents/human.py            Body state and physiology
agents/brain.py            Brain protocol, Action, Decision, RandomBrain
agents/observation.py      Immutable local sensory records
systems/communication.py   Meaning-free signals with spatial lookup
simulation/simulation.py   Seeded setup, tick loop, aggregate accounting
simulation/statistics.py   Aggregate snapshots and streaming CSV writer
simulation/events.py       Optional event sink and streaming JSONL writer
tests/test_basic.py        Automated tests
data/                     Optional experiment output
```

`Human` owns its body state and a replaceable brain. A brain implements
`choose_action(human, observation) -> Decision`. It may read its own body and the
provided observation; it must return decisions without mutating the body. No
world reference is passed to a brain. An `Action` selects movement, eating,
drinking, or waiting; `Decision.signal_id` optionally accompanies that action.

The observation consists of the tick number, nearby resource records
`(dx, dy, food, water)`, and heard signals `(signal_id, sender_id, dx, dy)`.
Offsets are relative to the observing agent. Records are immutable snapshots,
not references to world tiles. Brains receive no resource information outside
their perception radius. Different sensory modalities can later extend this
interface without changing the body model.

## Physical rules and tick order

- Coordinates are integer grid cells; north decreases `y`. Movement is one
  cardinal step, with hard boundaries and no wrapping. Agents may share cells.
- `INITIAL_FOOD` and `INITIAL_WATER` are total **resource units**, independently
  distributed with replacement. A tile can hold multiple units of both resources.
- Perception and hearing use inclusive square neighborhoods (Chebyshev distance).
  Radius zero includes the current cell. Perception is clipped at world edges.
- Each tick shuffles the living agents using a seeded ordering stream. Each agent
  observes, chooses, acts, optionally emits, gains hunger and thirst, ages one
  tick, and is checked for death. Actions are sequential: later agents can see
  resource depletion caused by earlier agents. Shuffling avoids permanent
  priority by ID; it does not make actions simultaneous.
- Eating/drinking consumes one available unit on the current tile and reduces
  the corresponding need by its configured relief, clamped at zero. Missing
  resources and blocked movement waste the action; survival costs still apply.
  Random agents can consume resources even when their need is already zero.
- Death occurs at `hunger >= max_hunger`, `thirst >= max_thirst`, or
  `age >= max_age`. Set `max_age=None` to disable aging deaths. If multiple limits
  are crossed together, the logged cause prioritizes hunger, thirst, then age.
- Dead bodies are removed from the active list after the tick; there is no
  retained body history. An already extinct simulation's `step()` is a no-op.
- Energy is stored but unchanged in v0.1. Reproductive types `0` and `1` are
  assigned at creation and currently have no behavioral effect. IDs are unique
  within a simulation, starting at zero.

Signals are integers in `[0, signal_vocab_size)`, without predefined meanings.
An emission is located at the sender's position **after its action** and is heard
only during the **next tick**, then expires. Range is measured between that
emission position and the receiver at observation time; receivers do not hear
themselves. An emission persists for that next tick even if its sender dies.
This one-tick delay avoids same-tick signal delivery depending on update order.
Signals have no cost in v0.1. `RandomBrain` emits with `signal_probability`; set it
to zero for silent random agents. Vocabulary sizes such as 4, 16, 256, and 4000
use the same representation. Spatial buckets avoid scanning all signals for
every receiver.

## Configuration and reproducibility

Edit the defaults in `config.py`, or construct an immutable `Config` per run:

```python
from config import Config
from simulation.simulation import Simulation

settings = Config(world_width=20, world_height=20, initial_population=30,
                  signal_vocab_size=256, max_age=None, max_ticks=200)
sim = Simulation.create_default(seed=42, config=settings)
for _ in range(settings.max_ticks):
    if sim.population == 0:
        break
    sim.step()
sim.print_status()
```

Every random stream derives from the configuration's seed using named
`random.Random` instances: world generation, agent placement, update order, and
one stream per brain. No global random state, wall clock, or unordered set
iteration influences the simulation. Output logging does not consume randomness.
The same configuration, seed, code, and Python version reproduce the same state,
statistics, and events. Record all four when comparing experiments; exact replay
across future code or Python versions is not promised. `create_default(seed=...)`
overrides the config seed, including when the supplied seed is zero.

`max_ticks` and `status_interval` control the CLI runner; `step()` advances a
single tick and does not impose a run-length limit.

## Statistics and optional events

`sim.statistics` contains `tick`, current `population`, and **cumulative** `deaths`,
`food_consumed`, `water_consumed`, and `signals_emitted`. CSV includes the initial
tick-zero snapshot and one row per completed tick. Subtract adjacent rows for
per-tick totals. `CSVStatisticsWriter.write(sim.statistics)` can also stream
snapshots from a custom experiment runner. The simulation keeps only the latest
aggregate snapshot, not an ever-growing list.

An optional callable `event_sink` receives `Event` records. The CLI's `--events`
uses `JSONLEventWriter`: one `run_started` record containing the effective config,
then one `agent_step` record per acting agent. Step events include the agent ID,
tick, starting position, body state before and after acting, complete local
observation (nearby resources and received signals), action and success,
emitted signal, consumption, and death cause. Custom sinks can sample/filter
events or calculate experimental measures without keeping them all in memory.
CSV and JSONL writers stream to caller-owned files; no event history or detailed
event payloads are created by the default simulation.

## Limitations and future direction

Random decisions do not learn from observations, so this is a baseline for
checking mechanics, not evidence of emergent communication. Finite resources,
no replenishment, and no reproduction make eventual extinction expected under
the default survival costs. There are no obstacles, plants, objects, combat,
communication semantics, learning, inheritance, or environmental cycles. The
brain interface is a programming contract, not a sandbox against malicious
custom brain code. Sequential competition is an explicit modeling choice.

Memory use is bounded with respect to elapsed ticks when outputs are streamed:
the world, living agents, latest statistics, and recent signals remain resident.
Perception allocates only local records; detailed logging adds serialization
cost. Large runs have not yet been benchmarked. There is no rendering,
multiprocessing, checkpointing, or experiment scheduler.

A sensible next step is a small seeded baseline experiment measuring survival,
resource use, and runtime while varying resource density and perception radius.
This establishes a reference before adding adaptive controllers. Later work can
add reproduction and inheritance, mutation, memory/learning, neural or evolving
brains, multiple senses, generic objects and material interactions, biological
growth, environmental cycles, persistent marks, and intergenerational learning.
None of those systems is implemented in v0.1; civilization-like outcomes remain
research questions rather than scripted development stages.
