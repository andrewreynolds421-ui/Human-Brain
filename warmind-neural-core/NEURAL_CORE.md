# Neural core — 2026-10-30

> **What the neural core is:** a spiking model of human association cortex
> (measured local wiring and physiology, 1.5–3 % of real neuron density,
> macaque long-range wiring, a hand-made action-selection circuit) that
> learns from the rewards RASPUTIN sends it. It does not think, read or
> talk; the chat stays Claude. See `synthetic-connectome/AUDIT.md`.


A live spiking simulation of a human cortical column running beside the
Warmind, visible on the dashboard, readable and perturbable by RASPUTIN.
Same deployment discipline as providers and dispatch: **inert until a flag is
set**, one patch to `main.py`, tests for every claim below.

## What it is, in one paragraph

About 1/10 of a 1 mm² column of human temporal cortex: 3,252 integrate-and-fire
neurons in 8 populations (layers 2/3, 4, 5, 6 × excitatory/inhibitory), 1.3 M
synapses. Wiring statistics come from the H01 electron-microscopy volume,
membrane constants from 413 patched human neurons (Allen Cell Types), and in
`full` mode the synaptic strengths and short-term plasticity from 17,490
tested human cell pairs (Allen Synaptic Physiology). It does not think or
talk. It is tissue dynamics: it fires, it responds to input, and its synapses
change when rewarded. The system prompt says exactly that to RASPUTIN.

## Files

| File | Status |
|---|---|
| `neural_core.py` | new — service, router, tool |
| `neural_panel.html` | new — dashboard panel (drop-in) |
| `main.py` | modified, 7 hunks (`main.py.neural.patch`) |
| `tests/test_neural_core.py` | new, 6 tests |
| `synthetic-connectome/` | new — the model (`synth/` package + `results/*.json` fits) |

## Ship with the flag OFF

```
NEURAL_CORE=0        # or unset
```

Verified: with the flag off the router answers 503 on every `/api/neural/*`
route, the `neural_core` tool is never appended to the tool list, the system
prompt is unchanged, and `tools_for()` counts are unchanged. `BUILD` bumped
to `2026-10-30.1`.

## Turn it on

1. Put the `synthetic-connectome` folder next to `main.py` (or set
   `NEURAL_CORE_PATH` to wherever it lives).
2. `pip install brian2 numpy` — Brian2 compiles with gcc if present
   (much faster); without a compiler it falls back to numpy and runs
   several times slower.
3. Set `NEURAL_CORE=1` and restart. The column builds and compiles in a
   background thread: **about 70 s the first time** (code generation),
   ~10 s on later starts (cached). `/api/neural/state` reports
   `status: warming` until then; nothing else waits on it.
4. Paste the three blocks of `neural_panel.html` into the dashboard and call
   `NeuralPanel.init({ api: API_BASE, accessCode: ACCESS_CODE, mount: '#neural-core' })`
   once. The panel polls once a second.

Optional:

```
NEURAL_CORE_MODE=membranes   # default: asynchronous-ish state, ~0.2–0.3x real time on 2 cores
NEURAL_CORE_MODE=full        # all measured synapses: periodic bursts, ~2x slower
NEURAL_CORE_SCALE=0.1        # 0.05 is ~3x faster (1,626 neurons) but can fall silent
NEURAL_CORE_CHUNK_MS=250     # biological ms per step; larger = faster, laggier panel
NEURAL_CORE_REALTIME=1       # never run faster than real time (only matters on a fast box)
```

## What RASPUTIN gets

A `neural_core` tool with four actions:

| action | effect |
|---|---|
| `status` | per-layer rates, regime (silent / asynchronous / synchronous / bursting), recent events |
| `stimulate` | a pulse of input into a layer (`L4` = sensory, `L23` = association, `L5`/`L6` = output). Use when something notable arrives. |
| `reward` | dopamine-like scalar in [−1, 1]. Excitatory synapses that were co-active in the last 300 ms are strengthened or weakened, then each neuron's total input is renormalised (synaptic scaling). Use when the user reacts to what the core did. |
| `reset` | restore the initial weights and state |

The tool is in `CORE_TOOL_NAMES`, so specialists can read the core too.
`reward` rewires the core (durable while it runs), so `neural_core` is in
`TAINTED_BLOCKED_TOOLS`: a turn carrying untrusted page text cannot touch it,
by the same reasoning that blocks `set_directive`.

## What the dashboard gets

The panel exposes, every second:

```
window.NeuralCore.activity   // 0..1 normalised mean rate — drive the sphere's idle glow
window.NeuralCore.burst      // true during a population burst — a flare
window.NeuralCore.state      // the full state object
window.NeuralCore.stimulate(layer, strength, duration_ms, label)
window.NeuralCore.reward(value, note)
```

and fires a `neural:state` event. A thumbs-up on a Direct Line reply can call
`NeuralCore.reward(1)`; a message arriving can call `NeuralCore.stimulate('L4')`.

## Thread model

Brian2 is not thread-safe. One worker thread owns the network; requests read
a snapshot it publishes after each step and queue commands it applies
between steps. No request blocks for longer than a dict copy. Build and
runtime errors are surfaced in `status: error` with the message, never
swallowed.

## Resources

One CPU core, continuously, while enabled. ~400 MB RAM at scale 0.1.
If the host is a 1-vCPU instance, run `NEURAL_CORE_SCALE=0.05` or leave the
flag off — the API is unaffected either way.

## Known limits

- Runs below real time. The panel shows simulation time, not wall time.
- `membranes` mode is synchronous rather than cleanly asynchronous; `full`
  mode bursts. Both are documented states of the underlying model
  (`synthetic-connectome/README.md`, open problems 5 and 8); the next model
  change (conductance-based synapses) is what moves it.
- In a bursting state almost every neuron is "recently active", so a reward
  touches most E→E synapses. The eligibility window (300 ms) is a parameter
  of `LiveColumn`; shorten it if the core is bursting.
- Nothing here persists across restarts: weights learned by reward are lost
  when the process stops. Saving/loading the weight vector is a small
  addition if it turns out to matter.

## Tests

`python -m pytest tests/test_neural_core.py -v` — flag-off inertness, the
`main.py` patch points, the tool contract, and a live run that builds a
small column, stimulates, rewards, pauses, resumes and resets it. The live
test takes about a minute the first time.


## Update: dendritic mode and model switching

`NEURAL_CORE_MODE=dendritic` runs the newest column: 12 populations
(fast-spiking and other interneurons per layer), two-compartment pyramidal
cells with each input on soma or dendrite, measured synaptic kinetics and an
in-vivo-like background. RASPUTIN can switch models while running:
`neural_core` action `set_model` with `model` = `membranes` | `full` |
`dendritic` (rebuild takes up to a few minutes; the request waits for it).
The dendritic mode is ~4x slower than `membranes`; the default is unchanged.
Tests: `tests/test_neural_core.py` (7, including a live dendritic run and a
model switch through the tool).


## Update: the learner model — RASPUTIN trains the network

`NEURAL_CORE_MODE=learner` (or `set_model` → `learner` at runtime) runs two
human-like prefrontal columns (areas 46d → 9/46d, joined by the measured
inter-areal pathway) that learn a cue → action mapping from reward
(`synthetic-connectome/synth/learner.py`, README "Stage 6b" / "Stage 7").

RASPUTIN's `neural_core` tool:

| action | what it does |
|---|---|
| `trial` with `cue` = `A` or `B` | shows the cue for 200 ms and returns the network's choice (action 1 or 2) and the spike counts of the two action groups (~10–25 s wall time) |
| `reward` with `value` in −1…+1 | trains the synapses behind that trial's choice; the reward may arrive any time later — the trial's synaptic "tags" wait for it. A reward with no trial waiting is refused. |
| `status` | adds learner progress: trials with feedback, mean reward over the last 20, whether a trial is waiting |

It learns whatever mapping is consistently rewarded (it has no built-in
"correct" answer). Tested: from chance (50%) to 73% over 300 trials on two
new networks with the standalone script, and end-to-end through this tool
(see README Stage 7). The default mode is unchanged (`membranes`). Tests: 8
(`tests/test_neural_core.py`), including a learner trial + reward through
the tool.

Learner synapses also form and disappear (`NEURAL_CORE_STRUCTURAL=1`, default):
every 10 rewards, synapses learning has weakened below 15% are removed and as
many new ones form at random sites; status reports the running totals.
In testing this raised accuracy over the last 100 of 300 trials from 73% to
82.5% (README Stage 8).

Learner surprise signal (`NEURAL_CORE_SURPRISE=3.0`, default): when recent
rewards fall below their long-run average (the rule RASPUTIN rewards has
changed), learning speeds up by up to 4x; status reports the current
surprise. With it, a reversed rule is relearned to 60% in ~150 trials
instead of ~235, while the first rule is learned better (82% vs 75%;
README Stage 10).

`NEURAL_CORE_MODE=learner8` (or `set_model` → `learner8`) runs the same task
inside all eight association areas, with plasticity on every long-range
input to the action cells; ~4–5x slower than `learner` (about 15–20 s per
trial here with fast mode). Tested: chance → 70.5% over the last 100 of 300
trials (README Stage 11).

Fast mode (`NEURAL_CORE_FAST=1`, default) runs both learner models about
twice as fast with less memory: Poisson inputs are drawn in one vectorised
step, and short-term-plasticity and inhibitory-plasticity state is kept per
neuron instead of per synapse. The synaptic part is exact (identical spike
trains to the original code); the merged inputs draw different random numbers
with the same statistics. Learning was re-tested in all eight areas
(README Stage 12). `NEURAL_CORE_FAST=0` restores the original code.

Quick learning (`NEURAL_CORE_QUICK=1`, default): a higher learning rate that is
boosted further while rewards are at chance and eases off as they come in
(expected uncertainty). In all eight areas, accuracy over trials 101–200 rose
from 67–77% to 80–82%, 75% was reached after 62–82 trials instead of 83–166,
and after a rule reversal the network was back to 60% in 56–72 trials instead
of 79–145 (README Stage 13). `status` reports the current `learning_rate_x`.
`NEURAL_CORE_QUICK=0` restores the previous rule.

Column calibration (`NEURAL_CORE_COLUMN=v2`, default): the Stage 25 column,
in which every cell population fires at or near its measured reference rate
(the earlier column had its non-fast-spiking interneurons silent in layer 2/3
and 2–7× too fast below). On it the learner reaches 96–97 % on the cue task
(was 82–83 %), gets to 75 % in 30–60 trials, and eight-area trials take about
half the time. `NEURAL_CORE_COLUMN=v1` restores the previous column.

Lean memory (`NEURAL_CORE_LEAN=1`, default): no reset snapshot is held,
wiring tables the running learner no longer reads are freed, build
temporaries are kept small, and build scratch memory is returned to the
system. Identical behaviour (same learned weights and choices). `reset` (and
`set_model`) frees the old network and rebuilds from the same seed, which
gives the same starting state; it takes as long as start-up. `learner8` runs
in ~1.05 GB and peaks at ~1.5 GB while starting (was 2.4 GB / 2.5 GB).

The spike record no longer grows: earlier versions kept every spike, plus
copies, so a long-running core slowly used more and more memory (a 1000-trial
run passed 4.8 GB). Memory now stays flat however long the core runs.

Model size: `learner` runs at double the previous neuron count by default
(`NEURAL_CORE_LEARNER_SCALE=0.03`), with action selection scaled to pool size
(README Stage 15); it learns sooner and better than the smaller one.
