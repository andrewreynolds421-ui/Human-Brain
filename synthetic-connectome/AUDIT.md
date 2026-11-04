# Audit — 4 November 2026

Where the project stands against the roadmap agreed on 30 October (seven
stages to a full brain, plus the Warmind integration discussed right after),
what was done well, what was not, and what should come next. Every number
below is from a file in `results/` or a script in `scripts/`.

## 1. The stages

### Stage 1 — one column in the right state  ▸ done, with caveats

**Goal as stated:** the measured column fires asynchronously at 1–10 Hz
with no tuning knobs beyond the external drive.

**What exists.** A 12-population column (L2/3, L4, L5, L6 × pyramidal /
fast-spiking / non-fast-spiking) with two-compartment pyramidal cells.
Wiring statistics and the per-cell input budget come from H01 (166 M
synapses, BigQuery); membrane constants from Allen Cell Types (413 human
neurons); synaptic strengths, kinetics and short-term plasticity from Allen
Synaptic Physiology (14 of 24 subtype pairs measured, the rest class
averages). At full in-degree it fires asynchronously: pyramids 1.5 / 5.0 /
4.5 / 3.0 Hz (L2/3 → L6), fast-spiking cells 8–19 Hz, CV of ISI 0.7–1.1,
reproducible across seeds and at double size.

**Caveats, honestly counted.**
* "No tuning knobs" was not achieved. Outside the data there are two
  *fitted* calibrations (deep-layer external input ×1.6 on L5E; E→FS strength
  ×0.35 outside L2/3), two *assumed* in-vivo settings (external gain 2.0,
  background conductance 2× leak), and two *assumed* morphology constants
  (30 % of FS synapses on the soma; 70 % of membrane in the dendrite). All are
  labelled "calibrated / assumed" in the code and README, which is the right
  practice, but they are six knobs, not zero.
* Two populations are off physiological range and were never fixed: L4
  non-fast-spiking interneurons at **33 Hz** (too high) and L2/3
  non-fast-spiking at **0.05 Hz** (near silent). Neither was flagged in a
  stage summary.
* The targets the calibrations aimed at (L5E 4 Hz, L6E 2 Hz) are generic
  "awake cortex" numbers, and the first human-column calibrations used
  **rodent** reference rates (Potjans & Diesmann 2014). There are human
  single-unit rate ranges in the literature that should replace both.

### Stage 2 — full-scale column  ▸ not done

Every result is at 1.5–3 % of the real neuron count (976 of 32,500 neurons
per column; 3,904–7,808 across eight areas) with full per-neuron in-degree,
which keeps each neuron's input realistic. The full column (130 M synapses)
was never run. Brian2's runtime mode is the limit: the 14.5 M-synapse
eight-area network runs at 0.03× real time on this machine after a 2×
speed-up (Stage 12). A full column needs a GPU simulator (GeNN or NEST GPU)
and ≥64 GB of RAM, and that is a port, not a tweak.

### Stage 3 — cortical sheet  ▸ started, with substitutes

Done: a multi-column sheet with horizontal connections carrying 70 % of
outside input (Markov 2011, Stepanyants 2009), and eight association areas
(46d, 9/46d, 8m, 8B, 24c, 7A, 7m, STPc) joined by measured connection
fractions. Activity spreads between areas in proportion to connection
strength.

Not done, and worth saying plainly: the long-range wiring is **macaque**
(Markov 2014, distances scaled ×2.3), not human HCP tractography; every area
is an identical copy of the temporal-cortex column (no BigBrain layer
thicknesses, no Allen Human Brain Atlas cell-type differences); it is 8 of
~180 areas; and there is no resting-state validation (the stage's "done
when" was fMRI network patterns and EEG spectra). The 25 % inter-areal
share is a reasoned estimate from one paper.

### Stage 4 — subcortex  ▸ not started

Everything subcortical is a stand-in: action selection is two to three
hand-made groups of 40 LIF cells with mutual inhibition (not measured
anywhere); the thalamus is Poisson input; reward is a number from outside;
"surprise" and "uncertainty" are two-line formulas standing in for
noradrenaline and acetylcholine. These stand-ins did real work (learning
only became reliable once the selection circuit existed), but they are the
biggest gap between the model and the roadmap, and they are hand-designed
in a project whose premise was "architecture from data".

### Stage 5 — senses and a body  ▸ partial, by a different route than planned

The plan was retina → thalamus → V1, a cochlea, and a motor body. What
exists instead: RASPUTIN as a toy body (cues in, a choice out, reward from
the tool), and two self-organising spiking sensory layers (Diehl & Cook
2015) trained without labels — one on real assistant messages, one on
printed letters (76 % correct on fonts never seen; chance 4 %). Speech is
not started; a spoken-digit dataset is reachable. There is no visual
front end with local receptive fields, which is what the remaining letter
confusions (b↔h, i/l/j) call for.

### Stage 6 — learning  ▸ most progress, but narrow

Built: three-factor reward learning with eligibility traces, action-gated
updates, a reward baseline, inhibitory plasticity for stability, synapse
turnover, a surprise signal, an expected-uncertainty signal, and an
evolutionary search over the rule's settings. Results: 2-area cue task
92.5 % (trials 101–150, double size), 8-area 80–82 % (trials 101–200),
reversal relearned in 56–72 trials, routing of unseen real messages 63 %
(chance 33 %).

Limits: plasticity is on **one pathway** (long-range inputs onto 9/46d's L5
pyramids, ~160 k of 14.5 M synapses), not "across the sheet"; reward never
comes from a basal-ganglia model; the rule has **nine settings found by
search** (η, trace time constant, depression ratio, baseline window, weight
cap, iSTDP rate, selection strength, cue fibres, uncertainty gain); choices
are 2–3-way; and the decision is limited by spike-count noise in pools of
30–60 cells (the ideal learner on the same inputs scores 13 points higher).
Also learned the hard way: the rule cannot learn many varied, overlapping
input patterns from one reward bit per trial (Stage 19) — a sensory layer
that organises inputs first is required.

### Stage 7 — validation against human data  ▸ not done

The only validation is per-population firing rate against reference
ranges, plus irregularity and synchrony indices. No LFP/EEG-like spectra, no
fMRI-like correlations, no behavioural comparison with people. This is the
cheapest unstarted stage: a population-rate spectrum from the existing
column against published human intracranial spectra (1/f slope, alpha/beta
peaks) is a few hours' work and would be the first human-data check.

### Stage 8 — Warmind / RASPUTIN integration  ▸ done for a first version

A live service inside the FastAPI backend (`neural_core.py`): five models,
tool actions (status / stimulate / reward / reset / set_model / trial), a
dashboard panel, 8 passing tests, documentation. Trained only through the
tool, the core reached 80 % on the cue task. A memory leak that would have
grown without bound in a long-running core was found and fixed (Stage 17);
fast and lean modes halve run time and cut memory 55 %.

Not done: the shadow-mode router (the brain learning which submind handles a
message) is built and tested on fake data but never run on the real
`tool_calls` log, because that log is on the user's machine; the sensory
layers are not used by the core yet; the brain still makes no real RASPUTIN
decision.

## 2. What was done well

* **Every parameter has a provenance line** (measured / class average /
  calibrated / assumed), printed when a preset is built. This is the
  project's best habit and should be kept.
* **Exactness checks before adopting speed-ups.** Fast mode was accepted only
  after spike-for-spike identical output (`scripts/verify_fast.py`); lean mode
  likewise. Nothing was traded for speed.
* **Negative results were recorded**, not buried: text routing on the raw
  code (seven variants, all at chance), double size in eight areas, the
  collapsed letter layers, normalisation of plastic weights. These saved time
  later.
* **Controls when a result looked too good or too bad:** the ideal-learner
  stand-ins (Stages 19–20) located the routing failure in the input
  representation within minutes; the fixed-pattern diagnostic showed the
  3-way choice itself was learnable.
* **Held-out tests throughout:** unseen networks for learning rules, unseen
  phrasings and messages for routing, unseen fonts for letters.

## 3. What could have been done better

1. **No version control.** The project has never been in a git repository;
   the only distribution is the artifact package and tarballs. One bad write
   or a lost workspace loses the history. This is the first thing to fix:
   `git init`, commit, push to the existing `Human-Brain` repo.
2. **Scratch scripts held results.** The readout that produced the 76 %
   letter figure, the ideal-learner baselines and the preset regression check
   lived in `/tmp` until this audit (now in `scripts/analysis/`).
3. **Small samples behind some conclusions.** Decided on two networks:
   "double size in eight areas is not clearly better", "fast mode learns at
   least as well", "quick rule works in eight areas". Seed-to-seed spread on
   this task is ±10 points, so these are indications, not findings. Rule:
   four networks minimum for any claim, and report the spread.
4. **Expensive runs before cheap stand-ins.** About two days went into
   spiking routing runs before the five-minute ideal-learner test showed the
   input code was the problem. The stand-in should come first every time.
5. **Checkpoint hygiene learned late.** Machine restarts killed long runs
   for days before checkpointing and a watchdog existed; then the watchdog
   resumed a collapsed layer for hours because the checkpoint was written
   before the health check. Both fixed; both avoidable.
6. **Validation targets drifted from human to generic.** See Stage 1.
7. **Test coverage is lopsided.** The RASPUTIN service has 8 tests; the
   simulation package has none beyond one exactness script and an ad hoc
   regression check. A preset regression test and `verify_fast` should run
   on every change.
8. **The README grew to 23 stages (1,600 lines) with a stale roadmap table**
   at the top (fixed today). A one-page status should stay current; the
   stage log is history.
9. **Framing.** "Simulated human brain" describes the goal, not the object.
   The object is: human association cortex, 2 % density, measured local
   wiring and physiology, macaque long-range wiring, hand-made selection
   circuit, external reward. Saying this every time keeps claims honest.
10. **Two off-range populations (L4N, L2/3N) were left for 20 stages.**
    Problems noticed in a stage summary should be tracked in *Open problems*
    until closed, not just mentioned once.

## 4. What should come next

In priority order, with rough cost on this machine (2 cores, 8 GB).

**A. Housekeeping (a day)**
1. Git repository, pushed to GitHub. Add a preset-regression test and
   `verify_fast.py` as tests.
2. Track the two off-range interneuron populations as open problems; check
   whether iSTDP (already in the learner) closes them.

**B. Things RASPUTIN would feel (weeks)**
3. Shadow-mode routing on the real log (`train_router_from_logs.py`, needs
   the user's `warmind.db`). Hand over one decision type only when it beats
   the current router on held-out history.
4. Speech: the free spoken-digit set through a cochlea-like front end (mel
   spectrogram → spike rates) into the same sensory layer. Same recipe as
   letters; a realistic first target is ten spoken digits.
5. Wire the sensory layers into `neural_core` so the core can take text and
   letter images as cues.

**C. The curriculum (research, months)**
6. Letters → words needs sequence memory: a second layer with local
   receptive fields for the remaining letter confusions, then a prediction
   rule (dendritic prediction, Urbanczik & Senn 2014, fits the two-compartment
   cells already built) so the network learns what comes next. This is the
   step with the most unknowns and the most value.

**D. Roadmap stages (the biology)**
7. **Stage 4 first, not Stage 2.** A small measured basal-ganglia/thalamus
   model (rodent circuit data) replacing the hand-made selection circuit and
   the external reward would close the largest gap between model and
   premise, at small scale, on this hardware.
8. **Stage 7, cheaply:** compare the column's population spectrum with
   human intracranial spectra. First human-data validation; hours of work.
9. **Stage 2 needs different hardware and a simulator port** (GeNN/NEST
   GPU; a machine with a GPU and 64 GB). Do not attempt it in Brian2 runtime.
10. **Stage 3 proper:** replace the macaque wiring with HCP tractography and
    give areas their BigBrain thicknesses once the sheet runs on better
    hardware.

**E. Standards for every future result**
* Four networks minimum; report the spread.
* A numerical stand-in before any spiking run longer than an hour.
* Checkpoint every long run; never checkpoint a failed state.
* Keep the provenance line; add to it, never around it.

## 5. What was fixed the same day (4 November, afternoon)

| audit item | action | where |
|---|---|---|
| 1 no version control | repository created; code pushed to branch `neural-core` of `andrewreynolds421-ui/Human-Brain`; every later fix its own commit | `jobs/sync_repo.sh` |
| 2 scratch scripts | seven analysis scripts moved into the repository | `scripts/analysis/` |
| 3 two-network conclusions | re-tested on four networks (below) | `results/areas8_four_network_retest.json` |
| 4 stand-ins first | made a rule | `METHODS.md` §3 |
| 5 checkpoint hygiene | learning runs write partial results every 20 trials; sensory training never checkpoints a collapsed layer | `scripts/reward_learning.py`, `scripts/train_letters.py` |
| 6 validation targets | sourced human (L2/3) and rodent (deep) rate ranges; a check script; 5 of 12 populations found out of range → open problem 0 | `results/rate_targets.json`, `scripts/check_rates.py` |
| 7 tests | 11 tests for the simulation package (8 fast, 3 slow); all pass | `tests/test_synth.py` |
| 8 stale README | roadmap table rewritten as a status table; stage log kept as history | `README.md` |
| 9 framing | "what the object is" paragraphs in the README and `NEURAL_CORE.md` | — |
| 10 untracked problems | the non-FS interneuron rates are open problem 0 with a sensitivity test | `README.md` *Open problems* |

**Re-test of the two-network conclusions** (eight areas, 150 trials, networks
9001–9004; accuracy in trials 101–150 per network, mean ± sd; first trial at
which a 30-trial average reaches 75 %):

| setting | per network | mean ± sd | reaches 75 % | conclusion |
|---|---|---|---|---|
| quick rule, current size | 82 / 82 / 80 / 88 | 83 ± 3 | 62 / 82 / 54 / 83 | **holds:** better than the old rule on every network |
| quick rule, double size (selection scaled) | 88 / 76 / 88 / 88 | 85 ± 5 | 54 / 76 / 81 / 73 | **holds:** +2 points, better on 3 of 4, worse on 1 — not clearly better; `learner8` stays at current size |
| old rule, fast mode | 68 / 72 / 84 / 72 | 74 ± 6 | 83 / 136 / 98 / 147 | **holds:** fast mode learns at least as well as the original code |
| old rule, original code | 64 / 74 / 58 / 76 | 68 ± 7 | — / 94 / 79 / 50 | |

So all three decisions stand, now on four networks each. The seed-to-seed
spread (sd 3–7 points) is the number to keep in mind when reading any
single-run figure in the stage log.
