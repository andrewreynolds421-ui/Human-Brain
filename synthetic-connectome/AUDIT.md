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

### Stage 4 — subcortex  ▸ started (5 November, Stage 26 — see §7)

At the time of the audit everything subcortical was a stand-in: action
selection was two to three hand-made groups of 40 LIF cells with mutual
inhibition (not measured anywhere); the thalamus Poisson input; reward a
number from outside; "surprise" and "uncertainty" two-line formulas standing
in for noradrenaline and acetylcholine. These stand-ins did real work
(learning only became reliable once the selection circuit existed, on the v1
column), but they were the biggest gap between the model and the roadmap,
hand-designed in a project whose premise was "architecture from data".
Since then the selection circuit has been replaced by a basal-ganglia–
thalamus loop built from published rat circuit data (§7); reward and the
neuromodulator formulas remain stand-ins.

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

## 6. Follow-up (4 November, evening): the interneuron problem fixed

Stage 25 recalibrated the column (`human_h01_dendritic_v2`): one sample-size
rule for measured subtype synapses, non-FS external input at the literature
ratio (×0.1), and the two fitted calibrations refitted. 9 of 12 populations
are now within their reference ranges (was 7; the misses went from 65–550 %
past an edge to 5–25 %). Learning on the recalibrated column, four networks
each: two areas 97 ± 2 % (was 82 ± 7), eight areas 96 ± 5 % (was 83 ± 3),
75 % reached in 30–63 trials instead of 43–97, and eight-area trials run in
7 s instead of 15 s. The v1 preset and all earlier results are unchanged;
RASPUTIN uses v2 by default. Knob count after the fix: 2 fitted, 5 assumed
(one more than before: the non-FS input ratio, sourced).

## 7. Follow-up (5 November): Stage 4 begun — the selection circuit from data

Recommendation 7 of §4 ("Stage 4 first, not Stage 2") is carried out in part.
`synth/basal_ganglia.py` is a spiking basal-ganglia–thalamus loop built from
Humphries, Stewart & Gurney (2006): populations, pathways, probabilities,
delays, synaptic kinetics, membranes, tonic currents, noise, dopamine level,
selection criterion and protocol from the paper; a thalamic relay and its two
pathways added and labelled assumed; four numbers fitted on a stand-in cortex
first (two cortical gains, two tonic currents) and one in the learner (the
thalamo-cortical gain), all recorded with their scans in
`results/bg_parameters.json` and `results/bg/`. On its own the loop sits in the
recorded resting ranges (STN 10, GP 19, SNr 25 spikes/s) and selects and
switches as the paper's does. Under the learner it replaces the hand-made
circuit with no loss: two areas 97.5 ± 0.9 % on four networks (WTA 97 ± 2),
eight areas 96 ± 1 % (WTA 96 ± 5) but reaching 75 % only at trial 64–85 against 20–40, because the loop perseverates on one action for its first 30–60 trials (below); the rewarded action can be read off the SNr/thalamus directly
(92 % in the last 50 trials). Two things the comparison showed that were not
known before: the loop works in a graded regime in the learner (SNr 20 vs 27
spikes/s, never below the paper's 5 Hz criterion, because the pools fire
4–8 spikes/s), and on the recalibrated column the two-area task no longer
needs any selection circuit (98 ± 1 % with none, reaching 75 % later: trial 33–52 vs 21–32 / 20–45) — so that task cannot separate the
circuits, while eight areas can (no circuit: 93 ± 3 % on two networks, 75 % at trial 48–78). The loop's slow start is perseveration — its first 30–60 choices are one action, since nothing in the loop learns from the reward and only the WTA groups' noise breaks ties — which points the next step at plasticity inside the loop (cortico-striatal learning and reward prediction), not at more fitting. Still stand-ins: reward (external), surprise and
uncertainty (formulas). Not done: dopamine effects in STN/GP, striatal
up/down states, rebound bursts, shunting, FSIs, TRN; hippocampus, cerebellum.
Cost: +1.9 s per trial, from object count, removable by merging populations.

## 8. Follow-up (5 November, afternoon): the reward signal moves inside the circuit

Stage 27 adds to the loop what §7 said it lacked. (i) Dopamine-gated
cortico-striatal plasticity (Reynolds 2001; Shen 2008; Yagishita 2014) on
cue-cell inputs to D1/D2 of every channel. (ii) A dopamine population with a
reward input, a habenula-type negative input and a striosomal critic that
learns to predict the outcome; its burst or pause after the outcome is the
prediction error that every plastic synapse — striatal and cortical — now
reads, in place of the number applied from outside. Result on four networks:
95 ± 2 % (Stage 26: 97.5 ± 1), 75 % at trial 34–55, and the dopamine cells
show the Schultz signature (burst to reward 21.5 → 13 spikes/s as the critic
rises 3.3 → 7.8 spikes/s; errors still a full pause). Negative result, stated
in the README and `results/bg/loop_only_diagnostics.json`: the loop's own
synapses learn in the right direction but cannot carry the task alone
(40–55 % with the cortical rule off), because the shared cue input moves all
channels together and this learner has no cortical competition between its
pools; two structural attempts (supralinear eligibility, thalamostriatal
input) did not fix it and one locked the loop. A process lesson recorded as
well: the first critic went silent and the 96 % learning result hid it —
the dopamine log, not the accuracy, showed the fault. Stage 4 status: basal
ganglia with plasticity and an internal prediction error, thalamic relay a
stand-in, reward still delivered as an outcome by the task; hippocampus,
cerebellum and neuromodulator circuits not started. Knob count in the loop:
MODEL constants from the paper; FITTED: 2 cortical gains, 2 tonic currents,
thalamic gain, cue gain, 2 learning rates, critic gain, 3 input rates;
ASSUMED: thalamic relay, critic offsets, 500 ms critic inhibition, dip gain,
weight bounds. Open problem 11 names the next structural step.

## 9. Follow-up (5–6 November): the thalamus as the pools' drive (Stage 28)

The handoff's first item, done. 15 % of each action pool's outside input now
arrives through the thalamic relay, held at the measured resting rate of
motor thalamus (13 spikes/s, Vitek et al. 1994) with weights computed so the
resting input is unchanged; only the share is assumed. On four networks the
loop learns the task with all cortical plasticity frozen (83 ± 7.5 %, was
56 %), and the full system loses its perseveration (75 % by trial 20–42,
was 34–55) at a cost of 2.5 points in final accuracy (92.5 ± 1.7 vs 95 ± 2.2)
because loop and cortex now both learn and partly compete. Open problems 10
and 11 are addressed; the loop/cortex balance, the TD circuit and the
eight-area test are what remains for Stage 4's basal ganglia and thalamus.

## 10. Follow-up (6 November, overnight): Stage 29

(a) Balancing loop and cortex: the scan's best setting (cortical rate ×0.5,
98 % on networks 31–32) did not hold on held-out networks 35–38 (89 ± 2 vs
91 ± 1 unchanged; WTA 96.5 ± 1.7) — METHODS rule 2 caught a seed effect.
(b) The routed loop reverses rules (all four networks relearn; ends 79 ± 5
vs WTA 80 ± 7, slower to recover) and runs in eight areas (89.5 ± 4.3, 75 %
by 33–59, perseveration halved vs Stage 26, ~6 points below WTA). (c) A
fast value pathway beside the slow critic: at moderate strength it fills the
cue-time dopamine dip but makes no burst; at four-fold strength the burst
moves to the cue (21 → 43 spikes/s while the reward burst fades 24 → 15) but
overshoots and costs learning (78 ± 8 %). New open problem 12 groups these
as missing homeostasis in the loop's populations. Two machine restarts
during the night; the job runner and partial files recovered every run.

## 11. Follow-up (6 November, day): Stage 30, homeostasis

Intrinsic homeostatic plasticity on the thalamic relay (target: its measured
13 spikes/s) removes the relays' drift and brings the routed loop to 95.0 ±
2.2 % on held-out networks 35–38 (was 91 ± 1; WTA 96.5 ± 1.7) — the gap
Stage 29 measured is now inside the seed spread. With homeostasis also on
the critic and value cells and the value pathway at half the overshooting
strength, dopamine at the cue doubles (13.6 → 25.3 spikes/s) while the
reward burst falls (23 → 13.5) and learning stays at 95.5 ± 3.3 %: the TD
signature, with the stated limit that the cue response starts above rest.
Negative: in eight areas relay homeostasis costs ~6 points (83.5 vs 89.5),
probably because one target rate under-drives the selected relay when the
pools get more input. Stage 4's basal ganglia and thalamus now work end to
end in two areas; eight areas is the open part.

## 12. Follow-up (6 November, afternoon): Stage 31

Two changes to how the relay's homeostat responds (slower average,
rest-gated) did not remove its eight-area cost; a clean control showed the
cost was real (88 % without the homeostat at the full cortical rate). What
removed it was routing 25 % rather than 15 % of the pools' drive through
the relay, which the homeostat makes safe. On held-out networks: two areas
93.5 ± 2.2 %, eight areas 88.5 ± 1.7 % (four networks; no-homeostat
87.5, WTA 94.5). Restarts: three during the day; every run recovered.

## 13. Follow-up (6 November, evening): Stage 32, RASPUTIN integration

The handoff's "putting it together" item, in shadow form. Offline, on real
assistant queries through the sensory layer, the basal-ganglia loop routes
unseen messages at 58.1 ± 8.4 % against 51.9 ± 5.7 % for the hand-made
circuit, better on all four networks. Live, `neural_router.py` runs the
same brain beside RASPUTIN, learns from each routing decision RASPUTIN
makes, logs its own pick and never acts on it. First measurement on real
traffic needs the user's running instance (the local log is empty). Also
built: checkpoint/resume for routing runs (the machine restarted ~10 times
on 6 Nov).

## 14. Follow-up (7 November): Stage 33, first human-recording comparison

The column's field-potential spectrum (Mazzoni LFP proxy) against awake
human intracranial EEG (Lendner 2020; Frauscher 2018 atlas): a beta rhythm
near the frontal band is present, but the 1/f background is absent (30–45 Hz
slope −1.1 vs −2.8/−3.0 in humans), slow power is missing, and doubling the
GABA_A decay does not steepen the slope. Diagnosis: the stand-in for the
rest of the brain — independent stationary Poisson input, 1.5–2× the
recurrent input — dominates the field. This is the first test against human
data that the model fails, and it fails for a reason that points at a
specific next step (open problem 13).

## 15. Follow-up (7 November): Stage 34 — Stage 33 corrected

Stage 33's failure was mostly the measurement: the LFP proxy used the
membrane potential, whose spike resets inject broadband power. With the
driving force fixed at rest, the resting column's 30–45 Hz slope is −2.20 ±
0.45 on held-out networks (human awake −2.75/−2.99). Stage 33's diagnosis
(independent Poisson input flattens the field) was wrong — a constant-input
control showed it. Remaining real gap: slow power ~25 dB short of the human
spectrum; shared slow input adds some but flattens the high-frequency slope,
and a slow-only variant failed criteria fixed in advance. A lesson for the
standards: a new measurement needs its own control (here, a signal whose
spectrum is known) before its result is interpreted.

## 16. Follow-up (8 November): Stage 35 — speed

The loop's six cell populations are now one simulator group (`merged=True`,
the default), with the populations as slices. The profiler showed the loop
was slow because of its many small objects, not its arithmetic. The merged
group is laid out in the separate groups' schedule order and uses their
generated arithmetic form, so it reproduces them exactly. Proof: every field
of the learner record was identical in 8/8 runs (four configurations, two
networks, 20 trials, including eight areas, the gated homeostat and the TD
value pathway), and a test guards it. The loop's time fell 28 %, about 1 s
per trial. That is half the overhead rather than most of it: the synapse
pathways are the larger remaining share, and merging them changes the
float-addition order, so it would need a seed-spread check.

## 17. Follow-up (9 November): Stage 36 — letters to words

Words are read letter by letter from real letter-layer responses in unseen
fonts. A prediction rule on the letter units, taught by the percept (the
winning letter unit; no labels), raises letter accuracy inside words. On
known words it goes from 77.6 to 85.1 %, +3.6 ± 1.7 points beyond a control
trained on shuffled-order words. On new words it goes from 78.7 to 81.3 %,
+1.9 ± 0.7 beyond the control. Both are paired on identical test words over
4 networks.

Teaching the rule with the raw evidence instead was found to smear
predictions over look-alike letters. That is now part of the record as a
reference row.

A spiking self-organising sequence layer was the planned core, and it does
not help (negative). Its winner-take-all templates lose the current letter
on new words. With weaker inhibition they keep it but add nothing a linear
predictor can use beyond the context, which a fitted decoder confirmed.

Process notes:
- The first shuffled control drew different test words (shared random
  stream). It was fixed before any spiking control ran, and the controls
  were then done for the context pathway only.
- The spiking controls were dropped once the layer was negative.
- The ideal-reader smoothing initially swamped real counts (frequencies are
  ~1e-5). It was caught because the 3-letter ideal scored below the 2-letter
  one, and fixed.

## 18. Follow-up (9 November): Stage 37 — measuring the headroom before building

The planned spiking letter-pair layer was not built. Exact offline versions
showed about 1 point of letter-reading headroom from order beyond the
context: even fully fitted, pairs added about 1 point, and learned online
they cost 1.4 / 0.1 points against plain unscaled context.

The Stage 36 target, "beat the context on new words", was mis-set. The ideal
reader itself gains only 0.9 points there.

What did help, measured on 4 networks and paired:
- 10× more reading and unscaled inputs to the prediction rule: +2.4 ± 1.5 /
  +1.5 ± 1.2 points.
- The online rule equals a fully fitted one taught by the same kind of
  teacher. The remaining 1–2 points are teacher errors.

The supporting checks are scratch scripts kept with their outputs; the main
table is reproduced exactly by `scripts/analysis/stage37.py`.

