# Handoff: everything a new session needs to put this together

Written 5 Nov 2026 for whoever (person or model) picks the project up next.
It says what exists, where it is, how to run it, what the numbers are, what
is honest about them, and what the next moves are — in that order. Read this,
then the status table at the top of `README.md`, then `METHODS.md`. Do not
read the 1,900-line README stage log end to end; it is history, and this file
points to the parts that still matter.

## 0. What the object is (five lines)

A spiking model of human *association cortex*: twelve cell populations per
column with local wiring measured from the H01 electron-microscopy volume,
membranes and synapses from the Allen human data, at 1.5–3 % of real neuron
count with full per-neuron in-degree; up to eight areas joined by *macaque*
long-range wiring; action selection by a basal-ganglia–thalamus loop built
from a published rat circuit model; outcomes turned into a learning signal by
a modelled dopamine population with a critic. It learns cue–action tasks
(95–97 %), routes real text at 58–63 % through a self-organising sensory
layer, reads printed letters at 76 % in unseen fonts, and runs live inside the
RASPUTIN assistant as a tool and as a shadow router that learns RASPUTIN's
routing decisions without changing them. Against human recordings: firing
rates in range for 9 of 12 populations; field-spectrum slope near awake human
iEEG, slow power far too low. It is not a brain and not full-scale.

## 1. The pieces and how they fit

```
 DATA                       SYNTHESIS                    SIMULATION / LEARNING                   USE
 H01 (BigQuery) ─┐                                       ┌─ LiveColumn (synth/live.py): step, stimulate, state
 Allen synphys ──┼─► presets (synth/__init__.py) ─► column ┼─ LearnerColumn (synth/learner.py): trial, reward
 Allen cell types┘   human_h01_dendritic_v2 (current)    │     ├─ selection: WTA groups | BasalGanglia loop
 macaque FLN ──────► build_areas / add_interareal        │     ├─ DopamineSystem (optional): outcome → δ
 (synth/areas.py)    (8 association areas)               │     └─ three-factor rule on 46d→9/46d L5 synapses
                                                         │
 text (CLINC150) ──► SensoryLayer (synth/sensory.py) ───►┘  cues: "A"/"B" | text | array | sensory spikes
 letters (fonts)  ─► SensoryLayer (letters)                                     │
                                                                                ▼
                                             RASPUTIN: warmind/neural_core.py wraps LearnerColumn as a tool
                                             (trial / reward / stimulate / status / set_model / reset)
```

* **Column** (`synth/subtypes_fit.py`, `synth/dendrite.py`, `synth/build.py`,
  `synth/network.py`): 12 populations (L23/L4/L5/L6 × E / fast-spiking /
  non-FS), two-compartment pyramids, conductance synapses, Tsodyks–Markram
  short-term plasticity, external drive from the H01 input budget. Preset
  `human_h01_dendritic_v2` (Stage 25) is current; `human_h01_dendritic` (v1)
  is kept and every earlier result reproduces on it. `python -m synth.h01_fit`
  prints every H01-derived number; each preset carries a `provenance` dict
  with every non-data number labelled CALIBRATED or ASSUMED.
* **Areas** (`synth/areas.py`, `results/interareal_macaque_assoc.json`): the
  same column in each of eight macaque association areas, joined by measured
  FLN weights; the learner uses 46d (cue) → 9/46d (action).
* **Learner** (`synth/learner.py`, `scripts/reward_learning.py` — the script is
  the reference implementation, the class is the live one; they share
  settings): cue = half of 46d L2/3 pyramids driven for 200 ms; action = which
  of two (or K) pools of 9/46d L5 pyramids fires more in the decision window;
  three-factor rule (eligibility × reward prediction error) on the plastic
  46d → 9/46d L5 synapses, gated to the chosen pool; synapse turnover
  (`synth/structural.py`); inhibitory plasticity; surprise and uncertainty
  terms on the learning rate (`QUICK_SETTINGS`). "Fast" and "lean" modes are
  exact (verified) speed/memory variants.
* **Selection**: `wta` = two hand-made groups of 40 LIF cells with mutual
  inhibition (a stand-in); `bg` = `synth/basal_ganglia.py`, the Humphries,
  Stewart & Gurney (2006) loop (D1/D2/STN/GP/SNr per channel + an assumed
  thalamic relay), parameters with labels and sources in
  `results/bg_parameters.json`.
* **Dopamine** (`DopamineSystem` in the same file): SNc/VTA cells, reward and
  habenula-type inputs, striosomal critic; with it on, the number given to
  `reward()` is an outcome event and the dopamine burst/pause is the learning
  signal for every plastic synapse.
* **Sensory layers** (`synth/sensory.py`, Diehl & Cook 2015 unsupervised
  STDP): text (`results/sensory/s400_final.npz`, 400 cells) and letters
  (`results/letters/l1600v_final.npz`, 1,600 cells); `synth/textcode.py` turns
  words into cell codes; `scripts/analysis/readout.py` learns a read-out.
* **RASPUTIN**: `warmind/neural_core.py` (service thread + tool), documented
  in `warmind/NEURAL_CORE.md`; `main.py.neural.patch` is the one hook into the
  assistant; 10 tests in `warmind/tests/test_neural_core.py`.

## 2. Where things live

| what | where |
|---|---|
| code, results, docs | GitHub `andrewreynolds421-ui/Human-Brain`, branch **`neural-core`** (`synthetic-connectome/` and `warmind-neural-core/`); one commit per closed item |
| working copies on this machine (may not survive) | `/home/claude/synthetic-connectome`, `/home/claude/warmind` (the whole RASPUTIN app incl. `warmind.db`, `DEPLOY_ALL.md`, `STEPS.md`), clone at `/home/claude/human-brain`, artifact mirror `/home/claude/warmind-neural-core` |
| shareable package | artifact "Warmind Neural Core Package", https://claude.ai/artifact/TodmXDrbBWakQBvyM7am2h (index page, every file; code served with a `.txt` suffix) |
| raw data | H01: BigQuery dataset `h01`, tables `synapses_c3` (166 M rows, loaded from `gs://h01-release/data/20210729/c3/synapses/exported/`) and `neurons` (3,111 labelled cells); the SQL is quoted in README "Phase 1" and "Stage 2b". Allen: `/home/claude/downloads/synphys/synphys_r2.1_small.sqlite` (also on the project's GitHub release), `data/cell_types_specimen_details.csv`. Text: `data/routing_clinc3.json` (CLINC150, 3 destinations). Letters: `data/letters.npz` (111 fonts, 89 train / 22 test). |
| summaries that make the raw data unnecessary for most work | `results/celltypes_human_summary.json`, `results/synphys_human_summary.json`, `results/h01_layer_stats.json` (`/home/claude/h01_layer_stats.json`), `results/interareal_macaque_assoc.json`, `results/rate_targets.json` |
| reference material | `/home/claude/ref/` (aisynphys, H01 analysis tables, Potjans 2014, multi-area-model, NEST) |
| job runner | `/home/claude/jobs/runner.py` (reads `queue.json`, 2 slots, a job is done when its `out` exists), `watch.sh`, `sync_repo.sh "message"` (copies the filtered tree into the clone, commits with attribution, pushes) |

Environment: Python 3.13, `pip install -r requirements.txt` (numpy, scipy,
matplotlib, brian2 ≥ 2.7, pytest); Brian2 needs a C compiler for its cython
target (first run of any new model compiles for 1–3 minutes; the cache is
`~/.cython/brian_extensions`, ~1.2 GB). Machine used: 2 CPUs, 7 GB RAM, no
GPU. Outbound network is an allowlist (package registries, GitHub, BigQuery
through the connector); the machine restarts without warning every day or
two — hence the runner, partial results and checkpoints.

## 3. Cookbook

```bash
cd synthetic-connectome
python -m pytest tests -q                       # 13 fast tests, seconds
python -m pytest tests -q -m slow               # 6 network tests, ~2 min
python scripts/check_rates.py --preset human_h01_dendritic_v2 --seeds 3   # rates vs reference ranges
python -m synth.h01_fit                          # every H01-derived number

# cue task, two areas, current best rule (fast + structural + surprise + uncertainty); ~7 min
python scripts/reward_learning.py --trials 150 --seed 31 --fast --preset human_h01_dendritic_v2 \
  --genome "$(cat results/genomes/quick.json)" --out results/x.json
#   add --areas all            eight areas (7 s/trial);  --n-scale 0.03  double size
#   add --selection bg          the basal-ganglia loop instead of the WTA groups
#   add --bg '{"plastic":true,"eta_bg":0.5,"ctx_gain_cue":1.5,"post_lin":0,"post_gain":1}' \
#       --da '{}'               dopamine population + critic; --cortical-da 0 keeps R - rbar for cortex
#   --reverse-at 100            flip the rule mid-run (reversal learning)
python scripts/analysis/bg_compare.py           # loop vs WTA vs none table from results/bg/

# the loop on its own (paper protocol, stand-in cortex)
python scripts/bg_selection_test.py --ctx-gain 2 --stn-gain 0.3 --seeds 1,2,3,4 --grid 4 --out results/bg/x.json

# text routing and letters
python scripts/train_sensory.py ...             # unsupervised text layer (see README Stage 20)
python scripts/routing_experiment.py --data clinc --sensory results/sensory/s400_final.npz --sens-lr 0.33 --balanced
python scripts/make_letters.py; python scripts/train_letters.py --n-e 1600 --epochs 3 --ink 100 --dt 0.25 --out results/letters/x
python scripts/analysis/readout.py results/letters/l1600v_final.npz

# field spectrum vs human iEEG (Stages 33-34; corrected proxy is the default)
python scripts/spectrum_validation.py --seeds 33,34,35,36 --t 10 --out results/spectra/x.json
#   --ext-mod '{"sigma": 0.2}'   shared slow modulation of the external input (off by default)

# routing benchmark with the loop (Stage 32; resumable)
python scripts/routing_experiment.py --data clinc --sensory results/sensory/s400_final.npz --sens-lr 0.33 --balanced \
  --preset human_h01_dendritic_v2 --selection bg --bg "<r25_hom bg from results/hom/configs.json>" --da '{"crit_gain": 2.5, "crit_eta": 0.05}' \
  --train 300 --n-test 90 --test-every 150 --ckpt results/router/ck_x --out results/router/x.json

# RASPUTIN (from warmind/)
NEURAL_CORE=1 NEURAL_CORE_MODE=learner uvicorn main:app          # the tool appears when the flag is on
NEURAL_CORE_SELECTION=bg NEURAL_CORE_DOPAMINE=1 ...              # loop + dopamine
python -m pytest tests/test_neural_core.py -q                    # 10 tests, ~3 min
python scripts/train_router_from_logs.py                         # sensory router from real warmind.db logs (offline)
NEURAL_ROUTER=1 uvicorn main:app                                 # live shadow router; GET /api/neural/shadow
python -m pytest tests/test_neural_router.py -q                  # 7 tests, seconds
```

Long runs: put them in `/home/claude/jobs/queue.json` and start the runner
(`/home/claude/jobs/ensure.sh` starts it if it is not running — nothing on
this machine starts it at boot, and the machine restarted ~10 times a day on
6–7 Nov, so call it after every check); every learning run writes
`<out>.partial.json` every 20 trials; sensory training checkpoints and never
checkpoints a collapsed layer (`_COLLAPSED` marker). Commit with
`bash /home/claude/jobs/sync_repo.sh "message"`.

## 4. Interfaces you will call

```python
from synth.learner import build_learner_network, LearnerColumn, QUICK_SETTINGS, WTA_REF_POOL
col = build_learner_network(n_scale=0.015, seed=1, preset="human_h01_dendritic_v2", areas=None | "all")
L = LearnerColumn(col, genome=dict(eta=QUICK_SETTINGS["eta"], wta_ref_pool=WTA_REF_POOL), seed=1,
                  structural=True, surprise_gain=3.0, uncert_gain=2.0, fast=True, lean=True,
                  selection="wta" | "bg", bg_kw={...}, da_kw=None | {...},
                  n_actions=2, pattern_cues=False, sensory=None | SensoryLayer | path)
r = L.trial("A" | "B" | "some text" | array)   # -> dict(cue, choice, pool_spikes[, snr_hz])
L.reward(+1.0 | -1.0)                           # -> dict(applied, prediction_error, ...[, dopamine])
L.step(ms); L.state(); L.stimulate(layer, strength, duration_ms); L.reset()   # lean: rebuild instead

from synth.basal_ganglia import BasalGanglia, DopamineSystem, DEFAULTS, DA_DEFAULTS
bg = BasalGanglia(sources=[(G, pool_idx), ...], seed, target=(G, "g_ed", w_nS), context=(G, cue_idx), **DEFAULTS-keys)
bg.rates(pop, t0, t1); bg.selected(t0, t1); bg.reward(delta); bg.weight_summary(); bg.clear(); bg.objects
da = DopamineSystem(bg, (G, cue_idx), seed, **DA_DEFAULTS-keys)
da.deliver(value); da.end_event(); da.delta(t0, t1); da.learn(delta); da.clear(); da.objects

from synth.sensory import SensoryLayer           # .train, .features(text|array), .spike_train, .save/.load
from synth.textcode import encode                # words -> cell code ("union" | "kwta")
from synth import PRESETS, synthesize            # PRESETS[name](**kw) -> params with .provenance
```

Settings that matter: the learner "genome" (`scripts/reward_learning.py`
`DEFAULT_GENOME`, `results/genomes/quick.json` = the current best:
eta 13.1, tau_elig 263 ms, a_minus 0.73, w_max ×30, istdp 0.0163, gate 1,
wta 8 / 3 mV, synapse turnover on, surprise 3, uncertainty 2); the loop
(`DEFAULTS`: ctx_gain 2.0, ctx_gain_stn 0.3, thal_gain 0.5 in use, tonic
currents 1.22/0.38/0.42 nA); the dopamine system (`DA_DEFAULTS`). RASPUTIN
environment variables are listed at the top of `warmind/neural_core.py`.

## 5. The numbers, and the files that back them

| claim | value | networks | file |
|---|---|---|---|
| column rates in reference ranges (v2) | 9 of 12 populations; misses 5–25 % past an edge | 3 seeds | `results/rate_check_v2.json`, `results/rate_targets.json` |
| cue task, 2 areas, WTA, v2 | 97.0 ± 2.2 %, 75 % by trial 21–32 | 4 | `results/v2/quick_2area_seed3[1-4].json` |
| cue task, 8 areas, WTA, v2 | 95.5 ± 4.6 %, 75 % by 20–40, 6.5 s/trial | 4 | `results/v2/quick_8area_seed900[1-4].json` |
| basal-ganglia loop instead of WTA | 2 areas 97.5 ± 0.9 %; 8 areas 96 ± 1 % but 75 % only by 64–85 (perseveration) | 4 + 4 | `results/bg/learning_comparison.json` |
| no selection circuit at all | 2 areas 98 ± 1 %; 8 areas 93 ± 3 % (2 nets) | 4 / 2 | same |
| loop alone, paper protocol | rest STN 10 / GP 19 / SNr 25 Hz in range; selects from 10–12 spikes/s; switches 80 %, keeps 85 % up to 16 spikes/s; both selected above | 4 | `results/bg/standalone_final.json` |
| loop vs WTA on held-out networks 35–38 (Stage 29a) | 91 ± 1 % vs 96.5 ± 1.7 % | 4 | `results/thal/stage29_summary.json` |
| routed loop, 8 areas / reversal (Stage 29b) | 89.5 ± 4.3 %; reversal end 79 ± 5 % (WTA 80 ± 7) | 4 + 4 | same |
| field spectrum vs awake human iEEG (Stages 33–34, corrected proxy) | 30–45 Hz slope −2.20 ± 0.45 (human −2.75…−2.99); slow power ~25 dB short | 4 held-out | `results/spectra/stage34_summary.json` |
| routing unseen CLINC messages, 3 destinations (Stage 32) | loop 58.1 ± 8.4 %, WTA 51.9 ± 5.7 % (loop better on 4/4) | 4 | `results/router/stage32_summary.json` |
| relay homeostasis, held-out networks 35–38 (Stage 30) | 95.0 ± 2.2 % (WTA 96.5 ± 1.7); 8 areas 83.5 ± 2.2 % (Stage 31 with 25 % routed: 88.5 ± 1.7 %, WTA 94.5) | 4 + 4 | `results/hom/stage30_summary.json` |
| dopamine burst moves to the cue (Stage 30, `td_hom2`) | cue 13.6 → 25.3, reward 23 → 13.5 spikes/s; learning 95.5 ± 3.3 % | 4 | same |
| thalamic relay carrying 15 % of the pools' drive (Stage 28) | loop alone 83 ± 7.5 %; full system 92.5 ± 1.7 %, 75 % by 20–42 | 4 | `results/thal/summary.json` |
| dopamine population as the learning signal | 95 ± 2 %, 75 % by 34–55; burst to reward 21.5 → 13 Hz as the critic rises 3.3 → 7.8 Hz | 4 | `results/bg/stage27_summary.json` |
| loop learning the task alone (cortex off) | **does not**: 40–55 % | 2 + probes | `results/bg/loop_only_diagnostics.json` |
| text routing, CLINC150, 3 ways | 58–63 % on unseen messages (chance 33 %) | — | `results/routing/clinc_s400_lr033*.json`, `results/sensory/s400_score.json` |
| printed letters, unseen fonts | 76 % (chance 4 %) | — | `results/letters/l1600v_score.json` |
| speed / memory | fast mode 2× exact; lean: 8-area learner 1.05 GB running, 1.5 GB peak | — | `results/speed/` (not in git), README Stages 12, 14, 17 |

"Reaches 75 %" = first trial at which the previous 20 average ≥ 75 %.
Seed-to-seed spread on the cue task is about ±10 points; two seeds are a
pilot. Costs per two-area trial at n_scale 0.015: WTA 2.9 s, loop 4.8 s,
loop + dopamine 8 s; eight areas ≈ ×2.3; double size ≈ ×2.

## 6. The standards (short form of `METHODS.md`)

Four networks per claim, spread reported; held-out networks, messages and
fonts; a numerical stand-in before any spiking run over an hour; exactness
before speed; every non-data number labelled with source or target; human
reference ranges where they exist; defaults never changed silently;
checkpoints for long runs, never of a collapsed state; results in
`results/`, nothing in `/tmp`; commit per closed item; tests before commits;
say what the object is; problems into *Open problems* the same day; negative
results written up; one status page kept current. The labels: MEASURED /
MODEL (from a cited paper), CALIBRATED / FITTED (to a stated target, scan on
file), ASSUMED (stated choice). The write-up of each stage ends with what the
result does and does not settle.

## 7. What to do next, in order, with the first concrete step of each

The audit (`AUDIT.md` §4, §7, §8) and the README's *Open problems* 0–11 are
the long form. Priorities as they stand on 5 Nov:

1. *(Done 6 Nov, README Stage 28: loop alone 83 %, full system 92.5 % reaching 75 % by trial 20–42. Remaining: the loop/cortex balance — both relays rise late in a run — and the eight-area and reversal tests with routing.)* **Thalamus as the pools' drive, and cortical competition between pools**
   (open problem 11; also the Stage 4 thalamus item). Today the thalamic relay
   adds to a pool's external Poisson drive; nothing is removed from an
   unselected pool, so the loop's striatal synapses cannot tell channels
   apart and the loop perseverates. Step: route a stated share of the pool
   cells' external input through the relay (relay tonic rate from awake
   rodent VA/VL recordings — find and cite a range), recalibrate the pool
   rate (`scripts/check_rates.py` still has to pass), then re-run
   `plastic_dl` (loop only) and `dasys` on four networks. Expect the
   "reaches 75 %" of the loop to fall toward the WTA's 20–40 and the loop-only
   run to leave chance. Cheaper partial step: a striatal fast-spiking
   interneuron population (Koos & Tepper 1999) with feed-forward inhibition.
   2–3 days.
2. *(Done 6 Nov, README Stage 30: with homeostasis on critic and value cells the burst moves to the cue — 13.6 → 25.3 spikes/s — while the reward burst falls and learning stays at 95.5 %.)* **Temporal-difference dopamine** so the burst transfers to the cue: replace
   the 500 ms critic decay with a proper TD circuit (eligibility in the
   critic, prediction at cue onset exciting the dopamine cells). Measure the
   Schultz transfer in `results/bg/stage27_summary.json`'s format. 1–2 days.
3. *(Done 6 Nov, README Stage 32: `warmind/neural_router.py`, `NEURAL_ROUTER=1`; CLINC benchmark loop 58 ± 8 % vs WTA 52 ± 6 %.)* **Shadow-mode routing on real logs**: `warmind/scripts/train_router_from_logs.py`
   on the user's `warmind.db`; report unseen-message accuracy per destination
   before anything is wired into dispatch. 1 day, needs the real log.
4. *(Started 9 Nov, README Stage 36: a prediction rule taught by the percept reads letters in words
   +7.4 points better on known words, +2.6 on new ones, and +3.6 / +1.9 beyond a shuffled-order control,
   on 4 networks; a self-organising spiking sequence layer adds nothing - negative, diagnosed.)*
   *(9 Nov, README Stage 37: the sequence-layer route has ~1 point of headroom here - letter-pair cells
   make reading slightly worse; 10x more reading + unscaled inputs give +2.4 / +1.5 points; the rest is
   the letter units' own errors.)* **Letters → words, what remains**: (a) better letters (82 % alone;
   also a better teacher) - e.g. cells that see parts of the image, as Stage 23 suggested; (b) words as
   units: word cells that use all of a word's letters at once (an ideal reader that knows the word list
   gets 63.7 % of known words vs 56 % now). Data: `results/words/`; `scripts/word_reading.py`,
   `scripts/analysis/stage37.py`. 1-2 days each.
   *(10-11 Nov, README Stage 38: the user's order is letters first, then capitals and how they relate to
   lowercase, then words.  Lowercase: new family-split test (300 Google Fonts typefaces, 60 never-seen
   designs); V1-like sparse edge cells feeding the learned layer: 86.9 +/- 0.4 % (4 networks), pixels 69.6 %;
   edge cells read directly 97.8 %; words follow - whole words ~95 % from edge-based letter units.)*
   **Next, in order**: (a) render letters on a common baseline / x-height (the i<->l errors are largely the
   per-letter box scaling); (b) capitals and their relation to lowercase (abstract letter identity);
   (c) words read through the learned layer, not only the edge cells; (d) finish the queued re-check of the
   Stage 36 sequence layer at normal inhibition (job queue `words_rec_*_i1_fixed`).
5. *(Done 7 Nov, README Stages 33–34: with a corrected LFP proxy the 30–45 Hz slope is near human (−2.2 vs −2.8/−3.0); slow power ~25 dB short, open problem 13.)* **Stage 7, cheaply**: compare the column's population spectrum with human
   intracranial spectra (`scripts/rhythm_diagnostics.py` is the start). Hours.
6. *(Cell groups done 8 Nov, README Stage 35: `BasalGanglia(merged=True)`, the default; spikes identical in
   8/8 learner runs, `scripts/verify_merged_bg.py`; loop time −28 %, about 1 s less per trial.)* **Speed**, what
   remains: merge the loop's 19 synapse pathways (~2 s per 2 s simulated). This is not exact (it changes the
   order of float additions), so it needs four networks within the seed spread. Optionally, merge the dopamine
   cells the exact way. Half a day.
7. **Stage 2 (full-scale column)** needs a GPU simulator port (GeNN or NEST
   GPU) and ≥ 64 GB; do not attempt it in Brian2 runtime on this machine.
8. **Hippocampus** is its own stage; **cerebellum** waits for a body (Stage
   5); **neuromodulator circuits** replace the surprise/uncertainty formulas
   once the dopamine system is a TD circuit.

*Current best loop settings (6 Nov): `results/hom/configs.json` → `r25_hom` (25 % routed + relay homeostasis; two and eight areas), plus the dopamine value pathway and critic/value homeostasis of `td_hom2` when the cue burst is wanted. Open: ~5-point gap to the hand-made circuit in eight areas.*

"Putting it all together" — one system rather than parts — is items 1, 3 and
4 wired through RASPUTIN: the sensory layers as the cue source
(`LearnerColumn(sensory=...)` already takes them), the loop with dopamine as
the selector and learning signal, the eight-area column as the cortex, and
RASPUTIN's own outcome signals (did the routed request succeed) as the
primary reward. Every interface for that exists; what is missing is the
structural step in item 1, without which the loop is a passenger.

## 8. Lessons that cost time (do not relearn them)

* Brian2: `e` is a reserved constant (use `elig`); `delay` inside `on_post`
  is the post pathway's (keep an explicit per-synapse `dax`); `x_post` and
  `tr_post` are reserved suffixes; `run_regularly` takes no `namespace`
  (put constants in the group's namespace); a `PoissonGroup` fires at most
  once per time step (spread strong drives over many fibres:
  `PATTERN_FIBRES`); `net.store()` builds a full in-RAM copy (lean mode
  avoids it; `reset` = rebuild from the seed).
* Memory: a slice of the spike monitor keeps the whole record alive — copy
  it and clear the monitor each step (`resize(0)` + `variables["N"].set_value(0)`).
* Runs die: the machine restarts; the runner, `.partial.json` files and
  checkpoints exist for that. Never leave anything in `/tmp`.
* Sensory layers collapse to one template under too much ink or too coarse a
  time step (`--ink 100`, `--dt 0.25`, `max_cells=50` safety valve); never
  checkpoint a collapsed state.
* Accuracy alone can hide a broken part: the first critic was silent while
  learning read 96 %; the dopamine log showed it. Log the signal, not just
  the score.
* The cue cells fire ~6 spikes/s during a cue, not the 30 of a Poisson
  stand-in; gains fitted on stand-ins must be re-fitted in the learner.
* Positive feedback into the striatum (thalamostriatal input) locks every
  channel on; the loop's working range is 10–16 spikes/s of cortical pool
  rate.
* Two seeds mislead (±10 points); four networks or it is a pilot.
* The double-size learner only helped once selection was scaled to pool
  size (`wta_ref_pool`); in eight areas double size was not clearly better.
* H01 within-slab probabilities are far below true ones (0.17 mm slab);
  the fit uses in-degree budgets and lateral fall-off, not raw pair counts.

## 9. How a stage is done here (the template)

Question → numerical stand-in (minutes) → spiking pilot on two networks →
four networks with held-out settings → write-up in `README.md` (stage
section: what was taken from data, what was fitted, what was assumed, what
was left out, the table, three honest readings, what it settles) → `AUDIT.md`
follow-up section and the status table → tests → `sync_repo.sh` commit →
artifact republish (`publish_files`, patch mode, index paragraph + file
list). Nothing is "done" until the last three are.
