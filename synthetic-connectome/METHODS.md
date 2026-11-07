# Working standards

Rules this project follows from 4 November 2026 on. Each one exists because
breaking it cost time or credibility earlier (see `AUDIT.md`, section 3).

## Before a result is believed

1. **Four networks minimum** for any claim about learning or dynamics, each a
   different synthesis seed, and the spread is reported with the mean. Two
   seeds are a pilot, and are called one. (Seed-to-seed spread on the cue
   task is about ±10 accuracy points.)
2. **Held-out everything:** networks not used in a search, messages or fonts
   never trained on, and — for the learning rule — settings chosen on one set
   of networks and reported on another.
3. **A numerical stand-in before any spiking run longer than an hour.** If an
   idealised version of the rule or the layer cannot learn the task, the
   spiking version will not either, and the stand-in takes minutes
   (`scripts/analysis/ideal*.py`, `proxy_sens.py` are the pattern).
4. **Exactness before speed.** A speed-up or memory saving is adopted only
   if it reproduces the previous output exactly (`scripts/verify_fast.py`,
   `tests/test_synth.py::test_learner_default_path_is_unchanged`) or, where
   randomness is unavoidable, matches it within the measured seed spread.

## Model parameters

5. **Every number outside the data is labelled** in the preset's provenance
   as `CALIBRATED` (fitted to a target) or `ASSUMED` (chosen from the
   literature or by judgement), with the source or the target it was fitted
   to. The provenance print-out is part of every stage write-up. Today's list:
   external gain, background conductance, deep external input (fitted), E→FS
   deep scaling (fitted), FS soma fraction, dendritic membrane share, the
   action-selection circuit, and the nine evolved learning-rule settings.
6. **Reference ranges are human where human data exist** and say when they
   are not (`results/rate_targets.json`). A population outside its range is
   an open problem until closed; `scripts/check_rates.py` is run after any
   change to the column.
7. **Defaults are not changed silently.** A change to a default that alters
   results goes with (a) the regression references regenerated on purpose,
   (b) `check_rates.py` and the learning checks rerun, (c) a stage entry
   saying what changed and why.

## Running experiments

8. **Checkpoint every run longer than an hour**; never write a checkpoint
   from a failed state; mark failed runs so a watchdog does not resume them.
   The machine restarts without warning.
9. **Results go in `results/`, code goes in `scripts/` or `synth/`, nothing
   in `/tmp`.** Every number in a write-up must be reproducible from a file
   in the repository.
10. **Commit after every closed item** (`/home/claude/jobs/sync_repo.sh
    "message"` from the working copy; the repository is the record, the
    artifact is the mirror).
11. **Tests run before a commit:** `python -m pytest tests -q` (seconds);
    `-m slow` (a minute) after any change to `synth/`.

## Writing it up

12. **Say what the object is.** It is human association cortex at 1.5–3 %
    neuron density with measured local wiring and physiology, macaque
    long-range wiring, a hand-made action-selection circuit, and reward
    supplied from outside. "Simulated human brain" is the goal, not the
    object.
13. **Problems noticed in a stage go into *Open problems* the same day**,
    numbered, and stay there until a stage closes them.
14. **Negative results are written up** with the same care as positive ones,
    including what was tried and what it ruled out.
15. **One status page stays current** (the roadmap table at the top of
    `README.md`); the stage log underneath is history and is not rewritten.

16. **A new measurement gets its own control before its result is read.**
    Stage 33 interpreted a field spectrum from a proxy that had never been
    checked on a signal with a known answer; the proxy itself was the cause
    of the mismatch (Stage 34). Run the measurement on a case whose result is
    known (a constant-input or shuffled control, or a synthetic signal)
    first.
