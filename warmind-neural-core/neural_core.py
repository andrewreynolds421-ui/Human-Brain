"""
Neural core: a live simulated human cortical column beside the Warmind.

Inert unless NEURAL_CORE=1.  With the flag off, the router answers 503 and
the tool is not registered, so an existing deployment behaves identically.

What it is
----------
A spiking-neuron model of ~1/10 of a 1 mm² column of human temporal cortex
(3,252 neurons, 1.3 M synapses) built from the synthetic-connectome project:
wiring statistics from the H01 electron-microscopy volume, membrane
constants from the Allen Cell Types database, synaptic strengths and
dynamics from the Allen Synaptic Physiology database.  It runs continuously
in a background thread at roughly 0.2-0.4x real time on two cores.

What the dashboard can do with it
---------------------------------
GET  /api/neural/state      per-layer firing rates, regime, recent spikes
POST /api/neural/stimulate  {layer, strength, duration_ms, label}  a pulse of
                            input into a layer (an event arrived)
POST /api/neural/reward     {value: -1..1, note}  dopamine-like signal: the
                            E->E synapses that were just co-active are
                            strengthened (value > 0) or weakened (< 0)
POST /api/neural/control    {action: pause | resume | reset}
GET  /api/neural/info       what the model is and where its numbers came from

RASPUTIN gets the same through the `neural_core` tool.

Environment
-----------
NEURAL_CORE=1               enable
NEURAL_CORE_PATH=...        directory containing the synthetic-connectome
                            package (default: ../synthetic-connectome, or a
                            `synthetic-connectome` folder next to this file)
NEURAL_CORE_MODE=membranes  "membranes" (asynchronous state, default),
                            "full" (all measured synapses; bursts, ~2x slower), or
                            "dendritic" (12 populations: fast-spiking / other
                            interneurons, two-compartment pyramidal cells,
                            measured kinetics, in-vivo background; slowest)
NEURAL_CORE_SCALE=0.1       fraction of the full column (0.05 is ~3x faster)
NEURAL_CORE_CHUNK_MS=250    biological ms simulated per step
NEURAL_CORE_REALTIME=0      1 = never run faster than real time
NEURAL_CORE_STRUCTURAL=1    learner model: synapses form and disappear (0 = weights only)
NEURAL_CORE_MODE=learner8   the learner inside all eight areas (~4-5x slower)
NEURAL_CORE_SURPRISE=3.0    learner model: learning speeds up when rewards stop matching
NEURAL_CORE_QUICK=1         learner models: learn in fewer trials (higher learning rate that
                            eases off as rewards come in; 0 = previous rule)
NEURAL_CORE_LEARNER_SCALE=0.03   learner size: 0.03 = 2x the neurons (default; ~1.4 GB, ~2x slower
                            per trial, learns sooner and better); 0.015 = previous size
NEURAL_CORE_LEARNER8_SCALE=0.015 learner8 size (2x not yet tested in all eight areas)
NEURAL_CORE_COLUMN=v2       column calibration: v2 (stage 25: every population at or near its
                            reference firing rate; learns faster) or v1 (stage 3d, the earlier results)
NEURAL_CORE_LEAN=1          learner models: same dynamics, ~40% less RAM while running
                            (reset snapshot on disk, unused wiring tables freed)
NEURAL_CORE_FAST=1          learner models: same dynamics, ~2x faster (merged inputs,
                            per-neuron synaptic state; 0 = original per-synapse code)
NEURAL_CORE_SELECTION=wta   learner models: how the action pools compete.  "wta" = the
                            hand-made selection groups; "bg" = the basal-ganglia-thalamus loop
                            built from Humphries et al. 2006 (stage 26; synth/basal_ganglia.py)
NEURAL_CORE_BG={}           JSON options for the loop (ctx_gain, ctx_gain_stn, thal_gain, ...)
                            expectations (unexpected uncertainty; 0 = off)

Thread model
------------
Brian2 is not thread-safe, so exactly one worker thread owns the network.
Requests never touch it: `state()` reads the snapshot the worker publishes
after every step; stimulate/reward/control are queued and applied by the
worker between steps.  Nothing in here blocks a request for longer than a
dict copy.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

NEURAL_CORE_ENABLED = os.environ.get("NEURAL_CORE", "0") == "1"
NEURAL_CORE_MODE = os.environ.get("NEURAL_CORE_MODE", "membranes")
NEURAL_CORE_SCALE = float(os.environ.get("NEURAL_CORE_SCALE", "0.1"))
NEURAL_CORE_CHUNK_MS = float(os.environ.get("NEURAL_CORE_CHUNK_MS", "250"))
NEURAL_CORE_REALTIME = os.environ.get("NEURAL_CORE_REALTIME", "0") == "1"
NEURAL_CORE_STRUCTURAL = os.environ.get("NEURAL_CORE_STRUCTURAL", "1") == "1"   # learner: synapses form / disappear
NEURAL_CORE_SURPRISE = float(os.environ.get("NEURAL_CORE_SURPRISE", "3.0"))       # learner: surprise boost (0 = off)
NEURAL_CORE_FAST = os.environ.get("NEURAL_CORE_FAST", "1") == "1"                  # learner: faster, same dynamics
NEURAL_CORE_QUICK = os.environ.get("NEURAL_CORE_QUICK", "1") == "1"                # learner: fewer trials to learn
NEURAL_CORE_LEAN = os.environ.get("NEURAL_CORE_LEAN", "1") == "1"                  # learner: less RAM, same dynamics
NEURAL_CORE_COLUMN = os.environ.get("NEURAL_CORE_COLUMN", "v2")                     # column calibration: v2 (stage 25) or v1
NEURAL_CORE_SELECTION = os.environ.get("NEURAL_CORE_SELECTION", "wta")               # learner: "wta" groups or the "bg" loop
NEURAL_CORE_BG = json.loads(os.environ.get("NEURAL_CORE_BG", "{}"))                  # learner: basal-ganglia loop options
COLUMN_PRESET = {"v1": "human_h01_dendritic", "v2": "human_h01_dendritic_v2"}.get(NEURAL_CORE_COLUMN, "human_h01_dendritic_v2")
NEURAL_CORE_LEARNER_SCALE = float(os.environ.get("NEURAL_CORE_LEARNER_SCALE", "0.03"))    # learner size (0.015 = previous)
NEURAL_CORE_LEARNER8_SCALE = float(os.environ.get("NEURAL_CORE_LEARNER8_SCALE", "0.015"))  # learner8 size

LAYERS = ("L23", "L4", "L5", "L6")

# mode -> (preset, preset kwargs, short description)
MODES = {
    "membranes": ("human_h01_membranes", {},
                  "8 populations, measured membranes, rodent-scaled synapses (asynchronous)."),
    "full": ("human_h01_synphys", {},
             "8 populations, all measured synapses and short-term plasticity (bursts)."),
    "dendritic": (COLUMN_PRESET, {"bg_ratio": 2.0, **({"ext_gain": 2.0} if COLUMN_PRESET == "human_h01_dendritic" else {})},
                  "12 populations (fast-spiking / other interneurons), two-compartment pyramidal "
                  "cells with inputs placed on soma or dendrite, measured synaptic kinetics, "
                  "in-vivo-like background conductance, calibrated deep-layer input; full connectivity."),
}
MODES["learner"] = ("two_area_learner", {},
                     "Two prefrontal areas (46d -> 9/46d, strongest measured link). A trial shows cue A or B; "
                     "the network chooses action 1 or 2; reward / punishment trains the synapses behind the "
                     "choice (evolved three-factor rule with action selection; chance -> 73% over 300 trials "
                     "in testing).")
MODES["learner8"] = ("eight_area_learner", {},
                      "The same learning task inside all eight association areas (46d, 9/46d, 8m, 8B, 24c, 7A, 7m, "
                      "STPc) joined by measured connections; plasticity on every long-range input to the action "
                      "cells. ~4-5x slower than 'learner' (chance -> 70% over 300 trials in testing).")
# modes that must keep every connection (scaled in-degrees distort them)
FULL_INDEGREE_MODES = {"dendritic"}

neural_router = APIRouter(prefix="/api/neural", tags=["neural core"])


# ---------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------

def _find_synth_path() -> Optional[str]:
    cands = [os.environ.get("NEURAL_CORE_PATH"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic-connectome"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "synthetic-connectome")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "synth")):
            return os.path.abspath(c)
    return None


class NeuralCoreService:
    def __init__(self, mode=NEURAL_CORE_MODE, scale=NEURAL_CORE_SCALE,
                 chunk_ms=NEURAL_CORE_CHUNK_MS, realtime=NEURAL_CORE_REALTIME, autostart=True):
        self.mode, self.scale, self.chunk_ms, self.realtime = mode, scale, chunk_ms, realtime
        self.status = "stopped"            # stopped | warming | running | paused | error
        self.error: Optional[str] = None
        self.snapshot: dict = {}
        self.info: dict = {}
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.column = None
        self.started_at = None
        if autostart:
            self.start()

    # ---- lifecycle ---------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        # Brian2 installs a SIGINT handler when first imported, which Python
        # only allows from the main thread: import it here, before the worker.
        path = _find_synth_path()
        if path and path not in sys.path:
            sys.path.insert(0, path)
        try:
            import brian2  # noqa: F401
        except Exception as e:
            self.error = f"brian2 import failed: {e}"
            self.status = "error"
            return
        self._stop.clear()
        self.status = "warming"
        self._thread = threading.Thread(target=self._run, name="neural-core", daemon=True)
        self._thread.start()

    def stop(self, timeout=10.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        self.status = "stopped"

    # ---- worker ------------------------------------------------------------
    def _build(self):
        path = _find_synth_path()
        if path is None:
            raise RuntimeError("synthetic-connectome package not found; set NEURAL_CORE_PATH")
        if path not in sys.path:
            sys.path.insert(0, path)
        from synth import synthesize, PRESETS          # noqa: E402  (heavy import, worker only)
        from synth.live import LiveColumn               # noqa: E402
        if self.mode not in MODES:
            raise ValueError(f"unknown NEURAL_CORE_MODE {self.mode!r}; use one of {sorted(MODES)}")
        preset, pkw, mdesc = MODES[self.mode]
        if self.mode in ("learner", "learner8"):
            from synth.learner import build_learner_network, LearnerColumn, QUICK_SETTINGS, WTA_REF_POOL   # noqa: E402
            n_scale = NEURAL_CORE_LEARNER8_SCALE if self.mode == "learner8" else NEURAL_CORE_LEARNER_SCALE
            col = build_learner_network(n_scale=n_scale, seed=1, areas="all" if self.mode == "learner8" else None,
                                        preset=COLUMN_PRESET)
            self.info = dict(model=col.name, mode=self.mode, neurons=col.N, synapses=col.S, layers=list(LAYERS),
                             selection=NEURAL_CORE_SELECTION,
                             modes={k: v[2] for k, v in MODES.items()},
                             description=("Two human-like prefrontal columns (H01 wiring, Allen human physiology) "
                                          "joined by the measured macaque 46d -> 9/46d pathway. " + mdesc))
            self.column = LearnerColumn(col, seed=1, structural=NEURAL_CORE_STRUCTURAL,
                                        surprise_gain=NEURAL_CORE_SURPRISE, fast=NEURAL_CORE_FAST, lean=NEURAL_CORE_LEAN,
                                        selection=NEURAL_CORE_SELECTION, bg_kw=NEURAL_CORE_BG,
                                        **(dict(genome=dict(eta=QUICK_SETTINGS["eta"], wta_ref_pool=WTA_REF_POOL),
                                                uncert_gain=QUICK_SETTINGS["uncert_gain"]) if NEURAL_CORE_QUICK else {}))
            self.chunk_ms = min(self.chunk_ms, 50.0)        # ~2 s wall per step at ~0.02x real time
            return
        params = PRESETS[preset](verbose=False, **pkw)
        k = 1.0 if self.mode in FULL_INDEGREE_MODES else self.scale
        col = synthesize(params, N_scaling=self.scale, K_scaling=k, seed=1)
        prov = getattr(params, "provenance", {})
        self.info = dict(
            model=col.name, mode=self.mode, scale=self.scale, neurons=col.N, synapses=col.S,
            description=("Spiking model of a human temporal-cortex column. Wiring statistics: H01 "
                         "electron-microscopy volume (Shapson-Coe et al. 2024). Membrane constants: "
                         "Allen Cell Types (413 human neurons). "
                         + ("Synaptic strengths and short-term plasticity: Allen Synaptic Physiology "
                            "(17,490 human pairs). " if self.mode in ("full", "dendritic") else
                            "Synaptic strengths: rodent reference, rebalanced. ")
                         + "Mode: " + mdesc + " "
                         + "Reward rule: reward-gated Hebbian update on recently co-active "
                           "excitatory synapses with synaptic scaling."),
            modes={k: v[2] for k, v in MODES.items()},
            provenance={k: (v if isinstance(v, (list, str)) else
                            f"{sum(1 for x in v.values() if not str(x).startswith(('rodent', 'none')))}/{len(v)} measured")
                        for k, v in prov.items()},
            layers=list(LAYERS),
        )
        self.column = LiveColumn(col, seed=1, eligibility_ms=300.0)

    def _run(self):
        try:
            self._build()
            self.column.step(self.chunk_ms)        # first step compiles the code objects
            self.started_at = time.time()
            self.status = "running"
        except Exception as e:                      # surface build errors instead of dying silently
            self.error = f"{type(e).__name__}: {e}"
            self.status = "error"
            traceback.print_exc()
            return
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.1)
                continue
            self._drain_queue()
            t0 = time.perf_counter()
            try:
                self.column.step(self.chunk_ms)
                snap = self.column.state()
                snap["status"] = self.status
                self.snapshot = snap
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                self.status = "error"
                traceback.print_exc()
                return
            if self.realtime:
                lag = self.chunk_ms / 1000.0 - (time.perf_counter() - t0)
                if lag > 0:
                    time.sleep(lag)

    def _drain_queue(self):
        while self._queue:
            with self._lock:
                cmd, kw, result = self._queue.popleft()
            try:
                if cmd == "stimulate":
                    self.column.stimulate(**kw); result["ok"] = True
                elif cmd == "reward":
                    result.update(self.column.reward(**kw)); result["ok"] = True
                elif cmd == "reset":
                    if getattr(self.column, "reset_needs_rebuild", False):
                        self.column = None                 # free the old network before building anew
                        import gc; gc.collect()
                        self._build()                      # same seed -> same starting state
                    else:
                        self.column.reset()
                    result["ok"] = True
                elif cmd == "trial":
                    if not hasattr(self.column, "trial"):
                        raise ValueError("trials need a learner model (set_model learner or learner8)")
                    result.update(self.column.trial(**kw)); result["ok"] = True
                elif cmd == "set_model":
                    if kw["mode"] not in MODES:
                        raise ValueError(f"unknown model {kw['mode']!r}; choose from {sorted(MODES)}")
                    old = self.mode
                    self.mode = kw["mode"]
                    self.column = None                     # free the old network first (no 2x memory)
                    import gc; gc.collect()
                    try:
                        self._build()
                        self.column.step(self.chunk_ms)
                    except Exception:
                        self.mode = old
                        self._build()
                        raise
                    result["ok"] = True; result["model"] = self.info.get("model")
            except Exception as e:
                result["ok"] = False; result["error"] = str(e)
            try:                                    # status reflects the command at once
                snap = self.column.state(); snap["status"] = self.status; self.snapshot = snap
            except Exception:
                pass
            result["done"] = True

    # ---- request side ------------------------------------------------------
    def enqueue(self, cmd: str, wait: Optional[float] = None, **kw) -> dict:
        if wait is None:
            # the worker takes commands between steps; slow models need longer waits
            wait = {"set_model": 900.0, "trial": 300.0, "reset": 900.0}.get(cmd, 60.0)
        if self.status not in ("running", "paused"):
            raise HTTPException(503, f"neural core is {self.status}" + (f": {self.error}" if self.error else ""))
        result: dict = {"done": False}
        with self._lock:
            self._queue.append((cmd, kw, result))
        if self.status == "paused":
            self._drain_queue()
        t0 = time.time()
        while not result["done"] and time.time() - t0 < wait:
            time.sleep(0.02)
        return result

    def state(self) -> dict:
        snap = dict(self.snapshot) if self.snapshot else {}
        snap["status"] = self.status
        if self.error:
            snap["error"] = self.error
        snap["uptime_s"] = round(time.time() - self.started_at) if self.started_at else 0
        return snap

    def control(self, action: str) -> dict:
        if action == "pause":
            self._pause.set(); self.status = "paused" if self.status == "running" else self.status
        elif action == "resume":
            self._pause.clear(); self.status = "running" if self.status == "paused" else self.status
        elif action == "reset":
            return self.enqueue("reset")
        else:
            raise HTTPException(400, "action must be pause, resume or reset")
        return {"ok": True, "status": self.status}


_service: Optional[NeuralCoreService] = None


def start_neural_core(**kw) -> Optional[NeuralCoreService]:
    """Call once at startup (main.py does this when NEURAL_CORE=1)."""
    global _service
    if not NEURAL_CORE_ENABLED and not kw.get("force"):
        return None
    kw.pop("force", None)
    if _service is None:
        _service = NeuralCoreService(**kw)
    return _service


def get_service() -> NeuralCoreService:
    if _service is None:
        raise HTTPException(503, "neural core disabled (set NEURAL_CORE=1)")
    return _service


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

class StimulateIn(BaseModel):
    layer: str = Field("L4", description="L23, L4, L5 or L6")
    strength: float = Field(1.0, ge=0.0, le=5.0)
    duration_ms: float = Field(100.0, ge=10.0, le=5000.0)
    label: str = ""


class RewardIn(BaseModel):
    value: float = Field(1.0, ge=-1.0, le=1.0)
    note: str = ""


class ControlIn(BaseModel):
    action: str


@neural_router.get("/state")
def neural_state():
    return get_service().state()


@neural_router.get("/info")
def neural_info():
    s = get_service()
    return dict(s.info, status=s.status, error=s.error)


@neural_router.post("/stimulate")
def neural_stimulate(body: StimulateIn):
    if body.layer not in LAYERS:
        raise HTTPException(400, f"layer must be one of {LAYERS}")
    return get_service().enqueue("stimulate", layer=body.layer, strength=body.strength,
                                 duration_ms=body.duration_ms, label=body.label)


@neural_router.post("/reward")
def neural_reward(body: RewardIn):
    return get_service().enqueue("reward", value=body.value, note=body.note)


@neural_router.post("/control")
def neural_control(body: ControlIn):
    return get_service().control(body.action)


# ---------------------------------------------------------------------------
# RASPUTIN tool
# ---------------------------------------------------------------------------

NEURAL_CORE_TOOL = {
    "name": "neural_core",
    "description": (
        "Read or act on the Warmind's neural core: a live spiking simulation of a human "
        "cortical column (built from real human connectome and patch-clamp data) running "
        "beside you. Actions: 'status' returns per-layer firing rates, regime (silent / "
        "asynchronous / synchronous / bursting) and recent events; 'stimulate' sends a pulse "
        "of input into a layer (use when something notable arrives: a message, news, a task); "
        "'reward' sends a dopamine-like signal in [-1, 1] that strengthens or weakens the "
        "synapses that were just active (use when the user approves or disapproves of what "
        "the core did); 'reset' restores the initial state; 'set_model' rebuilds the column as "
        "another model ('membranes', 'full', 'dendritic', or 'learner') and takes up to a few minutes. "
        "In the 'learner' and 'learner8' (eight areas, slower) models, 'trial' shows cue A or B to a two-area prefrontal network and returns "
        "its choice (action 1 or 2); follow it with 'reward' (+1 if the choice is the one you want for "
        "that cue, -1 if not) to train it - it learns whatever mapping you consistently reward. "
        "It does not think or answer; "
        "it is tissue dynamics you can watch and perturb. Report numbers as they are."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["status", "stimulate", "reward", "reset", "set_model", "trial"]},
            "cue": {"type": "string", "enum": ["A", "B"], "description": "for trial (learner model)"},
            "model": {"type": "string", "enum": list(MODES), "description": "for set_model ('learner8' = eight-area learner, slower)"},
            "layer": {"type": "string", "enum": list(LAYERS),
                      "description": "for stimulate: L4 = sensory input, L23 = association, L5/L6 = output layers"},
            "strength": {"type": "number", "minimum": 0, "maximum": 5, "description": "for stimulate; 1.0 = a strong event"},
            "duration_ms": {"type": "number", "minimum": 10, "maximum": 5000},
            "value": {"type": "number", "minimum": -1, "maximum": 1, "description": "for reward"},
            "note": {"type": "string", "description": "short label logged with the event"},
        },
        "required": ["action"],
    },
}


def run_neural_core_tool(inputs: dict) -> str:
    """Execute a neural_core tool call; returns the text result for the model."""
    try:
        s = get_service()
    except HTTPException as e:
        return f"neural core unavailable: {e.detail}"
    action = (inputs or {}).get("action", "status")
    if action == "status":
        st = s.state()
        if not st.get("rates"):
            return f"neural core status: {st.get('status')}" + (f" ({st.get('error')})" if st.get("error") else "")
        rates = ", ".join(f"{k} {v:.1f}" for k, v in st["layer_rates"].items())
        ev = "; ".join(f"t={e['t_ms']} {e['kind']} {e['detail']}" for e in st["events"][-4:]) or "none"
        ln = st.get("learner")
        extra = (f"; learner: {ln['trials_rewarded']} trials with feedback, mean reward over last 20 "
                 f"{ln['mean_reward_last20']}, waiting for feedback: {ln['waiting_for_feedback']}"
                 + (f", synapses removed / formed so far {ln['synapses_removed_total']} / {ln['synapses_formed_total']}"
                    if "synapses_removed_total" in ln else "")
                 + f", surprise {ln.get('surprise', 0)}, learning rate x{ln.get('learning_rate_x', 1.0)}") if ln else ""
        return (f"neural core {st['status']} [{s.mode}: {st.get('model')}]{extra}: sim time {st['t_ms'] / 1000:.1f} s, mean rate {st['mean_rate']} Hz, "
                f"regime {st['regime']} (synchrony {st['synchrony']}), layer rates (Hz): {rates}; "
                f"rewards so far {st['rewards']}, cumulative weight change {st['cumulative_weight_change']}; "
                f"recent events: {ev}")
    if action == "stimulate":
        r = s.enqueue("stimulate", layer=inputs.get("layer", "L4"), strength=float(inputs.get("strength", 1.0)),
                      duration_ms=float(inputs.get("duration_ms", 100.0)), label=str(inputs.get("note", "")))
        return "stimulus delivered" if r.get("ok") else f"stimulus failed: {r.get('error', 'timeout')}"
    if action == "trial":
        r = s.enqueue("trial", cue=str(inputs.get("cue", "A")))
        if r.get("ok"):
            return (f"cue {r['cue']} -> the network chose action {r['choice']} (spikes in action groups 1/2: "
                    f"{r['pool_spikes'][0]}/{r['pool_spikes'][1]}); send 'reward' +1 or -1 to train it")
        return f"trial failed: {r.get('error', 'timeout')}"
    if action == "reward":
        r = s.enqueue("reward", value=float(inputs.get("value", 1.0)), note=str(inputs.get("note", "")))
        if r.get("ok") and "applied" in r:
            if not r["applied"]:
                return f"reward not applied: {r['reason']}"
            return (f"reward {r['value']:+.2f} applied to the trial cue {r['cue']} -> action {r['choice']} "
                    f"(prediction error {r['prediction_error']:+.2f}, {r['eligible_synapses']} tagged synapses, "
                    f"relative weight change {r['relative_weight_change']:.4f})")
        if r.get("ok"):
            return (f"reward {r['value']:+.2f} applied to {r['eligible_synapses']} eligible synapses "
                    f"({r['active_neurons']} recently active neurons), relative weight change {r['relative_weight_change']:.4f}")
        return f"reward failed: {r.get('error', 'timeout')}"
    if action == "reset":
        r = s.enqueue("reset")
        return "neural core reset to its initial state" if r.get("ok") else f"reset failed: {r.get('error', 'timeout')}"
    if action == "set_model":
        mode = str(inputs.get("model", ""))
        if mode not in MODES:
            return f"unknown model {mode!r}; choose from {', '.join(MODES)}"
        if mode == s.mode:
            return f"already running {mode}"
        r = s.enqueue("set_model", mode=mode)
        if r.get("ok"):
            return f"neural core rebuilt as {mode} ({r.get('model')}): {MODES[mode][2]}"
        return f"set_model failed: {r.get('error', 'timeout')}"
    return f"unknown action {action!r}"


NEURAL_CORE_SYSTEM_NOTE = (
    "\n\nA live neural core runs beside you: a spiking simulation of a human cortical "
    "column built from real human connectome and patch-clamp data. Use the neural_core "
    "tool to read its state, to send it input when something notable arrives, or to "
    "reward/punish it when the user reacts to what it did. It does not think or speak; "
    "describe it accurately as tissue dynamics, never as your mind."
)
