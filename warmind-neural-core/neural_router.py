"""Shadow router: the spiking brain watches RASPUTIN route requests and learns to make the same call.

SHADOW ONLY.  Nothing here changes which submind RASPUTIN consults or what it says.  For every message
RASPUTIN answers, main.py hands this module (message, what RASPUTIN actually did); a background thread
shows the message to the brain, records the brain's own pick, and rewards it +1 when it matches and
-1/(K-1) when it does not (balanced, so guessing earns nothing on average).  Agreement is logged to the
`brain_shadow` table and reported at GET /api/neural/shadow, so you can see whether the brain could be
trusted with the decision long before it is given one.

The brain (README Stages 20-31 of synthetic-connectome):
  text -> self-organising sensory layer (unsupervised, results/sensory/s400_final.npz)
       -> 9/46d action pools, one per destination, in the v2 two-area column
       -> basal-ganglia loop: thalamic relay carrying 25 % of the pools' drive, relay homeostasis,
          dopamine population with a critic; the reward is the dopamine response to the outcome
Destinations: RASPUTIN (answered itself), PLUTUS, GEO (any regional submind), and ASCLEPIUS when
NEURAL_ROUTER_DESTS includes it.  Anything else is folded into RASPUTIN.

On start it replays up to NEURAL_ROUTER_REPLAY of the most recent logged decisions from `tool_calls`
(the same labels scripts/train_router_from_logs.py uses), oldest first, so a restart does not lose
what it learned; the live stream then continues the training.

Environment
-----------
NEURAL_ROUTER=0                 1 = run the shadow router (independent of NEURAL_CORE)
NEURAL_ROUTER_DESTS=RASPUTIN,PLUTUS,GEO
NEURAL_ROUTER_SENSORY=<path>    trained sensory layer (default: synthetic-connectome/results/sensory/s400_final.npz)
NEURAL_ROUTER_REPLAY=300        logged decisions replayed at start (0 = none)
NEURAL_ROUTER_SELECTION=bg      "bg" (the loop, Stage 31 settings) or "wta" (hand-made circuit)
"""
from __future__ import annotations

import collections
import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

GEO_SUBMINDS = {"VERDE", "KHEPRI", "BOREAS", "TAIGA", "KURNAI", "CHARLEMAGNE"}

NEURAL_ROUTER_ENABLED = os.environ.get("NEURAL_ROUTER", "0") == "1"
NEURAL_ROUTER_DESTS = [d.strip().upper() for d in os.environ.get("NEURAL_ROUTER_DESTS", "RASPUTIN,PLUTUS,GEO").split(",") if d.strip()]
NEURAL_ROUTER_REPLAY = int(os.environ.get("NEURAL_ROUTER_REPLAY", "300"))
NEURAL_ROUTER_SELECTION = os.environ.get("NEURAL_ROUTER_SELECTION", "bg")
DB_PATH = os.environ.get("DB_PATH", "warmind.db")

# Stage 31 loop settings (synthetic-connectome results/hom/configs.json, "r25_hom")
LOOP_BG = {"ctx_gain": 2.0, "ctx_gain_stn": 0.3, "thal_gain": 1.0, "plastic": True, "ctx_gain_cue": 1.5, "eta_bg": 0.5,
           "post_lin": 0.0, "post_gain": 1.0, "thal_drive_mV": 31.0, "thal_n_in": 64, "thal_route": 0.25, "thal_hom": 0.1}
LOOP_DA = {"crit_gain": 2.5, "crit_eta": 0.05}

SCHEMA = """CREATE TABLE IF NOT EXISTS brain_shadow (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,            -- 'live' or 'replay'
    message TEXT NOT NULL,
    label TEXT NOT NULL,             -- what RASPUTIN did
    brain TEXT NOT NULL,             -- what the brain picked
    correct INTEGER NOT NULL,
    detail TEXT                      -- JSON: pool spikes, SNr, dopamine
)"""


def route_label(consulted: Optional[str], dests=None) -> str:
    """Map what RASPUTIN did onto the router's destinations: the first submind consulted in its first
    response, GEO for any regional submind, RASPUTIN for none or anything the router does not cover."""
    dests = dests or NEURAL_ROUTER_DESTS
    c = (consulted or "").strip().upper()
    if not c:
        return "RASPUTIN"
    if c in GEO_SUBMINDS and "GEO" in dests:
        return "GEO"
    return c if c in dests else "RASPUTIN"


def logged_decisions(db_path: str, limit: int, dests=None):
    """(message, label) for RASPUTIN's first responses, oldest first, at most `limit` newest ones"""
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("SELECT user_message, tool_name, consulted, created_at FROM tool_calls "
                           "WHERE target = 'RASPUTIN' AND round = 0 ORDER BY created_at, id").fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        con.close()
    turns = collections.OrderedDict()
    for msg, tool, consulted, ts in rows:
        key = (msg, (ts or "")[:16])
        if turns.get(key, "RASPUTIN") == "RASPUTIN":
            turns[key] = route_label(consulted if tool == "consult_submind" else None, dests)
    items = [(m, lab) for (m, _), lab in turns.items()]
    return items[-limit:] if limit > 0 else []


def default_column_factory(dests):
    """the brain: v2 two-area column, one action pool per destination, sensory text layer, the loop"""
    from neural_core import _find_synth_path
    path = _find_synth_path()
    if path and path not in sys.path:
        sys.path.insert(0, path)
    from synth.learner import build_learner_network, LearnerColumn, QUICK_SETTINGS, WTA_REF_POOL
    sensory = os.environ.get("NEURAL_ROUTER_SENSORY") or os.path.join(path, "results", "sensory", "s400_final.npz")
    col = build_learner_network(n_scale=0.015, seed=41, preset="human_h01_dendritic_v2")
    kw = dict(selection="bg", bg_kw=dict(LOOP_BG), da_kw=dict(LOOP_DA)) if NEURAL_ROUTER_SELECTION == "bg" else {}
    L = LearnerColumn(col, seed=41, structural=True, surprise_gain=3.0, fast=True, lean=True,
                      genome=dict(eta=QUICK_SETTINGS["eta"], wta_ref_pool=WTA_REF_POOL), uncert_gain=QUICK_SETTINGS["uncert_gain"],
                      n_actions=len(dests), pattern_cues=True, sensory=sensory, sens_lr=0.33, **kw)
    L.step(500.0)
    return L


class ShadowRouter:
    def __init__(self, dests=None, db_path: str = DB_PATH, replay: int = NEURAL_ROUTER_REPLAY,
                 column_factory: Callable = default_column_factory, max_queue: int = 50):
        self.dests = list(dests or NEURAL_ROUTER_DESTS)
        if "RASPUTIN" not in self.dests:
            self.dests.insert(0, "RASPUTIN")
        self.db_path, self.replay_n, self.factory = db_path, int(replay), column_factory
        self.status, self.error = "stopped", None
        self.column = None
        self._q: collections.deque = collections.deque(maxlen=max_queue)   # oldest dropped if the brain falls behind
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.dropped = 0
        self.recent = collections.deque(maxlen=500)                        # (source, label, brain, correct)
        self._ensure_table()

    # ---- storage -----------------------------------------------------------
    def _db(self):
        return sqlite3.connect(self.db_path, timeout=15.0)

    def _ensure_table(self):
        con = self._db()
        try:
            con.execute(SCHEMA); con.commit()
        finally:
            con.close()

    def _record(self, source, message, label, brain, correct, detail):
        try:
            con = self._db()
            try:
                con.execute("INSERT INTO brain_shadow (created_at, source, message, label, brain, correct, detail) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (datetime.now(timezone.utc).isoformat(), source, message[:500], label, brain, int(correct),
                             json.dumps(detail)))
                con.commit()
            finally:
                con.close()
        except Exception as e:                                             # logging must never break anything
            print(f"[shadow-router] record failed (ignored): {e}")

    # ---- lifecycle -----------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        if self.factory is default_column_factory:
            # Brian2 installs a signal handler on first import, which only the main thread may do
            try:
                from neural_core import _find_synth_path
                path = _find_synth_path()
                if path and path not in sys.path:
                    sys.path.insert(0, path)
                import brian2  # noqa: F401
            except Exception as e:
                self.error, self.status = f"brian2 import failed: {e}", "error"
                return
        self.status = "warming"
        self._thread = threading.Thread(target=self._run, name="shadow-router", daemon=True)
        self._thread.start()

    def stop(self, timeout=10.0):
        self._stop.set(); self._wake.set()
        if self._thread:
            self._thread.join(timeout)
        self.status = "stopped"

    def observe(self, message: str, consulted: Optional[str]):
        """called by main.py after RASPUTIN's first response; returns at once"""
        if not message or not message.strip() or self.status in ("stopped", "error"):
            return
        if len(self._q) == self._q.maxlen:
            self.dropped += 1
        self._q.append((message, route_label(consulted, self.dests)))
        self._wake.set()

    # ---- worker ----------------------------------------------------------------
    def _step(self, message, label, source):
        r = self.column.trial(message)
        brain = self.dests[int(r["choice"]) - 1]
        correct = brain == label
        K = len(self.dests)
        rw = self.column.reward(1.0 if correct else -1.0 / max(K - 1, 1), note=f"shadow {source}")
        detail = dict(pool_spikes=r.get("pool_spikes"), snr_hz=r.get("snr_hz"), dopamine=rw.get("dopamine"))
        self.recent.append((source, label, brain, bool(correct)))
        self._record(source, message, label, brain, correct, detail)
        return brain, correct

    def _run(self):
        try:
            self.column = self.factory(self.dests)
            if self.replay_n > 0:
                self.status = "replaying"
                for m, lab in logged_decisions(self.db_path, self.replay_n, self.dests):
                    if self._stop.is_set():
                        return
                    self._step(m, lab, "replay")
            self.status = "running"
            while not self._stop.is_set():
                if not self._q:
                    self._wake.wait(1.0); self._wake.clear()
                    continue
                m, lab = self._q.popleft()
                self._step(m, lab, "live")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error, self.status = f"{type(e).__name__}: {e}", "error"

    # ---- report ----------------------------------------------------------------
    def stats(self) -> dict:
        live = [x for x in self.recent if x[0] == "live"]
        def acc(xs):
            return round(sum(c for *_, c in xs) / len(xs), 3) if xs else None
        conf = collections.Counter((lab, br) for _, lab, br, _ in live)
        per = {d: acc([x for x in live if x[1] == d]) for d in self.dests}
        base = (max(collections.Counter(x[1] for x in live).values()) / len(live)) if live else None
        return dict(status=self.status, error=self.error, destinations=self.dests, queued=len(self._q), dropped=self.dropped,
                    live_seen=len(live), agreement_last50=acc(live[-50:]), agreement_last200=acc(live[-200:]),
                    agreement_by_destination=per, majority_baseline=round(base, 3) if base else None,
                    replayed=sum(1 for x in self.recent if x[0] == "replay"),
                    confusion={f"{a}->{b}": n for (a, b), n in sorted(conf.items())},
                    note="shadow only: RASPUTIN's routing is never changed by this")


_router: Optional[ShadowRouter] = None


def start_shadow_router(**kw) -> Optional[ShadowRouter]:
    global _router
    if not NEURAL_ROUTER_ENABLED and not kw.pop("force", False):
        return None
    if _router is None:
        _router = ShadowRouter(**kw)
        _router.start()
    return _router


def shadow_observe(message: str, consulted: Optional[str]):
    """safe to call always: does nothing unless the shadow router is running"""
    if _router is not None:
        try:
            _router.observe(message, consulted)
        except Exception as e:
            print(f"[shadow-router] observe failed (ignored): {e}")


def shadow_stats() -> dict:
    return _router.stats() if _router is not None else dict(status="disabled", note="set NEURAL_ROUTER=1")
