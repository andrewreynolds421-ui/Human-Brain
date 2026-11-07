"""Shadow router plumbing, with a stand-in brain (no simulator): fast."""
import os
import sqlite3
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.pop("NEURAL_ROUTER", None)
import neural_router as nr  # noqa: E402


class StubBrain:
    """always picks destination `pick` (1-based); records rewards"""
    def __init__(self, pick=1):
        self.pick, self.rewards, self.trials = pick, [], []

    def trial(self, cue):
        self.trials.append(cue)
        return dict(cue=cue, choice=self.pick, pool_spikes=[1, 2, 3], snr_hz=[20.0, 5.0, 20.0])

    def reward(self, value, note=""):
        self.rewards.append(value)
        return dict(applied=True, dopamine=dict(da_hz=25.0, delta=1.0))


def make_db(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE tool_calls (id INTEGER PRIMARY KEY AUTOINCREMENT, client_id TEXT, target TEXT, "
                "user_message TEXT, tool_name TEXT, consulted TEXT, round INTEGER, created_at TEXT)")
    for i, (msg, tool, consulted) in enumerate(rows):
        con.execute("INSERT INTO tool_calls (client_id, target, user_message, tool_name, consulted, round, created_at) "
                    "VALUES ('c', 'RASPUTIN', ?, ?, ?, 0, ?)", (msg, tool, consulted, f"2026-11-0{1 + i // 50}T10:{i % 50:02d}:00"))
    con.commit(); con.close()


def wait_for(cond, timeout=10):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        time.sleep(0.02)
    return cond()


def test_disabled_by_default():
    assert nr.NEURAL_ROUTER_ENABLED is False
    assert nr.start_shadow_router() is None
    assert nr.shadow_stats()["status"] == "disabled"
    nr.shadow_observe("anything", None)          # no-op, no error


def test_route_label_folds_destinations():
    d = ["RASPUTIN", "PLUTUS", "GEO"]
    assert nr.route_label(None, d) == "RASPUTIN"
    assert nr.route_label("plutus", d) == "PLUTUS"
    assert nr.route_label("VERDE", d) == "GEO" and nr.route_label("TAIGA", d) == "GEO"
    assert nr.route_label("ASCLEPIUS", d) == "RASPUTIN"          # not a router destination
    assert nr.route_label("ASCLEPIUS", d + ["ASCLEPIUS"]) == "ASCLEPIUS"


def test_logged_decisions_reads_first_responses(tmp_path):
    db = str(tmp_path / "w.db")
    make_db(db, [("buy stocks?", "consult_submind", "PLUTUS"), ("hello", "__none__", None),
                 ("weather in Lisbon", "consult_submind", "VERDE"), ("graph it", "create_graph", None)])
    items = nr.logged_decisions(db, 10, ["RASPUTIN", "PLUTUS", "GEO"])
    assert items == [("buy stocks?", "PLUTUS"), ("hello", "RASPUTIN"), ("weather in Lisbon", "GEO"), ("graph it", "RASPUTIN")]
    assert nr.logged_decisions(db, 2, ["RASPUTIN", "PLUTUS", "GEO"]) == items[-2:]


def test_replay_then_live_records_and_rewards(tmp_path):
    db = str(tmp_path / "w.db")
    make_db(db, [("m%d" % i, "consult_submind" if i % 2 else "__none__", "PLUTUS" if i % 2 else None) for i in range(6)])
    brain = StubBrain(pick=1)                    # always "RASPUTIN"
    r = nr.ShadowRouter(dests=["RASPUTIN", "PLUTUS", "GEO"], db_path=db, replay=6, column_factory=lambda d: brain)
    r.start()
    assert wait_for(lambda: r.status == "running")
    assert len(brain.trials) == 6 and brain.rewards.count(1.0) == 3 and brain.rewards.count(-0.5) == 3
    r.observe("how is my portfolio", "PLUTUS")
    r.observe("tell me a joke", None)
    assert wait_for(lambda: r.stats()["live_seen"] == 2)
    st = r.stats()
    assert st["agreement_last50"] == 0.5 and st["confusion"] == {"PLUTUS->RASPUTIN": 1, "RASPUTIN->RASPUTIN": 1}
    assert st["replayed"] == 6
    con = sqlite3.connect(db)
    rows = con.execute("SELECT source, label, brain, correct FROM brain_shadow ORDER BY id").fetchall()
    con.close()
    assert rows[-2:] == [("live", "PLUTUS", "RASPUTIN", 0), ("live", "RASPUTIN", "RASPUTIN", 1)] and len(rows) == 8
    r.stop()


def test_observe_never_blocks_and_drops_oldest_when_behind(tmp_path):
    db = str(tmp_path / "w.db")
    make_db(db, [])
    class Slow(StubBrain):
        def trial(self, cue):
            time.sleep(0.2); return super().trial(cue)
    r = nr.ShadowRouter(dests=["RASPUTIN", "PLUTUS"], db_path=db, replay=0, column_factory=lambda d: Slow(), max_queue=3)
    r.start(); assert wait_for(lambda: r.status == "running")
    t0 = time.time()
    for i in range(10):
        r.observe(f"msg {i}", None)
    assert time.time() - t0 < 0.05               # the chat turn is never held up
    assert r.dropped >= 5
    r.stop()


def test_brain_failure_is_contained(tmp_path):
    db = str(tmp_path / "w.db")
    make_db(db, [])
    def broken(d):
        raise RuntimeError("no simulator here")
    r = nr.ShadowRouter(dests=["RASPUTIN", "PLUTUS"], db_path=db, replay=0, column_factory=broken)
    r.start()
    assert wait_for(lambda: r.status == "error")
    r.observe("x", None)                          # ignored, no exception
    assert "no simulator" in r.stats()["error"]


def test_main_py_wiring():
    src = open(os.path.join(HERE, "..", "main.py")).read()
    assert "start_shadow_router()" in src and '@app.get("/api/neural/shadow")' in src
    assert "shadow_observe(_trigger, _consult)" in src and 'if target == "RASPUTIN":' in src
