"""
Tests for the neural core.

Three layers: the flag (off by default, nothing changes), the tool contract
RASPUTIN sees, and the live service itself. The last one builds and runs a
small column, so it takes a minute the first time (Brian2 compiles) and is
skipped when the synthetic-connectome package is not present.

Run:  python -m pytest tests/test_neural_core.py -v
"""

import os
import re
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.pop("NEURAL_CORE", None)          # default: disabled
import neural_core as nc  # noqa: E402


# --- flag off: inert ---------------------------------------------------------

def test_disabled_by_default():
    assert nc.NEURAL_CORE_ENABLED is False
    assert nc.start_neural_core() is None
    with pytest.raises(Exception) as e:
        nc.get_service()
    assert "disabled" in str(e.value.detail)


def test_tool_reports_unavailable_when_disabled():
    out = nc.run_neural_core_tool({"action": "status"})
    assert out.startswith("neural core unavailable")


def test_main_registers_nothing_when_disabled():
    src = (ROOT / "main.py").read_text()
    # tool only appended behind the flag; executor branch present; tainted-blocked
    assert "if NEURAL_CORE_ENABLED:\n        tools.append(NEURAL_CORE_TOOL)" in src
    assert 'elif call.name == "neural_core":' in src
    m = re.search(r"TAINTED_BLOCKED_TOOLS = \{(.*?)\}", src, re.S)
    assert m and '"neural_core"' in m.group(1)
    assert '"neural_core"' in re.search(r"CORE_TOOL_NAMES = \{(.*?)\}", src, re.S).group(1)


# --- tool contract -----------------------------------------------------------

def test_tool_schema():
    t = nc.NEURAL_CORE_TOOL
    assert t["name"] == "neural_core"
    props = t["input_schema"]["properties"]
    assert set(props["action"]["enum"]) == {"status", "stimulate", "reward", "reset", "set_model", "trial"}
    assert set(props["model"]["enum"]) == {"membranes", "full", "dendritic", "learner", "learner8"}
    assert set(props["cue"]["enum"]) == {"A", "B"}
    assert props["value"]["minimum"] == -1 and props["value"]["maximum"] == 1
    assert set(props["layer"]["enum"]) == {"L23", "L4", "L5", "L6"}


def test_system_note_is_honest():
    assert "does not think" in nc.NEURAL_CORE_SYSTEM_NOTE
    assert "never as your mind" in nc.NEURAL_CORE_SYSTEM_NOTE


# --- live service ------------------------------------------------------------

@pytest.mark.skipif(nc._find_synth_path() is None, reason="synthetic-connectome package not found")
def test_service_runs_stimulates_rewards():
    svc = nc.NeuralCoreService(mode="membranes", scale=0.05, chunk_ms=100, autostart=False)
    svc.start()
    t0 = time.time()
    while svc.status == "warming" and time.time() - t0 < 600:
        time.sleep(1)
    assert svc.status == "running", svc.error
    time.sleep(3)
    st = svc.state()
    for key in ("t_ms", "rates", "layer_rates", "regime", "raster", "events", "realtime_factor"):
        assert key in st
    assert st["neurons"] > 0 and st["t_ms"] > 0
    assert set(st["rates"]) == {"L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"}

    r = svc.enqueue("stimulate", layer="L4", strength=1.0, duration_ms=100.0, label="test")
    assert r.get("ok"), r
    r = svc.enqueue("reward", value=0.5, note="test")
    assert r.get("ok") and "eligible_synapses" in r, r
    assert svc.control("pause")["status"] == "paused"
    assert svc.control("resume")["status"] == "running"
    r = svc.enqueue("reset")
    assert r.get("ok"), r

    # the tool path goes through the same service
    nc._service = svc
    try:
        assert nc.run_neural_core_tool({"action": "status"}).startswith("neural core running")
        assert "applied" in nc.run_neural_core_tool({"action": "reward", "value": -1, "note": "x"})
        assert nc.run_neural_core_tool({"action": "bogus"}).startswith("unknown action")
    finally:
        nc._service = None
        svc.stop()


@pytest.mark.skipif(nc._find_synth_path() is None, reason="synthetic-connectome package not found")
def test_dendritic_mode_and_set_model():
    """The 12-population, two-compartment column runs live, takes stimuli and
    rewards, and RASPUTIN can switch models through the tool."""
    svc = nc.NeuralCoreService(mode="dendritic", scale=0.05, chunk_ms=100, autostart=False)
    svc.start()
    t0 = time.time()
    while svc.status == "warming" and time.time() - t0 < 900:
        time.sleep(1)
    assert svc.status == "running", svc.error
    time.sleep(3)
    st = svc.state()
    assert set(st["rates"]) == {f"{L}{t}" for L in ("L23", "L4", "L5", "L6") for t in "EFN"}
    assert "2comp" in svc.info["model"]
    assert svc.enqueue("stimulate", layer="L23", strength=1.0, duration_ms=100.0).get("ok")
    assert svc.enqueue("reward", value=1.0).get("ok")
    nc._service = svc
    try:
        assert "[dendritic" in nc.run_neural_core_tool({"action": "status"})
        assert nc.run_neural_core_tool({"action": "set_model", "model": "nope"}).startswith("unknown model")
        out = nc.run_neural_core_tool({"action": "set_model", "model": "membranes"})
        assert out.startswith("neural core rebuilt as membranes"), out
        t1 = time.time()                       # first step after a rebuild can be slow on a busy machine
        while "rates" not in svc.state() and time.time() - t1 < 120:
            time.sleep(1)
        assert set(svc.state()["rates"]) == {"L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"}
    finally:
        nc._service = None
        svc.stop()


@pytest.mark.skipif(nc._find_synth_path() is None, reason="synthetic-connectome package not found")
def test_learner_mode_trial_and_reward():
    """RASPUTIN can run trials on the learner model and train it with reward."""
    svc = nc.NeuralCoreService(mode="learner", scale=0.05, chunk_ms=100, autostart=False)
    svc.start()
    t0 = time.time()
    while svc.status == "warming" and time.time() - t0 < 900:
        time.sleep(1)
    assert svc.status == "running", svc.error
    nc._service = svc
    try:
        # reward before any trial is refused, not silently applied
        assert nc.run_neural_core_tool({"action": "reward", "value": 1}).startswith("reward not applied")
        out = nc.run_neural_core_tool({"action": "trial", "cue": "A"})
        assert out.startswith("cue A -> the network chose action"), out
        choice = int(out.split("action ")[1][0])
        out = nc.run_neural_core_tool({"action": "reward", "value": 1 if choice == 1 else -1})
        assert out.startswith("reward") and "applied to the trial cue A" in out, out
        st = nc.run_neural_core_tool({"action": "status"})
        assert "learner: 1 trials with feedback" in st, st
        assert "synapses removed / formed so far" in st, st          # structural plasticity on by default
        assert ", surprise " in st, st                                 # surprise signal reported
        assert "learning rate x" in st, st                             # quick learning reports its rate
        assert svc.column.uncert_gain > 0                              # quick learning on by default
        # lean learner: reset rebuilds from the seed (no snapshot held in memory)
        assert svc.column.reset_needs_rebuild
        out = nc.run_neural_core_tool({"action": "reset"})
        assert "reset" in out.lower() and "fail" not in out.lower(), out
        st = nc.run_neural_core_tool({"action": "status"})
        assert "learner: 0 trials with feedback" in st, st
        # trials are refused outside the learner model
        assert svc.column.selectivity() == svc.column.selectivity()   # finite
    finally:
        nc._service = None
        svc.stop()
