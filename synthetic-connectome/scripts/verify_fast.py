"""Check the fast simulation paths in synth/network.py against the original code.

1. shared_stp (per-neuron STP + iSTDP state) must give the SAME spike train,
   spike for spike, as per-synapse STP/iSTDP (weights equal to rounding).
2. merged_inputs (one vectorised Poisson update) must give the same firing
   rates within run-to-run noise (it draws different random numbers).

Each variant runs in its own process: Brian2 orders same-slot objects by name and
names keep counting within one process, which alone changes the random stream.

    python scripts/verify_fast.py            # ~3 min
"""
import json, subprocess, sys, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = r"""
import sys, json, time, numpy as np
sys.path.insert(0, ROOT)
import brian2 as b2
from brian2 import ms
from synth import synthesize, PRESETS
from synth.network import build_network, stp_slot_overflows
from synth.learner import ISTDP_TARGET
shared, merged, T = SHARED, MERGED, TMS
p = PRESETS["human_h01_dendritic"](verbose=False, ext_gain=2.0, bg_ratio=2.0)
col = synthesize(p, N_scaling=0.03, K_scaling=1.0, seed=3)
b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms
nw = build_network(col, seed=1, bg_ratio=2.0, shared_stp=shared, merged_inputs=merged,
                   istdp=dict(eta=0.0163, tau_ms=20.0, target_hz=ISTDP_TARGET))
nw["net"].run(100 * ms); t0 = time.time(); nw["net"].run(T * ms); wall = time.time() - t0
W = np.empty(col.S); W[np.concatenate(nw["syn_index"])] = np.concatenate([np.asarray(S.w) for S in nw["synapses"]])
i = np.asarray(nw["spikes"].i); t = np.asarray(nw["spikes"].t / ms)
np.savez(OUT, i=i, t=t, W=W, wall=wall, pop=col.pop_of[i], npop=np.bincount(col.pop_of, minlength=12),
         ovf=stp_slot_overflows(col, i, t) if shared else -1)
"""


def child(shared, merged, T, out):
    code = (CHILD.replace("ROOT", repr(ROOT)).replace("SHARED", str(shared)).replace("MERGED", str(merged))
            .replace("TMS", str(T)).replace("OUT", repr(out)))
    subprocess.run([sys.executable, "-c", code], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.load(out)


def rates(d, T):
    m = d["t"] >= 100
    return np.bincount(d["pop"][m], minlength=12) / d["npop"] / (T / 1000)


def main(T=1000.0):
    a = child(False, False, T, "/tmp/vf_orig.npz")
    b = child(True, False, T, "/tmp/vf_shared.npz")
    c = child(True, True, T, "/tmp/vf_fast.npz")
    same = a["i"].size == b["i"].size and np.array_equal(a["i"], b["i"]) and np.allclose(a["t"], b["t"], atol=1e-9)
    wdiff = float(np.max(np.abs(a["W"] - b["W"]) / np.maximum(np.abs(a["W"]), 1e-15)))
    ra, rc = rates(a, T), rates(c, T)
    res = dict(shared_stp_identical_spikes=bool(same), shared_stp_weight_max_rel_diff=wdiff,
               slot_overflows=int(b["ovf"]),
               rates_orig=ra.round(2).tolist(), rates_fast=rc.round(2).tolist(),
               wall_s=dict(orig=round(float(a["wall"]), 1), shared=round(float(b["wall"]), 1), fast=round(float(c["wall"]), 1)))
    print(json.dumps(res, indent=1))
    ok = same and wdiff < 1e-3 and np.all(np.abs(rc - ra) <= 0.15 * ra + 0.5)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
