"""Rate scan used for the Stage 25 recalibration: python scripts/analysis/scan_rates.py '[["label", {preset kwargs}], ...]'"""
import sys, json, numpy as np
sys.path.insert(0, '/home/claude/synthetic-connectome')
import brian2 as b2
from brian2 import ms
from synth import synthesize, PRESETS
from synth.network import build_network
T = json.load(open('/home/claude/synthetic-connectome/results/rate_targets.json'))["populations"]
def rates(seed=3, t=2000.0, **kw):
    kw = dict(kw); eg = kw.pop("ext_gain", 2.0); p = PRESETS["human_h01_dendritic"](verbose=False, ext_gain=eg, bg_ratio=2.0, **kw)
    col = synthesize(p, N_scaling=0.03, K_scaling=1.0, seed=seed)
    b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms
    nw = build_network(col, seed=seed, bg_ratio=2.0, merged_inputs=True, shared_stp=True)
    nw["net"].run(300 * ms); n0 = nw["spikes"].num_spikes; nw["net"].run(t * ms)
    i = np.asarray(nw["spikes"].i)[n0:]
    r = np.bincount(col.pop_of[i], minlength=12) / np.bincount(col.pop_of, minlength=12) / (t / 1000)
    return col.pops, r
def show(label, pops, r):
    flags = []
    for k, q in enumerate(pops):
        lo, hi = T[q]["min"], T[q]["max"]
        flags.append(f"{q}={r[k]:.1f}" + ("" if lo <= r[k] <= hi else ("v" if r[k] < lo else "^")))
    bad = sum(1 for k, q in enumerate(pops) if not (T[q]["min"] <= r[k] <= T[q]["max"]))
    print(f"{label:48s} out={bad:2d}  " + " ".join(flags), flush=True)
configs = json.loads(sys.argv[1])
for label, kw in configs:
    pops, r = rates(**kw); show(label, pops, r)
