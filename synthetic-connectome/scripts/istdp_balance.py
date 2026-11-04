#!/usr/bin/env python3
"""Inhibitory plasticity (Vogels et al. 2011) sets the column's E/I balance.

1. Run the finished column with plastic inhibitory -> pyramidal synapses
   (target rate per pyramidal layer) for --learn seconds; track rates.
2. Freeze the learned weights and repeat the response test of
   scripts/gain_diagnostics.py (drive L2/3 or L5 pyramids), compared with
   the original weights.

    python scripts/istdp_balance.py --learn 30 --eta 0.01
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import brian2 as b2                                         # noqa: E402
from brian2 import ms, Hz, nS                               # noqa: E402
from synth import PRESETS, synthesize                       # noqa: E402
from synth.network import build_network                     # noqa: E402

TARGET = {"L23E": 2.0, "L4E": 2.0, "L5E": 4.0, "L6E": 2.0}


def response_test(col, target, trials=12, seed=1):
    b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms
    nw = build_network(col, seed=seed, bg_ratio=2.0); G, net = nw["G"], nw["net"]
    tgt = np.flatnonzero(col.pop_of == col.pops.index(target))
    rng = np.random.default_rng(seed + 5)
    pg = b2.PoissonGroup(2000, rates=0 * Hz)
    S = b2.Synapses(pg, G, model="w : siemens", on_pre="g_ed_post += w")
    ii = rng.integers(0, 2000, tgt.size * 1000); jj = np.repeat(tgt, 1000)
    S.connect(i=ii, j=jj); S.w = np.asarray(col.neuron["w_ext_pop"])[col.pop_of[jj]] * nS
    net.add(pg, S); net.run(400 * ms)
    onsets = []
    for k in range(trials):
        onsets.append(400 + k * 500.0)
        pg.rates = 30 * Hz; net.run(200 * ms); pg.rates = 0 * Hz; net.run(300 * ms)
    t = np.asarray(nw["spikes"].t / ms); i = np.asarray(nw["spikes"].i)
    out = {}
    for k, q in enumerate(col.pops):
        cells = col.pop_of == k; nk = cells.sum(); tk = t[cells[i]]
        r = np.mean([((tk >= on + 10) & (tk < on + 210)).sum() / nk / 0.2 - ((tk >= on - 300) & (tk < on)).sum() / nk / 0.3 for on in onsets])
        rest = np.mean([((tk >= on - 300) & (tk < on)).sum() / nk / 0.3 for on in onsets])
        out[q] = (round(float(rest), 1), round(float(r), 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", type=float, default=30.0, help="seconds of plasticity")
    ap.add_argument("--eta", type=float, default=0.01, help="learning rate (fraction of mean weight per pairing)")
    ap.add_argument("--n-scale", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    p = PRESETS["human_h01_dendritic"](verbose=False, ext_gain=2.0)
    col = synthesize(p, N_scaling=a.n_scale, K_scaling=1.0, seed=a.seed)
    # --- learning phase
    t0 = time.time()
    b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms
    nw = build_network(col, seed=a.seed, bg_ratio=2.0, istdp=dict(eta=a.eta, tau_ms=20.0, target_hz=TARGET))
    net, spk = nw["net"], nw["spikes"]
    hist = []
    step = 2.0
    for k in range(int(a.learn / step)):
        n0 = spk.num_spikes
        net.run(step * 1000 * ms)
        i = np.asarray(spk.i[n0:])
        rates = {q: round(float(np.isin(i, np.flatnonzero(col.pop_of == j)).sum() / (col.pop_of == j).sum() / step), 2)
                 for j, q in enumerate(col.pops)}
        hist.append(rates)
        print(f"t={step * (k + 1):5.0f}s  " + " ".join(f"{q}:{rates[q]:.1f}" for q in col.pops), flush=True)
    # copy learned weights back
    learned = col.weight_pA.copy()
    for S_, idx in zip(nw["synapses"], nw["syn_index"]):
        learned[idx] = np.asarray(S_.w / nS)
    changed = learned != col.weight_pA
    ratio = learned[changed].sum() / col.weight_pA[changed].sum()
    print(f"learning wall {time.time() - t0:.0f}s; plastic synapses {changed.sum():,}; total inhibitory weight onto pyramids x{ratio:.2f}")
    import copy
    col2 = copy.copy(col); col2.weight_pA = learned.astype(np.float32)
    res = {}
    for target in ("L23E", "L5E"):
        res[target] = dict(original=response_test(col, target, seed=a.seed), learned=response_test(col2, target, seed=a.seed))
        for lab in ("original", "learned"):
            r = res[target][lab]
            print(f"drive {target} {lab:8s} | " + "  ".join(f"{q} {v[0]}{v[1]:+.1f}" for q, v in r.items()), flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(dict(target_hz=TARGET, eta=a.eta, learn_s=a.learn, history=hist, weight_ratio=float(ratio), responses=res),
              open("results/istdp_balance.json", "w"), indent=1)
    np.save("results/istdp_learned_weights.npy", learned)


if __name__ == "__main__":
    main()
