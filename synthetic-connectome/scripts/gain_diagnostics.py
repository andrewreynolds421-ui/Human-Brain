#!/usr/bin/env python3
"""Which inhibitory pathway cancels driven pyramidal cells?

Single finished column (dendritic preset, K = 1).  Repeated 200 ms drives
(1,000 fibres x 30 Hz per cell, dendritic synapses) to one pyramidal
population; response = rate in [on+10, on+210] minus the 300 ms before.
``--knock`` scales one class of local synapses (diagnostic only):
  EF / EN / FE / NE   E->fast-spiking, E->other, fast-spiking->E, other->E (all layers)
  EF_trans            E->fast-spiking between different layers only
  EF_local            E->fast-spiking within the same layer only

    python scripts/gain_diagnostics.py --target L23E --knock none EF FE
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import brian2 as b2                                         # noqa: E402
from brian2 import ms, Hz                                   # noqa: E402
from synth import PRESETS, synthesize                       # noqa: E402
from synth.network import build_network                     # noqa: E402
from synth.params import pop_type, pop_layer                # noqa: E402


def run(target, knock, factor, n_scale=0.03, trials=12, seed=1, deep=None, base_efdeep=None):
    kw = {} if deep is None else dict(deep_ext_gain={"L5E": deep[0], "L6E": deep[1]})
    p = PRESETS["human_h01_dendritic"](verbose=False, ext_gain=2.0, **kw)
    col = synthesize(p, N_scaling=n_scale, K_scaling=1.0, seed=seed)
    if base_efdeep is not None:
        tp0 = np.array([pop_type(q) for q in col.pops]); ly0 = np.array([pop_layer(q) for q in col.pops])
        m0 = (tp0[col.pop_of[col.pre]] == "E") & (tp0[col.pop_of[col.post]] == "F") & (ly0[col.pop_of[col.post]] != "L23")
        col.weight_pA = col.weight_pA.copy(); col.weight_pA[m0] *= base_efdeep
    if knock.startswith("ext"):
        # scale external (outside-column) input onto one interneuron type, all layers
        which = knock[3:]                      # "F", "N" or "FN"
        tpp = [pop_type(q) for q in col.pops]
        col.K_ext = np.array([k * (factor if tpp[j] in which else 1.0) for j, k in enumerate(col.K_ext)])
    elif knock != "none":
        tp = np.array([pop_type(q) for q in col.pops]); ly = np.array([pop_layer(q) for q in col.pops])
        pre, post = col.pop_of[col.pre], col.pop_of[col.post]
        src, dst = {"EF": ("E", "F"), "EN": ("E", "N"), "FE": ("F", "E"), "NE": ("N", "E"),
                    "EF_trans": ("E", "F"), "EF_local": ("E", "F"), "EF_deep": ("E", "F"), "FE_trans": ("F", "E")}[knock]
        m = (tp[pre] == src) & (tp[post] == dst)
        if knock == "EF_trans":
            m &= ly[pre] != ly[post]
        if knock == "EF_local":
            m &= ly[pre] == ly[post]
        if knock == "FE_trans":
            m &= ly[pre] != ly[post]
        if knock == "EF_deep":                  # targets outside L2/3 (where E->FS is not measured)
            m &= ly[post] != "L23"
        col.weight_pA = col.weight_pA.copy(); col.weight_pA[m] *= factor
    b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms
    nw = build_network(col, seed=seed, bg_ratio=2.0); G, net = nw["G"], nw["net"]
    tgt = np.flatnonzero(col.pop_of == col.pops.index(target))
    rng = np.random.default_rng(seed + 5)
    pg = b2.PoissonGroup(2000, rates=0 * Hz)
    S = b2.Synapses(pg, G, model="w : siemens", on_pre="g_ed_post += w")
    ii = rng.integers(0, 2000, tgt.size * 1000); jj = np.repeat(tgt, 1000)
    S.connect(i=ii, j=jj); S.w = np.asarray(col.neuron["w_ext_pop"])[col.pop_of[jj]] * b2.nS
    net.add(pg, S); net.run(400 * ms)
    onsets = []
    for k in range(trials):
        onsets.append(400 + k * 500.0)
        pg.rates = 30 * Hz; net.run(200 * ms); pg.rates = 0 * Hz; net.run(300 * ms)
    t = np.asarray(nw["spikes"].t / ms); i = np.asarray(nw["spikes"].i)
    out = {}
    for k, q in enumerate(col.pops):
        cells = col.pop_of == k; nk = cells.sum(); tk = t[cells[i]]
        r = [((tk >= on + 10) & (tk < on + 210)).sum() / nk / 0.2 - ((tk >= on - 300) & (tk < on)).sum() / nk / 0.3 for on in onsets]
        rest = np.mean([((tk >= on - 300) & (tk < on)).sum() / nk / 0.3 for on in onsets])
        out[q] = (round(float(rest), 1), round(float(np.mean(r)), 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="L23E")
    ap.add_argument("--knock", nargs="+", default=["none"])
    ap.add_argument("--factor", type=float, nargs="+", default=[0.5])
    ap.add_argument("--deep", type=float, nargs=2, default=None, help="L5E and L6E external multipliers")
    ap.add_argument("--base-efdeep", type=float, default=None, help="apply this deep E->FS factor first")
    a = ap.parse_args()
    for kn in a.knock:
        for f in a.factor:
            t0 = time.time()
            out = run(a.target, kn, f, deep=a.deep, base_efdeep=a.base_efdeep)
            print(json.dumps(dict(target=a.target, knock=kn, factor=f, deep=a.deep, rest_and_response=out, wall=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    main()
