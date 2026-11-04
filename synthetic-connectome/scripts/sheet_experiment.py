#!/usr/bin/env python3
"""Multi-column sheet: does activity in one column reach its neighbours?

Builds an n x n sheet of the finished dendritic column (stage 3c), optionally
with horizontal connections (synth/sheet.py), runs it, then drives layer 4 of
the centre column with a 200 ms thalamus-like volley (2,000 fibres at 30 Hz,
200 per cell, ordinary external-type synapses) and measures each
column's pyramidal-cell response (stimulus window minus baseline), by
distance from the stimulated column.

    python scripts/sheet_experiment.py --n 3 --frac 0 0.25 0.5
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import brian2 as b2                                                   # noqa: E402
from brian2 import ms, Hz, pA                                         # noqa: E402
from synth import PRESETS, synthesize                                 # noqa: E402
from synth.sheet import make_sheet, add_horizontal, column_of         # noqa: E402
from synth.network import build_network                               # noqa: E402
from rhythm_diagnostics import bursts                                 # noqa: E402


def run(n, frac, n_scale, seed=1, t_stim=(1200.0, 1400.0), t_end=2000.0, stim_rate=30.0, fibres=2000, per_cell=200,
        targets=("L4E", "L4F")):
    p = make_sheet(PRESETS["human_h01_dendritic"](verbose=False, ext_gain=2.0), n)
    col = synthesize(p, N_scaling=n_scale, K_scaling=1.0, seed=seed)
    if frac > 0:
        col = add_horizontal(col, p, frac=frac, verbose=False, seed=seed)
    b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms
    nw = build_network(col, seed=seed, bg_ratio=2.0)
    G, net = nw["G"], nw["net"]
    cid = column_of(col.pos_um, n)
    centre = (n // 2) * n + n // 2
    l4 = np.array([q in targets for q in col.pops])[col.pop_of]      # default: thalamic targets (L4 pyramidal + fast-spiking)
    tgt = np.flatnonzero((cid == centre) & l4)
    # thalamus-like volley: a pool of fibres, each L4 cell of the centre column
    # receives per_cell of them through ordinary external-type synapses
    # (dendrite of pyramidal cells, soma of interneurons)
    rng = np.random.default_rng(seed + 11)
    pg = b2.PoissonGroup(fibres, rates=0 * Hz)
    w_ext = np.asarray(col.neuron["w_ext_pop"], float)
    two = bool(col.neuron.get("two_compartment"))
    added = []
    for exc in (True, False):
        cells = tgt[col.is_exc[col.pop_of[tgt]] == exc]
        if cells.size == 0:
            continue
        var = ("g_ed" if two else "g_e") if exc else "g_e"
        S = b2.Synapses(pg, G, model="w : siemens", on_pre=f"{var}_post += w")
        ii = rng.integers(0, fibres, cells.size * per_cell); jj = np.repeat(cells, per_cell)
        S.connect(i=ii, j=jj)
        S.w = w_ext[col.pop_of[jj]] * b2.nS
        added.append(S)
    net.add(pg, *added)
    net.run(t_stim[0] * ms)
    pg.rates = stim_rate * Hz
    net.run((t_stim[1] - t_stim[0]) * ms)
    pg.rates = 0 * Hz
    net.run((t_end - t_stim[1]) * ms)
    t = np.asarray(nw["spikes"].t / ms); i = np.asarray(nw["spikes"].i)
    E = col.is_exc[col.pop_of]
    cx, cy = np.arange(n * n) % n, np.arange(n * n) // n
    dx = np.minimum(abs(cx - n // 2), n - abs(cx - n // 2)); dy = np.minimum(abs(cy - n // 2), n - abs(cy - n // 2))
    dist = np.sqrt(dx ** 2 + dy ** 2)
    pop_resp = {}
    for pi, q in enumerate(col.pops):
        cells = (cid == centre) & (col.pop_of == pi); nk = cells.sum(); sel = cells[i]
        b_ = ((t > 400) & (t < t_stim[0]) & sel).sum() / nk / ((t_stim[0] - 400) / 1000)
        s_ = ((t >= t_stim[0] + 10) & (t < t_stim[1] + 10) & sel).sum() / nk / ((t_stim[1] - t_stim[0]) / 1000)
        pop_resp[q] = round(s_ - b_, 1)
    out = []
    for k in range(n * n):
        cells = (cid == k) & E
        nk = cells.sum()
        sel = cells[i]
        base = ((t > 400) & (t < t_stim[0]) & sel).sum() / nk / ((t_stim[0] - 400) / 1000)
        stim = ((t >= t_stim[0] + 10) & (t < t_stim[1] + 10) & sel).sum() / nk / ((t_stim[1] - t_stim[0]) / 1000)
        out.append(dict(column=k, dist_mm=float(dist[k]), base_hz=round(base, 2), stim_hz=round(stim, 2),
                        delta_hz=round(stim - base, 2)))
    res = dict(spike_t_ms=t, spike_i=i, t_sim_ms=t_end)
    fano = bursts(dict(res, t_sim_ms=t_stim[0]), t0=400.0)["fano"]
    return col, out, fano, pop_resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--frac", type=float, nargs="+", default=[0.0, 0.25])
    ap.add_argument("--n-scale", type=float, default=0.015)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per-cell", type=int, default=1000, help="volley fibres per target cell")
    ap.add_argument("--targets", nargs="+", default=["L4E", "L4F"], help="populations driven in the centre column")
    a = ap.parse_args()
    for f in a.frac:
        t0 = time.time()
        col, out, fano, pop_resp = run(a.n, f, a.n_scale, a.seed, per_cell=a.per_cell, targets=tuple(a.targets))
        by = {}
        for o in out:
            by.setdefault(o["dist_mm"], []).append(o["delta_hz"])
        summary = {f"{d:.2f} mm": round(float(np.mean(v)), 2) for d, v in sorted(by.items())}
        print(json.dumps(dict(frac=f, neurons=col.N, synapses=col.S, baseline_fano=round(fano, 1),
                              pyramidal_response_by_distance=summary, centre_column_response_by_population=pop_resp, wall_s=round(time.time() - t0),
                              columns=out)), flush=True)


if __name__ == "__main__":
    main()
