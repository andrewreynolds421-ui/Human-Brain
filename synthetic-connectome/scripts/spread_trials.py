#!/usr/bin/env python3
"""Trial-averaged spread test on a multi-column sheet.

One long run per network: after a 500 ms settle, ``--trials`` stimuli, each
a 200 ms drive (1,000 fibres x 30 Hz per target cell, ordinary excitatory
synapses) to the centre column's L2/3 and L5 pyramidal cells, followed by
600 ms of rest.  Each trial's response in each column = pyramidal rate in
[onset+10, onset+210] ms minus rate in [onset-300, onset].  Reports mean,
SEM and t per distance class, plus a PSTH of neighbour columns.

    python scripts/spread_trials.py --n 3 --frac 0 0.25 --seeds 1 2 --trials 30
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import brian2 as b2                                                   # noqa: E402
from brian2 import ms, Hz                                             # noqa: E402
from synth import PRESETS, synthesize                                 # noqa: E402
from synth.sheet import make_sheet, add_horizontal, column_of         # noqa: E402
from synth.network import build_network                               # noqa: E402

ON, OFF, SETTLE = 200.0, 600.0, 500.0


def run(n, frac, seed, trials, n_scale, targets=("L23E", "L5E"), per_cell=1000, fibres=2000, rate=30.0):
    p = make_sheet(PRESETS["human_h01_dendritic"](verbose=False, ext_gain=2.0), n)
    col = synthesize(p, N_scaling=n_scale, K_scaling=1.0, seed=seed)
    if frac > 0:
        col = add_horizontal(col, p, frac=frac, verbose=False, seed=seed)
    b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms
    nw = build_network(col, seed=seed, bg_ratio=2.0)
    G, net = nw["G"], nw["net"]
    cid = column_of(col.pos_um, n)
    centre = (n // 2) * n + n // 2
    tmask = np.array([q in targets for q in col.pops])[col.pop_of]
    tgt = np.flatnonzero((cid == centre) & tmask)
    rng = np.random.default_rng(seed + 11)
    pg = b2.PoissonGroup(fibres, rates=0 * Hz)
    w_ext = np.asarray(col.neuron["w_ext_pop"], float)
    S = b2.Synapses(pg, G, model="w : siemens", on_pre="g_ed_post += w")
    ii = rng.integers(0, fibres, tgt.size * per_cell); jj = np.repeat(tgt, per_cell)
    S.connect(i=ii, j=jj); S.w = w_ext[col.pop_of[jj]] * b2.nS
    net.add(pg, S)
    net.run(SETTLE * ms)
    onsets = []
    for k in range(trials):
        onsets.append(SETTLE + k * (ON + OFF))
        pg.rates = rate * Hz; net.run(ON * ms)
        pg.rates = 0 * Hz; net.run(OFF * ms)
    t = np.asarray(nw["spikes"].t / ms); i = np.asarray(nw["spikes"].i)
    E = col.is_exc[col.pop_of]
    cx, cy = np.arange(n * n) % n, np.arange(n * n) // n
    dx = np.minimum(abs(cx - n // 2), n - abs(cx - n // 2)); dy = np.minimum(abs(cy - n // 2), n - abs(cy - n // 2))
    dist = np.round(np.sqrt(dx ** 2 + dy ** 2), 2)
    resp = np.zeros((trials, n * n))
    psth_edges = np.arange(-100, 400, 10)
    psth = {d: np.zeros(psth_edges.size - 1) for d in np.unique(dist)}
    for k in range(n * n):
        cells = (cid == k) & E
        nk = cells.sum(); tk = t[cells[i]]
        for j, on in enumerate(onsets):
            post = ((tk >= on + 10) & (tk < on + 210)).sum() / nk / 0.2
            pre = ((tk >= on - 300) & (tk < on)).sum() / nk / 0.3
            resp[j, k] = post - pre
            psth[dist[k]] += np.histogram(tk - on, psth_edges)[0] / nk / 0.01 / trials / (dist == dist[k]).sum()
    summary = {}
    for d in np.unique(dist):
        r = resp[:, dist == d].mean(1)          # per trial, averaged over the columns at this distance
        sem = r.std(ddof=1) / np.sqrt(trials)
        summary[f"{d:.2f}"] = dict(mean=round(float(r.mean()), 3), sem=round(float(sem), 3),
                                   t=round(float(r.mean() / sem), 1) if sem > 0 else None)
    base = np.mean([((t[E[i]] >= on - 300) & (t[E[i]] < on)).sum() for on in onsets]) / E.sum() / 0.3
    # resting state (300 ms before each stimulus): every population's rate, and
    # synchrony of the whole sheet (Fano factor of 2 ms population counts / columns)
    pre = np.zeros(t.size, bool)
    for on in onsets:
        pre |= (t >= on - 300) & (t < on)
    pre_rates = {q: round(float(np.isin(i[pre], np.flatnonzero(col.pop_of == k)).sum() / (col.pop_of == k).sum() / (0.3 * trials)), 2)
                 for k, q in enumerate(col.pops)}
    fanos = []
    for on in onsets:
        h, _ = np.histogram(t[(t >= on - 300) & (t < on)], np.arange(on - 300, on + 2, 2))
        if h.mean() > 0:
            fanos.append(h.var() / h.mean())
    rest_fano_per_column = round(float(np.mean(fanos)) / (n * n), 2) if fanos else None
    return dict(n=n, frac=frac, seed=seed, trials=trials, neurons=col.N, synapses=col.S,
                baseline_pyr_hz=round(float(base), 2), rest_rates_hz=pre_rates, rest_fano_per_column=rest_fano_per_column,
                response_by_distance_mm=summary,
                psth_edges_ms=psth_edges.tolist(), psth_hz={f"{d:.2f}": np.round(v, 2).tolist() for d, v in psth.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--frac", type=float, nargs="+", default=[0.0, 0.25])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--n-scale", type=float, default=0.015)
    ap.add_argument("--out", default="results/spread_trials.jsonl")
    a = ap.parse_args()
    for seed in a.seeds:
        for f in a.frac:
            t0 = time.time()
            r = run(a.n, f, seed, a.trials, a.n_scale)
            import gc; gc.collect()
            r["wall_s"] = round(time.time() - t0)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(json.dumps({k: v for k, v in r.items() if not k.startswith("psth")}), flush=True)


if __name__ == "__main__":
    main()
