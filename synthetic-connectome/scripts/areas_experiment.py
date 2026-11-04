#!/usr/bin/env python3
"""Eight association areas joined by measured long-range connections.

Rest: firing rates per area and synchrony.  Then 30 stimuli (200 ms drive to
the L2/3 + L5 pyramidal cells of one area, 600 ms rest); response of every
area's pyramidal cells (stimulus window minus 300 ms pre-stimulus), with the
measured connection strength (FLNe) from the stimulated area.

    python scripts/areas_experiment.py --stim 7A --interareal 1
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import brian2 as b2                                                   # noqa: E402
from brian2 import ms, Hz                                             # noqa: E402
from synth import PRESETS                                             # noqa: E402
from synth.areas import build_areas, add_interareal, load_interareal  # noqa: E402
from synth.network import build_network                               # noqa: E402

ON, OFF, SETTLE = 200.0, 600.0, 500.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", default="7A")
    ap.add_argument("--interareal", type=int, default=1)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--n-scale", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/areas_trials.jsonl")
    a = ap.parse_args()
    t0 = time.time()
    data = load_interareal()
    areas = data["areas"]
    p = PRESETS["human_h01_dendritic"](verbose=False, ext_gain=2.0)
    col = build_areas(p, areas, n_scale=a.n_scale, seed=a.seed)
    if a.interareal:
        col = add_interareal(col, data, seed=a.seed)
    b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms
    nw = build_network(col, seed=a.seed, bg_ratio=2.0)
    G, net = nw["G"], nw["net"]
    sa = areas.index(a.stim)
    tmask = np.array([q in ("L23E", "L5E") for q in col.pops])[col.pop_of]
    tgt = np.flatnonzero((col.area_of == sa) & tmask)
    rng = np.random.default_rng(a.seed + 11)
    pg = b2.PoissonGroup(2000, rates=0 * Hz)
    S = b2.Synapses(pg, G, model="w : siemens", on_pre="g_ed_post += w")
    ii = rng.integers(0, 2000, tgt.size * 1000); jj = np.repeat(tgt, 1000)
    S.connect(i=ii, j=jj); S.w = np.asarray(col.neuron["w_ext_pop"])[col.pop_of[jj]] * b2.nS
    net.add(pg, S)
    net.run(SETTLE * ms)
    onsets = []
    for k in range(a.trials):
        onsets.append(SETTLE + k * (ON + OFF))
        pg.rates = 30 * Hz; net.run(ON * ms); pg.rates = 0 * Hz; net.run(OFF * ms)
    t = np.asarray(nw["spikes"].t / ms); i = np.asarray(nw["spikes"].i)
    E = col.is_exc[col.pop_of]
    out = dict(stim=a.stim, interareal=a.interareal, seed=a.seed, neurons=col.N, synapses=col.S, areas={})
    edges = np.arange(-100, 400, 10)
    for k, name in enumerate(areas):
        cells = (col.area_of == k) & E; nk = cells.sum(); tk = t[cells[i]]
        r = np.array([((tk >= on + 10) & (tk < on + 210)).sum() / nk / 0.2 - ((tk >= on - 300) & (tk < on)).sum() / nk / 0.3
                      for on in onsets])
        rest = np.mean([((tk >= on - 300) & (tk < on)).sum() / nk / 0.3 for on in onsets])
        psth = sum(np.histogram(tk - on, edges)[0] for on in onsets) / nk / 0.01 / len(onsets)
        allc = (col.area_of == k)
        rest_pop = {q: round(float(((t[allc[i] & (col.pop_of[i] == pi)] % (ON + OFF)) >= 0).sum()), 0) for pi, q in enumerate(col.pops)}
        out["areas"][name] = dict(fln_from_stim=round(data["fln_target_from_source"][name].get(a.stim, 0.0), 4) if name != a.stim else None,
                                  distance_mm_human=round(data["distance_mm_macaque"][a.stim][name] * 2.3, 1) if name != a.stim else 0,
                                  rest_pyr_hz=round(float(rest), 2), response_hz=round(float(r.mean()), 3),
                                  sem=round(float(r.std(ddof=1) / np.sqrt(len(r))), 3),
                                  t=round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 1),
                                  psth_hz=np.round(psth - psth[edges[:-1] < 0].mean(), 2).tolist())
    out["psth_edges_ms"] = edges.tolist(); out["wall_s"] = round(time.time() - t0)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    print(f"{a.stim} interareal={a.interareal} neurons {col.N} synapses {col.S:,} wall {out['wall_s']}s")
    for name, v in out["areas"].items():
        print(f"  {name:>6s}  FLN {v['fln_from_stim']}  dist {v['distance_mm_human']:5.0f} mm  rest {v['rest_pyr_hz']:.2f} Hz  "
              f"response {v['response_hz']:+.3f} (t {v['t']:+.1f})")


if __name__ == "__main__":
    main()
