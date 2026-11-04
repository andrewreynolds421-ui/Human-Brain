#!/usr/bin/env python3
"""Fit the one calibrated number of the dendritic column: extra external
input onto L5 and L6 pyramidal cells, so they fire at awake target rates.

Everything else stays as measured. Runs at full connectivity (K = 1) with the
H01 external budget and in-vivo background, multiplicative updates
m <- m * (target / rate)^alpha, until both rates are within tolerance.
Writes results/deep_ext_calibration.json, which the dendritic preset loads.

    python scripts/calibrate_deep.py --target-l5 4 --target-l6 2
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from synth import PRESETS, synthesize                      # noqa: E402
from synth.simulate import simulate                        # noqa: E402
from synth.analyze import population_rates                 # noqa: E402
from rhythm_diagnostics import bursts                      # noqa: E402

BASE = dict(ext_gain=2.0, ext_budget="h01")
BG = 2.0


def run(m5, m6, k, t_ms, seed=1):
    p = PRESETS["human_h01_dendritic"](verbose=False, deep_ext_gain={"L5E": m5, "L6E": m6}, **BASE)
    col = synthesize(p, N_scaling=0.1, K_scaling=k, seed=seed)
    res = simulate(col, t_sim_ms=t_ms, seed=seed, verbose=False, bg_ratio=BG)
    r = population_rates(res, 300)
    return dict(zip(p.populations, map(float, r))), bursts(res)["fano"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-l5", type=float, default=4.0)
    ap.add_argument("--target-l6", type=float, default=2.0)
    ap.add_argument("--tol", type=float, default=0.35, help="relative tolerance")
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--t", type=float, default=1500.0)
    ap.add_argument("--start", type=float, nargs=2, default=[3.0, 3.0])
    ap.add_argument("--iters", type=int, default=8)
    a = ap.parse_args()
    m5, m6 = a.start
    hist = []
    for it in range(a.iters):
        r, fano = run(m5, m6, a.k, a.t)
        hist.append(dict(m5=m5, m6=m6, L5E=r["L5E"], L6E=r["L6E"], fano=fano, rates=r))
        print(json.dumps(hist[-1]), flush=True)
        ok5 = abs(r["L5E"] - a.target_l5) <= a.tol * a.target_l5
        ok6 = abs(r["L6E"] - a.target_l6) <= a.tol * a.target_l6
        if ok5 and ok6:
            break
        f = lambda m, rate, tgt: float(np.clip(m * ((tgt + 0.2) / (rate + 0.2)) ** 0.5, m / 2, m * 2))
        m5 = m5 if ok5 else f(m5, r["L5E"], a.target_l5)
        m6 = m6 if ok6 else f(m6, r["L6E"], a.target_l6)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "deep_ext_calibration.json")
    json.dump(dict(deep_ext_gain={"L5E": hist[-1]["m5"], "L6E": hist[-1]["m6"]},
                   targets={"L5E": a.target_l5, "L6E": a.target_l6}, setup=dict(**BASE, bg_ratio=BG, K=a.k),
                   history=hist), open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
