#!/usr/bin/env python3
"""Two-compartment (dendritic) column: activity and synchrony vs external
drive and in-vivo background conductance.

    python scripts/dendrite_scan.py --gains 1.0 1.5 2.0 --bg 0 2
    python scripts/dendrite_scan.py --preset human_h01_12pop --gains 1.0     # one-compartment reference
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from synth import PRESETS, synthesize                         # noqa: E402
from synth.simulate import simulate                           # noqa: E402
from synth.analyze import population_rates, irregularity      # noqa: E402
from rhythm_diagnostics import bursts                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="human_h01_dendritic")
    ap.add_argument("--gains", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--bg", type=float, nargs="+", default=[0.0, 2.0], help="background conductance / leak")
    ap.add_argument("--cap-frac", type=float, default=None, help="dendritic share of membrane (default 0.7)")
    ap.add_argument("--t", type=float, default=2500.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    for bg in a.bg:
        for g in a.gains:
            kw = dict(verbose=False, ext_gain=g)
            if a.cap_frac is not None:
                kw["dend_cap_frac"] = a.cap_frac
            p = PRESETS[a.preset](**kw)
            col = synthesize(p, N_scaling=0.1, K_scaling=0.25, seed=a.seed)
            res = simulate(col, t_sim_ms=a.t, seed=a.seed, verbose=False, bg_ratio=bg or None)
            r = population_rates(res, 300)
            b = bursts(res)
            print(json.dumps(dict(preset=a.preset, gain=g, bg=bg, fano=round(b["fano"], 1),
                                  beat_ms=b["ac_peak_lag_ms"], cv=round(float(np.nanmean(irregularity(res, 300))), 2),
                                  rates={q: round(float(x), 1) for q, x in zip(p.populations, r)})), flush=True)


if __name__ == "__main__":
    main()
