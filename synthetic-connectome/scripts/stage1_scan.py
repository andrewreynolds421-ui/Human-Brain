#!/usr/bin/env python3
"""Stage 1 test: does the fully measured column find an asynchronous state
with no per-population tuning?

Model: conductance-based synapses (measured PSPs converted at their own
holding potentials, -70 mV for EPSPs and -55 mV for IPSPs), per-contact
amplitudes (unitary / synapses-per-connection from H01), measured
short-term plasticity, measured membranes, spike-frequency adaptation
(generic adaptive-LIF constants).  The single remaining knob is the gain on
the external (non-column) drive.

Usage:  python scripts/stage1_scan.py [--k-scale 0.25] [--gains 0.3 0.5 0.8]
"""
import argparse, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synth import synthesize, PRESETS                          # noqa: E402
from synth.simulate import simulate                           # noqa: E402
from synth.analyze import population_rates, synchrony, irregularity, plot_summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-scale", type=float, default=0.25)
    ap.add_argument("--n-scale", type=float, default=0.1)
    ap.add_argument("--gains", type=float, nargs="+", default=[0.3, 0.5, 0.8])
    ap.add_argument("--adapt-b", type=float, default=60.0, help="adaptation increment for E cells (pA); I cells get 1/6")
    ap.add_argument("--t", type=float, default=1500.0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--preset", default="human_h01_coba", help="human_h01_coba (8 populations) or human_h01_12pop")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    best = None
    header = False
    for g in args.gains:
        if args.preset == "human_h01_coba":
            p = PRESETS[args.preset](verbose=False, ext_gain=g, adapt_b_exc_pA=args.adapt_b, adapt_b_inh_pA=args.adapt_b / 6)
        else:
            p = PRESETS[args.preset](verbose=False, ext_gain=g, adapt_b_exc_pA=args.adapt_b)
        if not header:
            print(f"{'ext gain':>8s} | " + " ".join(f"{q:>6s}" for q in p.populations) + " | Fano    CV   regime")
            header = True
        col = synthesize(p, N_scaling=args.n_scale, K_scaling=args.k_scale, seed=args.seed)
        res = simulate(col, t_sim_ms=args.t, seed=args.seed, verbose=False)
        r = population_rates(res, args.t / 3); sy = float(np.nanmean(synchrony(res, args.t / 3))); cv = float(np.nanmean(irregularity(res, args.t / 3)))
        regime = "silent" if r.mean() < 0.1 else "saturated" if r.max() > 150 else "bursting" if sy > 8 else "synchronous" if sy > 3 else "asynchronous"
        rows.append((g, r, sy, cv, regime))
        print(f"{g:8.2f} | " + " ".join(f"{x:6.1f}" for x in r) + f" | {sy:6.1f} {cv:5.2f}  {regime}")
        score = (regime == "asynchronous", -abs(np.log((r.mean() + 0.1) / 5.0)))
        if best is None or score > best[0]:
            best = (score, g, res, col)
    _, g, res, col = best
    png = plot_summary(res, col, None, path=os.path.join(args.out, f"stage1_{args.preset}_activity.png"))
    np.save(os.path.join(args.out, f"stage1_{args.preset}_scan.npy"),
            np.array([(g, *r, sy, cv) for g, r, sy, cv, _ in rows]))
    print(f"\nfigure for ext gain {g}: {png}")


if __name__ == "__main__":
    main()
