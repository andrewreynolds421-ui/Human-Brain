#!/usr/bin/env python3
"""Synthesize a column from a preset, simulate it and report activity.

    python scripts/run_column.py --preset rodent            # validation (~1 min)
    python scripts/run_column.py --preset human_h01
    python scripts/run_column.py --preset human_h01_synphys
    python scripts/run_column.py --preset human_h01_12pop --ext-gain 0.4

Prints the network summary and, per population, firing rate, irregularity
(CV of inter-spike intervals) and synchrony; for the rodent preset it also
compares against the published full-scale rates (Potjans & Diesmann 2014).
Saves a raster + rate figure to results/<name>_activity.png.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synth import synthesize, PRESETS                     # noqa: E402
from synth.simulate import simulate                      # noqa: E402
from synth.analyze import report, plot_summary           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="rodent", choices=sorted(PRESETS))
    ap.add_argument("--n-scale", type=float, default=0.1, help="fraction of neurons kept")
    ap.add_argument("--k-scale", type=float, default=0.1, help="fraction of inputs per neuron kept")
    ap.add_argument("--t", type=float, default=1000.0, help="simulated time (ms)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ext-gain", type=float, default=None,
                    help="external-drive gain (conductance presets human_h01_coba / human_h01_12pop)")
    ap.add_argument("--no-spatial", action="store_true", help="ignore lateral distance when wiring")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    kw = {}
    if args.preset != "rodent" and args.preset != "human":
        kw["verbose"] = True
    if args.ext_gain is not None:
        kw["ext_gain"] = args.ext_gain
    if args.preset in ("human_h01_coba",):
        kw.setdefault("adapt_b_exc_pA", 60.0); kw.setdefault("adapt_b_inh_pA", 10.0)
    params = PRESETS[args.preset](**kw)

    col = synthesize(params, N_scaling=args.n_scale, K_scaling=args.k_scale,
                     spatial=not args.no_spatial, seed=args.seed)
    print(col.describe())
    res = simulate(col, t_sim_ms=args.t, seed=args.seed)

    ref = params.full_mean_rates if args.preset == "rodent" else None
    text, rates, cv, sy = report(res, ref)
    print(text)
    if ref is not None:
        err = np.abs(rates - ref)
        rank_ok = (np.argsort(rates) == np.argsort(ref)).all()
        print(f"max |rate - published| = {err.max():.2f} spikes/s; rank order identical: {rank_ok}")
    os.makedirs(args.out, exist_ok=True)
    png = plot_summary(res, col, ref, path=os.path.join(args.out, f"{params.name}_activity.png"))
    print("saved", png)


if __name__ == "__main__":
    main()
