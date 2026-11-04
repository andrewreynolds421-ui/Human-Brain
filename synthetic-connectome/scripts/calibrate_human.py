#!/usr/bin/env python3
"""Calibrate per-population inhibitory weights of the H01-fitted human column.

H01 provides synapse *counts*; synaptic *strengths* are unknown.  Starting
from the in-degree-balance rebalance in ``synth/h01_fit.py``, this script
adjusts one multiplier per target population on inhibitory weights until the
population firing rates land near target values, by simple multiplicative
fixed-point iteration (a crude stand-in for the mean-field calibration used
in large-scale cortical models).

Targets default to the rodent reference rates, the only layer-resolved
rates available; human cortex in vivo is thought to fire lower.

Usage:  python scripts/calibrate_human.py --iters 12 --t 800
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synth import synthesize, rodent_column                 # noqa: E402
from synth.h01_fit import human_h01_column                 # noqa: E402
from synth.simulate import simulate                        # noqa: E402
from synth.analyze import population_rates                 # noqa: E402
from synth.params import POPULATIONS                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--t", type=float, default=800.0)
    ap.add_argument("--eta", type=float, default=0.2)
    ap.add_argument("--clip", type=float, default=0.25)
    ap.add_argument("--n-scale", type=float, default=0.1)
    ap.add_argument("--k-scale", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--preset", default="human_h01", choices=["human_h01", "human_h01_synphys", "human_h01_membranes"])
    args = ap.parse_args()

    target = rodent_column().full_mean_rates
    from synth import PRESETS
    from synth.h01_fit import human_h01_column as _h
    params = PRESETS[args.preset]() if args.preset != "human_h01" else _h()
    if args.out is None:
        args.out = f"results/{params.name}_calibration.json"
    if params.g_per_target is None:          # measured PSP matrix: scale inhibitory columns directly
        base_psp = params.PSP_matrix_mean.copy()
    g = np.array(params.g_per_target if params.g_per_target is not None else np.ones(8), float)
    history = []
    for it in range(args.iters):
        if params.psp_matrix_override is not None:
            m = base_psp.copy(); m[:, 1::2] *= g[:, None]; params.psp_matrix_override = m
        else:
            params.g_per_target = g
        col = synthesize(params, N_scaling=args.n_scale, K_scaling=args.k_scale, seed=it)
        res = simulate(col, t_sim_ms=args.t, seed=it, verbose=False)
        rates = population_rates(res, t_start_ms=200.0)
        err = np.abs(rates - target).mean()
        history.append(dict(iter=it, g=g.round(3).tolist(), rates=rates.round(2).tolist(), mean_abs_err=float(err)))
        print(f"iter {it:2d}  mean|rate-target| = {err:5.2f}   rates = " +
              " ".join(f"{p}:{r:5.1f}" for p, r in zip(POPULATIONS, rates)))
        # more inhibition where too fast, less where too slow (log-space step)
        step = args.eta * np.log((rates + 0.2) / (target + 0.2))
        g = g * np.exp(np.clip(step, -args.clip, args.clip))
        g = np.clip(g, 0.2, 10.0)
    # select by log-ratio error so that a silent population counts as a large miss
    def score(h):
        r = np.array(h["rates"])
        return float(np.mean(np.abs(np.log((r + 0.2) / (target + 0.2)))) + 2.0 * np.sum(r < 0.2) + 5.0 * np.sum(r > 60))
    best = min(history, key=score)
    print(f"\nbest iteration {best['iter']}: mean |rate - target| = {best['mean_abs_err']:.2f}")
    print("inhibitory multipliers:", dict(zip(POPULATIONS, best["g"])))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(target_rates=target.tolist(), best=best, history=history), open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
