#!/usr/bin/env python3
"""Train the reward-modulated STDP testbed and plot learning curves.

Usage:
    python scripts/run_learning.py --seeds 10 --trials 800
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from learning import RSTDPConfig, train        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--trials", type=int, default=800)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    curves, finals, example = [], [], None
    for seed in range(args.seeds):
        net, log = train(RSTDPConfig(seed=seed), n_trials=args.trials, verbose=False)
        c = np.array([r["correct"] for r in log], float)
        curves.append(np.convolve(c, np.ones(50) / 50, mode="valid"))
        finals.append(c[-200:].mean())
        if example is None:
            example = log
        print(f"seed {seed}: accuracy first 100 = {c[:100].mean():.2f}, "
              f"last 200 = {c[-200:].mean():.2f}")
    curves = np.array(curves)
    print(f"\nfinal accuracy: mean {np.mean(finals):.2f}, "
          f"{sum(f >= 0.9 for f in finals)}/{args.seeds} seeds >= 0.90")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    x = np.arange(curves.shape[1]) + 50
    for cu in curves:
        ax.plot(x, cu, color="#1f4e79", alpha=0.25, lw=1)
    ax.plot(x, curves.mean(0), color="#1f4e79", lw=2.5, label=f"mean of {args.seeds} seeds")
    ax.axhline(0.5, color="k", ls=":", lw=1, label="chance")
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("trial")
    ax.set_ylabel("accuracy (50-trial moving average)")
    ax.set_title("reward-gated STDP learns the two-pattern task")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    wA = np.array([r["wA"] for r in example])
    wB = np.array([r["wB"] for r in example])
    ax.plot(wA[:, 0], color="#1f4e79", lw=2, label="pattern A -> action 0 (correct)")
    ax.plot(wA[:, 1], color="#1f4e79", lw=1, ls="--", label="pattern A -> action 1")
    ax.plot(wB[:, 1], color="#c0392b", lw=2, label="pattern B -> action 1 (correct)")
    ax.plot(wB[:, 0], color="#c0392b", lw=1, ls="--", label="pattern B -> action 0")
    ax.set_xlabel("trial")
    ax.set_ylabel("mean synaptic weight")
    ax.set_title("weights, seed 0")
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    path = os.path.join(args.out, "rstdp_learning.png")
    fig.savefig(path, dpi=130)
    print(f"figure: {path}")


if __name__ == "__main__":
    main()
