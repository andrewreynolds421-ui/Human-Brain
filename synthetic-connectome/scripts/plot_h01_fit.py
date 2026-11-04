#!/usr/bin/env python3
"""Figure: what H01 measured vs the rodent reference, and the lateral profile."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synth import rodent_column                                              # noqa: E402
from synth.h01_fit import (H01_P_SLAB, H01_PAIRS_MEASURED, H01_LATERAL_PROFILE,   # noqa: E402
                           fit_lateral_sigma, human_h01_column, MIN_PAIRS_TRUSTED)
from synth.params import POPULATIONS                                         # noqa: E402

rodent = rodent_column()
human = human_h01_column()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))

def heat(ax, M, title, vmax, label):
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(8)); ax.set_xticklabels(POPULATIONS, rotation=60, fontsize=8)
    ax.set_yticks(range(8)); ax.set_yticklabels(POPULATIONS, fontsize=8)
    ax.set_xlabel("source"); ax.set_ylabel("target"); ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=label)
    return im

heat(axes[0], rodent.conn_probs, "rodent reference (Potjans & Diesmann 2014)", 0.4, "connection probability")
heat(axes[1], H01_P_SLAB * 1e4, "H01 measured within the 0.17 mm slab", 10, "probability x 1e-4")
trusted = H01_PAIRS_MEASURED >= MIN_PAIRS_TRUSTED
for i in range(8):
    for j in range(8):
        if trusted[i, j]:
            axes[1].text(j, i, str(H01_PAIRS_MEASURED[i, j]), ha="center", va="center", fontsize=6,
                         color="white" if H01_P_SLAB[i, j] * 1e4 > 5 else "black")
axes[1].text(0.0, -0.32, "numbers = connected pairs behind the estimate (shown where >= 10)",
             transform=axes[1].transAxes, fontsize=7)

ax = axes[2]
for t, color in (("E", "#1f4e79"), ("I", "#c0392b")):
    prof = H01_LATERAL_PROFILE[t]
    d = np.array(list(prof)); p = np.array(list(prof.values()))
    sig = fit_lateral_sigma(prof)
    ax.semilogy(d, p, "o", color=color, label=f"{t} sources, measured")
    dd = np.linspace(0, 500, 100)
    a = np.exp(np.polyfit(-d ** 2 / (2 * sig ** 2), np.log(p), 1)[1])
    ax.semilogy(dd, a * np.exp(-dd ** 2 / (2 * sig ** 2)), "-", color=color, alpha=0.7,
                label=f"Gaussian fit, sigma = {sig:.0f} um")
ax.set_xlabel("lateral soma distance (um)"); ax.set_ylabel("same-layer connection probability")
ax.set_title("fall-off with distance (H01)", fontsize=10)
ax.legend(fontsize=8, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
path = os.path.join("results", "h01_fit.png")
fig.savefig(path, dpi=130)
print("figure:", path)
