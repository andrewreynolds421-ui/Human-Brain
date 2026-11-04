"""Activity statistics and plots for a simulated column."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .params import POPULATIONS

COLORS = {"E": "#1f4e79", "I": "#c0392b", "F": "#c0392b", "N": "#e08e0b"}


def _pops(res):
    return list(res.get("populations") or POPULATIONS)


def population_rates(res, t_start_ms=200.0):
    """Mean firing rate (spikes/s) per population, discarding the transient."""
    t, i = res["spike_t_ms"], res["spike_i"]
    keep = t >= t_start_ms
    dur_s = (res["t_sim_ms"] - t_start_ms) / 1000.0
    counts = np.bincount(res["pop_of"][i[keep]], minlength=len(_pops(res)))
    return counts / (res["n_per_pop"] * dur_s)


def irregularity(res, t_start_ms=200.0):
    """Mean coefficient of variation of inter-spike intervals per population.

    ~1 means Poisson-like (asynchronous irregular), <<1 regular firing.
    """
    t, i = res["spike_t_ms"], res["spike_i"]
    keep = t >= t_start_ms
    t, i = t[keep], i[keep]
    order = np.argsort(i, kind="stable")
    t, i = t[order], i[order]
    N_POP = len(_pops(res))
    cv = np.full(N_POP, np.nan)
    for p in range(N_POP):
        cvs = []
        sel = res["pop_of"][i] == p
        for n in np.unique(i[sel]):
            ts = np.sort(t[i == n])
            if ts.size >= 4:
                isi = np.diff(ts)
                cvs.append(isi.std() / isi.mean())
        if cvs:
            cv[p] = np.mean(cvs)
    return cv


def synchrony(res, t_start_ms=200.0, bin_ms=3.0):
    """Fano-factor of the population spike count histogram (1 ~ asynchronous)."""
    t, i = res["spike_t_ms"], res["spike_i"]
    keep = t >= t_start_ms
    edges = np.arange(t_start_ms, res["t_sim_ms"] + bin_ms, bin_ms)
    N_POP = len(_pops(res))
    out = np.zeros(N_POP)
    for p in range(N_POP):
        sel = res["pop_of"][i[keep]] == p
        h, _ = np.histogram(t[keep][sel], edges)
        out[p] = h.var() / h.mean() if h.mean() > 0 else np.nan
    return out


def report(res, reference=None):
    rates = population_rates(res)
    cv = irregularity(res)
    sy = synchrony(res)
    lines = [f"{'pop':5s} {'rate Hz':>8s} {'ref Hz':>7s} {'CV_ISI':>7s} {'sync':>6s}"]
    pops = _pops(res)
    for p in range(len(pops)):
        ref = f"{reference[p]:7.2f}" if reference is not None else "     --"
        lines.append(f"{pops[p]:5s} {rates[p]:8.2f} {ref} {cv[p]:7.2f} {sy[p]:6.2f}")
    return "\n".join(lines), rates, cv, sy


def plot_summary(res, col, reference=None, path="column_activity.png",
                 raster_window_ms=(500, 900), neurons_per_pop=60):
    rates = population_rates(res)
    t, i = res["spike_t_ms"], res["spike_i"]
    slices = col.pop_slices()
    POPS = col.pops

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw=dict(width_ratios=[2.2, 1]))

    # --- raster, a fixed number of neurons per population, layers top-down
    ax = axes[0]
    y0, yticks, ylabels = 0, [], []
    rng = np.random.default_rng(0)
    for p in POPS:
        sl = slices[p]
        ids = np.arange(sl.start, sl.stop)
        ids = rng.choice(ids, min(neurons_per_pop, ids.size), replace=False)
        ids.sort()
        remap = {nid: y0 + k for k, nid in enumerate(ids)}
        sel = np.isin(i, ids) & (t >= raster_window_ms[0]) & (t <= raster_window_ms[1])
        ys = np.array([remap[n] for n in i[sel]])
        ax.scatter(t[sel], ys, s=2, color=COLORS[p[-1]], rasterized=True)
        yticks.append(y0 + ids.size / 2)
        ylabels.append(p)
        y0 += ids.size + 8
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("time (ms)")
    ax.set_title(f"{col.name}: {col.N:,} neurons, {col.S:,} synapses")

    # --- rates vs reference
    ax = axes[1]
    x = np.arange(len(POPS))
    ax.barh(x, rates, color=[COLORS[p[-1]] for p in POPS], alpha=0.85,
            label="simulated")
    if reference is not None:
        ax.scatter(reference, x, marker="|", s=300, color="k", zorder=3,
                   label="published full-scale")
    ax.set_yticks(x)
    ax.set_yticklabels(POPS)
    ax.invert_yaxis()
    ax.set_xlabel("firing rate (spikes/s)")
    ax.set_title("population rates")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
