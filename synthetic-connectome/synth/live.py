"""A column that keeps running: step it, stimulate it, reward it, read it.

``LiveColumn`` wraps a synthesized column in a persistent Brian2 network so
an outside process (a dashboard, an agent, a body) can interact with it
while it runs:

* ``step(ms)``                advance biological time in chunks
* ``stimulate(layer, ...)``   extra Poisson drive into a layer for a while
                              (a "thalamic" pulse: an input event arrived)
* ``reward(value)``           a dopamine-like scalar, -1..1, applied to the
                              recurrent excitatory synapses between neurons
                              that were co-active in the last eligibility
                              window (reward-gated Hebbian rule, see below)
* ``state()``                 per-layer rates, burst flag, synchrony, the
                              most recent spikes, realtime factor

Reward rule
-----------
A simplified three-factor rule, cheap enough to apply to a million synapses
in one vectorised update: a synapse is *eligible* if both its pre- and
postsynaptic neuron spiked within ``eligibility_ms`` before the reward.
Eligible E->E weights change by ``eta * value`` (relative), then every
neuron's total excitatory input weight is renormalised to its original sum
(synaptic scaling), and weights are clipped to [0, w_max].  Positive reward
strengthens whatever the column was just doing; negative weakens it.  This
is the same idea as ``learning/rstdp.py`` with the spike-timing kernel
replaced by a co-activity window.
"""
from __future__ import annotations

import time
from collections import deque
import numpy as np
import brian2 as b2
from brian2 import ms, mV, pA, pF, Hz

from .params import layers_of
from .build import SynthesizedColumn


class LiveColumn:
    def __init__(self, col: SynthesizedColumn, dt_ms=0.1, seed=0, target="cython",
                 rate_window_ms=200.0, burst_bin_ms=5.0, burst_factor=4.0,
                 eligibility_ms=500.0, eta=0.05, w_max_factor=3.0, verbose=False):
        self.col = col
        self.pops = col.pops
        self.layers = layers_of(self.pops)
        self.dt_ms = dt_ms
        self.rate_window_ms = rate_window_ms
        self.burst_bin_ms = burst_bin_ms
        self.burst_factor = burst_factor
        self.eligibility_ms = eligibility_ms
        self.eta = eta
        self.w_max_factor = w_max_factor
        self.t_ms = 0.0
        self.wall_total = 0.0
        self.bio_total = 0.0
        self.events: deque = deque(maxlen=200)       # (t_ms, kind, detail)
        self._recent: deque = deque(maxlen=400)      # (t_ms array, i array) per step
        self._stim = []                              # active stimuli
        self._build(seed, target, verbose)

    # ------------------------------------------------------------------ build
    def _build(self, seed, target, verbose):
        col = self.col
        b2.start_scope()
        b2.prefs.codegen.target = target
        b2.defaultclock.dt = self.dt_ms * ms
        from .network import build_network
        nw = build_network(col, seed=seed, with_stimulus=True)
        self.G, self.spikes, self.net = nw["G"], nw["spikes"], nw["net"]
        self._synapses = nw["synapses"]
        self._stim_groups = nw["stim_groups"]
        self.w_ext_pop = nw["w_ext_pop"]
        self.coba = nw["coba"]
        self._wunit = nw["wunit"]
        # map each synapse group back to its index range in the column arrays
        self._syn_index = nw["syn_index"]
        self._store("init")
        # reward bookkeeping: E->E synapses, original per-target weight sums
        self.ee_mask = (col.is_exc[col.pop_of[col.pre]] & col.is_exc[col.pop_of[col.post]])
        self.ee_idx = np.flatnonzero(self.ee_mask)
        self.w0 = np.asarray(col.weight_pA, float)
        self.w_max = self.w0[self.ee_idx].max() * self.w_max_factor
        self.ee_target_sum0 = np.bincount(col.post[self.ee_idx], weights=self.w0[self.ee_idx], minlength=col.N)
        self.n_rewards = 0
        self.cum_weight_change = 0.0
        self._warm = False

    # ------------------------------------------------------------------- step
    def step(self, chunk_ms=50.0):
        """Advance the column by chunk_ms of biological time."""
        t0 = time.perf_counter()
        # expire stimuli
        now = self.t_ms
        self._stim = [s for s in self._stim if s["until_ms"] > now]
        for layer, (pg, Sst, n) in self._stim_groups.items():
            active = [s for s in self._stim if s["layer"] == layer]
            rate = sum(s["rate_hz"] for s in active)
            pg.rates = rate * Hz
            Sst.w_stim = (max((s["w_pA"] for s in active), default=0.0)) * pA
        n_before = self.spikes.num_spikes
        self.net.run(chunk_ms * ms)
        self.t_ms += chunk_ms
        # .copy(): a slice would keep the full copy of the whole spike record alive
        t = np.asarray(self.spikes.t / ms)[n_before:].copy()
        i = np.asarray(self.spikes.i)[n_before:].copy()
        self._recent.append((t, i))
        if not getattr(self, "_in_trial", False):
            self._clear_spikes()            # recent spikes live in _recent; the monitor would grow forever
        wall = time.perf_counter() - t0
        self.wall_total += wall
        self.bio_total += chunk_ms
        self._warm = True
        return wall

    def _clear_spikes(self):
        """empty the spike monitor in place (Brian2 has no public clear)"""
        self.spikes.resize(0)
        self.spikes.variables["N"].set_value(0)

    # ----------------------------------------------------------- interaction
    def stimulate(self, layer="L4", strength=1.0, duration_ms=100.0, label=""):
        """Extra Poisson drive into a layer: strength 1.0 ~ a strong sensory event."""
        layer = layer if layer in self.layers else "L4"
        rate_hz = 200.0 * float(strength)
        w_pA = float(np.mean(self.w_ext_pop)) * 2.0
        self._stim.append(dict(layer=layer, rate_hz=rate_hz, w_pA=w_pA,
                               until_ms=self.t_ms + float(duration_ms), label=label))
        self.events.append((round(self.t_ms), "stimulus", f"{layer} x{strength:g} for {duration_ms:g} ms {label}".strip()))

    def reward(self, value=1.0, note=""):
        """Dopamine-like scalar in [-1, 1] applied to recently co-active E->E synapses."""
        value = float(np.clip(value, -1.0, 1.0))
        t_cut = self.t_ms - self.eligibility_ms
        last = np.full(self.col.N, -np.inf)
        for t, i in reversed(self._recent):
            if t.size and t.max() < t_cut:
                break
            if i.size:
                np.maximum.at(last, i, t)
        active = last >= t_cut
        pre, post = self.col.pre[self.ee_idx], self.col.post[self.ee_idx]
        elig = active[pre] & active[post]
        w = np.empty(self.col.N * 0 + self.col.S, float)
        for S_, idx in zip(self._synapses, self._syn_index):
            w[idx] = np.asarray(S_.w[:] / self._wunit, float)
        before = w[self.ee_idx].copy()
        w_ee = before.copy()
        w_ee[elig] *= (1.0 + self.eta * value)
        # synaptic scaling: keep each neuron's total E->E input at its original sum
        sums = np.bincount(post, weights=w_ee, minlength=self.col.N)
        scale = np.ones(self.col.N)
        nz = sums > 0
        scale[nz] = self.ee_target_sum0[nz] / sums[nz]
        w_ee *= scale[post]
        np.clip(w_ee, 0.0, self.w_max, out=w_ee)
        w[self.ee_idx] = w_ee
        for S_, idx in zip(self._synapses, self._syn_index):
            S_.w[:] = w[idx] * self._wunit
        change = float(np.abs(w_ee - before).sum() / max(before.sum(), 1e-9))
        self.cum_weight_change += change
        self.n_rewards += 1
        self.events.append((round(self.t_ms), "reward", f"{value:+.2f} on {int(elig.sum())} eligible synapses "
                                                        f"({active.sum()} active neurons) {note}".strip()))
        return dict(value=value, eligible_synapses=int(elig.sum()), active_neurons=int(active.sum()),
                    relative_weight_change=change)

    def _store(self, name):
        """snapshot for reset(): in RAM, or on disk when self._store_file is set (lean mode)"""
        f = getattr(self, "_store_file", None)
        self.net.store(name, filename=f) if f else self.net.store(name)

    def reset(self):
        f = getattr(self, "_store_file", None)
        self.net.restore("init", filename=f) if f else self.net.restore("init")
        self.t_ms = 0.0
        self._recent.clear()
        self._stim.clear()
        self.events.append((0, "reset", ""))

    # ------------------------------------------------------------------ state
    def _window(self, ms_back):
        t_cut = self.t_ms - ms_back
        ts, iss = [], []
        for t, i in reversed(self._recent):
            if t.size == 0:
                continue
            if t.max() < t_cut:
                break
            keep = t >= t_cut
            ts.append(t[keep]); iss.append(i[keep])
        if not ts:
            return np.empty(0), np.empty(0, int)
        return np.concatenate(ts[::-1]), np.concatenate(iss[::-1])

    def state(self, raster_ms=300.0, raster_max=1500):
        t, i = self._window(self.rate_window_ms)
        dur_s = self.rate_window_ms / 1000.0
        n_pop = len(self.pops)
        counts = np.bincount(self.col.pop_of[i], minlength=n_pop) if i.size else np.zeros(n_pop)
        rates = counts / (self.col.n_per_pop * dur_s)
        # burst detector: population count per bin vs. its mean over the window
        edges = np.arange(self.t_ms - self.rate_window_ms, self.t_ms + self.burst_bin_ms, self.burst_bin_ms)
        h, _ = np.histogram(t, edges) if t.size else (np.zeros(len(edges) - 1), None)
        mean_bin = max(h.mean(), 1e-9) if len(h) else 1e-9
        fano = float(h.var() / mean_bin) if len(h) and mean_bin > 0 else 0.0
        burst_now = bool(len(h) and h[-1] > self.burst_factor * mean_bin and h[-1] >= 5)
        rt, ri = self._window(raster_ms)
        if rt.size > raster_max:
            sel = np.sort(np.random.default_rng(0).choice(rt.size, raster_max, replace=False))
            rt, ri = rt[sel], ri[sel]
        layer_rates = {L: float(np.average(rates[idx], weights=np.maximum(self.col.n_per_pop[idx], 1)))
                       for L, idx in self.layers.items()}
        mean_rate = float(np.average(rates, weights=self.col.n_per_pop))
        return dict(
            t_ms=round(self.t_ms, 1),
            neurons=int(self.col.N), synapses=int(self.col.S), model=self.col.name,
            rates={p: round(float(r), 2) for p, r in zip(self.pops, rates)},
            layer_rates={k: round(v, 2) for k, v in layer_rates.items()},
            mean_rate=round(mean_rate, 2),
            synchrony=round(fano, 2),
            bursting=burst_now,
            regime=("bursting" if fano > 8 else "synchronous" if fano > 3 else "asynchronous") if mean_rate > 0.1 else "silent",
            active_stimuli=[dict(layer=s["layer"], label=s["label"], ms_left=round(s["until_ms"] - self.t_ms))
                            for s in self._stim],
            rewards=self.n_rewards, cumulative_weight_change=round(self.cum_weight_change, 4),
            realtime_factor=round(self.bio_total / 1000.0 / self.wall_total, 2) if self.wall_total else None,
            raster=dict(t_ms=np.round(rt - (self.t_ms - raster_ms), 1).tolist(), i=ri.tolist(),
                        pop=self.col.pop_of[ri].tolist() if ri.size else [], window_ms=raster_ms),
            events=[dict(t_ms=e[0], kind=e[1], detail=e[2]) for e in list(self.events)[-12:]],
        )
