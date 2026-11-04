"""Structural plasticity: synapses that form and disappear.

Random formation + selective stabilisation, at constant synapse number:

* ``add_potential_synapses`` adds a pool of *potential* synapses (weight 0,
  absent) between the same cell populations as a set of existing synapses -
  in cortex many more axon-dendrite appositions exist than synapses (the
  "potential connectome").  Each copies a random existing synapse of the same
  populations for its properties (starting weight, delay, short-term
  plasticity, compartment).
* ``StructuralPlasticity.update`` (call after each weight update): synapses
  whose weight fell below ``prune_frac`` x their starting weight are removed
  (w = 0, absent); the same number of new synapses form at random absent
  sites with ``new_frac`` x their starting weight (new spines are small), so
  the total stays constant (adult spine density is stable while individual
  spines turn over).  New synapses keep their place only if learning
  strengthens them.

Time is compressed: here turnover happens over minutes of simulated time,
in cortex over days.
"""
from __future__ import annotations

import copy
import numpy as np


def add_potential_synapses(col, mask, factor=1.0, seed=0):
    """Return (new_col, potential, w_init): ``factor`` x mask.sum() absent
    synapses appended; ``potential`` marks them; ``w_init`` holds every
    synapse's starting weight (for potential ones, the template's)."""
    rng = np.random.default_rng(seed + 77)
    tmpl = np.flatnonzero(mask)
    m = int(round(factor * tmpl.size))
    pick = rng.choice(tmpl, m)
    groups = {}
    key_of = lambda i: (col.area_of[i], col.pop_of[i]) if hasattr(col, "area_of") else (0, col.pop_of[i])
    def pool(i):
        k = key_of(i)
        if k not in groups:
            groups[k] = np.flatnonzero((col.pop_of == k[1]) & ((col.area_of == k[0]) if hasattr(col, "area_of") else True))
        return groups[k]
    new_pre = np.array([rng.choice(pool(col.pre[t])) for t in pick], np.int32)
    new_post = np.array([rng.choice(pool(col.post[t])) for t in pick], np.int32)
    c = copy.copy(col)
    c.pre = np.concatenate([col.pre, new_pre]).astype(np.int32)
    c.post = np.concatenate([col.post, new_post]).astype(np.int32)
    w_init = np.concatenate([col.weight_pA, col.weight_pA[pick]]).astype(np.float32)
    c.weight_pA = np.concatenate([col.weight_pA, np.zeros(m, np.float32)]).astype(np.float32)
    c.delay_ms = np.concatenate([col.delay_ms, col.delay_ms[pick]]).astype(np.float32)
    if col.has_stp:
        c.stp_U = np.concatenate([col.stp_U, col.stp_U[pick]]).astype(np.float32)
        c.stp_tau_rec_ms = np.concatenate([col.stp_tau_rec_ms, col.stp_tau_rec_ms[pick]]).astype(np.float32)
        c.stp_tau_fac_ms = np.concatenate([col.stp_tau_fac_ms, col.stp_tau_fac_ms[pick]]).astype(np.float32)
    if col.syn_on_dend is not None:
        c.syn_on_dend = np.concatenate([col.syn_on_dend, col.syn_on_dend[pick]])
    potential = np.concatenate([np.zeros(col.S, bool), np.ones(m, bool)])
    full_mask = np.concatenate([mask, np.ones(m, bool)])
    return c, potential, w_init, full_mask


class StructuralPlasticity:
    def __init__(self, prune_frac=0.25, new_frac=0.5, seed=0):
        self.prune_frac, self.new_frac = prune_frac, new_frac
        self.rng = np.random.default_rng(seed + 99)
        self.formed = 0
        self.pruned = 0

    def update(self, w, w_init, present):
        """In place on arrays for one synapse group; returns (n_pruned, n_formed, formed_idx)."""
        weak = present & (w < self.prune_frac * w_init)
        n = int(weak.sum())
        w[weak] = 0.0
        present[weak] = False
        absent = np.flatnonzero(~present)
        absent = absent[~np.isin(absent, np.flatnonzero(weak))]      # a site just pruned does not reform at once
        k = min(n, absent.size)
        new = self.rng.choice(absent, k, replace=False) if k else np.empty(0, int)
        w[new] = self.new_frac * w_init[new]
        present[new] = True
        self.pruned += n; self.formed += k
        return n, k, new
