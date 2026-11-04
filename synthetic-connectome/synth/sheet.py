"""A sheet of cortical columns joined by horizontal connections.

``make_sheet(params, n)`` turns a 1 mm^2 column description into an n x n mm
sheet (n^2 columns): neuron counts scale with area, connection
probabilities are rescaled so every neuron keeps the same *local* in-degree
(local partners are still drawn with the H01 lateral profile, sigma ~120 um,
so local wiring stays local and runs continuously across column borders).

``add_horizontal(col, params, ...)`` then converts a fraction of each
population's external (Poisson) excitatory input into real horizontal
synapses from L2/3 and L5 pyramidal cells elsewhere in the sheet:

* reach      exp(-d / lambda), lambda = 1 mm (horizontal collaterals extend
             1-3 mm; Rockland 2022, Front Syst Neurosci); distances on a torus
             so no column sits on an edge
* sources    L2/3E and L5E pyramidal cells, 2:1 by synapse count (horizontal
             systems in L2/3 and L5 are aligned; L2/3 dominates)
* strength   each horizontal synapse copies the measured local synapse of the
             same population pair (amplitude, compartment, short-term
             plasticity); the *number* is set so the mean drive equals the
             external drive it replaces, at the single-column operating rates
             (so the calibrated column stays calibrated)
* delay      local delay + d / v, v = 0.3 m/s (unmyelinated horizontal axons,
             ~0.1-0.5 m/s)
* fraction   ``frac`` of the (uncalibrated) external input replaced; default
             0.7.  Derivation: of all neurons projecting to a patch of macaque
             cortex ~80% lie in the same area, ~15% in neighbouring areas, ~1%
             subcortical (Markov et al. 2011, Cereb Cortex); ~76% of excitatory
             synapses in an 800 um column come from outside it (Stepanyants et al.
             2009, PNAS).  Local ~24%, horizontal ~80-24 = 56%, so horizontal
             is ~56/76 ~ 0.7 of a column's outside input (range ~0.65-0.75).
"""
from __future__ import annotations

import copy
import numpy as np

from .params import pop_type, pop_layer
from .h01_fit import _indegree_from_probs, _probs_from_indegree

SINGLE_COLUMN_RATES = {"L23E": 1.6, "L5E": 4.8}     # finished column, stage 5b (Hz)


def make_sheet(params, n: int):
    p = copy.deepcopy(params)
    n0 = np.asarray(params.full_num_neurons, float)
    K = _indegree_from_probs(params.conn_probs, n0, n0)
    p.full_num_neurons = np.round(n0 * n * n).astype(int)
    p.conn_probs = _probs_from_indegree(K, p.full_num_neurons.astype(float), p.full_num_neurons.astype(float))
    p.footprint_um = params.footprint_um * n
    p.sheet_n = n
    p.name = f"{params.name}_sheet{n}x{n}"
    return p


def column_of(pos_um, n, col_um=1000.0):
    cx = np.clip((pos_um[:, 0] // col_um).astype(int), 0, n - 1)
    cy = np.clip((pos_um[:, 1] // col_um).astype(int), 0, n - 1)
    return cy * n + cx


def _torus_d(a, b, L):
    d = np.abs(a[:, None, :] - b[None, :, :])
    d = np.minimum(d, L - d)
    return np.sqrt((d ** 2).sum(-1))


DEFAULT_HORIZONTAL_FRAC = 0.7


def add_horizontal(col, params, frac=DEFAULT_HORIZONTAL_FRAC, lam_um=1000.0, v_m_s=0.3, source_split=(("L23E", 2 / 3), ("L5E", 1 / 3)),
                   rates=SINGLE_COLUMN_RATES, seed=0, verbose=True, calibrated_gain=None):
    """Return a new SynthesizedColumn with horizontal synapses added and the
    external input reduced by ``frac``.  Mutates nothing."""
    rng = np.random.default_rng(seed + 7)
    c = copy.copy(col)
    pops = col.pops
    L = params.footprint_um
    pop_of = col.pop_of
    pre_pop, post_pop = pop_of[col.pre], pop_of[col.post]
    w_ext = np.asarray(col.neuron["w_ext_pop"], float)            # nS per external spike
    K_ext = np.asarray(col.K_ext, float)
    # the calibrated deep-layer extra input stands for thalamic / apical input,
    # not horizontal connections: only the uncalibrated part is converted
    if calibrated_gain is None:
        import json, os
        f = os.path.join(os.path.dirname(__file__), "..", "results", "deep_ext_calibration.json")
        calibrated_gain = json.load(open(f))["deep_ext_gain"] if os.path.exists(f) else {}
    gain = np.array([float(calibrated_gain.get(q, 1.0)) for q in pops])
    K_conv = K_ext / gain
    new = dict(pre=[], post=[], w=[], d=[], U=[], trec=[], tfac=[], dend=[])
    report = {}
    for t, tp in enumerate(pops):
        tgt = np.flatnonzero(pop_of == t)
        if tgt.size == 0:
            continue
        drive = frac * K_conv[t] * w_ext[t] * col.bg_rate          # nS * Hz replaced per neuron
        report[tp] = {}
        for sp, share in source_split:
            s = pops.index(sp)
            sel = (pre_pop == s) & (post_pop == t)
            if not sel.any():
                continue
            w_eff = col.weight_pA[sel].astype(float)
            if col.has_stp:
                w_eff = w_eff * col.stp_U[sel]
            k = drive * share / (np.mean(w_eff) * rates[sp])        # horizontal synapses per target neuron
            m = int(round(k * tgt.size))
            if m == 0:
                continue
            src_ids = np.flatnonzero(pop_of == s)
            tg = rng.choice(tgt, m)
            order = np.argsort(tg, kind="stable"); tg = tg[order]
            counts = np.bincount(np.searchsorted(tgt, tg), minlength=tgt.size)
            srcs = np.empty(m, np.int64); dist = np.empty(m)
            a = 0
            for c0 in range(0, tgt.size, 128):
                c1 = min(c0 + 128, tgt.size)
                dd = _torus_d(col.pos_um[tgt[c0:c1], :2], col.pos_um[src_ids, :2], L)
                w = np.exp(-dd / lam_um)
                cdf = np.cumsum(w, 1); cdf /= cdf[:, -1:]
                for r in range(c1 - c0):
                    cnt = counts[c0 + r]
                    if cnt == 0:
                        continue
                    j = np.minimum(np.searchsorted(cdf[r], rng.random(cnt)), src_ids.size - 1)
                    srcs[a:a + cnt] = src_ids[j]; dist[a:a + cnt] = dd[r, j]; a += cnt
            idx = rng.choice(np.flatnonzero(sel), m)                 # template local synapses (same pair)
            new["pre"].append(srcs); new["post"].append(tg)
            new["w"].append(col.weight_pA[idx]); new["d"].append(col.delay_ms[idx] + dist / (v_m_s * 1000.0))
            if col.has_stp:
                new["U"].append(col.stp_U[idx]); new["trec"].append(col.stp_tau_rec_ms[idx]); new["tfac"].append(col.stp_tau_fac_ms[idx])
            if col.syn_on_dend is not None:
                new["dend"].append(col.syn_on_dend[idx])
            report[tp][sp] = dict(per_neuron=round(k, 1), mean_dist_um=round(float(dist.mean()), 0))
    cat = lambda k: np.concatenate(new[k]) if new[k] else None
    c.pre = np.concatenate([col.pre, cat("pre")]).astype(np.int32)
    c.post = np.concatenate([col.post, cat("post")]).astype(np.int32)
    c.weight_pA = np.concatenate([col.weight_pA, cat("w")]).astype(np.float32)
    c.delay_ms = np.concatenate([col.delay_ms, cat("d")]).astype(np.float32)
    if col.has_stp:
        c.stp_U = np.concatenate([col.stp_U, cat("U")]).astype(np.float32)
        c.stp_tau_rec_ms = np.concatenate([col.stp_tau_rec_ms, cat("trec")]).astype(np.float32)
        c.stp_tau_fac_ms = np.concatenate([col.stp_tau_fac_ms, cat("tfac")]).astype(np.float32)
    if col.syn_on_dend is not None:
        c.syn_on_dend = np.concatenate([col.syn_on_dend, cat("dend")])
    c.K_ext = K_ext - frac * K_conv
    c.DC_pA = np.asarray(col.DC_pA, float) * (c.K_ext / np.maximum(K_ext, 1e-9))
    c.name = col.name + f"_h{frac:g}"
    c.horizontal = dict(frac=frac, lam_um=lam_um, v_m_s=v_m_s, n_added=int(c.S - col.S), per_target=report)
    if verbose:
        print(f"horizontal synapses added: {c.S - col.S:,} ({(c.S - col.S) / c.N:.0f} per neuron); "
              f"external input reduced by {frac:.0%}")
    return c
