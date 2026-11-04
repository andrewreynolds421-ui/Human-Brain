"""Several cortical areas joined by long-range (white-matter) connections.

Each area is one copy of the finished column (``human_h01_dendritic``,
stage 3d); areas differ only in what connects to them.  Long-range synapses
replace part of each column's external (Poisson) input, drive-matched as in
synth/sheet.py so the calibration holds:

* share      of a column's outside input, ~25% comes from other areas
             (macaque: ~80% of projecting neurons intrinsic, ~15% neighbouring
             areas, ~1% subcortical, Markov et al. 2011; with ~76% of synapses
             from outside an 800 um column, Stepanyants et al. 2009, the
             inter-areal part is ~19/76 ~ 0.25).  Of that, the part from the
             *modelled* areas is 0.25 x sum of their FLNe into the target; the
             rest stays as Poisson input (areas not modelled).
* weights    split between source areas by FLNe (Markov et al. 2014,
             results/interareal_macaque_assoc.json)
* sources    L2/3 and L5 pyramidal cells, 2:1 (version 1: no feedforward /
             feedback laminar distinction - SLN is measured only for visual
             targets)
* targets    every population, in proportion to its external input
* delays     local delay + human distance / v; human distance = macaque
             white-matter distance x 2.3 (cube root of the ~12.6x brain volume
             ratio), v = 3.5 m/s (as in Schmidt et al. 2018)

The horizontal (within-area) share is left as Poisson input here: each area
is a single column.
"""
from __future__ import annotations

import copy
import json
import os
import numpy as np

from .build import synthesize

DATA = os.path.join(os.path.dirname(__file__), "..", "results", "interareal_macaque_assoc.json")
INTERAREAL_SHARE = 0.25
HUMAN_DISTANCE_SCALE = 2.3
VELOCITY_M_S = 3.5
SOURCE_SPLIT = (("L23E", 2 / 3), ("L5E", 1 / 3))
COLUMN_RATES = {"L23E": 1.6, "L5E": 4.8}          # finished column (stage 5b), Hz


def load_interareal(path=DATA):
    return json.load(open(path))


def build_areas(params, areas, n_scale=0.02, seed=1):
    """Synthesize one column per area and concatenate them into one network."""
    cols = [synthesize(params, N_scaling=n_scale, K_scaling=1.0, seed=seed + 100 * k) for k in range(len(areas))]
    base = cols[0]
    c = copy.copy(base)
    off = np.concatenate([[0], np.cumsum([x.N for x in cols])[:-1]])
    c.pop_of = np.concatenate([x.pop_of for x in cols])
    pos = []
    for k, x in enumerate(cols):
        p = x.pos_um.copy(); p[:, 0] += k * 10 * params.footprint_um      # areas far apart in the layout
        pos.append(p)
    c.pos_um = np.vstack(pos)
    c.n_per_pop = np.sum([x.n_per_pop for x in cols], 0)
    cat = lambda f: np.concatenate([getattr(x, f) for x in cols])
    c.pre = np.concatenate([x.pre + o for x, o in zip(cols, off)]).astype(np.int32)
    c.post = np.concatenate([x.post + o for x, o in zip(cols, off)]).astype(np.int32)
    for f in ("weight_pA", "delay_ms", "stp_U", "stp_tau_rec_ms", "stp_tau_fac_ms", "syn_on_dend"):
        if getattr(base, f) is not None:
            setattr(c, f, cat(f))
    # neurons must be grouped by population for pop_slices(); keep a permutation
    order = np.argsort(c.pop_of, kind="stable")
    inv = np.empty_like(order); inv[order] = np.arange(order.size)
    c.pop_of = c.pop_of[order]; c.pos_um = c.pos_um[order]
    c.pre = inv[c.pre].astype(np.int32); c.post = inv[c.post].astype(np.int32)
    area_of = np.concatenate([np.full(x.N, k) for k, x in enumerate(cols)])[order]
    c.area_of = area_of
    c.areas = list(areas)
    c.name = f"{base.name}_areas{len(areas)}"
    return c


def add_interareal(col, data=None, share=INTERAREAL_SHARE, rates=COLUMN_RATES, seed=0, verbose=True,
                   calibrated_gain=None):
    data = data or load_interareal()
    rng = np.random.default_rng(seed + 23)
    c = copy.copy(col)
    pops, areas = col.pops, col.areas
    pop_of, area_of = col.pop_of, col.area_of
    pre_pop, post_pop = pop_of[col.pre], pop_of[col.post]
    w_ext = np.asarray(col.neuron["w_ext_pop"], float)
    K_ext = np.asarray(col.K_ext, float)
    if calibrated_gain is None:
        f = os.path.join(os.path.dirname(__file__), "..", "results", "deep_ext_calibration.json")
        calibrated_gain = json.load(open(f))["deep_ext_gain"] if os.path.exists(f) else {}
    K_conv = K_ext / np.array([float(calibrated_gain.get(q, 1.0)) for q in pops])
    fln, dist = data["fln_target_from_source"], data["distance_mm_macaque"]
    new = dict(pre=[], post=[], w=[], d=[], U=[], trec=[], tfac=[], dend=[])
    # per-neuron external reduction (depends on the target area)
    ext_scale = np.ones(col.N)
    report = {}
    for ta, tname in enumerate(areas):
        tot_fln = sum(fln[tname].get(s, 0.0) for s in areas if s != tname)
        report[tname] = dict(modelled_fln=round(tot_fln, 3), from_modelled_share=round(share * tot_fln, 4))
        ext_scale[area_of == ta] = 1 - share * tot_fln * (K_conv / K_ext)[pop_of[area_of == ta]]
        for t, tp in enumerate(pops):
            tgt = np.flatnonzero((pop_of == t) & (area_of == ta))
            if tgt.size == 0:
                continue
            for sa, sname in enumerate(areas):
                fl = fln[tname].get(sname, 0.0) if sname != tname else 0.0
                if fl <= 0:
                    continue
                drive = share * fl * K_conv[t] * w_ext[t] * col.bg_rate
                d_mm = dist[sname][tname] * HUMAN_DISTANCE_SCALE
                for sp, sh in SOURCE_SPLIT:
                    s = pops.index(sp)
                    sel = (pre_pop == s) & (post_pop == t)
                    w_eff = col.weight_pA[sel].astype(float)
                    if col.has_stp:
                        w_eff = w_eff * col.stp_U[sel]
                    k = drive * sh / (np.mean(w_eff) * rates[sp])
                    m = int(rng.poisson(k * tgt.size))
                    if m == 0:
                        continue
                    src_pool = np.flatnonzero((pop_of == s) & (area_of == sa))
                    idx = rng.choice(np.flatnonzero(sel), m)
                    new["pre"].append(rng.choice(src_pool, m)); new["post"].append(rng.choice(tgt, m))
                    new["w"].append(col.weight_pA[idx])
                    new["d"].append(col.delay_ms[idx] + d_mm / VELOCITY_M_S)       # mm / (m/s) = ms
                    if col.has_stp:
                        new["U"].append(col.stp_U[idx]); new["trec"].append(col.stp_tau_rec_ms[idx]); new["tfac"].append(col.stp_tau_fac_ms[idx])
                    if col.syn_on_dend is not None:
                        new["dend"].append(col.syn_on_dend[idx])
    cat = lambda k: np.concatenate(new[k]) if new[k] else np.empty(0)
    c.pre = np.concatenate([col.pre, cat("pre")]).astype(np.int32)
    c.post = np.concatenate([col.post, cat("post")]).astype(np.int32)
    c.weight_pA = np.concatenate([col.weight_pA, cat("w")]).astype(np.float32)
    c.delay_ms = np.concatenate([col.delay_ms, cat("d")]).astype(np.float32)
    if col.has_stp:
        c.stp_U = np.concatenate([col.stp_U, cat("U")]).astype(np.float32)
        c.stp_tau_rec_ms = np.concatenate([col.stp_tau_rec_ms, cat("trec")]).astype(np.float32)
        c.stp_tau_fac_ms = np.concatenate([col.stp_tau_fac_ms, cat("tfac")]).astype(np.float32)
    if col.syn_on_dend is not None:
        c.syn_on_dend = np.concatenate([col.syn_on_dend, cat("dend").astype(bool)])
    c.ext_scale_per_neuron = ext_scale
    c.name = col.name + "_interareal"
    c.interareal = dict(share=share, report=report, n_added=int(c.S - col.S))
    if verbose:
        print(f"inter-areal synapses added: {c.S - col.S:,}; modelled share of outside input per area:",
              {a: r["from_modelled_share"] for a, r in report.items()})
    return c
