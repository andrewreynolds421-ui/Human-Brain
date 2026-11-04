"""Human cortical column preset fitted to the H01 dataset (Phase 1).

Every number in this file was measured from the H01 release (Shapson-Coe et
al., Science 2024) on 2026-10-30:

* cell types and cortical layers of 45,994 soma-bearing cells, from the
  authors' analysis tables (github.com/ashapsoncoe/h01:
  ``agglo_20200916c3_cell_data.json``, ``goog14r0s5c3_e_i_ratios_feb_2023.csv``,
  ``cortical_bounds_circles.json``);
* the full c3 synapse export (166,216,068 synapses,
  ``gs://h01-release/data/20210729/c3/synapses/exported/``) loaded into
  BigQuery and joined against a stratified sample of 3,111 labelled neurons
  (300 per layer x E/I where available).

What the volume can and cannot tell us
--------------------------------------
H01 is a slab ~3.9 x 2.4 mm in the plane of the cortical layers but only
0.17 mm thick.  Axons leave it quickly, so

* connection probabilities measured *within* the slab are far below the
  true column values (10^-4 .. 10^-3 instead of 10^-2 .. 10^-1), and
* pathways that travel a long way across layers (L2/3 -> L5, L6 -> L4) are
  barely sampled.

We therefore use H01 for (a) counts, densities, layer thicknesses and the
E/I split, which are not affected by truncation, (b) the *relative*
strength of the well-sampled local pathways, and (c) the lateral fall-off
of connection probability.  The absolute in-degree scale and the poorly
sampled pathways are carried over from the rodent reference model
(Potjans & Diesmann 2014); those entries are flagged in
``H01_PAIRS_MEASURED`` so they can be revisited when more human data exist.
"""
from __future__ import annotations

import json
import os
import numpy as np

from .params import ColumnParams, POPULATIONS, N_POP, CONN_PROBS as RODENT_CONN_PROBS
from .params import rodent_column

# ---------------------------------------------------------------------------
# 1. Geometry and counts (from the cell tables; see h01_layer_stats.json)
# ---------------------------------------------------------------------------
# Layer thickness: radial extent of soma positions between successive layer
# medians, measured from the white-matter boundary circle.  L1 is not
# measurable this way (no pia reference) and is set to a literature value.
H01_LAYER_THICKNESS_UM = {"L1": 250, "L23": 747, "L4": 377, "L5": 378, "L6": 453}

# Neuron density per layer (neurons / mm^3): cells with a classified soma,
# divided by the layer's occupied area (50 um grid) x the 0.168 mm slab.
# E = pyramidal + spiny stellate + atypical spiny; I = interneuron.
H01_DENSITY_PER_MM3 = {
    #        E       I
    "L2":  (15730,  8284),
    "L3":  ( 6468,  3626),
    "L4":  (20624,  5541),
    "L5":  (10702,  3392),
    "L6":  ( 9249,  1316),
}
H01_SUBLAYER_THICKNESS_UM = {"L2": 359, "L3": 388}


def h01_full_num_neurons(footprint_um=1000.0):
    """Neurons per population in a column with the given footprint."""
    a = (footprint_um / 1000.0) ** 2  # mm^2
    def n(dens, thick_um):
        return dens * (thick_um / 1000.0) * a
    L23E = n(H01_DENSITY_PER_MM3["L2"][0], 359) + n(H01_DENSITY_PER_MM3["L3"][0], 388)
    L23I = n(H01_DENSITY_PER_MM3["L2"][1], 359) + n(H01_DENSITY_PER_MM3["L3"][1], 388)
    L4E, L4I = n(H01_DENSITY_PER_MM3["L4"][0], 377), n(H01_DENSITY_PER_MM3["L4"][1], 377)
    L5E, L5I = n(H01_DENSITY_PER_MM3["L5"][0], 378), n(H01_DENSITY_PER_MM3["L5"][1], 378)
    L6E, L6I = n(H01_DENSITY_PER_MM3["L6"][0], 453), n(H01_DENSITY_PER_MM3["L6"][1], 453)
    return np.round([L23E, L23I, L4E, L4I, L5E, L5I, L6E, L6I]).astype(int)


# ---------------------------------------------------------------------------
# 2. Within-slab connection probabilities (rows = target, cols = source),
#    type-consistent synapses only (E sources counted through asymmetric
#    synapses, I sources through symmetric ones), 3,111-neuron sample.
# ---------------------------------------------------------------------------
H01_P_SLAB = np.array([
    # L23E     L23I     L4E      L4I      L5E      L5I      L6E      L6I     <- source
    [3.23e-4, 4.11e-4, 7.8e-5,  1.0e-4,  1.7e-5,  6.0e-6,  0.0,     0.0    ],  # -> L23E
    [4.78e-4, 2.87e-4, 8.9e-5,  4.4e-5,  6.0e-6,  0.0,     0.0,     0.0    ],  # -> L23I
    [7.2e-5,  6.1e-5,  2.45e-4, 5.0e-4,  4.4e-5,  1.78e-4, 0.0,     0.0    ],  # -> L4E
    [1.22e-4, 2.8e-5,  9.56e-4, 3.79e-4, 1.22e-4, 8.9e-5,  0.0,     0.0    ],  # -> L4I
    [2.2e-5,  6.0e-6,  6.7e-5,  1.33e-4, 2.79e-4, 8.0e-4,  1.11e-4, 1.2e-4 ],  # -> L5E
    [1.7e-5,  0.0,     2.78e-4, 3.3e-5,  6.0e-4,  4.91e-4, 3.3e-5,  6.0e-5 ],  # -> L5I
    [0.0,     0.0,     2.2e-5,  0.0,     2.22e-4, 2.0e-4,  7.02e-4, 6.91e-4],  # -> L6E
    [0.0,     0.0,     0.0,     0.0,     1.5e-4,  3.0e-5,  3.6e-4,  1.31e-3],  # -> L6I
])
# number of connected neuron pairs behind each estimate (statistical weight)
H01_PAIRS_MEASURED = np.array([
    [116, 148, 14, 18,  3,  1,  0,  0],
    [172, 103, 16,  8,  1,  0,  0,  0],
    [ 13,  11, 22, 45,  4, 16,  0,  0],
    [ 22,   5, 86, 34, 11,  8,  0,  0],
    [  4,   1,  6, 12, 25, 72, 10,  4],
    [  3,   0, 25,  3, 54, 44,  3,  2],
    [  0,   0,  2,  0, 20, 18, 63, 23],
    [  0,   0,  0,  0,  5,  1, 12, 16],
])
MIN_PAIRS_TRUSTED = 10

# Synapses per connected pair (multapses) in H01, pooled over well-sampled
# layer pairs; a patch-clamp unitary PSP is the sum over all contacts of one
# connection, so the per-synapse amplitude in the model is unitary / this.
H01_SYNAPSES_PER_CONNECTION = {("E", "E"): 1.5, ("I", "E"): 1.7, ("E", "I"): 1.8, ("I", "I"): 1.7}  # key = (target, source)

# Lateral fall-off of same-layer connection probability vs soma distance
# (um bins, all same-layer pairs).  Gaussian sigma fitted below.
H01_LATERAL_PROFILE = {
    "E": {50: 5.941e-3, 150: 2.200e-3, 250: 5.97e-4, 350: 1.90e-4, 450: 6.6e-5},
    "I": {50: 9.017e-3, 150: 1.689e-3, 250: 4.29e-4, 350: 6.7e-5,  450: 3.0e-5},
}


def fit_lateral_sigma(profile):
    d = np.array(list(profile.keys()), float)
    p = np.array(list(profile.values()), float)
    # ln p = a - d^2 / (2 sigma^2), weighted least squares on the well-sampled bins
    w = np.sqrt(p)
    A = np.vstack([np.ones_like(d), -d ** 2]).T * w[:, None]
    coef, *_ = np.linalg.lstsq(A, np.log(p) * w, rcond=None)
    return float(np.sqrt(1.0 / (2.0 * coef[1])))


# ---------------------------------------------------------------------------
# 3. Blend: H01 relative structure, rodent in-degree scale
# ---------------------------------------------------------------------------

def _indegree_from_probs(conn_probs, n_target, n_source):
    prod = np.outer(n_target, n_source).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.log(1.0 - conn_probs) / np.log((prod - 1.0) / prod)
    k[~np.isfinite(k)] = 0.0
    return k / n_target[:, None]


def _probs_from_indegree(K, n_target, n_source):
    prod = np.outer(n_target, n_source).astype(float)
    return 1.0 - ((prod - 1.0) / prod) ** (K * n_target[:, None])


def _h01_small_count_indegree(K_slab_scaled):
    """In-degree for every entry from H01's own pair counts, small counts
    included: Jeffreys estimate (k + 0.5) pairs.  The number of candidate
    pairs behind each entry, N = k / p_slab, is known where k > 0; it is
    modelled as n_target x n_source (log N = a_t + b_s, least squares) to
    fill entries with k = 0."""
    k = H01_PAIRS_MEASURED.astype(float)
    known = (k > 0) & (H01_P_SLAB > 0)
    T, S = np.nonzero(known)
    logN = np.log(k[known] / H01_P_SLAB[known])
    nt, ns = k.shape
    A = np.zeros((T.size, nt + ns))
    A[np.arange(T.size), T] = 1; A[np.arange(T.size), nt + S] = 1
    coef, *_ = np.linalg.lstsq(A, logN, rcond=None)
    N_fit = np.exp(coef[:nt][:, None] + coef[nt:][None, :])
    N = np.where(known, k / np.where(known, H01_P_SLAB, 1.0), N_fit)
    # K per pair found = K_slab_scaled / k where known; = scale-equivalent via N otherwise
    per_pair = np.where(known, K_slab_scaled / np.where(known, k, 1.0), np.nan)
    # per_pair ~ c * n_source / N  -> estimate c from known entries
    return k, N, per_pair


def h01_conn_probs(n_human, verbose=False, untrusted="rodent"):
    """Connection probability matrix for the human column.

    Expected in-degree from source s onto a target of type t in the slab
    is  p_slab[t,s] * N_s.  Those in-degrees are rescaled by one global
    factor so that, over the trusted entries, they match the rodent
    reference in-degrees in the median.  Untrusted entries (fewer than
    MIN_PAIRS_TRUSTED connected pairs) keep the rodent in-degree.
    """
    rodent = rodent_column()
    K_rodent = _indegree_from_probs(rodent.conn_probs, rodent.full_num_neurons.astype(float),
                                    rodent.full_num_neurons.astype(float))
    K_slab = H01_P_SLAB * n_human[None, :].astype(float)
    trusted = H01_PAIRS_MEASURED >= MIN_PAIRS_TRUSTED
    ratio = K_rodent[trusted] / K_slab[trusted]
    scale = float(np.median(ratio))
    if untrusted == "h01":
        # every entry from H01's own counts (Jeffreys, k + 0.5 pairs): the slab
        # spans all layers, so a low count for a vertical pathway is a measurement
        k, N, _ = _h01_small_count_indegree(K_slab * scale)
        p_j = (k + 0.5) / N
        K_small = p_j * n_human[None, :].astype(float) * scale
        K_human = np.where(trusted, K_slab * scale, K_small)
    else:
        K_human = np.where(trusted, K_slab * scale, K_rodent)
    probs = _probs_from_indegree(K_human, n_human.astype(float), n_human.astype(float))
    if verbose:
        print(f"in-degree scale factor slab->column: {scale:.1f}  "
              f"(spread of per-entry ratios: {np.percentile(ratio, 25):.1f} .. {np.percentile(ratio, 75):.1f})")
        print(f"trusted entries: {trusted.sum()} / {trusted.size}")
    return probs, K_human, trusted, scale


CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "..", "results", "human_H01_calibration.json")


def human_h01_column(footprint_um=1000.0, verbose=False, use_calibration=True, untrusted="rodent") -> ColumnParams:
    n = h01_full_num_neurons(footprint_um)
    probs, K, trusted, scale = h01_conn_probs(n, verbose=verbose, untrusted=untrusted)
    sigma_E = fit_lateral_sigma(H01_LATERAL_PROFILE["E"])
    sigma_I = fit_lateral_sigma(H01_LATERAL_PROFILE["I"])
    rodent = rodent_column()
    params = ColumnParams(
        name="human_H01",
        full_num_neurons=n,
        footprint_um=footprint_um,
        layer_thickness_um=dict(H01_LAYER_THICKNESS_UM),
        lateral_sigma_um=float(0.5 * (sigma_E + sigma_I)),
        conn_probs=probs,
        # external drive and reference rates are not measurable in H01;
        # keep the rodent values so the column receives comparable input
        K_ext=rodent.K_ext.copy(),
        bg_rate=rodent.bg_rate,
        full_mean_rates=rodent.full_mean_rates.copy(),
    )
    # H01 gives synapse *counts*, not strengths.  Inhibitory weights onto each
    # population are scaled so that the ratio of excitatory to inhibitory
    # in-degree-weighted input matches the validated rodent column; without
    # this, L4I, L5I and L6E receive 2-3x more relative excitation and the
    # network runs away (documented in README, "Open problems").
    from .params import IS_EXC
    K_rodent = _indegree_from_probs(rodent.conn_probs, rodent.full_num_neurons.astype(float),
                                    rodent.full_num_neurons.astype(float))
    ei_human = K[:, IS_EXC].sum(1) / K[:, ~IS_EXC].sum(1)
    ei_rodent = K_rodent[:, IS_EXC].sum(1) / K_rodent[:, ~IS_EXC].sum(1)
    params.g_per_target = ei_human / ei_rodent
    # refined multipliers from scripts/calibrate_human.py, if they exist
    if use_calibration and os.path.exists(CALIBRATION_FILE):
        cal = json.load(open(CALIBRATION_FILE))
        params.g_per_target = np.array(cal["best"]["g"], float)
        params.h01_calibrated = True
    else:
        params.h01_calibrated = False
    params.h01_indegree = K
    params.h01_trusted = trusted
    params.h01_sigma_E_um = sigma_E
    params.h01_sigma_I_um = sigma_I
    params.h01_scale = scale
    return params


def describe_fit():
    p = human_h01_column(verbose=True)
    rodent = rodent_column()
    print("\nneurons per 1 mm^2 column:")
    print(f"  {'pop':5s} {'human(H01)':>11s} {'rodent':>8s}")
    for i, name in enumerate(POPULATIONS):
        print(f"  {name:5s} {p.full_num_neurons[i]:11d} {rodent.full_num_neurons[i]:8d}")
    print(f"  total {p.full_num_neurons.sum():11d} {rodent.full_num_neurons.sum():8d}")
    print(f"\nlayer thickness (um): {p.layer_thickness_um}  total {p.total_thickness_um:.0f}")
    print(f"lateral sigma: E {p.h01_sigma_E_um:.0f} um, I {p.h01_sigma_I_um:.0f} um (slab-truncated, lower bounds)")
    print("inhibitory weight rebalance factor per target population:",
          dict(zip(POPULATIONS, np.round(p.g_per_target, 2))))
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print("\nconnection probabilities (rows target, cols source), H01-fitted:")
    print(p.conn_probs)
    print("\ntrusted (H01) entries marked 1, rodent fallback 0:")
    print(p.h01_trusted.astype(int))
    print("\nrecurrent in-degree per neuron (sum over sources):")
    for i, name in enumerate(POPULATIONS):
        print(f"  {name:5s} human {p.h01_indegree[i].sum():7.0f}   rodent "
              f"{_indegree_from_probs(rodent.conn_probs, rodent.full_num_neurons.astype(float), rodent.full_num_neurons.astype(float))[i].sum():7.0f}")


if __name__ == "__main__":
    describe_fit()


# ---------------------------------------------------------------------------
# 4. Postsynaptic input budget (per-synapse E/I labels, no presynaptic soma needed)
# ---------------------------------------------------------------------------
# For every labelled neuron (h01.neurons, 3,111 cells) all synapses in
# synapses_c3 whose postsynaptic neuron_id is that cell, split by the
# classifier's type (1 inhibitory, 2 excitatory); cells with >= 50 inputs.
# Query (BigQuery, 4.25 GB scanned) in README "Stage 2b".  Unlike the
# pair-based probabilities above, these counts do not require the
# presynaptic cell body to lie inside the 0.17 mm slab, so small, local
# interneuron populations are not undercounted.
#   layer: (E cells, E inh synapses, E exc synapses, I cells, I inh, I exc)
H01_POSTSYNAPTIC_INPUTS = {
    "2": (290, 271009, 376225, 283, 121481, 118881),
    "3": (292, 392045, 831273, 290, 169344, 223556),
    "4": (293, 146923, 262279, 289, 135659, 200810),
    "5": (291, 196228, 351391, 293, 130891, 198424),
    "6": (288, 242001, 490340, 102, 48092, 56247),
}
_H01_LAYER_GROUPS = {"L23": ["2", "3"], "L4": ["4"], "L5": ["5"], "L6": ["6"]}


def h01_input_budget():
    """Per model layer: inhibitory share of inputs onto E and onto I cells, and
    the ratio of mean total inputs per I cell to per E cell (slab-truncated
    counts; truncation removes more of the large pyramidal arbours, so the
    I/E ratio is if anything an overestimate)."""
    out = {}
    for L, ls in _H01_LAYER_GROUPS.items():
        nE = sum(H01_POSTSYNAPTIC_INPUTS[l][0] for l in ls); iE = sum(H01_POSTSYNAPTIC_INPUTS[l][1] for l in ls)
        eE = sum(H01_POSTSYNAPTIC_INPUTS[l][2] for l in ls)
        nI = sum(H01_POSTSYNAPTIC_INPUTS[l][3] for l in ls); iI = sum(H01_POSTSYNAPTIC_INPUTS[l][4] for l in ls)
        eI = sum(H01_POSTSYNAPTIC_INPUTS[l][5] for l in ls)
        out[L] = dict(inh_share_E=iE / (iE + eE), inh_share_I=iI / (iI + eI),
                      inputs_I_over_E=((iI + eI) / nI) / ((iE + eE) / nE))
    return out
