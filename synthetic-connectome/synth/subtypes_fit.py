"""12-population human column: each layer's interneurons split into
fast-spiking (F) and non-fast-spiking (N) cells (Phase S2, step 2).

Starts from the conductance-based 8-population column (``human_h01_coba``)
and splits every inhibitory population using
``results/synphys_human_subtypes.json`` (scripts/extract_synphys_subtypes.py)
and the F/N groups in ``results/celltypes_human_summary.json``
(scripts/extract_celltypes.py).  Nothing is tuned; each rule below is a
measured ratio applied on top of the 8-population numbers, with a stated
fallback when the data are too thin.

Cell numbers
    Each layer's interneurons (count from H01) are split by the measured
    FS fraction pooled over all layers (38%, 95% CI 29-48%).  Per-layer
    fractions outside L2/3 rest on 4-9 cells each and are not used.

Connections (in-degree per target neuron)
    The total number of inhibitory synapses onto a cell type is kept at the
    H01 value; only *who supplies them* changes.
    * source split (F vs N presynaptic): proportional to
      cell fraction x measured connection probability, when both subtypes
      have >= MIN_PROBED probed pairs onto that target class; otherwise
      proportional to cell fraction alone (no preference).
      I->E: measured (FS 13/70, nFS 2/75 -> FS supply ~81%).
      I->I: too few pairs -> cell fraction.
    * target split (inputs onto F vs N): in-degree proportional to the
      measured connection probability onto each subtype, normalised so the
      population total is unchanged; same MIN_PROBED rule.
      E->I: measured (E->FS 20/79, E->nFS 8/84).  I->I: equal.

Synaptic strength
    PSP[t,s] = 8-population value x (subtype-pair median / class-pair
    median) from the same recordings, when the subtype pair has >= MIN_N
    amplitudes; otherwise the class value is kept (nFS->E has no measured
    amplitude, so it keeps the I->E class value).

Short-term plasticity
    Release-model fits per subtype pair when >= MIN_N fits, else class.

Membranes
    tau_m, R_in -> C_m, threshold and rest per (layer, subtype) pooled from
    Synaptic Physiology intrinsic recordings and Cell Types (interneurons
    split by upstroke/downstroke ratio < 2.0); when a layer has < MIN_N cells
    of a subtype, the subtype is pooled across layers.

Adaptation
    Increment b for F and N = b_E x (measured adaptation index of the
    subtype / that of excitatory cells).  A proper fit is step 4.
"""
from __future__ import annotations

import json
import os
import numpy as np

from .params import pop_layer, pop_type
from .synphys_fit import human_synphys_column, RESULTS, CELLTYPES_FILE, POP_LAYERS
from .h01_fit import _indegree_from_probs, _probs_from_indegree, h01_input_budget

SUBTYPES_FILE = os.path.join(RESULTS, "synphys_human_subtypes.json")
MIN_N = 3
MIN_PROBED = 30
SUB = {"E": "E", "F": "FS", "N": "nFS"}          # population letter -> subtype label in the json
LAYER_KEYS = {"L23": ["23", "2", "3", "2/3"], "L4": ["4"], "L5": ["5"], "L6": ["6"]}


def _g(groups, key):
    return groups.get(key) or {}


def apply_input_budget(K, pops, n_per_pop, budget=None):
    """Rescale in-degrees so each population matches the H01 postsynaptic
    budget: (1) interneurons' mean total input = I/E ratio x the layer's
    excitatory cells' total (F/N keep their relative sizes); (2) the
    inhibitory share of every population's inputs = H01 share for its class
    and layer.  Within the excitatory and inhibitory sources, the relative
    contributions of each source population are kept as they were."""
    budget = budget or h01_input_budget()
    K = np.array(K, float)
    exc = np.array([pop_type(p) == "E" for p in pops])
    layers = [pop_layer(p) for p in pops]
    tot = K.sum(1)
    new_tot = tot.copy()
    for L, b in budget.items():
        e_idx = [i for i, p in enumerate(pops) if layers[i] == L and exc[i]]
        i_idx = [i for i, p in enumerate(pops) if layers[i] == L and not exc[i]]
        if not e_idx or not i_idx:
            continue
        w = np.asarray(n_per_pop, float)[i_idx]
        mean_I = np.average(tot[i_idx], weights=w)
        target_I = b["inputs_I_over_E"] * tot[e_idx[0]]
        new_tot[i_idx] = tot[i_idx] * target_I / mean_I
    for t, p in enumerate(pops):
        b = budget.get(layers[t])
        if b is None:
            continue
        share = b["inh_share_E"] if exc[t] else b["inh_share_I"]
        se, si = K[t, exc].sum(), K[t, ~exc].sum()
        if se > 0:
            K[t, exc] *= new_tot[t] * (1 - share) / se
        if si > 0:
            K[t, ~exc] *= new_tot[t] * share / si
    return K


def human_12pop_column(ext_gain=0.5, adapt_b_exc_pA=60.0, adapt_tau_ms=150.0, verbose=True,
                       subtypes_file=SUBTYPES_FILE, celltypes_file=CELLTYPES_FILE, fs_fraction=None,
                       input_budget=True, measured_kinetics=True, ext_budget=False, bg_ratio=None,
                       min_n_layer_pair=10, two_compartment=False, dend_cap_frac=0.7, fs_soma_fraction=1.0,
                       deep_ext_gain=None, ef_deep_factor=None, **kw):
    base = human_synphys_column(mode="full", coba=True, ext_gain=ext_gain, adapt_b_exc_pA=adapt_b_exc_pA,
                                adapt_b_inh_pA=adapt_b_exc_pA / 6, adapt_tau_ms=adapt_tau_ms,
                                verbose=False, min_n_layer_pair=min_n_layer_pair, **kw)
    sub = json.load(open(subtypes_file))
    groups = sub["groups"]
    ct = json.load(open(celltypes_file)) if os.path.exists(celltypes_file) else {"groups": {}}
    f = float(fs_fraction if fs_fraction is not None else sub["split_by_layer"]["all"]["FS_fraction"])
    prov = {"fs_fraction": f"{f:.3f} (pooled over layers; per-layer counts too small outside L2/3)"}

    pops8 = list(base.populations)
    pops12, parent = [], []
    for i, p in enumerate(pops8):
        if p.endswith("E"):
            pops12.append(p); parent.append(i)
        else:
            L = pop_layer(p)
            pops12 += [L + "F", L + "N"]; parent += [i, i]
    parent = np.array(parent)
    P = len(pops12)
    types = [pop_type(p) for p in pops12]
    frac = np.array([1.0 if t == "E" else (f if t == "F" else 1 - f) for t in types])

    # ---- neurons ---------------------------------------------------------
    n8 = base.full_num_neurons.astype(float)
    n12 = np.round(n8[parent] * frac).astype(int)

    # ---- in-degrees --------------------------------------------------------
    K8 = _indegree_from_probs(base.conn_probs, n8, n8)

    def p_of(pre_labels, post_labels):
        pr = sum(_g(groups, f"{a}->{b}").get("probed", 0) for a in pre_labels for b in post_labels)
        cn = sum(_g(groups, f"{a}->{b}").get("connected", 0) for a in pre_labels for b in post_labels)
        return pr, (cn / pr if pr else None)

    I_LABELS = ["FS", "nFS", "I?"]
    # source shares: who supplies inhibition onto a target class
    src_share = {}
    for tcls, tl in (("E", ["E"]), ("I", I_LABELS)):
        (pf_n, pf), (pn_n, pn) = p_of(["FS"], tl), p_of(["nFS"], tl)
        if pf_n >= MIN_PROBED and pn_n >= MIN_PROBED and (pf or pn):
            wF, wN = f * pf, (1 - f) * pn
            src_share[tcls] = {"F": wF / (wF + wN), "N": wN / (wF + wN)}
            prov[f"source split onto {tcls}"] = (f"measured: F {src_share[tcls]['F']:.2f} / N {src_share[tcls]['N']:.2f} "
                                                f"(p FS {pf:.3f} n={pf_n}, nFS {pn:.3f} n={pn_n})")
        else:
            src_share[tcls] = {"F": f, "N": 1 - f}
            prov[f"source split onto {tcls}"] = f"cell fraction (probed FS {pf_n}, nFS {pn_n} < {MIN_PROBED})"
    # target factors: how many inputs an F vs an N cell receives from a source class
    tgt_factor = {}
    for scls, sl in (("E", ["E"]), ("I", I_LABELS)):
        (pf_n, pf), (pn_n, pn) = p_of(sl, ["FS"]), p_of(sl, ["nFS"])
        if pf_n >= MIN_PROBED and pn_n >= MIN_PROBED and (pf or pn):
            norm = f * pf + (1 - f) * pn
            tgt_factor[scls] = {"F": pf / norm, "N": pn / norm}
            prov[f"target split from {scls}"] = (f"measured: F x{tgt_factor[scls]['F']:.2f}, N x{tgt_factor[scls]['N']:.2f} "
                                                f"(p onto FS {pf:.3f} n={pf_n}, onto nFS {pn:.3f} n={pn_n})")
        else:
            tgt_factor[scls] = {"F": 1.0, "N": 1.0}
            prov[f"target split from {scls}"] = f"equal (probed onto FS {pf_n}, onto nFS {pn_n} < {MIN_PROBED})"

    K12 = np.zeros((P, P))
    for t in range(P):
        for s in range(P):
            k = K8[parent[t], parent[s]]
            tcls = "E" if types[t] == "E" else "I"
            scls = "E" if types[s] == "E" else "I"
            if types[s] != "E":
                k *= src_share[tcls][types[s]]
            if types[t] != "E":
                k *= tgt_factor[scls][types[t]]
            K12[t, s] = k
    if input_budget:
        before = K12.copy()
        K12 = apply_input_budget(K12, pops12, n12)
        ex = np.array([t == "E" for t in types])
        prov["input budget (H01 postsynaptic counts)"] = {
            q: (f"inputs {before[i].sum():.0f}->{K12[i].sum():.0f}, inh share "
                f"{before[i, ~ex].sum() / before[i].sum():.2f}->{K12[i, ~ex].sum() / K12[i].sum():.2f}")
            for i, q in enumerate(pops12)}
    probs12 = _probs_from_indegree(K12, n12.astype(float), n12.astype(float))

    # ---- PSP amplitudes ------------------------------------------------------
    psp8 = base.PSP_matrix_mean
    psp12 = psp8[np.ix_(parent, parent)].copy()
    prov["psp"] = {}
    for tt in ("E", "F", "N"):
        for st in ("E", "F", "N"):
            if tt == "E" and st == "E":
                continue
            g = _g(groups, f"{SUB[st]}->{SUB[tt]}")
            c = _g(groups, f"class:{'E' if st == 'E' else 'I'}->class:{'E' if tt == 'E' else 'I'}")
            if g.get("psp_n", 0) >= MIN_N and c.get("psp_n", 0) >= MIN_N and g.get("psp_mV") and c.get("psp_mV"):
                r = g["psp_mV"] / c["psp_mV"]
                for t in range(P):
                    for s in range(P):
                        if types[t] == tt and types[s] == st:
                            psp12[t, s] *= r
                prov["psp"][f"{tt}<-{st}"] = f"x{r:.2f} (subtype {g['psp_mV']:.2f} mV n={g['psp_n']} / class {c['psp_mV']:.2f} mV)"
            else:
                prov["psp"][f"{tt}<-{st}"] = f"class value (subtype n={g.get('psp_n', 0)})"

    # CALIBRATION: E -> fast-spiking strength onto targets outside L2/3, where it
    # is not measured (human recordings: L2/3 n = 9-19; L5 n = 2, L6 n = 1, cross-
    # layer n = 3, all weaker).  Fitted with the deep external input so deep
    # fast-spiking cells fire at awake rates; scripts/gain_diagnostics.py, README
    # "Stage 5b".  Loaded from results/deep_ext_calibration.json when not given.
    if ef_deep_factor is None:
        cal = os.path.join(RESULTS, "deep_ext_calibration.json")
        if os.path.exists(cal):
            ef_deep_factor = json.load(open(cal)).get("ef_deep_factor")
    if ef_deep_factor:
        for t in range(P):
            if types[t] == "F" and pop_layer(pops12[t]) != "L23":
                for s_ in range(P):
                    if types[s_] == "E":
                        psp12[t, s_] *= float(ef_deep_factor)
        prov["CALIBRATED E->FS outside L2/3"] = f"x{float(ef_deep_factor):.2f} (fitted, not measured)"

    # ---- short-term plasticity ---------------------------------------------
    stp = dict(base.stp or {})
    prov["stp"] = {}
    for tt in ("E", "F", "N"):
        for st in ("E", "F", "N"):
            if tt == "E" and st == "E":
                continue
            g = _g(groups, f"{SUB[st]}->{SUB[tt]}")
            if g.get("model_n", 0) >= MIN_N and g.get("release_U") and g.get("dep_tau_ms"):
                tf = g["fac_tau_ms"] if (g.get("fac_amount") or 0) > 0.1 and g.get("fac_tau_ms") else 0.0
                stp[(tt, st)] = dict(U=float(np.clip(g["release_U"], 0.02, 0.95)), tau_rec_ms=float(g["dep_tau_ms"]),
                                     tau_fac_ms=float(tf))
                prov["stp"][f"{tt}<-{st}"] = (f"subtype (n={g['model_n']}): U {g['release_U']:.2f}, "
                                              f"tau_rec {g['dep_tau_ms']:.0f} ms, tau_fac {tf:.0f} ms")
            else:
                prov["stp"][f"{tt}<-{st}"] = "class value"

    # ---- membranes ------------------------------------------------------------
    tau8, C8, Vth8, EL8, _ = base.neuron_pop_arrays()
    tau, C, Vth, EL = tau8[parent].copy(), C8[parent].copy(), Vth8[parent].copy(), EL8[parent].copy()
    syn_intr = sub.get("intrinsic_by_layer", {})
    ctg = ct.get("groups", {})
    prov["neuron"] = {}

    def collect(t, layer_keys_syn, layer_keys_ct):
        taus, ris, vths, vrests = [], [], [], []
        for k, rec in syn_intr.items():
            l, c = k.split(":")
            if c == SUB[t] and l in layer_keys_syn:
                for lst, fld in ((taus, "tau_ms"), (ris, "input_resistance_MOhm"), (vths, "ap_threshold_v_mV")):
                    if rec.get(fld, {}).get("median") is not None:
                        lst.append((rec[fld]["median"], rec[fld]["n"]))
        for k, rec in ctg.items():
            l, c = k.split(":")
            if c == t and l in layer_keys_ct:
                if rec.get("tau"): taus.append((rec["tau"]["median"], rec["tau"]["n"]))
                if rec.get("ri"): ris.append((rec["ri"]["median"], rec["ri"]["n"]))
                if rec.get("vthresh"): vths.append((rec["vthresh"]["median"], rec["vthresh"]["n"]))
                elif rec.get("vthresh_est") is not None and rec.get("tau"):
                    vths.append((rec["vthresh_est"], rec["tau"]["n"]))
                if rec.get("vrest"): vrests.append((rec["vrest"]["median"], rec["vrest"]["n"]))
        return taus, ris, vths, vrests

    def wavg(pairs):
        n = sum(n for _, n in pairs)
        return (float(np.average([v for v, _ in pairs], weights=[n for _, n in pairs])), n) if n >= MIN_N else (None, n)

    all_syn = sorted({k.split(":")[0] for k in syn_intr} - {"1", "None", "?"})
    all_ct = sorted({k.split(":")[0] for k in ctg} - {"1", "?"})
    for i, p in enumerate(pops12):
        t = types[i]
        if t == "E":
            prov["neuron"][p] = "as 8-population E"
            continue
        L = pop_layer(p)
        local = collect(t, LAYER_KEYS[L], POP_LAYERS[L])
        pooled = collect(t, all_syn, all_ct)
        used = []
        vals = []
        for j, name in enumerate(("tau_m", "R_in", "V_th", "E_L")):
            v, n = wavg(local[j])
            lvl = "layer"
            if v is None:
                v, n = wavg(pooled[j]); lvl = "all layers"
            vals.append(v)
            if v is not None:
                used.append(f"{name} {v:.1f} ({lvl}, n={n})")
        t_, r_, v_, e_ = vals
        if t_: tau[i] = t_
        if t_ and r_: C[i] = t_ / r_ * 1e3
        if v_: Vth[i] = v_
        if e_: EL[i] = e_
        prov["neuron"][p] = "; ".join(used) if used else "as 8-population I"

    # ---- adaptation -------------------------------------------------------------
    ib = sub.get("intrinsic_by_subtype", {})
    a_E = ib.get("E", {}).get("adapt", {}).get("median")
    b = np.zeros(P)
    for i, t in enumerate(types):
        if t == "E":
            b[i] = adapt_b_exc_pA
        else:
            a_s = ib.get(SUB[t], {}).get("adapt", {}).get("median")
            b[i] = adapt_b_exc_pA * a_s / a_E if (a_s is not None and a_E) else adapt_b_exc_pA / 6
    prov["adaptation"] = {SUB[t]: round(float(b[types.index(t)]), 1) for t in ("E", "F", "N")}

    # ---- synaptic current decay (voltage-clamp PSC decay, somatic) -------------------
    te = np.full(P, base.tau_exc_ms); ti = np.full(P, base.tau_inh_ms)
    if measured_kinetics:
        prov["kinetics"] = {}
        for i, t in enumerate(types):
            for src, arr, generic in (("E", te, base.tau_exc_ms), ("I", ti, base.tau_inh_ms)):
                tl = SUB[t]
                g = _g(groups, f"{src}->{tl}") if src == "E" else _g(groups, f"class:I->class:{'E' if t == 'E' else 'I'}")
                if g.get("psc_decay_n", 0) >= MIN_N and g.get("psc_decay_ms"):
                    arr[i] = g["psc_decay_ms"]
                    prov["kinetics"][f"{pops12[i]}<-{src}"] = f"{arr[i]:.1f} ms (measured, n={g['psc_decay_n']})"
                else:
                    prov["kinetics"][f"{pops12[i]}<-{src}"] = f"{generic:.1f} ms (generic; n={g.get('psc_decay_n', 0)})"

    # ---- assemble -----------------------------------------------------------------
    p = base
    p.name = "human_H01_12pop_coba" + ("" if input_budget else "_nobudget") + ("_2comp" if two_compartment else "")
    p.populations = pops12
    p.full_num_neurons = n12
    p.conn_probs = probs12
    p.psp_matrix_override = psp12
    p.g_per_target = None
    p.stp = stp
    p.K_ext = np.asarray(base.K_ext, float)[parent]
    if ext_budget == "h01":
        # external (outside-column) fibres onto interneurons scaled by the H01
        # ratio of total excitatory synapses per interneuron to per pyramidal
        # cell in the same layer (postsynaptic counts; no F/N split exists in H01)
        hb = h01_input_budget()
        new = p.K_ext.copy()
        prov["external input budget"] = {}
        for i, q in enumerate(pops12):
            if types[i] == "E":
                continue
            L = pop_layer(q); bb = hb[L]
            ratio = bb["inputs_I_over_E"] * (1 - bb["inh_share_I"]) / (1 - bb["inh_share_E"])
            e_i = pops12.index(L + "E")
            new[i] = p.K_ext[e_i] * ratio
            prov["external input budget"][q] = f"{p.K_ext[i]:.0f} -> {new[i]:.0f} fibres (H01 excitatory-input ratio I/E {ratio:.2f})"
        p.K_ext = new
    elif ext_budget:
        # every cell in a layer gets the same fraction of its excitation from
        # outside the column as that layer's pyramidal cells: external fibres
        # scale with the cell's (H01-budgeted) recurrent excitatory in-degree
        exc_in = K12[:, np.array([t == "E" for t in types])].sum(1)
        new = p.K_ext.copy()
        prov["external input budget"] = {}
        for i, q in enumerate(pops12):
            if types[i] == "E":
                continue
            e_i = pops12.index(pop_layer(q) + "E")
            new[i] = p.K_ext[e_i] * exc_in[i] / exc_in[e_i]
            prov["external input budget"][q] = f"{p.K_ext[i]:.0f} -> {new[i]:.0f} fibres"
        p.K_ext = new
    p.bg_ratio = bg_ratio
    # CALIBRATION (the one fitted number set in this preset): extra external
    # fibres onto deep pyramidal cells, standing in for thalamic / long-range
    # input and apical-tuft input that the 0.17 mm H01 slab truncates.  Fitted by
    # scripts/calibrate_deep.py to awake target rates; see README "Stage 3c".
    if deep_ext_gain is None and two_compartment:
        cal = os.path.join(RESULTS, "deep_ext_calibration.json")
        if os.path.exists(cal):
            deep_ext_gain = json.load(open(cal))["deep_ext_gain"]
    if deep_ext_gain:
        prov["CALIBRATED deep external input"] = {}
        for q, m in deep_ext_gain.items():
            if q in pops12:
                i = pops12.index(q)
                p.K_ext[i] *= float(m)
                prov["CALIBRATED deep external input"][q] = f"x{float(m):.2f} external fibres (fitted, not measured)"
    p.adapt_b_pop = b
    p.tau_exc_pop, p.tau_inh_pop = te, ti
    if two_compartment:
        # Somatic voltage clamp of E->E currents is slowed by dendritic
        # filtering (7.2 ms) relative to E->interneuron currents (2.2 ms, compact
        # cells).  With a dendrite in the model that filtering is explicit, so
        # (a) E->E conductances use the unfiltered decay, and (b) the coupling
        # is set so the dendrite's clamped time constant tau_m (1 - A) equals the
        # measured clamp decay: A = 1 - tau_clamp / tau_m (per population).
        gEE, gEI = _g(groups, "class:E->class:E"), _g(groups, "class:E->class:I")
        tau_clamp, tau_true = gEE.get("psc_decay_ms"), gEI.get("psc_decay_ms")
        A = np.full(P, 0.5)
        prov["dendrite"] = {}
        for i, q in enumerate(pops12):
            if types[i] != "E":
                continue
            A[i] = float(np.clip(1 - tau_clamp / tau[i], 0.2, 0.95))
            te[i] = tau_true
            prov["dendrite"][q] = (f"attenuation {A[i]:.2f} (clamp decay {tau_clamp:.1f} ms, tau_m {tau[i]:.1f} ms); "
                                   f"E conductance decay {tau_true:.1f} ms; dendritic share of membrane {dend_cap_frac}")
        p.tau_exc_pop = te
        p.two_compartment = True
        p.dend_attenuation_pop = A
        p.dend_cap_frac = dend_cap_frac
        p.fs_soma_fraction = fs_soma_fraction
    p.full_mean_rates = None
    nrn = p.neuron
    nrn.tau_m_pop, nrn.C_m_pop, nrn.V_th_pop, nrn.E_L_pop = tau, C, Vth, EL
    nrn.V0_mean = EL + 0.3 * (Vth - EL)
    nrn.V0_std = np.full(P, 4.0)
    p.indegree_12 = K12
    p.provenance = {**getattr(base, "provenance", {}), "subtypes": prov}
    if verbose:
        print("12-population column (fast-spiking / non-fast-spiking interneurons):")
        for k, v in prov.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk:6s} {vv}")
            else:
                print(f"  {k}: {v}")
    return p
