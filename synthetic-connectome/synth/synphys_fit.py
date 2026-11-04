"""Human column preset: H01 wiring + Allen Synaptic Physiology strengths and
dynamics + Allen Cell Types membrane parameters (Phase 2).

Inputs (produced by the extraction scripts; absent files are skipped):

* ``results/synphys_human_summary.json``   from scripts/extract_synphys.py
* ``results/celltypes_human_summary.json`` from scripts/extract_celltypes.py

What is replaced relative to ``human_h01``:

* PSP amplitude per (target, source) population pair  <- synphys psp_mV
* weight spread per class pair                          <- synphys amplitude quartiles
* short-term plasticity per class pair (Tsodyks-Markram) <- synphys release-model fits
  (U = base release probability, tau_rec = depression tau, tau_fac =
  facilitation tau when facilitation is appreciable)
* membrane time constant, capacitance, threshold, rest  <- synphys intrinsic
  table, Cell Types export, or both (pooled)

Layer 2 and 3 groups are pooled into L2/3.  A quantity is used only when at
least ``MIN_N`` measurements back it; otherwise the next coarser level
(class pair across all layers), and finally the rodent value, is kept.  The
``provenance`` dict on the returned params records which level was used
for every entry, so a reader can see exactly how much is measured.
"""
from __future__ import annotations

import json
import os
import numpy as np

from .params import ColumnParams, POPULATIONS, N_POP, IS_EXC
from .h01_fit import human_h01_column, H01_SYNAPSES_PER_CONNECTION

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
SYNPHYS_FILE = os.path.join(RESULTS, "synphys_human_summary.json")
CELLTYPES_FILE = os.path.join(RESULTS, "celltypes_human_summary.json")
MIN_N = 3

POP_LAYERS = {"L23": ["2", "3", "2/3"], "L4": ["4"], "L5": ["5"], "L6": ["6"]}


def _pop_layer_cls(pop):
    return pop[:-1], ("ex" if pop[-1] == "E" else "in")


def _pooled(groups, keys, field):
    """n-weighted mean of group medians over the given keys; (value, n)."""
    vals, ns = [], []
    for k in keys:
        g = groups.get(k)
        if not g:
            continue
        v, n = g.get(field + "_median"), g.get(field + "_n", 0)
        if v is not None and n:
            vals.append(v); ns.append(n)
    if not ns:
        return None, 0
    return float(np.average(vals, weights=ns)), int(sum(ns))


def _pair_keys(groups, tgt_pop, src_pop, any_layer=False):
    tl, tc = _pop_layer_cls(tgt_pop)
    sl, sc = _pop_layer_cls(src_pop)
    out = []
    for k in groups:
        pre, post = k.split("->")
        pl, pc = pre.split(":"); ql, qc = post.split(":")
        if pc != sc or qc != tc:
            continue
        if any_layer or (pl in POP_LAYERS[sl] and ql in POP_LAYERS[tl]):
            out.append(k)
    return out


def human_synphys_column(synphys_file=SYNPHYS_FILE, celltypes_file=CELLTYPES_FILE,
                         verbose=True, mode="full", coba=False, ext_gain=1.0,
                         adapt_b_exc_pA=0.0, adapt_b_inh_pA=0.0, adapt_tau_ms=150.0,
                         min_n_layer_pair=MIN_N, **h01_kwargs) -> ColumnParams:
    """mode="full": measured membranes + measured synapses + STP (bursting state);
    mode="membranes": measured membranes only, rodent-scaled synapses (asynchronous state)."""
    params = human_h01_column(**h01_kwargs)
    params.name = "human_H01_synphys" if mode == "full" else "human_H01_membranes"
    if coba:
        params.name += "_coba"
        params.synapse_model = "coba"
    params.mode = mode
    params.ext_gain = ext_gain
    params.adapt_b_exc_pA, params.adapt_b_inh_pA, params.adapt_tau_ms = adapt_b_exc_pA, adapt_b_inh_pA, adapt_tau_ms
    prov = {"psp": {}, "stp": {}, "neuron": {}, "weight_spread": {}}
    syn = json.load(open(synphys_file)) if (mode == "full" and os.path.exists(synphys_file)) else None
    ct = json.load(open(celltypes_file)) if os.path.exists(celltypes_file) else None
    if syn is None and ct is None:
        if verbose:
            print("no synphys / celltypes summaries found; returning the H01-only column")
        params.provenance = prov
        return params

    # ---- synaptic strengths ------------------------------------------------
    if syn is not None:
        groups = syn["groups"]
        psp = params.PSP_matrix_mean.copy()            # starts as rodent values (E 0.15 mV, I -0.6 mV)
        for t, tp in enumerate(POPULATIONS):
            for s, sp in enumerate(POPULATIONS):
                v, n = _pooled(groups, _pair_keys(groups, tp, sp), "psp_mV")
                level = "layer-pair"
                if n < min_n_layer_pair:
                    v, n = _pooled(groups, _pair_keys(groups, tp, sp, any_layer=True), "psp_mV")
                    level = "class-pair"
                if n >= MIN_N and v is not None and (v > 0) == IS_EXC[s]:
                    psp[t, s] = v
                    prov["psp"][f"{tp}<-{sp}"] = f"{level} (n={n})"
                else:
                    prov["psp"][f"{tp}<-{sp}"] = "rodent"
        # the synthesizer builds PSPs from PSP_exc_mean, g and g_per_target;
        # override the whole matrix instead
        # Somatic unitary IPSPs measured in slices (~ -0.3 mV) are far weaker,
        # relative to EPSPs, than the g = -4 a current-based point-neuron
        # network needs for balance (dendritic location, driving force and
        # shunting are not representable here).  Start the inhibitory columns
        # at the multiplier that restores the rodent |IPSP/EPSP| ratio on top of
        # the H01 in-degree rebalance; scripts/calibrate_human.py refines it.
        # unitary amplitudes are per connection; the model's synapses are per contact
        for t in range(N_POP):
            for s_ in range(N_POP):
                psp[t, s_] /= H01_SYNAPSES_PER_CONNECTION[(POPULATIONS[t][-1], POPULATIONS[s_][-1])]
        prov["multapse_correction"] = "unitary PSP / synapses-per-connection (H01: E->E 1.5, I->E 1.7, E->I 1.8, I->I 1.7)"
        if coba:
            prov["inhibitory_start_multiplier"] = "none (conductance model uses measured amplitudes as measured)"
        else:
            bal = np.asarray(params.g_per_target, float)
            e_mean = np.array([np.mean(psp[t, IS_EXC]) for t in range(N_POP)])
            i_mean = np.array([np.mean(np.abs(psp[t, ~IS_EXC])) for t in range(N_POP)])
            g0 = 4.0 * e_mean / i_mean * bal
            psp[:, ~IS_EXC] *= g0[:, None]
            prov["inhibitory_start_multiplier"] = g0.round(2).tolist()
        params.psp_matrix_override = psp
        params.g_per_target = None

        # weight spread per class pair from amplitude quartiles
        spread = {}
        for tc in ("E", "I"):
            for sc in ("E", "I"):
                keys = _pair_keys(groups, "L23" + tc, "L23" + sc, any_layer=True)
                rel = []
                for k in keys:
                    g = groups[k]
                    if g.get("psp_mV_q25") is not None and g.get("psp_mV_median"):
                        rel.append((g["psp_mV_q75"] - g["psp_mV_q25"]) / (1.349 * abs(g["psp_mV_median"])))
                if rel:
                    spread[(tc, sc)] = float(np.median(rel))
                    prov["weight_spread"][f"{tc}<-{sc}"] = f"measured (n groups={len(rel)})"
        if spread:
            params.weight_rel_std_class = spread

        # short-term plasticity per class pair (Tsodyks-Markram mapping)
        stp = {}
        for tc in ("E", "I"):
            for sc in ("E", "I"):
                keys = _pair_keys(groups, "L23" + tc, "L23" + sc, any_layer=True)
                U, nU = _pooled(groups, keys, "release_prob")
                trec, nr = _pooled(groups, keys, "dep_tau_ms")
                famt, nf = _pooled(groups, keys, "fac_amount")
                tfac, _ = _pooled(groups, keys, "fac_tau_ms")
                if nU >= MIN_N and nr >= MIN_N and U:
                    stp[(tc, sc)] = dict(U=float(np.clip(U, 0.02, 0.95)), tau_rec_ms=float(trec),
                                         tau_fac_ms=float(tfac if (famt or 0) > 0.1 and tfac else 0.0))
                    prov["stp"][f"{tc}<-{sc}"] = f"measured (n={nU})"
                else:
                    prov["stp"][f"{tc}<-{sc}"] = "none (static)"
        params.stp = stp or None

    # ---- membrane parameters ----------------------------------------------
    tau = np.full(N_POP, params.neuron.tau_m); C = np.full(N_POP, params.neuron.C_m)
    Vth = np.full(N_POP, params.neuron.V_th); EL = np.full(N_POP, params.neuron.E_L)
    for i, pop in enumerate(POPULATIONS):
        layer, cls = _pop_layer_cls(pop)
        taus, ris, vths, vrests = [], [], [], []
        if syn is not None:
            for k, rec in syn.get("intrinsic", {}).items():
                l, c = k.split(":")
                if c == cls and l in POP_LAYERS[layer]:
                    if rec.get("tau_ms", {}).get("median"): taus.append((rec["tau_ms"]["median"], rec["tau_ms"]["n"]))
                    if rec.get("input_resistance_MOhm", {}).get("median"): ris.append((rec["input_resistance_MOhm"]["median"], rec["input_resistance_MOhm"]["n"]))
                    if rec.get("ap_threshold_v_mV", {}).get("median"): vths.append((rec["ap_threshold_v_mV"]["median"], rec["ap_threshold_v_mV"]["n"]))
        if ct is not None:
            for k, rec in ct.get("groups", {}).items():
                l, c = k.split(":")
                if c == pop[-1] and l in POP_LAYERS[layer]:
                    if rec.get("tau"): taus.append((rec["tau"]["median"], rec["tau"]["n"]))
                    if rec.get("ri"): ris.append((rec["ri"]["median"], rec["ri"]["n"]))
                    if rec.get("vthresh"): vths.append((rec["vthresh"]["median"], rec["vthresh"]["n"]))
                    elif rec.get("vthresh_est") is not None: vths.append((rec["vthresh_est"], rec["tau"]["n"] if rec.get("tau") else 1))
                    if rec.get("vrest"): vrests.append((rec["vrest"]["median"], rec["vrest"]["n"]))
        def wavg(pairs):
            return float(np.average([v for v, _ in pairs], weights=[n for _, n in pairs])) if sum(n for _, n in pairs) >= MIN_N else None
        t_, r_, v_, e_ = wavg(taus), wavg(ris), wavg(vths), wavg(vrests)
        used = []
        if t_: tau[i] = t_; used.append("tau_m")
        if t_ and r_: C[i] = t_ / r_ * 1e3; used.append("C_m")
        if e_: EL[i] = e_; used.append("E_L")
        if v_: Vth[i] = v_; used.append("V_th")
        prov["neuron"][pop] = ", ".join(used) if used else "rodent"
    params.neuron.tau_m_pop, params.neuron.C_m_pop = tau, C
    params.neuron.V_th_pop, params.neuron.E_L_pop = Vth, EL
    # start the membrane potentials between rest and threshold
    params.neuron.V0_mean = (EL + 0.3 * (Vth - EL))
    params.neuron.V0_std = np.full(N_POP, 4.0)
    # refined inhibitory multipliers for this preset, if calibrate_human.py produced them
    cal_file = os.path.join(RESULTS, f"{params.name}_calibration.json")
    if coba:
        params.g_per_target = None if params.psp_matrix_override is not None else params.g_per_target
        params.K_ext = params.K_ext * float(ext_gain)
        params.full_mean_rates = None     # no rate-based DC compensation: nothing is assumed about the rates
        prov["calibration"] = f"none; external drive gain {ext_gain}"
    elif h01_kwargs.get("use_calibration", True) and os.path.exists(cal_file):
        gcal = np.array(json.load(open(cal_file))["best"]["g"], float)
        if params.psp_matrix_override is not None:
            m = params.PSP_matrix_mean; m[:, 1::2] *= gcal[:, None]; params.psp_matrix_override = m
        else:
            params.g_per_target = gcal
        prov["calibration"] = {"inhibitory multipliers": gcal.round(2).tolist()}
    params.provenance = prov
    if verbose:
        print("provenance of human_H01_synphys parameters:")
        for section, d in prov.items():
            if not isinstance(d, dict) or section == "calibration":
                print(f"  {section:14s} {d}")
                continue
            measured = sum(1 for v in d.values() if not str(v).startswith("rodent") and not str(v).startswith("none"))
            print(f"  {section:14s} {measured}/{len(d)} entries from data")
    return params
