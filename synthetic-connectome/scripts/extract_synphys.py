#!/usr/bin/env python3
"""Extract human synaptic-physiology summaries from an Allen Synaptic
Physiology sqlite database (aisynphys release, any size).

Needs only the Python standard library, so it runs anywhere the sqlite file
is: ``python scripts/extract_synphys.py path/to/synphys_r2.0_small.sqlite``.

Output: ``results/synphys_human_summary.json`` with, per (pre layer, pre
class, post layer, post class) group:

* probed / connected pair counts and the raw connection probability
* resting-state PSP amplitude (mV), latency, rise time, decay tau
* short-term plasticity: paired-pulse ratio, 50 Hz induction, 250 ms recovery
* release-model fits: base release probability, depression/facilitation
  amount and time constants (Tsodyks-Markram-like)

plus connection probability vs lateral soma distance per presynaptic class,
and intrinsic membrane properties per (layer, class) for the neuron model.

Cell class is taken from ``cell.cell_class_nonsynaptic`` (morphology /
markers, independent of synaptic currents) as the aisynphys authors recommend
for connectivity measurements; a pair counts as "probed" when more than 10
test spikes were recorded at the holding potential relevant to the
presynaptic class, matching ``aisynphys.connectivity.pair_was_probed``.
"""
import argparse
import json
import os
import sqlite3
import statistics as st
from collections import defaultdict


def q(con, sql, params=()):
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def med(xs):
    xs = [x for x in xs if x is not None]
    return (st.median(xs), len(xs)) if xs else (None, 0)


def quartiles(xs):
    xs = sorted(x for x in xs if x is not None)
    if len(xs) < 4:
        return None, None
    q = st.quantiles(xs, n=4)
    return q[0], q[2]


def norm_layer(layer):
    if layer is None:
        return None
    s = str(layer).strip().upper().replace("LAYER", "").replace("L", "").strip()
    return s if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--out", default="results/synphys_human_summary.json")
    ap.add_argument("--species", default="human")
    ap.add_argument("--test-spikes", type=int, default=10)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    tables = {r["name"] for r in q(con, "SELECT name FROM sqlite_master WHERE type='table'")}
    need = {"slice", "experiment", "cell", "pair", "synapse"}
    missing = need - tables
    if missing:
        raise SystemExit(f"database is missing tables {sorted(missing)}; got {sorted(tables)}")
    has_loc = "cortical_cell_location" in tables
    has_dyn = "dynamics" in tables
    has_model = "synapse_model" in tables
    has_intr = "intrinsic" in tables

    # ---- cells of the requested species, with class and layer -------------
    cells = q(con, f"""
        SELECT c.id AS cell_id, c.cell_class_nonsynaptic AS cls, c.cell_class AS cls_syn,
               c.target_layer AS target_layer, e.project_name AS project
               {", loc.cortical_layer AS loc_layer, loc.distance_to_pia AS d_pia, loc.fractional_depth AS fdepth" if has_loc else ""}
        FROM cell c
        JOIN experiment e ON e.id = c.experiment_id
        JOIN slice s ON s.id = e.slice_id
        {"LEFT JOIN cortical_cell_location loc ON loc.cell_id = c.id" if has_loc else ""}
        WHERE s.species = ?""", (args.species,))
    info = {}
    for c in cells:
        layer = norm_layer(c.get("loc_layer")) or norm_layer(c.get("target_layer"))
        cls = c["cls"] if c["cls"] in ("ex", "in") else (c["cls_syn"] if c["cls_syn"] in ("ex", "in") else None)
        info[c["cell_id"]] = dict(layer=layer, cls=cls, project=c["project"],
                                  d_pia=c.get("d_pia"), fdepth=c.get("fdepth"))
    print(f"{len(info)} {args.species} cells; with class: {sum(1 for v in info.values() if v['cls'])}, "
          f"with layer: {sum(1 for v in info.values() if v['layer'])}")

    # ---- pairs ---------------------------------------------------------------
    pairs = q(con, """
        SELECT p.id, p.pre_cell_id, p.post_cell_id, p.has_synapse, p.has_electrical,
               p.n_ex_test_spikes, p.n_in_test_spikes, p.distance, p.lateral_distance, p.vertical_distance
        FROM pair p""")
    pairs = [p for p in pairs if p["pre_cell_id"] in info and p["post_cell_id"] in info]
    print(f"{len(pairs)} pairs between {args.species} cells")

    syn_by_pair = {s["pair_id"]: s for s in q(con, """
        SELECT pair_id, synapse_type, latency, psp_amplitude, psp_rise_time, psp_decay_tau, psc_amplitude
        FROM synapse""")}
    dyn_by_pair = {d["pair_id"]: d for d in q(con, """
        SELECT pair_id, qc_pass, n_source_events, paired_pulse_ratio_50hz, stp_initial_50hz,
               stp_induction_50hz, stp_recovery_250ms, variability_resting_state
        FROM dynamics""")} if has_dyn else {}
    model_by_pair = {m["pair_id"]: m for m in q(con, """
        SELECT pair_id, n_source_events, ml_n_release_sites, ml_base_release_probability,
               ml_mini_amplitude, ml_depression_amount, ml_depression_tau,
               ml_facilitation_amount, ml_facilitation_tau, ml_strength
        FROM synapse_model""")} if has_model else {}

    groups = defaultdict(lambda: dict(probed=0, connected=0, electrical=0,
                                      psp_mV=[], psc_pA=[], latency_ms=[], rise_ms=[], decay_ms=[],
                                      ppr_50hz=[], stp_initial_50hz=[], stp_induction_50hz=[],
                                      stp_recovery_250ms=[], cv_resting=[],
                                      release_prob=[], n_sites=[], dep_amount=[], dep_tau_ms=[],
                                      fac_amount=[], fac_tau_ms=[], distances_um=[]))
    dist_bins = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # cls -> bin -> [probed, connected]

    for p in pairs:
        pre, post = info[p["pre_cell_id"]], info[p["post_cell_id"]]
        if pre["cls"] is None or post["cls"] is None:
            continue
        n_test = p["n_ex_test_spikes"] if pre["cls"] == "ex" else p["n_in_test_spikes"]
        probed = (n_test or 0) > args.test_spikes
        if not probed:
            continue
        key = f"{pre['layer']}:{pre['cls']}->{post['layer']}:{post['cls']}"
        g = groups[key]
        g["probed"] += 1
        connected = bool(p["has_synapse"])
        g["connected"] += int(connected)
        g["electrical"] += int(bool(p["has_electrical"]))
        if p["lateral_distance"] is not None:
            b = int(p["lateral_distance"] * 1e6 // 50) * 50
            dist_bins[pre["cls"]][b][0] += 1
            dist_bins[pre["cls"]][b][1] += int(connected)
        if not connected:
            continue
        if p["distance"] is not None:
            g["distances_um"].append(p["distance"] * 1e6)
        s = syn_by_pair.get(p["id"])
        if s:
            # keep type-consistent synapses only (ex pre -> 'ex' synapse etc.)
            if s["synapse_type"] in (None, pre["cls"]):
                if s["psp_amplitude"] is not None: g["psp_mV"].append(s["psp_amplitude"] * 1e3)
                if s["psc_amplitude"] is not None: g["psc_pA"].append(s["psc_amplitude"] * 1e12)
                if s["latency"] is not None: g["latency_ms"].append(s["latency"] * 1e3)
                if s["psp_rise_time"] is not None: g["rise_ms"].append(s["psp_rise_time"] * 1e3)
                if s["psp_decay_tau"] is not None: g["decay_ms"].append(s["psp_decay_tau"] * 1e3)
        d = dyn_by_pair.get(p["id"])
        if d and (d["qc_pass"] is None or d["qc_pass"]):
            for k_src, k_dst in (("paired_pulse_ratio_50hz", "ppr_50hz"), ("stp_initial_50hz", "stp_initial_50hz"),
                                 ("stp_induction_50hz", "stp_induction_50hz"), ("stp_recovery_250ms", "stp_recovery_250ms"),
                                 ("variability_resting_state", "cv_resting")):
                if d[k_src] is not None:
                    g[k_dst].append(d[k_src])
        m = model_by_pair.get(p["id"])
        if m:
            if m["ml_base_release_probability"] is not None: g["release_prob"].append(m["ml_base_release_probability"])
            if m["ml_n_release_sites"] is not None: g["n_sites"].append(m["ml_n_release_sites"])
            if m["ml_depression_amount"] is not None: g["dep_amount"].append(m["ml_depression_amount"])
            if m["ml_depression_tau"] is not None: g["dep_tau_ms"].append(m["ml_depression_tau"] * 1e3)
            if m["ml_facilitation_amount"] is not None: g["fac_amount"].append(m["ml_facilitation_amount"])
            if m["ml_facilitation_tau"] is not None: g["fac_tau_ms"].append(m["ml_facilitation_tau"] * 1e3)

    summary = {}
    for key, g in sorted(groups.items()):
        row = dict(probed=g["probed"], connected=g["connected"], electrical=g["electrical"],
                   p_conn=(g["connected"] / g["probed"]) if g["probed"] else None)
        for k in ("psp_mV", "psc_pA", "latency_ms", "rise_ms", "decay_ms", "ppr_50hz", "stp_initial_50hz",
                  "stp_induction_50hz", "stp_recovery_250ms", "cv_resting", "release_prob", "n_sites",
                  "dep_amount", "dep_tau_ms", "fac_amount", "fac_tau_ms", "distances_um"):
            m, n = med(g[k])
            row[k + "_median"] = m
            row[k + "_n"] = n
        row["psp_mV_q25"], row["psp_mV_q75"] = quartiles(g["psp_mV"])
        summary[key] = row

    distance = {cls: {str(b): dict(probed=v[0], connected=v[1], p=(v[1] / v[0] if v[0] else None))
                      for b, v in sorted(bins.items())} for cls, bins in dist_bins.items()}

    intrinsic = {}
    if has_intr:
        rows = q(con, """SELECT cell_id, tau, input_resistance, input_resistance_ss, rheobase, ap_threshold_v,
                                sag, adaptation_index, ap_width FROM intrinsic""")
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            c = info.get(r["cell_id"])
            if not c or c["cls"] is None:
                continue
            k = f"{c['layer']}:{c['cls']}"
            for f, scale in (("tau", 1e3), ("input_resistance", 1e-6), ("input_resistance_ss", 1e-6),
                             ("rheobase", 1e12), ("ap_threshold_v", 1e3), ("sag", 1), ("adaptation_index", 1), ("ap_width", 1e3)):
                if r[f] is not None:
                    by[k][f].append(r[f] * scale)
        for k, d in sorted(by.items()):
            intrinsic[k] = {f + ("_ms" if f in ("tau", "ap_width") else "_MOhm" if "resistance" in f else "_pA" if f == "rheobase" else "_mV" if f == "ap_threshold_v" else ""):
                            dict(median=med(v)[0], n=len(v)) for f, v in d.items()}

    out = dict(source=os.path.basename(args.sqlite), species=args.species, test_spike_limit=args.test_spikes,
               n_cells=len(info), n_pairs=len(pairs), groups=summary, distance=distance, intrinsic=intrinsic)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)

    print(f"\n{len(summary)} layer/class groups with probed pairs. Largest:")
    for key, row in sorted(summary.items(), key=lambda kv: -kv[1]["probed"])[:12]:
        psp = row["psp_mV_median"]
        print(f"  {key:16s} probed {row['probed']:5d}  connected {row['connected']:4d}  p={row['p_conn']:.3f}  "
              f"PSP {psp if psp is None else round(psp, 3)} mV (n={row['psp_mV_n']})  "
              f"PPR {row['ppr_50hz_median'] if row['ppr_50hz_median'] is None else round(row['ppr_50hz_median'], 2)}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
