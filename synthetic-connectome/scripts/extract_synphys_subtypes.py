#!/usr/bin/env python3
"""Split human interneurons in an Allen Synaptic Physiology sqlite into
fast-spiking (FS) and other (non-FS) using intrinsic spike shape, then
summarise connectivity and synaptic properties per subtype.

Standard library only:
    python scripts/extract_synphys_subtypes.py path/to/synphys_r2.1_small.sqlite

Classification: an interneuron (cell_class_nonsynaptic == 'in') is FS when its
action-potential width is below --width-ms (default 0.7 ms, the gap in the
human interneuron spike-width distribution); it is 'other' when broader; and
'unknown' when it has no intrinsic recording. The upstroke/downstroke ratio is
reported per subtype as an independent check (FS cells have fast repolarisation,
so a lower ratio).

Pairs: same "probed" rule as extract_synphys.py (>10 test spikes at the holding
potential of the presynaptic class); synapse amplitudes are type-consistent only.
Output: results/synphys_human_subtypes.json
"""
import argparse
import json
import sqlite3
import statistics as st
from collections import defaultdict


def q(con, sql, params=()):
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def norm_layer(layer):
    if layer is None:
        return None
    s = str(layer).strip().upper().replace("LAYER", "").replace("L", "").strip()
    if s in ("2", "3", "2/3"):
        return "23"
    return s or None


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 3) if xs else None


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return round(max(0, c - h), 4), round(min(1, c + h), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--out", default="results/synphys_human_subtypes.json")
    ap.add_argument("--width-ms", type=float, default=0.7)
    ap.add_argument("--test-spikes", type=int, default=10)
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)

    cells = q(con, """
        SELECT c.id, c.cell_class_nonsynaptic AS cls, c.target_layer, loc.cortical_layer AS layer,
               i.ap_width, i.ap_upstroke_downstroke_ratio AS updown, i.firing_rate_40pa AS rate40,
               i.adaptation_index AS adapt, i.input_resistance AS rin, i.tau, i.rheobase,
               i.ap_threshold_v AS vth
        FROM cell c JOIN experiment e ON e.id = c.experiment_id JOIN slice s ON s.id = e.slice_id
        LEFT JOIN cortical_cell_location loc ON loc.cell_id = c.id
        LEFT JOIN intrinsic i ON i.cell_id = c.id
        WHERE s.species = 'human'""")
    info = {}
    widths_in = []
    for c in cells:
        if c["cls"] not in ("ex", "in"):
            continue
        layer = norm_layer(c["layer"]) or norm_layer(c["target_layer"])
        if c["cls"] == "ex":
            sub = "E"
        elif c["ap_width"] is None:
            sub = "I?"
        else:
            w = c["ap_width"] * 1e3
            widths_in.append(w)
            sub = "FS" if w < a.width_ms else "nFS"
        info[c["id"]] = dict(layer=layer, sub=sub, row=c)

    # --- the split itself ----------------------------------------------------
    hist = defaultdict(int)
    for w in widths_in:
        hist[round(int(w / 0.1) * 0.1, 1)] += 1
    by_layer = defaultdict(lambda: defaultdict(int))
    props = defaultdict(lambda: defaultdict(list))
    for v in info.values():
        if v["sub"] in ("FS", "nFS", "I?"):
            by_layer[v["layer"] or "?"][v["sub"]] += 1
        r = v["row"]
        for k, sc in (("ap_width", 1e3), ("updown", 1), ("rate40", 1), ("adapt", 1),
                      ("rin", 1e-6), ("tau", 1e3), ("rheobase", 1e12)):
            if r[k] is not None:
                props[v["sub"]][k].append(r[k] * sc)
    split = {}
    for L, d in sorted(by_layer.items()):
        known = d["FS"] + d["nFS"]
        split[L] = dict(FS=d["FS"], nFS=d["nFS"], unknown=d["I?"],
                        FS_fraction=round(d["FS"] / known, 3) if known else None,
                        FS_fraction_ci95=wilson(d["FS"], known))
    tot_fs = sum(d["FS"] for d in by_layer.values())
    tot_n = sum(d["nFS"] for d in by_layer.values())
    split["all"] = dict(FS=tot_fs, nFS=tot_n,
                        unknown=sum(d["I?"] for d in by_layer.values()),
                        FS_fraction=round(tot_fs / (tot_fs + tot_n), 3),
                        FS_fraction_ci95=wilson(tot_fs, tot_fs + tot_n))
    intrinsic = {s: {k: dict(median=med(v), n=len(v)) for k, v in d.items()}
                 for s, d in props.items() if s in ("E", "FS", "nFS")}
    # per layer, in the format of extract_synphys.py's "intrinsic" section
    lay = defaultdict(lambda: defaultdict(list))
    for v in info.values():
        if v["sub"] not in ("FS", "nFS"):
            continue
        r = v["row"]
        k = f"{v['layer']}:{v['sub']}"
        for f, src, sc in (("tau_ms", "tau", 1e3), ("input_resistance_MOhm", "rin", 1e-6),
                           ("ap_threshold_v_mV", "vth", 1e3), ("adaptation_index", "adapt", 1)):
            if r[src] is not None:
                lay[k][f].append(r[src] * sc)
    intrinsic_by_layer = {k: {f: dict(median=med(x), n=len(x)) for f, x in d.items()} for k, d in sorted(lay.items())}

    # --- pairs -----------------------------------------------------------------
    pairs = q(con, """SELECT id, pre_cell_id, post_cell_id, has_synapse, n_ex_test_spikes,
                             n_in_test_spikes FROM pair""")
    syn = {s["pair_id"]: s for s in q(con, """SELECT pair_id, synapse_type, psp_amplitude, latency,
                                                     psp_rise_time, psp_decay_tau, psc_decay_tau FROM synapse""")}
    dyn = {d["pair_id"]: d for d in q(con, """SELECT pair_id, qc_pass, paired_pulse_ratio_50hz,
                                                     stp_induction_50hz, stp_recovery_250ms FROM dynamics""")}
    mod = {m["pair_id"]: m for m in q(con, """SELECT pair_id, ml_base_release_probability AS U,
                                                     ml_depression_tau AS dtau, ml_facilitation_amount AS famt,
                                                     ml_facilitation_tau AS ftau FROM synapse_model""")}
    G = defaultdict(lambda: dict(probed=0, connected=0, psp=[], decay=[], ppr=[], ind=[], rec=[],
                                 U=[], dtau=[], famt=[], ftau=[], pscd=[]))
    for p in pairs:
        pre, post = info.get(p["pre_cell_id"]), info.get(p["post_cell_id"])
        if not pre or not post:
            continue
        pre_cls = "ex" if pre["sub"] == "E" else "in"
        n = p["n_ex_test_spikes"] if pre_cls == "ex" else p["n_in_test_spikes"]
        if (n or 0) <= a.test_spikes:
            continue
        pc = "E" if pre["sub"] == "E" else "I"
        qc = "E" if post["sub"] == "E" else "I"
        for key in (f"{pre['sub']}->{post['sub']}", f"L{pre['layer']}:{pre['sub']}->L{post['layer']}:{post['sub']}",
                    f"class:{pc}->class:{qc}"):
            g = G[key]
            g["probed"] += 1
            if not p["has_synapse"]:
                continue
            g["connected"] += 1
            s = syn.get(p["id"])
            if s and s["synapse_type"] in (None, pre_cls):
                if s["psp_amplitude"] is not None: g["psp"].append(s["psp_amplitude"] * 1e3)
                if s["psp_decay_tau"] is not None: g["decay"].append(s["psp_decay_tau"] * 1e3)
                if s["psc_decay_tau"] is not None: g["pscd"].append(s["psc_decay_tau"] * 1e3)
            d = dyn.get(p["id"])
            if d and d["qc_pass"] in (None, 1, True):
                if d["paired_pulse_ratio_50hz"] is not None: g["ppr"].append(d["paired_pulse_ratio_50hz"])
                if d["stp_induction_50hz"] is not None: g["ind"].append(d["stp_induction_50hz"])
                if d["stp_recovery_250ms"] is not None: g["rec"].append(d["stp_recovery_250ms"])
            m = mod.get(p["id"])
            if m:
                if m["U"] is not None: g["U"].append(m["U"])
                if m["dtau"] is not None: g["dtau"].append(m["dtau"] * 1e3)
                if m["famt"] is not None: g["famt"].append(m["famt"])
                if m["ftau"] is not None: g["ftau"].append(m["ftau"] * 1e3)
    groups = {}
    for k, g in sorted(G.items()):
        groups[k] = dict(probed=g["probed"], connected=g["connected"],
                         p=round(g["connected"] / g["probed"], 4) if g["probed"] else None,
                         p_ci95=wilson(g["connected"], g["probed"]),
                         psp_mV=med(g["psp"]), psp_n=len(g["psp"]), psp_decay_ms=med(g["decay"]), psc_decay_ms=med(g["pscd"]), psc_decay_n=len(g["pscd"]),
                         ppr_50hz=med(g["ppr"]), ppr_n=len(g["ppr"]),
                         stp_induction_50hz=med(g["ind"]), stp_recovery_250ms=med(g["rec"]),
                         release_U=med(g["U"]), dep_tau_ms=med(g["dtau"]), fac_amount=med(g["famt"]),
                         fac_tau_ms=med(g["ftau"]),
                         model_n=len(g["U"]))

    out = dict(width_threshold_ms=a.width_ms,
               interneuron_width_hist={str(k): v for k, v in sorted(hist.items())},
               split_by_layer=split, intrinsic_by_subtype=intrinsic,
               intrinsic_by_layer=intrinsic_by_layer, groups=groups)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"interneuron AP width histogram (ms bin: count), threshold {a.width_ms} ms")
    for k, v in sorted(hist.items()):
        print(f"  {k:.1f} {'#' * v} {v}")
    print("\nsplit by layer:")
    for L, d in split.items():
        print(f"  L{L:>4}: FS {d['FS']:3d}  nFS {d['nFS']:3d}  unknown {d['unknown']:3d}  FS frac {d['FS_fraction']}  CI {d['FS_fraction_ci95']}")
    print("\nintrinsic medians:")
    for s, d in intrinsic.items():
        print(f"  {s:4s} " + "  ".join(f"{k}={v['median']}(n{v['n']})" for k, v in d.items()))
    print("\nsubtype groups:")
    for k, g in groups.items():
        if k.startswith("L") or k.startswith("class"):
            continue
        print(f"  {k:10s} probed {g['probed']:5d} conn {g['connected']:3d} p {g['p']} {g['p_ci95']} "
              f"PSP {g['psp_mV']}(n{g['psp_n']}) PPR {g['ppr_50hz']}(n{g['ppr_n']}) "
              f"U {g['release_U']} dtau {g['dep_tau_ms']} (n{g['model_n']})")
    print("\nper-layer groups with any connections:")
    for k, g in groups.items():
        if k.startswith("L") and g["connected"] and ("FS" in k):
            print(f"  {k:22s} probed {g['probed']:4d} conn {g['connected']:2d} p {g['p']} PSP {g['psp_mV']} PPR {g['ppr_50hz']}")


if __name__ == "__main__":
    main()
