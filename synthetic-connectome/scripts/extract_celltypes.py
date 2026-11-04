#!/usr/bin/env python3
"""Summarise human neuron electrophysiology from the Allen Cell Types
Database export (``cell_types_specimen_details.csv``, the "Download" button
on celltypes.brain-map.org/data, or the AllenSDK equivalent).

    python scripts/extract_celltypes.py cell_types_specimen_details.csv

Output: ``results/celltypes_human_summary.json`` with, per (layer, E/I):
median membrane time constant, input resistance, resting potential,
rheobase, spike threshold (if present) and the implied capacitance.
E/I is taken from the dendrite type (spiny = excitatory, aspiny or sparsely
spiny = inhibitory), the convention used by the Allen Institute.

Column names in the export have changed between releases, so columns are
located by keyword rather than exact name; the script prints which columns
it used.
"""
import argparse
import csv
import json
import os
import statistics as st
from collections import defaultdict


def find_col(cols, *keywords, exclude=()):
    for c in cols:
        lc = c.lower()
        if all(k in lc for k in keywords) and not any(e in lc for e in exclude):
            return c
    return None


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="results/celltypes_human_summary.json")
    ap.add_argument("--fs-updown", type=float, default=2.0,
                    help="interneurons with an upstroke/downstroke ratio below this are fast-spiking (F), "
                         "others non-fast-spiking (N); the human aspiny distribution has its gap here")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8-sig")))
    cols = rows[0].keys()
    col = dict(
        species=find_col(cols, "species"),
        layer=find_col(cols, "layer"),
        dendrite=find_col(cols, "dendrite"),
        tau=find_col(cols, "ef__tau") or find_col(cols, "tau"),
        ri=find_col(cols, "ef__ri") or find_col(cols, "input_resistance"),
        vrest=find_col(cols, "vrest") or find_col(cols, "resting"),
        rheobase=find_col(cols, "threshold_i_long") or find_col(cols, "rheobase"),
        vthresh=find_col(cols, "threshold_v_long") or find_col(cols, "threshold_v"),
        sag=find_col(cols, "ef__sag") or find_col(cols, "sag"),
        rate=find_col(cols, "avg_firing_rate"),
        region=find_col(cols, "structure__acronym") or find_col(cols, "region"),
        updown=find_col(cols, "upstroke_downstroke"),
    )
    print("columns used:", {k: v for k, v in col.items() if v})

    by = defaultdict(lambda: defaultdict(list))
    n_human = 0
    for r in rows:
        if col["species"]:
            sp = str(r[col["species"]]).lower()
            if "human" not in sp and "homo" not in sp:
                continue
        n_human += 1
        layer = str(r[col["layer"]]).strip() if col["layer"] else "?"
        layer = layer.upper().replace("LAYER", "").replace("L", "").strip() or "?"
        dend = str(r[col["dendrite"]]).lower() if col["dendrite"] else ""
        cls = "E" if dend.strip() == "spiny" else ("I" if "aspiny" in dend or "sparsely" in dend else None)
        if cls is None:
            continue
        keys = [f"{layer}:{cls}"]
        ud = fnum(r[col["updown"]]) if col["updown"] else None
        if cls == "I" and ud is not None:
            keys.append(f"{layer}:{'F' if ud < args.fs_updown else 'N'}")    # subtype groups, in addition
        for key in keys:
            for f in ("tau", "ri", "vrest", "rheobase", "vthresh", "sag", "rate"):
                if col[f]:
                    v = fnum(r[col[f]])
                    if v is not None:
                        by[key][f].append(v)
            if col["region"]:
                by[key]["regions"].append(str(r[col["region"]]))
    print(f"{n_human} human cells; {sum(len(v['tau']) for v in by.values())} with a membrane time constant")

    out = {}
    for key, d in sorted(by.items()):
        rec = {}
        for f in ("tau", "ri", "vrest", "rheobase", "vthresh", "sag", "rate"):
            if d[f]:
                rec[f] = dict(median=st.median(d[f]), n=len(d[f]))
        if "tau" in rec and "ri" in rec:
            rec["C_m_pF"] = rec["tau"]["median"] / rec["ri"]["median"] * 1e3     # ms / MOhm -> nF -> pF
        if "vthresh" not in rec and all(k in rec for k in ("vrest", "rheobase", "ri")):
            rec["vthresh_est"] = rec["vrest"]["median"] + rec["rheobase"]["median"] * 1e-12 * rec["ri"]["median"] * 1e6 * 1e3
        rec["regions"] = sorted(set(d["regions"]))[:8]
        out[key] = rec
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(source=os.path.basename(args.csv), groups=out), open(args.out, "w"), indent=1)
    for key, rec in out.items():
        tau = rec.get("tau", {}).get("median"); ri = rec.get("ri", {}).get("median")
        print(f"  {key:5s} n={rec.get('tau', {}).get('n', 0):4d}  tau {tau if tau is None else round(tau, 1)} ms  "
              f"Rin {ri if ri is None else round(ri)} MOhm  C {rec.get('C_m_pF') and round(rec['C_m_pF'])} pF  "
              f"Vrest {rec.get('vrest', {}).get('median') and round(rec['vrest']['median'], 1)} mV")
    print("saved", args.out)


if __name__ == "__main__":
    main()
