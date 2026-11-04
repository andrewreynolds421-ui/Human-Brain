#!/usr/bin/env python3
"""Extract inter-areal connection strengths (FLNe) and white-matter distances
for a set of macaque association areas from the data bundled with the
multi-area model (Schmidt et al. 2018, github.com/INM-6/multi-area-model):

* FLNe: Markov et al. 2014, Cereb Cortex 24:17-36, raw table (29 injected
  areas x 91 sources); geometric mean over injections, as in the paper.
* distances: estimated white-matter distances between areas (Markov et al.
  2012 / "Thom" distances in the same bundle), mm, macaque.

    python scripts/extract_interareal.py path/to/multi-area-model/multiarea_model/data_multiarea/raw_data
"""
import json, math, os, sys, collections

AREAS = ["46d", "9/46d", "8m", "8B", "24c", "7A", "7m", "STPc"]
DIST_NAME = {"9/46d": "9-46d"}


def main(raw):
    rows = []
    for line in open(os.path.join(raw, "Markov2014_FLN_rawdata.csv")):
        p = [x for x in line.strip().split("\t") if x != ""]
        if len(p) >= 6 and p[0].isdigit():
            rows.append(p)
    g = collections.defaultdict(list)
    for r in rows:
        g[(r[2], r[3])].append(float(r[4]))           # (source, target) -> FLNe per injection
    fln = {t: {s: (math.exp(sum(map(math.log, g[(s, t)])) / len(g[(s, t)])) if (s, t) in g else 0.0)
               for s in AREAS if s != t} for t in AREAS}
    n_inj = {t: len({r[1] for r in rows if r[3] == t}) for t in AREAS}
    lines = [l.rstrip("\n").split("\t") for l in open(os.path.join(raw, "Thom_Distances_MERetal12.csv"))]
    hi = next(i for i, l in enumerate(lines) if "F6" in l)
    cols = {n: j for j, n in enumerate(lines[hi]) if n}
    drow = {l[0]: l for l in lines[hi + 1:] if l and l[0]}
    dist = {a: {b: float(drow[DIST_NAME.get(a, a)][cols[DIST_NAME.get(b, b)]]) for b in AREAS if b != a} for a in AREAS}
    out = dict(areas=AREAS, fln_target_from_source=fln, distance_mm_macaque=dist, injections_per_target=n_inj,
               sources=dict(fln="Markov et al. 2014 Cereb Cortex 24:17-36 (raw table via INM-6 multi-area-model)",
                            distance="white-matter distance estimates, Markov et al. 2012 (via INM-6 multi-area-model)"))
    dst = os.path.join(os.path.dirname(__file__), "..", "results", "interareal_macaque_assoc.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("saved", dst)
    for t in AREAS:
        print(f"{t:>6s} <- total from modelled areas {sum(fln[t].values()):.3f}  (injections {n_inj[t]})")


if __name__ == "__main__":
    main(sys.argv[1])
