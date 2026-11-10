#!/usr/bin/env python3
"""Lowercase letters a-z from 300 text typefaces of the Google Fonts collection (OFL; sans 184, serif 104,
mono 12; one regular font per family, sampled with seed 0 from the 1,124 text families with Latin),
split by FAMILY: 210 train / 30 validation / 60 test families, so test letters are from designs never
seen in any style (Stage 38; the older data/letters.npz split by font file - 21 of its 22 test fonts had
a sibling of the same family in training).  Rendering as scripts/make_letters.py.

    python scripts/make_letters_families.py --out data/letters_fam.npz
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_letters import LETTERS, render, usable   # noqa: E402

GF = "/home/claude/fonts_src/gf/ofl"
SEL = "/home/claude/fonts_src/selected.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-font", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "letters_fam.npz"))
    a = ap.parse_args()
    sel = json.load(open(SEL))
    fams = [s for s in sel if usable(os.path.join(GF, s["dir"], s["file"]))]
    order = np.random.default_rng(0).permutation(len(fams))
    n_te, n_va = 60, 30
    split = {"test": order[:n_te], "val": order[n_te:n_te + n_va], "train": order[n_te + n_va:]}
    rng = np.random.default_rng(a.seed)
    out = {}
    for name, idx in split.items():
        X, y, f = [], [], []
        for fi, k_f in enumerate(idx):
            p = os.path.join(GF, fams[k_f]["dir"], fams[k_f]["file"])
            for k, ch in enumerate(LETTERS):
                for _ in range(a.per_font):
                    img = render(p, ch, rng)
                    if img is not None:
                        X.append(img); y.append(k); f.append(fi)
        out[f"X_{name}"] = np.array(X); out[f"y_{name}"] = np.array(y); out[f"font_{name}"] = np.array(f)
        out[f"families_{name}"] = np.array([fams[k]["family"] for k in idx])
        out[f"category_{name}"] = np.array([fams[k]["category"] for k in idx])
    np.savez_compressed(a.out, **out)
    print(json.dumps({k: (len(v) if k.startswith("X_") else None) for k, v in out.items() if k.startswith("X_")}),
          f"{len(fams)} usable of {len(sel)} families")


if __name__ == "__main__":
    main()
