#!/usr/bin/env python3
"""Render data/pictures.npz from the four emoji sets (paths in /home/claude/pics_src/paths.json).
    python scripts/make_pictures.py --per 40
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from synth.pictures import DATA, STYLES, concepts, render, features, object_cells   # noqa: E402

SRC = "/home/claude/pics_src"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per", type=int, default=40); a = ap.parse_args()
    paths = json.load(open(os.path.join(SRC, "paths.json"))); names, _ = concepts()
    rng = np.random.default_rng(0); X, c, s = [], [], []
    for si, st in enumerate(STYLES):
        for ci, k in enumerate(names):
            for _ in range(a.per):
                X.append(render(os.path.join(SRC, paths[st][k]), rng)); c.append(ci); s.append(si)
    X = np.array(X)
    np.savez_compressed(DATA, X=X, concept=np.array(c), style=np.array(s), names=np.array(names), styles=np.array(STYLES), F=features(X), O=object_cells(X))
    print(X.shape)


if __name__ == "__main__":
    main()
