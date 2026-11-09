#!/usr/bin/env python3
"""Curriculum step 2 (words), input side: the frozen letter layer's response (spike count per cell) to
every image of data/letters.npz, so word sequences can be assembled from real letter-layer responses.
Inputs are scaled to the same total ink as in training (ink 100).  Each image is presented once (the
input spikes are random, so a second presentation would differ slightly).  Resumable; shardable.

    python scripts/letter_features.py --split test  --out results/words/feat_test.npz
    python scripts/letter_features.py --split train --shard 0/2 --out results/words/feat_train0.npz
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synth.sensory import SensoryLayer   # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default=os.path.join(ROOT, "results", "letters", "l1600v_final.npz"))
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ink", type=float, default=100.0)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    d = np.load(os.path.join(ROOT, "data", "letters.npz"))
    X = d[f"X_{a.split}"].reshape(len(d[f"X_{a.split}"]), -1) / 255.0
    X = np.clip(X * (a.ink / np.maximum(X.sum(1, keepdims=True), 1e-9)), 0, 1)
    idx = np.arange(len(X))[k::n]
    S = SensoryLayer.load(a.layer)
    F = np.zeros((len(idx), S.n_e), np.int16); done = 0
    if os.path.exists(a.out):
        old = np.load(a.out); F[:int(old["done"])] = old["F"][:int(old["done"])]; done = int(old["done"])
    t0 = time.time()
    for m in range(done, len(idx)):
        F[m] = S.features(X[idx[m]])
        if (m + 1) % a.save_every == 0 or m + 1 == len(idx):
            np.savez(a.out + ".tmp.npz", F=F, idx=idx, done=m + 1); os.replace(a.out + ".tmp.npz", a.out)
            print(f"{m + 1}/{len(idx)} images, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
