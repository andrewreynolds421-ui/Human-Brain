#!/usr/bin/env python3
"""Curriculum step 1: letters.  The spiking sensory layer (synth/sensory.py, Diehl & Cook 2015)
learns printed letters from data/letters.npz with no labels (checkpointed, resumable).
Scoring as in Diehl & Cook: each cell is assigned the letter it responds to most (using
training images), and a test image is read as the letter whose cells respond most.
Test images come from fonts never seen in training.  Chance = 1/26.

    python scripts/train_letters.py --n-e 400 --epochs 1 --out results/letters/l400
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synth.sensory import SensoryLayer   # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "letters.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-e", type=int, default=400)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--n-assign", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ink", type=float, default=100.0, help="total input intensity per image (0 = raw)")
    ap.add_argument("--dt", type=float, default=0.5, help="time step (ms)")
    ap.add_argument("--data", default=DATA, help="training images (test images always from data/letters.npz)")
    ap.add_argument("--noise", type=float, default=0.0, help="membrane noise SD (mV)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    d = dict(np.load(DATA))
    if a.data != DATA:
        dt_ = np.load(a.data); d["X_train"], d["y_train"] = dt_["X_train"], dt_["y_train"]
    def norm(X):     # same total ink per image (as MNIST, ~100), like a retina adapting to brightness
        X = X.reshape(len(X), -1) / 255.0
        return np.clip(X * (a.ink / np.maximum(X.sum(1, keepdims=True), 1e-9)), 0, 1)
    Xtr = norm(d["X_train"]); ytr = d["y_train"]
    Xte = norm(d["X_test"]); yte = d["y_test"]
    rng = np.random.default_rng(a.seed)
    n_total = int(a.epochs * len(Xtr))
    order = np.concatenate([rng.permutation(len(Xtr)) for _ in range(int(np.ceil(a.epochs)))])[:n_total]
    ck = a.out + "_ckpt.npz"
    S = SensoryLayer.load(ck) if os.path.exists(ck) else SensoryLayer(n_in=784, n_e=a.n_e, seed=a.seed, dt_ms=a.dt, noise_mV=a.noise)
    t0 = time.time()
    while S.trained_examples < n_total:
        start = S.trained_examples; stop = min(start + 1000, n_total)
        for k in order[start:stop]:
            S.present(Xtr[k], learn=True)
        Wc = np.asarray(S.S.w).reshape(784, S.n_e); Wc = Wc / np.linalg.norm(Wc, axis=0, keepdims=True)
        ic = np.random.default_rng(0).choice(S.n_e, min(S.n_e, 300), replace=False); Cc = Wc[:, ic].T @ Wc[:, ic]
        simc = float(Cc[~np.eye(len(ic), dtype=bool)].mean())
        print(f"trained {S.trained_examples}/{n_total} images ({time.time() - t0:.0f}s this session), cell similarity "
              f"{simc:.3f}, bursts skipped {getattr(S, 'skipped_bursts', 0)}", flush=True)
        if simc > 0.98:                      # never checkpoint a collapsed layer
            open(a.out + "_COLLAPSED", "w").write(f"collapsed by {S.trained_examples} images\n")
            raise SystemExit("cells have collapsed onto one template; stopping (checkpoint not overwritten)")
        S.save(ck)
    S.save(a.out + "_final.npz")
    ia = rng.choice(len(Xtr), a.n_assign, replace=False); it = rng.choice(len(Xte), a.n_test, replace=False)
    Fa = np.array([S.features(Xtr[k]) for k in ia]); Ft = np.array([S.features(Xte[k]) for k in it])
    mean_resp = np.array([Fa[ytr[ia] == c].mean(0) if (ytr[ia] == c).any() else np.zeros(S.n_e) for c in range(26)])
    assign = mean_resp.argmax(0)
    votes = np.array([[f[assign == c].mean() if (assign == c).any() else 0.0 for c in range(26)] for f in Ft])
    pred = votes.argmax(1)
    acc = float(np.mean(pred == yte[it]))
    per = {chr(97 + c): round(float(np.mean(pred[yte[it] == c] == c)), 2) for c in range(26)}
    Wn = np.asarray(S.S.w).reshape(784, S.n_e); Wn = Wn / np.linalg.norm(Wn, axis=0, keepdims=True)
    sim = float((Wn.T @ Wn)[~np.eye(S.n_e, dtype=bool)].mean())
    res = dict(n_e=a.n_e, images=n_total, ink=a.ink, cell_similarity=sim, test_accuracy_unseen_fonts=acc, chance=1 / 26,
               cells_per_letter=np.bincount(assign, minlength=26).tolist(), per_letter=per,
               silent_test_images=float((Ft.sum(1) == 0).mean()), mean_cells_firing=float((Ft > 0).sum(1).mean()))
    print(json.dumps(res))
    json.dump(res, open(a.out + "_score.json", "w"), indent=1)
    np.savez_compressed(a.out + "_features.npz", Fa=Fa, ya=ytr[ia], Ft=Ft, yt=yte[it])


if __name__ == "__main__":
    main()
