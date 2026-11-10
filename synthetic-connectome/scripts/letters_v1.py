#!/usr/bin/env python3
"""Stage 38: the self-organising spiking letter layer (synth/sensory.py) trained on V1-like complex cells
(synth/v1.py) instead of pixels, on the family-split letter set (data/letters_fam.npz: test families
never seen in any style).  --input pixels trains the same layer on pixels as the control.

    python scripts/letters_v1.py train --input v1 --n-e 400 --out results/letters_fam/v1_400_s0
    python scripts/letters_v1.py score results/letters_fam/v1_400_s0 results/letters_fam/px_400_s0

train: one or more passes without labels (checkpointed, resumable, stops on collapse), then the frozen
layer's spike counts for a readout subset of training images, all validation and all test images.
score: letter units (logistic readout, C chosen on validation families) on the layer, on the V1 cells
directly, and on both; accuracy on the 60 test families, per letter.
"""
import argparse, json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from synth.v1 import complex_cells, as_rates, sparsify   # noqa: E402

DATASETS = {"fam": "letters_fam", "line": "letters_line"}     # box-scaled (Stage 38) / on a text line (Stage 39)


def paths(data="fam"):
    b = os.path.join(ROOT, "data", DATASETS[data])
    return b + ".npz", b + "_v1.npz"


def inputs(kind, total, data="fam"):
    DATA, V1_CACHE = paths(data)
    d = np.load(DATA)
    if kind in ("v1", "v1sparse"):
        if not os.path.exists(V1_CACHE):
            np.savez_compressed(V1_CACHE, **{f"V_{k}": complex_cells(d[f"X_{k}"] / 255.0) for k in ("train", "val", "test")})
        v = np.load(V1_CACHE)
        f = sparsify if kind == "v1sparse" else (lambda x: x)
        X = {k: as_rates(f(v[f"V_{k}"]), total) for k in ("train", "val", "test")}
    else:                                    # pixels, same total drive (Stage 21 ink normalisation)
        X = {}
        for k in ("train", "val", "test"):
            P = d[f"X_{k}"].reshape(len(d[f"X_{k}"]), -1) / 255.0
            X[k] = np.clip(P * (total / np.maximum(P.sum(1, keepdims=True), 1e-9)), 0, 1)
    return X, {k: d[f"y_{k}"] for k in ("train", "val", "test")}


def train(a):
    from synth.sensory import SensoryLayer
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"))
    X, y = inputs(a.input, a.total, a.data)
    n_in = X["train"].shape[1]
    rng = np.random.default_rng(a.seed)
    n_total = int(a.epochs * len(X["train"]))
    order = np.concatenate([rng.permutation(len(X["train"])) for _ in range(int(np.ceil(a.epochs)))])[:n_total]
    ck = os.path.join(a.out, "layer.npz")
    S = SensoryLayer.load(ck) if os.path.exists(ck) else SensoryLayer(n_in=n_in, n_e=a.n_e, seed=a.seed, dt_ms=a.dt)
    t0 = time.time()
    while S.trained_examples < n_total:
        start = S.trained_examples; stop = min(start + 1000, n_total)
        for k in order[start:stop]:
            S.present(X["train"][k], learn=True)
        W = np.asarray(S.S.w).reshape(n_in, S.n_e); W = W / np.linalg.norm(W, axis=0, keepdims=True)
        sim = float((W.T @ W)[~np.eye(S.n_e, dtype=bool)].mean())
        print(f"trained {S.trained_examples}/{n_total} ({time.time() - t0:.0f} s), cell similarity {sim:.3f}, "
              f"bursts skipped {getattr(S, 'skipped_bursts', 0)}", flush=True)
        if sim > 0.98:
            open(os.path.join(a.out, "COLLAPSED"), "w").write(f"collapsed by {S.trained_examples} images\n")
            raise SystemExit("collapsed; checkpoint not overwritten")
        S.save(ck + ".tmp.npz"); os.replace(ck + ".tmp.npz", ck)
    sub = np.sort(np.random.default_rng(100 + a.seed).choice(len(X["train"]), a.n_readout, replace=False))
    for name, idx in (("train", sub), ("val", np.arange(len(X["val"]))), ("test", np.arange(len(X["test"])))):
        f = os.path.join(a.out, f"counts_{name}_i1.npz")       # thresholds are held during reads (Stage 38)
        C = np.zeros((len(idx), S.n_e), np.int16); done = 0
        if os.path.exists(f):
            dd = np.load(f); done = int(dd["done"]); C[:done] = dd["C"][:done]
        for m in range(done, len(idx)):
            C[m] = S.features(X[name][idx[m]])
            if (m + 1) % 500 == 0 or m + 1 == len(idx):
                np.savez(f + ".tmp.npz", C=C, idx=idx, done=m + 1); os.replace(f + ".tmp.npz", f)
                print(f"recorded {name} {m + 1}/{len(idx)} ({time.time() - t0:.0f} s)", flush=True)
    open(os.path.join(a.out, "DONE_i1"), "w").write("ok\n")
    open(os.path.join(a.out, "DONE"), "w").write("ok\n")


def record(a):
    """re-read a trained layer with lateral inhibition x a.inhib: counts_<set>_i<inhib>.npz"""
    from synth.sensory import SensoryLayer
    ra = argparse.Namespace(**json.load(open(os.path.join(a.out, "args.json"))))
    X, y = inputs(ra.input, ra.total, getattr(ra, 'data', 'fam'))
    S = SensoryLayer.load(os.path.join(a.out, "layer.npz"))
    f0 = os.path.join(a.out, "counts_train.npz")
    sub = np.load(f0 if os.path.exists(f0) else os.path.join(a.out, "counts_train_i1.npz"))["idx"]
    t0 = time.time()
    for name, idx in (("train", sub), ("val", np.arange(len(X["val"]))), ("test", np.arange(len(X["test"])))):
        f = os.path.join(a.out, f"counts_{name}_i{a.inhib:g}.npz")
        C = np.zeros((len(idx), S.n_e), np.int16); done = 0
        if os.path.exists(f):
            dd = np.load(f); done = int(dd["done"]); C[:done] = dd["C"][:done]
        for m in range(done, len(idx)):
            C[m] = S.features(X[name][idx[m]], inhib=a.inhib)
            if (m + 1) % 500 == 0 or m + 1 == len(idx):
                np.savez(f + ".tmp.npz", C=C, idx=idx, done=m + 1); os.replace(f + ".tmp.npz", f)
                print(f"recorded {name} {m + 1}/{len(idx)} at inhibition x{a.inhib:g} ({time.time() - t0:.0f} s)", flush=True)
    open(os.path.join(a.out, f"DONE_i{a.inhib:g}"), "w").write("ok\n")


def readout(Ftr, ytr, Fva, yva, Fte, yte, Cs=(0.03, 0.1, 0.3, 1.0)):
    from sklearn.linear_model import LogisticRegression
    best = None
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=4000).fit(Ftr, ytr)
        acc = m.score(Fva, yva)
        if best is None or acc > best[0]:
            best = (acc, C, m)
    acc, C, m = best
    pred = m.predict(Fte)
    per = {chr(97 + c): round(float(np.mean(pred[yte == c] == c)), 3) for c in range(26)}
    conf = np.zeros((26, 26), int)
    np.add.at(conf, (yte, pred), 1)
    off = [(int(conf[i, j]), chr(97 + i), chr(97 + j)) for i in range(26) for j in range(26) if i != j]
    return dict(C=C, validation=round(float(acc), 4), test_new_families=round(float(np.mean(pred == yte)), 4),
                per_letter=per, top_confusions=[f"{t}->{p}: {n}" for n, t, p in sorted(off, reverse=True)[:8]])


def score(a):
    rows = []
    for run in a.runs:
        ra = json.load(open(os.path.join(run, "args.json")))
        DATA, V1_CACHE = paths(ra.get("data", "fam"))
        v = np.load(V1_CACHE)
        d = np.load(DATA)
        y = {k: d[f"y_{k}"] for k in ("train", "val", "test")}
        suf = f"_i{a.inhib:g}"                     # recorded with frozen thresholds (Stage 38 fix)
        C = {k: np.load(os.path.join(run, f"counts_{k}{suf}.npz")) for k in ("train", "val", "test")}
        idx = C["train"]["idx"]
        L = {k: np.log1p(C[k]["C"].astype(float)) for k in C}
        yy = {"train": y["train"][idx], "val": y["val"], "test": y["test"]}
        V = {"train": v["V_train"][idx], "val": v["V_val"], "test": v["V_test"]}
        res = dict(run=run, inhib=a.inhib, data=ra.get("data", "fam"), input=ra["input"], n_e=ra["n_e"], epochs=ra["epochs"], seed=ra["seed"],
                   cells_firing_test=round(float((C["test"]["C"] > 0).sum(1).mean()), 2),
                   silent_test=round(float((C["test"]["C"].sum(1) == 0).mean()), 4))
        res["letter_units_on_layer"] = readout(L["train"], yy["train"], L["val"], yy["val"], L["test"], yy["test"])
        res["letter_units_on_v1_cells"] = readout(V["train"], yy["train"], V["val"], yy["val"], V["test"], yy["test"])
        both = {k: np.hstack([L[k], V[k]]) for k in L}
        res["letter_units_on_layer_and_v1"] = readout(both["train"], yy["train"], both["val"], yy["val"], both["test"], yy["test"])
        rows.append(res)
        print(json.dumps({k: (v_["test_new_families"] if isinstance(v_, dict) else v_) for k, v_ in res.items()}), flush=True)
    if a.save:
        json.dump(rows, open(a.save, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "record", "score"))
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--input", choices=("v1", "v1sparse", "pixels"), default="v1")
    ap.add_argument("--data", choices=tuple(DATASETS), default="fam")
    ap.add_argument("--n-e", type=int, default=400)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total", type=float, default=50.0, help="total input drive per image (rates sum)")
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--n-readout", type=int, default=10000, help="labelled training images for the letter units")
    ap.add_argument("--inhib", type=float, default=1.0, help="record/score: lateral inhibition factor at readout")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    {"train": train, "record": record, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
