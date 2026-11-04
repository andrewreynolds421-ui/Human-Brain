#!/usr/bin/env python3
"""Train the spiking sensory layer (synth/sensory.py) on unlabeled messages, with
checkpoints (resumes after an interruption), then score it.

Scoring uses labels only to *measure* the frozen layer, never to train it:
an idealised reward learner (noise-free version of the brain's rule) learns
destination from the layer's spike counts, as in the numerical stand-in.

    python scripts/train_sensory.py --n-e 1000 --messages 20000 --out results/sensory/s1000
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synth.sensory import SensoryLayer   # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "routing_clinc3.json")


def reward_score(F_tr, y_tr, F_te, y_te, trials, k, seed=0, eta=0.05, sigma=0.05):
    rng = np.random.default_rng(seed)
    K = int(max(y_tr.max(), y_te.max()) + 1)

    def topk(f):
        g = np.zeros_like(f, float)
        if f.sum() > 0:
            g[np.argsort(f)[-k:]] = 1.0
            g[f == 0] = 0.0
        return g
    V = np.ones((K, F_tr.shape[1])); rbar = 0.0
    for _ in range(trials):
        n = rng.integers(len(y_tr)); f = topk(F_tr[n])
        c = int(np.argmax(V @ f / k + rng.normal(0, sigma, K))); R = 1.0 if c == y_tr[n] else -1.0 / (K - 1)
        V[c] += eta * (R - rbar) * f; rbar += (R - rbar) / 10
    return float(np.mean([np.argmax(V @ topk(f)) == y for f, y in zip(F_te, y_te)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-e", type=int, default=1000)
    ap.add_argument("--messages", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--out", required=True, help="path prefix")
    ap.add_argument("--n-score", type=int, default=3000, help="train messages used to score")
    a = ap.parse_args()
    D = json.load(open(DATA)); labels = D["labels"]
    rng = np.random.default_rng(a.seed)
    order = [D["train"][i][0] for i in np.concatenate([rng.permutation(len(D["train"])) for _ in range(a.messages // len(D["train"]) + 1)])][: a.messages]
    ck = a.out + "_ckpt.npz"
    S = SensoryLayer.load(ck) if os.path.exists(ck) else SensoryLayer(n_in=1000, n_e=a.n_e, seed=a.seed)
    t0 = time.time()
    while S.trained_examples < a.messages:
        start = S.trained_examples
        S.train(order[start: min(start + a.checkpoint_every, a.messages)], log_every=a.checkpoint_every // 2)
        S.save(ck)
        print(f"checkpoint at {S.trained_examples} messages ({time.time() - t0:.0f}s this session)", flush=True)
    S.save(a.out + "_final.npz")
    # score the frozen layer
    tr = [D["train"][i] for i in rng.choice(len(D["train"]), a.n_score, replace=False)]
    te = D["test"]
    F_tr = np.array([S.features(m) for m, _ in tr]); y_tr = np.array([labels.index(l) for _, l in tr])
    F_te = np.array([S.features(m) for m, _ in te]); y_te = np.array([labels.index(l) for _, l in te])
    res = dict(n_e=a.n_e, messages=a.messages, cells_used=int((F_tr.sum(0) > 0).sum()),
               mean_cells_firing=float((F_tr > 0).sum(1).mean()), silent_messages=float((F_tr.sum(1) == 0).mean()))
    for k in (3, 5):
        res[f"reward_acc_k{k}"] = {T: round(np.mean([reward_score(F_tr, y_tr, F_te, y_te, T, k, seed=s) for s in range(3)]), 3)
                                   for T in (300, 1000, 3000)}
    print(json.dumps(res))
    json.dump(res, open(a.out + "_score.json", "w"), indent=1)
    np.savez_compressed(a.out + "_features.npz", F_tr=F_tr, y_tr=y_tr, F_te=F_te, y_te=y_te)


if __name__ == "__main__":
    main()
