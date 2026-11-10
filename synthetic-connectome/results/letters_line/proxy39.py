import sys, numpy as np
sys.path.insert(0, ".")
import synth.v1 as v1
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
def unit(X): return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
def code(km, X, k=3):
    S = unit(X) @ unit(km.cluster_centers_).T
    out = np.zeros_like(S); top = np.argsort(-S, 1)[:, :k]
    np.put_along_axis(out, top, np.take_along_axis(S, top, 1), 1); return out
for name, f in (("box (Stage 38)", "data/letters_fam.npz"), ("line", "data/letters_line.npz")):
    d = np.load(f)
    V = {k: unit(v1.sparsify(v1.complex_cells(d[f"X_{k}"] / 255.0))) for k in ("train", "test")}
    sub = np.random.default_rng(0).choice(len(V["train"]), 10000, replace=False)
    for seed in (0, 1):
        km = MiniBatchKMeans(400, random_state=seed, n_init=1, batch_size=1024, max_iter=60).fit(V["train"])
        m = LogisticRegression(C=3.0, max_iter=3000).fit(code(km, V["train"][sub]), d["y_train"][sub])
        p = m.predict(code(km, V["test"])); y = d["y_test"]
        per = {chr(97 + c): round(float((p[y == c] == c).mean()), 2) for c in range(26)}
        il = int(((y == 8) & (p == 11)).sum()), int(((y == 11) & (p == 8)).sum())
        print(f"{name:15s} templates seed {seed}: {np.mean(p == y):.3f} | i {per['i']} l {per['l']} j {per['j']} f {per['f']} g {per['g']} | i->l {il[0]}, l->i {il[1]} | hardest {sorted(per.items(), key=lambda x: x[1])[:4]}", flush=True)
