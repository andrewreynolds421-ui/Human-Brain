import sys, numpy as np
sys.path.insert(0, ".")
import synth.v1 as v1
from sklearn.linear_model import LogisticRegression
def feats(X, lam, sigma):
    v1.KERNELS = [(v1.gabor(th, 0.0, lam, sigma), v1.gabor(th, np.pi / 2, lam, sigma)) for th in np.arange(8) * np.pi / 8]
    return v1.complex_cells(X / 255.0)
for name, f in (("box (Stage 38)", "data/letters_fam.npz"), ("line", "data/letters_line.npz")):
    d = np.load(f); sub = np.random.default_rng(0).choice(len(d["X_train"]), 10000, replace=False)
    for lam, sig in ((6.0, 2.5), (4.0, 1.7)):
        if name.startswith("box") and lam != 6.0: continue
        V = {k: feats(d[f"X_{k}"][sub] if k == "train" else d[f"X_{k}"], lam, sig) for k in ("train", "test")}
        ytr, yte = d["y_train"][sub], d["y_test"]
        for kind, g in (("dense", lambda x: x), ("sparse", v1.sparsify)):
            m = LogisticRegression(C=0.1, max_iter=3000).fit(g(V["train"]), ytr); p = m.predict(g(V["test"]))
            per = {chr(97 + c): round(float((p[yte == c] == c).mean()), 2) for c in range(26)}
            worst = sorted(per.items(), key=lambda x: x[1])[:5]
            print(f"{name:15s} wavelength {lam:.0f} {kind:6s}: {np.mean(p == yte):.3f} | i {per['i']} l {per['l']} j {per['j']} | hardest {worst}", flush=True)
