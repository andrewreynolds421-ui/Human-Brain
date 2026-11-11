import numpy as np, sys
sys.path.insert(0, ".")
import synth.v1 as v1
from sklearn.linear_model import LogisticRegression
d = np.load("data/letters_line48.npz"); v = np.load("data/letters_line48_v1.npz")
sub = np.random.default_rng(0).choice(len(d["y_train"]), 10000, replace=False); y = d["y_test"]
for kind, g in (("dense", lambda x: x), ("sparse", v1.sparsify)):
    m = LogisticRegression(C=0.1, max_iter=3000).fit(g(v["V_train"][sub]), d["y_train"][sub]); p = m.predict(g(v["V_test"]))
    per = {chr(97 + c): round(float((p[y == c] == c).mean()), 2) for c in range(26)}
    print(f"line 48x40 {kind}: {np.mean(p == y):.3f} | i {per['i']} l {per['l']} | hardest {sorted(per.items(), key=lambda x: x[1])[:5]}", flush=True)
