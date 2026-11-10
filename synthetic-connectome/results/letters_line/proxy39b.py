import sys, numpy as np
sys.path.insert(0, ".")
exec(open("/tmp/proxy39.py").read().split("for name, f in")[0])
d = np.load("data/letters_line.npz")
V = {k: unit(v1.sparsify(v1.complex_cells(d[f"X_{k}"] / 255.0))) for k in ("train", "val", "test")}
np.savez_compressed("data/letters_line_v1.npz", **{f"V_{k}": v1.complex_cells(d[f"X_{k}"] / 255.0) for k in ("train", "val", "test")})
sub = np.random.default_rng(0).choice(len(V["train"]), 10000, replace=False); y = d["y_test"]
for n in (400, 1600):
    km = MiniBatchKMeans(n, random_state=0, n_init=1, batch_size=2048, max_iter=60).fit(V["train"])
    for k in (3, 10):
        m = LogisticRegression(C=3.0, max_iter=3000).fit(code(km, V["train"][sub], k), d["y_train"][sub]); p = m.predict(code(km, V["test"], k))
        print(f"line: {n} templates, top {k} respond: {np.mean(p == y):.3f} | i {(p[y == 8] == 8).mean():.2f} l {(p[y == 11] == 11).mean():.2f}", flush=True)
