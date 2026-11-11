import sys, numpy as np, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import capitals as C
from synth.v1 import sparsify
from sklearn.cluster import MiniBatchKMeans
V, y = C.load()
def unit(X): return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
F = {k: unit(sparsify(V[k])).astype(np.float32) for k in V}
def code(cent, X, k):
    S = X @ cent.T; idx = np.argpartition(-S, k, 1)[:, :k]; val = np.take_along_axis(S, idx, 1)
    return idx, val
def votes_fit(idx, val, lab, n, power):
    N = np.full((n, 26), 0.01)
    np.add.at(N, (idx.ravel(), np.repeat(lab, idx.shape[1])), (val.ravel() ** power))
    return N / N.sum(1, keepdims=True)
def read(P, idx, val, power):
    return np.einsum("nk,nkl->nl", val ** power, P[idx])
for n in (800, 3200):
    t0 = time.time(); km = MiniBatchKMeans(n, random_state=0, n_init=1, batch_size=4096, max_iter=40).fit(F["train"]); cent = unit(km.cluster_centers_).astype(np.float32)
    for k in (1, 5, 20):
        itr, vtr = code(cent, F["train"], k); ite, vte = code(cent, F["test"], k)
        for power in (1, 8):
            P = votes_fit(itr, vtr, y["train"] % 26, n, power)
            pred = read(P, ite, vte, power).argmax(1); ok = pred == y["test"] % 26
            print(f"{n} detectors, top {k:2d}, sharpness {power}: lowercase {ok[y['test'] < 26].mean():.3f} capitals {ok[y['test'] >= 26].mean():.3f}", flush=True)
    print(f"  ({time.time() - t0:.0f} s)", flush=True)
