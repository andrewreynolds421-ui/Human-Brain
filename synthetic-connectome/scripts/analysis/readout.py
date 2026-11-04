"""Score a frozen sensory layer's saved responses three ways (cell vote, nearest letter average, learned letter names).
    python scripts/analysis/readout.py results/letters/l1600v_features.npz
(kept from the session's scratch files for reproducibility; paths may need adjusting)"""
import sys, numpy as np
d = np.load(sys.argv[1]); Fa, ya, Ft, yt = d["Fa"].astype(float), d["ya"], d["Ft"].astype(float), d["yt"]
K = 26
# 1. Diehl & Cook vote (as scored)
mr = np.array([Fa[ya == c].mean(0) for c in range(K)]); assign = mr.argmax(0)
vote = np.array([[f[assign == c].mean() if (assign == c).any() else 0 for c in range(K)] for f in Ft]).argmax(1)
# 2. nearest class-average response (cosine)
def nz(X): return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
cent = nz(mr); near = (nz(Ft) @ cent.T).argmax(1)
# 3. learned letter names: softmax readout on the cells' spike counts
X = nz(Fa); Xt = nz(Ft); W = np.zeros((K, X.shape[1])); b = np.zeros(K)
for ep in range(300):
    Z = X @ W.T + b; P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    P[np.arange(len(ya)), ya] -= 1; W -= 2.0 * (P.T @ X / len(ya) + 1e-4 * W); b -= 2.0 * P.mean(0)
lin = (Xt @ W.T + b).argmax(1)
for name, p in (("cell vote (Diehl & Cook)", vote), ("nearest letter average", near), ("learned letter names", lin)):
    print(f"  {name:26s} {np.mean(p == yt):.1%}")
