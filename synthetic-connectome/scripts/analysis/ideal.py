"""Noise-free stand-in for the brain's reward rule on the synthetic routing messages (Stage 19 reference).
(kept from the session's scratch files for reproducibility; paths may need adjusting)"""
import sys, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from synth.textcode import encode
from routing_experiment import DATA, LABELS
def run(cf, sigma, trials, n=122, seed=0, eta=0.05):
    rng = np.random.default_rng(seed)
    tr = [(encode(m, n, cf)[0], k) for k, l in enumerate(LABELS) for m in DATA[l]["train"]]
    te = [(encode(m, n, cf)[0], k) for k, l in enumerate(LABELS) for m in DATA[l]["test"]]
    W = np.ones((3, n)); rbar = 0
    for t in range(trials):
        x, k = tr[rng.integers(len(tr))]
        s = W @ x; s = s / max(x.sum(), 1) + rng.normal(0, sigma, 3)
        c = int(np.argmax(s)); R = 1 if c == k else -1
        W[c] += eta * (R - rbar) * x; rbar += (R - rbar) / 10
    ok = [int(np.argmax(W @ x / max(x.sum(), 1) + rng.normal(0, sigma, 3)) == k) for x, k in te for _ in range(20)]
    return np.mean(ok)
for sigma in (0.05, 0.2):
    for cf in (0.05, 0.15, 0.3):
        print(f"noise {sigma}, cells/word {cf}: held-out after 300 / 1000 / 3000 trials:",
              [round(np.mean([run(cf, sigma, T, seed=s) for s in range(5)]), 2) for T in (300, 1000, 3000)])
