"""Numerical stand-in for a competitive sensory layer (k-winners) on CLINC, before the spiking version was built (Stage 20).
(kept from the session's scratch files for reproducibility; paths may need adjusting)"""
import sys, json, numpy as np
sys.path.insert(0, '/home/claude/synthetic-connectome')
from synth.textcode import encode
D = json.load(open('/home/claude/synthetic-connectome/data/routing_clinc3.json')); L = D["labels"]
NIN = 1000
cache = {}
def enc(m):
    if m not in cache: cache[m] = encode(m, NIN, 0.02)[0]
    return cache[m]
def train_sensory(n_units, k, epochs_msgs, seed=0, lr=0.05):
    rng = np.random.default_rng(seed)
    W = rng.random((n_units, NIN)); W /= W.sum(1, keepdims=True)
    theta = np.zeros(n_units)
    for t in range(epochs_msgs):
        x = enc(D["train"][rng.integers(len(D["train"]))][0]); x = x / max(x.sum(), 1)
        a = W @ x - theta
        win = np.argsort(a)[-k:]
        W[win] += lr * (x - W[win]); W[win] /= W[win].sum(1, keepdims=True)    # Hebbian move toward input, normalised
        theta[win] += 0.002; theta *= 0.999                                       # homeostasis: winners get harder to win
    return W, theta
def feats(W, theta, m, k):
    x = enc(m); x = x / max(x.sum(), 1); a = W @ x - theta; f = np.zeros(len(a)); f[np.argsort(a)[-k:]] = 1; return f
def reward(W, theta, k, trials, sigma=0.05, seed=0, eta=0.05):
    rng = np.random.default_rng(seed + 1); n = W.shape[0]
    V = np.ones((3, n)); rbar = 0
    for t in range(trials):
        m, lab = D["train"][rng.integers(len(D["train"]))]; f = feats(W, theta, m, k); c_true = L.index(lab)
        s = V @ f / k + rng.normal(0, sigma, 3); c = int(np.argmax(s)); R = 1 if c == c_true else -0.5
        V[c] += eta * (R - rbar) * f; rbar += (R - rbar) / 10
    sel = rng.choice(len(D["test"]), 600, replace=False)
    return np.mean([int(np.argmax(V @ feats(W, theta, D["test"][i][0], k)) == L.index(D["test"][i][1])) for i in sel])
for n_units, k in ((100, 5), (300, 10), (300, 3)):
    W, th = train_sensory(n_units, k, 20000)
    print(f"sensory {n_units} units, {k} winners (20k unlabeled msgs): reward test acc after 300/1000/3000 trials:",
          [round(np.mean([reward(W, th, k, T, seed=s) for s in range(3)]), 2) for T in (300, 1000, 3000)], flush=True)
