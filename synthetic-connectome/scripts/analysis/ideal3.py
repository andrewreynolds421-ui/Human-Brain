"""Same stand-in on CLINC routing data + supervised bag-of-words ceiling (Stage 20 reference).
(kept from the session's scratch files for reproducibility; paths may need adjusting)"""
import sys, json, numpy as np
sys.path.insert(0, '/home/claude/synthetic-connectome')
from synth.textcode import encode
D = json.load(open('/home/claude/synthetic-connectome/data/routing_clinc3.json')); L = D["labels"]
def enc(m, n, cf): return encode(m, n, cf)[0]
def run(cf, sigma, trials, n=122, seed=0, eta=0.05):
    rng = np.random.default_rng(seed)
    tr = D["train"]; te = D["test"]
    W = np.ones((3, n)); rbar = 0
    for t in range(trials):
        m, lab = tr[rng.integers(len(tr))]; x = enc(m, n, cf); k = L.index(lab)
        s = W @ x / max(x.sum(), 1) + rng.normal(0, sigma, 3)
        c = int(np.argmax(s)); R = 1 if c == k else -0.5
        W[c] += eta * (R - rbar) * x; rbar += (R - rbar) / 10
    sel = rng.choice(len(te), 600, replace=False)
    return np.mean([int(np.argmax(W @ enc(te[i][0], n, cf) / max(enc(te[i][0], n, cf).sum(), 1)) == L.index(te[i][1])) for i in sel])
for n in (122, 1000):
    print(f"ideal reward learner, {n} cue cells, 5%/word, noise 0.05: test acc after 300/1000/3000 trials:",
          [round(np.mean([run(0.05, 0.05, T, n=n, seed=s) for s in range(3)]), 2) for T in (300, 1000, 3000)], flush=True)
# supervised bag-of-words ceiling (logistic regression) for reference
from collections import Counter
from synth.textcode import words
vocab = {w: i for i, (w, c) in enumerate(Counter(w for m, _ in D["train"] for w in set(words(m))).most_common(3000))}
def bow(m):
    x = np.zeros(len(vocab)); [x.__setitem__(vocab[w], 1) for w in words(m) if w in vocab]; return x
Xtr = np.array([bow(m) for m, _ in D["train"]]); ytr = np.array([L.index(l) for _, l in D["train"]])
Xte = np.array([bow(m) for m, _ in D["test"]]); yte = np.array([L.index(l) for _, l in D["test"]])
W = np.zeros((3, Xtr.shape[1])); b = np.zeros(3)
for ep in range(30):
    P = np.exp(Xtr @ W.T + b); P /= P.sum(1, keepdims=True); G = P.copy(); G[np.arange(len(ytr)), ytr] -= 1
    W -= 0.5 * G.T @ Xtr / len(ytr); b -= 0.5 * G.mean(0)
print("supervised bag-of-words (logistic regression, ceiling for word-only methods):", round(float(np.mean(np.argmax(Xte @ W.T + b, 1) == yte)), 3))
