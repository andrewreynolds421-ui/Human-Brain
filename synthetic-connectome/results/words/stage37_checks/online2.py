import sys, json, argparse, numpy as np
sys.argv = ["x"]; sys.path.insert(0, ".")
import scripts.word_reading as W
from synth.words import combine, L
from sklearn.linear_model import LogisticRegression
exec(open("/tmp/conj.py").read().split("for s in (1, 2, 3, 4):")[0].split("from sklearn")[1].split("\n", 1)[1])
class Raw:                                   # delta rule without input normalisation
    def __init__(s, n, eta): s.W = np.zeros((L, n)); s.b = np.zeros(L); s.eta = eta
    def predict(s, x):
        z = s.W @ x + s.b; z -= z.max(); p = np.exp(z); return p / p.sum()
    def learn(s, x, t):
        e = t - s.predict(x); s.W += s.eta * np.outer(e, x); s.b += s.eta * 0.1 * e
def run_rule(Xp, toks, eta, epochs=1):
    P = Raw(Xp.shape[1], eta); Ev = np.concatenate([W.evidence_of(t, E["train"]) for t in toks])
    for _ in range(epochs):
        for x, e in zip(Xp, Ev): P.learn(x, np.eye(L)[combine(e, P.predict(x), 1.0).argmax()])
    return P
def letters(Pfun, Fv, Ft, sets):
    pv = Pfun(Fv); best = max((W.read(sets["val"], E["train"], pv, b)["letter"], b) for b in (0.5, 0.75, 1.0, 1.5))[1]
    return [W.read(sets[n], E["test"], Pfun(Ft[n]), best)["letter"] for n in ("known", "held")]
for s in (1, 2):
    ra = argparse.Namespace(**json.load(open(f"results/words/s{s}/args.json")))
    words, freqs, tr, ho, sets = W.build_sets(s, False, ra)
    ro, va = W.font_split()
    toks = W.sample_set(np.random.default_rng(777 + s), tr, 20000, ro, freqs[tr], None, words)
    Fp = feats(toks, E["train"], ra.decay); Fv = feats(sets["val"], E["train"], ra.decay)
    Ft = {n: feats(sets[n], E["test"], ra.decay) for n in ("known", "held")}
    yp = np.concatenate([t["let"] for t in toks])
    for k in ("context", "conj+ctx"):
        clf = LogisticRegression(C=0.3, max_iter=4000).fit(Fp[k], yp)
        def pf(F, clf=clf, k=k):
            p = np.zeros((len(F[k]), L)); p[:, clf.classes_] = clf.predict_proba(F[k]); return p
        kn, he = letters(pf, Fv, Ft, sets)
        print(f"s{s} {k:9s} FITTED (true letters, many passes): letters known {kn:.3f} held {he:.3f}", flush=True)
        for eta in (0.1, 0.3, 1.0):
            P = run_rule(Fp[k], toks, eta)
            kn, he = letters(lambda F, P=P, k=k: np.array([P.predict(x) for x in F[k]]), Fv, Ft, sets)
            print(f"s{s} {k:9s} online rule, raw inputs, eta {eta}: letters known {kn:.3f} held {he:.3f}", flush=True)
