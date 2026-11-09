import sys, json, argparse, numpy as np
sys.argv = ["x"]; sys.path.insert(0, ".")
import scripts.word_reading as W
from synth.words import combine, L
from sklearn.linear_model import LogisticRegression
exec(open("/tmp/online2.py").read().split("for s in (1, 2):")[0].split("from sklearn.linear_model import LogisticRegression\n")[1])
class Anneal(Raw):
    def __init__(s, n, eta, tau): super().__init__(n, eta); s.eta0, s.tau, s.t = eta, tau, 0
    def learn(s, x, t):
        s.eta = s.eta0 / (1 + s.t / s.tau); s.t += 1; super().learn(x, t)
R = {}
for s in (1, 2, 3, 4):
    ra = argparse.Namespace(**json.load(open(f"results/words/s{s}/args.json")))
    words, freqs, tr, ho, sets = W.build_sets(s, False, ra)
    ro, va = W.font_split()
    toks = W.sample_set(np.random.default_rng(777 + s), tr, 20000, ro, freqs[tr], None, words)
    k = "context"
    Fp = feats(toks, E["train"], ra.decay)[k]; Fv = feats(sets["val"], E["train"], ra.decay)
    Ft = {n: feats(sets[n], E["test"], ra.decay) for n in ("known", "held")}
    Ev = np.concatenate([W.evidence_of(t, E["train"]) for t in toks])
    ytrue = np.concatenate([t["let"] for t in toks]); ywin = Ev.argmax(1)
    def fitted(y):
        clf = LogisticRegression(C=0.3, max_iter=4000).fit(Fp, y)
        def pf(F):
            p = np.full((len(F[k]), L), 1e-6); p[:, clf.classes_] = clf.predict_proba(F[k]); return p / p.sum(1, keepdims=True)
        return pf
    def online(P):
        for x, e in zip(Fp, Ev): P.learn(x, np.eye(L)[combine(e, P.predict(x), 1.0).argmax()])
        return lambda F: np.array([P.predict(x) for x in F[k]])
    rows = {"fitted, true letters": fitted(ytrue), "fitted, winner of evidence": fitted(ywin),
            "online, fixed rate 0.1": online(Raw(Fp.shape[1], 0.1)),
            "online, settling rate 1/(1+t/5000)": online(Anneal(Fp.shape[1], 0.5, 5000.0)),
            "online, settling rate 1/(1+t/20000)": online(Anneal(Fp.shape[1], 0.5, 20000.0))}
    for name, pf in rows.items():
        R.setdefault(name, []).append(letters(pf, Fv, Ft, sets))
    print("done", s, flush=True)
for name, v in R.items():
    v = 100 * np.array(v); print(f"{name:38s} known {v[:,0].mean():.1f} ± {v[:,0].std(ddof=1):.1f}  held {v[:,1].mean():.1f} ± {v[:,1].std(ddof=1):.1f}")
