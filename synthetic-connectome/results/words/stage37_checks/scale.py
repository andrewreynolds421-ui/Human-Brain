import sys, json, argparse, numpy as np
sys.argv = ["x"]; sys.path.insert(0, ".")
import scripts.word_reading as W
from sklearn.linear_model import LogisticRegression
exec(open("/tmp/conj.py").read().split("for s in (1, 2, 3, 4):")[0].split("from sklearn")[1].split("\n", 1)[1])
from synth.words import load_words, split_words
for s in (1, 2):
    ra = argparse.Namespace(**json.load(open(f"results/words/s{s}/args.json")))
    words, freqs, tr, ho, sets = W.build_sets(s, False, ra)
    ro, va = W.font_split()
    rng = np.random.default_rng(777 + s)
    big = W.sample_set(rng, tr, 40000, ro, freqs[tr], None, words)
    yt = {n: np.concatenate([t["let"] for t in sets[n]]) for n in ("known", "held")}
    Ft = {n: feats(sets[n], E["test"], ra.decay) for n in yt}
    for ntok in (2000, 8000, 40000):
        toks = big[:ntok]; Fp = feats(toks, E["train"], ra.decay); yp = np.concatenate([t["let"] for t in toks])
        out = []
        for k in ("context", "conj+ctx"):
            clf = LogisticRegression(C=0.3, max_iter=4000).fit(Fp[k], yp)
            r = []
            for n in ("known", "held"):
                p = np.zeros((len(yt[n]), 26)); p[:, clf.classes_] = clf.predict_proba(Ft[n][k]); r.append(f"{np.mean(p.argmax(1) == yt[n]):.3f}")
            out.append(f"{k} known {r[0]} held {r[1]}")
        print(f"s{s} {ntok:6d} words read | " + " | ".join(out), flush=True)
