import sys, json, argparse, numpy as np
sys.argv = ["x"]; sys.path.insert(0, ".")
import scripts.word_reading as W
from sklearn.linear_model import LogisticRegression
ev = np.load(W.EVID); E = {"train": ev["E_train"], "test": ev["E_test"]}
def feats(toks, Ex, decay):
    out = {"context": [], "conj": [], "conj+ctx": []}
    for t in toks:
        for g in W.contexts(W.evidence_of(t, Ex), decay):
            cur, tr, st = g[:26], g[26:52], g[52]
            prev = np.r_[tr, st]                       # earlier letters + start
            cj = np.outer(np.r_[cur, st], prev).ravel()  # current letter (or start) x earlier letters
            out["context"].append(g); out["conj"].append(cj); out["conj+ctx"].append(np.r_[g, cj])
    return {k: np.array(v) for k, v in out.items()}
for s in (1, 2, 3, 4):
    ra = argparse.Namespace(**json.load(open(f"results/words/s{s}/args.json")))
    _, _, _, _, sets = W.build_sets(s, False, ra)
    y = {n: np.concatenate([t["let"] for t in sets[n]]) for n in ("pred", "known", "held")}
    F = {n: feats(sets[n], E["train" if n == "pred" else "test"], ra.decay) for n in y}
    line = []
    for k in ("context", "conj", "conj+ctx"):
        for C in (0.3,):
            clf = LogisticRegression(C=C, max_iter=4000).fit(F["pred"][k], y["pred"])
            r = []
            for n in ("known", "held"):
                p = np.zeros((len(y[n]), 26)); p[:, clf.classes_] = clf.predict_proba(F[n][k])
                r.append(f"{np.mean(p.argmax(1) == y[n]):.3f}/{-np.mean(np.log2(np.maximum(p[np.arange(len(p)), y[n]], 1e-12))):.2f}b")
            line.append(f"{k}: known {r[0]} held {r[1]}")
    print(f"s{s} | " + " | ".join(line), flush=True)
