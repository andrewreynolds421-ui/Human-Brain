"""Stage 37: where the remaining reading accuracy is - order beyond the context (letter-pair cells), the
prediction rule's input scaling, the amount of reading, and the teacher.  Four networks (word splits of
runs s1-s4), paired on identical test words; the prediction rule is the percept-taught delta rule.
Each network gets a separate practice stream of 20,000 frequency-weighted words in the readout fonts.

    python scripts/analysis/stage37.py            # ~15 min, writes results/words/stage37_learning.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import scripts.word_reading as W                      # noqa: E402
from synth.words import L, Predictor, combine, letter_pairs   # noqa: E402

BETAS = (0.5, 0.75, 1.0, 1.5)


def feats(toks, E, decay):
    g = np.array([c for t in toks for c in W.contexts(W.evidence_of(t, E), decay)])
    return dict(context=g, pairs=np.array([letter_pairs(x) for x in g]))


def norm(X):
    return X / np.maximum(X.sum(1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=20000)
    ap.add_argument("--out", default=os.path.join(W.WD, "stage37_learning.json"))
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    ev = np.load(W.EVID); E = {"train": ev["E_train"], "test": ev["E_test"]}
    R = {}
    for s in (1, 2, 3, 4):
        ra = argparse.Namespace(**json.load(open(os.path.join(W.WD, f"s{s}", "args.json"))))
        words, freqs, tr, ho, sets = W.build_sets(s, False, ra)
        ro, va = W.font_split()
        big = W.sample_set(np.random.default_rng(777 + s), tr, a.tokens, ro, freqs[tr], None, words)
        Fb = feats(big, E["train"], ra.decay); Fv = feats(sets["val"], E["train"], ra.decay)
        Ft = {n: feats(sets[n], E["test"], ra.decay) for n in ("known", "held")}
        Evb = np.concatenate([W.evidence_of(t, E["train"]) for t in big]); yb = np.concatenate([t["let"] for t in big])
        nstep = {nt: sum(len(t["let"]) for t in big[:nt]) for nt in (2000, a.tokens)}
        ctx = lambda F: F["context"]
        both = lambda F: np.hstack([norm(F["context"]), norm(F["pairs"])])

        def best_online(scale, getX, nt):
            best = None
            for eta in (0.03, 0.1, 0.3):
                X = getX(Fb)
                P = Predictor(X.shape[1], eta=eta, scale=scale)
                for x, e in zip(X[:nstep[nt]], Evb[:nstep[nt]]):
                    P.learn(x, np.eye(L)[combine(e, P.predict(x), 1.0).argmax()])
                pf = lambda F, P=P: np.array([P.predict(x) for x in getX(F)])
                pv = pf(Fv)
                acc = max(W.read(sets["val"], E["train"], pv, b)["letter"] for b in BETAS)
                if best is None or acc > best[0]:
                    best = (acc, pf)
            return best[1]

        clf = LogisticRegression(C=0.3, max_iter=4000).fit(Fb["context"], yb)

        def fitted(F):
            p = np.full((len(F["context"]), L), 1e-6); p[:, clf.classes_] = clf.predict_proba(F["context"])
            return p / p.sum(1, keepdims=True)
        rows = {"Stage 36 rule (inputs scaled to sum 1), 2,000 words": best_online("sum", ctx, 2000),
                "unscaled inputs, 2,000 words": best_online("none", ctx, 2000),
                f"unscaled inputs, {a.tokens:,} words": best_online("none", ctx, a.tokens),
                f"unscaled, {a.tokens:,} words, + letter-pair cells": best_online("none", both, a.tokens),
                f"reference: fitted to the true letters, {a.tokens:,} words": fitted}
        for name, pf in rows.items():
            pv = pf(Fv)
            b = max((W.read(sets["val"], E["train"], pv, b)["letter"], b) for b in BETAS)[1]
            r = [W.read(sets[n], E["test"], pf(Ft[n]), b) for n in ("known", "held")]
            R.setdefault(name, []).append([r[0]["letter"], r[1]["letter"], r[0]["word"], r[1]["word"]])
        al = [W.read(sets[n], E["test"], np.full((len(Ft[n]["context"]), L), 1.0 / L), 0) for n in ("known", "held")]
        R.setdefault("letters alone", []).append([al[0]["letter"], al[1]["letter"], al[0]["word"], al[1]["word"]])
        print(f"network {s} done", flush=True)
    base = 100 * np.array(R["Stage 36 rule (inputs scaled to sum 1), 2,000 words"])
    out = {}
    for name, v in R.items():
        v = 100 * np.array(v); d = v - base
        out[name] = dict(letters_known=[round(v[:, 0].mean(), 1), round(v[:, 0].std(ddof=1), 1)],
                         letters_new=[round(v[:, 1].mean(), 1), round(v[:, 1].std(ddof=1), 1)],
                         words_known=round(v[:, 2].mean(), 1), words_new=round(v[:, 3].mean(), 1),
                         vs_stage36_known=[round(d[:, 0].mean(), 1), round(d[:, 0].std(ddof=1), 1)],
                         vs_stage36_new=[round(d[:, 1].mean(), 1), round(d[:, 1].std(ddof=1), 1)],
                         per_network=v.round(2).tolist())
    print(json.dumps(out, indent=1))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
