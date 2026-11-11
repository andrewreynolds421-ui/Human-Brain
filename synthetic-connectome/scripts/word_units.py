#!/usr/bin/env python3
"""Stage 42: common words as units.

Letter units (lowercase taught with labels; capitals learned from 100 fonts of alphabet-book pairs with
replay, Stage 41; results/capitals/units_s<seed>.npz) read the edge cells.  One word unit per taught word.

Teaching: a word is shown (lowercase, a random training font each time) and named; its unit strengthens
synapses from the letter units active at each letter position - the average of what the letter units
reported over the showings (Hebbian; ASSUMED position slots counted from the first letter, and word length
known).  How often a word is met follows its measured frequency (wordfreq); that sets the unit's resting
level (log frequency).

Reading: every word unit of the right length adds up, over positions, the log of the match between its
synapses and the letter units' evidence, plus its resting level; the best unit is the word read
(interactive activation, McClelland & Rumelhart 1981, reduced to one bottom-up pass and a winner).
"Not a word I know": the winner's match per letter is below a threshold (FITTED on validation fonts).

Tests (60 never-seen font families; lowercase / Capitalised / ALL CAPS): whole-word recognition by
vocabulary size and number of showings; letters corrected by the word (capital I); unknown words.

    python scripts/word_units.py --seed 1   ->  results/word_units/s1.json
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capitals as C                                              # noqa: E402
from synth.words import L, Predictor, combine, encode, load_words   # noqa: E402

OUT = os.path.join(ROOT, "results", "word_units")
FLOOR = 0.01                 # ASSUMED: a word unit's match with a letter never drops below this (lapse)


class Reader:
    def __init__(self, seed):
        u = np.load(os.path.join(C.OUT, f"units_s{seed}.npz"))
        V, self.y = C.load()
        self.V = {k: ((V[k] - u["mu"]) / u["sd"]).astype(np.float32) for k in V}
        self.W, self.b = u["W"], u["b"]
        self.P = Predictor(4 * L + 2, scale="none"); self.P.W, self.P.b = u["P_W"], u["P_b"]; self.beta = float(u["beta"])
        self.nf = {k: len(self.V[k]) // (C.NCLS * C.PER) for k in self.V}

    def evidence(self, split, fam, cls, r):
        z = self.V[split][C.img_index(fam, cls, r)] @ self.W.T + self.b
        z -= z.max(-1, keepdims=True); p = np.exp(z)
        return p / p.sum(-1, keepdims=True)


def show(R, rng, split, word, style):
    let = encode(word); T = len(let)
    case = np.zeros(T, int)
    if style == "Capital":
        case[0] = 1
    elif style == "ALLCAPS":
        case[:] = 1
    return R.evidence(split, int(rng.integers(R.nf[split])), let + 26 * case, rng.integers(0, C.PER, T)), let, case


class Lexicon:
    """word units; M[w] (T x 26) = synapses from the letter units at each position"""

    def __init__(self, words, freqs):
        self.words, self.logf = list(words), np.log(np.asarray(freqs) / np.sum(freqs))
        self.len = np.array([len(w) for w in words])
        self.M = [np.zeros((len(w), L)) for w in words]; self.n = np.zeros(len(words))
        self.by_len = {}

    def teach(self, w, ev):
        self.n[w] += 1
        self.M[w] += (ev - self.M[w]) / self.n[w]                 # running average of the letter units' report

    def exact(self):
        for w, word in enumerate(self.words):
            self.M[w] = np.eye(L)[encode(word)]; self.n[w] = 1

    def finish(self):
        for T in np.unique(self.len):
            idx = np.nonzero(self.len == T)[0]
            self.by_len[int(T)] = (idx, np.stack([self.M[w] for w in idx]))       # (n, T, 26)

    def read(self, ev):
        """returns (word index or -1, match per letter of the winner, posterior of the winner)"""
        T = len(ev)
        if T not in self.by_len:
            return -1, -np.inf, 0.0
        idx, Mw = self.by_len[T]
        match = np.log(np.einsum("ntl,tl->nt", Mw, ev) * (1 - FLOOR) + FLOOR)
        s = match.sum(1) + self.logf[idx]
        k = int(np.argmax(s)); z = s - s.max(); post = np.exp(z) / np.exp(z).sum()
        return int(idx[k]), float(match[k].mean()), float(post[k])


def letters_with_prediction(R, ev):
    return np.array([combine(ev[t], R.P.predict(C.context2(ev, t)), R.beta).argmax() for t in range(len(ev))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--vocab", type=int, nargs="+", default=[100, 500, 2000, 5000])
    ap.add_argument("--showings", type=int, nargs="+", default=[1, 3, 10])
    ap.add_argument("--n-test", type=int, default=1500)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    R = Reader(a.seed)
    words, freqs = load_words(6000)
    unknown = np.arange(5000, 6000)                               # never taught: the next 1,000 words by frequency
    rng = np.random.default_rng(a.seed)
    res = dict(seed=a.seed, vocab={})
    for N in a.vocab:
        ids = np.arange(N); p = freqs[:N] / freqs[:N].sum()
        row = {}
        for k in a.showings + ["exact"]:
            lex = Lexicon([words[i] for i in ids], freqs[:N])
            if k == "exact":
                lex.exact()                                       # reference: units given the spelling itself
            else:
                tr = np.random.default_rng(100 + a.seed)
                for w in ids:
                    for _ in range(k):
                        lex.teach(w, show(R, tr, "train", words[w], "lower")[0])
            lex.finish()
            # threshold for "not a word I know": validation fonts, known vs unknown words, lowercase
            vr = np.random.default_rng(200 + a.seed)
            mk = [lex.read(show(R, vr, "val", words[w], "lower")[0])[1] for w in vr.choice(ids, 400, p=p)]
            mu_ = [lex.read(show(R, vr, "val", words[w], "lower")[0])[1] for w in vr.choice(unknown, 400)]
            cand = np.sort(np.r_[mk, mu_])
            thr = max(((np.mean(np.array(mk) >= c) + np.mean(np.array(mu_) < c)) / 2, c) for c in cand)[1]
            out = dict(threshold=round(float(thr), 3))
            for style in ("lower", "Capital", "ALLCAPS"):
                tr2 = np.random.default_rng(300 + a.seed)
                st = dict(word=[], word_types=[], letters_alone=[], letters_pred=[], letters_word=[], all_alone=[], all_pred=[],
                          capI_alone=[], capI_word=[], known_flagged_unknown=[])
                for mode, sample in (("tokens", tr2.choice(ids, a.n_test, p=p)), ("types", tr2.choice(ids, min(a.n_test, 3 * N)))):
                    for w in sample:
                        ev, let, case = show(R, tr2, "test", words[w], style)
                        g, match, post = lex.read(ev)
                        if mode == "types":
                            st["word_types"].append(g == w); continue
                        st["word"].append(g == w)
                        la = ev.argmax(1); lp = letters_with_prediction(R, ev); lw = encode(lex.words[g]) if g >= 0 else la
                        st["letters_alone"].append(np.mean(la == let)); st["letters_pred"].append(np.mean(lp == let)); st["letters_word"].append(np.mean(lw == let))
                        st["all_alone"].append(np.all(la == let)); st["all_pred"].append(np.all(lp == let))
                        st["known_flagged_unknown"].append(match < thr)
                        for t in np.nonzero((let == 8) & (case == 1))[0]:
                            st["capI_alone"].append(la[t] == 8); st["capI_word"].append(lw[t] == 8)
                out[style] = {k_: (round(float(np.mean(v)), 4) if len(v) else None) for k_, v in st.items()}
                out[style]["capI_n"] = len(st["capI_alone"])
            ur = np.random.default_rng(400 + a.seed)
            uk = [lex.read(show(R, ur, "test", words[w], "lower")[0]) for w in ur.choice(unknown, 600)]
            out["unknown_words_flagged"] = round(float(np.mean([m < thr for _, m, _ in uk])), 4)
            row[str(k)] = out
            print(N, k, {s_: (out[s_]["word"], out[s_]["all_alone"], out[s_]["all_pred"]) for s_ in ("lower", "Capital", "ALLCAPS")},
                  "unknown flagged", out["unknown_words_flagged"], "known flagged", out["lower"]["known_flagged_unknown"], flush=True)
        res["vocab"][str(N)] = row
    json.dump(res, open(os.path.join(OUT, f"s{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
