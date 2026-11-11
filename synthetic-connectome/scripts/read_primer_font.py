#!/usr/bin/env python3
"""Stage 44: words teach the letters the book's font.

A book is printed in one font.  While reading, each word recognised with confidence (not "new", and the
winning word unit's posterior at least `conf`) teaches the letter units: the unit's spelling names each
letter, and a delta rule changes FAST weights from the edge cells (added to the letter units' own weights,
reset to zero with every new book).  No labels: the teacher is the model's own word recognition.

Protocol (per network): A - the First Reader read from no word units in ONE never-seen font family, with
the font learning off / on.  B - a second copy in 4 other never-seen fonts, word units frozen, font learning
off / on (starting from zero each time).  Also the letters of each B font (all 52 shapes, alone) read before
and after reading the book in it, and the Second Reader (never read) in a new font.
eta_font and conf are FITTED on validation fonts (book B-style, with the word units of A-off).

    python scripts/read_primer_font.py --seed 1   ->  results/primer_font/s1.json
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capitals as C                                   # noqa: E402
from word_units import Reader, FLOOR                   # noqa: E402
from read_primer import Words, SECOND, read_book       # noqa: E402
from synth.primer import lessons                       # noqa: E402
from synth.words import L, encode                      # noqa: E402

OUT = os.path.join(ROOT, "results", "primer_font")
VAL_FONTS = (5, 7, 15)        # ordinary validation families (family 0, "Digital Numbers", is a seven-segment display face: 28 % letters)


class FontReader(Reader):
    def __init__(self, seed):
        super().__init__(seed)
        self.xn2 = float(np.mean(np.sum(self.V["val"][::50] ** 2, 1)))
        self.reset()

    def reset(self):
        self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b)

    def ev_x(self, X):
        z = X @ (self.W + self.dW).T + self.b + self.db
        z -= z.max(-1, keepdims=True); p = np.exp(z)
        return p / p.sum(-1, keepdims=True)

    def x(self, split, fam, cls, r):
        return self.V[split][C.img_index(fam, cls, r)]

    def teach(self, x, k, eta):
        err = -self.ev_x(x[None])[0]; err[k] += 1.0
        self.dW += (eta / self.xn2) * np.outer(err, x); self.db += (eta / self.xn2) * err

    def letters_alone(self, split, fam):
        cls = np.repeat(np.arange(C.NCLS), C.PER); r = np.tile(np.arange(C.PER), C.NCLS)
        return float(np.mean(self.ev_x(self.x(split, fam, cls, r)).argmax(1) == cls % 26))


def winner(W, ev):
    idx, match, s = W.scores(ev)
    if idx is None:
        return -1, -np.inf, 0.0
    k = int(np.argmax(s)); z = s - s.max()
    return idx[k], float((match[k] - W._self).mean()), float(1.0 / np.exp(z).sum())


def read(R, W, book, rng, split, fam, learn, eta=0.0, conf=0.9):
    R.reset()
    known = set(W.truth); rec = []
    for lesson in book:
        for sent in lesson["sentences"]:
            for tok in sent:
                let = encode(tok.lower()); cls = let + 26 * np.array([c.isupper() for c in tok], int)
                X = R.x(split, fam, cls, rng.integers(0, C.PER, len(let))); ev = R.ev_x(X); w = tok.lower()
                first = w not in known
                u, rel, post = winner(W, ev)
                new = u < 0 or rel < W.thr
                if new and learn:
                    u = W.recruit(ev, w); known.add(w)
                elif not new and learn:
                    W.refresh(u, ev)
                taught = False
                if eta > 0 and not new and post >= conf:
                    for t, k in enumerate(W.M[u].argmax(1)):          # the word unit's spelling names the letters
                        R.teach(X[t], int(k), eta)
                    taught = True
                rec.append(dict(w=w, read=(None if (new and not learn) else W.truth[u]), new=bool(new), first=first,
                                letters_right=float(np.mean(ev.argmax(1) == let)), all_letters=bool(np.all(ev.argmax(1) == let)),
                                taught=taught, lesson_wrong=bool(taught and W.truth[u] != w)))
    return rec


def summary_A(W, rec):
    fs = [r for r in rec if r["first"]]; kn = [r for r in rec if not r["first"]]
    return dict(units=len(W.M), duplicates=sum(1 for i, t in enumerate(W.truth) if t in W.truth[:i]), words_with_a_unit=len(set(W.truth)),
                first_sighting_made_a_unit=float(np.mean([r["new"] for r in fs])),
                known_read_right=float(np.mean([(not r["new"]) and r["read"] == r["w"] for r in kn])),
                known_taken_for_new=float(np.mean([r["new"] for r in kn])),
                letters_right=float(np.mean([r["letters_right"] for r in rec])),
                every_letter_right=float(np.mean([r["all_letters"] for r in rec])))


def summary_B(W, rec):
    has = set(W.truth); k = [r for r in rec if r["w"] in has]; u = [r for r in rec if r["w"] not in has]
    t = [r for r in rec if r["taught"]]
    return dict(known_share=len(k) / len(rec), known_read_right=float(np.mean([r["read"] == r["w"] for r in k])),
                known_called_unknown=float(np.mean([r["new"] for r in k])),
                unknown_flagged=(float(np.mean([r["new"] for r in u])) if u else None),
                letters_right=float(np.mean([r["letters_right"] for r in rec])),
                every_letter_right=float(np.mean([r["all_letters"] for r in k])),
                words_that_taught=len(t) / len(rec), lessons_wrong=(float(np.mean([r["lesson_wrong"] for r in t])) if t else None),
                read_right_by_fifth=[float(np.mean([r["read"] == r["w"] for r in ch if r["w"] in has])) for ch in np.array_split(np.array(rec, dtype=object), 5)])


def avg(ds):
    out = {}
    for k in ds[0]:
        v = [d[k] for d in ds if d[k] is not None]
        out[k] = (np.mean(v, 0).round(4).tolist() if isinstance(v[0], list) else round(float(np.mean(v)), 4)) if v else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eta", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--conf", type=float, nargs="+", default=[0.9, 0.99])
    ap.add_argument("--n-fonts", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    R = FontReader(a.seed)
    book = lessons(); book2 = lessons(SECOND)
    # threshold for "new word": as Stage 43 (validation fonts, true words known)
    Wv = Words(-np.inf, 0.0)
    rv = read_book(R, Wv, book, np.random.default_rng(50 + a.seed), "val", np.arange(R.nf["val"]), learn=True, oracle=True)
    kn = np.array([r["match"] for r in rv if not r["first"]]); nw = np.array([r["match"] for r in rv if r["first"] and np.isfinite(r["match"])])
    thr = max(((np.mean(kn >= c) + np.mean(nw < c)) / 2, c) for c in np.quantile(np.r_[kn, nw], np.linspace(0.01, 0.99, 99)))[1]
    fr = np.random.default_rng(10 + a.seed)
    famA = int(fr.integers(30)); famsB = [int(f) for f in 30 + fr.choice(30, a.n_fonts, replace=False)]; fam2 = int(fr.integers(30))
    res = dict(seed=a.seed, threshold=round(float(thr), 3), font_A=famA, fonts_B=famsB)
    # A without font learning
    W0 = Words(thr, 0.0)
    res["A_off"] = summary_A(W0, read(R, W0, book, np.random.default_rng(a.seed), "test", famA, True))
    # fit eta, conf: validation fonts, second copy with the units of A_off
    best = None
    for eta in a.eta:
        for cf in a.conf:
            sc = np.mean([summary_B(W0, read(R, W0, book, np.random.default_rng(70 + f), "val", f, False, eta, cf))["known_read_right"] for f in VAL_FONTS])
            print("validation", eta, cf, round(float(sc), 4), flush=True)
            if best is None or sc > best[0]:
                best = (sc, eta, cf)
    v_off = float(np.mean([summary_B(W0, read(R, W0, book, np.random.default_rng(70 + f), "val", f, False))["known_read_right"] for f in VAL_FONTS]))
    _, eta, cf = best
    res.update(eta_font=eta, conf=cf, validation=dict(off=round(v_off, 4), on=round(float(best[0]), 4)))
    # A with font learning
    W1 = Words(thr, 0.0)
    res["A_on"] = summary_A(W1, read(R, W1, book, np.random.default_rng(a.seed), "test", famA, True, eta, cf))

    def second_copies(W, e, bk=book, fams=famsB):
        out, la, lb = [], [], []
        for f in fams:
            R.reset(); la.append(R.letters_alone("test", f))
            out.append(summary_B(W, read(R, W, bk, np.random.default_rng(900 + a.seed + f), "test", f, False, e, cf)))
            lb.append(R.letters_alone("test", f))                     # fast weights as left by the reading
        d = avg(out); d["letters_of_the_font_alone_before"] = round(float(np.mean(la)), 4); d["letters_of_the_font_alone_after"] = round(float(np.mean(lb)), 4)
        return d
    res["B_units_off_reading_off"] = second_copies(W0, 0.0)
    res["B_units_off_reading_on"] = second_copies(W0, eta)
    res["B_units_on_reading_on"] = second_copies(W1, eta)
    res["C_second_reader_off"] = second_copies(W0, 0.0, book2, [fam2])
    res["C_second_reader_on"] = second_copies(W1, eta, book2, [fam2])
    json.dump(res, open(os.path.join(OUT, f"s{a.seed}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
