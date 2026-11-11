#!/usr/bin/env python3
"""Stage 43: reading a primer.  The model starts with letter units (Stage 41) and NO word units, and reads
McGuffey's First Reader lesson by lesson (synth/primer.py), each lesson printed in one font family the
letter units have never seen.

New words on the fly: for every word, the word units of that length compete (Stage 42 reading).  If the
best unit's match per letter is below a threshold - "not a word I know" - a new unit is recruited from that
single sighting (no name, no label); otherwise the winner is the word read and its synapses are refreshed
(running average, at most 20 sightings).  A unit's resting level is the log of how often it has won.

Sentences: while reading, the model counts which unit followed which (a Hebbian count between word units,
with a sentence-start unit); the next word is predicted as P(next | previous) and added to the word
units' input with weight beta_w (ASSUMED 1).

Tests: (B) a second copy of the book in other never-seen fonts, units frozen; (C) the Second Reader, never
read: how many of its words are known, are they recognised, are unknown ones flagged, is the next word
predicted.  References: units taught with names (Stage 42), and reading letter by letter.

The threshold is FITTED on validation fonts (book read with the true words known, to collect matches).

    python scripts/read_primer.py --seed 1   ->  results/primer/s1.json
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capitals as C                                   # noqa: E402
from word_units import Reader, FLOOR, letters_with_prediction   # noqa: E402
from synth.primer import lessons, ROOT as _R          # noqa: E402
from synth.words import L, encode                     # noqa: E402

OUT = os.path.join(ROOT, "results", "primer")
SECOND = os.path.join(ROOT, "data", "primer", "mcguffey_second_reader.txt")
START = -1


def show(R, rng, split, fam, tok):
    let = encode(tok.lower()); case = np.array([c.isupper() for c in tok], int)
    return R.evidence(split, fam, let + 26 * case, rng.integers(0, C.PER, len(let))), let


class Words:
    """word units recruited while reading"""

    def __init__(self, thr, beta_w=1.0):
        self.M, self.n, self.truth, self.len = [], [], [], []
        self.by_len, self.thr, self.beta_w = {}, thr, beta_w
        self.follow = {}                               # (previous unit, next unit) -> count
        self.after = {}                                # previous unit -> total

    def scores(self, ev, prev=None):
        idx = self.by_len.get(len(ev))
        if not idx:
            return None, None, None
        Mw = np.stack([self.M[u] for u in idx])
        match = np.log(np.einsum("ntl,tl->nt", Mw, ev) * (1 - FLOOR) + FLOOR)
        # relative match: against what a brand-new word spelled exactly as the letters were read would get.
        # An unsure letter costs a known word little; a letter read with confidence that disagrees costs a lot.
        self._self = np.log(ev.max(1) * (1 - FLOOR) + FLOOR)
        n = np.array([self.n[u] for u in idx], float)
        s = match.sum(1) + np.log(n / max(sum(self.n), 1))
        if prev is not None and self.beta_w > 0:
            s = s + self.beta_w * np.log(self.predict(prev, idx))
        return idx, match, s

    def predict(self, prev, idx, boost=True):
        """how much more likely each unit of idx is after `prev` than in general:
        (lam * P(next | previous) + (1 - lam) * P(next)) / P(next), lam = 0.5 ASSUMED; Hebbian counts.
        boost=False returns the probability itself (for guessing the next word unseen)."""
        tot = self.after.get(prev, 0); N = max(sum(self.n), 1)
        uni = np.array([self.n[u] / N for u in idx])
        bi = np.array([self.follow.get((prev, u), 0) / tot for u in idx]) if tot else uni
        p = 0.5 * bi + 0.5 * uni
        return p / uni if boost else p

    def read(self, ev, prev=None):
        idx, match, s = self.scores(ev, prev)
        if idx is None:
            return -1, -np.inf
        k = int(np.argmax(s))
        return idx[k], float((match[k] - self._self).mean())

    def recruit(self, ev, truth):
        self.M.append(ev.copy()); self.n.append(1); self.truth.append(truth); self.len.append(len(ev))
        self.by_len.setdefault(len(ev), []).append(len(self.M) - 1)
        return len(self.M) - 1

    def refresh(self, u, ev):
        self.n[u] += 1
        self.M[u] += (ev - self.M[u]) / min(self.n[u], 20)

    def link(self, prev, u):
        self.follow[(prev, u)] = self.follow.get((prev, u), 0) + 1
        self.after[prev] = self.after.get(prev, 0) + 1


def read_book(R, W, book, rng, split, fams, learn, oracle=False, use_pred=False, log=None):
    """returns per-token records: (true word, unit truth or None, recruited?, match, first sighting?)"""
    known_truth = {t for t in W.truth}
    rec = []
    for lesson in book:
        fam = int(rng.choice(fams))
        for sent in lesson["sentences"]:
            prev = START
            for tok in sent:
                ev, let = show(R, rng, split, fam, tok); w = tok.lower()
                first = w not in known_truth
                u, match = W.read(ev, prev if use_pred else None)
                if oracle:                              # validation only: bookkeeping by the true word
                    new = first
                    if not new:
                        u = W.truth.index(w)
                else:
                    new = u < 0 or match < W.thr
                if new and learn:
                    u = W.recruit(ev, w); known_truth.add(w)
                elif not new and learn:
                    W.refresh(u, ev)
                rec.append(dict(w=w, read=(None if (new and not learn) else W.truth[u]), new=bool(new), match=match, first=first,
                                all_alone=bool(np.all(ev.argmax(1) == let)), all_pred=bool(np.all(letters_with_prediction(R, ev) == let))))
                if learn and not oracle:
                    W.link(prev, u)
                prev = u if not (new and not learn) else START
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--passes", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    R = Reader(a.seed); rng = np.random.default_rng(a.seed)
    book = lessons(); book2 = lessons(SECOND)
    # threshold: validation fonts, book read with the true words known
    Wv = Words(-np.inf, 0.0)
    rv = read_book(R, Wv, book, np.random.default_rng(50 + a.seed), "val", np.arange(R.nf["val"]), learn=True, oracle=True)
    # (matches of first sightings are against the units present at that moment)
    kn = np.array([r["match"] for r in rv if not r["first"]]); nw = np.array([r["match"] for r in rv if r["first"] and np.isfinite(r["match"])])
    thr = max(((np.mean(kn >= c) + np.mean(nw < c)) / 2, c) for c in np.quantile(np.r_[kn, nw], np.linspace(0.01, 0.99, 99)))[1]
    res = dict(seed=a.seed, threshold=round(float(thr), 3), book=dict(lessons=len(book), words=sum(len(s) for l in book for s in l["sentences"]),
               different_words=len({t.lower() for l in book for s in l["sentences"] for t in s})))
    # A: reading the book from nothing, never-seen fonts (test families 0-29)
    W = Words(thr)
    A = []
    for p in range(a.passes):
        A.append(read_book(R, W, book, rng, "test", np.arange(30), learn=True))
    ra = A[0]
    fs = [r for r in ra if r["first"]]; kn = [r for r in ra if not r["first"]]
    dup = sum(1 for i, t in enumerate(W.truth) if t in W.truth[:i])
    res["A_reading_from_nothing"] = dict(
        units=len(W.M), duplicates=dup, words_with_a_unit=len(set(W.truth)),
        first_sightings=len(fs), first_sighting_made_a_unit=round(float(np.mean([r["new"] for r in fs])), 4),
        first_sighting_taken_for_another_word=round(float(np.mean([not r["new"] for r in fs])), 4),
        known_tokens=len(kn), known_read_right=round(float(np.mean([(not r["new"]) and r["read"] == r["w"] for r in kn])), 4),
        known_taken_for_new=round(float(np.mean([r["new"] for r in kn])), 4),
        known_read_as_other_word=round(float(np.mean([(not r["new"]) and r["read"] != r["w"] for r in kn])), 4),
        by_tenth_of_book=[round(float(np.mean([(r["read"] == r["w"]) for r in ch])), 3) for ch in np.array_split(np.array(ra, dtype=object), 10)])

    # B: a second copy in other never-seen fonts, units frozen
    def test(Wd, bk, use_pred, seed):
        r = read_book(R, Wd, bk, np.random.default_rng(seed), "test", np.arange(30, 60), learn=False, use_pred=use_pred)
        has = set(Wd.truth)
        k = [x for x in r if x["w"] in has]; u = [x for x in r if x["w"] not in has]
        return dict(tokens=len(r), known_share=round(len(k) / len(r), 4),
                    known_read_right=round(float(np.mean([x["read"] == x["w"] for x in k])), 4),
                    known_called_unknown=round(float(np.mean([x["new"] for x in k])), 4),
                    unknown_flagged=(round(float(np.mean([x["new"] for x in u])), 4) if u else None),
                    letter_by_letter=round(float(np.mean([x["all_alone"] for x in k])), 4),
                    letter_by_letter_with_prediction=round(float(np.mean([x["all_pred"] for x in k])), 4))
    res["B_second_copy"] = dict(without_word_prediction=test(W, book, False, 900 + a.seed), with_word_prediction=test(W, book, True, 900 + a.seed))
    # reference: units taught with names (3 showings in training fonts), frequency from the book
    Wn = Words(thr); tr = np.random.default_rng(300 + a.seed)
    cnt = {}
    for l in book:
        for s in l["sentences"]:
            for t in s:
                cnt[t.lower()] = cnt.get(t.lower(), 0) + 1
    for w, c in cnt.items():
        u = Wn.recruit(show(R, tr, "train", int(tr.integers(R.nf["train"])), w)[0], w)
        for _ in range(2):
            Wn.refresh(u, show(R, tr, "train", int(tr.integers(R.nf["train"])), w)[0])
        Wn.n[u] = c
    res["B_second_copy"]["reference_units_taught_with_names"] = test(Wn, book, False, 900 + a.seed)
    # C: the Second Reader, never read
    res["C_second_reader"] = dict(lessons=len(book2), without_word_prediction=test(W, book2, False, 950 + a.seed),
                                  with_word_prediction=test(W, book2, True, 950 + a.seed))
    def guess_next(Wd, bk):
        idx_of = {}
        for u, t in enumerate(Wd.truth):
            idx_of.setdefault(t, u)
        allu = list(range(len(Wd.M))); hit = base = n = 0
        top = int(np.argmax(Wd.n))
        for l in bk:
            for s_ in l["sentences"]:
                prev = START
                for tok in s_:
                    u = idx_of.get(tok.lower())
                    if u is not None and prev is not None:
                        n += 1; hit += int(Wd.truth[allu[int(np.argmax(Wd.predict(prev, allu, boost=False)))]] == tok.lower()); base += int(u == top)
                    prev = u
        return dict(cases=n, right=round(hit / max(n, 1), 4), most_common_word_baseline=round(base / max(n, 1), 4))
    res["C_second_reader"]["next_word_guessed"] = guess_next(W, book2)
    # D: go on reading - Second Reader lessons 1-45 (learning on the fly), then its last 20 lessons as the test
    last = book2[45:]
    res["D_before_reading_on"] = dict(last_20_lessons=test(W, last, True, 970 + a.seed), next_word_guessed=guess_next(W, last))
    read_book(R, W, book2[:45], rng, "test", np.arange(30), learn=True)
    res["D_after_reading_on"] = dict(units=len(W.M), words_with_a_unit=len(set(W.truth)), last_20_lessons=test(W, last, True, 970 + a.seed),
                                     last_20_without_word_prediction=test(W, last, False, 970 + a.seed), next_word_guessed=guess_next(W, last))
    json.dump(res, open(os.path.join(OUT, f"s{a.seed}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
