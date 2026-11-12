#!/usr/bin/env python3
"""Stage 45: pictures with words, as in children's books.

The model reads with letter units (Stage 41) and learns words on the fly with font learning (Stages 43-44);
it now also SEES the pictures.  Picture units are recruited on the fly like word units: a picture that
matches no picture unit well enough (cosine of the standardised picture cells, synth/pictures.py, below a
threshold FITTED on another drawing style) gets a new unit; otherwise the best unit wins and is refreshed.
No labels on either side.

Word and picture are linked by counting, page by page, which picture units and which word units were
active together (Hebbian; each unit counted once per page).  The link strength is how much more often
the word comes with the picture than without it (delta P = P(word | picture) - P(word | no picture)).

Books (each network: one drawing style for the pictures, one never-seen font for the text):
  wordbook - a first-words picture book: one page per thing (34), the picture with its word printed below
  primer   - McGuffey's First Reader: each lesson's picture (the things named in its description; ASSUMED
             mapping by keyword, data/pictures/concepts.json) shown as separate objects, with its sentences
  both     - the word book, then the primer
  control  - the primer with the lesson pictures shuffled between lessons
Each book is read `--passes` times (ASSUMED 3, as children re-read).

Tests (new renderings of the pictures):
  naming       - a picture is shown; the word unit most strongly linked to its picture unit is "said"
  choosing     - a word is printed (another never-seen font); which of 4 pictures goes with it (chance 25 %)
  other style  - naming pictures drawn in the three styles the book did not use
  help reading - words printed under their picture in a hard font: read with / without the picture's links

    python scripts/picture_book.py --seed 1   ->  results/picture_book/s1.json
"""
import argparse, json, os, re, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capitals as C                                         # noqa: E402
from read_primer import Words                                # noqa: E402
from read_primer_font import FontReader, winner              # noqa: E402
from synth.primer import lessons                             # noqa: E402
from synth.pictures import DATA, STYLES, concepts            # noqa: E402
from synth.words import encode                               # noqa: E402

OUT = os.path.join(ROOT, "results", "picture_book")
BETA_PIC = 3.0              # ASSUMED: weight of the picture's links when reading the word below it
WORD_THR = -0.76            # Stage 43 threshold for "new word" (network 1; refitted per network below)


class Pictures:
    def __init__(self, thr):
        self.M, self.n, self.truth, self.thr = [], [], [], thr

    def see(self, f, truth=None, learn=True):
        f = f / (np.linalg.norm(f) + 1e-9)
        if self.M:
            sims = np.array(self.M) @ f; u = int(np.argmax(sims))
            if sims[u] >= self.thr or not learn:
                if learn:
                    self.n[u] += 1; m = self.M[u] + (f - self.M[u]) / min(self.n[u], 20); self.M[u] = m / np.linalg.norm(m)
                return u
        if not learn:
            return -1
        self.M.append(f); self.n.append(1); self.truth.append(truth)
        return len(self.M) - 1


class Links:
    def __init__(self):
        self.pw, self.p, self.w, self.pages = {}, {}, {}, 0

    def page(self, pics, words):
        self.pages += 1
        for p in set(pics):
            self.p[p] = self.p.get(p, 0) + 1
            for w in set(words):
                self.pw[(p, w)] = self.pw.get((p, w), 0) + 1
        for w in set(words):
            self.w[w] = self.w.get(w, 0) + 1

    def strength(self, p, w):
        """delta P (Allan 1980): how much more often the word comes with this picture than without it.
        Words met everywhere ("the") and words met once score low; plain co-occurrence ratios favoured the rare."""
        n_p = self.p.get(p, 0); n_w = self.w.get(w, 0); n_pw = self.pw.get((p, w), 0); N = max(self.pages, 1)
        if n_p == 0:
            return 0.0
        rest = N - n_p
        return n_pw / n_p - ((n_w - n_pw) / rest if rest > 0 else 0.0)

    def name(self, p, units):
        cand = [w for (p_, w) in self.pw if p_ == p]
        if not cand:
            return None
        return max(cand, key=lambda w: self.strength(p, w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--passes", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    R = FontReader(a.seed)
    d = np.load(DATA); F = d["O"].astype(float); cid, sid = d["concept"], d["style"]
    names, pat = concepts(); nC = len(names)
    style = (a.seed - 1) % 4; vstyle = (style + 1) % 4
    mu, sd = F[sid != style].mean(0), F[sid != style].std(0) + 1e-3            # adaptation to pictures in general
    Fz = (F - mu) / sd
    rng = np.random.default_rng(a.seed)
    PER = int(np.sum((cid == 0) & (sid == 0)))
    half = PER // 2                                                              # renderings 0..half-1 for the books, rest for tests

    def pic(k, st, test=False):
        idx = np.nonzero((cid == k) & (sid == st))[0]
        return Fz[idx[rng.integers(half, PER)] if test else idx[rng.integers(0, half)]]

    # picture threshold: FITTED on the validation style (truth used only there) by simulating three passes
    # through the 34 pictures and scoring how often a new rendering lands on a unit made from the same thing
    def purity(th):
        P_ = Pictures(th)
        for _ in range(a.passes):
            for k in rng.permutation(nC):
                P_.see(pic(k, vstyle), k)
        ok = [P_.truth[u] == k for k in range(nC) for _ in range(5) for u in [P_.see(pic(k, vstyle, test=True), learn=False)] if u >= 0]
        return float(np.mean(ok)) - 0.002 * max(len(P_.M) - nC, 0)            # small cost per extra unit
    grid = np.round(np.arange(0.5, 0.96, 0.05), 2)
    pscore = {float(t): purity(t) for t in grid}
    pthr = max(pscore, key=pscore.get)

    words = lessons()
    pics = [[names.index(k) for k in names if re.search(r"\b(" + pat[k] + r")\b", " ".join(l["pictures"]).lower())] for l in words]
    fam = int(np.random.default_rng(20 + a.seed).integers(30)); fam_test = 30 + int(np.random.default_rng(30 + a.seed).integers(30))
    res = dict(seed=a.seed, style=STYLES[style], validation_style=STYLES[vstyle], picture_threshold=round(float(pthr), 3),
               threshold_scores={str(k): round(v, 3) for k, v in pscore.items()}, font=fam, test_font=fam_test)

    pf = json.load(open(os.path.join(ROOT, "results", "primer_font", f"s{a.seed}.json")))
    ETA_FONT, CONF_FONT = float(pf["eta_font"]), float(pf["conf"])          # FITTED per network in Stage 44

    def read_word(W, tok, fam_, learn, eta=None, prior=None):
        eta = ETA_FONT if eta is None else eta
        let = encode(tok.lower()); cls = let + 26 * np.array([ch.isupper() for ch in tok], int)
        X = R.x("test", fam_, cls, rng.integers(0, C.PER, len(let))); ev = R.ev_x(X)
        idx, match, s = W.scores(ev)
        if idx is None:
            u, rel, post = -1, -np.inf, 0.0
        else:
            if prior is not None:
                s = s + np.array([prior(u_) for u_ in idx])
            k = int(np.argmax(s)); z = s - s.max()
            u, rel, post = idx[k], float((match[k] - W._self).mean()), float(1 / np.exp(z).sum())
        new = u < 0 or rel < W.thr
        if learn:
            if new:
                u = W.recruit(ev, tok.lower())
            else:
                W.refresh(u, ev)
                if post >= CONF_FONT:
                    for t, kk in enumerate(W.M[u].argmax(1)):
                        R.teach(X[t], int(kk), eta)
        return (None if (new and not learn) else u), tok.lower()

    def run(book):
        R.reset(); W = Words(WORD_THR, 0.0); P = Pictures(pthr); Lk = Links()
        pages = []
        if book in ("wordbook", "both"):
            pages += [("word", [k], [[names[k]]]) for k in range(nC)]
        if book in ("primer", "both", "control"):
            pp = list(pics)
            if book == "control":
                pp = [pp[i] for i in np.random.default_rng(99 + a.seed).permutation(len(pp))]
            pages += [("lesson", pp[i], l["sentences"]) for i, l in enumerate(words)]
        for _ in range(a.passes):
            for kind, things, sents in pages:
                pu = [P.see(pic(k, style), k) for k in things]
                wu = [read_word(W, t, fam, True)[0] for s in sents for t in s]
                Lk.page(pu, wu)
        return W, P, Lk

    def evaluate(W, P, Lk, things):
        def said(k, st):
            u = P.see(pic(k, st, test=True), learn=False)
            if u < 0:
                return None
            w = Lk.name(u, None)
            return None if w is None else W.truth[w]
        ok = lambda k, w: w is not None and w in (names[k], names[k] + "s")
        naming = {names[k]: float(np.mean([ok(k, said(k, style)) for _ in range(10)])) for k in things}
        other = {names[k]: float(np.mean([ok(k, said(k, st)) for st in range(4) if st != style for _ in range(4)])) for k in things}
        # choosing: the word read in the test font -> its unit; 4 pictures; pick the strongest link
        R0 = (R.dW.copy(), R.db.copy()); R.reset()                 # a new book's font: the fast weights start again
        ch = []
        for k in things:
            for _ in range(10):
                w, _t = read_word(W, names[k], fam_test, False)
                if w is None:
                    ch.append(False); continue
                opts = [k] + list(rng.choice([j for j in range(nC) if j != k], 3, replace=False))
                sc = []
                for j in opts:
                    u = P.see(pic(j, style, test=True), learn=False)
                    sc.append(Lk.strength(u, w) if u >= 0 else -9)
                ch.append(int(np.argmax(sc)) == 0 and max(sc) > min(sc))
        # help reading: the word under its picture, hard font, with / without the picture's links
        hard = [f for f in range(30, 60)]
        lacc = [R.letters_alone("test", f) for f in hard]; hf = hard[int(np.argmin(lacc))]
        withp = withoutp = n = 0
        for k in things:
            if names[k] not in W.truth:
                continue
            for _ in range(10):
                u = P.see(pic(k, style, test=True), learn=False)
                prior = (lambda w_, u=u: BETA_PIC * max(Lk.strength(u, w_), 0.0)) if u >= 0 else None
                rs = rng.bit_generator.state
                w1, _t = read_word(W, names[k], hf, False); rng.bit_generator.state = rs
                w2, _t = read_word(W, names[k], hf, False, prior=prior)
                withoutp += int(w1 is not None and W.truth[w1] == names[k]); withp += int(w2 is not None and W.truth[w2] == names[k]); n += 1
        R.dW, R.db = R0
        return dict(naming=round(float(np.mean(list(naming.values()))), 4), naming_per_thing={k: round(v, 2) for k, v in naming.items()},
                    choosing=round(float(np.mean(ch)), 4), naming_other_styles=round(float(np.mean(list(other.values()))), 4),
                    reading_hard_font=dict(font=int(hf), letters_alone=round(float(min(lacc)), 3), without_picture=round(withoutp / max(n, 1), 4),
                                           with_picture=round(withp / max(n, 1), 4), words=n),
                    picture_units=len(P.M), word_units=len(W.M))

    named_in_primer = [k for k in range(nC) if any(k in p and names[k] in {t.lower() for s in words[i]["sentences"] for t in s} for i, p in enumerate(pics))]
    res["things_pictured_and_named_in_the_primer"] = [names[k] for k in named_in_primer]
    for book in ("wordbook", "primer", "both", "control"):
        W, P, Lk = run(book)
        res[book] = dict(all_34=evaluate(W, P, Lk, range(nC)), primer_things=evaluate(W, P, Lk, named_in_primer))
        print(book, res[book]["all_34"]["naming"], res[book]["primer_things"]["naming"], res[book]["all_34"]["choosing"], res[book]["all_34"]["reading_hard_font"], flush=True)
    json.dump(res, open(os.path.join(OUT, f"s{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
