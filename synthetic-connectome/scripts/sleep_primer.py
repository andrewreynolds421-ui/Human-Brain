#!/usr/bin/env python3
"""Stage 48: sleep after reading.

Day 1: the model reads McGuffey's First Reader from no word units in one never-seen font (Stages 43-44: word
units recruited on the fly, recognised words teach the book's font to FAST letter weights; settings as fitted
per network in Stage 44).  It keeps the day's episodes: the edge-cell input of each letter of each word it
recognised, with the word unit that recognised it.

Night (no input, no labels), one of:
  none          - nothing (Stage 44: the fast weights are discarded with the book)
  awake         - CONTROL: the book is read once more, awake, learning on (the same number of word sightings
                  as the replay below), so any gain from sleep is not just more practice
  merge         - duplicate word units are merged: units of the same length whose letters read the same
                  (same most likely spelling) become one (templates averaged by use, counts and links added)
                  (lexical integration of new words after sleep: Dumay & Gaskell 2007)
  replay        - the day's episodes are replayed: each replayed word's spelling teaches the SLOW letter units
                  (consolidation from fast to slow; complementary learning systems, McClelland et al. 1995),
                  interleaved one-for-one with rehearsal of old knowledge - letters of the training fonts named
                  by a stable copy of the units as they were before the night (as Stage 41)
  merge+replay  - both

Day 2 (all conditions alike): (B) a second copy in 4 other never-seen fonts, word units frozen, font learning
from zero; (C) the Second Reader, never read, in another font; (A2) the same book in the first font again;
(L) letters alone in 10 other never-seen fonts, slow units only (does consolidation cost general reading?).
The replay step size is FITTED on validation fonts (B-style) among REPLAY_ETAS.

    python scripts/sleep_primer.py --seed 1   ->  results/sleep/s1.json
"""
import argparse, copy, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capitals as C                                                     # noqa: E402
from read_primer import Words, SECOND, read_book                         # noqa: E402
from read_primer_font import FontReader, winner, read, summary_B, VAL_FONTS   # noqa: E402
from synth.primer import lessons                                         # noqa: E402
from synth.words import L, encode                                        # noqa: E402

OUT = os.path.join(ROOT, "results", "sleep")
REPLAY_ETAS = (0.03, 0.1, 0.3)


def day1(R, W, book, rng, fam, eta, conf):
    """read the book (learning on) and keep the episodes of recognised words"""
    R.reset(); known = set(W.truth); episodes = []
    for lesson in book:
        for sent in lesson["sentences"]:
            for tok in sent:
                let = encode(tok.lower()); cls = let + 26 * np.array([c.isupper() for c in tok], int)
                X = R.x("test", fam, cls, rng.integers(0, C.PER, len(let))); ev = R.ev_x(X); w = tok.lower()
                u, rel, post = winner(W, ev)
                new = u < 0 or rel < W.thr
                if new:
                    u = W.recruit(ev, w); known.add(w)
                else:
                    W.refresh(u, ev)
                    if post >= conf:
                        for t, k in enumerate(W.M[u].argmax(1)):
                            R.teach(X[t], int(k), eta)
                        episodes.append((X, u))
    return episodes


def merge(W):
    """replay each word unit's own pattern (weakest first); if another, more used unit of the same length
    recognises it as a word it knows (match relative to a new word above the reading threshold), the two
    become one - templates averaged by use, counts and links added.  No labels."""
    where = {u: u for u in range(len(W.M))}
    alive = set(range(len(W.M)))
    for u in sorted(range(len(W.M)), key=lambda u: W.n[u]):
        cand = [v for v in W.by_len.get(W.len[u], []) if v != u and v in alive and W.n[v] >= W.n[u]]
        if not cand:
            continue
        ev = W.M[u]
        Mw = np.stack([W.M[v] for v in cand])
        match = np.log(np.einsum("ntl,tl->nt", Mw, ev) * (1 - 0.01) + 0.01)
        self_ = np.log(ev.max(1) * (1 - 0.01) + 0.01)
        rel = (match - self_).mean(1); k = int(np.argmax(rel))
        if rel[k] >= W.thr:
            v = cand[k]; n_u, n_v = W.n[u], W.n[v]
            W.M[v] = (W.M[v] * n_v + W.M[u] * n_u) / (n_u + n_v); W.n[v] = n_u + n_v
            alive.discard(u); where[u] = v
    for u in list(where):                               # follow chains
        while where[where[u]] != where[u]:
            where[u] = where[where[u]]
    keep = sorted(alive)
    groups = {}
    for u, v in where.items():
        groups.setdefault(v, []).append(u)
    merged_wrong = sum(1 for g in groups.values() if len({W.truth[u] for u in g}) > 1)
    merged_right = sum(len(g) - 1 for g in groups.values() if len({W.truth[u] for u in g}) == 1)
    old = W; N = Words(old.thr, old.beta_w); new_id = {}
    for k in keep:
        new_id[k] = len(N.M); N.M.append(old.M[k]); N.n.append(old.n[k]); N.truth.append(old.truth[k]); N.len.append(old.len[k])
        N.by_len.setdefault(old.len[k], []).append(new_id[k])
    for (p, q), c in old.follow.items():
        a = new_id[where[p]] if p >= 0 else p; b = new_id[where[q]]
        N.follow[(a, b)] = N.follow.get((a, b), 0) + c; N.after[a] = N.after.get(a, 0) + c
    return N, len(old.M) - len(N.M), merged_wrong, merged_right


def replay(R, W, episodes, n, eta_s, rng, Rstable):
    """consolidate: replayed words teach the slow letter units, interleaved with rehearsal of old letters"""
    R.reset()
    for _ in range(n):
        X, u = episodes[int(rng.integers(len(episodes)))]
        spelling = W.M[u].argmax(1)
        for t, k in enumerate(spelling):
            slow_teach(R, X[t], int(k), eta_s)
            x = R.V["train"][C.img_index(int(rng.integers(R.nf["train"])), int(rng.integers(C.NCLS)), int(rng.integers(C.PER)))]
            slow_teach(R, x, int(Rstable(x)), eta_s)


def slow_teach(R, x, k, eta):
    z = x @ R.W.T + R.b; z -= z.max(); p = np.exp(z); p /= p.sum()
    err = -p; err[k] += 1.0
    R.W += (eta / R.xn2) * np.outer(err, x); R.b += (eta / R.xn2) * err


def letters_slow(R, fams):
    R.reset()
    return float(np.mean([R.letters_alone("test", f) for f in fams]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pf = json.load(open(os.path.join(ROOT, "results", "primer_font", f"s{a.seed}.json")))
    eta, conf, thr = float(pf["eta_font"]), float(pf["conf"]), float(pf["threshold"])
    famA, famsB = int(pf["font_A"]), [int(f) for f in pf["fonts_B"]]
    fr = np.random.default_rng(40 + a.seed); others = [int(f) for f in fr.choice([f for f in range(60) if f != famA and f not in famsB], 11, replace=False)]
    fam2, famsL = others[0], others[1:]
    book, book2 = lessons(), lessons(SECOND)
    R = FontReader(a.seed); W0slow = (R.W.copy(), R.b.copy())
    W = Words(thr, 0.0)
    episodes = day1(R, W, book, np.random.default_rng(a.seed), famA, eta, conf)
    day = dict(units=len(W.M), duplicates=sum(1 for i, t in enumerate(W.truth) if t in W.truth[:i]), episodes=len(episodes))
    Wday, dWday = copy.deepcopy(W), (R.dW.copy(), R.db.copy())
    n_tokens = sum(len(s) for l in book for s in l["sentences"])

    def stable(x, Ws=W0slow):
        z = x @ Ws[0].T + Ws[1]; return int(np.argmax(z))

    def night(kind, eta_s):
        R.W, R.b = W0slow[0].copy(), W0slow[1].copy(); R.dW, R.db = dWday[0].copy(), dWday[1].copy()
        Wn = copy.deepcopy(Wday); info = {}
        if kind == "awake":
            r = read(R, Wn, book, np.random.default_rng(500 + a.seed), "test", famA, True, eta, conf)
            info["new_units_while_rereading"] = len(Wn.M) - len(Wday.M)
        if kind in ("merge", "merge+replay"):
            Wn, removed, wrong, right = merge(Wn)
            info.update(units_merged_away=removed, merges_of_the_same_word=right, groups_merging_different_words=wrong)
        if kind in ("replay", "merge+replay"):
            # episodes point at day-1 units; their spellings are read from the day-1 lexicon (merging renumbers)
            replay(R, Wday, episodes, n_tokens, eta_s, np.random.default_rng(600 + a.seed), stable)
            info["replayed_words"] = n_tokens
        return Wn, info

    def day2(Wn, fams=famsB, split="test"):
        out = []
        for f in fams:
            out.append(summary_B(Wn, read(R, Wn, book, np.random.default_rng(900 + a.seed + f), split, f, False, eta, conf)))
        return {k: round(float(np.mean([o[k] for o in out if o[k] is not None])), 4) for k in ("known_read_right", "known_called_unknown", "every_letter_right", "letters_right")}

    # fit the replay step on validation fonts
    vs = {}
    for e in REPLAY_ETAS:
        Wn, _ = night("replay", e); vs[e] = day2(Wn, VAL_FONTS, "val")["known_read_right"]
    eta_s = max(vs, key=vs.get)
    res = dict(seed=a.seed, font_A=famA, fonts_B=famsB, font_C=fam2, fonts_L=famsL, replay_eta=eta_s, validation_by_eta=vs, day1=day)
    for kind in ("none", "awake", "merge", "replay", "merge+replay"):
        Wn, info = night(kind, eta_s)
        d = dict(info, units=len(Wn.M), duplicates=sum(1 for i, t in enumerate(Wn.truth) if t in Wn.truth[:i]))
        d["B_second_copy"] = day2(Wn)
        d["C_second_reader"] = day2(Wn, [fam2]) if False else None
        rC = read(R, Wn, book2, np.random.default_rng(950 + a.seed), "test", fam2, False, eta, conf)
        d["C_second_reader"] = {k: round(float(v), 4) for k, v in summary_B(Wn, rC).items() if k in ("known_read_right", "known_called_unknown", "unknown_flagged", "every_letter_right")}
        rA = read(R, Wn, book, np.random.default_rng(970 + a.seed), "test", famA, False, eta, conf)
        d["A2_same_book_same_font"] = {k: round(float(v), 4) for k, v in summary_B(Wn, rA).items() if k in ("known_read_right", "known_called_unknown", "every_letter_right")}
        d["L_letters_other_fonts_slow_units"] = round(letters_slow(R, famsL), 4)
        d["L_letters_book_font_slow_units"] = round(letters_slow(R, [famA]), 4)
        res[kind] = d
        print(kind, {k: v for k, v in d.items() if not isinstance(v, dict)}, d["B_second_copy"]["known_read_right"], d["C_second_reader"]["known_read_right"], flush=True)
    json.dump(res, open(os.path.join(OUT, f"s{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
