#!/usr/bin/env python3
"""Stage 50: grammar from text alone (no eyes).

Corpus: 32 children's books from Project Gutenberg (GITenberg mirror; /home/claude/corpora/books), ~1.8 M
words, split by BOOK: TEST_BOOKS are never read, VAL_BOOKS choose settings.

What the model learns, with the kinds of learning it already has (event level, not spiking):
  word units      - one per word met at least 3 times (others share an "unknown" unit)
  context counts  - each word unit counts which of the 500 most frequent words come just before and just
                    after it (Hebbian co-occurrence); its profile = log counts, unit length
  category units  - K units compete for the word profiles (winner-take-all competitive learning; mini-batch
                    k-means is its batch form), each word joining the unit it matches best: word classes
                    learned from distribution alone (Redington, Chater & Finch 1998; Mintz 2003)
  sequence counts - which category follows which, and which follows a pair (Hebbian counts), with the
                    sentence start / end as units; the grammar is these transition counts
  sentence score  - log P(categories in order) + log P(each word | its category), interpolated with a
                    word-level trigram model of the same text (the reference without categories)

Tests:
  A  categories vs parts of speech (spaCy en_core_web_sm tags, used for grading only): purity, V-measure,
     against random categories of the same sizes
  B  a NEW word: 300 words held out of category learning; from 1 / 3 / 10 sentences containing it, the
     category it joins - is that category's majority part of speech the word's own?
  C  sentences of never-read books: the real sentence vs the same words with two neighbours swapped
     (word order), and vs a random shuffle - which does the model prefer?
  D  BLiMP (Warstadt et al. 2020): 67 phenomena x 1,000 minimal pairs, grammatical vs not - which does
     the model score higher?  Humans 88.6 %; a 5-gram model trained on far more text 61.2 %.

    python scripts/grammar.py   ->  results/grammar/summary.json
"""
import glob, json, math, os, re, sys
from collections import Counter, defaultdict
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BOOKS = "/home/claude/corpora/books"
BLIMP = "/home/claude/corpora/blimp/data"
OUT = os.path.join(ROOT, "results", "grammar")
TEST_BOOKS = ("Treasure-Island", "The-Secret-Garden", "Just-So-Stories", "McGuffey-s-Third")
VAL_BOOKS = ("Black-Beauty", "Ozma-of-Oz")
S, E, UNK = "<s>", "</s>", "<unk>"


def book_text(d):
    fs = [f for f in glob.glob(os.path.join(d, "*.txt")) if not re.search(r"-(8|0)\.txt$", f)] or glob.glob(os.path.join(d, "*.txt"))
    t = open(fs[0], encoding="latin-1").read()
    a = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", t); b = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", t)
    if a: t = t[a.end():]
    if b: t = t[:b.start()]
    return t


def sentences(text):
    text = re.sub(r"\[[^\]]*\]", " ", text).replace("\r", "")
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if len(para) < 10 or para.isupper():
            continue
        for s in re.split(r"(?<=[.!?])\s+", para):
            toks = re.findall(r"[a-z]+(?:'[a-z]+)?", s.lower())
            if 3 <= len(toks) <= 25:
                out.append(toks)
    return out


def corpus():
    out = {"train": [], "val": [], "test": []}
    for d in sorted(glob.glob(os.path.join(BOOKS, "*"))):
        name = os.path.basename(d)
        split = "test" if name.startswith(TEST_BOOKS) else "val" if name.startswith(VAL_BOOKS) else "train"
        out[split] += sentences(book_text(d))
    return out


class Grammar:
    def __init__(self, sents, K=40, n_ctx=500, min_count=3, seed=0, hold=()):
        cnt = Counter(w for s in sents for w in s)
        self.vocab = {w for w, c in cnt.items() if c >= min_count}
        self.ctx = [w for w, _ in cnt.most_common(n_ctx)]; ci = {w: i for i, w in enumerate(self.ctx)}
        self.hold = set(hold)
        L = defaultdict(Counter); R = defaultdict(Counter)
        for s in sents:
            s2 = [S] + s + [E]
            for i in range(1, len(s2) - 1):
                w = s2[i]
                if w in self.hold or w not in self.vocab:
                    continue
                if s2[i - 1] in ci: L[w][ci[s2[i - 1]]] += 1
                if s2[i + 1] in ci: R[w][ci[s2[i + 1]]] += 1
                if s2[i - 1] == S: L[w][n_ctx] += 1
                if s2[i + 1] == E: R[w][n_ctx] += 1
        self.n_ctx, self.ci = n_ctx, ci
        words = sorted(w for w in self.vocab if w not in self.hold)
        X = np.array([self.profile(L[w], R[w]) for w in words])
        from sklearn.cluster import MiniBatchKMeans
        big = np.array([cnt[w] >= 10 for w in words])
        km = MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=2048).fit(X[big])
        self.cent = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        self.cat = {w: int(np.argmax(self.cent @ x)) for w, x in zip(words, X)}
        self.K = K
        self.fit_sequences(sents)

    def profile(self, l, r):
        v = np.zeros(2 * (self.n_ctx + 1))
        for i, c in l.items(): v[i] = np.log1p(c)
        for i, c in r.items(): v[self.n_ctx + 1 + i] = np.log1p(c)
        return v / (np.linalg.norm(v) + 1e-9)

    def c(self, w):
        return self.cat.get(w, self.K)                    # unknown words: their own category K

    def fit_sequences(self, sents):
        K = self.K + 3; self.cs, self.ce = self.K + 1, self.K + 2
        self.t1 = np.full(K, 0.1); self.t2 = np.full((K, K), 0.1); self.t3 = defaultdict(Counter)
        self.em = defaultdict(Counter); self.cn = Counter()
        self.w1 = Counter(); self.w2 = defaultdict(Counter); self.w3 = defaultdict(Counter)
        for s in sents:
            cs = [self.cs, self.cs] + [self.c(w) for w in s] + [self.ce]
            ws = [S, S] + [w if w in self.vocab else UNK for w in s] + [E]
            for i in range(2, len(cs)):
                self.t1[cs[i]] += 1; self.t2[cs[i - 1], cs[i]] += 1; self.t3[(cs[i - 2], cs[i - 1])][cs[i]] += 1
                self.w1[ws[i]] += 1; self.w2[ws[i - 1]][ws[i]] += 1; self.w3[(ws[i - 2], ws[i - 1])][ws[i]] += 1
                if i < len(cs) - 1:
                    self.em[cs[i]][ws[i]] += 1; self.cn[cs[i]] += 1
        self.N1 = sum(self.w1.values()); self.V = len(self.w1) + 1

    def p_cat(self, a, b, c):
        t = self.t3.get((a, b)); n3 = sum(t.values()) if t else 0
        p2 = self.t2[b, c] / self.t2[b].sum(); p1 = self.t1[c] / self.t1.sum()
        p3 = t[c] / n3 if n3 else 0.0
        lam = n3 / (n3 + 5.0)
        return lam * p3 + (1 - lam) * (0.9 * p2 + 0.1 * p1)

    def p_word(self, u, v, w):
        t = self.w3.get((u, v)); n3 = sum(t.values()) if t else 0
        b = self.w2.get(v); n2 = sum(b.values()) if b else 0
        p1 = (self.w1[w] + 0.5) / (self.N1 + 0.5 * self.V)
        p2 = (b[w] / n2) if n2 else 0.0; p3 = (t[w] / n3) if n3 else 0.0
        l3, l2 = n3 / (n3 + 3.0), n2 / (n2 + 3.0)
        return l3 * p3 + (1 - l3) * (l2 * p2 + (1 - l2) * p1)

    def score(self, s, mix=0.5, parts=False):
        cs = [self.cs, self.cs] + [self.c(w) for w in s] + [self.ce]
        ws = [S, S] + [w if w in self.vocab else UNK for w in s] + [E]
        lc = lw = lm = 0.0
        for i in range(2, len(cs)):
            pc = self.p_cat(cs[i - 2], cs[i - 1], cs[i])
            pe = 1.0 if i == len(cs) - 1 else (self.em[cs[i]][ws[i]] + 0.1) / (self.cn[cs[i]] + 0.1 * self.V)
            pw = self.p_word(ws[i - 2], ws[i - 1], ws[i])
            lc += math.log(pc * pe); lw += math.log(pw); lm += math.log(mix * pc * pe + (1 - mix) * pw)
        return (lc, lw, lm) if parts else lm

    def place(self, word, sents):
        """category for a new word from the sentences it occurs in"""
        L = Counter(); R = Counter()
        for s in sents:
            s2 = [S] + s + [E]
            for i in range(1, len(s2) - 1):
                if s2[i] != word: continue
                if s2[i - 1] in self.ci: L[self.ci[s2[i - 1]]] += 1
                if s2[i + 1] in self.ci: R[self.ci[s2[i + 1]]] += 1
                if s2[i - 1] == S: L[self.n_ctx] += 1
                if s2[i + 1] == E: R[self.n_ctx] += 1
        return int(np.argmax(self.cent @ self.profile(L, R)))


def pos_of_words(sents, n_max=60000):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    tags = defaultdict(Counter)
    for doc, s in zip(nlp.pipe((" ".join(s) for s in sents[:n_max]), batch_size=512), sents[:n_max]):
        for t in doc:
            tags[t.text][t.pos_] += 1
    return {w: c.most_common(1)[0][0] for w, c in tags.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    C = corpus(); rng = np.random.default_rng(a.seed)
    res = dict(sentences={k: len(v) for k, v in C.items()}, words={k: sum(map(len, v)) for k, v in C.items()})
    print(res, flush=True)
    pos = pos_of_words(C["train"])
    # K chosen on validation books: average log-probability per word of the category model
    best = None
    for K in (20, 40, 80, 160):
        G = Grammar(C["train"], K, seed=a.seed)
        lp = np.mean([G.score(s, parts=True)[0] / (len(s) + 1) for s in C["val"][:4000]])
        print("K", K, "validation log-prob per word (categories)", round(float(lp), 3), flush=True)
        if best is None or lp > best[0]:
            best = (lp, K, G)
    _, K, G = best; res["K"] = K
    # mixture weight on validation: real vs swapped neighbours
    def swapped(s, r):
        i = int(r.integers(0, len(s) - 1)); t = list(s); t[i], t[i + 1] = t[i + 1], t[i]; return t
    vr = np.random.default_rng(1 + 10 * a.seed); vp = [(s, swapped(s, vr)) for s in C["val"][:3000] if len(set(s)) > 2]
    mix = max((np.mean([G.score(a, m) > G.score(b, m) for a, b in vp if a != b]), m) for m in (0.1, 0.3, 0.5, 0.7, 0.9))[1]; res["mix"] = mix

    # A: categories vs parts of speech
    from sklearn.metrics import v_measure_score
    ws = [w for w in G.cat if w in pos]
    cats = np.array([G.cat[w] for w in ws]); tags = np.array([pos[w] for w in ws])
    def purity(c):
        return float(sum(Counter(tags[c == k]).most_common(1)[0][1] for k in np.unique(c)) / len(c))
    perm = rng.permutation(cats)
    tok = Counter(w for s in C["train"] for w in s)
    wt = np.array([tok[w] for w in ws], float)
    def purity_w(c):
        tot = 0.0
        for k in np.unique(c):
            m = c == k; t = Counter(); [t.update({g: x}) for g, x in zip(tags[m], wt[m])]
            tot += t.most_common(1)[0][1]
        return float(tot / wt.sum())
    examples = {}
    for k in range(K):
        m = [w for w in ws if G.cat[w] == k]
        if m:
            m.sort(key=lambda w: -tok[w]); examples[k] = dict(main_pos=Counter(pos[w] for w in m).most_common(2), words=m[:10], size=len(m))
    res["A_categories"] = dict(words=len(ws), purity=round(purity(cats), 3), purity_random=round(purity(perm), 3),
                               purity_by_token=round(purity_w(cats), 3), purity_by_token_random=round(purity_w(perm), 3),
                               v_measure=round(float(v_measure_score(tags, cats)), 3), v_measure_random=round(float(v_measure_score(tags, perm)), 3),
                               categories=examples)
    print("A", {k: v for k, v in res["A_categories"].items() if k != "categories"}, flush=True)

    # B: new words placed from 1 / 3 / 10 sentences (held out of category learning)
    cand = [w for w, c in tok.items() if 20 <= c <= 400 and w in pos and pos[w] in ("NOUN", "VERB", "ADJ", "ADV")]
    hold = list(rng.choice(cand, 300, replace=False))
    Gh = Grammar(C["train"], K, hold=hold, seed=a.seed)
    major = {}
    for k in range(K):
        m = [pos[w] for w in Gh.cat if Gh.cat[w] == k and w in pos]
        major[k] = Counter(m).most_common(1)[0][0] if m else None
    occ = defaultdict(list)
    for s in C["train"]:
        for w in set(s) & set(hold):
            occ[w].append(s)
    B = {}
    for n in (1, 3, 10):
        ok = []
        for w in hold:
            if len(occ[w]) >= n:
                some = [occ[w][i] for i in rng.choice(len(occ[w]), n, replace=False)]
                ok.append(major.get(Gh.place(w, some)) == pos[w])
        B[str(n)] = round(float(np.mean(ok)), 3)
    prior = Counter(pos[w] for w in hold); B["chance_most_common_class"] = round(prior.most_common(1)[0][1] / len(hold), 3)
    B["held_out_words_by_class"] = dict(prior)
    res["B_new_words"] = B; print("B", B, flush=True)

    # C: never-read books - real vs swapped neighbours / shuffled
    tr_ = np.random.default_rng(2 + 10 * a.seed); Cres = {}
    test = [s for s in C["test"] if len(set(s)) > 3][:6000]
    for name, make in (("two_neighbours_swapped", lambda s: swapped(s, tr_)), ("shuffled", lambda s: list(tr_.permutation(s)))):
        pairs = [(s, make(s)) for s in test]; pairs = [(a, b) for a, b in pairs if a != b]
        sc = [(G.score(a, parts=True), G.score(b, parts=True), G.score(a, mix), G.score(b, mix)) for a, b in pairs]
        Cres[name] = dict(pairs=len(pairs), categories=round(float(np.mean([x[0] > y[0] for x, y, _, _ in sc])), 4),
                          words_trigram=round(float(np.mean([x[1] > y[1] for x, y, _, _ in sc])), 4),
                          both=round(float(np.mean([m1 > m2 for _, _, m1, m2 in sc])), 4))
    res["C_word_order_never_read_books"] = Cres; print("C", Cres, flush=True)

    # D: BLiMP
    D = {}
    for f in sorted(glob.glob(os.path.join(BLIMP, "*.jsonl"))):
        rows = [json.loads(l) for l in open(f)]
        field = rows[0].get("linguistics_term", "?")
        r = dict(categories=[], words=[], both=[], known=[])
        for x in rows:
            g = re.findall(r"[a-z]+(?:'[a-z]+)?", x["sentence_good"].lower()); b = re.findall(r"[a-z]+(?:'[a-z]+)?", x["sentence_bad"].lower())
            sg, sb = G.score(g, parts=True), G.score(b, parts=True)
            r["categories"].append(sg[0] > sb[0]); r["words"].append(sg[1] > sb[1]); r["both"].append(G.score(g, mix) > G.score(b, mix))
            r["known"].append(all(w in G.vocab for w in g + b))
        kn = np.array(r["known"])
        D[os.path.basename(f)[:-6]] = dict(field=field, **{k: round(float(np.mean(v)), 3) for k, v in r.items()},
                                           known_pairs=int(kn.sum()), **{f"{k}_known_words_only": (round(float(np.mean(np.array(r[k])[kn])), 3) if kn.sum() else None) for k in ("categories", "words", "both")})
    fields = defaultdict(list)
    for k, v in D.items():
        fields[v["field"]].append(v)
    tot = sum(v["known_pairs"] for v in D.values())
    known_only = {m: round(float(sum(v[f"{m}_known_words_only"] * v["known_pairs"] for v in D.values() if v["known_pairs"]) / tot), 3) for m in ("categories", "words", "both")}
    res["D_blimp"] = dict(overall={m: round(float(np.mean([v[m] for v in D.values()])), 3) for m in ("categories", "words", "both", "known")},
                          pairs_with_known_words_only=dict(pairs=tot, **known_only),
                          by_field={f: {m: round(float(np.mean([v[m] for v in vs])), 3) for m in ("categories", "words", "both")} for f, vs in sorted(fields.items())},
                          by_phenomenon=D)
    print("D", res["D_blimp"]["overall"], "known words only", res["D_blimp"]["pairs_with_known_words_only"], flush=True)
    for f, v in res["D_blimp"]["by_field"].items():
        print("   ", f, v, flush=True)
    res["seed"] = a.seed
    json.dump(res, open(os.path.join(OUT, "summary.json" if a.seed == 0 else f"s{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
