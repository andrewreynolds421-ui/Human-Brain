#!/usr/bin/env python3
"""Stage 86: reading comprehension of real books, grade by grade (Children's Book Test style).

Hill et al. 2016 (the Children's Book Test, from Project Gutenberg children's books): read 20 consecutive
sentences of a book, then fill a blank in the 21st from 10 candidate words. Here: the blank is a noun (spaCy tag
NN / NNS / NNP) that also occurs in the 20-sentence passage - so remembering the passage helps - and the 9 other
candidates are other nouns of the same passage. Chance 10 %.

Books: the graded collection of Stage 53 (`data/grades/catalogue.json`: school readers labelled by grade, other
books by calibrated Flesch-Kincaid), bands 1 .. 6, 7+. TEST items come only from the test books (never used in
training anywhere), and an item is dropped if its blanked sentence shares a 6-word run with anything VEGA's
reading network read (books, extra books, the grade 4-6 year) - the rule of the grade test, made stricter.
TRAIN items come from the training-split books.

    python scripts/passage_test.py            # writes data/passages.json.gz (+ results/passages/prepare.json)
"""
import gzip, json, os, pickle, sys, zlib
from collections import Counter, defaultdict
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar as GR  # noqa: E402
CAT = os.path.join(ROOT, "data", "grades", "catalogue.json")
OUTF = os.path.join(ROOT, "data", "passages.json.gz")
BANDS = ["1", "2", "3", "4", "5", "6", "7+"]
CTX, N_TEST, N_TRAIN, N_CAND, MAXW = 20, 300, 2500, 10, 400


def grams(s): return {zlib.crc32(" ".join(s[i:i + 6]).encode()) for i in range(len(s) - 5)}


def reader_reading():
    """6-word runs of everything the reading network read (Stage 73 networks: books + extra books + the year-4 books)"""
    import grammar_memory as GM
    C = GR.corpus(); C = GM.corpus_big(C); seen = set()
    for s in C["train"]: seen |= grams(s)
    for s in pickle.load(open("/home/claude/models/year4/sentences.pkl", "rb")): seen |= grams(s)
    return seen


def make(book_sents, nlp, rng, n, seen=None):
    items, tries, dropped = [], 0, 0
    starts = [(b, i) for b, ss in enumerate(book_sents) for i in range(0, len(ss) - CTX - 1, 3)]
    for k in rng.permutation(len(starts)):
        if len(items) >= n or tries > 40 * n: break
        tries += 1; b, i = starts[k]; ss = book_sents[b]; ctx = ss[i:i + CTX]; qs = ss[i + CTX]
        if not (6 <= len(qs) <= 30) or sum(map(len, ctx)) > MAXW: continue
        if seen is not None and grams(qs) & seen: dropped += 1; continue
        doc = nlp(" ".join(qs)); tags = [t.tag_ for t in doc]
        if len(doc) != len(qs): continue
        ctxw = set(w for s in ctx for w in s)
        nouns_ctx = sorted({t.text for s in nlp.pipe([" ".join(s) for s in ctx]) for t in s if t.tag_ in ("NN", "NNS", "NNP") and len(t.text) > 2})
        pos = [j for j, (w, tg) in enumerate(zip(qs, tags)) if tg in ("NN", "NNS", "NNP") and w in ctxw and len(w) > 2]
        if not pos: continue
        j = int(rng.choice(pos)); ans = qs[j]; others = [w for w in nouns_ctx if w != ans]
        if len(others) < N_CAND - 1: continue
        cands = [ans] + list(rng.choice(others, N_CAND - 1, replace=False)); order = rng.permutation(N_CAND)
        cands = [cands[o] for o in order]; y = int(np.nonzero(order == 0)[0][0])
        q = qs[:j] + ["something"] + qs[j + 1:]                                    # the blank, as an English placeholder
        items.append(dict(story=[w for s in ctx for w in s], ends=[int(x) for x in np.cumsum([len(s) for s in ctx]) - 1], q=q,
                          cands=[(c,) for c in cands], y=y, ans=ans, sent=qs, pos=j))
    return items, dropped


def main():
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    cat = json.load(open(CAT))["books"]; rng = np.random.default_rng(86)
    books = {sp: defaultdict(list) for sp in ("train", "test")}
    for c in cat:
        books[c["split"]][c["band"]].append(GR.sentences(GR.book_text(c["dir"])))
    print("reading the reader's reading ...", flush=True); seen = reader_reading(); print(len(seen), "6-word runs", flush=True)
    out = dict(train={}, test={}); info = dict(test_dropped={}, n={}, books={b: len(books["test"][b]) for b in BANDS})
    for b in BANDS:
        out["test"][b], info["test_dropped"][b] = make(books["test"][b], nlp, rng, N_TEST, seen)
        out["train"][b], _ = make(books["train"][b], nlp, rng, N_TRAIN)
        info["n"][b] = dict(test=len(out["test"][b]), train=len(out["train"][b])); print(b, info["n"][b], "dropped", info["test_dropped"][b], flush=True)
    for b in BANDS:                                                                  # examples for the write-up
        it = out["test"][b][0]; info.setdefault("examples", {})[b] = " ".join(it["story"][-25:]) + " || " + " ".join(it["q"]) + "  [" + ", ".join(c[0] for c in it["cands"]) + "] answer: " + it["ans"]
    json.dump(out, gzip.open(OUTF, "wt")); os.makedirs(os.path.join(ROOT, "results", "passages"), exist_ok=True)
    json.dump(info, open(os.path.join(ROOT, "results", "passages", "prepare.json"), "w"), indent=1); print(json.dumps(info, indent=1)[:3000])


if __name__ == "__main__":
    main()
