#!/usr/bin/env python3
"""Stage 59: real illustrated books - can the model's eyes use them?

Books: 23 illustrated picture books from Project Gutenberg's HTML editions (/home/claude/corpora/illustrated:
19 Beatrix Potter tales, Raggedy Ann and Raggedy Andy, two Three Little Pigs). Each picture is paired with the
text up to the next picture (and its caption): ~900 picture-text pairs.

Eyes: Stage 45's object cells (orientation energy and colour, pooled for position tolerance), applied to the
whole picture and to five crops (four quarters and the centre), each scaled to 48 x 48.

What a page "is about": which of the picture system's 34 things its text names (concepts.json keywords and
plurals) - e.g. rabbit, cat, duck, mouse, hen, pig, dog, tree, house, basket.

A  untrained: does the emoji-trained prototype of a thing score higher on pages whose text names it?
B  learned from the books: per thing, a reader of the object cells learned from the other books' pictures and
   their text (leave one book out); does it find the thing in the held-out book's pictures?
   CONTROL: the same with the text labels shuffled among the training pages.
Both measured WITHIN each book (pages of one book naming the thing vs pages of the same book not naming it),
so a book's drawing style cannot stand in for the thing. Area under the curve: 50 % = chance.

    python scripts/illustrated.py
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from synth.pictures import object_cells                                         # noqa: E402
BOOKS = "/home/claude/corpora/illustrated"
OUT = os.path.join(ROOT, "results", "illustrated")


def pairs():
    out = []
    for h in sorted(glob.glob(os.path.join(BOOKS, "*", "*-h", "*.htm*"))):
        book = os.path.basename(os.path.dirname(os.path.dirname(h))); html = open(h, encoding="latin-1").read()
        html = html[html.find("<body"):]
        parts = re.split(r"(<img[^>]*>)", html, flags=re.I)
        cur = None
        for p in parts:
            m = re.match(r"<img[^>]*src=\"([^\"]+)\"[^>]*>", p, flags=re.I)
            if m:
                if cur and cur["text"].strip(): out.append(cur)
                alt = re.search(r"alt=\"([^\"]*)\"", p, flags=re.I)
                path = os.path.join(os.path.dirname(h), m.group(1))
                cur = dict(book=book, image=path, text=(alt.group(1) if alt else "") + " ") if os.path.exists(path) else None
            elif cur is not None:
                t = re.sub(r"<[^>]+>", " ", p); t = re.sub(r"&[a-z]+;|&#\d+;", " ", t)
                cur["text"] += " " + " ".join(t.split())
        if cur and cur["text"].strip(): out.append(cur)
    return [p for p in out if len(p["text"].split()) >= 3 and "gutenberg" not in p["text"].lower()]


def views(path):
    from PIL import Image
    im = Image.open(path).convert("RGB"); W, H = im.size
    boxes = [(0, 0, W, H), (0, 0, W // 2, H // 2), (W // 2, 0, W, H // 2), (0, H // 2, W // 2, H), (W // 2, H // 2, W, H), (W // 4, H // 4, 3 * W // 4, 3 * H // 4)]
    return np.stack([np.asarray(im.crop(b).resize((48, 48), Image.BILINEAR)) for b in boxes])


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return None
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def within_book(score, Y, books, c):
    vals = []
    for b in set(books):
        m = np.array([x == b for x in books])
        a = auc(score[m & Y[:, c]], score[m & ~Y[:, c]])
        if a is not None and (m & Y[:, c]).sum() >= 2 and (m & ~Y[:, c]).sum() >= 2: vals.append(a)
    return vals


def main():
    os.makedirs(OUT, exist_ok=True)
    P = pairs(); print(len(P), "picture-text pairs from", len({p["book"] for p in P}), "books", flush=True)
    conc = json.load(open(os.path.join(ROOT, "data", "pictures", "concepts.json"))); names = list(conc)
    pat = {n: re.compile(r"\b(" + "|".join(set(conc[n][1].split("|")) | {n, n + "s"}) + r")\b", re.I) for n in names}
    Y = np.array([[bool(pat[n].search(p["text"])) for n in names] for p in P])
    print("pages naming each thing:", {n: int(Y[:, k].sum()) for k, n in enumerate(names) if Y[:, k].sum() >= 10}, flush=True)
    cache = os.path.join("/home/claude/models", "illustrated_cells.npz")
    if os.path.exists(cache): V = np.load(cache)["V"]
    else:
        V = np.stack([object_cells(views(p["image"])) for p in P]); np.savez_compressed(cache, V=V)
    Vn = V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-9)                       # (pages, 6 views, 400)
    d = np.load(os.path.join(ROOT, "data", "pictures.npz"), allow_pickle=True); O = d["O"] / (np.linalg.norm(d["O"], axis=1, keepdims=True) + 1e-9)
    proto = np.stack([O[d["concept"] == k].mean(0) for k in range(len(names))]); proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    books = [p["book"] for p in P]; res = dict(pairs=len(P), books=sorted(set(books)), A={}, B={}, B_shuffled={})
    usable = [k for k in range(len(names)) if Y[:, k].sum() >= 15]
    # A: untrained, emoji prototypes
    S = (Vn @ proto.T).max(1)                                                         # best view, per thing
    for k in usable:
        v = within_book(S[:, k], Y, books, k)
        if v: res["A"][names[k]] = [round(float(np.mean(v)), 3), len(v)]
    # B: learned from other books' pictures and text
    from sklearn.linear_model import LogisticRegression
    F = np.hstack([Vn.mean(1), Vn.max(1)]); rng = np.random.default_rng(0)
    for k in usable:
        for shuffled in (False, True):
            score = np.zeros(len(P))
            for b in set(books):
                te = np.array([x == b for x in books]); y = Y[~te, k].copy()
                if shuffled: rng.shuffle(y)
                if y.sum() < 5: score[te] = 0.0; continue
                clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced").fit(F[~te], y); score[te] = clf.decision_function(F[te])
            v = within_book(score, Y, books, k)
            if v: (res["B_shuffled"] if shuffled else res["B"])[names[k]] = [round(float(np.mean(v)), 3), len(v)]
    for key in ("A", "B", "B_shuffled"):
        vals = [v[0] for v in res[key].values()]
        res[key + "_mean"] = round(float(np.mean(vals)), 3) if vals else None
    print(json.dumps({k: v for k, v in res.items() if k != "books"}, indent=1))
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
