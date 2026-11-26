#!/usr/bin/env python3
"""Stage 90a: word meaning from VEGA's own reading - a "semantic cortex" beside the reading network.

VEGA's reading network was trained to predict the next word; its word representations carry form and grammar
("wolves" nearer "cats" than "wolf": number before meaning - Stage 80). Meaning in the brain is learned slowly from
which words occur together (distributional learning; semantic memory in the anterior temporal lobe: Patterson et al.
2007; Landauer & Dumais 1997 for the method). Here: from every sentence VEGA's reading network read (books, extra
books, the grade 4-6 year), count which words occur within 5 words of each other, weigh the counts by positive
pointwise mutual information (context counts smoothed with power 0.75, Levy et al. 2015) and keep the 128 strongest
dimensions (SVD). No other text, no pretrained model. MODEL (a statistical summary, not a trained network).

    python scripts/semantic_cortex.py        # writes data/semantic_vectors.npz and results/semantic/probe.json
"""
import json, os, pickle, sys
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
OUTF = os.path.join(ROOT, "data", "semantic_vectors.npz")
WINDOW, DIM, MIN_COUNT = 5, 128, 5


def load(path=OUTF):
    z = np.load(path, allow_pickle=False); return {w: v for w, v in zip(z["words"].tolist(), z["vectors"].astype(np.float32))}


def main():
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds
    import grammar as GR, grammar_memory as GM
    C = GM.corpus_big(GR.corpus()); sents = list(C["train"]) + list(pickle.load(open("/home/claude/models/year4/sentences.pkl", "rb")))
    cnt = Counter(w for s in sents for w in s); vocab = sorted(w for w, n in cnt.items() if n >= MIN_COUNT); ix = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    ids = np.fromiter((ix.get(w, -1) for s in sents for w in s), dtype=np.int32); sid = np.repeat(np.arange(len(sents)), [len(s) for s in sents])
    print(len(sents), "sentences", len(ids), "words", V, "vocabulary", flush=True)
    M = sp.csr_matrix((V, V), dtype=np.float64)
    for d in range(1, WINDOW + 1):
        ok = (sid[:-d] == sid[d:]) & (ids[:-d] >= 0) & (ids[d:] >= 0); r, c = ids[:-d][ok], ids[d:][ok]
        M = M + sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(V, V)).tocsr()
    M = M + M.T; tot = M.sum(); rw = np.asarray(M.sum(1)).ravel(); cw = rw ** 0.75; cw = cw / cw.sum()
    M = M.tocoo(); pmi = np.log(M.data / tot / ((rw[M.row] / tot) * cw[M.col])); keep = pmi > 0
    PP = sp.csr_matrix((pmi[keep], (M.row[keep], M.col[keep])), shape=(V, V))
    U, S, _ = svds(PP, k=DIM, random_state=0); vec = (U * np.sqrt(S)).astype(np.float32); vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6
    np.savez_compressed(OUTF, words=np.array(vocab), vectors=vec.astype(np.float16))
    # probe: is a noun's plural its nearest word? (data/noun_pairs.json)
    pairs = [p for p in json.load(open(os.path.join(ROOT, "data", "noun_pairs.json"))) if p[0] in ix and p[1] in ix]
    ranks = []
    for a, b in pairs:
        sim = vec @ vec[ix[a]]; sim[ix[a]] = -9; ranks.append(int((sim > sim[ix[b]]).sum()) + 1)
    cos = lambda a, b: round(float(vec[ix[a]] @ vec[ix[b]]), 2) if a in ix and b in ix else None
    near = lambda w: [vocab[i] for i in np.argsort(-(vec @ vec[ix[w]]))[1:7]] if w in ix else None
    probe = dict(sentences=len(sents), words=int(len(ids)), vocabulary=V, plural_pairs=len(pairs), plural_is_nearest=round(float(np.mean(np.array(ranks) == 1)), 3),
                 plural_in_top10=round(float(np.mean(np.array(ranks) <= 10)), 3), median_rank=float(np.median(ranks)),
                 cosines={f"{a}-{b}": cos(a, b) for a, b in [("wolf", "wolves"), ("wolves", "cats"), ("mouse", "mice"), ("mice", "cats"), ("cat", "cats"), ("sheep", "wolf"), ("kitchen", "garden"), ("kitchen", "wolf")]},
                 nearest={w: near(w) for w in ("wolf", "mouse", "kitchen", "afraid", "gave", "queen", "ship")})
    os.makedirs(os.path.join(ROOT, "results", "semantic"), exist_ok=True); json.dump(probe, open(os.path.join(ROOT, "results", "semantic", "probe.json"), "w"), indent=1); print(json.dumps(probe, indent=1))


if __name__ == "__main__":
    main()
