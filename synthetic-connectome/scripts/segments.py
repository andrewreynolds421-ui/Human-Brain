#!/usr/bin/env python3
"""Stage 68: memory in segments (the user's idea).

Once something is learned it is saved in a folder apart from the network; the network (or another version of it)
adds a segment to its active memory when the task needs it, and leaves it on disk otherwise. Here a segment is
a Stage 67 episodic store written to a folder: the search index, the words that followed, the projection used,
and a description (what it holds, who wrote it, its size).

Segments, written by each Stage 65 *books* network without any retraining:
  books   every word of the books it was trained on
  speech  speech to children (CHILDES adult utterances) - material this network never trained on; it only
          reads it once and files it
Questions: (1) does the matching segment help, and does the wrong one do nothing? on held-out book text and
held-out speech, with none / books / speech / both segments active (log-probability per word), and on one
speech skill the books network lacks (Stage 61's "had Peter ever called" test); (2) what does an active segment
cost (megabytes, seconds to load, time per word)? (3) can another version use it? - a network with different
weights loads this network's books segment (--borrow).

    python scripts/segments.py --seed 0            # write the segments and test
    python scripts/segments.py --seed 0 --borrow 1 # network 0 uses network 1's books segment
"""
import argparse, json, os, pickle, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar_memory as GM, output_order as OO, picture_check as PC, sleep_grammar as SG, childes as CH, episodic as EP   # noqa: E402
OUT = os.path.join(ROOT, "results", "segments")
DATA = "/home/claude/models/segments"
GRID = [(lam, t) for lam in (0.05, 0.1, 0.2, 0.4) for t in (0.25, 1.0, 4.0)]


def save(store, folder, **meta):
    os.makedirs(folder, exist_ok=True)
    store.faiss.write_index(store.index, os.path.join(folder, "index.faiss")); np.save(os.path.join(folder, "next_word.npy"), store.vals)
    np.savez(os.path.join(folder, "projection.npz"), P=store.P, mu=store.mu)
    mb = sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)) / 1e6
    json.dump(dict(meta, moments=int(store.index.ntotal), megabytes=round(mb, 1)), open(os.path.join(folder, "segment.json"), "w"), indent=1)


class Segment:
    """a segment added to active memory"""
    def __init__(self, folder):
        import faiss
        t0 = time.time(); self.meta = json.load(open(os.path.join(folder, "segment.json")))
        self.index = faiss.read_index(os.path.join(folder, "index.faiss")); self.index.nprobe = 16
        self.vals = np.load(os.path.join(folder, "next_word.npy")); z = np.load(os.path.join(folder, "projection.npz")); self.P, self.mu = z["P"], z["mu"]
        self.load_seconds = round(time.time() - t0, 2)

    def share(self, H, y, T):
        D, I = self.index.search(((H - self.mu) @ self.P.T).astype(np.float32), EP.K)
        return EP.knn_share(y, D, self.vals[np.maximum(I, 0)], I >= 0, T)

    def median_distance(self, H):
        return float(np.median(self.index.search(((H - self.mu) @ self.P.T).astype(np.float32), 1)[0])) + 1e-6


def prepared(M, sents):
    """network log-probs, full states and word ids for a text, concatenated; and where each sentence starts"""
    F = EP.feats(M, sents); cut = np.cumsum([0] + [len(f[0]) for f in F])
    return np.concatenate([f[0] for f in F]), np.concatenate([f[1] for f in F]), np.concatenate([f[2] for f in F]), cut


def mixed(lp, shares):
    """shares: [(weight, share per word)] of the active segments; the network keeps the rest"""
    rest = 1.0 - sum(w for w, _ in shares); return np.log(rest * np.exp(lp) + sum(w * s for w, s in shares) + 1e-30)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--borrow", type=int, default=-1)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); seen = {}

    def corpus_seen(f):
        def g(C_):
            Ct = f(C_); seen["s"] = Ct["train"]; return Ct
        return g

    def test(M, C, tok, seed, res):
        rng = np.random.default_rng(a.seed); home = os.path.join(DATA, f"vega_books_s{a.seed}")
        speech = pickle.load(open(os.path.join(CH.DATA, "adult_speech.pkl"), "rb")); held = speech[-20000:]; speech = speech[:-20000]
        texts = dict(books=dict(val=C["val"][:1500], test=C["test"][:3000]), speech=dict(val=held[:1500], test=held[5000:8000]))
        if a.borrow < 0:
            for name, sents, src in (("books", seen["s"], "the books of Stage 65's books group (Project Gutenberg)"),
                                     ("speech", speech, "CHILDES adult utterances (MacWhinney 2000), all but the last 20,000")):
                if not os.path.exists(os.path.join(home, name, "segment.json")):
                    save(EP.Store(M, sents, rng), os.path.join(home, name), name=name, holds=src, words=int(sum(map(len, sents))), written_by=f"vega_books_s{a.seed}", format="state (64 of 320 hidden units, PCA) -> next word id")
        folders = dict(books=os.path.join(home if a.borrow < 0 else os.path.join(DATA, f"vega_books_s{a.borrow}"), "books"), speech=os.path.join(home, "speech"))
        segs = {k: Segment(f) for k, f in folders.items() if a.borrow < 0 or k == "books"}
        out = dict(seed=a.seed, borrowed_from=a.borrow if a.borrow >= 0 else None, segments={k: dict(s.meta, load_seconds=s.load_seconds) for k, s in segs.items()}, fitted={}, logp_per_word={}, built={})
        prep = {d: {sp: prepared(M, texts[d][sp]) for sp in ("val", "test")} for d in texts}
        par = {}
        for k, s in segs.items():                               # each segment's weight and sharpness, fitted on validation text of its own kind
            lp, H, y, _ = prep[k]["val"]; med = s.median_distance(H[:5000]); sc = {}
            for t in sorted({t for _, t in GRID}):
                sh = s.share(H, y, t * med)
                for lam in sorted({l for l, _ in GRID}): sc[(lam, t)] = float(mixed(lp, [(lam, sh)]).mean())
            b = max(sc, key=sc.get); par[k] = (b[0], b[1] * med); out["fitted"][k] = dict(weight=b[0], sharpness=b[1], val_logp=round(sc[b], 4), val_logp_none=round(float(lp.mean()), 4))
        conds = [(), ("books",)] + ([("speech",), ("books", "speech")] if "speech" in segs else [])
        tn, _ = PC.nouns(); pairs = SG.adult_tests(tn, np.random.default_rng(43))["question_ever"]
        pg, pb = prepared(M, [p[0] for p in pairs]), prepared(M, [p[1] for p in pairs])
        for d in texts:
            lp, H, y, _ = prep[d]["test"]; out["logp_per_word"][d] = {}
            for c in conds:
                t0 = time.time(); v = mixed(lp, [(par[k][0], segs[k].share(H, y, par[k][1])) for k in c])
                out["logp_per_word"][d]["+".join(c) or "none"] = round(float(v.mean()), 4)
                if d == "books": out.setdefault("ms_per_word", {})["+".join(c) or "none"] = round(1000 * (time.time() - t0) / len(lp), 3)
        for c in conds:
            tot = []
            for lp, H, y, cut in (pg, pb):
                v = mixed(lp, [(par[k][0], segs[k].share(H, y, par[k][1])) for k in c]); tot.append(np.add.reduceat(v, cut[:-1]))
            out["built"]["+".join(c) or "none"] = round(float(np.mean(tot[0] > tot[1])), 4)
        out["megabytes_active"] = {"+".join(c) or "none": round(sum(segs[k].meta["megabytes"] for k in c), 1) for c in conds}
        json.dump(out, open(os.path.join(OUT, f"s{a.seed}{'_borrow' + str(a.borrow) if a.borrow >= 0 else ''}.json"), "w"), indent=1); print(json.dumps(out), flush=True)
        raise SystemExit

    GM.evaluate = test; GM.corpus_big = corpus_seen(GM.corpus_big); OO.OUT = "/tmp/segments_scratch"
    OO.run(argparse.Namespace(cond="books", seed=a.seed))


if __name__ == "__main__":
    main()
