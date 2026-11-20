#!/usr/bin/env python3
"""Stage 67: a memory of everything VEGA has read.

Until now the network kept only what its weights absorbed; every sentence was read alone and then gone. Here it
also keeps an episodic record (MODEL: a stand-in for a hippocampal store, not spiking): at every word of
everything it reads, the state it was in (its 320 hidden units, reduced to 64 numbers) is filed together with
the word that came next. When reading later, it looks up the moments most like the present one and what
followed them, and mixes that with its own expectation (weight and sharpness FITTED on the validation items).

Two stores:
  lifelong  every word of the books it was trained on (Stage 65's books group; 5.3 M moments)
  passage   the 30 sentences just before the test sentence in the same book ("what I have just read")
Test: Stage 66's grade-by-grade reading test (a blanked content word, four choices), with
  none / lifelong / passage / lifelong + passage, and as a control the passage store filled with 30 sentences
  from elsewhere in the same grade's books (same kind of text, not what came before).
Items are also split by whether the missing word occurs in the 30 sentences before.

    python scripts/episodic.py --seed 0
"""
import argparse, json, os, pickle, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar_memory as GM, graded_reading as GD, output_order as OO   # noqa: E402
OUT = os.path.join(ROOT, "results", "episodic")
DIM, K, WINDOW = 64, 32, 30
GRID = [(lam, t) for lam in (0.05, 0.1, 0.2, 0.4) for t in (0.06, 0.25, 1.0, 4.0)]


def feats(M, sents, P=None, mu=None, bs=128):
    """per sentence: log P of each word (and the end) from the network, the state before each of them, and their ids"""
    torch = M.torch; M.m.eval(); out = [None] * len(sents)
    idx = np.argsort([len(s) for s in sents], kind="stable")
    with torch.no_grad():
        for k in range(0, len(idx), bs):
            g = idx[k:k + bs]; x, objs, om = M.batch([dict(words=sents[i]) for i in g]); o, _ = M.m(x[:, :-1], objs, om); y = M._targets(x[:, 1:])
            keep = (y > 0).reshape(-1); full = torch.zeros(y.numel())
            full[keep] = M.m.o(o.reshape(-1, o.shape[-1])[keep], y.reshape(-1)[keep]).output
            full = full.reshape(y.shape).numpy(); h = o.numpy(); yy = y.numpy()
            for r, i in enumerate(g):
                n = len(sents[i]) + 1; hh = h[r, :n]
                if P is not None: hh = (hh - mu) @ P.T
                out[i] = (full[r, :n].copy(), hh.astype(np.float32), yy[r, :n].astype(np.int32))
    return out


class Store:
    """the lifelong record: states (64 numbers, 8 bits each) in a searchable index, and the word that followed each"""
    def __init__(self, M, sents, rng, log=print):
        import faiss
        self.faiss = faiss; t0 = time.time()
        samp = [sents[i] for i in rng.choice(len(sents), min(len(sents), 8000), replace=False)]
        H = np.concatenate([f[1] for f in feats(M, samp)]); self.mu = H.mean(0)
        _, _, Vt = np.linalg.svd(H - self.mu, full_matrices=False); self.P = Vt[:DIM].astype(np.float32)
        Z = (H - self.mu) @ self.P.T
        nlist = int(min(2048, max(16, len(Z) // 40)))
        self.index = faiss.index_factory(DIM, f"IVF{nlist},SQ8"); self.index.train(Z.astype(np.float32)); vals = []
        for k in range(0, len(sents), 20000):
            F = feats(M, sents[k:k + 20000], self.P, self.mu)
            self.index.add(np.concatenate([f[1] for f in F])); vals.append(np.concatenate([f[2] for f in F]))
        self.vals = np.concatenate(vals); self.index.nprobe = 16
        self.megabytes = round((self.index.ntotal * (DIM + 8) + self.vals.nbytes) / 1e6, 1)
        log("store", self.index.ntotal, "moments,", self.megabytes, "MB,", round((time.time() - t0) / 60, 1), "min", flush=True)

    def search(self, Z):
        D, I = self.index.search(Z, K); return D, self.vals[np.maximum(I, 0)], I >= 0


def knn_share(y, D, V, ok, T):
    """the share of the remembered moments (nearer ones count more) that were followed by each word in question"""
    D = np.where(ok, D, 0.0); w = np.exp(-(D - D[:, :1]) / T) * ok
    return (w * (V == y[:, None])).sum(1) / np.maximum(w.sum(1), 1e-9)


def passage_search(Z, Kp, Vp):
    if len(Kp) == 0: return np.full((len(Z), K), np.inf, np.float32), np.zeros((len(Z), K), np.int32)
    d = ((Z[:, None, :] - Kp[None]) ** 2).sum(-1); o = np.argsort(d, 1)[:, :K]
    D = np.take_along_axis(d, o, 1); V = Vp[o]
    if D.shape[1] < K:
        pad = K - D.shape[1]; D = np.pad(D, ((0, 0), (0, pad)), constant_values=np.inf); V = np.pad(V, ((0, 0), (0, pad)))
    return D.astype(np.float32), V


def locate(S):
    """where each sentence of a grade's test text sits (first occurrence)"""
    pos = {}
    for i, s in enumerate(S): pos.setdefault(tuple(s), i)
    return pos


def run_items(M, store, items, S, shift):
    """for each item: per option, the network's word log-probs and the neighbours from each store"""
    pos = locate(S); rec = []
    need = set()
    for it in items:
        i = pos.get(tuple(it["sent"]))
        if i is None: continue
        for j in range(max(0, i - WINDOW), i): need.add(j)
        c = (i + shift) % len(S)
        for j in range(max(0, c - WINDOW), c): need.add(j)
    need = sorted(need); F = dict(zip(need, feats(M, [S[j] for j in need], store.P, store.mu)))
    opts = [it["sent"][:it["pos"]] + [o] + it["sent"][it["pos"] + 1:] for it in items for o in it["options"]]
    FO = feats(M, opts, store.P, store.mu)
    Dl, Vl, okl = store.search(np.concatenate([f[1] for f in FO])); cut = np.cumsum([0] + [len(f[0]) for f in FO])
    for n, it in enumerate(items):
        i = pos.get(tuple(it["sent"]))
        if i is None: continue
        r = dict(answer=it["answer"], opts=[])
        for name, c in (("passage", i), ("elsewhere", (i + shift) % len(S))):
            js = list(range(max(0, c - WINDOW), c))
            r[name] = (np.concatenate([F[j][1][:-1] for j in js]) if js else np.zeros((0, DIM), np.float32),   # the end-of-sentence moments are left out
                       np.concatenate([F[j][2][:-1] for j in js]) if js else np.zeros(0, np.int32))
        r["in_passage"] = it["options"][it["answer"]] in {w for j in range(max(0, i - WINDOW), i) for w in S[j]}
        for q in range(4):
            f = FO[4 * n + q]; a, b = cut[4 * n + q], cut[4 * n + q + 1]; o = dict(lp=f[0], y=f[2], life=(Dl[a:b], Vl[a:b], okl[a:b]))
            for name in ("passage", "elsewhere"):
                o[name] = passage_search(f[1], *r[name])
            r["opts"].append(o)
        rec.append(r)
    return rec


def score(rec, cond, par):
    """par: {store: (weight, sharpness)}; the network's own probability keeps the rest of the weight"""
    hit = []
    for r in rec:
        tot = []
        for o in r["opts"]:
            p = np.exp(o["lp"]); rest = 1.0; mix = 0.0
            for part in ([] if cond == "none" else cond.split("+")):
                lam, T = par[part]
                if part == "lifelong": D, V, ok = o["life"]
                else: D, V = o[part]; ok = np.isfinite(D)
                mix = mix + lam * knn_share(o["y"], D, V, ok, T); rest -= lam
            tot.append(np.log(rest * p + mix + 1e-30).sum())
        hit.append(int(np.argmax(tot)) == r["answer"])
    return np.array(hit)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    G = pickle.load(open("/home/claude/models/graded/data.pkl", "rb")); seen = {}

    def corpus_seen(f):
        def g(C_):
            Ct = f(C_); seen["s"] = Ct["train"]; return Ct
        return g

    def test(M, C, tok, seed, res):
        rng = np.random.default_rng(a.seed); read = seen["s"][:a.limit] if a.limit else seen["s"]; had = {tuple(s) for s in read}
        store = Store(M, read, rng)
        out = dict(seed=a.seed, moments=int(store.index.ntotal), megabytes=store.megabytes, words_read=int(sum(map(len, read))), fitted={}, test={})
        val = [r for b in GD.BANDS for r in run_items(M, store, [it for it in G["items"]["val"][b] if tuple(it["sent"]) not in had], G["S"]["val"][b], 97)]
        med = float(np.median(np.concatenate([o["life"][0][:, 0] for r in val for o in r["opts"]]))) + 1e-6
        par = {}
        for part in ("lifelong", "passage"):                    # each store's weight and sharpness fitted alone on the validation items
            sc = {(lam, t): float(score(val, part, {part: (lam, t * med)}).mean()) for lam, t in GRID}
            b_ = max(sc, key=sc.get); par[part] = (b_[0], b_[1] * med); out["fitted"][part] = dict(weight=b_[0], sharpness=b_[1], val=round(sc[b_], 4))
        par["elsewhere"] = par["passage"]
        conds = ("none", "lifelong", "passage", "lifelong+passage", "elsewhere", "lifelong+elsewhere")
        out["val_none"] = round(float(score(val, "none", par).mean()), 4)
        allhit = {c: [] for c in conds}; inp = []
        for b in GD.BANDS:
            rec = run_items(M, store, [it for it in G["items"]["test"][b] if tuple(it["sent"]) not in had], G["S"]["test"][b], 2003)
            out["test"][b] = dict(n=len(rec))
            for c in conds:
                h = score(rec, c, par); out["test"][b][c] = round(float(h.mean()), 4); allhit[c].append(h)
            inp.append(np.array([r["in_passage"] for r in rec]))
        inp = np.concatenate(inp); out["share_answer_in_passage"] = round(float(inp.mean()), 4)
        for c in conds:
            h = np.concatenate(allhit[c]); out["test"].setdefault("all", {})[c] = round(float(h.mean()), 4)
            out["test"].setdefault("answer_in_passage", {})[c] = round(float(h[inp].mean()), 4); out["test"].setdefault("answer_not_in_passage", {})[c] = round(float(h[~inp].mean()), 4)
        json.dump(out, open(os.path.join(OUT, f"s{a.seed}{'_limit' if a.limit else ''}.json"), "w"), indent=1); print(json.dumps(out), flush=True)
        raise SystemExit

    GM.evaluate = test; GM.corpus_big = corpus_seen(GM.corpus_big); OO.OUT = "/tmp/episodic_scratch"
    OO.run(argparse.Namespace(cond="books", seed=a.seed))


if __name__ == "__main__":
    main()
