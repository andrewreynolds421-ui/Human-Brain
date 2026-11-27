#!/usr/bin/env python3
"""Stage 94: is VEGA's reading network too small to carry a story? Bigger readers, raised on connected text.

Stage 91: the reading network (one 256-unit recurrent layer, raised on single sentences) cannot use the story so
far; teaching it connected text afterwards only stopped the story from hurting. The BabyLM findings (Warstadt et al.
2023; Hu et al. 2024) are that at child-scale data the architecture, the objective and the amount of computation all
matter. Here three readers learn from scratch, the same way, from the same books read in running order:

  --kind gru  --width 256  --layers 1     the size of VEGA's reading network (control)
  --kind gru  --width 1024 --layers 2     the same kind, ~16x the recurrent weights
  --kind attn --width 384  --layers 6     attention over everything read so far (a small causal transformer)

Text: the training-split books of the graded catalogue present on the machine, last 5 % of each held out; sentences
that share a 6-word run with the old reader's test / validation text or a passage-test question are dropped. A
lesson is 32 runs of 192 running tokens, each starting at a sentence start; sentences are separated by an end mark.
Words: those seen 5+ times; every word is also given its last 1 / 2 / 3 letters (as Stage 58), so unknown words
keep their endings. Output: adaptive softmax over the vocabulary (frequency order). MODEL throughout.

Measured (held-out text, never read): log-probability per word of a sentence read alone and read after the
sentences before it; the passage test (Stage 86) by the reader alone, alone / in context, with the most-frequent
baseline. Checkpoints (half precision, without optimiser state) every 10 minutes; stops cleanly at TIME_LIMIT
minutes and continues when started again.

    WORDS=40000000 python scripts/big_reader.py --kind gru --width 1024 --layers 2 --seed 0
"""
import argparse, gzip, json, math, os, sys, time, zlib
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
OUT = os.environ.get("EV_OUT", os.path.join(ROOT, "results", "big_reader"))
DATA = os.environ.get("EV_DATA", "/home/claude/models/big_reader")
WORDS = int(float(os.environ.get("WORDS", 40e6))); BS, LEN, E = 32, 192, 256
TIME_LIMIT = float(os.environ.get("TIME_LIMIT", 300)); T0 = time.time()
PASSAGES = os.environ.get("PASSAGES", os.path.join(ROOT, "data", "passages.json.gz"))
EOS = "</s>"


def grams(s): return {zlib.crc32(" ".join(s[i:i + 6]).encode()) for i in range(len(s) - 5)}


def books(P):
    import grammar as GR, grammar_memory as GM
    C = GM.corpus_big(GR.corpus()); held = set()
    for x in C["test"] + C["val"]: held |= grams(x)
    for bd in P["test"]:
        for it in P["test"][bd]: held |= grams(it["sent"])
    cat = [c for c in json.load(open(os.path.join(ROOT, "data", "grades", "catalogue.json")))["books"] if c["split"] == "train" and os.path.isdir(c["dir"])]
    if os.environ.get("READER_BOOKS") == "bundle": cat = [c for c in cat if "/books_graded/" not in c["dir"]]     # the books the GitHub runners have (Stage 94's readers)
    train, heldout = [], []
    for c in cat:
        ss = [s for s in GR.sentences(GR.book_text(c["dir"])) if not (grams(s) & held)]; k = max(30, len(ss) // 20)
        if len(ss) > 5 * k: train.append(ss[:-k]); heldout.append(ss[-k:])
    return train, heldout


class Vocab:
    def __init__(s, train):
        cnt = Counter(w for b in train for x in b for w in x)
        s.itos = ["<pad>", "<unk>", EOS] + [w for w, n in cnt.most_common() if n >= 5]; s.stoi = {w: i for i, w in enumerate(s.itos)}
        s.ends = [{x: i + 1 for i, x in enumerate(sorted({w[-k:] for w in s.itos[3:] if len(w) > k}))} for k in (1, 2, 3)]

    def enc(s, words):                                                              # (ids, three ending ids) per word
        return [s.stoi.get(w, 1) for w in words], [[s.ends[k].get(w[-k - 1:], 0) if len(w) > k + 1 and w != EOS else 0 for w in words] for k in range(3)]


def build(torch, V, n_ends, kind, width, layers):
    nn = torch.nn

    class Reader(nn.Module):
        def __init__(s):
            super().__init__()
            s.base = nn.Embedding(V, E, padding_idx=0); s.tabs = nn.ModuleList([nn.Embedding(n + 1, E, padding_idx=0) for n in n_ends]); s.drop = nn.Dropout(0.1)
            if kind == "gru": s.rnn = nn.GRU(E, width, layers, batch_first=True, dropout=0.1 if layers > 1 else 0.0)
            else:
                s.inp = nn.Linear(E, width); s.pos = nn.Embedding(1024, width)
                s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, 6, 4 * width, 0.1, batch_first=True, norm_first=True), layers); s.ln = nn.LayerNorm(width)
            s.o = nn.AdaptiveLogSoftmaxWithLoss(width, V, cutoffs=[2000, 10000], div_value=4.0); s.width = width

        def embed(s, x, en):
            return s.base(x) + sum(t(en[:, k]) for k, t in enumerate(s.tabs))

        def states(s, x, en):                                                       # state after each token (predicts the next)
            e = s.drop(s.embed(x, en))
            if kind == "gru": return s.drop(s.rnn(e)[0])
            T = x.shape[1]; h = s.inp(e) + s.pos(torch.arange(T).clamp(max=1023))[None]
            return s.ln(s.tr(h, mask=torch.triu(torch.full((T, T), float("-inf")), 1), is_causal=True))

        def logp(s, x, en):                                                         # log-probability of token t+1 given tokens <= t: (B, T-1)
            h = s.states(x[:, :-1], en[:, :, :-1]); y = x[:, 1:]; keep = (y > 0).reshape(-1); out = torch.zeros(y.numel())
            out[keep] = s.o(h.reshape(-1, h.shape[-1])[keep], y.reshape(-1)[keep]).output; return out.reshape(y.shape)
    return Reader()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--kind", choices=("gru", "attn"), default="gru")
    ap.add_argument("--width", type=int, default=256); ap.add_argument("--layers", type=int, default=1); a = ap.parse_args()
    import torch
    torch.manual_seed(940 + a.seed); torch.set_num_threads(int(os.environ.get("THREADS", 2))); os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    tag = f"bigreader_{a.kind}{a.width}x{a.layers}_"; path = os.path.join(DATA, f"{tag}s{a.seed}.pt"); ck = path + ".ckpt"
    P = json.load(gzip.open(PASSAGES, "rt")); NT = int(os.environ.get("EV_NTEST", 0)) or None
    train, heldout = books(P); VB = Vocab(train); V = len(VB.itos)
    net = build(torch, V, [len(d) for d in VB.ends], a.kind, a.width, a.layers); npar = sum(p.numel() for p in net.parameters())
    info = dict(kind=a.kind, width=a.width, layers=a.layers, parameters=npar, books=len(train), train_words=int(sum(len(x) for b in train for x in b)), vocabulary=V, words_budget=WORDS)
    print(info, flush=True)
    # each book as one token stream with sentence ends; lessons start at sentence starts
    streams, starts = [], []
    for b in train:
        ids, en = VB.enc([w for x in b for w in x + [EOS]]); streams.append((np.array(ids, np.int64), np.array(en, np.int64)))
        st = np.cumsum([0] + [len(x) + 1 for x in b[:-1]]); starts.append(st[st < len(ids) - LEN - 1])
    pb = np.array([len(s) for s in starts], float); pb /= pb.sum(); rng = np.random.default_rng(941 + a.seed)

    def lesson():
        xs, es = [], []
        for _ in range(BS):
            b = int(rng.choice(len(streams), p=pb)); i = int(rng.choice(starts[b])); xs.append(streams[b][0][i:i + LEN + 1]); es.append(streams[b][1][:, i:i + LEN + 1])
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(es))

    def score(contexts, sents, bs=48):                                               # log-probability of each sentence (and its end mark) after its context
        out = np.zeros(len(sents)); order = np.argsort([len(c) + len(s) for c, s in zip(contexts, sents)], kind="stable"); net.eval()
        with torch.no_grad():
            for k in range(0, len(order), bs):
                g = order[k:k + bs]; seqs = [[EOS] + list(contexts[i]) + list(sents[i]) + [EOS] for i in g]; L = max(map(len, seqs))
                x = torch.zeros(len(g), L, dtype=torch.long); en = torch.zeros(len(g), 3, L, dtype=torch.long)
                for r, q in enumerate(seqs): ids, ee = VB.enc(q); x[r, :len(q)] = torch.tensor(ids); en[r, :, :len(q)] = torch.tensor(ee)
                lp = net.logp(x, en).numpy()
                for r, i in enumerate(g): n = len(sents[i]) + 1; e = len(seqs[r]) - 1; out[i] = lp[r, e - n:e].sum()
        net.train(); return out

    er = np.random.default_rng(94); ev = []
    while len(ev) < (NT or 600):
        b = heldout[int(er.integers(len(heldout)))]; i = int(er.integers(5, len(b))); ctx = []; j = i - 1
        while j >= 0 and len(ctx) + len(b[j]) + 1 <= 150: ctx = b[j] + [EOS] + ctx; j -= 1
        if 5 <= len(b[i]) <= 40 and ctx: ev.append((ctx[:-1] + [EOS], b[i]))

    def measure(NT=NT):
        S = [s for _, s in ev][:NT]; nw = sum(len(s) + 1 for s in S); m = dict(heldout_logp_per_word=dict(alone=round(float(score([[]] * len(S), S).sum() / nw), 4), in_context=round(float(score([c for c, _ in ev][:len(S)], S).sum() / nw), 4)))
        m["passages"] = {"alone": {}, "in_context": {}, "most_frequent": {}}
        for bd, its in P["test"].items():
            its = its[:NT]; y = np.array([it["y"] for it in its]); nc = len(its[0]["cands"])
            S2 = [it["sent"][:it["pos"]] + [c[0]] + it["sent"][it["pos"] + 1:] for it in its for c in it["cands"]]
            ctxs = []
            for it in its:
                c, a0 = [], 0
                for e in it["ends"]: c += it["story"][a0:e + 1] + [EOS]; a0 = e + 1
                ctxs += [c] * nc
            m["passages"]["alone"][bd] = round(float(np.mean(score([[]] * len(S2), S2).reshape(-1, nc).argmax(1) == y)), 4)
            m["passages"]["in_context"][bd] = round(float(np.mean(score(ctxs, S2).reshape(-1, nc).argmax(1) == y)), 4)
            m["passages"]["most_frequent"][bd] = round(float(np.mean([max(range(nc), key=lambda j: (it["story"].count(it["cands"][j][0]), -j)) == it["y"] for it in its])), 4)
        for k in ("alone", "in_context", "most_frequent"): m["passages"][k + "_mean"] = round(float(np.mean(list(m["passages"][k].values()))), 4)
        return m

    steps = WORDS // (BS * LEN); start = 0; log = []; used = 0.0; done = os.path.exists(path)
    lr0 = 1e-3 if a.kind == "gru" else 5e-4; opt = torch.optim.Adam(net.parameters(), lr=lr0)
    if done: net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in torch.load(path).items()})
    elif os.path.exists(ck):
        z = torch.load(ck, weights_only=False); net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in z["net"].items()}); start = z["step"]; log = z["log"]; rng.bit_generator.state = z["rng"]; used = z["used"]
        print("resumed at step", start, "of", steps, flush=True)

    def save_ck(step):
        torch.save(dict(net={k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()}, step=step, log=log, rng=rng.bit_generator.state, used=used + time.time() - T0), ck + ".tmp"); os.replace(ck + ".tmp", ck)
    last = time.time(); partial = False
    if not done:
        for step in range(start, steps):
            if (time.time() - T0) / 60 > TIME_LIMIT: save_ck(step); partial = True; print("time limit: stopped at step", step, "of", steps, flush=True); break
            if time.time() - last > 600: save_ck(step); last = time.time()
            for g in opt.param_groups: g["lr"] = lr0 * min(1.0, (step - start + 1) / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))   # warm-up (also after a restart), cosine decay
            x, en = lesson(); loss = -net.logp(x, en).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if (step + 1) % int(os.environ.get("LOG_EVERY", 250)) == 0: log.append(dict(step=step + 1, words=(step + 1) * BS * LEN, minutes=round((used + time.time() - T0) / 60, 1), loss=round(float(loss.detach()), 3))); print(log[-1], flush=True)
        if not partial:
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items()}, path)
            if os.path.exists(ck): os.remove(ck)
    out = dict(seed=a.seed, **info, finished=not partial, words_read=(log[-1]["words"] if log else 0), training=log, **measure(60 if partial else NT))                    # a stopped run gets a short interim test
    json.dump(out, open(os.path.join(OUT, f"{tag}s{a.seed}.json"), "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "training"}), flush=True)


if __name__ == "__main__":
    main()
