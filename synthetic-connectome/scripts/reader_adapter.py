#!/usr/bin/env python3
"""Stage 95: a Stage 94 reading network (big_reader.py) in the place of VEGA's old reading network.

The hippocampus (hippo_events.py) asks its reading network for: a state after every word, each word's embedding,
embeddings of answer words, and the probability of a sentence. BigM gives those from a big_reader model, in the
old layout (position 0 = before the first word, position i + 1 = after word i), while the model itself reads the
story the way it was raised: sentences separated by the end mark.

    BIG_READER=/path/bigreader_gru1024x2_s0.pt python scripts/hippo_events.py --sup 1 --roles --seed 0
    python scripts/reader_adapter.py /path/bigreader_gru1024x2_s0.pt       # grade test (Stage 53 / 66 items) with this reader
"""
import gzip, json, os, pickle, re, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import big_reader as BR  # noqa: E402


class BigM:
    def __init__(s, torch, path, passages=BR.PASSAGES):
        m = re.search(r"bigreader_(gru|attn)(\d+)x(\d+)_s(\d+)", os.path.basename(path)); kind, width, layers = m.group(1), int(m.group(2)), int(m.group(3))
        P = json.load(gzip.open(passages, "rt")); train, _ = BR.books(P); s.VB = BR.Vocab(train); s.torch = torch; s.H = width; s.name = os.path.basename(path)
        s.net = BR.build(torch, len(s.VB.itos), [len(d) for d in s.VB.ends], kind, width, layers)
        s.net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in torch.load(path).items()}); s.net.eval()   # fails if the books (vocabulary) differ from the reader's
        for p in s.net.parameters(): p.requires_grad_(False)
        outer = s

        class Emb:
            base = s.net.base
            def __call__(self, x): return outer._gather(outer.net.embed(x["tok"], x["en"]), x)

        class Shim:
            e = Emb()
            def __call__(self, x, objs=None, om=None): return outer._gather(outer.net.states(x["tok"], x["en"]), x), None
            def eval(self): return self
            def parameters(self): return outer.net.parameters()
        s.m = Shim()

    def _tensors(s, seqs):
        torch = s.torch; L = max(map(len, seqs)); x = torch.zeros(len(seqs), L, dtype=torch.long); en = torch.zeros(len(seqs), 3, L, dtype=torch.long)
        for r, q in enumerate(seqs): ids, ee = s.VB.enc(q); x[r, :len(q)] = torch.tensor(ids); en[r, :, :len(q)] = torch.tensor(ee)
        return x, en

    def batch(s, items):
        """items: dict(words=[...], ends=[index of each sentence's last word]) -> the model's tokens (with end marks) and where each word sits"""
        seqs, pos = [], []
        for it in items:
            ends = set(it.get("ends") or []); toks = [BR.EOS]; wp = [0]
            for i, w in enumerate(it["words"]):
                toks.append(str(w)); wp.append(len(toks) - 1)
                if i in ends: toks.append(BR.EOS)
            seqs.append(toks); pos.append(wp)
        Lw = max(map(len, pos)); idx = s.torch.zeros(len(items), Lw, dtype=s.torch.long)
        for r, wp in enumerate(pos): idx[r, :len(wp)] = s.torch.tensor(wp)
        tok, en = s._tensors(seqs); return dict(tok=tok, en=en, idx=idx), None, None

    def _gather(s, h, x):
        return s.torch.gather(h, 1, x["idx"][..., None].expand(-1, -1, h.shape[-1]))

    def word_emb(s, words):
        tok, en = s._tensors([list(map(str, words))]); return s.net.embed(tok, en)[0]

    def score(s, contexts, sents, bs=48):
        """log-probability of each sentence (and its end mark) after its context (tokens, end marks included; [] = alone)"""
        torch = s.torch; out = np.zeros(len(sents)); order = np.argsort([len(c) + len(x) for c, x in zip(contexts, sents)], kind="stable")
        with torch.no_grad():
            for k in range(0, len(order), bs):
                g = order[k:k + bs]; seqs = [[BR.EOS] + list(contexts[i]) + list(map(str, sents[i])) + [BR.EOS] for i in g]; tok, en = s._tensors(seqs); lp = s.net.logp(tok, en).numpy()
                for r, i in enumerate(g): n = len(sents[i]) + 1; e = len(seqs[r]) - 1; out[i] = lp[r, e - n:e].sum()
        return out

    def logp(s, sents): return list(s.score([[]] * len(sents), sents))


def grade_test(path):
    """Stage 53's reading-level test (cloze: one content word blanked, 4 options of the same part of speech and frequency)"""
    import torch
    torch.set_num_threads(int(os.environ.get("THREADS", 2))); M = BigM(torch, path); D = pickle.load(open("/home/claude/models/graded/data.pkl", "rb"))["items"]["test"]; res = {}
    for b, items in D.items():
        sents = [it["sent"][:it["pos"]] + [o] + it["sent"][it["pos"] + 1:] for it in items for o in it["options"]]
        res[b] = round(float(np.mean(np.array(M.logp(sents)).reshape(len(items), 4).argmax(1) == np.array([it["answer"] for it in items]))), 4)
    res["mean"] = round(float(np.mean(list(res.values()))), 4); res["reads_at_75"] = [b for b in D if res[b] >= 0.75]; return res


if __name__ == "__main__":
    out = {os.path.basename(p): grade_test(p) for p in sys.argv[1:]}; print(json.dumps(out, indent=1))
    os.makedirs(os.path.join(ROOT, "results", "big_reader"), exist_ok=True); json.dump(out, open(os.path.join(ROOT, "results", "big_reader", "grade_test.json"), "w"), indent=1)
