#!/usr/bin/env python3
"""Stage 54: a gated slot for the subject.

Stage 52's memory learned number but followed the nearest noun: one mixed state cannot keep "the subject
was singular" while "soldiers" passes through. Here the recurrent memory (GRU, 256 units) gets two SLOTS
beside it (64 units each). Each slot has a write gate, computed from the memory and the current word:
open -> the slot takes in the present; closed -> it keeps what it holds. The slots feed back into the
memory and into the prediction. This is gated working memory as in prefrontal cortex / basal ganglia
models (PBWM: Frank, Loughry & O'Reilly 2001; O'Reilly & Frank 2006), where the basal ganglia decide when a
prefrontal stripe updates and when it maintains.

  learned  - the gates learn only from predicting the next word
  taught   - slot 1's gate is also told (during reading only) where the main clause's subject is, from
             spaCy's parser: open at the subject, closed elsewhere. Upper bound: can a slot that holds the
             subject fix agreement? (Children get no parser, but do get meaning and prosody; MODEL.)
  open     - CONTROL: the same slots with the gates fixed open (the slots then hold only the present word):
             same size, same reading, no gating.

Same books as Stage 52 (4.9 M words), three passes, adaptive softmax output (as Stage 53), four networks
each. Tests: Stage 51's battery (grammar_memory.evaluate), plus where the gates open in the built sentences.

    python scripts/subject_slot.py prepare                    # parse the training sentences (taught gates)
    python scripts/subject_slot.py run --kind learned --seed 0
"""
import argparse, json, os, pickle, sys, time
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar as GR, grammar_memory as GM                                   # noqa: E402
DATA = "/home/claude/models/slot"
OUT = os.path.join(ROOT, "results", "subject_slot")
K, D, PASSES, LAMBDA = 2, 64, 3, 1.0


def corpus():
    C = GR.corpus(); return C, GM.corpus_big(C)


def prepare():
    """main-clause subject position in each training sentence (-1: none; -2: parser tokens do not match)"""
    import spacy
    os.makedirs(DATA, exist_ok=True)
    _, Ct = corpus()
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    pos, t0 = [], time.time()
    for k, (doc, s) in enumerate(zip(nlp.pipe((" ".join(s) for s in Ct["train"]), batch_size=512), Ct["train"])):
        if [t.text for t in doc] != s:
            pos.append(-2); continue
        subj = [t.i for t in doc if t.dep_ in ("nsubj", "nsubjpass") and t.head.dep_ == "ROOT"]
        pos.append(subj[0] if subj else -1)
        if k % 50000 == 0: print(k, round((time.time() - t0) / 60, 1), "min", flush=True)
    pos = np.array(pos)
    pickle.dump(pos, open(os.path.join(DATA, "subject_positions.pkl"), "wb"))
    print("sentences", len(pos), "with a main subject", int((pos >= 0).sum()), "none", int((pos == -1).sum()), "mismatch", int((pos == -2).sum()))


class SlotNet:
    def __init__(self, kind, C, seed, subj=None):
        import torch, torch.nn as nn
        torch.manual_seed(seed); torch.set_num_threads(int(os.environ.get("THREADS", "1")))
        self.torch, self.kind = torch, kind
        cnt = Counter(w for s in C["train"] for w in s)
        self.itos = ["<pad>", "<s>", "</s>", "<unk>"] + sorted(w for w, c in cnt.items() if c >= 5)
        self.stoi = {w: i for i, w in enumerate(self.itos)}; self.vocab = set(self.itos[4:]); V = len(self.itos)

        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.e = nn.Embedding(V, 128, padding_idx=0); s.cell = nn.GRUCell(128 + K * D, 256)
                s.gate = nn.Linear(256 + 128, K); s.cand = nn.Linear(256, K * D); s.d = nn.Dropout(0.25)
                nn.init.constant_(s.gate.bias, -2.0)                          # slots start mostly closed
                s.o = nn.AdaptiveLogSoftmaxWithLoss(256 + K * D, V, cutoffs=[2000, 10000] if V > 12000 else [V // 4, V // 2], div_value=4.0)

            def forward(s, x, kind):
                B, T = x.shape; e = s.d(s.e(x)); h = e.new_zeros(B, 256); m = e.new_zeros(B, K, D); outs, gates = [], []
                for t in range(T):
                    h = s.cell(torch.cat([e[:, t], m.reshape(B, -1)], 1), h)
                    g = torch.ones(B, K) if kind == "open" else torch.sigmoid(s.gate(torch.cat([h, e[:, t]], 1)))
                    c = torch.tanh(s.cand(h)).reshape(B, K, D)
                    m = (1 - g.unsqueeze(-1)) * m + g.unsqueeze(-1) * c
                    outs.append(torch.cat([h, m.reshape(B, -1)], 1)); gates.append(g)
                return s.d(torch.stack(outs, 1)), torch.stack(gates, 1)
        self.m = M(); self.opt = torch.optim.Adam(self.m.parameters(), lr=2e-3)
        self.C, self.seed, self.subj = C, seed, subj

    def enc(self, s):
        return [1] + [self.stoi.get(w, 3) for w in s] + [2]

    def pad(self, seqs):
        L = max(map(len, seqs)); x = np.zeros((len(seqs), L), np.int64)
        for k, q in enumerate(seqs): x[k, :len(q)] = q
        return self.torch.from_numpy(x)

    def train(self, path):
        torch = self.torch; F = torch.nn.functional; tr = self.C["train"]; ck = path + ".ckpt"
        st = dict(ep=0, b=0, log=[], sums=[0.0, 0, 0.0])
        if os.path.exists(ck):
            z = torch.load(ck, weights_only=False); self.m.load_state_dict(z["m"]); self.opt.load_state_dict(z["o"]); st = z["st"]
            print("resumed at pass", st["ep"] + 1, "batch", st["b"], flush=True)
        save_s, last = 60 * float(os.environ.get("SAVE_MIN", "10")), time.time()

        def save():
            torch.save(dict(m=self.m.state_dict(), o=self.opt.state_dict(), st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck)
        idx = np.argsort([len(s) for s in tr], kind="stable"); groups = [idx[i:i + 64] for i in range(0, len(idx), 64)]
        while st["ep"] < PASSES:
            order = np.random.default_rng(1000 * self.seed + st["ep"]).permutation(len(groups)); t0 = time.time(); self.m.train()
            for bi in range(st["b"], len(order)):
                g = groups[order[bi]]; x = self.pad([self.enc(tr[i]) for i in g])
                out, gates = self.m(x[:, :-1], self.kind); y = x[:, 1:]; keep = (y > 0).reshape(-1)
                loss = self.m.o(out.reshape(-1, out.shape[-1])[keep], y.reshape(-1)[keep]).loss; aux = torch.zeros(())
                if self.kind == "taught":
                    sp = self.subj[g]; ok = sp != -2
                    if ok.any():
                        tgt = torch.zeros(len(g), y.shape[1])
                        for r, p in enumerate(sp):
                            if p >= 0: tgt[r, p + 1] = 1.0                    # input at time p+1 is word p
                        mask = (x[:, 1:] > 0).float() * torch.from_numpy(ok.astype(np.float32)).unsqueeze(1)
                        aux = (F.binary_cross_entropy(gates[:, :, 0].clamp(1e-5, 1 - 1e-5), tgt, reduction="none") * mask).sum() / mask.sum()
                self.opt.zero_grad(); (loss + LAMBDA * aux).backward(); torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0); self.opt.step()
                k = int(keep.sum()); st["sums"][0] += float(loss.detach()) * k; st["sums"][1] += k; st["sums"][2] += float(aux.detach())
                st["b"] = bi + 1
                if time.time() - last > save_s:
                    save(); last = time.time()
            self.m.eval()
            val = -np.sum(self.logp(self.C["val"])) / sum(len(s) + 1 for s in self.C["val"])
            st["log"].append(dict(epoch=st["ep"] + 1, train=round(st["sums"][0] / max(st["sums"][1], 1), 3), val=round(float(val), 3),
                                  aux=round(st["sums"][2] / max(len(order), 1), 4), minutes=round((time.time() - t0) / 60, 1)))
            print(st["log"][-1], flush=True)
            st["ep"] += 1; st["b"] = 0; st["sums"] = [0.0, 0, 0.0]
            for gp in self.opt.param_groups: gp["lr"] *= 0.7
            save()
        torch.save(self.m.state_dict(), path); json.dump(st["log"], open(path + ".json", "w")); os.remove(ck)
        self.log = st["log"]

    def run_batches(self, sents, bs=128):
        torch = self.torch; self.m.eval()
        idx = np.argsort([len(s) for s in sents], kind="stable")
        with torch.no_grad():
            for k in range(0, len(idx), bs):
                g = idx[k:k + bs]; x = self.pad([self.enc(sents[i]) for i in g]); out, gates = self.m(x[:, :-1], self.kind); y = x[:, 1:]
                keep = (y > 0).reshape(-1); full = torch.zeros(y.numel())
                full[keep] = self.m.o(out.reshape(-1, out.shape[-1])[keep], y.reshape(-1)[keep]).output
                yield g, full.reshape(y.shape).numpy(), gates.numpy()

    def token_logp(self, sents):
        out = [None] * len(sents)
        for g, lp, _ in self.run_batches(sents):
            for r, i in enumerate(g): out[i] = lp[r, :len(sents[i]) + 1]
        return out

    def logp(self, sents):
        return [float(v.sum()) for v in self.token_logp(sents)]

    def gates_of(self, sents):
        out = [None] * len(sents)
        for g, _, gt in self.run_batches(sents):
            for r, i in enumerate(g): out[i] = gt[r, 1:len(sents[i]) + 1]      # gate when READING word j (input time j+1)
        return out


def gate_profile(M, tok, vocab):
    """mean gate of each slot at each word of the built sentences ('the bird that the soldiers saw is ...')"""
    E2 = GM.controlled(tok, vocab); prof = {}
    for kind in ("pp", "rel"):
        X = [x["good"] for x in E2 if x["kind"] == kind]
        G = M.gates_of(X); n = 6 if kind == "rel" else 5                     # words up to the verb
        names = ["the", "SUBJECT", "that", "the", "noun", "verb", "IS/ARE"] if kind == "rel" else ["the", "SUBJECT", "prep", "the", "noun", "IS/ARE"]
        arr = np.array([g[:len(names)] for g in G])
        prof[kind] = {f"slot{k + 1}": {nm: round(float(arr[:, j, k].mean()), 3) for j, nm in enumerate(names)} for k in range(K)}
    return prof


def run(a):
    os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    C, Ct = corpus(); tok = Counter(w for s in C["train"] for w in s)
    subj = pickle.load(open(os.path.join(DATA, "subject_positions.pkl"), "rb")) if a.kind == "taught" else None
    M = SlotNet(a.kind, Ct, a.seed, subj)
    path = os.path.join(DATA, f"{a.kind}_s{a.seed}.pt")
    if os.path.exists(path) and os.path.exists(path + ".json"):
        M.m.load_state_dict(M.torch.load(path)); M.log = json.load(open(path + ".json"))
    else:
        M.train(path)
    res = dict(model=a.kind, seed=a.seed, training=M.log)
    res = GM.evaluate(M, C, tok, a.seed, res)
    res["gates"] = gate_profile(M, tok, {w for w, c in tok.items() if c >= 3}); print("gates", res["gates"], flush=True)
    json.dump(res, open(os.path.join(OUT, f"{a.kind}_s{a.seed}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("prepare", "run"))
    ap.add_argument("--kind", choices=("learned", "taught", "open"), default="learned"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); prepare() if a.cmd == "prepare" else run(a)


if __name__ == "__main__":
    main()
