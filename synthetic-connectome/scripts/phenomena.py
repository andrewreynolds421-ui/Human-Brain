#!/usr/bin/env python3
"""Stage 60: teaching grammar phenomena the same way agreement was taught.

The recipe that taught agreement (Stages 56-58): many nouns, many frames, sentences where the easy cue misleads,
word form visible - and only GRAMMATICAL sentences, learned by predicting the next word (no negative examples,
as children mostly get none). Here it is applied to four of the phenomena the model fails on BLiMP:

  determiners   this / that + singular, these / those + plural ("these big dogs were ..."), as subject and object
  reflexives    himself / herself / itself / themselves agreeing with the subject in number and gender, across
                phrases and relative clauses ("the queen near the boys saw herself"); the nearest subject binds
                ("the girls said that the king saw himself")
  any / ever    only after a licensor: not / n't / never / no / only / a question ("the boy didn't see any dogs",
                "did the girl ever ...", "no cat saw any birds"); otherwise "some"; a licensor inside a relative
                clause does not reach the main clause ("the boys that didn't sing saw some birds")
  what / that   "knew what the girl saw" (gap) vs "knew that the girl saw the dog" (no gap), also across
                "said that" (long distance) and subject gaps ("knew who saw the dog")

Also: words not in the vocabulary still show their ending (a reader sees the letters of an unknown word);
before, unknown words had none, which is part of why BLiMP's adult-vocabulary agreement items failed.

GROUPS (four networks each): base = books + Stage 57-58 agreement pages (150 k per pass), with word endings and
unknown-word endings; phenomena = base + 150 k phenomenon pages per pass. Two passes.
TESTS: BLiMP (its own vocabulary and frames, never seen) by phenomenon; built pairs with held-out verbs and nouns.

    python scripts/phenomena.py run --cond phenomena --seed 0
"""
import argparse, glob, json, os, re, sys, time
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar as GR, grammar_memory as GM, picture_check as PC, frames as FR  # noqa: E402
DATA = "/home/claude/models/phenomena"
OUT = os.path.join(ROOT, "results", "phenomena")
FEM = {"girl", "woman", "mother", "queen", "princess", "lady", "aunt", "sister", "daughter", "wife", "grandmother", "witch", "actress", "nurse", "maid", "widow", "duchess"}
MASC = {"boy", "man", "father", "king", "prince", "uncle", "brother", "son", "husband", "grandfather", "gentleman", "lord", "duke", "knight", "farmer", "soldier", "sailor"}
ADJ = ["big", "little", "old", "young", "good", "bad", "tall", "small", "happy", "poor", "kind", "brave", "pretty", "clever", "red", "white", "black", "grey"]
HELD_ADJ = ["green", "strange", "quiet", "lucky"]
SAY = ["said", "thought", "knew", "heard", "believed", "felt", "dreamed", "hoped"]
ASK = ["knew", "wondered", "asked", "forgot", "remembered", "learned", "found out", "guessed"]
PH = ["det", "refl", "npi", "gap"]


def refl(n, plural):
    if plural: return "themselves"
    return "herself" if n in FEM else "himself" if n in MASC else "itself"


def phenomenon_page(rng, nouns, verbs, adjs=ADJ):
    N = lambda: nouns[rng.integers(len(nouns))]; V = lambda: str(rng.choice(verbs)); A = lambda: str(rng.choice(adjs))
    kind = PH[rng.integers(len(PH))]; a, b, c = N(), N(), N(); na, nb, nc = (int(rng.integers(2)) for _ in range(3))
    be = lambda n: ("were" if n else "was") if rng.integers(2) else ("are" if n else "is")
    if kind == "det":
        det = lambda n: str(rng.choice(["these", "those"] if n else ["this", "that"]))
        adj = [A()] if rng.integers(2) else []
        if rng.integers(2): w = [det(na)] + adj + [a[na], be(na)] + list(FR.PRED[rng.integers(len(FR.PRED))])
        else: w = ["the", a[na], V(), det(nb)] + adj + [b[nb]]
    elif kind == "refl":
        t = rng.integers(4)
        if t == 0: w = ["the", a[na], V(), refl(a[0], na)]
        elif t == 1: w = ["the", a[na], str(rng.choice(FR.PREPS)), "the", b[nb], V(), refl(a[0], na)]
        elif t == 2: w = ["the", a[na], "that", "the", b[nb], V(), V(), refl(a[0], na)]
        else: w = ["the", a[na], str(rng.choice(SAY)), "that", "the", b[nb], V(), refl(b[0], nb)]
    elif kind == "npi":
        t = rng.integers(7); npi = str(rng.choice(["any", "any", "ever"]))
        if t == 0: w = ["the", a[na], "didn't"] + (["see", "any", b[1]] if npi == "any" else ["ever", "see", "the", b[nb]])
        elif t == 1: w = ["the", a[na], "never"] + ([V(), "any", b[1]] if npi == "any" else ["saw", "the", b[nb]])
        elif t == 2: w = ["no", a[na], V(), "any", b[1]]
        elif t == 3: w = ["only", "the", a[na], "ever", V(), "the", b[nb]]
        elif t == 4: w = ["did", "the", a[na], "ever", "see", "the", b[nb]]
        elif t == 5: w = ["the", a[na], "that", "didn't", "sing", V(), "some", b[1]]     # licensor in the relative clause: no NPI outside it
        else: w = ["the", a[na], V(), "some", b[1]]                                      # no licensor: "some"
    else:
        t = rng.integers(5)
        if t == 0: w = ["the", a[na], str(rng.choice(ASK)).split()[0], "what", "the", b[nb], V()]
        elif t == 1: w = ["the", a[na], str(rng.choice(SAY)), "that", "the", b[nb], V(), "the", c[nc]]
        elif t == 2: w = ["the", a[na], str(rng.choice(ASK)).split()[0], "who", V(), "the", b[nb]]
        elif t == 3: w = ["the", a[na], str(rng.choice(ASK)).split()[0], "what", "the", b[nb], str(rng.choice(SAY)), "that", "the", c[nc], V()]
        else: w = ["what", "did", "the", a[na], "see"]
    return dict(words=[x for x in w if x], kind=kind)


BASE = {"helped": "help", "loved": "love", "met": "meet", "called": "call", "saw": "see", "found": "find", "liked": "like"}


def built_tests(nouns, verbs, adjs, rng, n=400):
    """good / bad pairs per phenomenon (the two differ only where the phenomenon is), with held-out verbs,
    adjectives, prepositions and (optionally) nouns"""
    out = {k: [] for k in ("det", "refl_number", "refl_gender", "refl_across", "npi", "npi_scope", "gap_wh", "gap_that")}
    gendered = [p for p in nouns if p[0] in FEM | MASC]
    for _ in range(n):
        a, b = nouns[rng.integers(len(nouns))], nouns[rng.integers(len(nouns))]; na = int(rng.integers(2)); nb = 1 - na
        v = str(rng.choice(verbs)); vb = BASE[v]; adj = str(rng.choice(adjs)); prep = str(rng.choice(FR.HELD_PREPS))
        out["det"].append((["the", b[nb], v, "these" if na else "this", adj, a[na]], ["the", b[nb], v, "this" if na else "these", adj, a[na]]))
        out["refl_number"].append((["the", a[na], v, refl(a[0], na)], ["the", a[na], v, "itself" if na else "themselves"]))
        if gendered:
            g = gendered[rng.integers(len(gendered))]
            out["refl_gender"].append((["the", g[0], v, refl(g[0], 0)], ["the", g[0], v, "himself" if g[0] in FEM else "herself"]))
        out["refl_across"].append((["the", a[na], prep, "the", b[nb], v, refl(a[0], na)], ["the", a[na], prep, "the", b[nb], v, refl(b[0], nb)]))
        out["npi"].append((["the", a[na], "didn't", vb, "any", b[1]], ["the", a[na], "really", v, "any", b[1]]))
        out["npi_scope"].append((["the", a[na], "that", "liked", "the", b[nb], "didn't", vb, "any", "dogs"],
                                 ["the", a[na], "that", "didn't", "like", "the", b[nb], v, "any", "dogs"]))
        out["gap_wh"].append((["the", a[na], "knew", "what", "the", b[nb], v], ["the", a[na], "knew", "that", "the", b[nb], v]))
        out["gap_that"].append((["the", a[na], "knew", "that", "the", b[nb], v, "the", "dog"], ["the", a[na], "knew", "what", "the", b[nb], v, "the", "dog"]))
    return out


class Reader(PC.CheckNet):
    """Stage 58's reader (word + endings), where unknown words keep their endings too"""
    def __init__(self, vocab, seed):
        super().__init__(vocab, seed, False)
        import spelling as SP
        SP.spelled(self); torch = self.torch
        ids = self.m.e.ids; del self.m.e._buffers["ids"]; self.m.e.register_buffer("ids", ids, persistent=False)   # grows with unknown words
        self.V = len(self.itos); e = self.m.e
        self.end_idx = []
        for k in (1, 2, 3):
            ends = sorted({w[-k:] for w in self.itos[4:] if len(w) > k}); self.end_idx.append({x: i + 1 for i, x in enumerate(ends)})
        V0 = self.V

        def fwd(x):
            base = e.base(torch.where(x >= V0, torch.full_like(x, 3), x))
            return base + sum(t(e.ids[k][x]) for k, t in enumerate(e.tabs))
        e.forward = fwd

    def enc(self, w):
        out = [1]
        for x in w:
            i = self.stoi.get(x)
            if i is None:                                   # unknown word: a new id that shares the "unknown" word unit
                i = len(self.itos); self.itos.append(x); self.stoi[x] = i
                col = self.torch.tensor([[self.end_idx[k].get(x[-k - 1:], 0) if len(x) > k + 1 else 0] for k in range(3)])
                self.m.e.ids = self.torch.cat([self.m.e.ids, col], 1)
            out.append(i)
        return out + [2]

    def _targets(self, y):
        return self.torch.where(y >= self.V, self.torch.full_like(y, 3), y)

    def step(self, items, questions=False):
        torch = self.torch; self.m.train()
        x, objs, om = self.batch([dict(words=it["words"]) for it in items])
        out, _ = self.m(x[:, :-1], objs, om); y = self._targets(x[:, 1:]); keep = (y > 0).reshape(-1)
        loss = self.m.o(out.reshape(-1, out.shape[-1])[keep], y.reshape(-1)[keep]).loss
        self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0); self.opt.step()

    def token_logp_items(self, items, bs=128):
        torch = self.torch; self.m.eval(); out = [None] * len(items)
        idx = np.argsort([len(it["words"]) for it in items], kind="stable")
        with torch.no_grad():
            for k in range(0, len(idx), bs):
                g = idx[k:k + bs]; x, objs, om = self.batch([items[i] for i in g]); o, att = self.m(x[:, :-1], objs, om); y = self._targets(x[:, 1:])
                keep = (y > 0).reshape(-1); full = torch.zeros(y.numel())
                full[keep] = self.m.o(o.reshape(-1, o.shape[-1])[keep], y.reshape(-1)[keep]).output
                full = full.reshape(y.shape).numpy()
                for r, i in enumerate(g): out[i] = (full[r, :len(items[i]["words"]) + 1], att[r].numpy())
        return out


def run(a):
    os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    C = GR.corpus(); Ct = GM.corpus_big(C); tok = Counter(w for s in C["train"] for w in s)
    train_nouns, held = PC.nouns(); look = PC.Look()
    words = {w for p in train_nouns for w in p} | set(FR.PREPS) | set(FR.HELD_PREPS) | set(FR.REL) | set(FR.HELD_REL) | {w for p in FR.PRED + FR.HELD_PRED for w in p} | \
            set(ADJ) | set(HELD_ADJ) | {w for s in SAY + ASK for w in s.split()} | {"the", "that", "see", "sees", "saw", "is", "are", "was", "were", "this", "these", "those", "himself",
                                                                                    "herself", "itself", "themselves", "any", "ever", "some", "no", "only", "never", "didn't", "did", "not", "what", "who", "sing", "like", "really", "dogs", "dog", "help", "love", "meet", "call"}
    cnt = Counter(w for s in Ct["train"] for w in s)
    M = Reader(sorted({w for w, c in cnt.items() if c >= 5} | words), a.seed)
    torch = M.torch; path = os.path.join(DATA, f"{a.cond}_s{a.seed}.pt"); ck = path + ".ckpt"
    st = dict(ep=0, b=0, log=[])
    if os.path.exists(path) and os.path.exists(path + ".json"):
        M.m.load_state_dict(torch.load(path)); st = json.load(open(path + ".json"))
    else:
        if os.path.exists(ck):
            z = torch.load(ck, weights_only=False); M.m.load_state_dict(z["m"]); M.opt.load_state_dict(z["o"]); st = z["st"]; print("resumed at pass", st["ep"] + 1, "batch", st["b"], flush=True)
        last = time.time()
        while st["ep"] < 2:
            rng = np.random.default_rng(1000 * a.seed + st["ep"]); t0 = time.time()
            items = [dict(words=s) for s in Ct["train"]] + [dict(words=FR.make_page(rng, train_nouns, look)["words"]) for _ in range(150_000)]
            if a.cond == "phenomena":
                items += [phenomenon_page(rng, train_nouns, FR.REL) for _ in range(150_000)]
            order = np.argsort([len(it["words"]) for it in items], kind="stable")
            groups = [order[i:i + 64] for i in range(0, len(order), 64)]; gorder = rng.permutation(len(groups))
            for bi in range(st["b"], len(gorder)):
                M.step([items[i] for i in groups[gorder[bi]]]); st["b"] = bi + 1
                if time.time() - last > 600:
                    torch.save(dict(m=M.m.state_dict(), o=M.opt.state_dict(), st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck); last = time.time()
            M.m.eval(); val = -np.sum(M.logp(C["val"])) / sum(len(s) + 1 for s in C["val"])
            st["log"].append(dict(epoch=st["ep"] + 1, val=round(float(val), 3), minutes=round((time.time() - t0) / 60, 1)))
            print(st["log"][-1], flush=True); st["ep"] += 1; st["b"] = 0
            for gp in M.opt.param_groups: gp["lr"] *= 0.7
            torch.save(dict(m=M.m.state_dict(), o=M.opt.state_dict(), st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck)
        torch.save(M.m.state_dict(), path); json.dump(st, open(path + ".json", "w")); os.remove(ck)
    res = dict(cond=a.cond, seed=a.seed, training=st["log"])
    bt = {}
    for name, nl in (("page_nouns", train_nouns), ("held_out_nouns", held)):
        T = built_tests(nl, FR.HELD_REL, HELD_ADJ, np.random.default_rng(42))
        for k, prs in T.items():
            if not prs: continue
            g = M.logp([p[0] for p in prs]); b = M.logp([p[1] for p in prs]); bt[f"{name}_{k}"] = round(float(np.mean(np.array(g) > np.array(b))), 4)
    res["built"] = bt; print("built", bt, flush=True)
    res = GM.evaluate(M, C, tok, a.seed, res)
    json.dump(res, open(os.path.join(OUT, f"{a.cond}_s{a.seed}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--cond", choices=("base", "phenomena"), default="phenomena"); ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
