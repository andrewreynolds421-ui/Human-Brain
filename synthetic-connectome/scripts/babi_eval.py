#!/usr/bin/env python3
"""Stage 78: VEGA's hippocampus on the bAbI tasks (an outside test with published results).

bAbI (Weston et al. 2015, "Towards AI-complete question answering"; v1.2, English, 1k; copy from
github.com/booydar/babilong data/tasks_1-20_v1-2.zip): short generated stories with questions. Task 1 (single
supporting fact: "Where is Mary?") and task 2 (two supporting facts: "Where is the milk?" after someone carried it)
are what the hippocampus of Stage 77 does; their wording differs from VEGA's world ("journeyed", "went back to",
"got ... there", "discarded", "grabbed"; places such as hallway and bathroom; names Sandra and Daniel).

  zero-shot   Stage 77's networks as they are - never trained on bAbI
  trained     the same, then trained on the task's 1,000 training questions (5 passes), as published models are

Answer: the place among bAbI's six (bathroom, bedroom, garden, hallway, kitchen, office) whose word
representation is closest to the recalled one - an open choice over all places, as in the published results.
The reading network stays frozen in both.

    python scripts/babi_eval.py --seed 0
"""
import argparse, json, os, re, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar_memory as GM, output_order as OO, hippocampus as HP, hippo_pointer as PT   # noqa: E402
BABI = "/home/claude/corpora/babi/tasks_1-20_v1-2/en"
OUT = os.path.join(ROOT, "results", "babi")
PLACES = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]
TASKS = {1: "qa1_single-supporting-fact", 2: "qa2_two-supporting-facts"}


def words(s):
    return re.findall(r"[a-z]+", s.lower())


def load(task, split):
    """bAbI file -> items in the form the hippocampus code reads: story words, question as statement options"""
    items, story = [], []
    for line in open(os.path.join(BABI, f"{TASKS[task]}_{split}.txt")):
        n, rest = line.split(" ", 1)
        if n == "1": story = []
        if "\t" in rest:
            q, ans, _ = rest.split("\t"); q = words(q)                       # "where is the milk" / "where is mary"
            subj = q[2:]; opts = [subj + ["is", "in", "the", ans]] + [subj + ["is", "in", "the", p] for p in PLACES if p != ans]
            items.append(dict(story=[w for s in story for w in s], options=opts, kind="where_object" if subj[0] == "the" else "where_person"))
        else:
            story.append(words(rest))
    return items


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)

    def run(M, C, tok, seed, res):
        torch = M.torch; M.m.eval()
        for p in M.m.parameters(): p.requires_grad_(False)

        def batch(items):
            seqs, spos, cands, ql = [], [], [], []
            for it in items:
                q, c, _ = HP.query_of(it); seqs.append(it["story"] + q); spos.append(len(it["story"])); ql.append(len(q)); cands.append(c)
            x, objs, om = M.batch([dict(words=s) for s in seqs])
            with torch.no_grad():
                o, _ = M.m(x, objs, om); e = M.m.e(x); ce = M.m.e(torch.tensor([[M.enc([w])[1] for w in c] for c in cands]))
            T = max(spos); L = max(ql); B = len(items); mask = torch.zeros(B, T); qmask = torch.zeros(B, L); qe = torch.zeros(B, L, e.shape[-1])
            for i in range(B):
                mask[i, :spos[i]] = 1; qmask[i, :ql[i]] = 1; qe[i, :ql[i]] = e[i, spos[i] + 1:spos[i] + 1 + ql[i]]
            q = o[torch.arange(B), torch.tensor([spos[i] + ql[i] for i in range(B)])]
            return o[:, 1:T + 1] * mask[..., None], e[:, 1:T + 1] * mask[..., None], mask, q, qe, qmask, ce

        def logits(net, b):
            z = net(*b[:6]); ce = b[6]
            return net.temp * torch.einsum("be,bce->bc", z / (z.norm(dim=-1, keepdim=True) + 1e-6), ce / (ce.norm(dim=-1, keepdim=True) + 1e-6))

        def accuracy(net, items):
            net.eval(); hit = []
            with torch.no_grad():
                for k in range(0, len(items), 32): hit.append((logits(net, batch(items[k:k + 32])).argmax(1) == 0).numpy())
            return float(np.concatenate(hit).mean())

        out = dict(seed=a.seed, tasks={})
        for t in TASKS:
            train, test = load(t, "train"), load(t, "test"); r = dict(n_test=len(test), story_words_max=max(len(i["story"]) for i in test))
            net = PT.build(M, a.seed); net.load_state_dict(torch.load(os.path.join(PT.DATA, f"pointer_s{a.seed}.pt")))
            r["zero_shot"] = round(accuracy(net, test), 4); print(t, "zero-shot", r["zero_shot"], flush=True)
            opt = torch.optim.Adam(net.parameters(), lr=3e-4); rng = np.random.default_rng(a.seed)
            for ep in range(5):
                net.train()
                for k in range(0, len(train), 16):
                    b = list(batch([train[i] for i in rng.permutation(len(train))[k:k + 16]]))
                    perm = torch.tensor(np.array([rng.permutation(6) for _ in range(b[6].shape[0])]))
                    b[6] = torch.gather(b[6], 1, perm[..., None].expand(-1, -1, b[6].shape[-1])); target = (perm == 0).float().argmax(1)
                    loss = torch.nn.functional.cross_entropy(logits(net, b), target); opt.zero_grad(); loss.backward(); opt.step()
            r["trained_1k"] = round(accuracy(net, test), 4); print(t, "trained", r["trained_1k"], flush=True)
            out["tasks"][TASKS[t]] = r
        json.dump(out, open(os.path.join(OUT, f"s{a.seed}.json"), "w"), indent=1); print(json.dumps(out), flush=True)
        raise SystemExit

    GM.evaluate = run; OO.DATA = "/home/claude/models/day_night/year4_cycle"; OO.OUT = "/tmp/hippo_scratch"; os.makedirs(OO.OUT, exist_ok=True)
    OO.run(argparse.Namespace(cond="books", seed=a.seed))


if __name__ == "__main__":
    main()
