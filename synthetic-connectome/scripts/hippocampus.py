#!/usr/bin/env python3
"""Stage 75: a hippocampus beside the reading network.

Stage 74: VEGA's reading network does not keep track of what happens in a story, and training it on stories hurt
its reading. In the brain, who-what-where is bound by the hippocampus, beside the cortex. Here VEGA gets one
(MODEL: a rate model wired after the hippocampal circuit, not spiking, not from a connectome):

  entorhinal cortex in   at every word, from the reading network's state (frozen): a "what" key, a "where /
                         with whom" value and an encoding strength (how much this moment should be stored)
  perforant path         the direct entorhinal -> CA3 path (learned): a dense CA3 address for every key, which
                         carries the retrieval cue (first version without it: the sparse codes of question and
                         memory never overlapped, recall was empty and nothing could be learned)
  dentate gyrus          a fixed random expansion of the key to 1,024 cells, of which only the 5 % most active
                         stay on (pattern separation: similar moments get distinct codes)       ASSUMED sizes
  CA3                    fast synapses written during the story (reset for each story): the DG code is bound to
                         the value; the change is the value minus what CA3 already recalls for that code (the CA1
                         comparison) - so a new fact replaces an old one instead of piling on top
  recall                 a question cue recalls through CA3; the recalled pattern is fed back as a new cue once
                         more (pattern completion in two steps: "ball" -> "with Fred" -> "in the barn")
  CA1 / EC out           compares the recall with the question and says which of the offered words is the answer
Learned by error-driven training across many stories (EC in, CA1 / EC out): the hippocampus learns HOW to store
and recall; WHAT is stored is written in one exposure per story.

Groups (four networks each, each on one of Stage 73's reading networks, which stay unchanged):
  hippocampus   the full circuit
  no_dg         CA3 addressed by the perforant path only (no dentate pattern separation)
  cortex_only   the same CA1 / EC out read-out from the reading network's own state, no hippocampus
                (control: same training, same answers to learn from)
Training: 100,000 world stories (Stage 74's world, training names / objects / places, half of the questions
about carried objects); tests: Stage 74's four test sets.

    python scripts/hippocampus.py --cond hippocampus --seed 0
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar_memory as GM, output_order as OO, world as W   # noqa: E402
OUT = os.environ.get("HIPPO_OUT", os.path.join(ROOT, "results", "hippocampus"))
DATA = os.environ.get("HIPPO_DATA", "/home/claude/models/hippocampus")
N_DG, ACTIVE, D_KEY, D_VAL, BS = 1024, 0.05, 64, 64, 32
STORIES = int(os.environ.get("STORIES", 100_000))


def query_of(it):
    """a test item -> question words, candidate answer words (true first), kind"""
    t = it["options"]
    if it["kind"] == "who_has": return ["who", "has", "the", t[0][-1]], [o[0] for o in t], it["kind"]
    subj = t[0][:t[0].index("is")]; return ["where", "is"] + subj, [o[-1] for o in t], it["kind"]


def make_items(rng, n):
    out = []
    for i in range(n):
        want = ("carried", "who_has", "where_person", "carried")[i % 4]
        while True:
            it = W.question(rng, *W.story(rng, W.NAMES, W.OBJECTS, W.PLACES, int(rng.integers(4, 11))), W.PLACES, want)
            if it: out.append(it); break
    return out


def build(M, cond, seed):
    torch = M.torch; nn = torch.nn; torch.manual_seed(1000 + seed); H = 320; E = M.m.e.base.weight.shape[1] if hasattr(M.m.e, "base") else 64

    class Hippo(nn.Module):
        def __init__(s):
            super().__init__()
            s.ec_k = nn.Linear(H, D_KEY); s.ec_v = nn.Linear(H, D_VAL); s.ec_g = nn.Linear(H, 1); s.q_k = nn.Linear(H, D_KEY)
            s.hop = nn.Linear(D_VAL, D_KEY); s.pp = nn.Linear(D_KEY, N_DG)
            g = torch.Generator().manual_seed(77 + seed); s.register_buffer("dg", torch.randn(D_KEY, N_DG, generator=g) / D_KEY ** 0.5)   # fixed, not learned
            s.ca1 = nn.Sequential(nn.Linear(H + 2 * D_VAL, 256), nn.Tanh(), nn.Linear(256, E))

        def code(s, k):
            """CA3 address of a key: the direct entorhinal path (perforant path, learned, dense) plus, except in
            no_dg, the dentate gyrus's sparse code (fixed expansion, 5 % active)"""
            a = s.pp(k)
            if cond != "no_dg":
                d = torch.relu(k @ s.dg); kth = d.topk(int(ACTIVE * N_DG), dim=-1).values[..., -1:]
                d = d * (d >= kth); a = a + d / (d.norm(dim=-1, keepdim=True) + 1e-6)
            return a / (a.norm(dim=-1, keepdim=True) + 1e-6)

        def forward(s, hs, mask, q):
            """hs (B, T, H) reading states over the story, mask (B, T), q (B, H) state after the question"""
            B, T, _ = hs.shape
            if cond == "cortex_only": z = torch.zeros(B, D_VAL); return s.ca1(torch.cat([q, z, z], -1))
            Mem = torch.zeros(B, N_DG, D_VAL)
            keys = s.code(s.ec_k(hs)); vals = torch.tanh(s.ec_v(hs)); gate = torch.sigmoid(s.ec_g(hs)) * mask[..., None]
            for t in range(T):                                   # CA3 fast synapses, delta rule (CA1 mismatch)
                c = keys[:, t]; old = torch.einsum("bn,bnv->bv", c, Mem)
                Mem = Mem + gate[:, t, :, None] * c[:, :, None] * (vals[:, t] - old)[:, None, :]
            r1 = torch.einsum("bn,bnv->bv", s.code(s.q_k(q)), Mem)
            r2 = torch.einsum("bn,bnv->bv", s.code(s.hop(r1)), Mem)
            return s.ca1(torch.cat([q, r1, r2], -1))
    return Hippo()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cond", choices=("hippocampus", "no_dg", "cortex_only"), default="hippocampus"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)

    def run(M, C, tok, seed, res):
        torch = M.torch; M.m.eval()
        for p in M.m.parameters(): p.requires_grad_(False)
        net = build(M, a.cond, a.seed); opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        def emb(words):
            with torch.no_grad(): return M.m.e(torch.tensor([[M.enc([w])[1] for w in words]]))[0]

        def batch(items):
            seqs, qpos, spos, cands = [], [], [], []
            for it in items:
                q, c, _ = query_of(it); seqs.append(it["story"] + q); spos.append(len(it["story"])); qpos.append(len(it["story"]) + len(q)); cands.append(c)
            x, objs, om = M.batch([dict(words=s) for s in seqs])
            with torch.no_grad(): o, _ = M.m(x, objs, om)                     # state after each token (position 0 = start)
            T = max(spos); hs = o[:, 1:T + 1]; mask = torch.zeros(len(items), T)
            for i, n in enumerate(spos): mask[i, :n] = 1
            q = o[torch.arange(len(items)), torch.tensor(qpos)]
            ce = torch.stack([emb(c) for c in cands])                         # (B, 4, E)
            return hs * mask[..., None], mask, q, ce

        def predict(items):
            net.eval(); hits = []
            with torch.no_grad():
                for k in range(0, len(items), 64):
                    hs, mask, q, ce = batch(items[k:k + 64]); z = net(hs, mask, q)
                    hits.append((torch.einsum("be,bce->bc", z, ce).argmax(1) == 0).numpy())
            net.train(); return np.concatenate(hits)

        path = os.path.join(DATA, f"{a.cond}_s{a.seed}.pt"); log = []
        rng = np.random.default_rng(500 + a.seed); val = make_items(np.random.default_rng(99), 400); t0 = time.time()
        if os.path.exists(path): net.load_state_dict(torch.load(path)); log = json.load(open(path + ".json"))
        else:
            for step in range(STORIES // BS):
                items = make_items(rng, BS); hs, mask, q, ce = batch(items)
                perm = torch.tensor(np.array([rng.permutation(4) for _ in items]))      # answer position shuffled
                ce = torch.gather(ce, 1, perm[..., None].expand(-1, -1, ce.shape[-1])); target = (perm == 0).float().argmax(1)
                logits = torch.einsum("be,bce->bc", net(hs, mask, q), ce)
                loss = torch.nn.functional.cross_entropy(logits, target); opt.zero_grad(); loss.backward(); opt.step()
                if (step + 1) % 500 == 0 or step + 1 == STORIES // BS:
                    v = float(predict(val).mean()); log.append(dict(stories=(step + 1) * BS, val=round(v, 4), minutes=round((time.time() - t0) / 60, 1))); print(log[-1], flush=True)
            torch.save(net.state_dict(), path); json.dump(log, open(path + ".json", "w"))
        out = dict(cond=a.cond, seed=a.seed, training=log)
        for name, items in W.test_sets(seed=0).items():
            hit = predict(items); r = dict(all=round(float(hit.mean()), 4))
            for k in ("where_person", "who_has"): m = np.array([it["kind"] == k for it in items]); r[k] = round(float(hit[m].mean()), 4)
            car = np.array([it["carried"] for it in items]); r["carried_objects"] = round(float(hit[car].mean()), 4)
            cue = np.array([W.last_mention(it) for it in items]); r["when_cue_wrong"] = round(float(hit[~cue].mean()), 4); out[name] = r
        json.dump(out, open(os.path.join(OUT, f"{a.cond}_s{a.seed}.json"), "w"), indent=1); print(json.dumps(out), flush=True)
        raise SystemExit

    GM.evaluate = run; OO.DATA = "/home/claude/models/day_night/year4_cycle"; OO.OUT = "/tmp/hippo_scratch"; os.makedirs(OO.OUT, exist_ok=True)
    OO.run(argparse.Namespace(cond="books", seed=a.seed))


if __name__ == "__main__":
    main()
