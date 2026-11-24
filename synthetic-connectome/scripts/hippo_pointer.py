#!/usr/bin/env python3
"""Stage 77: the hippocampus stores the cortex's own word representations.

Stage 75-76: the hippocampus tracked stories but could not answer about places and objects it never met in a
story - what it stored for "garden" was its own learned code, read out by a map learned for the trained words
only. In the brain the hippocampus is thought to store an index to the cortex's own representation of the thing
(hippocampal indexing theory, Teyler & DiScenna 1986), which the cortex then reinstates. Here:

  content   what CA3 stores and recalls is the reading network's representation of a word (its embedding, the
            same one it reads with; frozen) - chosen from the last six words by attention from the reading
            network's state ("garden" in "mary went to the garden", "fred" when "fred took the lamp")
  key       likewise a word representation from the last six words, mapped into CA3 through the perforant path
            and the dentate gyrus (as Stage 75)
  recall    the question's key word ("ball" in "where is the ball"); a second recall uses the first recall's
            word as the cue ("ball" -> fred -> barn)
  answer    the recalled representation is compared with the representations of the offered words (cosine);
            nothing word-specific is learned on the way out
Learned (error-driven, across stories): which recent word to store, under which, when, and how to combine the
two recalls. Four networks, 100,000 stories, Stage 74 / 75 tests; compare Stage 75.
Control *look_back*: no memory - the question's state attends directly over the story's words and states (a
reader looking back over the page) and the attended word is the answer; same read-out and training.

    python scripts/hippo_pointer.py --cond pointer --seed 0
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar_memory as GM, output_order as OO, world as W, hippocampus as HP   # noqa: E402
OUT = os.environ.get("HIPPO_OUT", os.path.join(ROOT, "results", "hippocampus"))
DATA = os.environ.get("HIPPO_DATA", "/home/claude/models/hippocampus")
WIN, N_DG, ACTIVE, D_KEY = 6, 1024, 0.05, 64


def build(M, seed, cond="pointer"):
    torch = M.torch; nn = torch.nn; torch.manual_seed(2000 + seed); H = 320; E = M.m.e.base.embedding_dim

    class Pointer(nn.Module):
        def __init__(s):
            super().__init__()
            s.pick_v = nn.Linear(H, WIN); s.pick_k = nn.Linear(H, WIN); s.gate = nn.Linear(H, 1)
            s.wk = nn.Linear(E, D_KEY); s.pp = nn.Linear(D_KEY, N_DG); s.q_pick = nn.Linear(H, 1); s.mix = nn.Linear(H, 2)
            s.temp = nn.Parameter(torch.tensor(5.0))
            g = torch.Generator().manual_seed(77 + seed); s.register_buffer("dg", torch.randn(D_KEY, N_DG, generator=g) / D_KEY ** 0.5)

        def code(s, e):
            k = s.wk(e); a = s.pp(k); d = torch.relu(k @ s.dg); kth = d.topk(int(ACTIVE * N_DG), dim=-1).values[..., -1:]
            a = a + (d * (d >= kth)) / ((d * (d >= kth)).norm(dim=-1, keepdim=True) + 1e-6)
            return a / (a.norm(dim=-1, keepdim=True) + 1e-6)

        def forward(s, hs, es, mask, q, qe, qmask):
            """hs (B,T,H) reading states, es (B,T,E) word representations of the story; q (B,H) state after the
            question, qe (B,L,E) its words, qmask (B,L)"""
            B, T, _ = hs.shape
            if cond == "look_back":                              # control: attend over the story's words from the question, no memory
                sc = (es * s.lb(q)[:, None, :]).sum(-1) / es.shape[-1] ** 0.5 + (hs * s.lb_h(q)[:, None, :]).sum(-1) / H ** 0.5 + torch.log(mask + 1e-9)   # scaled, as in attention
                return (torch.softmax(sc, -1)[..., None] * es).sum(1)
            pad = torch.cat([torch.zeros(B, WIN - 1, es.shape[-1]), es], 1)
            win = torch.stack([pad[:, j:j + T] for j in range(WIN)], 2)                   # (B,T,WIN,E): last six words
            v = (torch.softmax(s.pick_v(hs), -1)[..., None] * win).sum(2)
            kw = (torch.softmax(s.pick_k(hs), -1)[..., None] * win).sum(2)
            keys = s.code(kw); g = torch.sigmoid(s.gate(hs)) * mask[..., None]
            Mem = torch.zeros(B, N_DG, es.shape[-1])
            for t in range(T):
                c = keys[:, t]; old = torch.einsum("bn,bne->be", c, Mem)
                Mem = Mem + g[:, t, :, None] * c[:, :, None] * (v[:, t] - old)[:, None, :]
            qw = s.query_word(q, qe, qmask)
            r1 = torch.einsum("bn,bne->be", s.code(qw), Mem); r2 = torch.einsum("bn,bne->be", s.code(r1), Mem)
            w = torch.softmax(s.mix(q), -1); return w[:, :1] * r1 + w[:, 1:] * r2

        def query_word(s, q, qe, qmask):
            """the question's key word: attention over the question's words from the state after it"""
            sc = (qe * s.qproj(q)[:, None, :]).sum(-1) + torch.log(qmask + 1e-9)
            return (torch.softmax(sc, -1)[..., None] * qe).sum(1)

    net = Pointer(); net.qproj = nn.Linear(H, M.m.e.base.embedding_dim); net.lb = nn.Linear(H, M.m.e.base.embedding_dim); net.lb_h = nn.Linear(H, H); return net


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--cond", choices=("pointer", "look_back"), default="pointer"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)

    def run(M, C, tok, seed, res):
        torch = M.torch; M.m.eval()
        for p in M.m.parameters(): p.requires_grad_(False)
        net = build(M, a.seed, a.cond); opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        def batch(items):
            seqs, spos, cands, ql = [], [], [], []
            for it in items:
                q, c, _ = HP.query_of(it); seqs.append(it["story"] + q); spos.append(len(it["story"])); ql.append(len(q)); cands.append(c)
            x, objs, om = M.batch([dict(words=s) for s in seqs])
            with torch.no_grad():
                o, _ = M.m(x, objs, om); e = M.m.e(x)
                ce = M.m.e(torch.tensor([[M.enc([w])[1] for w in c] for c in cands]))
            T = max(spos); L = max(ql); B = len(items); mask = torch.zeros(B, T); qmask = torch.zeros(B, L); qe = torch.zeros(B, L, e.shape[-1])
            for i in range(B):
                mask[i, :spos[i]] = 1; qmask[i, :ql[i]] = 1; qe[i, :ql[i]] = e[i, spos[i] + 1:spos[i] + 1 + ql[i]]
            q = o[torch.arange(B), torch.tensor([spos[i] + ql[i] for i in range(B)])]
            return o[:, 1:T + 1] * mask[..., None], e[:, 1:T + 1] * mask[..., None], mask, q, qe, qmask, ce

        def logits(b):
            z = net(*b[:6]); ce = b[6]
            return net.temp * torch.einsum("be,bce->bc", z / (z.norm(dim=-1, keepdim=True) + 1e-6), ce / (ce.norm(dim=-1, keepdim=True) + 1e-6))

        def predict(items):
            net.eval(); hits = []
            with torch.no_grad():
                for k in range(0, len(items), 64): hits.append((logits(batch(items[k:k + 64])).argmax(1) == 0).numpy())
            net.train(); return np.concatenate(hits)

        path = os.path.join(DATA, f"{a.cond}_s{a.seed}.pt"); ck = path + ".ckpt"; log = []; start = 0
        rng = np.random.default_rng(500 + a.seed); val = HP.make_items(np.random.default_rng(99), 400); t0 = time.time(); last = time.time()
        if os.path.exists(ck):                                   # resume after a machine restart
            z = torch.load(ck, weights_only=False); net.load_state_dict(z["net"]); opt.load_state_dict(z["opt"]); start = z["step"]; log = z["log"]
            rng.bit_generator.state = z["rng"]; print("resumed at", start * HP.BS, "stories", flush=True)
        for step in range(start, HP.STORIES // HP.BS):
            if time.time() - last > 600:
                torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, log=log, rng=rng.bit_generator.state), ck + ".tmp"); os.replace(ck + ".tmp", ck); last = time.time()
            items = HP.make_items(rng, HP.BS); b = list(batch(items))
            perm = torch.tensor(np.array([rng.permutation(4) for _ in items]))
            b[6] = torch.gather(b[6], 1, perm[..., None].expand(-1, -1, b[6].shape[-1])); target = (perm == 0).float().argmax(1)
            loss = torch.nn.functional.cross_entropy(logits(b), target); opt.zero_grad(); loss.backward(); opt.step()
            if (step + 1) % 500 == 0 or step + 1 == HP.STORIES // HP.BS:
                log.append(dict(stories=(step + 1) * HP.BS, val=round(float(predict(val).mean()), 4), minutes=round((time.time() - t0) / 60, 1))); print(log[-1], flush=True)
        torch.save(net.state_dict(), path)
        if os.path.exists(ck): os.remove(ck)
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
