#!/usr/bin/env python3
"""Stage 79: how far does VEGA's story comprehension go? All 20 bAbI tasks.

bAbI v1.2 (Weston et al. 2015), English, 10k training examples per task, trained jointly (one network for all 20
tasks), tested on each task's 1,000 test questions; a task is passed at 95 % (the field's convention). Stories are
cut to their last 50 sentences (as published memory networks limit their memory).

The hippocampus of Stage 77 with three additions (MODEL):
  recall while storing   at every word CA3 is also cued from the recent words, and the recalled word joins the
                         recent words as something that can be stored ("mary dropped the milk" can file the milk
                         under where Mary is, which the sentence does not say)
  more recall            three recall steps from the question's main word, and one from a second question word
                         ("what did fred give to JEFF", "is john in the KITCHEN")
  CA1 decision           besides passing a recalled word on, CA1 can answer from a comparison (recall vs. the
                         second question word, for yes / no; how strongly something is recalled, for counting)
Answers are chosen among the task's answer words seen in training (single words, or two for lists / paths),
by closeness to the produced representation. The reading network (Stage 73) stays frozen. VEGA's world stories
(Stage 74, including long ones) are mixed in, a fifth of the lessons, so that skill is kept.

    python scripts/hippo_babi.py --seed 0
"""
import argparse, json, os, re, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar_memory as GM, output_order as OO, world as W, hippocampus as HP   # noqa: E402
BABI = "/home/claude/corpora/babi/tasks_1-20_v1-2"
OUT = os.environ.get("BABI_OUT", os.path.join(ROOT, "results", "babi_all"))
DATA = os.environ.get("BABI_DATA", "/home/claude/models/babi_all")
WIN, N_DG, ACTIVE, D_KEY, BS, MAX_SENT = 8, 512, 0.05, 64, 16, 50
QUESTIONS = int(os.environ.get("QUESTIONS", 120_000))


def words(s):
    return re.findall(r"[a-z]+", s.lower())


def load_task(path):
    items, story = [], []
    for line in open(path):
        n, rest = line.split(" ", 1)
        if n == "1": story = []
        if "\t" in rest:
            q, ans, _ = rest.split("\t")
            items.append(dict(story=[w for s in story[-MAX_SENT:] for w in s], q=words(q), ans=tuple(words(ans.replace(",", " ")))))
        else:
            story.append(words(rest))
    return items


def load_all():
    tasks = {}
    for f in sorted(os.listdir(os.path.join(BABI, "en-10k"))):
        m = re.match(r"qa(\d+)_(.+)_train.txt", f)
        if not m: continue
        t = int(m.group(1)); name = f"qa{t}_{m.group(2)}"
        tr = load_task(os.path.join(BABI, "en-10k", f)); te = load_task(os.path.join(BABI, "en", f.replace("_train", "_test")))
        cands = sorted({it["ans"] for it in tr})
        for it in tr + te: it["cands"] = cands; it["y"] = cands.index(it["ans"]) if it["ans"] in cands else -1
        tasks[t] = dict(name=name, train=tr, test=te, cands=cands)
    return tasks


def world_items(rng, n, test=None):
    """VEGA's world (Stage 74, stories of 4-18 events) in the same form: question words and four candidate words.
    Training items get the candidates shuffled; test items keep the true one first."""
    if test is not None: src = test
    else:
        src = []
        while len(src) < n:
            it = W.question(rng, *W.story(rng, W.NAMES, W.OBJECTS, W.PLACES, int(rng.integers(4, 19))), W.PLACES, ("carried", "who_has", "where_person", "carried")[len(src) % 4])
            if it: src.append(it)
    out = []
    for it in src:
        q, c, _ = HP.query_of(it); cands = [(w,) for w in c]; y = 0
        if test is None:
            order = rng.permutation(len(cands)); cands = [cands[k] for k in order]; y = int(np.nonzero(order == 0)[0][0])
        out.append(dict(story=it["story"], q=q, cands=cands, y=y))
    return out


def build(M, seed):
    torch = M.torch; nn = torch.nn; torch.manual_seed(3000 + seed); H = 320; E = M.m.e.base.embedding_dim

    class Hippo2(nn.Module):
        def __init__(s):
            super().__init__()
            s.pick_k = nn.Linear(H, WIN); s.pick_r = nn.Linear(H, WIN); s.pick_v = nn.Linear(H, WIN + 1); s.gate = nn.Linear(H, 1)
            s.wk = nn.Linear(E, D_KEY); s.pp = nn.Linear(D_KEY, N_DG); s.q1 = nn.Linear(H, E); s.q2 = nn.Linear(H, E)
            s.mix = nn.Linear(H, 6); s.dec = nn.Sequential(nn.Linear(H + 6 * E + 4, 256), nn.Tanh(), nn.Linear(256, E)); s.temp = nn.Parameter(torch.tensor(5.0))
            g = torch.Generator().manual_seed(77 + seed); s.register_buffer("dg", torch.randn(D_KEY, N_DG, generator=g) / D_KEY ** 0.5)

        def code(s, e):
            k = s.wk(e); a = s.pp(k); d = torch.relu(k @ s.dg); kth = d.topk(int(ACTIVE * N_DG), dim=-1).values[..., -1:]
            d = d * (d >= kth); a = a + d / (d.norm(dim=-1, keepdim=True) + 1e-6); return a / (a.norm(dim=-1, keepdim=True) + 1e-6)

        def forward(s, hs, es, mask, q, qe, qmask):
            B, T, _ = hs.shape; Ed = es.shape[-1]
            pad = torch.cat([torch.zeros(B, WIN - 1, Ed), es], 1); win = torch.stack([pad[:, j:j + T] for j in range(WIN)], 2)
            ak = torch.softmax(s.pick_k(hs), -1); ar = torch.softmax(s.pick_r(hs), -1); av = torch.softmax(s.pick_v(hs), -1)
            keys = s.code((ak[..., None] * win).sum(2)); rkeys = s.code((ar[..., None] * win).sum(2)); g = torch.sigmoid(s.gate(hs)) * mask[..., None]
            Mem = torch.zeros(B, N_DG, Ed)
            for t in range(T):
                rec = torch.einsum("bn,bne->be", rkeys[:, t], Mem)                                  # recall while storing
                opts = torch.cat([win[:, t], rec[:, None]], 1); v = (av[:, t, :, None] * opts).sum(1)
                c = keys[:, t]; old = torch.einsum("bn,bne->be", c, Mem)
                Mem = Mem + g[:, t, :, None] * c[:, :, None] * (v - old)[:, None, :]
            read = lambda e: torch.einsum("bn,bne->be", s.code(e), Mem)
            lm = torch.log(qmask + 1e-9)
            c1 = (torch.softmax((qe * s.q1(q)[:, None]).sum(-1) / Ed ** 0.5 + lm, -1)[..., None] * qe).sum(1)
            c2 = (torch.softmax((qe * s.q2(q)[:, None]).sum(-1) / Ed ** 0.5 + lm, -1)[..., None] * qe).sum(1)
            r1 = read(c1); r2 = read(r1); r3 = read(r2); s1 = read(c2)
            cos = lambda a, b: (a * b).sum(-1, keepdim=True) / (a.norm(dim=-1, keepdim=True) * b.norm(dim=-1, keepdim=True) + 1e-6)
            feats = torch.cat([cos(r1, c2), cos(r2, c2), cos(r3, c2), r1.norm(dim=-1, keepdim=True)], -1)
            w = torch.softmax(s.mix(q), -1); opts = [r1, r2, r3, s1, c2]
            z = sum(w[:, i:i + 1] * o for i, o in enumerate(opts)) + w[:, 5:6] * s.dec(torch.cat([q, r1, r2, r3, s1, c1, c2, feats], -1))
            return z
    return Hippo2()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)

    def run(M, C, tok, seed, res):
        torch = M.torch; M.m.eval()
        for p in M.m.parameters(): p.requires_grad_(False)
        net = build(M, a.seed); opt = torch.optim.Adam(net.parameters(), lr=1e-3); cache = {}

        def emb(c):
            if c not in cache:
                with torch.no_grad(): e = M.m.e(torch.tensor([[M.enc([w])[1] for w in c]]))[0]; cache[c] = e[0] + (0.5 * e[1] if len(c) > 1 else 0)
            return cache[c]

        def batch(items):
            seqs = [it["story"] + it["q"] for it in items]; spos = [len(it["story"]) for it in items]; ql = [len(it["q"]) for it in items]
            x, objs, om = M.batch([dict(words=s) for s in seqs])
            with torch.no_grad(): o, _ = M.m(x, objs, om); e = M.m.e(x)
            T = max(max(spos), 1); L = max(ql); B = len(items); mask = torch.zeros(B, T); qmask = torch.zeros(B, L); qe = torch.zeros(B, L, e.shape[-1])
            for i in range(B): mask[i, :spos[i]] = 1; qmask[i, :ql[i]] = 1; qe[i, :ql[i]] = e[i, spos[i] + 1:spos[i] + 1 + ql[i]]
            q = o[torch.arange(B), torch.tensor([spos[i] + ql[i] for i in range(B)])]
            hs = o[:, 1:T + 1]; es = e[:, 1:T + 1]
            if hs.shape[1] < T: hs = torch.nn.functional.pad(hs, (0, 0, 0, T - hs.shape[1])); es = torch.nn.functional.pad(es, (0, 0, 0, T - es.shape[1]))
            ce = torch.stack([torch.stack([emb(c) for c in it["cands"]]) for it in items])
            return hs * mask[..., None], es * mask[..., None], mask, q, qe, qmask, ce

        def logits(b):
            z = net(*b[:6]); ce = b[6]
            return net.temp * torch.einsum("be,bce->bc", z / (z.norm(dim=-1, keepdim=True) + 1e-6), ce / (ce.norm(dim=-1, keepdim=True) + 1e-6))

        def accuracy(items):
            net.eval(); hit = []
            with torch.no_grad():
                for k in range(0, len(items), 32):
                    it = items[k:k + 32]; hit.append((logits(batch(it)).argmax(1).numpy() == np.array([i["y"] for i in it])))
            net.train(); return float(np.concatenate(hit).mean())

        tasks = load_all(); rng = np.random.default_rng(700 + a.seed); T = sorted(tasks)
        path = os.path.join(DATA, f"s{a.seed}.pt"); ck = path + ".ckpt"; start = 0; log = []; t0 = time.time(); last = time.time()
        if os.path.exists(ck):
            z = torch.load(ck, weights_only=False); net.load_state_dict(z["net"]); opt.load_state_dict(z["opt"]); start = z["step"]; log = z["log"]; rng.bit_generator.state = z["rng"]
            print("resumed at step", start, flush=True)
        if not os.path.exists(path):
            for step in range(start, QUESTIONS // BS):
                if time.time() - last > 600:
                    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, log=log, rng=rng.bit_generator.state), ck + ".tmp"); os.replace(ck + ".tmp", ck); last = time.time()
                if rng.random() < 0.2: items = world_items(rng, BS)
                else:
                    t = T[int(rng.integers(len(T)))]; tr = tasks[t]["train"]; items = [tr[i] for i in rng.choice(len(tr), BS, replace=False)]
                b = list(batch(items)); y = torch.tensor([i["y"] for i in items])
                loss = torch.nn.functional.cross_entropy(logits(b), y); opt.zero_grad(); loss.backward(); opt.step()
                if (step + 1) % 1000 == 0 or step + 1 == QUESTIONS // BS:
                    log.append(dict(questions=(step + 1) * BS, minutes=round((time.time() - t0) / 60, 1), loss=round(float(loss.detach()), 3))); print(log[-1], flush=True)
            torch.save(net.state_dict(), path)
            if os.path.exists(ck): os.remove(ck)
        else: net.load_state_dict(torch.load(path))
        out = dict(seed=a.seed, questions=QUESTIONS, training=log, tasks={})
        for t in T:
            acc = accuracy(tasks[t]["test"]); out["tasks"][tasks[t]["name"]] = round(acc, 4); print(tasks[t]["name"], round(acc, 4), flush=True)
        out["passed_95"] = int(sum(v >= 0.95 for v in out["tasks"].values())); out["mean_error"] = round(100 * (1 - np.mean(list(out["tasks"].values()))), 2)
        out["world"] = {k: round(accuracy(world_items(None, 0, v)), 4) for k, v in W.test_sets(seed=0).items()}
        json.dump(out, open(os.path.join(OUT, f"s{a.seed}.json"), "w"), indent=1); print(json.dumps(out), flush=True)
        raise SystemExit

    GM.evaluate = run; OO.DATA = "/home/claude/models/day_night/year4_cycle"; OO.OUT = "/tmp/hippo_scratch"; os.makedirs(OO.OUT, exist_ok=True)
    OO.run(argparse.Namespace(cond="books", seed=a.seed))


if __name__ == "__main__":
    main()
