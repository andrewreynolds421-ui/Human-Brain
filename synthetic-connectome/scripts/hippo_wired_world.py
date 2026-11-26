#!/usr/bin/env python3
"""Stage 81a: the data-wired hippocampus (hippo_wired.py) on VEGA's world stories - same task, reader, training
and tests as Stage 77 (hippo_pointer.py), plus a crowded test.

Stage 77's picking of what to store (attention over the last six words), the gate and the question's key word are
kept; only the memory changes: Stage 77's dentate-addressed table -> CA3 / CA1 populations with mossy fibres,
recurrent collaterals and Schaffer synapses (hippo_wired.WiredCore).

Conditions: wired (full) | no_rc (no recurrent completion) | dense_mf (mossy fibres all-to-all and weak)
            | old (evaluate the Stage 77 pointer_s*.pt models here, no training)
SCORE_BOTH=1: each recall is also scored on its own (the better one takes the credit) and the chooser (mix) is
trained to pick the recall that answered better.
BIG_LOOP=1: the second recall is cued from entorhinal layer V output (via EC V -> EC II), not the decoded word.
CURRICULUM=1: one-step questions first, then short stories with carried objects, then the normal mix.
Tests: Stage 74/77's four sets, plus *crowded*: 7 people and 7 objects in 4 places, 20-27 events - many similar
memories, where pattern separation should matter.

    python scripts/hippo_wired_world.py --cond wired --seed 0
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar_memory as GM, output_order as OO, world as W, hippocampus as HP, hippo_pointer as PT, hippo_wired as HW   # noqa: E402
OUT = os.environ.get("WIRED_OUT", os.path.join(ROOT, "results", "hippo_wired"))
DATA = os.environ.get("WIRED_DATA", "/home/claude/models/hippo_wired")
WIN = PT.WIN


def crowded_set(seed=0, n=1000):
    rng = np.random.default_rng(20_000 + seed); items = []
    for want, k in (("carried", n // 2), ("who_has", n // 4), ("where_person", n - n // 2 - n // 4)):
        got = 0
        while got < k:
            it = W.question(rng, *W.story(rng, W.NAMES, W.OBJECTS, W.PLACES, int(rng.integers(20, 28)), n_people=7, n_obj=7), W.PLACES, want)
            if it: items.append(it); got += 1
    return items


def curriculum_items(rng, n, frac):
    """CURRICULUM=1: stage 1 (first 20 % of training) one-step questions only (where a person is, who has what) in
    short stories (4-6 events); stage 2 (20-50 %) all kinds incl. carried objects (two-step) in short stories;
    stage 3 the normal mix (HP.make_items, 4-10 events)"""
    if frac >= 0.5: return HP.make_items(rng, n)
    kinds = ("where_person", "who_has") if frac < 0.2 else ("carried", "who_has", "where_person", "carried")
    out = []
    while len(out) < n:
        it = W.question(rng, *W.story(rng, W.NAMES, W.OBJECTS, W.PLACES, int(rng.integers(4, 7))), W.PLACES, kinds[len(out) % len(kinds)])
        if it: out.append(it)
    return out


def build(M, seed, mode):
    torch = M.torch; nn = torch.nn; torch.manual_seed(4000 + seed); H = 320; E = M.m.e.base.embedding_dim

    class WiredPointer(nn.Module):
        def __init__(s):
            super().__init__()
            s.pick_v = nn.Linear(H, WIN); s.pick_k = nn.Linear(H, WIN); s.gate = nn.Linear(H, 1)
            s.mix = nn.Linear(H, 2); s.temp = nn.Parameter(torch.tensor(5.0)); s.qproj = nn.Linear(H, E)
            s.core = HW.build_core(torch, E, seed, mode); s.diag = {}

        def forward(s, hs, es, mask, q, qe, qmask):
            B, T, _ = hs.shape
            pad = torch.cat([torch.zeros(B, WIN - 1, es.shape[-1]), es], 1)
            win = torch.stack([pad[:, j:j + T] for j in range(WIN)], 2)
            v = (torch.softmax(s.pick_v(hs), -1)[..., None] * win).sum(2)
            kw = (torch.softmax(s.pick_k(hs), -1)[..., None] * win).sum(2)
            g = torch.sigmoid(s.gate(hs)) * mask[..., None]
            mem = s.core.new_memory(B)
            for t in range(T): s.core.store(mem, kw[:, t], v[:, t], g[:, t])
            sc = (qe * s.qproj(q)[:, None, :]).sum(-1) + torch.log(qmask + 1e-9)
            qw = (torch.softmax(sc, -1)[..., None] * qe).sum(1)
            if os.environ.get("BIG_LOOP") == "1":                 # second recall cued from the entorhinal output itself (CA1 -> SUB -> EC V -> EC II)
                r1, c1, ec5 = s.core.recall_full(mem, qw); r2, _, _ = s.core.recall_full(mem, ec2=s.core.ec5_ec2(ec5))
            else:
                r1, c1 = s.core.recall(mem, qw); r2, _ = s.core.recall(mem, r1)
            ml = s.mix(q); w = torch.softmax(ml, -1); s.last = (r1, r2, ml)
            with torch.no_grad():
                cs = s.core.ca3_store(qw); s.diag = dict(gate=round(float(g.sum() / mask.sum()), 3), mix1=round(float(w[:, 0].mean()), 3),
                    cue_store=round(float((c1 * cs).sum(-1).mean()), 3), mf_gain=round(float(s.core.mf_gain), 2))
            return w[:, :1] * r1 + w[:, 1:] * r2
    net = WiredPointer()
    if os.environ.get("PRETRAIN_LOOP", "1") == "1": print("EC-CA1 loop", HW.pretrain_loop(torch, net.core, M.m.e.base.weight, seed=seed), flush=True)
    return net


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", choices=("wired", "no_rc", "dense_mf", "shared", "old"), default="wired"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    stories = int(os.environ.get("STORIES", HP.STORIES))

    def run(M, C, tok, seed, res):
        torch = M.torch; torch.set_num_threads(int(os.environ.get("THREADS", 1))); M.m.eval()
        for p in M.m.parameters(): p.requires_grad_(False)
        if a.cond == "old":
            net = PT.build(M, a.seed); net.load_state_dict(torch.load(os.path.join(PT.DATA, f"pointer_s{a.seed}.pt")))
        else:
            net = build(M, a.seed, a.cond)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

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

        log = []; t0 = time.time(); tr_acc = []

        def train_step(items, lr, rng):
            for g in opt.param_groups: g["lr"] = lr
            b = list(batch(items)); perm = torch.tensor(np.array([rng.permutation(4) for _ in items]))
            b[6] = torch.gather(b[6], 1, perm[..., None].expand(-1, -1, b[6].shape[-1])); target = (perm == 0).float().argmax(1)
            lg = logits(b); loss = torch.nn.functional.cross_entropy(lg, target)
            if os.environ.get("SCORE_BOTH") == "1":
                r1, r2, ml = net.last; ce = b[6] / (b[6].norm(dim=-1, keepdim=True) + 1e-6)
                per = torch.stack([torch.nn.functional.cross_entropy(net.temp * torch.einsum("be,bce->bc", r / (r.norm(dim=-1, keepdim=True) + 1e-6), ce), target, reduction="none") for r in (r1, r2)], 1)
                loss = loss + per.min(1).values.mean() + torch.nn.functional.cross_entropy(ml, per.detach().argmin(1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()

        sleep_mode = os.environ.get("SLEEP_MODE")
        if sleep_mode and a.cond != "old":
            # SLEEP_MODE=awake|sleep|reread, INIT_MODEL=<trained .pt>: continue training in DAYS days of new stories
            # (stories / DAYS each). sleep: after each day a night replays as many stored episodes (half that day's,
            # half earlier days') at a third of the step size, with no new input (Stage 76). reread: the same extra
            # lessons shuffled into the day at the day's step size (equal-lessons control). awake: days only.
            days = int(os.environ.get("DAYS", 4)); per_day = stories // days; DAY_LR, NIGHT_LR = 1e-3, 1e-3 / 3
            path = os.path.join(DATA, f"{sleep_mode}_s{a.seed}.pt"); ck = path + ".ckpt"
            net.load_state_dict(torch.load(os.environ["INIT_MODEL"])); rng = np.random.default_rng(900 + a.seed)
            val = HP.make_items(np.random.default_rng(99), 400); episodes = []; d0 = 0
            log.append(dict(when="start", val=round(float(predict(val).mean()), 4)))
            if os.path.exists(ck):
                z = torch.load(ck, weights_only=False); net.load_state_dict(z["net"]); opt.load_state_dict(z["opt"]); d0 = z["day"]
                log = z["log"]; episodes = z["episodes"]; rng.bit_generator.state = z["rng"]; print("resumed after day", d0, flush=True)
            for d in range(d0, days):
                today = HP.make_items(rng, per_day)
                replay = []
                if sleep_mode != "awake":
                    old_eps = episodes if episodes else today
                    replay = [today[i] for i in rng.choice(len(today), per_day // 2, replace=False)] + [old_eps[i] for i in rng.choice(len(old_eps), per_day - per_day // 2, replace=len(old_eps) < per_day)]
                lessons = today + replay if sleep_mode == "reread" else list(today)
                order = rng.permutation(len(lessons))
                for k in range(0, len(order) - HP.BS + 1, HP.BS): train_step([lessons[i] for i in order[k:k + HP.BS]], DAY_LR, rng)
                log.append(dict(when=f"day {d + 1}", val=round(float(predict(val).mean()), 4), minutes=round((time.time() - t0) / 60, 1))); print(log[-1], flush=True)
                if sleep_mode == "sleep":
                    order = rng.permutation(len(replay))
                    for k in range(0, len(order) - HP.BS + 1, HP.BS): train_step([replay[i] for i in order[k:k + HP.BS]], NIGHT_LR, rng)
                    log.append(dict(when=f"night {d + 1}", val=round(float(predict(val).mean()), 4), minutes=round((time.time() - t0) / 60, 1))); print(log[-1], flush=True)
                episodes += today
                torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), day=d + 1, log=log, episodes=episodes, rng=rng.bit_generator.state), ck + ".tmp"); os.replace(ck + ".tmp", ck)
            torch.save(net.state_dict(), path)
        elif a.cond != "old":
            path = os.path.join(DATA, f"{a.cond}_s{a.seed}.pt"); ck = path + ".ckpt"; start = 0; last = time.time()
            rng = np.random.default_rng(500 + a.seed); val = HP.make_items(np.random.default_rng(99), 400)
            if os.path.exists(path): net.load_state_dict(torch.load(path)); start = stories // HP.BS
            elif os.path.exists(ck):
                z = torch.load(ck, weights_only=False); net.load_state_dict(z["net"]); opt.load_state_dict(z["opt"]); start = z["step"]; log = z["log"]
                rng.bit_generator.state = z["rng"]; print("resumed at", start * HP.BS, "stories", flush=True)
            for step in range(start, stories // HP.BS):
                if time.time() - last > 600:
                    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, log=log, rng=rng.bit_generator.state), ck + ".tmp"); os.replace(ck + ".tmp", ck); last = time.time()
                frac = step / max(stories // HP.BS, 1)
                items = curriculum_items(rng, HP.BS, frac) if os.environ.get("CURRICULUM") == "1" else HP.make_items(rng, HP.BS); b = list(batch(items))
                perm = torch.tensor(np.array([rng.permutation(4) for _ in items]))
                b[6] = torch.gather(b[6], 1, perm[..., None].expand(-1, -1, b[6].shape[-1])); target = (perm == 0).float().argmax(1)
                lg = logits(b); tr_acc.append(float((lg.argmax(1) == target).float().mean())); loss = torch.nn.functional.cross_entropy(lg, target)
                if os.environ.get("SCORE_BOTH") == "1" and a.cond != "old":       # score each recall on its own; the chooser learns which one answered
                    r1, r2, ml = net.last; ce = b[6] / (b[6].norm(dim=-1, keepdim=True) + 1e-6)
                    per = torch.stack([torch.nn.functional.cross_entropy(net.temp * torch.einsum("be,bce->bc", r / (r.norm(dim=-1, keepdim=True) + 1e-6), ce), target, reduction="none") for r in (r1, r2)], 1)
                    loss = loss + per.min(1).values.mean() + torch.nn.functional.cross_entropy(ml, per.detach().argmin(1))
                opt.zero_grad(); loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                if os.environ.get("DEBUG") and step < 5: print("step", step, "loss", float(loss), "gradnorm", float(gn), {n: (None if p.grad is None else round(float(p.grad.abs().sum()), 4)) for n, p in net.named_parameters()}, flush=True)
                if (step + 1) % int(os.environ.get('LOG_EVERY', 250)) == 0 or step + 1 == stories // HP.BS:
                    log.append(dict(stories=(step + 1) * HP.BS, val=round(float(predict(val).mean()), 4), minutes=round((time.time() - t0) / 60, 1),
                                    sparsity=dict(net.core.sparsity), train=round(float(np.mean(tr_acc[-50:])), 3), diag=dict(net.diag))); print(log[-1], flush=True)
            torch.save(net.state_dict(), path)
            if os.path.exists(ck): os.remove(ck)
        out = dict(cond=a.cond, sleep_mode=sleep_mode, init_model=os.environ.get("INIT_MODEL"), seed=a.seed, stories=stories, training=log, provenance=HW.PROVENANCE, sizes=HW.SIZE)
        sets = W.test_sets(seed=0); sets["crowded"] = crowded_set(0)
        for name, items in sets.items():
            hit = predict(items); r = dict(all=round(float(hit.mean()), 4))
            for k in ("where_person", "who_has"): m = np.array([it["kind"] == k for it in items]); r[k] = round(float(hit[m].mean()), 4)
            car = np.array([it["carried"] for it in items]); r["carried_objects"] = round(float(hit[car].mean()), 4)
            cue = np.array([W.last_mention(it) for it in items]); r["when_cue_wrong"] = round(float(hit[~cue].mean()), 4); out[name] = r
        json.dump(out, open(os.path.join(OUT, f"{sleep_mode or a.cond}_s{a.seed}.json"), "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k not in ("training", "provenance")}), flush=True)
        raise SystemExit

    GM.evaluate = run; OO.DATA = "/home/claude/models/day_night/year4_cycle"; OO.OUT = "/tmp/hippo_scratch"; os.makedirs(OO.OUT, exist_ok=True)
    OO.run(argparse.Namespace(cond="books", seed=a.seed))


if __name__ == "__main__":
    main()
