#!/usr/bin/env python3
"""Stage 76: the hippocampus with sleep.

Stage 75's hippocampus learned from 100,000 stories in one go. Here its experience comes in days, and the
stories it lived through are kept as episodes (the hippocampal record: the story and what was asked). Groups
(four networks each, the same reading networks and the same 100,000 stories, split over 4 days):
  awake    days only
  sleep    after each day a night: the hippocampus replays stored episodes with no new input - half from that
           day, half from earlier days - as many as the day had, at a third of the step size (as Stage 70)
  reread   the same extra replays, but while awake: shuffled into the day at the daytime step (equal lessons;
           the control of Stage 71)
Tests: Stage 74 / 75's four test sets.

    python scripts/hippo_sleep.py --cond sleep --seed 0
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar_memory as GM, output_order as OO, world as W, hippocampus as HP   # noqa: E402
OUT = os.path.join(ROOT, "results", "hippo_sleep")
DATA = "/home/claude/models/hippo_sleep"
DAYS, DAY_LR, NIGHT_LR = 4, 1e-3, 1e-3 / 3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cond", choices=("awake", "sleep", "reread"), default="sleep"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)

    def run(M, C, tok, seed, res):
        torch = M.torch; M.m.eval()
        for p in M.m.parameters(): p.requires_grad_(False)
        net = HP.build(M, "hippocampus", a.seed); opt = torch.optim.Adam(net.parameters(), lr=DAY_LR)
        emb_cache = {}

        def emb(words):
            for w in words:
                if w not in emb_cache:
                    with torch.no_grad(): emb_cache[w] = M.m.e(torch.tensor([[M.enc([w])[1]]]))[0, 0]
            return torch.stack([emb_cache[w] for w in words])

        def batch(items):
            seqs, qpos, spos, cands = [], [], [], []
            for it in items:
                q, c, _ = HP.query_of(it); seqs.append(it["story"] + q); spos.append(len(it["story"])); qpos.append(len(it["story"]) + len(q)); cands.append(c)
            x, objs, om = M.batch([dict(words=s) for s in seqs])
            with torch.no_grad(): o, _ = M.m(x, objs, om)
            T = max(spos); hs = o[:, 1:T + 1]; mask = torch.zeros(len(items), T)
            for i, n in enumerate(spos): mask[i, :n] = 1
            return hs * mask[..., None], mask, o[torch.arange(len(items)), torch.tensor(qpos)], torch.stack([emb(c) for c in cands])

        def step(items, lr, rng):
            for g in opt.param_groups: g["lr"] = lr
            hs, mask, q, ce = batch(items); perm = torch.tensor(np.array([rng.permutation(4) for _ in items]))
            ce = torch.gather(ce, 1, perm[..., None].expand(-1, -1, ce.shape[-1])); target = (perm == 0).float().argmax(1)
            loss = torch.nn.functional.cross_entropy(torch.einsum("be,bce->bc", net(hs, mask, q), ce), target)
            opt.zero_grad(); loss.backward(); opt.step()

        def predict(items):
            net.eval(); hits = []
            with torch.no_grad():
                for k in range(0, len(items), 64):
                    hs, mask, q, ce = batch(items[k:k + 64]); hits.append((torch.einsum("be,bce->bc", net(hs, mask, q), ce).argmax(1) == 0).numpy())
            net.train(); return np.concatenate(hits)

        rng = np.random.default_rng(500 + a.seed); per_day = HP.STORIES // DAYS; BS = HP.BS
        val = HP.make_items(np.random.default_rng(99), 400); episodes = []; curve = []; t0 = time.time()
        for d in range(DAYS):
            today = HP.make_items(rng, per_day)                                  # the same stories in every group (same seed)
            prior = episodes[:]
            replay = [today[i] for i in rng.choice(len(today), per_day // 2, replace=False)] + \
                     ([prior[i] for i in rng.choice(len(prior), per_day - per_day // 2, replace=False)] if prior else [today[i] for i in rng.choice(len(today), per_day - per_day // 2, replace=False)])
            if a.cond == "reread": lessons = [(it, DAY_LR) for it in today + replay]; rng.shuffle(lessons)
            else: lessons = [(it, DAY_LR) for it in today]
            for k in range(0, len(lessons), BS): step([x for x, _ in lessons[k:k + BS]], lessons[k][1], rng)
            curve.append(dict(when=f"day {d + 1}", val=round(float(predict(val).mean()), 4), minutes=round((time.time() - t0) / 60, 1))); print(curve[-1], flush=True)
            if a.cond == "sleep":
                order = rng.permutation(len(replay))
                for k in range(0, len(order), BS): step([replay[i] for i in order[k:k + BS]], NIGHT_LR, rng)
                curve.append(dict(when=f"night {d + 1}", val=round(float(predict(val).mean()), 4), minutes=round((time.time() - t0) / 60, 1))); print(curve[-1], flush=True)
            episodes += today
        torch.save(net.state_dict(), os.path.join(DATA, f"{a.cond}_s{a.seed}.pt"))
        out = dict(cond=a.cond, seed=a.seed, curve=curve)
        for name, items in W.test_sets(seed=0).items():
            hit = predict(items); r = dict(all=round(float(hit.mean()), 4))
            for k in ("where_person", "who_has"): m = np.array([it["kind"] == k for it in items]); r[k] = round(float(hit[m].mean()), 4)
            car = np.array([it["carried"] for it in items]); r["carried_objects"] = round(float(hit[car].mean()), 4); out[name] = r
        json.dump(out, open(os.path.join(OUT, f"{a.cond}_s{a.seed}.json"), "w"), indent=1); print(json.dumps(out), flush=True)
        raise SystemExit

    GM.evaluate = run; OO.DATA = "/home/claude/models/day_night/year4_cycle"; OO.OUT = "/tmp/hippo_scratch"; os.makedirs(OO.OUT, exist_ok=True)
    OO.run(argparse.Namespace(cond="books", seed=a.seed))


if __name__ == "__main__":
    main()
