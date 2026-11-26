#!/usr/bin/env python3
"""Stage 81b: the data-wired hippocampus on all 20 bAbI tasks - Stage 79's network (hippo_babi.py: recall while
storing, three recalls from the question's main word, one from a second word, CA1 decision) with its memory
replaced by hippo_wired.WiredCore. Training, data and tests are hippo_babi.py's own (it is imported and run).

    QUESTIONS=40000 BABI_OUT=results/hippo_wired_babi BABI_DATA=/home/claude/models/hippo_wired_babi \
        python scripts/hippo_wired_babi.py --seed 0
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hippo_babi as HB, hippo_wired as HW   # noqa: E402


def build(M, seed):
    torch = M.torch; nn = torch.nn; torch.manual_seed(5000 + seed); H = 320; E = M.m.e.base.embedding_dim; WIN = HB.WIN
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))

    class WiredBabi(nn.Module):
        def __init__(s):
            super().__init__()
            s.pick_k = nn.Linear(H, WIN); s.pick_r = nn.Linear(H, WIN); s.pick_v = nn.Linear(H, WIN + 1); s.gate = nn.Linear(H, 1)
            s.q1 = nn.Linear(H, E); s.q2 = nn.Linear(H, E)
            s.mix = nn.Linear(H, 6); s.dec = nn.Sequential(nn.Linear(H + 6 * E + 4, 256), nn.Tanh(), nn.Linear(256, E)); s.temp = nn.Parameter(torch.tensor(5.0))
            s.core = HW.build_core(torch, E, seed, os.environ.get("WIRED_MODE", "wired"))

        def forward(s, hs, es, mask, q, qe, qmask):
            B, T, _ = hs.shape; Ed = es.shape[-1]
            pad = torch.cat([torch.zeros(B, WIN - 1, Ed), es], 1); win = torch.stack([pad[:, j:j + T] for j in range(WIN)], 2)
            ak = torch.softmax(s.pick_k(hs), -1); ar = torch.softmax(s.pick_r(hs), -1); av = torch.softmax(s.pick_v(hs), -1)
            kw = (ak[..., None] * win).sum(2); rw = (ar[..., None] * win).sum(2); g = torch.sigmoid(s.gate(hs)) * mask[..., None]
            mem = s.core.new_memory(B)
            for t in range(T):
                rec, _ = s.core.recall(mem, rw[:, t])                                                  # recall while storing
                opts = torch.cat([win[:, t], rec[:, None]], 1); v = (av[:, t, :, None] * opts).sum(1)
                s.core.store(mem, kw[:, t], v, g[:, t])
            read = lambda e: s.core.recall(mem, e)[0]
            lm = torch.log(qmask + 1e-9)
            c1 = (torch.softmax((qe * s.q1(q)[:, None]).sum(-1) / Ed ** 0.5 + lm, -1)[..., None] * qe).sum(1)
            c2 = (torch.softmax((qe * s.q2(q)[:, None]).sum(-1) / Ed ** 0.5 + lm, -1)[..., None] * qe).sum(1)
            if os.environ.get("BIG_LOOP") == "1":                 # later recalls cued from the entorhinal output (EC V -> EC II), not the decoded word
                r1, _, e1 = s.core.recall_full(mem, c1); r2, _, e2 = s.core.recall_full(mem, ec2=s.core.ec5_ec2(e1))
                r3, _, _ = s.core.recall_full(mem, ec2=s.core.ec5_ec2(e2))
            else:
                r1 = read(c1); r2 = read(r1); r3 = read(r2)
            s1 = read(c2)
            cos = lambda a, b: (a * b).sum(-1, keepdim=True) / (a.norm(dim=-1, keepdim=True) * b.norm(dim=-1, keepdim=True) + 1e-6)
            feats = torch.cat([cos(r1, c2), cos(r2, c2), cos(r3, c2), r1.norm(dim=-1, keepdim=True)], -1)
            ml = s.mix(q); w = torch.softmax(ml, -1); d = s.dec(torch.cat([q, r1, r2, r3, s1, c1, c2, feats], -1))
            opts = [r1, r2, r3, s1, c2, d]; s.last = (opts, ml)
            return sum(w[:, i:i + 1] * o for i, o in enumerate(opts))
    net = WiredBabi()
    if os.environ.get("PRETRAIN_LOOP", "1") == "1": print("EC-CA1 loop", HW.pretrain_loop(torch, net.core, M.m.e.base.weight, seed=seed), flush=True)
    return net


def patched_main():
    """hippo_babi.main with SCORE_BOTH=1 (each of the six answers scored on its own, the chooser trained to pick the one
    that answered best) and gradient clipping always on"""
    import inspect
    src = inspect.getsource(HB.main)
    # hippo_babi.main's training step now spans several lines (Stage 80+ options); patch the main loop's backward step
    old = ("opt.zero_grad(); loss.backward()\n"
           "                if a.episodes or a.steady: torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)")
    assert src.count(old) == 1
    new = ("if os.environ.get('SCORE_BOTH') == '1' and hasattr(net, 'last'):\n"
           "                    opts, ml = net.last; ce = b[6] / (b[6].norm(dim=-1, keepdim=True) + 1e-6)\n"
           "                    per = torch.stack([torch.nn.functional.cross_entropy(net.temp * torch.einsum('be,bce->bc', o / (o.norm(dim=-1, keepdim=True) + 1e-6), ce), y, reduction='none') for o in opts], 1)\n"
           "                    loss = loss + per.min(1).values.mean() + torch.nn.functional.cross_entropy(ml, per.detach().argmin(1))\n"
           "                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)")
    src = src.replace(old, new)
    exec(compile(src, HB.__file__, "exec"), HB.__dict__); HB.main()


if __name__ == "__main__":
    HB.MAX_SENT = int(os.environ.get("MAX_SENT", HB.MAX_SENT))      # shorter stories: this machine has 2 CPUs
    if os.environ.get("WIRED_MODE", "wired") != "old": HB.build = build   # old = Stage 79's own network, same budget
    patched_main()
