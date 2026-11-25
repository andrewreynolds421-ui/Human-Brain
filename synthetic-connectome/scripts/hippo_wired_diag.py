"""Stage 81 diagnosis: where does recall fail in the trained wired model? For each test question:
cue -> CA3 overlap with the stored pattern, recalled CA1 vs the CA1 pattern of the right answer, decoded output vs answer."""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar_memory as GM, output_order as OO, world as W, hippocampus as HP, hippo_wired_world as WW   # noqa
def run(M, C, tok, seed, res):
    torch = M.torch; torch.set_num_threads(1); M.m.eval()
    os.environ["PRETRAIN_LOOP"] = "0"; net = WW.build(M, 0, os.environ.get("DIAG_MODE", "wired"))
    net.load_state_dict(torch.load(os.environ.get("DIAG_MODEL", "/home/claude/models/hippo_wired/wired_s0.pt"))); net.eval(); core = net.core
    items = W.test_sets(seed=0)["same"]; out = {}
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=-1)
    for kind in ("where_person", "carried"):
        sel = [it for it in items if (it["kind"] == kind if kind != "carried" else it["carried"])][:96]
        seqs, spos, cands, ql = [], [], [], []
        for it in sel:
            q, c, _ = HP.query_of(it); seqs.append(it["story"] + q); spos.append(len(it["story"])); ql.append(len(q)); cands.append(c)
        x, objs, om = M.batch([dict(words=s) for s in seqs])
        with torch.no_grad():
            o, _ = M.m(x, objs, om); e = M.m.e(x); ce = M.m.e(torch.tensor([[M.enc([w])[1] for w in c] for c in cands]))
            T = max(spos); L = max(ql); B = len(sel); mask = torch.zeros(B, T); qmask = torch.zeros(B, L); qe = torch.zeros(B, L, e.shape[-1])
            for i in range(B): mask[i, :spos[i]] = 1; qmask[i, :ql[i]] = 1; qe[i, :ql[i]] = e[i, spos[i] + 1:spos[i] + 1 + ql[i]]
            q = o[torch.arange(B), torch.tensor([spos[i] + ql[i] for i in range(B)])]
            hs, es = o[:, 1:T + 1] * mask[..., None], e[:, 1:T + 1] * mask[..., None]
            WIN = WW.WIN; pad = torch.cat([torch.zeros(B, WIN - 1, es.shape[-1]), es], 1); win = torch.stack([pad[:, j:j + T] for j in range(WIN)], 2)
            v = (torch.softmax(net.pick_v(hs), -1)[..., None] * win).sum(2); kw = (torch.softmax(net.pick_k(hs), -1)[..., None] * win).sum(2)
            g = torch.sigmoid(net.gate(hs)) * mask[..., None]; mem = core.new_memory(B)
            for t in range(T): core.store(mem, kw[:, t], v[:, t], g[:, t])
            sc = (qe * net.qproj(q)[:, None, :]).sum(-1) + torch.log(qmask + 1e-9); qw = (torch.softmax(sc, -1)[..., None] * qe).sum(1)
            c1 = core.ca3_cue_mem(qw, mem); a1 = core.apply(mem, "eS", c1); r1 = core.readout(a1)
            right = ce[:, 0]; wrong = ce[:, 1:]
            ca1_right = core.ca1_target(right); ca1_wrong = torch.stack([core.ca1_target(wrong[:, j]) for j in range(3)], 1)
            # does recalled CA1 point at the right answer (in CA1 space)? and does the decoded output?
            ca1_hit = (cos(a1, ca1_right) > cos(a1[:, None], ca1_wrong).max(1).values).float().mean()
            dec_hit = (cos(r1, right) > cos(r1[:, None], wrong).max(1).values).float().mean()
            # decode the recalled CA1 pattern with an ideal read-out: nearest candidate in CA1 space vs. learned loop
            out[kind] = dict(one_hop_ca1_points_right=round(float(ca1_hit), 3), one_hop_decoded_right=round(float(dec_hit), 3),
                             ca1_norm_ratio=round(float(a1.norm(dim=-1).mean() / ca1_right.norm(dim=-1).mean()), 3),
                             cos_recall_vs_right_ca1=round(float(cos(a1, ca1_right).mean()), 3))
    print(json.dumps(out), flush=True); raise SystemExit
GM.evaluate = run; OO.DATA = "/home/claude/models/day_night/year4_cycle"; OO.OUT = "/tmp/hippo_scratch"; os.makedirs(OO.OUT, exist_ok=True)
OO.run(argparse.Namespace(cond="books", seed=0))
