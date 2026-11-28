#!/usr/bin/env python3
"""Stage 89: a hippocampus that stores whole sentences as events, and is taught each recall step.

What Stages 79-88 and the review (AUDIT 64) found: VEGA's hippocampus recalled single facts but did not chain them
(the two-step strategy pays only when both steps work, so training settled for guessing), and on real books its
memory hardly beat counting words (it had been built on toy sentences with an 8-word window). Two changes:

1. EVENTS ARE SENTENCES. Each sentence of the story is one stored event (hippocampal event segmentation at
   boundaries: Zacks et al. 2007; Ben-Yakov & Henson 2018). An event holds its words (each filed under its own
   dentate-gyrus code, as before, with a learned key strength from the word's role) and the reading network's
   state at the sentence end (the gist). Recall is pattern completion over events: any word of an event - or a
   gist cue - brings the event back, by attention over all stored events (one step of a modern Hopfield network,
   the standard model of CA3 autoassociation: Ramsauer et al. 2020), with learned recency and "before / after
   the event just recalled" (time cells). What comes back is the rest of the event: its other words, weighted by
   learned importance. Three recall steps; a prefrontal working memory (GRU, MODEL) holds the question and what
   was recalled, and forms the next cue through the learned re-entry loop of Stage 84.
   Recognition as well as recall: each candidate answer is also offered to the memory as a cue, and how strongly
   the recalled events respond to it is added to its score (familiarity / recognition memory; starts at zero).
2. EACH STEP IS TAUGHT (--sup 1). bAbI marks which sentences support each answer, in the order of the reasoning.
   Recall step k is rewarded for landing on supporting sentence k (more supports than steps: on any of them).
   This is the "strong supervision" of Weston et al. 2015; --sup 0 is the control (answer only).
Real book passages (Stage 86's passage test) are in the training mix from the start, not a last "day".

Tested like for like (AUDIT 64): ONE answer list for all 20 bAbI tasks; the passage test is reported next to its
"most frequent candidate" baseline. Compare with: Stage 84 one-list (8, 5, 7, 8 passed, 24.7 %), LSTM 4 / 36.4 %,
MemN2N (answer only) 14 / 7.5 %, strongly supervised MemNN (Weston et al. 2015; supporting facts, as here).

    QUESTIONS=160000 PASSAGES=data/passages.json.gz python scripts/hippo_events.py --sup 1 --seed 0
"""
import argparse, gzip, json, os, re, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar_memory as GM, output_order as OO, world as W, hippocampus as HP, hippo_babi as HB   # noqa: E402
OUT = os.environ.get("EV_OUT", os.path.join(ROOT, "results", "events"))
DATA = os.environ.get("EV_DATA", "/home/claude/models/events")
HOPS, N_DG, ACTIVE, D_KEY, BS, MAX_SENT = 3, 512, 0.05, 64, 16, 50
QUESTIONS = int(os.environ.get("QUESTIONS", 160_000))
SUP_W = 1.0                                                              # weight of the step teaching (ASSUMED)
REWARD = False       # Stage 96: dopamine - each recall is a CHOICE of one event (sampled); a critic predicts success; choices that did
                     # better than predicted are strengthened, worse weakened (reward prediction error; Schultz et al. 1997)
HINDSIGHT = False    # Stage 96: after the answer is given, the sentences that mention it are where recall should have gone (no labels needed)
ENTITIES = False     # Stage 93: a running situation model - memory cells updated sentence by sentence while reading (Henaff et al. 2017; Zwaan & Radvansky 1998)
N_CELLS, D_ENT = 20, 128
SPACE_SUP = False    # Stage 92b: the map is taught - for path questions the answer's moves give the displacement it should read (training answers only)
MOVES = {"n": (0.0, 1.0), "s": (0.0, -1.0), "e": (1.0, 0.0), "w": (-1.0, 0.0)}
SPACE = False        # Stage 92: a spatial map - things are placed in 2-D so that the stated relations agree (path integration)
ROLES = False        # Stage 90b: role-aware recall - read one word of the recalled event by its role in the sentence
SEM = False          # Stage 90a: word meaning from the semantic cortex (scripts/semantic_cortex.py) joined to every word
D_SEM = 128
P_WORLD, P_PASS = 0.15, 0.25                                             # share of training batches (ASSUMED)


# Stage 99 --aug: words that can be swapped for one another without changing what a story means
AUG_PERSONS = "antoine bernhard bill brian daniel emily fred gertrude greg jason jeff jessica john julie julius lily mary sandra sumit winona yann".split()
AUG_PLACES = "bathroom bedroom garden hallway kitchen office cinema park school".split()
AUG_OBJECTS = "apple football milk".split()
AUG_COLOURS = "gray green white yellow".split()
AUG_SKIP = {20}
PRIO_EVERY = int(os.environ.get("PRIO_EVERY", 4000))                                  # steps between re-weighting tasks by error (--prio)                                                                       # motives: hungry -> kitchen is world knowledge, not a name


def load_task(path):
    items, sents, ids = [], [], []
    for line in open(path):
        n, rest = line.split(" ", 1); n = int(n)
        if n == 1: sents, ids = [], []
        if "\t" in rest:
            q, ans, sup = rest.rstrip("\n").split("\t")
            kept, kid = sents[-MAX_SENT:], ids[-MAX_SENT:]; pos = {i: j for j, i in enumerate(kid)}; want = [int(x) for x in sup.split()]
            items.append(dict(sents=kept, q=HB.words(q), ans=tuple(HB.words(ans.replace(",", " "))),
                              supp=[pos[x] for x in want] if all(x in pos for x in want) else None))     # None: a supporting sentence was cut off
        else:
            sents.append(HB.words(rest)); ids.append(n)
    return items


def load_all():
    tasks = {}
    for f in sorted(os.listdir(os.path.join(HB.BABI, "en-10k"))):
        m = re.match(r"qa(\d+)_(.+)_train.txt", f)
        if not m: continue
        tasks[int(m.group(1))] = dict(name=f"qa{m.group(1)}_{m.group(2)}", train=load_task(os.path.join(HB.BABI, "en-10k", f)),
                                      test=load_task(os.path.join(HB.BABI, "en", f.replace("_train", "_test"))))
    allc = sorted({it["ans"] for t in tasks for it in tasks[t]["train"]})              # ONE answer list for all tasks (AUDIT 64)
    for t in tasks:
        for it in tasks[t]["train"] + tasks[t]["test"]: it["cands"] = allc; it["y"] = allc.index(it["ans"]) if it["ans"] in allc else -1
    return tasks, allc


def split(words, ends):
    out, a = [], 0
    for e in ends: out.append(list(map(str, words[a:e + 1]))); a = e + 1
    return out


def world_item(it, rng=None):
    q, c, _ = HP.query_of(it); cands = [(str(w),) for w in c]; y = 0
    if rng is not None:
        o = rng.permutation(len(cands)); cands = [cands[k] for k in o]; y = int(np.nonzero(o == 0)[0][0])
    return dict(sents=split(it["story"], it["ends"]), q=list(map(str, q)), cands=cands, y=y, supp=None)


def world_batch(rng, n):
    out = []
    while len(out) < n:
        it = W.question(rng, *W.story(rng, W.NAMES, W.OBJECTS, W.PLACES, int(rng.integers(4, 19))), W.PLACES, ("carried", "who_has", "where_person", "carried")[len(out) % 4])
        if it: out.append(world_item(it, rng))
    return out


def passage_item(it):
    return dict(sents=split(it["story"], it["ends"]), q=it["q"], cands=[tuple(c) for c in it["cands"]], y=it["y"], supp=None, sent=it["sent"], pos=it["pos"])


def build(M, seed):
    torch = M.torch; nn = torch.nn; torch.manual_seed(8900 + seed); H = getattr(M, "H", 320); E = M.m.e.base.embedding_dim + (D_SEM if SEM else 0); P, D = 256, 128; NR = 2 if ROLES else 1

    class EventMemory(nn.Module):
        def __init__(s):
            super().__init__()
            s.wk = nn.Linear(E, D_KEY); s.pp = nn.Linear(D_KEY, N_DG)                              # EC -> DG / CA3 key (as Stage 75)
            s.ksal = nn.Linear(H, 1); s.sal = nn.Linear(E, 1); s.sproj = nn.Linear(H, D)           # key strength (role), importance, event gist
            s.pfc0 = nn.Linear(H, P); s.pfc = nn.GRUCell((1 + NR) * E + D, P)                             # prefrontal working memory
            s.cue_q = nn.Linear(P, E); s.qs = nn.Linear(P, D); s.tw = nn.Linear(P, 3)              # cue word, gist cue, time (recency / before / after)
            s.beta = nn.Parameter(torch.tensor(10.0)); s.temp = nn.Parameter(torch.tensor(5.0))
            s.reent = nn.Sequential(nn.Linear(E + P, 256), nn.Tanh(), nn.Linear(256, E))           # big loop (Stage 84); starts as a copy
            s.mix = nn.Linear(H, NR * HOPS + 1 + (1 if ENTITIES else 0)); s.dec = nn.Sequential(nn.Linear(P + H + NR * HOPS * E + HOPS + 1 + (64 if SPACE else 0) + (D_ENT if ENTITIES else 0), 256), nn.Tanh(), nn.Linear(256, E))
            if ROLES: s.hproj = nn.Linear(H, D); s.rq = nn.Linear(P + E, D)                          # a word's role (from its context) / the role asked for
            if SPACE:                                                                              # entorhinal-like map (MODEL)
                s.sp_h = nn.Linear(H, D); s.sp_u = nn.Parameter(torch.randn(2, D) * 0.1)             # which two things a sentence relates (by role)
                s.sp_d = nn.Linear(H, 2); s.sp_g = nn.Linear(H, 1)                                   # the displacement it states; whether it is spatial at all
                s.sp_q = nn.Linear(H, 2 * E); s.sp_out = nn.Sequential(nn.Linear(10, 64), nn.Tanh())  # the two things asked about; what is read off the map
                with torch.no_grad(): s.sp_g.bias.fill_(-2.0)                                        # sentences are not spatial until learned otherwise
                s.register_buffer("dirs", torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]))
            if ENTITIES:                                                                           # the state of the story, kept up while reading (MODEL)
                s.en_keys = nn.Parameter(torch.randn(N_CELLS, D_ENT) * 0.1); s.en_s = nn.Linear(H + E, D_ENT)   # what each cell is about; a sentence as read
                s.en_U = nn.Linear(D_ENT, D_ENT, bias=False); s.en_V = nn.Linear(D_ENT, D_ENT, bias=False); s.en_W = nn.Linear(D_ENT, D_ENT, bias=False); s.en_a = nn.PReLU()
                s.en_q = nn.Linear(H, D_ENT); s.en_o = nn.Sequential(nn.Linear(2 * D_ENT, D_ENT), nn.PReLU()); s.en_e = nn.Linear(D_ENT, E)
            if REWARD: s.critic = nn.Linear(H, 1)                                                  # expected success, from the question
            s.recw = nn.Linear(H, HOPS + 1)                                                        # recognition gains; start at zero
            with torch.no_grad():
                s.reent[2].weight.zero_(); s.reent[2].bias.zero_(); s.recw.weight.zero_(); s.recw.bias.zero_(); s.tw.weight.zero_(); s.tw.bias.zero_()
            g = torch.Generator().manual_seed(89 + seed); s.register_buffer("dg", torch.randn(D_KEY, N_DG, generator=g) / D_KEY ** 0.5)

        def code(s, e):
            k = s.wk(e); a = s.pp(k); d = torch.relu(k @ s.dg); kth = d.topk(int(ACTIVE * N_DG), dim=-1).values[..., -1:]
            d = d * (d >= kth); a = a + d / (d.norm(dim=-1, keepdim=True) + 1e-6); return a / (a.norm(dim=-1, keepdim=True) + 1e-6)

        def forward(s, hs, es, mask, q, qe, qmask, sid, ends, smask, ce):
            B, T, _ = hs.shape; S = ends.shape[1]; Ed = es.shape[-1]
            keys = s.code(es); kstr = torch.sigmoid(s.ksal(hs))[..., 0] * mask; imp = torch.sigmoid(s.sal(es))[..., 0] * mask
            Mh = (sid[:, None, :] == torch.arange(S)[None, :, None]).float()                         # event x word membership
            send = torch.gather(hs, 1, ends[..., None].expand(-1, -1, hs.shape[-1])) * smask[..., None]; ssum = s.sproj(send)
            tpos = (torch.arange(S, dtype=torch.float32)[None] + 1) / smask.sum(1, keepdim=True).clamp(min=1)
            lsm = torch.log(smask + 1e-9); lm = torch.log(qmask + 1e-9)
            st = torch.tanh(s.pfc0(q)); tau = torch.ones(B); found, rs, scs, atts, counts, rr = [], [], [], [], [], []; pol = torch.zeros(B); ent = torch.zeros(B)
            if ROLES: hrole = s.hproj(hs); lmask = torch.log(mask + 1e-9)
            for k in range(HOPS):
                Sq = torch.cat([qe] + [f[:, None] for f in found], 1); Sm = torch.cat([lm] + [torch.zeros(B, 1)] * len(found), 1)
                cue = (torch.softmax((Sq * s.cue_q(st)[:, None]).sum(-1) / Ed ** 0.5 + Sm, -1)[..., None] * Sq).sum(1)
                wm = torch.einsum("btn,bn->bt", keys, s.code(cue))                                   # how well each stored word matches the cue
                sm = torch.einsum("bst,bt->bs", Mh, wm * kstr)                                       # ... summed per event: any word recalls it
                tw = s.tw(st)
                sc = (s.beta * sm + (ssum * s.qs(st)[:, None]).sum(-1) / D ** 0.5 + 10 * tw[:, :1] * tpos
                      + 10 * tw[:, 1:2] * torch.sigmoid(20 * (tau[:, None] - tpos)) + 10 * tw[:, 2:3] * torch.sigmoid(20 * (tpos - tau[:, None])) + lsm)
                a = torch.softmax(sc, -1)                                                            # pattern completion over events
                if REWARD:                                                                           # commit to ONE event: explore while learning, the strongest when tested
                    jk = torch.multinomial(a, 1)[:, 0] if s.training else a.argmax(-1); lsel = torch.log_softmax(sc, -1)
                    pol = pol + lsel.gather(1, jk[:, None])[:, 0]; ent = ent - (a * lsel.clamp(min=-30)).sum(-1); a = torch.nn.functional.one_hot(jk, S).float()
                ww = torch.einsum("bs,bst->bt", a, Mh) * imp * (1 - wm.clamp(0, 1))                  # the rest of the event (not the cue word itself)
                r = (ww[..., None] * es).sum(1) / (ww.sum(1, keepdim=True) + 1e-3)
                tau = (a * tpos).sum(-1); counts.append((torch.sigmoid(s.beta * sm - 5) * smask).sum(-1, keepdim=True))
                extra = []
                if ROLES:                                                                            # which word of the recalled event: the one in the role asked for
                    wsc = (hrole * s.rq(torch.cat([st, cue], -1))[:, None]).sum(-1) / D ** 0.5
                    w2 = torch.softmax(torch.log(torch.einsum("bs,bst->bt", a, Mh) + 1e-9) + wsc + lmask, -1)
                    r2 = (w2[..., None] * es).sum(1); extra = [r2]; rr.append(r2)
                st = s.pfc(torch.cat([r, cue] + extra + [(a[..., None] * ssum).sum(1)], -1), st)
                found += extra
                found.append(r + s.reent(torch.cat([r, st], -1))); rs.append(r); scs.append(torch.log_softmax(sc, -1)); atts.append(a)
            spf = []
            if SPACE:
                wr = torch.softmax(torch.einsum("btd,rd->brt", s.sp_h(hs), s.sp_u)[:, :, None, :] + torch.log(Mh + 1e-9)[:, None], -1)   # (B, 2, S, T)
                kc = s.code(torch.einsum("brst,bte->brse", wr, es)); d = s.sp_d(send); g = torch.sigmoid(s.sp_g(send)) * smask[..., None]
                Pm = torch.zeros(B, N_DG, 2)                                                         # the map: a position for every dentate code
                for _ in range(8):                                                                   # settle: every relation pulls its two things into place
                    err = g * (torch.einsum("bsn,bnk->bsk", kc[:, 0], Pm) - torch.einsum("bsn,bnk->bsk", kc[:, 1], Pm) - d)
                    Pm = (Pm - 0.3 * torch.einsum("bsn,bsk->bnk", kc[:, 0], err) + 0.3 * torch.einsum("bsn,bsk->bnk", kc[:, 1], err)).clamp(-8, 8)
                allp = torch.cat([torch.einsum("bsn,bnk->bsk", kc[:, 0], Pm), torch.einsum("bsn,bnk->bsk", kc[:, 1], Pm)], 1); allg = torch.cat([g, g], 1)[..., 0]
                aq = torch.softmax((qe[:, None] * s.sp_q(q).view(B, 2, Ed)[:, :, None]).sum(-1) / Ed ** 0.5 + lm[:, None], -1)
                pq = torch.einsum("brn,bnk->brk", s.code(torch.einsum("brl,ble->bre", aq, qe)), Pm)  # where the two things asked about are
                occ = [torch.tanh((torch.exp(-((pq[:, r, None, None, :] + s.dirs[None, :, None, :] - allp[:, None]) ** 2).sum(-1) / 0.25) * allg[:, None]).sum(-1)) for r in (0, 1)]   # is something one step away, in each direction?
                spf = [s.sp_out(torch.cat([pq[:, 1] - pq[:, 0]] + occ, -1))]; s.space = dict(delta=pq[:, 1] - pq[:, 0], gate=g[..., 0], d=d)
            if ENTITIES:
                bag = torch.einsum("bst,bte->bse", Mh * imp[:, None], es) / (torch.einsum("bst,bt->bs", Mh, imp)[..., None] + 1e-3)
                sv = s.en_s(torch.cat([send, bag], -1)); hc = torch.zeros(B, N_CELLS, D_ENT); wk = s.en_keys[None].expand(B, -1, -1)
                for j in range(S):                                                                  # read on: each cell decides whether this sentence concerns it
                    sj = sv[:, j, None]; gt = torch.sigmoid((sj * hc).sum(-1) + (sj * wk).sum(-1))[..., None] * smask[:, j, None, None]
                    hn = hc + gt * s.en_a(s.en_U(hc) + s.en_V(wk) + s.en_W(sj)); hc = hn / (hn.norm(dim=-1, keepdim=True).clamp(min=1.0))
                qd = s.en_q(q); pe = torch.softmax((hc * qd[:, None]).sum(-1), -1); ue = s.en_o(torch.cat([(pe[..., None] * hc).sum(1), qd], -1)); spf = spf + [ue]
            if REWARD: s.rl = dict(pol=pol, ent=ent, value=torch.sigmoid(s.critic(q))[:, 0])
            w = torch.softmax(s.mix(q), -1)
            outs = rs + rr
            recalled = list(outs)
            if ENTITIES: outs = outs + [s.en_e(ue)]
            z = sum(w[:, k:k + 1] * outs[k] for k in range(len(outs))) + w[:, len(outs):] * s.dec(torch.cat([st, q] + recalled + counts + [rs[0].norm(dim=-1, keepdim=True)] + spf, -1))
            lg = s.temp * torch.einsum("be,bce->bc", z / (z.norm(dim=-1, keepdim=True) + 1e-6), ce / (ce.norm(dim=-1, keepdim=True) + 1e-6))
            wmc = torch.sigmoid(s.beta * torch.einsum("btn,bcn->bct", keys, s.code(ce)) - 5) * kstr[:, None]     # recognition: each candidate as a cue
            g = s.recw(q); lg = lg + g[:, :1] * wmc.sum(-1)
            for k in range(HOPS): lg = lg + g[:, k + 1:k + 2] * torch.einsum("bct,bt->bc", wmc, torch.einsum("bs,bst->bt", atts[k], Mh))
            return lg, scs
    return EventMemory()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sup", type=int, default=1); ap.add_argument("--roles", action="store_true"); ap.add_argument("--sem", action="store_true"); ap.add_argument("--space", action="store_true"); ap.add_argument("--space-sup", action="store_true"); ap.add_argument("--entities", action="store_true"); ap.add_argument("--reward", action="store_true"); ap.add_argument("--hindsight", action="store_true")
    # Stage 99: how VEGA is taught (all off by default; earlier stages reproduce unchanged)
    ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--sched", default=""); ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--mix", action="store_true"); ap.add_argument("--prio", action="store_true"); ap.add_argument("--fade", type=float, default=0.0)
    ap.add_argument("--aug", type=float, default=0.0); ap.add_argument("--val", action="store_true"); a = ap.parse_args()
    global ROLES, SEM, SPACE, SPACE_SUP, ENTITIES, REWARD, HINDSIGHT; ENTITIES = a.entities; REWARD = a.reward; HINDSIGHT = a.hindsight; ROLES = a.roles; SEM = a.sem; SPACE_SUP = a.space_sup; SPACE = a.space or a.space_sup
    os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True); tag = f"events_sup{a.sup}_" + ("roles_" if a.roles else "") + ("sem_" if a.sem else "") + ("spacesup_" if a.space_sup else "space_" if a.space else "") + ("entities_" if a.entities else "") + ("reward_" if a.reward else "") + ("hindsight_" if a.hindsight else "") + ("bigreader_" if os.environ.get("BIG_READER") else "") + ("val_" if a.val else "") + (f"wd{a.wd:g}_" if a.wd else "") + (f"{a.sched}_" if a.sched else "") + (f"drop{a.drop:g}_" if a.drop else "") + ("mix_" if a.mix else "") + ("prio_" if a.prio else "") + (f"fade{a.fade:g}_" if a.fade else "") + (f"aug{a.aug:g}_" if a.aug else "") + os.environ.get("EV_TAG", "")

    def run(M, C, tok, seed, res):
        torch = M.torch; torch.set_num_threads(int(os.environ.get("THREADS", 2)))
        if os.environ.get("BIG_READER"):                                                      # Stage 95: a Stage 94 reading network in the old one's place
            import reader_adapter as RA
            M = RA.BigM(torch, os.environ["BIG_READER"]); print("reading network:", M.name, "width", M.H, flush=True)
        if os.environ.get("STORY_READER"): M.m.load_state_dict(torch.load(os.environ["STORY_READER"]), strict=False); print("reading network:", os.environ["STORY_READER"], flush=True)   # Stage 91
        M.m.eval()
        for p in M.m.parameters(): p.requires_grad_(False)
        net = build(M, a.seed); cache = {}
        LR0 = 5e-4
        opt = torch.optim.AdamW(net.parameters(), lr=LR0, weight_decay=a.wd) if a.wd else torch.optim.Adam(net.parameters(), lr=LR0)
        if SEM:
            import semantic_cortex as SC
            SV = SC.load(); scale = float(M.m.e.base.weight.norm(dim=1).mean()); zero = np.zeros(D_SEM, np.float32)     # meaning vectors at the size of the reader's own
            sem = lambda ws: torch.tensor(np.stack([SV.get(str(w), zero) for w in ws]) * scale) if len(ws) else torch.zeros(0, D_SEM)

        def emb(c):                                                                           # an answer of several words: all of them, earlier words weigh more
            if c not in cache:
                with torch.no_grad(): e = M.word_emb(c) if hasattr(M, "word_emb") else M.m.e(torch.tensor([[M.enc([w])[1] for w in c]]))[0]
                if SEM: e = torch.cat([e, sem(list(c))], -1)
                cache[c] = sum(e[i] * 0.5 ** i for i in range(len(c)))
            return cache[c]

        def batch(items):
            flat = [[w for s in it["sents"] for w in s] for it in items]; spos = [len(f) for f in flat]; ql = [len(it["q"]) for it in items]
            x, objs, om = M.batch([dict(words=f + it["q"], ends=[int(v) - 1 for v in np.cumsum([len(sn) for sn in it["sents"]])]) for f, it in zip(flat, items)])
            with torch.no_grad(): o, _ = M.m(x, objs, om); e = M.m.e(x)
            if SEM:
                sv = torch.zeros(e.shape[0], e.shape[1], D_SEM)
                for i, (f, it) in enumerate(zip(flat, items)): ws = f + it["q"]; sv[i, 1:1 + len(ws)] = sem(ws)[:e.shape[1] - 1]
                e = torch.cat([e, sv], -1)
            T = max(spos); L = max(ql); B = len(items); S = max(len(it["sents"]) for it in items)
            mask = torch.zeros(B, T); qmask = torch.zeros(B, L); qe = torch.zeros(B, L, e.shape[-1]); sid = torch.full((B, T), -1, dtype=torch.long)
            ends = torch.zeros(B, S, dtype=torch.long); smask = torch.zeros(B, S)
            for i, it in enumerate(items):
                mask[i, :spos[i]] = 1; qmask[i, :ql[i]] = 1; qe[i, :ql[i]] = e[i, spos[i] + 1:spos[i] + 1 + ql[i]]; p = 0
                for j, sn in enumerate(it["sents"]): sid[i, p:p + len(sn)] = j; p += len(sn); ends[i, j] = p - 1; smask[i, j] = 1
            q = o[torch.arange(B), torch.tensor([spos[i] + ql[i] for i in range(B)])]
            hs = o[:, 1:T + 1]; es = e[:, 1:T + 1]
            if hs.shape[1] < T: hs = torch.nn.functional.pad(hs, (0, 0, 0, T - hs.shape[1])); es = torch.nn.functional.pad(es, (0, 0, 0, T - es.shape[1]))
            ce = torch.stack([torch.stack([emb(c) for c in it["cands"]]) for it in items])
            if a.drop and net.training:                                                       # Stage 99: dropout (DMN+, robust DNC)
                D = torch.nn.functional.dropout; hs, es, q, qe = D(hs, a.drop), D(es, a.drop), D(q, a.drop), D(qe, a.drop)
            return hs * mask[..., None], es * mask[..., None], mask, q, qe, qmask, sid, ends, smask, ce

        def sup_loss(scs, items):
            tot, n = 0.0, 0
            for i, it in enumerate(items):
                sp = it.get("supp")
                if not sp: continue
                for k in range(HOPS):
                    if len(sp) <= HOPS:
                        if k < len(sp): tot = tot - scs[k][i, sp[k]]; n += 1                        # step k lands on supporting sentence k
                    else: tot = tot - torch.logsumexp(scs[k][i, sp], 0); n += 1                     # many supports: any of them
            return tot / max(n, 1)

        def hind_loss(scs, items):                                                            # hindsight: some recall step should land on a sentence that mentions the answer
            tot, n = 0.0, 0
            for i, it in enumerate(items):
                aw = set(it["cands"][it["y"]]); Hs = [j for j, sn in enumerate(it["sents"]) if aw & set(sn)]
                if Hs: tot = tot - torch.stack([torch.logsumexp(scs[k][i, Hs], 0) for k in range(HOPS)]).max(); n += 1
            return tot / max(n, 1)

        def accuracy(items, hops=None):
            net.eval(); hit = []
            with torch.no_grad():
                for k in range(0, len(items), 32):
                    it = items[k:k + 32]; lg, scs = net(*batch(it)); hit.append(lg.argmax(1).numpy() == np.array([i["y"] for i in it]))
                    if hops is not None:
                        for i, x in enumerate(it):
                            sp = x.get("supp")
                            if sp and len(sp) <= HOPS: hops.append([int(scs[j][i].argmax()) == sp[j] for j in range(len(sp))])
            net.train(); return float(np.concatenate(hit).mean())

        NT = int(os.environ.get("EV_NTEST", 0)) or None                                        # smoke tests: fewer test items
        tasks, allc = load_all(); T = sorted(tasks); rng = np.random.default_rng(890 + a.seed)
        if a.val:                                                                             # Stage 99: last 10 % of each task's training stories held out
            for t in T:
                tr = tasks[t]["train"]; k = len(tr) // 10; tasks[t]["val"] = tr[-k:]; tasks[t]["train"] = tr[:-k]
        ans_words = {w for c in allc for w in c}
        CLASSES = [AUG_PERSONS, AUG_PLACES, AUG_OBJECTS, AUG_COLOURS]

        def augment(it, t):                                                                  # Stage 99: the same story with its people, places, things renamed
            if not a.aug or t in AUG_SKIP or rng.random() >= a.aug: return it
            mp = {}
            for cl in CLASSES:
                for part in ([w for w in cl if w in ans_words], [w for w in cl if w not in ans_words]):   # answers stay answers
                    if len(part) > 1: mp.update(zip(part, [part[i] for i in rng.permutation(len(part))]))
            sub = lambda ws: [mp.get(w, w) for w in ws]
            ans = tuple(sub(it["ans"]))
            if ans not in allc: return it                                                    # a list answer in an order never seen: keep the original
            new = dict(it, sents=[sub(sn) for sn in it["sents"]], q=sub(it["q"]), ans=ans, y=allc.index(ans))
            if os.environ.get("AUG_SHOW") and rng.random() < 0.02: print("AUG", t, " | ".join(" ".join(x) for x in it["sents"][-3:]), "?", " ".join(it["q"]), it["ans"], "=>", " | ".join(" ".join(x) for x in new["sents"][-3:]), "?", " ".join(new["q"]), new["ans"], flush=True)
            return new

        tw = np.ones(len(T)) / len(T)                                                         # how often each task is practised (--prio changes this)

        def task_items(n):
            if a.mix:                                                                         # interleaved: every question from its own task
                ts = rng.choice(len(T), n, p=tw); out = []
                for k in ts:
                    tr = tasks[T[k]]["train"]; out.append(augment(tr[int(rng.integers(len(tr)))], T[k]))
                return out
            k = int(rng.choice(len(T), p=tw)) if a.prio else int(rng.integers(len(T))); tr = tasks[T[k]]["train"]   # uniform draw kept bit-identical to earlier stages
            return [augment(tr[i], T[k]) for i in rng.choice(len(tr), n, replace=False)]

        def reprioritise():                                                                   # spaced repetition by error: practise what is still wrong
            nonlocal tw
            src = "val" if a.val else "train"
            err = np.array([1 - accuracy(tasks[t][src][:64]) for t in T])
            tw = 0.5 / len(T) + 0.5 * (err + 0.02) / (err + 0.02).sum()
            return err
        PASS = None
        if os.environ.get("PASSAGES"):
            raw = json.load(gzip.open(os.environ["PASSAGES"], "rt")); PASS = {sp: {b: [passage_item(it) for it in raw[sp][b]] for b in raw[sp]} for sp in raw}
        path = os.path.join(DATA, f"{tag}s{a.seed}.pt"); ck = path + ".ckpt"; start = 0; log = []; t0 = time.time(); last = time.time(); used = 0.0
        if os.path.exists(ck):
            z = torch.load(ck, weights_only=False); net.load_state_dict(z["net"]); opt.load_state_dict(z["opt"]); start = z["step"]; log = z["log"]; rng.bit_generator.state = z["rng"]; used = z.get("used", 0.0)
            print("resumed at step", start, flush=True)
        if not os.path.exists(path):
            for step in range(start, QUESTIONS // BS):
                if time.time() - last > 600:
                    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, log=log, rng=rng.bit_generator.state, used=used + time.time() - t0), ck + ".tmp"); os.replace(ck + ".tmp", ck); last = time.time()
                u = rng.random()
                if u < P_WORLD: items = world_batch(rng, BS)
                elif PASS and u < P_WORLD + P_PASS:
                    bd = list(PASS["train"])[int(rng.integers(len(PASS["train"])))]; tr = PASS["train"][bd]; items = [tr[i] for i in rng.choice(len(tr), BS, replace=False)]
                else:
                    items = task_items(BS)
                nsteps = QUESTIONS // BS; frac = step / nsteps
                if a.sched == "cos":                                                              # Stage 99: 2 % warm-up, then cosine down to 5 %
                    lr = LR0 * (frac / 0.02 if frac < 0.02 else 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * (frac - 0.02) / 0.98)))
                    for g in opt.param_groups: g["lr"] = lr
                if a.prio and step % PRIO_EVERY == 0 and step > 0: print("task weights", np.round(tw, 3).tolist(), "errors", np.round(reprioritise(), 2).tolist(), flush=True)
                lg, scs = net(*batch(items)); loss = torch.nn.functional.cross_entropy(lg, torch.tensor([i["y"] for i in items]))
                sw = SUP_W * (1 - (1 - a.fade) * min(1.0, frac / 0.8) if a.fade else 1.0)          # Stage 99: step teaching fades to --fade of its weight by 80 % of training
                if a.sup: loss = loss + sw * sup_loss(scs, items)
                if HINDSIGHT: loss = loss + hind_loss(scs, items)
                if REWARD:
                    rwd = (lg.argmax(1) == torch.tensor([i["y"] for i in items])).float(); rl = net.rl; rpe = rwd - rl["value"].detach()   # better or worse than expected
                    loss = loss - (rpe * rl["pol"]).mean() + ((rl["value"] - rwd) ** 2).mean() - 0.01 * rl["ent"].mean()
                if SPACE_SUP:
                    ix = [i for i, it in enumerate(items) if it.get("ans") and all(w in MOVES for w in it["ans"])]
                    if ix: loss = loss + ((net.space["delta"][ix] - torch.tensor([np.sum([MOVES[w] for w in items[i]["ans"]], 0) for i in ix], dtype=torch.float32)) ** 2).sum(-1).mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                if (step + 1) % 1000 == 0 or step + 1 == QUESTIONS // BS:
                    log.append(dict(questions=(step + 1) * BS, minutes=round((used + time.time() - t0) / 60, 1), loss=round(float(loss.detach()), 3))); print(log[-1], flush=True)
            torch.save(net.state_dict(), path)
            if os.path.exists(ck): os.remove(ck)
        else: net.load_state_dict(torch.load(path))
        if os.environ.get("PROBE_SPACE") and SPACE:                                          # what the map made of each sentence
            net.eval()
            err = []
            for k in range(0, 400, 32):                                                       # does the map read the path's displacement?
                its = tasks[19]["test"][k:k + 32]
                with torch.no_grad(): net(*batch(its))
                err += ((net.space["delta"] - torch.tensor([np.sum([MOVES[w] for w in it["ans"]], 0) for it in its], dtype=torch.float32)).abs() < 0.5).all(-1).tolist()
            print("path questions where the map's displacement is within half a step of the true one:", round(float(np.mean(err)), 3))
            for t in (17, 19, 1):
                for it in tasks[t]["test"][:2]:
                    with torch.no_grad(): lg, _ = net(*batch([it]))
                    print(f"\n{tasks[t]['name']} Q: {' '.join(it['q'])} | answer {it['ans']} | said {it['cands'][int(lg.argmax())]} | delta {[round(x, 2) for x in net.space['delta'][0].tolist()]}")
                    for sn, g, d in zip(it["sents"], net.space["gate"][0].tolist(), net.space["d"][0].tolist()): print(f"    gate {g:.2f} d ({d[0]:+.2f}, {d[1]:+.2f})  {' '.join(sn)}")
            raise SystemExit
        out = dict(seed=a.seed, questions=QUESTIONS, training=log, tasks={}, step_hits={},
                   config=dict(reader=os.path.basename(os.environ.get("BIG_READER", "") or os.environ.get("STORY_READER", "")) or "stage 73", sup=a.sup, roles=ROLES, sem=SEM, space=SPACE, space_sup=SPACE_SUP, entities=ENTITIES, reward=REWARD, hindsight=HINDSIGHT, passages=bool(PASS), one_answer_list=len(allc), hops=HOPS, tag=tag, n_params=sum(p.numel() for p in net.parameters())))
        for t in T:
            hops = []; acc = accuracy(tasks[t]["test"][:NT], hops); out["tasks"][tasks[t]["name"]] = round(acc, 4)
            if hops: out["step_hits"][tasks[t]["name"]] = [round(float(np.mean([h[j] for h in hops if len(h) > j])), 3) for j in range(max(map(len, hops)))]   # does recall step j land on supporting sentence j?
            print(tasks[t]["name"], round(acc, 4), out["step_hits"].get(tasks[t]["name"]), flush=True)
        out["passed_95"] = int(sum(v >= 0.95 for v in out["tasks"].values())); out["mean_error"] = round(100 * (1 - np.mean(list(out["tasks"].values()))), 2)
        out["recipe"] = dict(wd=a.wd, sched=a.sched, drop=a.drop, mix=a.mix, prio=a.prio, fade=a.fade, aug=a.aug, val=a.val)
        if a.val:
            out["val"] = {tasks[t]["name"]: round(accuracy(tasks[t]["val"][:NT]), 4) for t in T}
            out["val_mean_error"] = round(100 * (1 - np.mean(list(out["val"].values()))), 2)
        out["train_sample"] = {tasks[t]["name"]: round(accuracy(tasks[t]["train"][:300]), 4) for t in T}       # how well it knows what it practised
        if a.prio: out["task_weights"] = {tasks[t]["name"]: round(float(w), 4) for t, w in zip(T, tw)}
        out["world"] = {k: round(accuracy([world_item(it) for it in v]), 4) for k, v in W.test_sets(seed=0).items() for v in [v[:NT]]}
        if PASS:
            out["passages"], out["passages_reader"], out["passages_both"], out["passages_most_frequent"] = {}, {}, {}, {}
            for bd, its in PASS["test"].items():
                its = its[:NT]; y = np.array([it["y"] for it in its]); net.eval(); hz = []
                with torch.no_grad():
                    for k0 in range(0, len(its), 32): hz.append(torch.log_softmax(net(*batch(its[k0:k0 + 32]))[0], -1).numpy())
                net.train(); hz = np.concatenate(hz)
                lp = np.array(M.logp([it["sent"][:it["pos"]] + [c[0]] + it["sent"][it["pos"] + 1:] for it in its for c in it["cands"]])).reshape(len(its), -1)
                flat = [[w for s in it["sents"] for w in s] for it in its]
                out["passages"][bd] = round(float(np.mean(hz.argmax(1) == y)), 4); out["passages_reader"][bd] = round(float(np.mean(lp.argmax(1) == y)), 4)
                out["passages_both"][bd] = round(float(np.mean((hz + lp - lp.max(1, keepdims=True)).argmax(1) == y)), 4)
                if hasattr(M, "score"):                                                           # Stage 95: the big reader after reading the passage (sentences end-marked)
                    ctx = []
                    for it in its: ctx += [[w for sn in it["sents"] for w in sn + ["</s>"]]] * len(it["cands"])
                    lc = M.score(ctx, [it["sent"][:it["pos"]] + [c[0]] + it["sent"][it["pos"] + 1:] for it in its for c in it["cands"]]).reshape(len(its), -1)
                    out.setdefault("passages_reader_context", {})[bd] = round(float(np.mean(lc.argmax(1) == y)), 4)
                    out.setdefault("passages_both_context", {})[bd] = round(float(np.mean((hz + lc - lc.max(1, keepdims=True)).argmax(1) == y)), 4)
                if os.environ.get("STORY_READER"):                                                # the reading network after reading the passage (Stage 91)
                    import story_reader as SR
                    lc = SR.context_logp(M, [f for f, it in zip(flat, its) for _ in it["cands"]], [it["sent"][:it["pos"]] + [c[0]] + it["sent"][it["pos"] + 1:] for it in its for c in it["cands"]]).reshape(len(its), -1)
                    out.setdefault("passages_reader_context", {})[bd] = round(float(np.mean(lc.argmax(1) == y)), 4)
                    out.setdefault("passages_both_context", {})[bd] = round(float(np.mean((hz + lc - lc.max(1, keepdims=True)).argmax(1) == y)), 4)
                out["passages_most_frequent"][bd] = round(float(np.mean([max(range(len(it["cands"])), key=lambda j: (f.count(it["cands"][j][0]), -j)) == it["y"] for it, f in zip(its, flat)])), 4)
            print({k: out[k] for k in out if k.startswith("passages")}, flush=True)
        json.dump(out, open(os.path.join(OUT, f"{tag}s{a.seed}.json"), "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "training"})[:3000], flush=True)
        raise SystemExit

    GM.evaluate = run; OO.DATA = "/home/claude/models/day_night/year4_cycle"; OO.OUT = "/tmp/hippo_scratch"; os.makedirs(OO.OUT, exist_ok=True)
    OO.run(argparse.Namespace(cond="books", seed=a.seed))


if __name__ == "__main__":
    main()
