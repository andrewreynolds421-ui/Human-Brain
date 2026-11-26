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

Stage 80 (--episodes): an episode record beside CA3 - every stored event is kept in order, not overwritten (as
the hippocampus records sequences of events: time cells, sequence replay). Four extra recalls: the latest event
about the question's word; the event where it was with the second question word; what was true just before that
("where was the apple before the bathroom"); and the latest event about what was recalled (chaining). A count of
the events matching the question word goes to CA1. Learning steps clipped and a smaller step size (one Stage 79
network failed to learn). 160,000 training questions.

    python scripts/hippo_babi.py --seed 0
    QUESTIONS=160000 python scripts/hippo_babi.py --episodes --seed 0

Stage 82 (--concepts, includes the episode record): (1) CA1's output and the answer words pass through one learned
concept map (starts as identity) before they are compared - VEGA's reading network keeps "wolves" nearer "cats"
than "wolf" (number before meaning), so a correctly recalled "mice" could not be matched to the answer "mouse";
(2) reverse recall: from a recalled word to whose event had it as content ("swan" -> lily), then the latest event
about that one (induction: "lily is a swan, lily is white, bernhard is a swan - what colour is bernhard?").

    QUESTIONS=160000 python scripts/hippo_babi.py --concepts --seed 0

Stage 83 (--controller, includes the episode record; compare with Stage 80): a recall loop instead of fixed steps.
Prefrontal working memory (GRU, MODEL) holds the question and everything recalled so far; each step it forms a cue
(attention over the question words and earlier recalls), the basal ganglia (MODEL) choose how to ask the
hippocampus (CA3 lookup / latest event about the cue / the event before the last one found) and whether to answer
or recall again (at most 6 steps; a small cost per step, ASSUMED 0.003). Logs mean steps per task.
--steady: control - Stage 79's circuit with only Stage 80's learning settings (lr 5e-4, clipping).

    QUESTIONS=160000 python scripts/hippo_babi.py --controller --seed 0

Stage 84 (--loop, includes the episode record; with --controller the controller's recalls re-enter too): chained
recall goes through a learned re-entry map (CA1 -> deep EC -> superficial EC; "big-loop recurrence", Kumaran &
McClelland 2012) instead of reusing the recalled word as the next cue unchanged. Starts as an exact copy (Stage 80).

    QUESTIONS=160000 python scripts/hippo_babi.py --loop --seed 0
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
EPISODES = False                                                        # Stage 80: the episode record (set by --episodes)
CONTROLLER = False                                                      # controller: prefrontal working memory + basal-ganglia recall loop (set by --controller)
K_STEPS = 6; PONDER = 0.003; CTRL_ONLY = False                         # --ctrl-only: the controller is the only way to chain recalls; step cost phased in                                             # most recall steps; cost per step (ASSUMED)
PATTERN = False                                                         # Stage 85: event store with pattern completion (--pattern; implies --loop --allcues)
ALLCUES = False                                                         # Stage 84b: chain from every question word in parallel (--allcues)
LOOP = False                                                            # Stage 84: learned big-loop re-entry (CA1 -> EC deep -> EC superficial) for chained recall (--loop)
CONCEPTS = False                                                        # Stage 82: concept map + reverse recall (set by --concepts)


def words(s):
    return re.findall(r"[a-z]+", s.lower())


def load_task(path):
    items, story = [], []
    for line in open(path):
        n, rest = line.split(" ", 1)
        if n == "1": story = []
        if "\t" in rest:
            q, ans, _ = rest.split("\t")
            kept = story[-MAX_SENT:]; ends = list(np.cumsum([len(x) for x in kept]) - 1)          # Stage 85: where each sentence ends (the full stop)
            items.append(dict(story=[w for s in kept for w in s], q=words(q), ans=tuple(words(ans.replace(",", " "))), ends=ends))
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
            n_ep = (4 if EPISODES else 0) + (2 if CONCEPTS else 0) + (1 if CONTROLLER else 0) + (2 if ALLCUES else 0)
            s.mix = nn.Linear(H, 6 + n_ep); s.dec = nn.Sequential(nn.Linear(H + (6 + n_ep) * E + 4 + (3 if EPISODES else 0), 256), nn.Tanh(), nn.Linear(256, E)); s.temp = nn.Parameter(torch.tensor(5.0))
            if EPISODES: s.beta = nn.Parameter(torch.tensor(10.0)); s.tw = nn.Parameter(torch.zeros(8 if CONTROLLER else 5, 3))      # per episode read: weights of recency, "just before", value match
            if CONCEPTS:                                                                          # CA1 output -> concepts: one map for every word form, learned, starts as identity
                s.cmap = nn.Linear(E, E); s.beta_inv = nn.Parameter(torch.tensor(10.0))
                with torch.no_grad(): s.cmap.weight.copy_(torch.eye(E)); s.cmap.bias.zero_()
            if CONTROLLER:                                                                        # PFC: working memory of the question and what was recalled
                P = 256; s.pfc0 = nn.Linear(H, P); s.pfc = nn.GRUCell(2 * E + 3, P); s.cue_q = nn.Linear(P, E)
                s.act = nn.Linear(P, 4 if CONCEPTS else 3)                                        # basal ganglia: which way to ask the hippocampus
                s.stop = nn.Linear(P, 1)                                                          # basal ganglia: answer now, or recall again
                s.out = nn.Sequential(nn.Linear(P + E, 256), nn.Tanh(), nn.Linear(256, E))
                with torch.no_grad(): s.stop.bias.fill_(-1.0)
            if ALLCUES: s.qa = nn.Linear(H, E)
            if PATTERN:                                                                           # event store: boundary gate, word importance
                s.bgate = nn.Linear(H, 1); s.bw = nn.Parameter(torch.tensor(4.0)); s.sal = nn.Linear(E, 1); s.ksal = nn.Linear(H, 1)   # importance as content; strength as a key (from context: role in the sentence)
                with torch.no_grad(): s.bgate.bias.fill_(-3.0)                                                    # which question word's chain to trust
            if LOOP:                                                                              # re-entry: recalled content -> next cue; starts as a plain copy
                s.reent = nn.Sequential(nn.Linear(E + H, 256), nn.Tanh(), nn.Linear(256, E))
                with torch.no_grad(): s.reent[2].weight.zero_(); s.reent[2].bias.zero_()
            g = torch.Generator().manual_seed(77 + seed); s.register_buffer("dg", torch.randn(D_KEY, N_DG, generator=g) / D_KEY ** 0.5)

        def code(s, e):
            k = s.wk(e); a = s.pp(k); d = torch.relu(k @ s.dg); kth = d.topk(int(ACTIVE * N_DG), dim=-1).values[..., -1:]
            d = d * (d >= kth); a = a + d / (d.norm(dim=-1, keepdim=True) + 1e-6); return a / (a.norm(dim=-1, keepdim=True) + 1e-6)

        def forward(s, hs, es, mask, q, qe, qmask, bnd=None):
            B, T, _ = hs.shape; Ed = es.shape[-1]
            pad = torch.cat([torch.zeros(B, WIN - 1, Ed), es], 1); win = torch.stack([pad[:, j:j + T] for j in range(WIN)], 2)
            ak = torch.softmax(s.pick_k(hs), -1); ar = torch.softmax(s.pick_r(hs), -1); av = torch.softmax(s.pick_v(hs), -1)
            keys = s.code((ak[..., None] * win).sum(2)); rkeys = s.code((ar[..., None] * win).sum(2)); g = torch.sigmoid(s.gate(hs)) * mask[..., None]
            Mem = torch.zeros(B, N_DG, Ed); vals = []
            for t in range(T):
                rec = torch.einsum("bn,bne->be", rkeys[:, t], Mem)                                  # recall while storing
                opts = torch.cat([win[:, t], rec[:, None]], 1); v = (av[:, t, :, None] * opts).sum(1); vals.append(v)
                c = keys[:, t]; old = torch.einsum("bn,bne->be", c, Mem)
                Mem = Mem + g[:, t, :, None] * c[:, :, None] * (v - old)[:, None, :]
            read = lambda e: torch.einsum("bn,bne->be", s.code(e), Mem)
            if PATTERN:                                                                           # Stage 85: store each event so any of its words recalls the rest
                bnd = torch.zeros(B, T) if bnd is None else bnd[:, :T]
                seg = torch.cumsum(bnd, 1) - bnd                                                  # sentence number of each word (an end belongs to its sentence)
                spad = torch.cat([torch.full((B, WIN - 1), -1.0), seg], 1); sw = torch.stack([spad[:, j:j + T] for j in range(WIN)], 2)
                inc = (sw == seg[..., None]).float()                                              # window words from the same sentence
                imp = torch.sigmoid(s.sal(win))[..., 0] * inc                                     # learned importance ("is", "a" should be low)
                tot = (imp[..., None] * win).sum(2, keepdim=True)
                vals_ev = (tot - imp[..., None] * win) / ((imp.sum(2, keepdim=True) - imp)[..., None] + 1e-3)   # value for each word = the rest of its event
                hpad = torch.cat([torch.zeros(B, WIN - 1, hs.shape[-1]), hs], 1); hwin = torch.stack([hpad[:, j:j + T] for j in range(WIN)], 2)
                kstr = torch.sigmoid(s.ksal(hwin))[..., 0] * inc                                  # how strongly each word files the event (subject vs object)
                ckeys = s.code(win); gate_ev = torch.sigmoid(s.bgate(hs)[..., 0] + s.bw * bnd) * mask
                Mem2 = torch.zeros(B, N_DG, Ed)
                for t in range(T):
                    c = ckeys[:, t]; old = torch.einsum("bjn,bne->bje", c, Mem2); wgt = (gate_ev[:, t, None] * kstr[:, t])[..., None, None]
                    Mem2 = Mem2 + (wgt * c[..., None] * (vals_ev[:, t] - old)[:, :, None, :]).sum(1)
                s.ev = dict(gate=gate_ev, imp=imp, kstr=kstr); s.Mem2 = Mem2
            lm = torch.log(qmask + 1e-9)
            c1 = (torch.softmax((qe * s.q1(q)[:, None]).sum(-1) / Ed ** 0.5 + lm, -1)[..., None] * qe).sum(1)
            c2 = (torch.softmax((qe * s.q2(q)[:, None]).sum(-1) / Ed ** 0.5 + lm, -1)[..., None] * qe).sum(1)
            R = (lambda x: x + s.reent(torch.cat([x, q], -1))) if LOOP else (lambda x: x)        # Kumaran & McClelland 2012: recurrence through EC
            r1 = read(c1); r2 = read(R(r1)); r3 = read(R(r2)); s1 = read(c2)
            s.trace = dict(cue1=c1, cue2=c2, r1=r1, R_r1=R(r1), r2=r2, r3=r3, s1=s1)                 # for probing (PROBE=1)
            allc = []
            if ALLCUES:                                                                           # every question word starts its own two-step chain
                MemC = Mem2 if PATTERN else Mem                                                   # Stage 85: chains read the event store
                X1 = torch.einsum("bln,bne->ble", s.code(qe), MemC); Lq = qe.shape[1]
                X1R = X1 + s.reent(torch.cat([X1, q[:, None].expand(-1, Lq, -1)], -1)) if LOOP else X1
                X2 = torch.einsum("bln,bne->ble", s.code(X1R), MemC)
                aq = torch.softmax((qe * s.qa(q)[:, None]).sum(-1) / Ed ** 0.5 + lm, -1)[..., None]
                ch1 = (aq * X1).sum(1); ch2 = (aq * X2).sum(1); allc = [ch1, ch2]; s.trace.update(chain1=ch1, chain2=ch2)
                if os.environ.get("PROBE"): s.trace.update({f"w{j}_1": X1[:, j] for j in range(Lq)}); s.trace.update({f"w{j}_2": X2[:, j] for j in range(Lq)}); s.trace["trust"] = aq[..., 0]
            if CTRL_ONLY: r2 = torch.zeros_like(r1); r3 = torch.zeros_like(r1)                  # no fixed second / third step
            cos = lambda a, b: (a * b).sum(-1, keepdim=True) / (a.norm(dim=-1, keepdim=True) * b.norm(dim=-1, keepdim=True) + 1e-6)
            feats = torch.cat([cos(r1, c2), cos(r2, c2), cos(r3, c2), r1.norm(dim=-1, keepdim=True)], -1)
            opts = [r1, r2, r3, s1, c2] + allc; extra = list(allc)
            if EPISODES:                                                                          # the episode record: every stored event, in order
                V = torch.stack(vals, 1); tpos = torch.arange(T, dtype=torch.float32)[None] / max(T, 1)
                logg = torch.log(g[..., 0] + 1e-6) + torch.log(mask + 1e-9)
                Vn = V / (V.norm(dim=-1, keepdim=True) + 1e-6)

                def ep(i, cue, val=None, tau=None):
                    sc = s.beta * (keys * s.code(cue)[:, None]).sum(-1) + logg + s.tw[i, 0] * 10 * tpos
                    if val is not None: sc = sc + s.tw[i, 2] * 10 * (Vn * (val / (val.norm(dim=-1, keepdim=True) + 1e-6))[:, None]).sum(-1)
                    if tau is not None: sc = sc - 1e4 * (tpos >= tau[:, None] - 1e-6) + s.tw[i, 1] * 10 * tpos     # only events before tau
                    a = torch.softmax(sc, -1); return (a[..., None] * V).sum(1), (a * tpos).sum(-1)
                A, tA = ep(0, c1)                                    # the latest event about the question's word
                F, tF = ep(1, c1, val=c2)                            # the event where it was with the second question word ("... the bathroom")
                Cb, _ = ep(2, c1, tau=tF)                            # what was true just before that
                D, _ = ep(3, R(A))                                   # chain: the latest event about what was recalled
                if CTRL_ONLY: D = torch.zeros_like(A); Cb = torch.zeros_like(A)
                count = (torch.sigmoid(s.beta * (keys * s.code(c1)[:, None]).sum(-1) - 5) * g[..., 0]).sum(-1, keepdim=True)
                s.trace.update(A=A, F=F, Cb=Cb, D=D)
                opts += [A, Cb, D, F]; extra += [A, Cb, D, F, count, cos(A, c2), cos(D, c2)]
                if CONCEPTS:                                                                      # reverse recall: whose event had this as its content ("swan" -> lily)
                    KW = (ak[..., None] * win).sum(2); KWn = KW / (KW.norm(dim=-1, keepdim=True) + 1e-6)
                    own = (KWn * (c1 / (c1.norm(dim=-1, keepdim=True) + 1e-6))[:, None]).sum(-1)          # not the question's own subject
                    sc = s.beta_inv * (Vn * (A / (A.norm(dim=-1, keepdim=True) + 1e-6))[:, None]).sum(-1) + logg - 10 * own
                    I = (torch.softmax(sc, -1)[..., None] * KW).sum(1)
                    Ia, _ = ep(4, I)                                                              # then: the latest event about that one
                    opts += [I, Ia]; extra += [I, Ia]
                if CONTROLLER:                                                                    # recall loop: cue -> chosen read -> working memory -> stop?
                    hp = torch.tanh(s.pfc0(q)); found = []; tau = torch.full((B,), 1.01); remain = torch.ones(B); zc = 0; steps = torch.zeros(B)
                    for k in range(K_STEPS):
                        S = torch.cat([qe] + [f[:, None] for f in found], 1); Sm = torch.cat([qmask] + [torch.ones(B, 1)] * len(found), 1)
                        cue = (torch.softmax((S * s.cue_q(hp)[:, None]).sum(-1) / Ed ** 0.5 + torch.log(Sm + 1e-9), -1)[..., None] * S).sum(1)
                        r0 = read(cue); r1e, t1 = ep(5, cue); r2e, t2 = ep(6, cue, tau=tau); acts = [r0, r1e, r2e]
                        if CONCEPTS:
                            sc = s.beta_inv * (Vn * (cue / (cue.norm(dim=-1, keepdim=True) + 1e-6))[:, None]).sum(-1) + logg - 10 * own
                            acts.append((torch.softmax(sc, -1)[..., None] * KW).sum(1))
                        pa = torch.softmax(s.act(hp), -1); r = sum(pa[:, i:i + 1] * x for i, x in enumerate(acts))
                        tau = pa[:, 0] * tau + pa[:, 1] * t1 + pa[:, 2] * t2 + (pa[:, 3] * tau if CONCEPTS else 0)   # the time pointer moves with episode reads
                        hp = s.pfc(torch.cat([r, cue, cos(r, c1), cos(r, c2), r.norm(dim=-1, keepdim=True)], -1), hp); found.append(R(r))
                        p = torch.sigmoid(s.stop(hp))[:, 0] if k < K_STEPS - 1 else torch.ones(B)
                        zc = zc + (remain * p)[:, None] * s.out(torch.cat([hp, r], -1)); steps = steps + remain; remain = remain * (1 - p)
                    s.steps = steps; opts += [zc]; extra += [zc]
            w = torch.softmax(s.mix(q), -1); s.w = w
            z = sum(w[:, i:i + 1] * o for i, o in enumerate(opts)) + w[:, len(opts):len(opts) + 1] * s.dec(torch.cat([q, r1, r2, r3, s1, c1, c2, feats] + extra, -1))
            return z
    return Hippo2()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--episodes", action="store_true"); ap.add_argument("--concepts", action="store_true"); ap.add_argument("--controller", action="store_true"); ap.add_argument("--steady", action="store_true"); ap.add_argument("--ctrl-only", action="store_true"); ap.add_argument("--loop", action="store_true"); ap.add_argument("--allcues", action="store_true"); ap.add_argument("--pattern", action="store_true")   # control: Stage 79 circuit, Stage 80 learning settings only
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    global EPISODES, CONCEPTS, CONTROLLER, CTRL_ONLY, LOOP, ALLCUES, PATTERN; PATTERN = a.pattern; a.loop = a.loop or a.pattern; a.allcues = a.allcues or a.pattern; LOOP = a.loop; ALLCUES = a.allcues; a.episodes = a.episodes or a.loop; CTRL_ONLY = a.ctrl_only; a.controller = a.controller or a.ctrl_only; CONCEPTS = a.concepts; CONTROLLER = a.controller; EPISODES = a.episodes or a.concepts or a.controller; a.episodes = EPISODES
    if a.steady: tag = "steady_"
    elif a.pattern: tag = "pattern_" + ("concepts_" if a.concepts else "") + ("controller_" if a.controller else "")
    elif a.loop: tag = "loop_" + ("allcues_" if a.allcues else "") + ("concepts_" if a.concepts else "") + ("controller_" if a.controller else "")
    else: tag = ("controller2_" if a.ctrl_only else "controller_" if a.controller else "") + ("concepts_" if a.concepts else "" if a.controller else "episodes_" if a.episodes else "")

    def run(M, C, tok, seed, res):
        torch = M.torch; M.m.eval()
        for p in M.m.parameters(): p.requires_grad_(False)
        net = build(M, a.seed); opt = torch.optim.Adam(net.parameters(), lr=5e-4 if (a.episodes or a.steady) else 1e-3); cache = {}

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
            bnd = torch.zeros(B, T)
            for i, it in enumerate(items):
                for k in it.get("ends", []):
                    if k < T: bnd[i, k] = 1
            return hs * mask[..., None], es * mask[..., None], mask, q, qe, qmask, ce, bnd

        def logits(b):
            z = net(*b[:6], bnd=b[7]); ce = b[6]
            if CONCEPTS: z = net.cmap(z); ce = net.cmap(ce)
            return net.temp * torch.einsum("be,bce->bc", z / (z.norm(dim=-1, keepdim=True) + 1e-6), ce / (ce.norm(dim=-1, keepdim=True) + 1e-6))

        how = {}; dump = {}; DUMP = os.environ.get("DUMP"); ONLY = os.environ.get("ONLY_TASKS")   # diagnosis: record every answer; test only some tasks

        def accuracy(items, name=None):
            net.eval(); hit = []; st = []; wc = []
            with torch.no_grad():
                for k in range(0, len(items), 32):
                    it = items[k:k + 32]; pr = logits(batch(it)).argmax(1).numpy(); hit.append(pr == np.array([i["y"] for i in it]))
                    if DUMP and name and (DUMP == "1" or any(name.startswith(f"qa{n}_") for n in DUMP.split(","))): dump.setdefault(name, []).extend(dict(story=" ".join(" ".join(x) if isinstance(x, (list, tuple)) else str(x) for x in i["story"][-80:]), q=" ".join(map(str, i["q"])), y=" ".join(map(str, i["cands"][i["y"]])), p=" ".join(map(str, i["cands"][j]))) for i, j in zip(it, pr))
                    if CONTROLLER: st.append(net.steps.numpy()); wc.append(net.w[:, -2].numpy())      # recall steps taken; weight on the controller's answer
            if CONTROLLER and name: how[name] = dict(steps=round(float(np.concatenate(st).mean()), 2), weight=round(float(np.concatenate(wc).mean()), 3))
            net.train(); return float(np.concatenate(hit).mean())

        tasks = load_all(); rng = np.random.default_rng(700 + a.seed); T = sorted(tasks)
        path = os.path.join(DATA, f"{tag}s{a.seed}.pt"); ck = path + ".ckpt"; start = 0; log = []; t0 = time.time(); last = time.time()
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
                loss = torch.nn.functional.cross_entropy(logits(b), y)
                if CONTROLLER: loss = loss + PONDER * (min(1.0, step / (QUESTIONS // BS / 2)) if CTRL_ONLY else 1.0) * net.steps.mean()                            # thinking costs time
                opt.zero_grad(); loss.backward()
                if a.episodes or a.steady: torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)               # Stage 80: steadier learning
                opt.step()
                if (step + 1) % 1000 == 0 or step + 1 == QUESTIONS // BS:
                    log.append(dict(questions=(step + 1) * BS, minutes=round((time.time() - t0) / 60, 1), loss=round(float(loss.detach()), 3))); print(log[-1], flush=True)
            torch.save(net.state_dict(), path)
            if os.path.exists(ck): os.remove(ck)
        else: net.load_state_dict(torch.load(path))
        if os.environ.get("PROBE"):                                                           # what is recalled at each step, as the nearest words
            words = sorted({w for t in T for it in tasks[t]["test"][:300] for w in it["story"] + it["q"] if isinstance(w, str)})
            WE = torch.stack([emb((w,)) for w in words]); WEn = WE / (WE.norm(dim=-1, keepdim=True) + 1e-6); net.eval()
            for t in T:
                if not any(tasks[t]["name"].startswith(f"qa{n}_") for n in os.environ["PROBE"].split(",")): continue
                for it in tasks[t]["test"][:int(os.environ.get("PROBE_N", "6"))]:
                    with torch.no_grad(): z = logits(batch([it]))
                    print("\nSTORY:", " ".join(it["story"][-60:]), "| Q:", " ".join(it["q"]), "| answer", it["cands"][it["y"]], "| said", it["cands"][int(z.argmax())])
                    for k, v in net.trace.items():
                        if k == "trust": print("   trust ", " ".join(f"{w}:{x:.2f}" for w, x in zip(it["q"], v[0].tolist()))); continue
                        v = v[0] / (v[0].norm() + 1e-6); sim = WEn @ v; top = sim.topk(3)
                        print(f"   {k:6s}", ", ".join(f"{words[i]} {x:.2f}" for x, i in zip(top.values.tolist(), top.indices.tolist())))
                    print("   mix   ", " ".join(f"{x:.2f}" for x in net.w[0].tolist()))
            raise SystemExit
        out = dict(seed=a.seed, questions=QUESTIONS, training=log, tasks={})
        out["config"] = dict(EPISODES=EPISODES, CONCEPTS=CONCEPTS, CONTROLLER=CONTROLLER, CTRL_ONLY=CTRL_ONLY, LOOP=LOOP, ALLCUES=ALLCUES, PATTERN=PATTERN, tag=tag, n_params=sum(p.numel() for p in net.parameters()))   # what was actually on
        for t in T:
            if ONLY and not any(tasks[t]["name"].startswith(f"qa{n}_") for n in ONLY.split(",")): continue
            acc = accuracy(tasks[t]["test"], tasks[t]["name"]); out["tasks"][tasks[t]["name"]] = round(acc, 4); print(tasks[t]["name"], round(acc, 4), flush=True)
        out["passed_95"] = int(sum(v >= 0.95 for v in out["tasks"].values())); out["mean_error"] = round(100 * (1 - np.mean(list(out["tasks"].values()))), 2)
        out["controller"] = how
        if DUMP: out["dump"] = dump
        if ONLY: json.dump(out, open(os.path.join(OUT, f"{tag}s{a.seed}.json"), "w"), indent=1); raise SystemExit
        out["world"] = {k: round(accuracy(world_items(None, 0, v)), 4) for k, v in W.test_sets(seed=0).items()}
        json.dump(out, open(os.path.join(OUT, f"{tag}s{a.seed}.json"), "w"), indent=1); print(json.dumps(out), flush=True)
        raise SystemExit

    GM.evaluate = run; OO.DATA = "/home/claude/models/day_night/year4_cycle"; OO.OUT = "/tmp/hippo_scratch"; os.makedirs(OO.OUT, exist_ok=True)
    OO.run(argparse.Namespace(cond="books", seed=a.seed))


if __name__ == "__main__":
    main()
