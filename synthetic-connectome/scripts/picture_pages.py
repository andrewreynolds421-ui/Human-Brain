#!/usr/bin/env python3
"""Stage 55: picture-book pages - seeing who is doing it.

Stage 54: a slot that held the subject did not help, because ordinary text rarely rewards reading the verb's
number from the subject. In a picture book it does: "the bird that the cats see IS singing" is right because
the one singing in the picture is one bird. Children are thought to find the subject this way - the noun for
the one who acts (semantic bootstrapping, Pinker 1984).

PAGES (generated): a scene of 2-3 kinds of things, one or several of each, from the 34 things the picture
system can see (Stage 45: object cells of one of 40 drawings per thing); one kind ACTS (singing, sleeping,
eating, running, playing, jumping). The model sees each thing as (object cells, one-or-several, action seen) -
the number and action codes stand for what the eye would give (MODEL; actions are not drawn). Sentences of
rising difficulty describe the page:
  simple   the bird is singing                   pp     the bird by the cats is singing
  rel_obj  the bird that the cats see is singing  rel_subj  the bird that sees the cats is singing
  comp     the cats see that the bird is singing
The other kind's number is random, so half the pages have a misleading noun before the verb.

MODEL: the gated memory (GRU, 256) that, at every word, LOOKS at the scene (attention over the things seen,
plus a "nothing" object for text without pictures) and uses what it looks at to predict the next word.

CONDITIONS (four networks each; 4.9 M words of books, two passes; + 150 k pages per pass where used):
  books      books only
  text       books + the pages' sentences WITHOUT pictures       (do harder sentences alone help?)
  pictures   books + pages with pictures
  questions  books + pages with pictures; on up to 10 % of pages, those where the model is least sure which
             thing to look at when the verb comes, it ASKS "which one?" and is shown the one acting
             (its looking is corrected toward it)

TESTS: Stage 51's battery, TEXT ONLY (does what pictures taught carry over to reading?); agreement in the
built sentences split by whether the subject is a thing seen in pictures; and new pages (unseen drawing
style, unseen pairs of things) with and without the picture.

    python scripts/picture_pages.py run --cond pictures --seed 0
"""
import argparse, json, os, sys, time
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar as GR, grammar_memory as GM                                   # noqa: E402
DATA = "/home/claude/models/pages"
OUT = os.path.join(ROOT, "results", "picture_pages")
PASSES, PAGES_PER_PASS, Q_BUDGET = 2, 150_000, 0.10
ACTORS = {"girl": "girls", "boy": "boys", "woman": "women", "man": "men", "baby": "babies", "dog": "dogs", "cat": "cats", "horse": "horses",
          "hen": "hens", "chick": "chicks", "goat": "goats", "bird": "birds", "rabbit": "rabbits", "rat": "rats", "cow": "cows", "pig": "pigs", "duck": "ducks"}
PLACES = {"tree": "trees", "house": "houses", "chair": "chairs", "basket": "baskets", "flower": "flowers", "boat": "boats", "ball": "balls", "hat": "hats"}
ACTIONS = ["singing", "sleeping", "eating", "running", "playing", "jumping"]
PREPS = ["by", "near", "with", "behind", "beside"]
TEMPLATES = ["simple", "pp", "rel_obj", "rel_subj", "comp"]
DOBJ = 400 + 2 + len(ACTIONS) + 1


class PictureBank:
    def __init__(self):
        d = np.load(os.path.join(ROOT, "data", "pictures.npz"), allow_pickle=True)
        self.names = list(d["names"]); O = d["O"]; O = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-9)
        self.O, self.concept, self.style = O.astype(np.float32), d["concept"], d["style"]

    def look(self, name, rng, styles):
        c = self.names.index(name); idx = np.nonzero((self.concept == c) & np.isin(self.style, styles))[0]
        return self.O[int(rng.choice(idx))]


def make_page(rng, bank, styles, pairs=None):
    """a scene and one sentence about it. Returns words, objects (n, DOBJ), actor index, index of the verb"""
    while True:
        a, b = rng.choice(sorted(ACTORS), 2, replace=False)
        if pairs is None or (a, b) in pairs: break
    t = str(rng.choice(sorted(PLACES))); act = int(rng.integers(len(ACTIONS)))
    na, nb, nt = (int(rng.integers(2)) for _ in range(3))                    # 0 one, 1 several
    tmpl = str(rng.choice(TEMPLATES)); past = bool(rng.integers(2))
    A = ACTORS[a] if na else a; B = ACTORS[b] if nb else b; T = PLACES[t] if nt else t
    be = ("were" if na else "was") if past else ("are" if na else "is")
    see_b = ("saw" if past else ("see" if nb else "sees")); see_a = ("saw" if past else ("see" if na else "sees"))
    actw = ACTIONS[act]
    if tmpl == "simple":   w = ["the", A, be, actw, str(rng.choice(PREPS)), "the", T]
    elif tmpl == "pp":     w = ["the", A, str(rng.choice(PREPS)), "the", B, be, actw]
    elif tmpl == "rel_obj": w = ["the", A, "that", "the", B, see_b, be, actw]
    elif tmpl == "rel_subj": w = ["the", A, "that", see_a, "the", B, be, actw]
    else:                  w = ["the", B, see_b, "that", "the", A, be, actw]
    objs = []
    for name, n, acting in ((a, na, True), (b, nb, False), (t, nt, False)):
        v = np.zeros(DOBJ, np.float32); v[:400] = bank.look(name, rng, styles); v[400 + n] = 1.0
        v[402 + (act if acting else len(ACTIONS))] = 1.0; objs.append(v)
    order = rng.permutation(3); objs = np.stack(objs)[order]
    return dict(words=w, objs=objs, actor=int(np.nonzero(order == 0)[0][0]), verb=w.index(be), tmpl=tmpl, attractor=(na != nb), plural=bool(na), subj=a)


class SceneNet:
    def __init__(self, vocab, seed):
        import torch, torch.nn as nn
        torch.manual_seed(seed); torch.set_num_threads(int(os.environ.get("THREADS", "1")))
        self.torch = torch
        self.itos = ["<pad>", "<s>", "</s>", "<unk>"] + sorted(vocab); self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.vocab = set(vocab); V = len(self.itos)

        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.e = nn.Embedding(V, 128, padding_idx=0); s.cell = nn.GRUCell(128 + 64, 256); s.d = nn.Dropout(0.25)
                s.obj = nn.Linear(DOBJ, 64); s.null = nn.Parameter(torch.zeros(1, 1, 64)); s.q = nn.Linear(256, 64)
                s.o = nn.AdaptiveLogSoftmaxWithLoss(256 + 64, V, cutoffs=[2000, 10000] if V > 12000 else [V // 4, V // 2], div_value=4.0)

            def forward(s, x, objs, omask):
                """objs (B, n, DOBJ), omask (B, n) 1 = present; slot 0 is always 'nothing'"""
                B, T = x.shape; e = s.d(s.e(x))
                ov = torch.cat([s.null.expand(B, 1, 64), torch.tanh(s.obj(objs))], 1)          # (B, n+1, 64)
                m = torch.cat([torch.ones(B, 1), omask], 1)
                h = e.new_zeros(B, 256); r = e.new_zeros(B, 64); outs, atts = [], []
                for t in range(T):
                    h = s.cell(torch.cat([e[:, t], r], 1), h)
                    sc = (ov @ s.q(h).unsqueeze(-1)).squeeze(-1) / 8.0
                    a = torch.softmax(sc.masked_fill(m == 0, -1e9), 1); r = (a.unsqueeze(-1) * ov).sum(1)
                    outs.append(torch.cat([h, r], 1)); atts.append(a)
                return s.d(torch.stack(outs, 1)), torch.stack(atts, 1)               # atts (B, T, n+1)
        self.m = M(); self.opt = torch.optim.Adam(self.m.parameters(), lr=2e-3)

    def enc(self, w):
        return [1] + [self.stoi.get(x, 3) for x in w] + [2]

    def batch(self, items):
        torch = self.torch; seqs = [self.enc(it["words"]) for it in items]; L = max(map(len, seqs))
        x = np.zeros((len(items), L), np.int64)
        for k, q in enumerate(seqs): x[k, :len(q)] = q
        objs = np.zeros((len(items), 3, DOBJ), np.float32); om = np.zeros((len(items), 3), np.float32)
        for k, it in enumerate(items):
            if it.get("objs") is not None: objs[k] = it["objs"]; om[k] = 1.0
        return torch.from_numpy(x), torch.from_numpy(objs), torch.from_numpy(om)

    def step(self, items, questions):
        """one learning step; returns number of questions asked"""
        torch = self.torch; F = torch.nn.functional; self.m.train()
        x, objs, om = self.batch(items); out, att = self.m(x[:, :-1], objs, om); y = x[:, 1:]; keep = (y > 0).reshape(-1)
        loss = self.m.o(out.reshape(-1, out.shape[-1])[keep], y.reshape(-1)[keep]).loss; asked = 0
        if questions:
            pages = [k for k, it in enumerate(items) if it.get("objs") is not None]
            if pages:
                # looking at the time the verb is predicted: input word verb-1 is at time index verb
                a = torch.stack([att[k, items[k]["verb"], 1:] for k in pages])                  # (P, 3)
                ent = -(a.clamp(1e-6) * a.clamp(1e-6).log()).sum(1)
                n = int(np.floor(Q_BUDGET * len(pages) + np.random.default_rng(len(items)).random()))
                if n:
                    ask = torch.argsort(ent, descending=True)[:n]                                # least sure
                    tgt = torch.tensor([items[pages[int(i)]]["actor"] for i in ask])
                    loss = loss + F.nll_loss(a[ask].clamp(1e-6).log(), tgt); asked = n
        self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0); self.opt.step()
        return asked

    def token_logp_items(self, items, bs=128):
        torch = self.torch; self.m.eval(); out = [None] * len(items)
        idx = np.argsort([len(it["words"]) for it in items], kind="stable")
        with torch.no_grad():
            for k in range(0, len(idx), bs):
                g = idx[k:k + bs]; x, objs, om = self.batch([items[i] for i in g]); o, att = self.m(x[:, :-1], objs, om); y = x[:, 1:]
                keep = (y > 0).reshape(-1); full = torch.zeros(y.numel())
                full[keep] = self.m.o(o.reshape(-1, o.shape[-1])[keep], y.reshape(-1)[keep]).output
                full = full.reshape(y.shape).numpy(); att = att.numpy()
                for r, i in enumerate(g): out[i] = (full[r, :len(items[i]["words"]) + 1], att[r])
        return out

    # text-only interface for grammar_memory.evaluate
    def token_logp(self, sents):
        return [v[0] for v in self.token_logp_items([dict(words=s) for s in sents])]

    def logp(self, sents):
        return [float(v.sum()) for v in self.token_logp(sents)]


def pages_test(M, bank, rng, pairs, with_picture, n=3000):
    items = [make_page(rng, bank, styles=[3], pairs=pairs) for _ in range(n)]
    bad = [dict(it, words=it["words"][:it["verb"]] + [GM.FLIP[it["words"][it["verb"]]]] + it["words"][it["verb"] + 1:]) for it in items]
    if not with_picture:
        items = [dict(it, objs=None) for it in items]; bad = [dict(it, objs=None) for it in bad]
    g = M.token_logp_items(items); b = M.token_logp_items(bad)
    ok = np.array([g[k][0][it["verb"]] > b[k][0][it["verb"]] for k, it in enumerate(items)])
    look = np.array([g[k][1][it["verb"], 1 + it["actor"]] for k, it in enumerate(items)]) if with_picture else None
    res = {}
    for tm in TEMPLATES:
        for at in (False, True):
            m = np.array([it["tmpl"] == tm and it["attractor"] == at for it in items])
            if m.sum(): res[f"{tm}_{'attractor' if at else 'same'}"] = round(float(ok[m].mean()), 4)
    res["all"] = round(float(ok.mean()), 4)
    if look is not None: res["looks_at_actor_when_verb_comes"] = round(float(look.mean()), 3)
    return res


def run(a):
    os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    C = GR.corpus(); Ct = GM.corpus_big(C); tok = Counter(w for s in C["train"] for w in s)
    bank = PictureBank()
    allpairs = [(x, y) for x in sorted(ACTORS) for y in sorted(ACTORS) if x != y]
    rp = np.random.default_rng(5); held = set(map(tuple, np.array(allpairs)[rp.choice(len(allpairs), len(allpairs) // 5, replace=False)].tolist()))
    train_pairs = set(allpairs) - held
    page_words = set(ACTIONS) | set(PREPS) | set(ACTORS) | set(ACTORS.values()) | set(PLACES) | set(PLACES.values()) | {"the", "that", "see", "sees", "saw", "is", "are", "was", "were"}
    cnt = Counter(w for s in Ct["train"] for w in s)
    vocab = sorted({w for w, c in cnt.items() if c >= 5} | page_words)
    M = SceneNet(vocab, a.seed)
    path = os.path.join(DATA, f"{a.cond}_s{a.seed}.pt"); ck = path + ".ckpt"
    torch = M.torch
    st = dict(ep=0, b=0, log=[], asked=0)
    if os.path.exists(path) and os.path.exists(path + ".json"):
        M.m.load_state_dict(torch.load(path)); st = json.load(open(path + ".json"))
    else:
        if os.path.exists(ck):
            z = torch.load(ck, weights_only=False); M.m.load_state_dict(z["m"]); M.opt.load_state_dict(z["o"]); st = z["st"]
            print("resumed at pass", st["ep"] + 1, "batch", st["b"], flush=True)
        last = time.time()
        while st["ep"] < PASSES:
            rng = np.random.default_rng(1000 * a.seed + st["ep"]); t0 = time.time()
            items = [dict(words=s) for s in Ct["train"]]
            if a.cond != "books":
                pages = [make_page(rng, bank, styles=[0, 1, 2], pairs=train_pairs) for _ in range(PAGES_PER_PASS)]
                if a.cond == "text":
                    for p in pages: p["objs"] = None
                items += pages
            order = np.argsort([len(it["words"]) for it in items], kind="stable")
            groups = [order[i:i + 64] for i in range(0, len(order), 64)]; gorder = rng.permutation(len(groups))
            for bi in range(st["b"], len(gorder)):
                st["asked"] += M.step([items[i] for i in groups[gorder[bi]]], a.cond == "questions"); st["b"] = bi + 1
                if time.time() - last > 600:
                    torch.save(dict(m=M.m.state_dict(), o=M.opt.state_dict(), st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck); last = time.time()
            M.m.eval()
            val = -np.sum(M.logp(C["val"])) / sum(len(s) + 1 for s in C["val"])
            st["log"].append(dict(epoch=st["ep"] + 1, val=round(float(val), 3), minutes=round((time.time() - t0) / 60, 1), questions_so_far=st["asked"]))
            print(st["log"][-1], flush=True); st["ep"] += 1; st["b"] = 0
            for gp in M.opt.param_groups: gp["lr"] *= 0.7
            torch.save(dict(m=M.m.state_dict(), o=M.opt.state_dict(), st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck)
        torch.save(M.m.state_dict(), path); json.dump(st, open(path + ".json", "w")); os.remove(ck)
    res = dict(cond=a.cond, seed=a.seed, training=st["log"], questions_asked=st["asked"])
    rt = np.random.default_rng(77)
    res["pages_new_with_picture"] = pages_test(M, bank, rt, held, True)
    res["pages_new_text_only"] = pages_test(M, bank, np.random.default_rng(77), held, False)
    print("pages", res["pages_new_with_picture"], res["pages_new_text_only"], flush=True)
    res = GM.evaluate(M, C, tok, a.seed, res)
    E2 = GM.controlled(tok, {w for w, c in tok.items() if c >= 3})
    seen = set(ACTORS) | set(ACTORS.values())
    tg = M.token_logp([x["good"] for x in E2]); tb = M.token_logp([x["bad"] for x in E2])
    split = {}
    for name, sel in (("subject_seen_in_pictures", lambda x: x["good"][1] in seen), ("subject_not_in_pictures", lambda x: x["good"][1] not in seen)):
        for kind in ("pp", "rel"):
            for at in (False, True):
                ks = [k for k, x in enumerate(E2) if sel(x) and x["kind"] == kind and x["attractor"] == at]
                if not ks: continue
                i = [next(j for j, (p, q) in enumerate(zip(E2[k]["good"], E2[k]["bad"])) if p != q) for k in ks]
                ok = [tg[k][j] > tb[k][j] for k, j in zip(ks, i)]
                split[f"{name}_{kind}_{'attractor' if at else 'same'}"] = [round(float(np.mean(ok)), 4), len(ks)]
    res["E2_by_noun"] = split; print("E2 split", split, flush=True)
    json.dump(res, open(os.path.join(OUT, f"{a.cond}_s{a.seed}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--cond", choices=("books", "text", "pictures", "questions"), default="pictures"); ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
