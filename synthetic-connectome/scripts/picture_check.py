#!/usr/bin/env python3
"""Stage 56: many nouns, and the picture as a check instead of a crutch.

Stage 55: with the picture in view the model looked instead of reading; without pictures, the pages'
sentences taught agreement only for the 17 nouns on the pages. Two changes, together:

MANY NOUNS. Pages are built from ~900 nouns of the books that have singular and plural forms
(`data/noun_pairs.json`); a quarter are HELD OUT - they never appear on a page, only in ordinary book
reading. If the model learns the rule, agreement should hold for them too.

PICTURE AS A CHECK. The model reads the page with NO picture in view. When the verb comes it GUESSES what
the picture shows - how many of the acting thing there are, and what it looks like - then sees the picture
and learns from the mismatch (as a child guesses, then is shown). Appearance: the object cells of a drawing
for the 25 things the picture system can draw; for other nouns a fixed random 400-number "look" per noun
(MODEL - a stand-in for how it looks).

GROUPS (four networks each; same books, passes and number of pages as Stage 55):
  text   books + many-noun pages, text only
  check  books + many-noun pages + the picture check
Stage 55's 17-noun text group is the reference.

TESTS (text only): built sentences in new frames ("the X that the Y helped is here", "the X of the Y are very
happy") with subjects (a) from the pages, (b) held out; Stage 51's battery.

    python scripts/picture_check.py run --cond check --seed 0
"""
import argparse, json, os, sys, time, zlib
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar as GR, grammar_memory as GM, picture_pages as PP                # noqa: E402
DATA = "/home/claude/models/check"
OUT = os.path.join(ROOT, "results", "picture_check")
STOP = {"time", "way", "kind", "sort", "matter", "people", "air", "water", "work", "part", "end", "side", "mind", "word", "thing", "moment", "ground", "light", "fire",
        "morning", "night", "day", "home", "world", "place", "name", "heart", "hand", "head", "face", "voice", "look", "turn", "means", "use", "rest", "minute", "hour", "year"}


def nouns():
    pairs = [tuple(p) for p in json.load(open(os.path.join(ROOT, "data", "noun_pairs.json"))) if p[0] not in STOP]
    pic = set(PP.ACTORS) | set(PP.PLACES)
    rng = np.random.default_rng(11); free = [p for p in pairs if p[0] not in pic]
    held = {free[i] for i in rng.choice(len(free), len(free) // 4, replace=False)}
    train = [p for p in pairs if p not in held]
    return train, sorted(held)


class Look:
    """what each thing looks like: drawings for the things the picture system knows, otherwise a fixed random look"""
    def __init__(self):
        self.bank = PP.PictureBank(); self.cache = {}

    def __call__(self, sing, rng):
        if sing in self.bank.names:
            return self.bank.look(sing, rng, [0, 1, 2])
        if sing not in self.cache:
            v = np.random.default_rng(zlib.crc32(sing.encode())).standard_normal(400).astype(np.float32); self.cache[sing] = v / np.linalg.norm(v)
        return self.cache[sing]


def make_page(rng, noun_list, look):
    i, j = rng.choice(len(noun_list), 2, replace=False); a, b = noun_list[i], noun_list[j]
    na, nb = int(rng.integers(2)), int(rng.integers(2)); A, B = a[na], b[nb]
    past = bool(rng.integers(2)); be = ("were" if na else "was") if past else ("are" if na else "is")
    see_b = "saw" if past else ("see" if nb else "sees"); see_a = "saw" if past else ("see" if na else "sees")
    act = str(rng.choice(PP.ACTIONS)); tmpl = str(rng.choice(PP.TEMPLATES)); prep = str(rng.choice(PP.PREPS))
    if tmpl == "simple":     w = ["the", A, be, act, prep, "the", B]
    elif tmpl == "pp":       w = ["the", A, prep, "the", B, be, act]
    elif tmpl == "rel_obj":  w = ["the", A, "that", "the", B, see_b, be, act]
    elif tmpl == "rel_subj": w = ["the", A, "that", see_a, "the", B, be, act]
    else:                    w = ["the", B, see_b, "that", "the", A, be, act]
    return dict(words=w, verb=w.index(be), number=na, look=look(a[0], rng), tmpl=tmpl, attractor=(na != nb))


def built_tests(train_nouns, held, rng, n=1500):
    """new frames (verbs and endings never on the pages)"""
    rel_v = ["helped", "loved", "liked", "met", "followed", "found", "knew", "called"]
    ends = [["here"], ["very", "happy"], ["quite", "still"], ["there"], ["afraid"], ["in", "the", "garden"]]
    allp = train_nouns + held; out = {}
    for name, subj in (("page_nouns", train_nouns), ("held_out_nouns", held)):
        items = []
        for _ in range(n):
            a = subj[rng.integers(len(subj))]; b = allp[rng.integers(len(allp))]
            if a == b: continue
            kind = str(rng.choice(["pp", "rel"])); sp, bp = int(rng.integers(2)), int(rng.integers(2))
            v = ("were" if sp else "was") if rng.integers(2) else ("are" if sp else "is")
            mid = ["of", "the", b[bp]] if kind == "pp" else ["that", "the", b[bp], str(rng.choice(rel_v))]
            good = ["the", a[sp]] + mid + [v] + list(ends[rng.integers(len(ends))]); bad = list(good); bad[2 + len(mid)] = GM.FLIP[v]
            items.append(dict(good=good, bad=bad, kind=kind, attractor=(sp != bp), verb=2 + len(mid)))
        out[name] = items
    return out


class CheckNet(PP.SceneNet):
    """the Stage 55 reader with the scene always empty, plus a guess at the verb: number (2) and look (400)"""
    def __init__(self, vocab, seed, check):
        super().__init__(vocab, seed); torch = self.torch; nn = torch.nn
        self.check = check; self.num = nn.Linear(256, 2); self.look = nn.Linear(256, 400)
        self.opt = torch.optim.Adam(list(self.m.parameters()) + list(self.num.parameters()) + list(self.look.parameters()), lr=2e-3)

    def step(self, items, questions=False):
        torch = self.torch; F = torch.nn.functional; self.m.train()
        x, objs, om = self.batch([dict(words=it["words"]) for it in items])
        out, _ = self.m(x[:, :-1], objs, om); y = x[:, 1:]; keep = (y > 0).reshape(-1)
        loss = self.m.o(out.reshape(-1, out.shape[-1])[keep], y.reshape(-1)[keep]).loss
        pages = [k for k, it in enumerate(items) if "number" in it]
        if self.check and pages:
            h = torch.stack([out[k, items[k]["verb"], :256] for k in pages])           # the reading state as the verb comes
            num = torch.tensor([items[k]["number"] for k in pages]); lk = torch.from_numpy(np.stack([items[k]["look"] for k in pages]))
            loss = loss + F.cross_entropy(self.num(h), num) + (1 - F.cosine_similarity(self.look(h), lk)).mean()
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.m.parameters()) + list(self.num.parameters()) + list(self.look.parameters()), 1.0); self.opt.step()
        return 0


def run(a):
    os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    C = GR.corpus(); Ct = GM.corpus_big(C); tok = Counter(w for s in C["train"] for w in s)
    train_nouns, held = nouns(); look = Look()
    page_words = {w for p in train_nouns for w in p} | set(PP.ACTIONS) | set(PP.PREPS) | {"the", "that", "see", "sees", "saw", "is", "are", "was", "were"}
    cnt = Counter(w for s in Ct["train"] for w in s)
    M = CheckNet(sorted({w for w, c in cnt.items() if c >= 5} | page_words), a.seed, a.cond == "check")
    torch = M.torch; path = os.path.join(DATA, f"{a.cond}_s{a.seed}.pt"); ck = path + ".ckpt"
    params = lambda: dict(m=M.m.state_dict(), n=M.num.state_dict(), l=M.look.state_dict())
    st = dict(ep=0, b=0, log=[])
    if os.path.exists(path) and os.path.exists(path + ".json"):
        z = torch.load(path); M.m.load_state_dict(z["m"]); M.num.load_state_dict(z["n"]); M.look.load_state_dict(z["l"]); st = json.load(open(path + ".json"))
    else:
        if os.path.exists(ck):
            z = torch.load(ck, weights_only=False); M.m.load_state_dict(z["p"]["m"]); M.num.load_state_dict(z["p"]["n"]); M.look.load_state_dict(z["p"]["l"])
            M.opt.load_state_dict(z["o"]); st = z["st"]; print("resumed at pass", st["ep"] + 1, "batch", st["b"], flush=True)
        last = time.time()
        while st["ep"] < PP.PASSES:
            rng = np.random.default_rng(1000 * a.seed + st["ep"]); t0 = time.time()
            items = [dict(words=s) for s in Ct["train"]] + [make_page(rng, train_nouns, look) for _ in range(PP.PAGES_PER_PASS)]
            order = np.argsort([len(it["words"]) for it in items], kind="stable")
            groups = [order[i:i + 64] for i in range(0, len(order), 64)]; gorder = rng.permutation(len(groups))
            for bi in range(st["b"], len(gorder)):
                M.step([items[i] for i in groups[gorder[bi]]]); st["b"] = bi + 1
                if time.time() - last > 600:
                    torch.save(dict(p=params(), o=M.opt.state_dict(), st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck); last = time.time()
            M.m.eval()
            val = -np.sum(M.logp(C["val"])) / sum(len(s) + 1 for s in C["val"])
            st["log"].append(dict(epoch=st["ep"] + 1, val=round(float(val), 3), minutes=round((time.time() - t0) / 60, 1)))
            print(st["log"][-1], flush=True); st["ep"] += 1; st["b"] = 0
            for gp in M.opt.param_groups: gp["lr"] *= 0.7
            torch.save(dict(p=params(), o=M.opt.state_dict(), st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck)
        torch.save(params(), path); json.dump(st, open(path + ".json", "w")); os.remove(ck)
    res = dict(cond=a.cond, seed=a.seed, training=st["log"], nouns=dict(on_pages=len(train_nouns), held_out=len(held)))
    T = built_tests(train_nouns, held, np.random.default_rng(99)); bt = {}
    for name, items in T.items():
        tg = M.token_logp([x["good"] for x in items]); tb = M.token_logp([x["bad"] for x in items])
        ok = np.array([tg[k][x["verb"]] > tb[k][x["verb"]] for k, x in enumerate(items)])
        sp = np.array([x["good"][x["verb"]] in ("are", "were") for x in items])
        for kind in ("pp", "rel"):
            for at in (False, True):
                m = np.array([x["kind"] == kind and x["attractor"] == at for x in items])
                bt[f"{name}_{kind}_{'attractor' if at else 'same'}"] = dict(right=round(float(ok[m].mean()), 4), plural_right=round(float(ok[m & sp].mean()), 4),
                                                                          singular_right=round(float(ok[m & ~sp].mean()), 4), n=int(m.sum()))
    res["built"] = bt; print("built", {k: v["right"] for k, v in bt.items()}, flush=True)
    # the guess itself: number and look of the acting thing on new pages (held-out nouns as subjects)
    if a.cond == "check":
        rng = np.random.default_rng(5); pg = [make_page(rng, held, look) for _ in range(2000)]
        x, objs, om = M.batch([dict(words=p["words"]) for p in pg])
        with torch.no_grad():
            M.m.eval(); out, _ = M.m(x[:, :-1], objs, om); h = torch.stack([out[k, p["verb"], :256] for k, p in enumerate(pg)])
            nok = (M.num(h).argmax(1).numpy() == np.array([p["number"] for p in pg]))
        res["guess_number_held_out_nouns"] = {f"{t}_{'attractor' if at else 'same'}": round(float(nok[[p["tmpl"] == t and p["attractor"] == at for p in pg]].mean()), 4)
                                              for t in PP.TEMPLATES for at in (False, True)}
        print("guess", res["guess_number_held_out_nouns"], flush=True)
    res = GM.evaluate(M, C, tok, a.seed, res)
    json.dump(res, open(os.path.join(OUT, f"{a.cond}_s{a.seed}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--cond", choices=("text", "check"), default="check"); ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
