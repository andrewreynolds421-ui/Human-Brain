#!/usr/bin/env python3
"""Stage 61: adult constructions, and sleep.

1. ADULT CONSTRUCTIONS. Stage 60's rules carried to new words but not to constructions never met; BLiMP uses
   auxiliaries, plain present verbs, who / which, names, bare plurals and quantifiers, contractions, auxiliary
   questions. New pages (grammatical only) in those constructions:
     aux agreement   the customer who visited the children has worn ... / the boys that saw the dog have ...
     verb agreement  boys that saw the dog suffer / the boy near the trees suffers
     negation        isn't / aren't / wasn't / weren't / hasn't / haven't / doesn't / don't
     who gaps        a lady remembered who the actors hid / knew what the boys hadn't seen
     names           Mary can't help herself / John saw himself / Mary has seen the dogs
     quantifiers     all / most / some / many / few + plural; every / each + singular; bare plurals
     questions       had the boy ever seen the dog? / do some dogs ever sing? / can Mary ever ...? (any, ever)
   A quarter of the names, and helped / loved / met / called, are held out.

2. SLEEP. Stage 60 also lost some of what it knew (irregular forms 84 -> 78 %, reflexive gender 69 -> 59 %) -
   new learning overwriting old, the interference sleep is thought to prevent by replaying recent and older
   memories interleaved, offline (complementary learning systems: McClelland, McNaughton & O'Reilly 1995).
     awake    read straight through (two passes)
     sleep    each pass is four "days"; after each day a NIGHT: no new input, replay of sentences sampled from
              everything read so far this pass (old and new mixed), a quarter as many lessons as the day, at a
              third of the step size
     reread   CONTROL: the same number of extra lessons, from that day's sentences only, at the day's step size
Books + Stage 57-58 agreement pages + Stage 60 phenomenon pages + adult pages (150 k each per pass).
Four networks each. Tests: BLiMP by field and sub-set; Stage 60's built pairs; new built pairs with held-out
verbs and names.

    python scripts/sleep_grammar.py run --cond sleep --seed 0
"""
import argparse, json, os, sys, time
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar as GR, grammar_memory as GM, picture_check as PC, frames as FR, phenomena as PH   # noqa: E402
DATA = "/home/claude/models/sleep_grammar"
OUT = os.path.join(ROOT, "results", "sleep_grammar")
DAYS, NIGHT_SHARE, NIGHT_STEP = 4, 0.25, 1 / 3
FNAMES = ["mary", "anna", "emma", "lucy", "jane", "sarah", "alice", "grace", "rose", "kate", "susan", "helen", "martha", "nancy", "betty", "ruth", "laura", "julia",
          "clara", "ellen", "katherine", "rebecca", "danielle", "natalie", "karla", "marie", "elizabeth", "margaret", "dorothy", "emily", "amy", "lisa", "diana", "sandra"]
MNAMES = ["john", "tom", "james", "peter", "george", "henry", "jack", "william", "charles", "robert", "david", "paul", "frank", "edward", "joseph", "harry", "fred", "sam",
          "bruce", "patrick", "michael", "richard", "thomas", "daniel", "mark", "steven", "kenneth", "carl", "dennis", "gary", "brian", "eric", "roger", "walter"]
rng0 = np.random.default_rng(21)
HELD_F = set(rng0.choice(FNAMES, len(FNAMES) // 4, replace=False)); HELD_M = set(rng0.choice(MNAMES, len(MNAMES) // 4, replace=False))
TRAIN_F = [n for n in FNAMES if n not in HELD_F]; TRAIN_M = [n for n in MNAMES if n not in HELD_M]
BASEV = {"saw": "see", "heard": "hear", "watched": "watch", "chased": "chase", "fed": "feed", "carried": "carry", "caught": "catch", "kept": "keep", "left": "leave",
         "taught": "teach", "told": "tell", "visited": "visit", "liked": "like", "found": "find", "followed": "follow", "knew": "know", "asked": "ask", "thanked": "thank",
         "pushed": "push", "pulled": "pull", "wanted": "want", "missed": "miss", "praised": "praise", "feared": "fear", "trusted": "trust", "warned": "warn", "kissed": "kiss",
         "greeted": "greet", "noticed": "notice", "remembered": "remember", "served": "serve", "guarded": "guard", "led": "lead", "named": "name", "dressed": "dress",
         "helped": "help", "loved": "love", "met": "meet", "called": "call"}
PART = dict({v: v for v in BASEV}, saw="seen", knew="known", **{})
INTR = ["sing", "sleep", "suffer", "wait", "laugh", "cry", "run", "hide", "work", "play", "fall", "smile", "listen", "rest", "speak", "win"]
QSING = ["every", "each", "this", "that"]; QPL = ["all", "most", "some", "many", "few", "these", "those", "the", "the", ""]


def s3(v):
    if v.endswith(("s", "sh", "ch", "x", "o")): return v + "es"
    if v.endswith("y") and v[-2] not in "aeiou": return v[:-1] + "ies"
    return v + "s"


def ing(v):
    if v in ("run", "win"): return v + v[-1] + "ing"
    return (v[:-1] if v.endswith("e") and not v.endswith("ee") else v) + "ing"


def adult_page(rng, nouns, verbs, fnames, mnames):
    N = lambda: nouns[rng.integers(len(nouns))]; V = lambda: str(rng.choice(verbs)); I = lambda: str(rng.choice(INTR))
    pl = int(rng.integers(2)); a, b, c = N(), N(), N()
    subj = ([str(rng.choice(QPL))] if pl else [str(rng.choice(QSING + ["the", "the", "a"]))]); subj = [x for x in subj if x] + [a[pl]]
    rel = lambda: ([str(rng.choice(["who", "that", "which"])), V(), str(rng.choice(["the", "some", "all", "most"])), b[1]] if rng.integers(2)
                   else [str(rng.choice(["who", "that", "which"])), "the", b[int(rng.integers(2))], V()])
    mid = rel() if rng.integers(2) else ([str(rng.choice(FR.PREPS)), "the", b[int(rng.integers(2))]] if rng.integers(2) else [])
    k = int(rng.integers(7)); v = V()
    if k == 0:   w = subj + mid + [("have" if pl else "has") + str(rng.choice(["", "n't"])), PART[v], "the", c[int(rng.integers(2))]]
    elif k == 1: w = subj + mid + [I() if pl else s3(I())]
    elif k == 2:
        neg = str(rng.choice(["aren't", "weren't", "don't"] if pl else ["isn't", "wasn't", "doesn't"]))
        w = subj + mid + [neg] + ([I()] if neg in ("don't", "doesn't") else ([ing(I())] if rng.integers(2) else ["here"]))
    elif k == 3:
        who = str(rng.choice(["who", "what"])); aux = str(rng.choice(["", "had", "hadn't", "didn't"]))
        w = ["the", a[pl], str(rng.choice(["remembered", "knew", "forgot", "learned", "wondered"])), who, "the", b[int(rng.integers(2))]] + \
            ([aux, PART[v] if aux.startswith("had") else BASEV[v]] if aux else [v])
    elif k == 4:
        fem = bool(rng.integers(2)); n = str(rng.choice(fnames if fem else mnames))
        modal = str(rng.choice(["", "can't", "couldn't", "won't", "could", "will", "might"]))
        w = [n] + ([modal, BASEV[v]] if modal else [v]) + ["herself" if fem else "himself"] if rng.integers(2) else \
            [n, str(rng.choice(["has", "had", "hasn't"])), PART[v], "the", c[int(rng.integers(2))]]
    elif k == 5:
        aux = str(rng.choice(["had", "has", "did", "can", "will", "could"])) if not pl else str(rng.choice(["had", "have", "did", "do", "can", "will"]))
        if aux in ("do", "did") and not pl: aux = "does" if aux == "do" else "did"
        verb = PART[v] if aux in ("had", "has", "have") else BASEV[v]
        w = [aux] + subj + ["ever", verb, str(rng.choice(["any", "the"])), b[1]] if rng.integers(2) else [aux] + subj + [verb, "the", b[int(rng.integers(2))]]
    else:
        w = subj + [str(rng.choice(["had", "has", "have"] if pl else ["had", "has"])) if True else "", PART[v], "some", b[1]]
        if pl and w[len(subj)] == "has": w[len(subj)] = "have"
        if not pl and w[len(subj)] == "have": w[len(subj)] = "has"
    return dict(words=[x for x in w if x], kind="adult")


def adult_tests(nouns, rng, n=400):
    """good / bad pairs with held-out verbs (helped / loved / met / called) and held-out names"""
    out = {k: [] for k in ("aux_agree_relative", "verb_agree_relative", "who_gap", "name_reflexive", "question_ever")}
    hf, hm = sorted(HELD_F), sorted(HELD_M)
    for _ in range(n):
        a, b = nouns[rng.integers(len(nouns))], nouns[rng.integers(len(nouns))]; pl = int(rng.integers(2)); v = str(rng.choice(FR.HELD_REL)); base = BASEV[v]
        rel = ["who", v, "most", b[1]]
        out["aux_agree_relative"].append((["this" if not pl else "these"][:0] + ["the", a[pl]] + rel + ["have" if pl else "has", "seen", "the", "dog"],
                                          ["the", a[pl]] + rel + ["has" if pl else "have", "seen", "the", "dog"]))
        i = str(rng.choice(INTR))
        out["verb_agree_relative"].append(([a[1], "that", "aren't", ing(base), "the", b[0], i], [a[1], "that", "aren't", ing(base), "the", b[0], s3(i)]))
        out["who_gap"].append((["a", a[0], "has", "remembered", "who", "the", b[1], base], ["a", a[0], "has", "remembered", "that", "the", b[1], base]))
        fem = bool(rng.integers(2)); nm = str(rng.choice(hf if fem else hm))
        out["name_reflexive"].append(([nm, "can't", base, "herself" if fem else "himself"], [nm, "can't", base, "himself" if fem else "herself"]))
        out["question_ever"].append((["had", nm, "ever", PART[v]], [nm, "had", "ever", PART[v]]))
    return out


def run(a):
    os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    C = GR.corpus(); Ct = GM.corpus_big(C); tok = Counter(w for s in C["train"] for w in s)
    train_nouns, held = PC.nouns(); look = PC.Look(); verbs = FR.REL
    words = {w for p in train_nouns for w in p} | set(FR.PREPS) | set(FR.HELD_PREPS) | set(FR.REL) | set(FR.HELD_REL) | {w for p in FR.PRED + FR.HELD_PRED for w in p} | \
            set(PH.ADJ) | set(PH.HELD_ADJ) | {w for s in PH.SAY + PH.ASK for w in s.split()} | set(BASEV.values()) | set(PART.values()) | set(INTR) | {s3(i) for i in INTR} | {ing(i) for i in INTR} | {ing(b) for b in BASEV.values()} | \
            set(TRAIN_F) | set(TRAIN_M) | set(QSING) | {q for q in QPL if q} | {"the", "a", "that", "who", "which", "what", "see", "sees", "saw", "is", "are", "was", "were", "this",
            "these", "those", "himself", "herself", "itself", "themselves", "any", "ever", "some", "no", "only", "never", "didn't", "did", "not", "sing", "like", "really", "dogs",
            "dog", "has", "have", "had", "hasn't", "haven't", "hadn't", "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't", "does", "do", "can", "can't", "could", "couldn't",
            "will", "won't", "might", "here", "remembered", "forgot", "learned", "wondered", "seen", "known"}
    cnt = Counter(w for s in Ct["train"] for w in s)
    M = PH.Reader(sorted({w for w, c in cnt.items() if c >= 5} | words), a.seed)
    torch = M.torch; path = os.path.join(DATA, f"{a.cond}_s{a.seed}.pt"); ck = path + ".ckpt"
    st = dict(ep=0, b=0, log=[])
    if os.path.exists(path) and os.path.exists(path + ".json"):
        M.m.load_state_dict(torch.load(path)); st = json.load(open(path + ".json"))
    else:
        if os.path.exists(ck):
            z = torch.load(ck, weights_only=False); M.m.load_state_dict(z["m"]); M.opt.load_state_dict(z["o"]); st = z["st"]; print("resumed at pass", st["ep"] + 1, "lesson", st["b"], flush=True)
        last = time.time()
        while st["ep"] < 2:
            rng = np.random.default_rng(1000 * a.seed + st["ep"]); t0 = time.time()
            items = [dict(words=s) for s in Ct["train"]] + [dict(words=FR.make_page(rng, train_nouns, look)["words"]) for _ in range(150_000)] + \
                    [PH.phenomenon_page(rng, train_nouns, verbs) for _ in range(150_000)] + [adult_page(rng, train_nouns, verbs, TRAIN_F, TRAIN_M) for _ in range(150_000)]
            order = np.argsort([len(it["words"]) for it in items], kind="stable")
            groups = [order[i:i + 64] for i in range(0, len(order), 64)]; gorder = list(rng.permutation(len(groups)))
            # the pass as a fixed schedule of (lesson, step scale): days, and after each day a night (sleep) or extra re-reading (reread)
            sched = []; days = np.array_split(np.arange(len(gorder)), DAYS)
            for d, day in enumerate(days):
                sched += [(gorder[i], 1.0) for i in day]
                n_extra = int(NIGHT_SHARE * len(day))
                if a.cond == "sleep":
                    seen = np.concatenate(days[:d + 1]); sched += [(gorder[int(i)], NIGHT_STEP) for i in rng.choice(seen, n_extra)]
                elif a.cond == "reread":
                    sched += [(gorder[int(i)], 1.0) for i in rng.choice(day, n_extra)]
            base_lr = 2e-3 * 0.7 ** st["ep"]
            for bi in range(st["b"], len(sched)):
                g, scale = sched[bi]
                for gp in M.opt.param_groups: gp["lr"] = base_lr * scale
                M.step([items[i] for i in groups[g]]); st["b"] = bi + 1
                if time.time() - last > 600:
                    torch.save(dict(m=M.m.state_dict(), o=M.opt.state_dict(), st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck); last = time.time()
            M.m.eval(); val = -np.sum(M.logp(C["val"])) / sum(len(s) + 1 for s in C["val"])
            st["log"].append(dict(epoch=st["ep"] + 1, val=round(float(val), 3), lessons=len(sched), minutes=round((time.time() - t0) / 60, 1)))
            print(st["log"][-1], flush=True); st["ep"] += 1; st["b"] = 0
            torch.save(dict(m=M.m.state_dict(), o=M.opt.state_dict(), st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck)
        torch.save(M.m.state_dict(), path); json.dump(st, open(path + ".json", "w")); os.remove(ck)
    res = dict(cond=a.cond, seed=a.seed, training=st["log"]); bt = {}
    for name, nl in (("page_nouns", train_nouns), ("held_out_nouns", held)):
        for k, prs in list(PH.built_tests(nl, FR.HELD_REL, PH.HELD_ADJ, np.random.default_rng(42)).items()) + list(adult_tests(nl, np.random.default_rng(43)).items()):
            if not prs: continue
            g = M.logp([p[0] for p in prs]); b = M.logp([p[1] for p in prs]); bt[f"{name}_{k}"] = round(float(np.mean(np.array(g) > np.array(b))), 4)
    res["built"] = bt; print("built", bt, flush=True)
    res = GM.evaluate(M, C, tok, a.seed, res)
    json.dump(res, open(os.path.join(OUT, f"{a.cond}_s{a.seed}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--cond", choices=("awake", "sleep", "reread"), default="sleep"); ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
