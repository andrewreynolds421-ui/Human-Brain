#!/usr/bin/env python3
"""Stage 70: days and nights as a routine.

Stage 69: one night of replaying stored experience moved it into the network; replaying old and new together
kept what was known. Here that becomes VEGA's routine. Start: Stage 65's *books* networks. New experience,
split over DAYS days: speech to children (CHILDES adult utterances, all but the held-out last 20,000) and the
training books of Stage 53's graded collection (different books from the grade test's), shuffled together.

  awake  each day: read that day's new material once (daytime step size); no nights
  cycle  each day the same; each night: replay that day's experience together with as many sentences drawn
         from older memory (the original books, teaching pages, earlier days), at a third of the step size
Equal-lessons controls (Stage 71): the same sentences and number of lessons as *cycle*, arranged differently
  interleave      one long day per day: today's material, the night's replay of it and the night's old memories
                  shuffled together, at the daytime step size; no separate night
  cycle_fullstep  as *cycle*, but the night at the daytime step size

The vocabulary does not grow: new words are read through their endings (Stage 58) and predicted as "unknown".
After each day and night: held-out speech and book text (log-prob per word) - a learning / forgetting curve.
At the end: the full test battery (BLiMP, built pairs, agreement, word order) and Stage 66's grade test.

Stage 73 (a grade 4-6 school year): --start day_night/cycle --new year4 starts from Stage 70's cycle networks and
gives each day new grade 4-6 books (scripts/year4_books.py; sentences sharing a 6-word run with the grade test
dropped); old memory for the nights then also holds Stage 70's experience. --new reread: the same routine, but
each day re-reads as many sentences of Stage 70's experience instead (control: practice without new books).

    python scripts/day_night.py --cond cycle --seed 0
    python scripts/day_night.py --cond cycle --start day_night/cycle --new year4 --tag year4 --seed 0
"""
import argparse, json, os, pickle, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import frames as FR, grammar_memory as GM, graded_reading as GD, output_order as OO, phenomena as PH, picture_check as PC, sleep_grammar as SG, childes as CH, segments as SM, grade_test as GT   # noqa: E402
OUT = os.path.join(ROOT, "results", "day_night")
DATA = "/home/claude/models/day_night"
DAYS, DAY_STEP, NIGHT_STEP, BS = 4, 1e-3, 5e-4, 64                 # day step: the next pass's size in Stages 61-65 (2e-3 * 0.7^2 ~ 1e-3); night a third of that (ASSUMED)


def year4_sentences(G):
    """sentences of the grade 4-6 books, without any sharing a 6-word run with the grade test's sentences (cached)"""
    cache = "/home/claude/models/year4/sentences.pkl"
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    import grammar as GR
    books = json.load(open(os.path.join(ROOT, "data", "grades", "year4.json")))["kept"]
    sh = lambda x: {tuple(x[i:i + 6]) for i in range(len(x) - 5)}
    held = set().union(*[sh(s) for b in GD.BANDS for s in G["S"]["test"][b]])
    out = [s for b in books for s in GR.sentences(GR.book_text(b["dir"])) if len(s) >= 2 and not (sh(s) & held)]
    os.makedirs(os.path.dirname(cache), exist_ok=True); pickle.dump(out, open(cache, "wb")); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cond", choices=("awake", "cycle", "interleave", "cycle_fullstep"), default="cycle"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default=""); ap.add_argument("--new", choices=("first", "year4", "reread", "world"), default="first"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); grp = (a.tag + "_" if a.tag else "") + a.cond; odir = os.path.join(OUT, grp); mdir = os.path.join(DATA, grp, "books"); os.makedirs(odir, exist_ok=True); os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, f"awake_s{a.seed}.pt"); real_eval, real_corpus = GM.evaluate, GM.corpus_big; seen = {}

    def corpus_seen(f):
        def g(C_):
            Ct = f(C_); seen["s"] = Ct["train"]; return Ct
        return g

    def live(M, C, tok, seed, res):
        torch = M.torch; rng = np.random.default_rng(7000 + a.seed)
        speech = pickle.load(open(os.path.join(CH.DATA, "adult_speech.pkl"), "rb")); held = speech[-20000:]; speech = speech[:-20000]
        G = pickle.load(open("/home/claude/models/graded/data.pkl", "rb")); graded = [s for b in GD.BANDS for s in G["S"]["train"][b]]
        new = speech + graded; new = [new[i] for i in rng.permutation(len(new)) if len(new[i]) >= 2]
        lived = []
        if a.new != "first":                                     # Stage 70's experience is now old memory
            lived = new; y4 = year4_sentences(G); graded = graded + y4
            if a.new == "world":                                 # Stage 74: the school year is old memory too; days read world stories
                import world as W
                lived = lived + y4; new = [W.training_story(rng) for _ in range(W.TRAIN_STORIES)]
            else:
                new = y4 if a.new == "year4" else [lived[i] for i in rng.choice(len(lived), len(y4), replace=False)]
            new = [new[i] for i in rng.permutation(len(new))]
        days = np.array_split(np.arange(len(new)), DAYS)
        tn, _ = PC.nouns(); look = PC.Look()
        pages = [FR.make_page(rng, tn, look)["words"] for _ in range(150_000)] + [PH.phenomenon_page(rng, tn, FR.REL)["words"] for _ in range(150_000)] + \
                [SG.adult_page(rng, tn, FR.REL, SG.TRAIN_F, SG.TRAIN_M)["words"] for _ in range(150_000)]
        old0 = seen["s"] + pages + lived
        probe = dict(speech=held[5000:6500], books=C["test"][:1500])

        def measure():
            return {d: round(float(SM.prepared(M, s)[0].mean()), 4) for d, s in probe.items()}

        def plan(d, night):
            """the day's or night's lessons: a fixed list of sentence groups (shuffled, similar lengths together)"""
            today = [new[i] for i in days[d]]

            def night_items():
                r = np.random.default_rng(100 * a.seed + 2 * d + 1)          # the same old memories in every group that uses them
                earlier = [new[i] for i in np.concatenate(days[:d])] if d else []
                pool_n = len(old0) + len(earlier); pick = r.choice(pool_n, len(today), replace=False)
                return today + [old0[i] if i < len(old0) else earlier[i - len(old0)] for i in pick]
            r = np.random.default_rng(100 * a.seed + 2 * d + night)
            if night: items = night_items()
            elif a.cond == "interleave": items = today + night_items()
            else: items = today
            o = np.argsort([len(s) for s in items], kind="stable"); groups = [o[i:i + BS] for i in range(0, len(o), BS)]
            return items, [groups[i] for i in r.permutation(len(groups))]

        ck = path + ".ckpt"; st = dict(d=0, night=0, b=0, curve=[])
        if os.path.exists(ck):
            z = torch.load(ck, weights_only=False); M.m.load_state_dict(z["m"]); M.opt.load_state_dict(z["o"]); st = z["st"]; print("resumed", {k: v for k, v in st.items() if k != "curve"}, flush=True)
        else:
            st["curve"].append(dict(when="start", **measure())); print(st["curve"][-1], flush=True)
        last = time.time()
        while st["d"] < DAYS:
            if st["night"] and a.cond in ("awake", "interleave"): st.update(d=st["d"] + 1, night=0, b=0); continue
            items, groups = plan(st["d"], st["night"])
            for gp in M.opt.param_groups: gp["lr"] = NIGHT_STEP if (st["night"] and a.cond == "cycle") else DAY_STEP
            for bi in range(st["b"], len(groups)):
                M.step([dict(words=items[j]) for j in groups[bi]]); st["b"] = bi + 1
                if time.time() - last > 600:
                    torch.save(dict(m=M.m.state_dict(), o=M.opt.state_dict(), st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck); last = time.time()
            M.m.eval(); st["curve"].append(dict(when=f"{'night' if st['night'] else 'day'} {st['d'] + 1}", lessons=len(groups), **measure())); print(st["curve"][-1], flush=True)
            if st["night"]: st.update(d=st["d"] + 1, night=0, b=0)
            else: st.update(night=1, b=0)
            torch.save(dict(m=M.m.state_dict(), o=M.opt.state_dict(), st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck); last = time.time()
        had = {tuple(s) for s in seen["s"]} | {tuple(s) for s in graded}
        grade = {}
        for b in GD.BANDS:
            its = [it for it in G["items"]["test"][b] if tuple(it["sent"]) not in had]; grade[b] = round(GT.cloze(M, its), 4)
        out = dict(cond=a.cond, start=a.start or "Stage 65 books", new=a.new, seed=a.seed, days=DAYS, new_sentences=len(new), new_words=int(sum(map(len, new))), curve=st["curve"], grade=grade,
                   grade_mean=round(float(np.mean(list(grade.values()))), 4))
        json.dump(out, open(os.path.join(odir, f"life_s{a.seed}.json"), "w"), indent=1); print(out, flush=True)
        torch.save(M.m.state_dict(), path); json.dump(dict(ep=2, b=0, log=[dict(life=a.cond)]), open(path + ".json", "w"))
        if os.path.exists(ck): os.remove(ck)
        raise SystemExit

    if not os.path.exists(path + ".json"):
        GM.evaluate = live; GM.corpus_big = corpus_seen(real_corpus); OO.OUT = "/tmp/day_night_scratch"
        if a.start: OO.DATA = os.path.join("/home/claude/models", a.start)
        try: OO.run(argparse.Namespace(cond="books", seed=a.seed))
        except SystemExit: pass
    GM.evaluate = real_eval; GM.corpus_big = real_corpus                 # the full battery on the network after its days (and nights)
    OO.DATA = os.path.join(DATA, grp); OO.OUT = odir
    OO.run(argparse.Namespace(cond="books", seed=a.seed))


if __name__ == "__main__":
    main()
