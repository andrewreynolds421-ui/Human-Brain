#!/usr/bin/env python3
"""Stage 91: the reading network learns to read stories, not sentences.

Every lesson of VEGA's reading network (Stages 43-73) was ONE sentence: it starts each sentence with an empty state,
so nothing it knows spans sentences - yet since Stage 74 it is handed whole stories and the hippocampus is built on
its states. Meaning beyond the sentence (who is being talked about, what is going on) can only be learned from
connected text. Here the reading network keeps training ("slow cortical learning", a quarter of its old step size)
on the books it can already read, now in running order:

  --mode story    half the lessons are runs of consecutive sentences from one book (40-240 words), read as one
                  sequence: the next word is predicted from the story so far. The other half single sentences
                  (rehearsal, so sentence reading is kept).
  --mode single   CONTROL: the same number of lessons and words, single sentences only.

Books: the training split of the graded catalogue (Stage 53) that is on this machine; the last 5 % of every book is
held out, and any sentence sharing a 6-word run with the reading network's own test / validation sentences or with a
passage-test question is dropped. Passage-test and grade-test books are the catalogue's test split: never read.

Measured before and after: (1) how well a held-out sentence is predicted alone and after its preceding sentences
(does the story so far help?); (2) the passage test (Stage 86) with the reading network alone - sentence 21 scored
alone, and scored after reading the 20 sentences before it. Then (--then-events) Stage 89/90's hippocampus is trained
on the new reading network (STORY_READER) with the same budget as before.

    STEPS=6000 python scripts/story_reader.py --mode story --seed 0 --then-events "--sup 1 --roles --sem"
"""
import argparse, gzip, json, os, subprocess, sys, time, zlib
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar as GR, grammar_memory as GM, output_order as OO   # noqa: E402
OUT = os.environ.get("EV_OUT", os.path.join(ROOT, "results", "story_reader"))
DATA = os.environ.get("EV_DATA", "/home/claude/models/story_reader")
STEPS = int(os.environ.get("STEPS", 6000)); BS = 16; LR = 5e-4                    # the reader was trained at 2e-3 (step size ASSUMED)
PASSAGES = os.environ.get("PASSAGES", os.path.join(ROOT, "data", "passages.json.gz"))


def grams(s): return {zlib.crc32(" ".join(s[i:i + 6]).encode()) for i in range(len(s) - 5)}


def context_logp(M, contexts, sents, bs=64):
    """log-probability of each sentence read after its context (a list of words; [] = read alone)"""
    out = np.zeros(len(sents)); order = np.argsort([len(c) + len(s) for c, s in zip(contexts, sents)], kind="stable")
    for k in range(0, len(order), bs):
        g = order[k:k + bs]; tl = M.token_logp([list(contexts[i]) + list(sents[i]) for i in g])
        for i, v in zip(g, tl): out[i] = float(np.sum(v[len(contexts[i]):]))                 # the sentence's words and its end
    return out


def passage_scores(M, P, n=None):
    """passage test with the reading network alone: sentence 21 alone / after the 20 sentences before it"""
    res = {"alone": {}, "in_context": {}}
    for bd, its in P["test"].items():
        its = its[:n]; y = np.array([it["y"] for it in its]); S = [it["sent"][:it["pos"]] + [c[0]] + it["sent"][it["pos"] + 1:] for it in its for c in it["cands"]]
        C = [it["story"] for it in its for _ in it["cands"]]; nc = len(its[0]["cands"])
        res["alone"][bd] = round(float(np.mean(context_logp(M, [[]] * len(S), S).reshape(-1, nc).argmax(1) == y)), 4)
        res["in_context"][bd] = round(float(np.mean(context_logp(M, C, S).reshape(-1, nc).argmax(1) == y)), 4)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--mode", choices=("story", "single"), default="story")
    ap.add_argument("--then-events", default=""); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    tag = f"reader_{a.mode}_"; path = os.path.join(DATA, f"{tag}s{a.seed}.pt")

    def run(M, C, tok, seed, res):
        torch = M.torch; torch.set_num_threads(int(os.environ.get("THREADS", 2))); rng = np.random.default_rng(910 + a.seed)
        P = json.load(gzip.open(PASSAGES, "rt")); NT = int(os.environ.get("EV_NTEST", 0)) or None
        held = set()
        for x in C["test"] + C["val"]: held |= grams(x)
        for bd in P["test"]:
            for it in P["test"][bd]: held |= grams(it["sent"])
        cat = [c for c in json.load(open(os.path.join(ROOT, "data", "grades", "catalogue.json")))["books"] if c["split"] == "train" and os.path.isdir(c["dir"])]
        train, heldout = [], []
        for c in cat:
            ss = [s for s in GR.sentences(GR.book_text(c["dir"])) if not (grams(s) & held)]; k = max(30, len(ss) // 20)
            if len(ss) > 5 * k: train.append(ss[:-k]); heldout.append(ss[-k:])
        sizes = np.array([len(b) for b in train], float); pb = sizes / sizes.sum()
        info = dict(books=len(train), sentences=int(sizes.sum()), words=int(sum(len(s) for b in train for s in b)), mode=a.mode, steps=STEPS, lr=LR)
        print(info, flush=True)

        def run_of(books, r, target):                                                        # consecutive sentences of one book
            b = books[int(r.choice(len(books), p=pb if books is train else None))]; i = int(r.integers(len(b))); w = []
            while i < len(b) and len(w) + len(b[i]) <= target: w += b[i]; i += 1
            return w or b[min(i, len(b) - 1)][:target]

        def single(r, n_words):                                                              # single sentences, about n_words in all
            out, tot = [], 0
            while tot < n_words: b = train[int(r.choice(len(train), p=pb))]; s = b[int(r.integers(len(b)))]; out.append(s); tot += len(s)
            return out

        er = np.random.default_rng(91); ev = []                                              # held-out: a sentence and the sentences before it
        while len(ev) < (NT or 600):
            b = heldout[int(er.integers(len(heldout)))]; i = int(er.integers(5, len(b))); ctx = []; j = i - 1
            while j >= 0 and len(ctx) + len(b[j]) <= 150: ctx = b[j] + ctx; j -= 1
            if 5 <= len(b[i]) <= 40 and ctx: ev.append((ctx, b[i]))

        def measure():
            S = [s for _, s in ev]; nw = sum(len(s) + 1 for s in S)
            return dict(heldout_logp_per_word=dict(alone=round(float(context_logp(M, [[]] * len(S), S).sum() / nw), 4), in_context=round(float(context_logp(M, [c for c, _ in ev], S).sum() / nw), 4)),
                        passages_reader=passage_scores(M, P, NT))
        out = dict(seed=a.seed, **info, before=measure()); print("before", out["before"], flush=True)
        if os.path.exists(path): M.m.load_state_dict(torch.load(path), strict=False)
        else:
            for g in M.opt.param_groups: g["lr"] = LR
            t0 = time.time(); words = 0
            for step in range(STEPS):
                tgt = BS * int(rng.integers(40, 241))                                         # words in this lesson (same for both modes)
                if a.mode == "story" and step % 2 == 0: items = [dict(words=run_of(train, rng, tgt // BS)) for _ in range(BS)]
                else: items = [dict(words=s) for s in single(rng, tgt)]
                words += sum(len(i["words"]) for i in items); M.step(items)
                if (step + 1) % int(os.environ.get("LOG_EVERY", 500)) == 0: print(dict(step=step + 1, minutes=round((time.time() - t0) / 60, 1), words=words), flush=True)
            out["words_read"] = words; torch.save({k: v for k, v in M.m.state_dict().items()}, path)
        out["after"] = measure(); print("after", out["after"], flush=True)
        json.dump(out, open(os.path.join(OUT, f"{tag}s{a.seed}.json"), "w"), indent=1)
        raise SystemExit

    GM.evaluate = run; OO.DATA = "/home/claude/models/day_night/year4_cycle"; OO.OUT = "/tmp/hippo_scratch"; os.makedirs(OO.OUT, exist_ok=True)
    try: OO.run(argparse.Namespace(cond="books", seed=a.seed))
    except SystemExit: pass
    if a.then_events:                                                                        # the hippocampus on the new reading network
        env = dict(os.environ, STORY_READER=path, EV_TAG=os.environ.get("EV_TAG", "") + f"{a.mode}reader_", QUESTIONS=os.environ.get("EV_QUESTIONS", "160000"))
        raise SystemExit(subprocess.call([sys.executable, os.path.join(ROOT, "scripts", "hippo_events.py"), "--seed", str(a.seed)] + a.then_events.split(), env=env))


if __name__ == "__main__":
    main()
