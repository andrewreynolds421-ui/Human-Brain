#!/usr/bin/env python3
"""Stage 69: sleep - replaying a memory segment into the network.

Stage 68: a segment of speech the network never trained on makes it predict speech better while the segment is
active, but gives no skill (how questions are built stays unknown), and costs memory and time per word. In
sleep the stored experience is replayed with no new input, at a small step size, so that the weights take it
over (the hippocampus teaching the cortex).

A segment holds, in order, the word that followed every moment - so the sentences can be re-read from it (words
outside the network's vocabulary come back as "unknown"; the vocabulary does not grow in sleep).

Start: Stage 65's *books* networks (books + teaching pages; never trained on speech), with Stage 68's segments.
Groups (four networks each), one night:
  speech  replay the speech segment only
  mixed   replay the speech segment, the books segment and teaching pages, shuffled together (pages are
          generated again, standing in for a pages segment)
Before / after: held-out speech and book text (log-probability per word), with no segment and with the
segments written BEFORE sleep (are old memories still readable by the changed network?); then the full test
battery (BLiMP, built tests, agreement, word order), to compare with Stage 65's books group (no sleep) and its
books + speech group (trained on the mix from the start).

    python scripts/sleep_replay.py --cond mixed --seed 0
"""
import argparse, json, os, pickle, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import frames as FR, grammar_memory as GM, output_order as OO, phenomena as PH, picture_check as PC, sleep_grammar as SG, childes as CH, segments as SM   # noqa: E402
OUT = os.path.join(ROOT, "results", "sleep_replay")
DATA = "/home/claude/models/sleep_replay"
NIGHT_STEP = 5e-4                                               # about a third of the step size at the end of training (as in Stage 61); ASSUMED


def sentences_of(folder, itos):
    """the sentences a segment holds, re-read from its record of next words (id 2 ends a sentence)"""
    v = np.load(os.path.join(folder, "next_word.npy")); ends = np.flatnonzero(v == 2); starts = np.concatenate([[0], ends[:-1] + 1])
    return lambda i: [itos[int(x)] for x in v[starts[i]:ends[i]]], ends - starts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cond", choices=("speech", "mixed"), default="mixed"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); odir = os.path.join(OUT, a.cond); mdir = os.path.join(DATA, a.cond, "books"); os.makedirs(odir, exist_ok=True); os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, f"awake_s{a.seed}.pt"); real_eval = GM.evaluate; real_corpus = GM.corpus_big

    def night(M, C, tok, seed, res):
        torch = M.torch; home = os.path.join(SM.DATA, f"vega_books_s{a.seed}")
        segs = {k: SM.Segment(os.path.join(home, k)) for k in ("books", "speech")}
        held = pickle.load(open(os.path.join(CH.DATA, "adult_speech.pkl"), "rb"))[-20000:]
        texts = dict(books=C["test"][:3000], speech=held[5000:8000]); par = dict(books=0.2, speech=0.4)

        def measure():
            o = {}
            for d, sents in texts.items():
                lp, H, y, _ = SM.prepared(M, sents); med = segs[d].median_distance(H[:5000])
                o[d] = dict(none=round(float(lp.mean()), 4), old_segment=round(float(SM.mixed(lp, [(par[d], segs[d].share(H, y, med))]).mean()), 4))
            return o

        ck = path + ".ckpt"; st = dict(b=0, before=None)
        rng = np.random.default_rng(5000 + a.seed)
        src = [sentences_of(os.path.join(home, "speech"), M.itos)]
        if a.cond == "mixed":
            src.append(sentences_of(os.path.join(home, "books"), M.itos))
            tn, _ = PC.nouns(); look = PC.Look()
            pages = [FR.make_page(rng, tn, look)["words"] for _ in range(150_000)] + [PH.phenomenon_page(rng, tn, FR.REL)["words"] for _ in range(150_000)] + \
                    [SG.adult_page(rng, tn, FR.REL, SG.TRAIN_F, SG.TRAIN_M)["words"] for _ in range(150_000)]
            src.append((lambda i: pages[i], np.array([len(p) for p in pages])))
        which = np.concatenate([np.full(len(n), k) for k, (_, n) in enumerate(src)]); idx = np.concatenate([np.arange(len(n)) for _, n in src]); L = np.concatenate([n for _, n in src])
        keep = L >= 2; which, idx, L = which[keep], idx[keep], L[keep]
        order = np.argsort(L, kind="stable"); groups = [order[i:i + 64] for i in range(0, len(order), 64)]; gorder = rng.permutation(len(groups))
        if os.path.exists(ck):
            z = torch.load(ck, weights_only=False); M.m.load_state_dict(z["m"]); M.opt.load_state_dict(z["o"]); st = z["st"]; print("resumed at lesson", st["b"], flush=True)
        else:
            st["before"] = measure(); print("before", st["before"], flush=True)
        for gp in M.opt.param_groups: gp["lr"] = NIGHT_STEP
        last = time.time(); t0 = time.time()
        for bi in range(st["b"], len(gorder)):
            M.step([dict(words=src[which[j]][0](int(idx[j]))) for j in groups[gorder[bi]]]); st["b"] = bi + 1
            if time.time() - last > 600:
                torch.save(dict(m=M.m.state_dict(), o=M.opt.state_dict(), st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck); last = time.time()
        M.m.eval(); after = measure(); print("after", after, flush=True)
        out = dict(cond=a.cond, seed=a.seed, lessons=len(gorder), sentences_replayed={k: int((which == i).sum()) for i, k in enumerate(("speech", "books", "pages")[:len(src)])},
                   minutes_this_run=round((time.time() - t0) / 60, 1), before=st["before"], after=after)
        json.dump(out, open(os.path.join(odir, f"replay_s{a.seed}.json"), "w"), indent=1)
        torch.save(M.m.state_dict(), path); json.dump(dict(ep=2, b=0, log=[dict(night=a.cond, lessons=len(gorder))]), open(path + ".json", "w"))
        if os.path.exists(ck): os.remove(ck)
        raise SystemExit

    if not os.path.exists(path + ".json"):                      # the night: load Stage 65's books network, replay, save
        GM.evaluate = night; OO.OUT = "/tmp/sleep_replay_scratch"
        try: OO.run(argparse.Namespace(cond="books", seed=a.seed))
        except SystemExit: pass
    GM.evaluate = real_eval; GM.corpus_big = real_corpus        # the morning: the full test battery on the slept network
    OO.DATA = os.path.join(DATA, a.cond); OO.OUT = odir
    OO.run(argparse.Namespace(cond="books", seed=a.seed))


if __name__ == "__main__":
    main()
