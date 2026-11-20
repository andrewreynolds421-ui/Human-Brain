#!/usr/bin/env python3
"""Stage 65: the output layer with the words in the right order.

The adaptive output layer (Stages 53-64) gives the first 2,000 words a full-size read-out (320 units), the next
8,000 a quarter (80) and the rest a sixteenth (20). It assumes the words are ordered from most to least common.
They were in alphabetical order: "the", "that", "was", "who", "she", "you" sat in the 20-unit part, which carried
58 % of all running words with books and 74 % once speech was added (the larger vocabulary also moved "he",
"her", "herself", "himself" from the 80-unit to the 20-unit part - the reason names with reflexives fell to
chance in Stages 62-64).

Here the words are ordered by how often they occur in the reading (corpus counts + a sample of the pages), and two
earlier groups are re-run with nothing else changed (four networks each):
  books   books + pages                      (Stage 61's awake group)
  both    books + speech + pages, two markers, pages under both   (Stage 64)

    python scripts/output_order.py run --cond books --seed 0
"""
import argparse, os, pickle, sys
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import frames as FR, grammar_memory as GM, phenomena as PH, picture_check as PC, picture_pages as PP, sleep_grammar as SG, childes as CH, register as RG   # noqa: E402
DATA = "/home/claude/models/output_order"
OUT = os.path.join(ROOT, "results", "output_order")


def page_counts(n=20_000, per_pass=150_000):
    """word counts in a sample of the three kinds of teaching page, scaled to one pass"""
    rng = np.random.default_rng(0); tn, _ = PC.nouns(); look = PC.Look(); c = Counter()
    for _ in range(n):
        for it in (PC_page(rng, tn, look), PHEN(rng, tn, FR.REL), ADULT(rng, tn, FR.REL, SG.TRAIN_F, SG.TRAIN_M)): c.update(it["words"])
    return Counter({w: int(v * per_pass / n) for w, v in c.items()})


PC_page, PHEN, ADULT = FR.make_page, PH.phenomenon_page, SG.adult_page          # the unwrapped page makers


def ordered(corpus):
    """wrap a corpus function: when the reading is assembled, set the output order from its word counts"""
    def f(C_):
        Ct = corpus(C_); PP.ORDER = Counter(w for s in Ct["train"] for w in s) + page_counts(); return Ct
    return f


def run(a):
    if a.cond == "books":
        GM.corpus_big = ordered(GM.corpus_big); SG.DATA = os.path.join(DATA, "books"); SG.OUT = os.path.join(OUT, "books")
        os.makedirs(SG.DATA, exist_ok=True); os.makedirs(SG.OUT, exist_ok=True)
        SG.run(argparse.Namespace(cond="awake", seed=a.seed)); return
    speech = [RG.Talk(s) for s in pickle.load(open(os.path.join(CH.DATA, "adult_speech.pkl"), "rb"))]
    books = GM.corpus_big
    GM.corpus_big = ordered(lambda C_: dict(C_, train=books(C_)["train"] + speech))
    FR.make_page = RG.half_talk(FR.make_page); PH.phenomenon_page = RG.half_talk(PH.phenomenon_page); SG.adult_page = RG.half_talk(SG.adult_page)
    PH.Reader = RG.Tagged; SG.DATA = os.path.join(DATA, "both")
    for tag in ("book", "talk"):
        RG.Tagged.EVAL = tag; SG.OUT = os.path.join(OUT, "both_" + tag)
        os.makedirs(SG.DATA, exist_ok=True); os.makedirs(SG.OUT, exist_ok=True)
        if not os.path.exists(os.path.join(SG.OUT, f"awake_s{a.seed}.json")):
            SG.run(argparse.Namespace(cond="awake", seed=a.seed))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run",)); ap.add_argument("--cond", choices=("books", "both"), default="books")
    ap.add_argument("--seed", type=int, default=0); run(ap.parse_args())


if __name__ == "__main__":
    main()
