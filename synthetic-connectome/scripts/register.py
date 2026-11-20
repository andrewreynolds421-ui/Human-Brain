#!/usr/bin/env python3
"""Stage 63: telling the model whether it is reading talk or a book.

Stage 62's mix of books and speech to children kept most book skills but only part of what speech alone taught
(ever-questions 44 % against 75 %), although speech was already 62 % of the sentences in every pass and
question-order "ever" outnumbered statement-order "ever" 3 to 1 in the mix. So the two kinds of language
interfere. A listener knows whether they are in a conversation or reading a book; this model did not.

Here the same mix is read with a register marker: every sentence's start token says "book" (books and teaching
pages) or "talk" (CHILDES adult speech). Nothing else changes (Stage 62's books_speech group is the control:
same material, same passes, one start token). After training every test is run twice, once under each marker.

Stage 64 (--pages both): the same, but each teaching page carries the talk marker or the book marker, half each
(fixed per page by a checksum of its words), to see whether the page rules can then be used under both markers.
Results in results/register/both_book and both_talk.

    python scripts/register.py run --seed 0
    python scripts/register.py run --pages both --seed 0
"""
import argparse, os, pickle, sys, zlib

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import frames as FR, grammar_memory as GM, phenomena as PH, sleep_grammar as SG, childes as CH   # noqa: E402
DATA = "/home/claude/models/register"
OUT = os.path.join(ROOT, "results", "register")
TAG = "xxtalkstart"                                             # vocabulary entry used as the "talk" start token (never a target)


class Talk(list):
    """a sentence heard as speech"""


class Tagged(PH.Reader):
    """Stage 60's reader with two start tokens: 1 = book, the TAG word's id = talk"""
    EVAL = "book"                                               # marker given to sentences that carry none (the tests)

    def __init__(self, vocab, seed):
        super().__init__(sorted(set(vocab) | {TAG}), seed); self.T = self.stoi[TAG]

    def enc(self, w):
        out = super().enc(w)
        if isinstance(w, Talk) or Tagged.EVAL == "talk": out[0] = self.T
        return out


def half_talk(make):
    """wrap a page maker: half of its pages (by checksum of the words) are read under the talk marker"""
    def f(*x, **k):
        it = make(*x, **k)
        if zlib.crc32(" ".join(it["words"]).encode()) & 1: it = dict(it, words=Talk(it["words"]))
        return it
    return f


def run(a):
    pre = ""
    if a.pages == "both":
        pre = "both_"; FR.make_page = half_talk(FR.make_page); PH.phenomenon_page = half_talk(PH.phenomenon_page); SG.adult_page = half_talk(SG.adult_page)
    speech = [Talk(s) for s in pickle.load(open(os.path.join(CH.DATA, "adult_speech.pkl"), "rb"))]
    books = GM.corpus_big
    GM.corpus_big = lambda C_: dict(C_, train=books(C_)["train"] + speech)
    PH.Reader = Tagged; SG.DATA = os.path.join(DATA, "models" + ("_both" if pre else ""))
    for tag in ("book", "talk"):                                # first call trains (tests under "book"), second loads and tests under "talk"
        Tagged.EVAL = tag; SG.OUT = os.path.join(OUT, pre + tag)
        os.makedirs(SG.DATA, exist_ok=True); os.makedirs(SG.OUT, exist_ok=True)
        if not os.path.exists(os.path.join(SG.OUT, f"awake_s{a.seed}.json")):
            SG.run(argparse.Namespace(cond="awake", seed=a.seed))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run",)); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--pages", choices=("book", "both"), default="book")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
