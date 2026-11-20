#!/usr/bin/env python3
"""Stage 63: telling the model whether it is reading talk or a book.

Stage 62's mix of books and speech to children kept most book skills but only part of what speech alone taught
(ever-questions 44 % against 75 %), although speech was already 62 % of the sentences in every pass and
question-order "ever" outnumbered statement-order "ever" 3 to 1 in the mix. So the two kinds of language
interfere. A listener knows whether they are in a conversation or reading a book; this model did not.

Here the same mix is read with a register marker: every sentence's start token says "book" (books and teaching
pages) or "talk" (CHILDES adult speech). Nothing else changes (Stage 62's books_speech group is the control:
same material, same passes, one start token). After training every test is run twice, once under each marker.

    python scripts/register.py run --seed 0
"""
import argparse, os, pickle, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar_memory as GM, phenomena as PH, sleep_grammar as SG, childes as CH   # noqa: E402
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


def run(a):
    speech = [Talk(s) for s in pickle.load(open(os.path.join(CH.DATA, "adult_speech.pkl"), "rb"))]
    books = GM.corpus_big
    GM.corpus_big = lambda C_: dict(C_, train=books(C_)["train"] + speech)
    PH.Reader = Tagged; SG.DATA = os.path.join(DATA, "models")
    for tag in ("book", "talk"):                                # first call trains (tests under "book"), second loads and tests under "talk"
        Tagged.EVAL = tag; SG.OUT = os.path.join(OUT, tag)
        os.makedirs(SG.DATA, exist_ok=True); os.makedirs(SG.OUT, exist_ok=True)
        if not os.path.exists(os.path.join(SG.OUT, f"awake_s{a.seed}.json")):
            SG.run(argparse.Namespace(cond="awake", seed=a.seed))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run",)); ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
