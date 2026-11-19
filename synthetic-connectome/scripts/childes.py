#!/usr/bin/env python3
"""Stage 62: learning from speech to children (CHILDES).

Stage 61's model was uneven: good at the rules drilled with teaching pages, poor at what children get early from
everyday talk (questions, who / what, irregular verbs). Its reading was old storybooks, not conversation. Here it
also hears real speech to children: the CHILDES archive (MacWhinney 2000; North American English transcripts,
via the copy in github.com/enochlev/CHILDES_LLM_training, data/child_full.csv) - 1.3 M utterances, 10.2 M words;
we use only what the ADULTS say (mother, father, investigator, other adults: 7.4 M words, 38 % questions), not the
children's own speech. CHILDES data are for research use under TalkBank's rules (cite MacWhinney 2000); the
corpus is not copied into this repository.

Groups (four networks each; same teaching pages and two passes as Stage 61's awake group, which is the reference
"books + pages"):
  speech        CHILDES adult speech + teaching pages (no books)
  books_speech  books + CHILDES adult speech + teaching pages
Tests: BLiMP by field; Stage 60-61 built pairs; Stage 51 battery.

    python scripts/childes.py prepare
    python scripts/childes.py run --cond speech --seed 0
"""
import argparse, json, os, pickle, re, sys, time
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar as GR, grammar_memory as GM, picture_check as PC, frames as FR, phenomena as PH, sleep_grammar as SG   # noqa: E402
CSV = "/home/claude/corpora/childes_probe/enochlev_CHILDES_LLM_training/data/child_full.csv"
DATA = "/home/claude/models/childes"
OUT = os.path.join(ROOT, "results", "childes")
JUNK = {"xxx", "yyy", "www", "xx", "yy"}


def prepare():
    import pandas as pd
    os.makedirs(DATA, exist_ok=True)
    d = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    d = d[d.speaker != "CHI"]
    sents, kept = [], 0
    for t in d.text:
        for part in re.split(r"[.?!;]+", t.lower()):
            w = [x for x in re.findall(r"[a-z]+(?:'[a-z]+)?", part) if x not in JUNK]
            if len(w) >= 2: sents.append(w)
    pickle.dump(sents, open(os.path.join(DATA, "adult_speech.pkl"), "wb"))
    n = sum(map(len, sents))
    print("adult sentences", len(sents), "words", n, "mean length", round(n / len(sents), 2))
    for s in sents[1000:1012]: print("  ", " ".join(s))


def run(a):
    """Stage 61's training loop (awake, no nights) with this stage's reading"""
    speech = pickle.load(open(os.path.join(DATA, "adult_speech.pkl"), "rb"))
    books = GM.corpus_big
    GM.corpus_big = (lambda C_: dict(C_, train=speech)) if a.cond == "speech" else (lambda C_: dict(C_, train=books(C_)["train"] + speech))
    SG.DATA = os.path.join(DATA, "models", a.cond); SG.OUT = os.path.join(OUT, a.cond)
    os.makedirs(SG.DATA, exist_ok=True); os.makedirs(SG.OUT, exist_ok=True)
    SG.run(argparse.Namespace(cond="awake", seed=a.seed))       # files are named awake_s{seed} inside the group's folder


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("prepare", "run"))
    ap.add_argument("--cond", choices=("speech", "books_speech"), default="speech"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); prepare() if a.cmd == "prepare" else run(a)


if __name__ == "__main__":
    main()
