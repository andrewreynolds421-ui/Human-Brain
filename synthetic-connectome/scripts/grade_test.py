#!/usr/bin/env python3
"""Stage 66: where the fixed model (VEGA, Stage 65) reads, grade by grade.

Stage 53's reading test (multiple-choice cloze: a content word blanked, three distractors of the same part of
speech and frequency; test books binned by Flesch-Kincaid grade 1 .. 7+) applied to Stage 65's networks, which
were trained on other material. Items whose sentence also occurs in the network's own reading are dropped
(reported). "Reads at" a grade = score >= 75 % (ASSUMED, as in Stage 53). Chance is 25 %.

    python scripts/grade_test.py --cond books --seed 0        # also: --cond both (tested under each marker)
"""
import argparse, json, os, pickle, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar as GR, grammar_memory as GM, graded_reading as GD, sleep_grammar as SG, register as RG, output_order as OO   # noqa: E402
OUT = os.path.join(ROOT, "results", "grade_test")


def cloze(M, items):
    sents = [it["sent"][:it["pos"]] + [o] + it["sent"][it["pos"] + 1:] for it in items for o in it["options"]]
    lp = np.array(M.logp(sents)).reshape(len(items), 4)
    return float(np.mean(lp.argmax(1) == np.array([it["answer"] for it in items])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cond", choices=("books", "both"), default="books"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    items = pickle.load(open("/home/claude/models/graded/data.pkl", "rb"))["items"]["test"]
    seen = {}

    def corpus_seen(f):                                        # remember the network's own reading, to drop test sentences it has read
        def g(C_):
            Ct = f(C_); seen["s"] = {tuple(s) for s in Ct["train"]}; return Ct
        return g

    def test(M, C, tok, seed, res):
        out = dict(cond=a.cond, seed=a.seed, dropped={}, n={})
        clean = {}
        for b in GD.BANDS:
            clean[b] = [it for it in items[b] if tuple(it["sent"]) not in seen["s"]]; out["dropped"][b] = len(items[b]) - len(clean[b]); out["n"][b] = len(clean[b])
        for tag in (("book", "talk") if a.cond == "both" else ("book",)):
            RG.Tagged.EVAL = tag; out[tag] = {b: round(cloze(M, clean[b]), 4) for b in GD.BANDS}
            for pos in ("NOUN", "VERB", "ADJ", "ADV"):
                out[tag + "_" + pos] = round(cloze(M, [it for b in GD.BANDS for it in clean[b] if it["tag"] == pos]), 4)
        json.dump(out, open(os.path.join(OUT, f"{a.cond}_s{a.seed}.json"), "w"), indent=1); print(out, flush=True)
        raise SystemExit

    GM.evaluate = test; SG_OUT = "/tmp/grade_scratch"; os.makedirs(SG_OUT, exist_ok=True)
    real = GM.corpus_big; GM.corpus_big = corpus_seen(real)
    OO.OUT = SG_OUT                                             # the stage's own result files are not touched
    OO.run(argparse.Namespace(cond=a.cond, seed=a.seed))


if __name__ == "__main__":
    main()
