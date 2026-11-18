#!/usr/bin/env python3
"""Stage 57: many frames as well as many nouns.

Stage 56: with ~700 nouns the agreement rule began to carry to held-out nouns (relative clauses 60-65 %),
but not to "of the", a kind of phrase never on a page - the model learned the drilled frames. Here the pages
vary the frames too, and some frames are held out:

  prepositions  ~28 on pages; HELD OUT: of, under, across, toward
  verbs in relative clauses  ~35 (past) + see / sees; HELD OUT: helped, loved, met, called
  what is said of the subject  actions + adjectives; HELD OUT: here, there, afraid, happy, still, in the garden
  sentence types  Stage 55's five + two phrases deep ("the X by the Y near the Z is ...") + relative clause
                  inside a phrase ("the X near the Y that the Z saw is ...")

Same nouns split as Stage 56 (679 on pages, 218 held out), same reading (books, two passes, 150 k pages per
pass), same groups (text / check), four networks each. Test: built sentences made ONLY of held-out frames
("the X of the Y ___ here", "the X that the Y helped ___ afraid"), with page nouns and with held-out nouns.

    python scripts/frames.py run --cond check --seed 0
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import picture_check as PC, picture_pages as PP, grammar_memory as GM          # noqa: E402

PREPS = ["in", "on", "at", "near", "by", "with", "behind", "beside", "from", "among", "over", "around", "inside", "outside", "along", "beneath",
         "above", "below", "beyond", "against", "upon", "within", "through", "into", "about", "after", "before", "between"]
HELD_PREPS = ["of", "under", "across", "toward"]
REL = ["saw", "heard", "watched", "chased", "fed", "carried", "caught", "kept", "left", "taught", "told", "visited", "liked", "found", "followed", "knew",
       "asked", "thanked", "pushed", "pulled", "wanted", "missed", "praised", "feared", "trusted", "warned", "kissed", "greeted", "noticed", "remembered",
       "served", "guarded", "led", "named", "dressed"]
HELD_REL = ["helped", "loved", "met", "called"]
PRED = [[a] for a in PP.ACTIONS] + [[w] for w in ("hungry", "tired", "sleepy", "cold", "ready", "busy", "glad", "sad", "angry", "safe", "lost", "asleep", "quiet", "late")] + \
       [["very", "tired"], ["so", "glad"], ["in", "the", "house"], ["on", "the", "hill"], ["at", "the", "door"]]
HELD_PRED = [["here"], ["there"], ["afraid"], ["very", "happy"], ["quite", "still"], ["in", "the", "garden"]]
TYPES = ["simple", "pp", "rel_obj", "rel_subj", "comp", "pp2", "pp_rel"]


def make_page(rng, nouns, look, preps=PREPS, rel=REL, pred=PRED):
    i, j, k = rng.choice(len(nouns), 3, replace=False); a, b, c = nouns[i], nouns[j], nouns[k]
    na, nb, nc = (int(rng.integers(2)) for _ in range(3)); A, B, Cc = a[na], b[nb], c[nc]
    past = bool(rng.integers(2)); be = ("were" if na else "was") if past else ("are" if na else "is")
    P = lambda: str(rng.choice(preps)); R = lambda n: str(rng.choice(rel)) if (past or rng.integers(2)) else ("see" if n else "sees")
    ph = list(pred[rng.integers(len(pred))]); t = str(rng.choice(TYPES))
    if t == "simple":     w = ["the", A, be] + ph
    elif t == "pp":       w = ["the", A, P(), "the", B, be] + ph
    elif t == "rel_obj":  w = ["the", A, "that", "the", B, R(nb), be] + ph
    elif t == "rel_subj": w = ["the", A, "that", R(na), "the", B, be] + ph
    elif t == "comp":     w = ["the", B, R(nb), "that", "the", A, be] + ph
    elif t == "pp2":      w = ["the", A, P(), "the", B, P(), "the", Cc, be] + ph
    else:                 w = ["the", A, P(), "the", B, "that", "the", Cc, R(nc), be] + ph
    v = [m for m, x in enumerate(w) if x == be and m >= 2][0] if t != "comp" else w.index(be, 5)
    mis = (na != nb) or (t in ("pp2", "pp_rel") and na != nc)
    return dict(words=w, verb=v, number=na, look=look(a[0], rng), tmpl=t, attractor=mis)


def built_tests(train_nouns, held, rng, n=1500):
    """only held-out frames: 'of / under / across / toward the Y', 'that the Y helped / loved / met / called', held-out endings"""
    allp = train_nouns + held; out = {}
    for name, subj in (("page_nouns", train_nouns), ("held_out_nouns", held)):
        items = []
        while len(items) < n:
            a = subj[rng.integers(len(subj))]; b = allp[rng.integers(len(allp))]
            if a == b: continue
            kind = str(rng.choice(["pp", "rel"])); sp, bp = int(rng.integers(2)), int(rng.integers(2))
            v = ("were" if sp else "was") if rng.integers(2) else ("are" if sp else "is")
            mid = [str(rng.choice(HELD_PREPS)), "the", b[bp]] if kind == "pp" else ["that", "the", b[bp], str(rng.choice(HELD_REL))]
            good = ["the", a[sp]] + mid + [v] + list(HELD_PRED[rng.integers(len(HELD_PRED))]); bad = list(good); bad[2 + len(mid)] = GM.FLIP[v]
            items.append(dict(good=good, bad=bad, kind=kind, attractor=(sp != bp), verb=2 + len(mid), prep=mid[0] if kind == "pp" else None))
        out[name] = items
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--cond", choices=("text", "check"), default="check"); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    # reuse Stage 56's training and testing with this stage's pages, frames and output folder
    PC.make_page = make_page; PC.built_tests = built_tests
    PC.DATA = "/home/claude/models/frames"; PC.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "frames")
    PC.PP.ACTIONS = PP.ACTIONS
    extra = set(PREPS) | set(HELD_PREPS) | set(REL) | set(HELD_REL) | {w for p in PRED + HELD_PRED for w in p}
    orig_nouns = PC.nouns
    PC.nouns = lambda: orig_nouns()
    PC.PP.PREPS = sorted(set(PP.PREPS) | extra)                                  # so every frame word is in the vocabulary
    PC.run(a)


if __name__ == "__main__":
    main()
