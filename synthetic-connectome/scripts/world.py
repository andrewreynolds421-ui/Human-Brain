#!/usr/bin/env python3
"""Stage 74: a small world - reading for what is true, not only for what is likely.

VEGA predicts words; it has no model of what a text is about. Here it reads stories about a small world - people
moving between places and picking up, putting down and giving objects - and is asked what is true at the end.
The answer is decided by the world's state, not by which words are common: the ball is where the person who
picked it up went later, though "garden" was the last word said near "ball".

STORY: each person and object starts somewhere ("mary was in the kitchen", "the ball was in the garden"), then
4-10 events (training) in random order:
  move   mary went to / walked to / ran to the garden
  take   mary picked up / took the ball            (only what is in the same place and not held)
  drop   mary put down / dropped the ball          (it stays where she is)
  give   mary gave the ball to john                (both in the same place)
Training stories end with two true statements ("the ball is in the garden", "john has the ball").
QUESTIONS (tests): the story, then one statement with four choices; VEGA picks the one it finds most likely.
  where_person  mary is in the X          where_object  the ball is in the X        who_has  X has the ball
Choices: the true one, where the thing was before (when it moved), others from the story, then random.
Each where_object item is marked *carried* when the object's place changed only because its holder moved -
the case no surface cue can answer (the last place said near the object is wrong).
TEST SETS (1,000 stories each; half about carried objects, a quarter who has what, a quarter where a person is): same kind as training; held-out names; held-out objects and places (never in
any story); long stories (14-18 events). Chance is 25 %.

Training (`scripts/day_night.py --cond cycle --start day_night/year4_cycle --new world --tag world`): Stage 73's
networks, 4 days of 120,000 training stories with gentle nights replaying older memory (books, pages, speech,
the school year).

    python scripts/world.py eval --models /home/claude/models/day_night/year4_cycle --tag start --seed 0
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "world")
PLACES = ["kitchen", "garden", "barn", "hall", "bedroom", "yard", "school", "office"]
HELD_PLACES = ["cellar", "attic", "library", "orchard"]
OBJECTS = ["ball", "hat", "key", "book", "apple", "cup", "box", "coat", "bell", "doll", "lamp", "ring"]
HELD_OBJECTS = ["bag", "bottle", "basket", "letter", "knife", "cake"]
NAMES = ["mary", "anna", "emma", "jane", "sarah", "alice", "grace", "kate", "susan", "martha", "nancy", "ruth", "laura", "julia", "ellen", "marie",
         "john", "tom", "james", "george", "henry", "william", "charles", "robert", "david", "paul", "frank", "edward", "joseph", "harry", "fred", "sam"]
HELD_NAMES = ["clara", "diana", "helen", "lucy", "rebecca", "rose", "carl", "jack", "michael", "peter"]
TRAIN_STORIES = 120_000
MOVE = ["went to the", "walked to the", "ran to the"]; TAKE = ["picked up the", "took the"]; DROP = ["put down the", "dropped the"]


def story(rng, names, objects, places, n_events, n_people=3, n_obj=3):
    """returns (words, state, history): state = person -> place, object -> (place, holder)"""
    P = list(rng.choice(names, n_people, replace=False)); O = list(rng.choice(objects, n_obj, replace=False)); L = list(rng.choice(places, min(4, len(places)), replace=False))
    where = {p: str(rng.choice(L)) for p in P}; obj = {o: [str(rng.choice(L)), None] for o in O}
    hist = {x: [] for x in P + O}; carried = {o: False for o in O}
    w = []; ends = []                                                   # ends: index of each sentence's last word (Stage 89)
    for p in P: w += [p, "was", "in", "the", where[p]]; ends.append(len(w) - 1)
    for o in O: w += ["the", o, "was", "in", "the", obj[o][0]]; ends.append(len(w) - 1)
    for _ in range(n_events):
        for _try in range(20):
            k = int(rng.integers(4)); p = str(rng.choice(P))
            if k == 0:
                to = str(rng.choice([l for l in L if l != where[p]])); hist[p].append(where[p]); where[p] = to
                for o in O:
                    if obj[o][1] == p: hist[o].append(obj[o][0]); obj[o][0] = to; carried[o] = True
                w += [p] + str(rng.choice(MOVE)).split() + [to]; ends.append(len(w) - 1); break
            here = [o for o in O if obj[o][1] is None and obj[o][0] == where[p]]; mine = [o for o in O if obj[o][1] == p]
            if k == 1 and here:
                o = str(rng.choice(here)); obj[o][1] = p; w += [p] + str(rng.choice(TAKE)).split() + [o]; ends.append(len(w) - 1); break
            if k == 2 and mine:
                o = str(rng.choice(mine)); obj[o][1] = None; carried[o] = False; w += [p] + str(rng.choice(DROP)).split() + [o]; ends.append(len(w) - 1); break
            others = [q for q in P if q != p and where[q] == where[p]]
            if k == 3 and mine and others:
                o = str(rng.choice(mine)); q = str(rng.choice(others)); obj[o][1] = q; w += [p, "gave", "the", o, "to", q]; ends.append(len(w) - 1); break
    return w, dict(where=where, obj=obj, hist=hist, carried=carried, people=P, objects=O, places=L, ends=ends)


def statements(st):
    """every true end statement of a story: (kind, words, subject)"""
    out = [("where_person", [p, "is", "in", "the", st["where"][p]], p) for p in st["people"]]
    out += [("where_object", ["the", o, "is", "in", "the", st["obj"][o][0]], o) for o in st["objects"]]
    out += [("who_has", [st["obj"][o][1], "has", "the", o], o) for o in st["objects"] if st["obj"][o][1]]
    return out


def training_story(rng):
    w, st = story(rng, NAMES, OBJECTS, PLACES, int(rng.integers(4, 11)))
    s = statements(st); pick = rng.choice(len(s), min(2, len(s)), replace=False)
    for i in pick: w = w + s[i][1]
    return w


def question(rng, w, st, places, want=None):
    """one test item: the story plus four versions of a final statement, the first true. want="carried": ask
    about an object moved only by its holder walking (None if the story has none)"""
    S = statements(st)
    if want == "carried": S = [x for x in S if x[0] == "where_object" and st["carried"][x[2]]]
    elif want: S = [x for x in S if x[0] == want]
    if not S: return None
    kind, true, subj = S[int(rng.integers(len(S)))]
    if kind == "who_has":
        alts = [p for p in st["people"] if p != true[0]]; opts = [true] + [[a] + true[1:] for a in alts]
    else:
        t = true[-1]; pool = list(dict.fromkeys(st["hist"][subj][::-1] + st["places"] + list(rng.permutation(places))))
        alts = [x for x in pool if x != t][:3]; opts = [true] + [true[:-1] + [a] for a in alts]
    opts = opts[:4]
    while len(opts) < 4: opts.append(opts[-1])
    moved = bool(st["hist"][subj]) if kind != "who_has" else False
    return dict(story=w, ends=list(st.get("ends", [])), options=opts, kind=kind, carried=bool(kind == "where_object" and st["carried"][subj]), moved=moved,
                previous_offered=bool(kind != "who_has" and st["hist"][subj] and st["hist"][subj][-1] in [o[-1] for o in opts[1:]]))


def test_sets(seed=0, n=1000):
    """each set: half carried objects (the hard case), a quarter who has what, a quarter where a person is"""
    rng = np.random.default_rng(10_000 + seed); T = {}
    spec = dict(same=(NAMES, OBJECTS, PLACES, 4, 11), held_names=(HELD_NAMES, OBJECTS, PLACES, 4, 11),
                held_objects_places=(NAMES, HELD_OBJECTS, HELD_PLACES, 4, 11), long=(NAMES, OBJECTS, PLACES, 14, 19))
    for name, (nm, ob, pl, lo, hi) in spec.items():
        items = []
        for want, k in (("carried", n // 2), ("who_has", n // 4), ("where_person", n - n // 2 - n // 4)):
            got = 0
            while got < k:
                it = question(rng, *story(rng, nm, ob, pl, int(rng.integers(lo, hi))), pl, want)
                if it: items.append(it); got += 1
        T[name] = items
    return T


def answer(M, items):
    seqs = [it["story"] + o for it in items for o in it["options"]]
    lp = np.array(M.logp(seqs)).reshape(len(items), 4)
    return lp.argmax(1) == 0


def last_mention(it):
    """surface cue: the last place word said in the story right after the subject's name (or, for who_has, the last
    name said before the object) - what a reader without a model of the world might guess"""
    w = it["story"]; t = it["options"][0]
    if it["kind"] == "who_has":
        o = t[-1]; names = [w[i] for i in range(len(w)) if i + 3 < len(w) and w[i + 3] == o or (i + 2 < len(w) and w[i + 2] == o and w[i + 1] in ("took",))]
        guess = names[-1] if names else None; return guess == t[0]
    subj = t[1] if t[0] == "the" else t[0]; places = set(o[-1] for o in it["options"])
    last = None
    for i, x in enumerate(w):
        if x == subj:
            nxt = [y for y in w[i + 1:i + 7] if y in places]
            if nxt: last = nxt[0]
    return last == t[-1]


def evaluate(M, tag, seed, extra=None):
    T = test_sets(seed=0); out = dict(tag=tag, seed=seed, **(extra or {}))
    for name, items in T.items():
        hit = answer(M, items); r = dict(n=len(items), all=round(float(hit.mean()), 4))
        for k in ("where_person", "where_object", "who_has"):
            m = np.array([it["kind"] == k for it in items]); r[k] = round(float(hit[m].mean()), 4) if m.any() else None
        car = np.array([it["carried"] for it in items]); r["carried_objects"] = round(float(hit[car].mean()), 4) if car.any() else None; r["n_carried"] = int(car.sum())
        cue = np.array([last_mention(it) for it in items]); r["surface_cue_right"] = round(float(cue.mean()), 4)
        r["when_cue_wrong"] = round(float(hit[~cue].mean()), 4) if (~cue).any() else None
        out[name] = r
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, f"{tag}_s{seed}.json"), "w"), indent=1); print(json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("eval", "show")); ap.add_argument("--models", default="/home/claude/models/day_night/year4_cycle")
    ap.add_argument("--tag", default="start"); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    if a.cmd == "show":
        rng = np.random.default_rng(1); print(" ".join(training_story(rng)))
        for k, v in test_sets(n=3).items(): print(k, " ".join(v[0]["story"]), "|", [" ".join(o) for o in v[0]["options"]], v[0]["kind"], v[0]["carried"])
        return
    import grammar_memory as GM, output_order as OO
    def ev(M, C, tok, seed, res):
        evaluate(M, a.tag, a.seed); raise SystemExit
    GM.evaluate = ev; OO.DATA = a.models; OO.OUT = "/tmp/world_scratch"; os.makedirs(OO.OUT, exist_ok=True)
    OO.run(argparse.Namespace(cond="books", seed=a.seed))


if __name__ == "__main__":
    main()
