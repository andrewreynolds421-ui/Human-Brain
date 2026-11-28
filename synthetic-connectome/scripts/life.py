#!/usr/bin/env python3
"""Stage 100: VEGA's life - a test of long-term memory.

Up to Stage 99, VEGA reads one story and is questioned about it straight away; nothing it read survives the
question except what changed in its weights. Here it lives through DAYS in one small world with one cast. Every
day brings a few episodes (scenes in a place: people come and go, pick things up, put them down, give them to each
other) and, now and then, a lasting fact said once ("anna has a dog called rex", "mary lives in the barn", later
"mary now lives in the yard"). At the end of each day VEGA is questioned about today AND about the past.

QUESTION KINDS (answer = one word; `supp` = the sentences that decide it, as (day, episode, sentence)):
  today      where is the key now?                         the object's place at the end of today
  long_ago   where was the key three days ago?             its place at the end of that day (1-29 days back)
  told_once  what is anna's dog called?                    a pet named once, any time before
  changed    where does mary live?                         the newest home; older homes are the trap
  never      what is tom's cat called? -> unknown          tom was never said to have a cat (guessing is caught)
Every item carries `delay` = days between now and the newest sentence that decides it, so accuracy can be drawn
against delay (the forgetting curve).

The memory sees each episode as a list of sentences (lists of words) with its day; a question gives "now". The
generator is pure Python and seeded; `python scripts/life.py --seed 0 --stats` prints what one life contains.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import world as W                                                    # Stage 74's verbs, so the reading network knows them

PEOPLE = ["mary", "anna", "emma", "jane", "sarah", "alice", "john", "tom", "james", "george", "henry", "paul"]
OBJECTS = ["ball", "hat", "key", "book", "apple", "cup", "box", "coat", "bell", "doll"]
PLACES = ["kitchen", "garden", "barn", "hall", "bedroom", "yard", "school", "office"]
PETS = ["dog", "cat", "horse", "bird"]
PET_NAMES = ["rex", "max", "spot", "rover", "daisy", "molly", "bella", "buster", "ginger", "pepper", "shadow", "lady"]
NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
       "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
UNKNOWN = "unknown"


def number(n):                                                       # 1..29 as words: "twenty three"
    return [NUM[n]] if n <= 20 else ["twenty", NUM[n - 20]]


def ago(k):
    return ["yesterday"] if k == 1 else number(k) + ["days", "ago"]


def live(seed=0, days=30, episodes=5, questions_per_day=12, facts_per_day=0.6):
    rng = np.random.default_rng(seed)
    where = {p: str(rng.choice(PLACES)) for p in PEOPLE}
    obj = {o: [str(rng.choice(PLACES)), None] for o in OBJECTS}               # place, holder
    obj_src = {o: None for o in OBJECTS}                                       # (day, ep, sent) that last set it
    home, home_src, home_old = {}, {}, {}
    pets = {}                                                                  # person -> (kind, name, src)
    untold_pets = [(p, k) for p in PEOPLE for k in PETS]; rng.shuffle(untold_pets)
    names_left = list(rng.permutation(PET_NAMES))
    snapshot = {}                                                              # day -> {object: (place, src)}
    life, qs = [], []

    for d in range(days):
        for e in range(episodes):
            here = str(rng.choice(PLACES)); cast = list(rng.choice(PEOPLE, int(rng.integers(2, 4)), replace=False))
            S = []

            def say(words):
                S.append(words); return (d, e, len(S) - 1)

            for p in cast:                                                     # everyone in the scene comes here
                if where[p] != here:
                    for o in OBJECTS:
                        if obj[o][1] == p: obj[o][0] = here; obj_src[o] = None   # carried along: decided by the move below
                    where[p] = here; src = say([p] + str(rng.choice(W.MOVE)).split() + [here])
                    for o in OBJECTS:
                        if obj[o][1] == p: obj_src[o] = src
            for _ in range(int(rng.integers(3, 7))):
                k = int(rng.integers(3)); p = str(rng.choice(cast))
                loose = [o for o in OBJECTS if obj[o][1] is None and obj[o][0] == here]; mine = [o for o in OBJECTS if obj[o][1] == p]
                others = [q for q in cast if q != p]
                if k == 0 and loose:
                    o = str(rng.choice(loose)); obj[o][1] = p; obj_src[o] = say([p] + str(rng.choice(W.TAKE)).split() + [o])
                elif k == 1 and mine:
                    o = str(rng.choice(mine)); obj[o][1] = None; obj[o][0] = here; obj_src[o] = say([p] + str(rng.choice(W.DROP)).split() + [o])
                elif k == 2 and mine and others:
                    o = str(rng.choice(mine)); q = str(rng.choice(others)); obj[o][1] = q; obj_src[o] = say([p, "gave", "the", o, "to", q])
            if rng.random() < facts_per_day / episodes * 3 and untold_pets and names_left:   # a lasting fact, said once
                p, kind = untold_pets.pop(); nm = str(names_left.pop()); names_left.insert(0, nm)
                if p not in pets:
                    pets[p] = (kind, nm, say([p, "has", "a", kind, "called", nm]))
            if rng.random() < facts_per_day / episodes * 2:
                p = str(rng.choice(PEOPLE)); new = str(rng.choice([x for x in PLACES if x != home.get(p)]))
                if p in home: home_old.setdefault(p, []).append(home[p]); home_src[p] = say([p, "now", "lives", "in", "the", new])
                else: home_src[p] = say([p, "lives", "in", "the", new])
                home[p] = new
            if S: life.append(dict(day=d, episode=e, place=here, sents=S))
        snapshot[d] = {o: (obj[o][0], obj_src[o]) for o in OBJECTS}

        # end of day: questions about today and the past
        def add(kind, q, ans, src, extra=None):
            if src is None: return
            it = dict(day=d, kind=kind, q=q, ans=(ans,), supp=[list(src)], delay=d - src[0])
            if extra: it.update(extra)
            qs.append(it)

        for _ in range(questions_per_day):
            r = rng.random()
            if r < 0.25:
                o = str(rng.choice(OBJECTS)); pl, src = snapshot[d][o]
                add("today", ["where", "is", "the", o, "now"], pl, src)
            elif r < 0.55 and d > 0:
                k = int(rng.integers(1, d + 1)); o = str(rng.choice(OBJECTS)); pl, src = snapshot[d - k][o]
                if pl != snapshot[d][o][0]:                                     # only when the answer differs from today's
                    add("long_ago", ["where", "was", "the", o] + ago(k), pl, src, dict(days_back=k))
            elif r < 0.75 and pets:
                p = str(rng.choice(list(pets))); kind, nm, src = pets[p]
                add("told_once", ["what", "is", p, "s", kind, "called"], nm, src)
            elif r < 0.9 and home_old:
                p = str(rng.choice(list(home_old)))
                add("changed", ["where", "does", p, "live"], home[p], home_src[p], dict(old=home_old[p]))
            else:
                p = str(rng.choice(PEOPLE)); kinds = [k for k in PETS if pets.get(p, (None,))[0] != k]
                kind = str(rng.choice(kinds))
                qs.append(dict(day=d, kind="never", q=["what", "is", p, "s", kind, "called"], ans=(UNKNOWN,), supp=[], delay=None))
    answers = sorted({a for it in qs for a in it["ans"]} | set(PLACES) | set(PET_NAMES) | {UNKNOWN})
    for it in qs: it["y"] = answers.index(it["ans"][0])
    return dict(seed=seed, days=days, episodes=life, questions=qs, answers=answers)


def stats(L):
    from collections import Counter
    n_s = sum(len(e["sents"]) for e in L["episodes"])
    print(f"seed {L['seed']}: {L['days']} days, {len(L['episodes'])} episodes, {n_s} sentences, {len(L['questions'])} questions")
    print("kinds:", dict(Counter(it["kind"] for it in L["questions"])))
    dl = [it["delay"] for it in L["questions"] if it["delay"] is not None]
    print("delay (days) quartiles:", np.percentile(dl, [25, 50, 75, 100]).tolist())
    for e in L["episodes"][:2]:
        print(f"day {e['day']} episode {e['episode']} ({e['place']}):", " | ".join(" ".join(s) for s in e["sents"]))
    for kind in ("today", "long_ago", "told_once", "changed", "never"):
        it = next((x for x in L["questions"] if x["kind"] == kind and x["day"] > 10), None)
        if it: print(f"  [{kind}] day {it['day']}: {' '.join(it['q'])}? -> {it['ans'][0]}  (decided {it['delay']} days earlier by {it['supp']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--stats", action="store_true"); ap.add_argument("--out", default="")
    a = ap.parse_args(); L = live(a.seed, a.days)
    if a.stats or not a.out: stats(L)
    if a.out: json.dump(L, open(a.out, "w")); print("wrote", a.out)
