#!/usr/bin/env python3
"""Stage 47: the brain as Warmind's intent router (the hybrid, option 1).

Task and data are Warmind's own (Warmind Audit Package, intent/): 10 intents (labels.py: LOCAL_* handled
on the phone, REMOTE_* picks the submind), 106 benchmark phrasings (bench_flyhash.py DATA), the splits of
bench_flyhash.make_splits (n per class, seeds), and the shipped router (char_wb 3-5-gram TF-IDF + logistic
regression, class-balanced) as the baseline to beat.  Copies in data/intent/.

The brain's router:
  units      - a unit per word, per word piece and per letter triple inside a word, recruited the first
               time it is seen (Stage 46 token units + letter-sequence detectors, cf. Stage 37 letter pairs)
  decisions  - 10 decision units read the units' activity (unit length) and learn by the delta rule
  memory     - every utterance met is kept (episodic memory) and replayed `replay` times after each new
               one (interleaved replay, as Stage 41) - learning from each use, no batch retraining
  teacher    - LABELLED: the intent the request turned out to need (Warmind logs it in tool_calls, so the
               label comes free after the fact); or REWARD ONLY: the brain picks an intent and is told only
               right / wrong (three-factor rule on the chosen unit, reward minus its expectation)

eta and replay are FITTED on splits with seeds 5-9; results are on seeds 0-4 (bench seeds), so the tuning
splits are not the test splits.

Also: an online-use run - the 106 phrasings arrive one at a time in random order, the router answers
before it is taught, then learns; the shipped router is refitted after every request for comparison.

    python scripts/intent_brain.py   ->  results/intent_brain/summary.json
"""
import json, os, re, sys
from collections import Counter
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = json.load(open(os.path.join(ROOT, "data", "intent", "bench_utterances.json")))
LABELS = list(DATA)
OUT = os.path.join(ROOT, "results", "intent_brain")


def make_splits(n_per_class, seed):
    """bench_flyhash.make_splits, unchanged"""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label, items in DATA.items():
        items = list(items)
        rng.shuffle(items)
        n = min(n_per_class, len(items) - 3)
        train += [(t, label) for t in items[:n]]
        test += [(t, label) for t in items[n:]]
    return train, test


class Units:
    ngrams = (3,)                                     # letter-sequence lengths (Stage 47 variant: 3, 4, 5)

    def __init__(self):
        self.id = {}

    def act(self, text, learn=True):
        a = {}
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            keys = ["w:" + w] + [f"t{n}:" + w[i:i + n] for n in self.ngrams for i in range(max(1, len(w) - n + 1))] + ["b:" + w[:3], "e:" + w[-3:]]
            for k in keys:
                if k not in self.id:
                    if not learn:
                        continue
                    self.id[k] = len(self.id)
                a[self.id[k]] = a.get(self.id[k], 0) + 1
        return a


class Brain:
    habituate = False                                 # Stage 47 variant: units fading with how often they fire

    def __init__(self, eta, replay, reward_only=False, seed=0, cap=20000):
        self.U = Units(); self.fired = np.zeros(cap); self.seen = 0; self.W = np.zeros((len(LABELS), cap)); self.b = np.zeros(len(LABELS))
        self.eta, self.replay, self.reward_only = eta, replay, reward_only
        self.mem = []; self.texts = []; self.rng = np.random.default_rng(seed); self.base = np.zeros(len(LABELS))

    def vec(self, text, learn=True, count=False):
        a = self.U.act(text, learn); v = np.zeros(self.W.shape[1])
        for u, c in a.items():
            v[u] = np.log1p(c)
        if count:                                     # habituation: a unit's response falls with how many requests
            self.seen += 1; self.fired[list(a)] += 1  # it has fired for, log(seen / fired) - an idf in neural form
        if self.habituate:
            v = v * np.log((self.seen + 1.0) / (self.fired + 1.0) + 1.0)
        return v / (np.linalg.norm(v) + 1e-9)

    def probs(self, v):
        z = self.W @ v + self.b; z -= z.max(); p = np.exp(z)
        return p / p.sum()

    def predict(self, text):
        p = self.probs(self.vec(text, learn=False))
        return LABELS[int(np.argmax(p))], float(np.max(p))

    def _lesson(self, v, y=None, choice=None, reward=None):
        p = self.probs(v)
        if not self.reward_only:
            err = -p; err[y] += 1
            self.W += self.eta * np.outer(err, v); self.b += 0.1 * self.eta * err
        else:                                         # three-factor: only the chosen unit, (reward - expectation)
            d = reward - self.base[choice]; self.base[choice] += 0.1 * d
            g = -p * d; g[choice] += d                # policy gradient of log p(choice)
            self.W += self.eta * np.outer(g, v); self.b += 0.1 * self.eta * g

    def learn(self, text, label):
        v = self.vec(text, count=True); y = LABELS.index(label)
        if self.reward_only:
            p = self.probs(v); c = int(self.rng.choice(len(LABELS), p=p)); r = 1.0 if c == y else -1.0
            self.mem.append((v, None, c, r)); self._lesson(v, choice=c, reward=r)
        else:
            self.mem.append((v, y, None, None)); self._lesson(v, y=y)
        self.texts.append(text)
        for _ in range(self.replay):                  # replay remembered requests
            for k in self.rng.permutation(len(self.mem)):
                v_, y_, c_, r_ = self.mem[k]
                if self.habituate:                     # a remembered request is re-seen with today's habituation
                    v_ = self.vec(self.texts[k], learn=False)
                if self.reward_only:                   # replay re-tries the remembered request, told right / wrong
                    p = self.probs(v_); c = int(self.rng.choice(len(LABELS), p=p))
                    yy = c_ if r_ > 0 else None
                    r = (1.0 if c == yy else -1.0) if yy is not None else (-1.0 if c == c_ else 0.0)
                    if r != 0.0:
                        self._lesson(v_, choice=c, reward=r)
                else:
                    self._lesson(v_, y=y_)


def shipped(train):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)), LogisticRegression(max_iter=2000, class_weight="balanced"))
    return pipe.fit([t for t, _ in train], [y for _, y in train])


def macro_f1(pred, gold):
    f = []
    for l in LABELS:
        tp = sum(p == l and g == l for p, g in zip(pred, gold)); fp = sum(p == l and g != l for p, g in zip(pred, gold)); fn = sum(p != l and g == l for p, g in zip(pred, gold))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def run_brain(n, seed, eta, replay, reward_only):
    train, test = make_splits(n, seed)
    B = Brain(eta, replay, reward_only, seed)
    for k in np.random.default_rng(100 + seed).permutation(len(train)):
        B.learn(*train[k])
    pred = [B.predict(t)[0] for t, _ in test]
    return float(np.mean([p == y for p, (_, y) in zip(pred, test)])), macro_f1(pred, [y for _, y in test])


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--ngrams", default="3"); ap.add_argument("--habituate", type=int, default=0)
    ap.add_argument("--out", default="summary.json"); a = ap.parse_args()
    Units.ngrams = tuple(int(x) for x in a.ngrams.split(",")); Brain.habituate = bool(a.habituate)
    os.makedirs(OUT, exist_ok=True)
    sizes = [1, 2, 3, 5, 8]
    res = dict(classes=len(LABELS), utterances=sum(len(v) for v in DATA.values()), ngrams=list(Units.ngrams), habituate=Brain.habituate)
    # fit eta / replay on seeds 5-9
    fits = {}
    for ro in (False, True):
        grid = [(e, r) for e in (0.3, 1.0, 3.0) for r in (0, 3, 10)]
        sc = {g: np.mean([run_brain(n, s, *g, ro)[0] for n in (2, 5) for s in range(5, 10)]) for g in grid}
        fits[ro] = max(sc, key=sc.get)
        print("fitted", "reward-only" if ro else "labelled", fits[ro], round(float(sc[fits[ro]]), 3), flush=True)
    res["fitted"] = {"labelled": fits[False], "reward_only": fits[True]}
    table = {}
    for n in sizes:
        row = {}
        for name, fn in (("shipped_router", lambda s: (lambda m, te: (float(np.mean(m.predict([t for t, _ in te]) == [y for _, y in te])),
                                                                         macro_f1(list(m.predict([t for t, _ in te])), [y for _, y in te])))(shipped(make_splits(n, s)[0]), make_splits(n, s)[1])),
                         ("brain_labelled", lambda s: run_brain(n, s, *fits[False], False)),
                         ("brain_reward_only", lambda s: run_brain(n, s, *fits[True], True))):
            v = np.array([fn(s) for s in range(5)])
            row[name] = dict(accuracy=round(float(v[:, 0].mean()), 3), accuracy_sd=round(float(v[:, 0].std(ddof=1)), 3), macro_f1=round(float(v[:, 1].mean()), 3))
        table[n] = row
        print(n, {k: v["accuracy"] for k, v in row.items()}, flush=True)
    res["by_examples_per_intent"] = table
    # online use: 106 requests one at a time, answer first, then learn
    curves = {"shipped_router": [], "brain_labelled": [], "brain_reward_only": []}
    allu = [(t, l) for l, v in DATA.items() for t in v]
    for s in range(5):
        order = [allu[i] for i in np.random.default_rng(200 + s).permutation(len(allu))]
        Bl = Brain(*fits[False], False, s); Br = Brain(*fits[True], True, s); seen = []
        hits = {k: [] for k in curves}
        for t, y in order:
            if len({l for _, l in seen}) >= 2:
                hits["shipped_router"].append(shipped(seen).predict([t])[0] == y)
            else:
                hits["shipped_router"].append(y == "REMOTE_CHAT")       # untrained: everything goes to RASPUTIN
            hits["brain_labelled"].append(Bl.predict(t)[0] == y if Bl.mem else y == "REMOTE_CHAT")
            hits["brain_reward_only"].append(Br.predict(t)[0] == y if Br.mem else y == "REMOTE_CHAT")
            Bl.learn(t, y); Br.learn(t, y); seen.append((t, y))
        for k in curves:
            curves[k].append([float(np.mean(c)) for c in np.array_split(np.array(hits[k], float), 5)])
    res["online_use"] = {k: dict(by_fifth=np.round(np.mean(v, 0), 3).tolist(), overall=round(float(np.mean(v)), 3)) for k, v in curves.items()}
    print(res["online_use"], flush=True)
    json.dump(res, open(os.path.join(OUT, a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
