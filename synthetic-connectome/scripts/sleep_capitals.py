#!/usr/bin/env python3
"""Stage 49: sleep for the capitals that got stuck.

Set-up as Stage 41 (scripts/capitals.py): letter units on the edge cells know lowercase (labels); capitals
are learned WITHOUT labels from words that contain them ("The" 70 %, "THE" 30 %), each capital's percept
(units' evidence x the word context's prediction from both sides) teaching the units, with replay of
lowercase from a stable copy so lowercase is kept.  Taught one sighting at a time, some capitals got stuck:
the units read capital G as "c" so confidently that the word context could never overrule it.

Four days, 5,000 words each.  Each evening, one of:
  awake         - nothing more (Stage 41's word reading, continued over the days)
  awake_reread  - CONTROL: the day's words are read again, awake, as Stage 41 (same number of lessons as sleep)
  sleep         - offline, no input, no labels: the day's capital sightings are replayed and judged TOGETHER.
                  For each remembered sighting, its 10 most similar remembered capital shapes are found, and
                  the context's predictions for all of them (what the words around them said, apart from the
                  shapes) are pooled; the pooled context, combined with the units' evidence, names the letter
                  that teaches the units - evidence gathered across the day, which a single sighting does not
                  have.  Interleaved one-for-one with lowercase replay from the stable copy.
  sleep_down    - sleep, preceded by synaptic downscaling: the capital-learning part of the weights (what the
                  units learned since lowercase) is scaled by 0.8 (synaptic homeostasis: Tononi & Cirelli 2014),
                  so confident errors become less confident and the next day's context can overrule them.

Tests after the last night (60 never-seen families): each capital alone, capitals in words (The / THE),
lowercase alone; the stuck letters.  Step sizes as Stage 41 (eta_w 0.1, lapse 0.01, beta 1, conf 0.9, as
fitted there for network 1); four networks.

    python scripts/sleep_capitals.py --seed 1   ->  results/sleep_capitals/s1.json
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capitals as C                                                           # noqa: E402
from synth.words import L, combine, load_words, split_words                   # noqa: E402

OUT = os.path.join(ROOT, "results", "sleep_capitals")
ETA, LAPSE, BETA_T, CONF, K_NEIGH, DOWN = 0.1, 0.01, 1.0, 0.9, 10, 0.8
ABC = C.ABC


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--per-day", type=int, default=5000)
    ap.add_argument("--diagnose", action="store_true"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    V, y = C.load(); rng = np.random.default_rng(a.seed)
    words, freqs = load_words(5000); tr, ho = split_words(words, 0.2, a.seed)
    nf = {k: len(V[k]) // (C.NCLS * C.PER) for k in V}
    low = np.nonzero(y["train"] < 26)[0]; sub = rng.choice(low, 10000, replace=False)
    mu = V["train"][sub].mean(0); sd = V["train"][sub].std(0) + 1e-3
    V = {k: ((V[k] - mu) / sd).astype(np.float32) for k in V}
    m0, _ = C.fit(V["train"][sub], y["train"][sub], V["val"][y["val"] < 26], y["val"][y["val"] < 26], Cs=(0.003, 0.01, 0.03))
    xn2 = float(np.mean(np.sum(V["train"][sub] ** 2, 1)))
    U0 = C.Units(m0, xn2)
    lw = C.sample_words(rng, tr, freqs[tr], 20000, nf["train"], words, "lower")
    P = C.train_predictor(lw, V["train"], U0, 0.1)
    beta = 1.0
    stable = C.Units(m0, xn2)
    rr = np.random.default_rng(700 + a.seed)

    def replay_lower(U):
        x = V["train"][C.img_index(int(rr.integers(nf["train"])), int(rr.integers(26)), int(rr.integers(C.PER)))]
        U.learn(x, np.eye(L)[int(stable.evidence(x[None])[0].argmax())], ETA)

    def read_day(U, toks, keep=None):
        """Stage 41 word reading; optionally remember each capital sighting (x, evidence, context prediction)"""
        for tok in toks:
            X = C.feats(tok, V["train"]); ev = U.evidence(X)
            for t in range(len(ev)):
                prior = P.predict(C.context2(ev, t))
                e_t = (1 - LAPSE) * ev[t] + LAPSE / L
                p = combine(e_t, prior, BETA_T); k = int(p.argmax())
                if tok["case"][t] and keep is not None:
                    keep.append((X[t], prior, tok["let"][t]))
                if p[k] < CONF:
                    continue
                U.learn(X[t], np.eye(L)[k], ETA)
                if tok["case"][t]:
                    replay_lower(U)

    def sleep(U, mem, down=False):
        if down:                                     # synaptic downscaling of what was learned since lowercase
            U.W = m0.coef_ + DOWN * (U.W - m0.coef_); U.b = m0.intercept_ + DOWN * (U.b - m0.intercept_)
        Xm = np.array([m[0] for m in mem]); Pm = np.array([m[1] for m in mem])
        Z = Xm / (np.linalg.norm(Xm, axis=1, keepdims=True) + 1e-9)
        ok = []
        for i in np.random.default_rng(800 + a.seed).permutation(len(mem)):
            sims = Z @ Z[i]; nb = np.argpartition(-sims, K_NEIGH)[:K_NEIGH]
            pooled = Pm[nb].mean(0)                        # what the words around similar shapes said
            ev = U.evidence(Xm[i][None])[0]
            p = combine((1 - LAPSE) * ev + LAPSE / L, pooled, BETA_T * 2.0); k = int(p.argmax())
            ok.append(k == mem[i][2])
            U.learn(Xm[i], np.eye(L)[k], ETA)
            replay_lower(U)
        return float(np.mean(ok))

    def report(U):
        cap, pred, yt = C.isolated(U, V["test"], y["test"], True)
        lo = C.isolated(U, V["test"], y["test"], False)[0]
        pc = {ABC.upper()[c]: round(float(np.mean(pred[yt == c] == c)), 3) for c in range(26)}
        tests = {st: C.sample_words(np.random.default_rng(300 + a.seed), tr, freqs[tr], 400, nf["test"], words, st) for st in ("Capital", "ALLCAPS")}
        w = {st: round(C.read_words(t, V["test"], U, P, beta)[1], 4) for st, t in tests.items()}
        return dict(capitals_alone=round(cap, 4), lowercase_alone=round(lo, 4), capitals_in_words=w, per_capital=pc,
                    stuck=[(c, pc[c]) for c in "GPYVIJQ"])

    if a.diagnose:
        U = C.Units(m0, xn2); mem = []
        read_day(U, C.sample_words(np.random.default_rng(1000 + 10 * a.seed), tr, freqs[tr], a.per_day, nf["train"], words, "mix"), mem)
        Xm = np.array([m[0] for m in mem]); Pm = np.array([m[1] for m in mem]); Y = np.array([m[2] for m in mem])
        Z = Xm / (np.linalg.norm(Xm, axis=1, keepdims=True) + 1e-9); E = U.evidence(Xm)
        own = E.argmax(1); single = Pm.argmax(1); pooled = []
        for i in range(len(mem)):
            nb = np.argpartition(-(Z @ Z[i]), K_NEIGH)[:K_NEIGH]; pooled.append(Pm[nb].mean(0))
        pooled = np.array(pooled); pa = pooled.argmax(1)
        for c in "GPYVIQ":
            k = ABC.index(c.lower()); m = Y == k
            if m.sum():
                top = np.bincount(pa[m], minlength=26).argmax()
                print(c, "sightings", int(m.sum()), "| units alone", round(float(np.mean(own[m] == k)), 2), "| context of one word", round(float(np.mean(single[m] == k)), 2),
                      "| pooled context", round(float(np.mean(pa[m] == k)), 2), "(most often says", ABC[top] + ")", flush=True)
        print("all capitals: units", round(float(np.mean(own == Y)), 3), "one word's context", round(float(np.mean(single == Y)), 3), "pooled", round(float(np.mean(pa == Y)), 3))
        return
    days = [C.sample_words(np.random.default_rng(1000 + 10 * a.seed + d), tr, freqs[tr], a.per_day, nf["train"], words, "mix") for d in range(a.days)]
    res = dict(seed=a.seed, days=a.days, per_day=a.per_day)
    res["before"] = report(C.Units(m0, xn2))
    for cond in ("awake", "awake_reread", "sleep", "sleep_down"):
        U = C.Units(m0, xn2); log = []
        for d in range(a.days):
            mem = []
            read_day(U, days[d], mem)
            if cond == "awake_reread":
                read_day(U, days[d])
            elif cond in ("sleep", "sleep_down"):
                log.append(sleep(U, mem, cond == "sleep_down"))
        res[cond] = dict(report(U), night_lessons_right=[round(x, 3) for x in log])
        print(cond, res[cond]["capitals_alone"], res[cond]["lowercase_alone"], res[cond]["capitals_in_words"], res[cond]["stuck"], log, flush=True)
    json.dump(res, open(os.path.join(OUT, f"s{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
