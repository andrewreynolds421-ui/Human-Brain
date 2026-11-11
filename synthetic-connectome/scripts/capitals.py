#!/usr/bin/env python3
"""Stage 40: capital letters and how they relate to lowercase.

Images: data/letters_case48.npz (scripts/make_letters_line.py --case both): a-z and A-Z of 299 font
families on one text line and one scale per font, 48 x 48; classes 0-25 lowercase, 26-51 capitals; family
split 209 train / 30 validation / 60 test as in Stage 38.  Letter units read the V1-like complex cells
(synth/v1.py, dense output, 1,152 cells) directly.

    python scripts/capitals.py features          # edge-cell responses for every image (cached)
    python scripts/capitals.py basics            # parts A and B  -> results/capitals/basics.json
    python scripts/capitals.py learn --seed 1    # part C         -> results/capitals/learn_s1.json

A. Capitals alone, lowercase alone, all 52 shapes, and letter identity regardless of case (labelled
   readouts; C chosen on validation families).
B. What carries over for free: letter units taught lowercase only, shown capitals.
C. Learning capitals WITHOUT labels, from words.  Letter units know lowercase (taught with labels, as
   before).  A word-context prediction (delta rule taught by the percept, Stage 36; here it uses the
   letters on BOTH sides) is learned from reading lowercase words.  Then words appear with capitals -
   "The" or "THE" (ASSUMED mix 70 % / 30 %) - and each capital's percept (evidence x prediction^beta)
   teaches the letter units' synapses from the edge cells (delta rule; no labels).  Every letter of
   the word teaches, so lowercase stays practised.
   Alphabet-book pairing (also no labels): a capital shown together with its lowercase letter in the same
   font; the unit that reads the lowercase letter teaches the capital's shape.  Alone, and followed by words.
   Controls: prediction learned from shuffled-order words (no sequence knowledge); no prediction (the
   units teach themselves from their own guess).  Reference: capitals taught with labels.
"""
import argparse, json, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from synth.words import L, Predictor, combine, encode, load_words, split_words   # noqa: E402

DATA = os.path.join(ROOT, "data", "letters_case48.npz")
CACHE = os.path.join(ROOT, "data", "letters_case48_v1.npz")
OUT = os.path.join(ROOT, "results", "capitals")
PER, NCLS = 6, 52
ABC = "abcdefghijklmnopqrstuvwxyz"


def features(a=None):
    from synth.v1 import complex_cells
    d = np.load(DATA)
    np.savez(CACHE, **{f"V_{k}": complex_cells(d[f"X_{k}"] / 255.0).astype(np.float32) for k in ("train", "val", "test")})


def load():
    d = np.load(DATA); v = np.load(CACHE)
    V = {k: v[f"V_{k}"] for k in ("train", "val", "test")}
    y = {k: d[f"y_{k}"] for k in V}
    return V, y


def fit(F, y, Fv, yv, Cs=(0.03, 0.1, 0.3)):
    from sklearn.linear_model import LogisticRegression
    best = None
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=3000).fit(F, y)
        acc = m.score(Fv, yv)
        if best is None or acc > best[0]:
            best = (acc, C, m)
    return best[2], best[1]


def per_class(pred, y, names):
    return {n: round(float(np.mean(pred[y == c] == c)), 3) for c, n in zip(np.unique(y), names)}


def confusions(pred, y, names, k=10):
    cl = list(np.unique(y)); out = []
    for i, ci in enumerate(cl):
        for j, cj in enumerate(cl):
            if i != j:
                out.append((int(np.sum((y == ci) & (pred == cj))), f"{names[i]}->{names[j]}"))
    return [f"{n}: {c}" for c, n in sorted(out, reverse=True)[:k]]


# ----------------------------------------------------------------------------------- parts A and B
def basics(a):
    os.makedirs(OUT, exist_ok=True)
    V, y = load()
    rng = np.random.default_rng(0)
    res = {}
    low = {k: y[k] < 26 for k in y}; up = {k: y[k] >= 26 for k in y}
    names52 = list(ABC) + list(ABC.upper())
    # A: four labelled readouts
    sets = dict(lowercase_alone=(low, lambda t: t, list(ABC)), capitals_alone=(up, lambda t: t - 26, list(ABC.upper())),
                all_52_shapes=({k: np.ones(len(y[k]), bool) for k in y}, lambda t: t, names52),
                letter_identity_either_case=({k: np.ones(len(y[k]), bool) for k in y}, lambda t: t % 26, list(ABC)))
    models = {}
    for name, (mask, lab, names) in sets.items():
        sub = np.nonzero(mask["train"])[0]
        sub = rng.choice(sub, min(len(sub), a.n_label), replace=False)
        m, C = fit(V["train"][sub], lab(y["train"][sub]), V["val"][mask["val"]], lab(y["val"][mask["val"]]))
        yt = lab(y["test"][mask["test"]]); pred = m.predict(V["test"][mask["test"]])
        r = dict(C=C, labelled_images=int(len(sub)), accuracy=round(float(np.mean(pred == yt)), 4),
                 hardest=sorted(per_class(pred, yt, names).items(), key=lambda t: t[1])[:8],
                 top_confusions=confusions(pred, yt, names))
        if name == "all_52_shapes":
            r["letter_right_case_wrong"] = round(float(np.mean((pred % 26 == yt % 26) & (pred != yt))), 4)
            r["letter_right"] = round(float(np.mean(pred % 26 == yt % 26)), 4)
            cw = {ABC[c]: round(float(np.mean((pred[yt % 26 == c] % 26 == c) & (pred[yt % 26 == c] != yt[yt % 26 == c]))), 3) for c in range(26)}
            r["case_mixed_up_most"] = sorted(cw.items(), key=lambda t: -t[1])[:10]
        if name == "letter_identity_either_case":
            isup = y["test"] >= 26
            r["on_lowercase"] = round(float(np.mean(pred[~isup] == yt[~isup])), 4)
            r["on_capitals"] = round(float(np.mean(pred[isup] == yt[isup])), 4)
        res[name] = r; models[name] = m
        print(name, r["accuracy"], flush=True)
    # B: lowercase-taught units shown capitals
    m = models["lowercase_alone"]
    yt = y["test"][up["test"]] - 26; pred = m.predict(V["test"][up["test"]])
    pc = per_class(pred, yt, list(ABC.upper()))
    res["free_transfer_lowercase_units_on_capitals"] = dict(
        accuracy=round(float(np.mean(pred == yt)), 4), per_capital=pc,
        read_as={ABC.upper()[c]: ABC[int(np.bincount(pred[yt == c], minlength=26).argmax())] for c in range(26)})
    print("free transfer", res["free_transfer_lowercase_units_on_capitals"]["accuracy"], flush=True)
    json.dump(res, open(os.path.join(OUT, "basics.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------ part C
class Units:
    """26 letter units reading the edge cells: softmax(W x + b); delta-rule updates"""

    def __init__(self, model, xnorm2=1.0):
        self.W = model.coef_.copy(); self.b = model.intercept_.copy()
        self.xnorm2 = xnorm2                      # mean squared input strength: the step is eta / xnorm2

    def evidence(self, X):
        z = X @ self.W.T + self.b
        z -= z.max(-1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(-1, keepdims=True)

    def learn(self, x, target, eta):
        err = target - self.evidence(x[None])[0]
        self.W += (eta / self.xnorm2) * np.outer(err, x); self.b += (eta / self.xnorm2) * err


def img_index(fam, cls, r):
    return fam * NCLS * PER + cls * PER + r


def context2(ev, t, decay=0.5):
    """word context of position t from the letters on both sides:
    [previous letter | next letter | fading trace before | fading trace after | first | last]"""
    T = len(ev); z = np.zeros(L)
    prev = ev[t - 1] if t > 0 else z
    nxt = ev[t + 1] if t < T - 1 else z
    tb = z.copy()
    for k in range(0, t - 1):                       # letters before the previous one
        tb = decay * (tb + ev[k])
    ta = z.copy()
    for k in range(T - 1, t + 1, -1):               # letters after the next one
        ta = decay * (ta + ev[k])
    return np.concatenate([prev, nxt, np.clip(tb, 0, 1), np.clip(ta, 0, 1), [float(t == 0), float(t == T - 1)]])


def sample_words(rng, ids, freqs, n, n_fam, words, style):
    """tokens: letters, family, class per letter (case), rendering.  style: lower / Capital / ALLCAPS / mix"""
    p = freqs / freqs.sum(); toks = []
    for w in rng.choice(ids, n, p=p):
        let = encode(words[w]); T = len(let)
        st = style if style != "mix" else ("Capital" if rng.random() < 0.7 else "ALLCAPS")
        case = np.zeros(T, int)
        if st == "Capital":
            case[0] = 1
        elif st == "ALLCAPS":
            case[:] = 1
        toks.append(dict(let=let, cls=let + 26 * case, case=case, fam=int(rng.integers(n_fam)), r=rng.integers(0, PER, T)))
    return toks


def feats(tok, V):
    return V[img_index(tok["fam"], tok["cls"], tok["r"])]


def train_predictor(toks, V, U, eta, shuffle_rng=None):
    P = Predictor(4 * L + 2, eta=eta, scale="none")
    for tok in toks:
        ev = U.evidence(feats(tok, V))
        if shuffle_rng is not None:
            ev = ev[shuffle_rng.permutation(len(ev))]
        for t in range(len(ev)):
            x = context2(ev, t)
            P.learn(x, np.eye(L)[combine(ev[t], P.predict(x), 1.0).argmax()])
    return P


def read_words(toks, V, U, P, beta):
    """letter accuracy with the word-context prediction: all letters, and the capitals among them"""
    ok, okc = [], []
    for tok in toks:
        ev = U.evidence(feats(tok, V))
        for t in range(len(ev)):
            p = combine(ev[t], P.predict(context2(ev, t)), beta) if (P is not None and beta > 0) else ev[t]
            hit = p.argmax() == tok["let"][t]
            ok.append(hit)
            if tok["case"][t]:
                okc.append(hit)
    return float(np.mean(ok)), (float(np.mean(okc)) if okc else None)


def isolated(U, V, y, case):
    m = (y >= 26) if case else (y < 26)
    pred = U.evidence(V[m]).argmax(1)
    return float(np.mean(pred == y[m] % 26)), pred, y[m] % 26


def learn(a):
    os.makedirs(OUT, exist_ok=True)
    V, y = load()
    rng = np.random.default_rng(a.seed)
    words, freqs = load_words(5000)
    tr, ho = split_words(words, 0.2, a.seed)
    nf = {k: len(V[k]) // (NCLS * PER) for k in V}
    # letter units taught lowercase with labels
    low = np.nonzero(y["train"] < 26)[0]; sub = rng.choice(low, a.n_label, replace=False)
    # each edge cell's signal is standardised (its usual level and spread over lowercase letters removed -
    # adaptation); without this the delta rule is badly conditioned (a large part common to every image)
    mu = V["train"][sub].mean(0); sd = V["train"][sub].std(0) + 1e-3
    V = {k: ((V[k] - mu) / sd).astype(np.float32) for k in V}
    m0, C0 = fit(V["train"][sub], y["train"][sub], V["val"][y["val"] < 26], y["val"][y["val"] < 26], Cs=(0.003, 0.01, 0.03))
    # word-context prediction from reading lowercase words
    lw = sample_words(rng, tr, freqs[tr], a.n_read, nf["train"], words, "lower")
    xn2 = float(np.mean(np.sum(V["train"][sub] ** 2, 1)))
    U0 = Units(m0, xn2)
    P = train_predictor(lw, V["train"], U0, a.eta_p)
    Pshuf = train_predictor(lw, V["train"], U0, a.eta_p, shuffle_rng=np.random.default_rng(99 + a.seed))
    vw = sample_words(rng, tr, freqs[tr], 300, nf["val"], words, "lower")
    beta = max((read_words(vw, V["val"], U0, P, b)[0], b) for b in (0.5, 0.75, 1.0, 1.5))[1]
    res = dict(seed=a.seed, beta=beta, C_lowercase=C0, args=vars(a))
    tw = {n: sample_words(rng, ids, freqs[ids], 400, nf["test"], words, "lower") for n, ids in (("known", tr), ("new", ho))}
    res["lowercase_words"] = {n: dict(letters_alone=read_words(t, V["test"], U0, None, 0)[0],
                                      with_both_side_prediction=read_words(t, V["test"], U0, P, beta)[0]) for n, t in tw.items()}
    # words with capitals
    cw = sample_words(rng, tr, freqs[tr], a.n_caps, nf["train"], words, "mix")
    tests = {f"{st}_{n}": sample_words(rng, ids, freqs[ids], 300, nf["test"], words, st)
             for st in ("Capital", "ALLCAPS") for n, ids in (("known", tr), ("new", ho))}
    vcap = sample_words(rng, tr, freqs[tr], 300, nf["val"], words, "mix")

    def report(U, Pred):
        cap, pred, yt = isolated(U, V["test"], y["test"], True)
        out = dict(capitals_isolated=round(cap, 4), lowercase_isolated=round(isolated(U, V["test"], y["test"], False)[0], 4),
                   per_capital={ABC.upper()[c]: round(float(np.mean(pred[yt == c] == c)), 3) for c in range(26)})
        for k, t in tests.items():
            al, cp = read_words(t, V["test"], U, Pred, beta)
            out[f"words_{k}"] = dict(all_letters=round(al, 4), capitals=round(cp, 4))
        return out

    res["before"] = report(Units(m0, xn2), P)

    def word_phase(U, Pred, b, eta_w, labels=False, lapse=0.0, conf=0.0):
        """read the words with capitals; every letter's percept (or label) teaches the units"""
        log = []
        for tok in cw:
            X = feats(tok, V["train"]); ev = U.evidence(X)
            for t in range(len(ev)):
                if labels:
                    k = tok["let"][t]
                else:
                    # lapse: the units may be wrong even when certain, so the evidence never rules a letter out
                    e_t = (1.0 - lapse) * ev[t] + lapse / L
                    p = combine(e_t, Pred.predict(context2(ev, t)), b) if b > 0 else ev[t]
                    k = int(p.argmax())
                    if p[k] < conf:                 # an unsure percept does not teach
                        continue
                if tok["case"][t]:
                    log.append(k == tok["let"][t])
                U.learn(X[t], np.eye(L)[k], eta_w)
        return float(np.mean(log)), float(np.mean(log[-2000:]))

    pair_fams = np.random.default_rng(500 + a.seed).permutation(nf["train"])

    def pair_phase(U, k, eta_w, passes=5):
        """alphabet-book pairing, no labels: a capital is shown with its lowercase letter in the same font
        (k fonts); the unit that reads the lowercase letter teaches the capital's shape"""
        r = np.random.default_rng(900 + a.seed); ok = []
        for _ in range(passes):
            for f in pair_fams[:k]:
                for c in r.permutation(26):
                    xl = V["train"][img_index(f, c, r.integers(PER))]; xu = V["train"][img_index(f, c + 26, r.integers(PER))]
                    t = int(U.evidence(xl[None])[0].argmax()); ok.append(t == c)
                    U.learn(xu, np.eye(L)[t], eta_w)
                    U.learn(xl, np.eye(L)[t], eta_w)          # the lowercase letter of the pair is rehearsed too
        return float(np.mean(ok))

    vlow = y["val"] < 26; vup = ~vlow

    def val_score(U):                               # capitals (in words and alone) AND lowercase kept; validation families
        ev = U.evidence(V["val"]).argmax(1) == y["val"] % 26
        return read_words(vcap, V["val"], U, P, beta)[1] + float(ev[vup].mean()) + 2 * float(ev[vlow].mean())

    def best_of(make):
        best = None
        for setting in make["grid"]:
            U = Units(m0, xn2) if make.get("start") is None else make["start"]()
            info = make["run"](U, *setting)
            sc = val_score(U)
            if best is None or sc > best[0]:
                best = (sc, setting, U, info)
        return best[1:]

    def copy_units(U):
        C = Units(m0, xn2); C.W = U.W.copy(); C.b = U.b.copy(); return C

    grid_w = [(b, e, lp, cf) for b in a.beta_teach for e in a.eta_w for lp in a.lapse for cf in a.conf]
    plans = dict(
        word_context=dict(grid=grid_w, run=lambda U, b, e, lp, cf: word_phase(U, P, b, e, False, lp, cf)),
        shuffled_order_control=dict(grid=grid_w, run=lambda U, b, e, lp, cf: word_phase(U, Pshuf, b, e, False, lp, cf)),
        no_prediction_control=dict(grid=[(0.0, e, 0.0, cf) for e in a.eta_w for cf in a.conf], run=lambda U, b, e, lp, cf: word_phase(U, None, b, e, False, 0.0, cf)),
        taught_with_labels=dict(grid=[(0.0, e, 0.0, 0.0) for e in a.eta_w], run=lambda U, b, e, lp, cf: word_phase(U, None, b, e, True)))
    kept = {}
    for name, plan in plans.items():
        (b, e, lp, cf), U, info = best_of(plan)
        res[name] = dict(eta_w=e, beta_teach=b, lapse=lp, confidence_needed=cf, teacher_correct_overall=round(info[0], 4), teacher_correct_last_2000=round(info[1], 4), **report(U, P))
        print(name, res[name]["capitals_isolated"], res[name]["lowercase_isolated"], "teacher", info, (b, e, lp, cf), flush=True)
    for k in a.pairs:
        (e,), U, info = best_of(dict(grid=[(e,) for e in a.eta_pair], run=lambda U, e, k=k: pair_phase(U, k, e)))
        res[f"pairs_{k}_fonts"] = dict(eta_w=e, pairings=int(5 * k * 26), lowercase_read_correctly_in_pairs=round(info, 4), **report(U, P))
        print(f"pairs {k}", res[f"pairs_{k}_fonts"]["capitals_isolated"], res[f"pairs_{k}_fonts"]["lowercase_isolated"], flush=True)
        (b, e2, lp, cf), U2, info2 = best_of(dict(grid=grid_w, start=lambda U=U: copy_units(U), run=lambda U_, b, e, lp, cf: word_phase(U_, P, b, e, False, lp, cf)))
        res[f"pairs_{k}_fonts_then_words"] = dict(eta_w=e2, beta_teach=b, lapse=lp, confidence_needed=cf, teacher_correct_overall=round(info2[0], 4),
                                                   teacher_correct_last_2000=round(info2[1], 4), **report(U2, P))
        print(f"pairs {k} + words", res[f"pairs_{k}_fonts_then_words"]["capitals_isolated"], res[f"pairs_{k}_fonts_then_words"]["lowercase_isolated"], "teacher", info2, (b, e2, lp, cf), flush=True)
    json.dump(res, open(os.path.join(OUT, f"learn_s{a.seed}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("features", "basics", "learn"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-label", type=int, default=10000, help="labelled lowercase images for the letter units")
    ap.add_argument("--n-read", type=int, default=20000, help="lowercase words read to learn the word-context prediction")
    ap.add_argument("--n-caps", type=int, default=20000, help="words with capitals (no labels)")
    ap.add_argument("--eta-p", type=float, default=0.1)
    ap.add_argument("--eta-w", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--beta-teach", type=float, nargs="+", default=[1.0])
    ap.add_argument("--lapse", type=float, nargs="+", default=[0.01], help="chance the units are wrong when certain (teaching percept)")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.0, 0.9, 0.99], help="a percept teaches only above this confidence")
    ap.add_argument("--pairs", type=int, nargs="+", default=[1, 3, 10, 30], help="fonts of alphabet-book pairs (x 26 letters x 5 showings)")
    ap.add_argument("--eta-pair", type=float, nargs="+", default=[0.3, 1.0])
    a = ap.parse_args()
    {"features": features, "basics": basics, "learn": learn}[a.cmd](a)


if __name__ == "__main__":
    main()
