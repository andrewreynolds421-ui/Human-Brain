#!/usr/bin/env python3
"""Stage 36: reading words letter by letter with a sequence layer and a prediction rule (synth/words.py).

Fonts: the 89 training fonts are split once into readout fonts (70: letter units, sequence learning)
and validation fonts (19: fitting the prediction's weight); the test uses the 22 unseen fonts.
Words: 5,000 most frequent English words (3-7 letters); 20 % of the types are held out (per seed).
All word streams, test included, are sampled in proportion to word frequency, as in running text.

    python scripts/word_reading.py prepare                       # letter units + evidence for every image
    python scripts/word_reading.py run --seed 1 --out results/words/s1          # spiking layer (resumable)
    python scripts/word_reading.py run --seed 1 --shuffle 1 --out results/words/s1_shuf   # control
    python scripts/word_reading.py record --inhib 0.1 --out results/words/s1   # re-read with weaker inhibition
    python scripts/word_reading.py analyse results/words/s1 [results/words/s2 ...] [--inhib 0.1]
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from synth.words import (LETTERS, L, PER_FONT, ROOT, LetterBank, LetterReadout, NGram, Predictor, combine,  # noqa
                         context_groups, context_vector, encode, load_words, split_words)

WD = os.path.join(ROOT, "results", "words")
EVID = os.path.join(WD, "evidence.npz")
N_READOUT_FONTS = 70


def font_split():
    perm = np.random.default_rng(0).permutation(89)
    return np.sort(perm[:N_READOUT_FONTS]), np.sort(perm[N_READOUT_FONTS:])


# ------------------------------------------------------------------------------------- prepare
def prepare(a):
    bank = LetterBank.load()
    ro, va = font_split()
    rows, y, _ = bank.rows("train", fonts=ro)
    vrows, vy, _ = bank.rows("train", fonts=va)
    best = None
    for C in (0.01, 0.03, 0.1, 0.3):
        R = LetterReadout(C=C).fit(bank.F["train"][rows], y)
        acc = float((R.evidence(bank.F["train"][vrows]).argmax(1) == vy).mean())
        print(f"readout C={C}: validation-font letter accuracy {acc:.3f}", flush=True)
        if best is None or acc > best[0]:
            best = (acc, C, R)
    acc, C, R = best
    trows, ty, _ = bank.rows("test")
    E_tr, E_te = R.evidence(bank.F["train"]), R.evidence(bank.F["test"])
    test_acc = float((E_te.argmax(1) == ty).mean())
    np.savez(EVID, E_train=E_tr, E_test=E_te, C=C, val_acc=acc, test_acc=test_acc, readout_fonts=ro, val_fonts=va)
    print(json.dumps(dict(C=C, validation_font_accuracy=round(acc, 4), unseen_font_accuracy=round(test_acc, 4))))


# ------------------------------------------------------------------------------------ sequences
def sample_set(rng, word_ids, n, fonts, weights=None, shuffle=None, words=None):
    """n word tokens: (word id, font, rendering per letter, letter order).  shuffle: None, or a separate
    random generator that permutes each token's letters (so the control draws the same words, fonts
    and renderings as the normal run and its test sets are identical)"""
    p = None if weights is None else weights / weights.sum()
    ids = rng.choice(word_ids, n, replace=True, p=p) if weights is not None else rng.choice(word_ids, n, replace=n > len(word_ids))
    out = []
    for w in ids:
        letters = encode(words[w])
        if shuffle is not None:
            letters = shuffle.permutation(letters)
        out.append(dict(w=int(w), font=int(rng.choice(fonts)), let=letters, r=rng.integers(0, PER_FONT, len(letters))))
    return out


def evidence_of(tok, E, split_n_fonts=None):
    k = tok["font"] * L * PER_FONT + tok["let"] * PER_FONT + tok["r"]
    return E[k]


def contexts(ev, decay):
    """sequence-layer input groups before each letter t (predicting letter t), from bottom-up evidence"""
    T = len(ev); out = []
    trace = np.zeros(L)
    for t in range(T):
        if t == 0:
            out.append(context_groups(np.zeros(L), np.zeros(L), True))
        else:
            out.append(context_groups(ev[t - 1], trace, False))
            trace = decay * (trace + ev[t - 1])
    return out


def build_sets(seed, shuffle, a):
    words, freqs = load_words(a.n_words)
    tr, ho = split_words(words, a.held_frac, seed)
    ro, va = font_split()
    rng = np.random.default_rng(1000 + seed)
    shuffle = np.random.default_rng(5000 + seed) if shuffle else None
    sets = dict(
        learn=sample_set(rng, tr, a.learn_tokens, ro, freqs[tr], shuffle, words),
        pred=sample_set(rng, tr, a.pred_tokens, ro, freqs[tr], shuffle, words),
        val=sample_set(rng, tr, a.val_words, va, freqs[tr], None, words),
        known=sample_set(rng, tr, a.test_words, np.arange(22), freqs[tr], None, words),
        held=sample_set(rng, ho, a.test_words, np.arange(22), freqs[ho], None, words))
    return words, freqs, tr, ho, sets


SPLIT_OF = dict(learn="train", pred="train", val="train", known="test", held="test")


# ----------------------------------------------------------------------------------- spiking run
def run(a):
    from synth.sensory import SensoryLayer
    os.makedirs(a.out, exist_ok=True)
    ev = np.load(EVID)
    E = {"train": ev["E_train"], "test": ev["E_test"]}
    words, freqs, tr, ho, sets = build_sets(a.seed, bool(a.shuffle), a)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"))
    if a.layer == "none":                      # prototype: prediction straight from the context, no spiking layer
        open(os.path.join(a.out, "DONE"), "w").write("ok\n"); return
    ck, layer_f = os.path.join(a.out, "progress.json"), os.path.join(a.out, "layer.npz")
    prog = json.load(open(ck)) if os.path.exists(ck) else dict(learned=0, recorded={})
    if os.path.exists(layer_f):
        S = SensoryLayer.load(layer_f)
    else:
        S = SensoryLayer(n_in=(2 * L + 1) * a.m, n_e=a.n_cells, seed=a.seed)
    # list of all context steps of the learning stream
    steps = [c for tok in sets["learn"] for c in contexts(evidence_of(tok, E["train"]), a.decay)]
    t0 = time.time()
    while prog["learned"] < len(steps):
        k = prog["learned"]
        x = np.repeat(steps[k], a.m)
        S.present(x, learn=True)
        prog["learned"] = k + 1
        if (k + 1) % a.save_every == 0 or k + 1 == len(steps):
            S.save(layer_f + ".tmp.npz"); os.replace(layer_f + ".tmp.npz", layer_f)
            json.dump(prog, open(ck + ".tmp", "w")); os.replace(ck + ".tmp", ck)
            print(f"learned {k + 1}/{len(steps)} steps ({time.time() - t0:.0f} s)", flush=True)
    for name in ("pred", "val", "known", "held"):
        f = os.path.join(a.out, f"counts_{name}.npz")
        toks = sets[name]
        nsteps = sum(len(t["let"]) for t in toks)
        C = np.zeros((nsteps, a.n_cells), np.int16); done = 0
        if os.path.exists(f):
            d = np.load(f); done = int(d["done"]); C[:done] = d["C"][:done]
        allc = [c for tok in toks for c in contexts(evidence_of(tok, E[SPLIT_OF[name]]), a.decay)]
        for k in range(done, nsteps):
            C[k] = S.features(np.repeat(allc[k], a.m))
            if (k + 1) % a.save_every == 0 or k + 1 == nsteps:
                np.savez(f + ".tmp.npz", C=C, done=k + 1); os.replace(f + ".tmp.npz", f)
                print(f"recorded {name} {k + 1}/{nsteps} ({time.time() - t0:.0f} s)", flush=True)
    open(os.path.join(a.out, "DONE"), "w").write("ok\n")


def record(a):
    """re-read a trained sequence layer (frozen) with lateral inhibition x a.inhib: counts_<set>_i<inhib>.npz"""
    from synth.sensory import SensoryLayer
    ra = argparse.Namespace(**json.load(open(os.path.join(a.out, "args.json"))))
    ev = np.load(EVID)
    E = {"train": ev["E_train"], "test": ev["E_test"]}
    words, freqs, tr, ho, sets = build_sets(ra.seed, bool(ra.shuffle), ra)
    S = SensoryLayer.load(os.path.join(a.out, "layer.npz"))
    t0 = time.time()
    for name in ("pred", "val", "known", "held"):
        f = os.path.join(a.out, f"counts_{name}_i{a.inhib:g}.npz")
        toks = sets[name]
        allc = [c for tok in toks for c in contexts(evidence_of(tok, E[SPLIT_OF[name]]), ra.decay)]
        C = np.zeros((len(allc), ra.n_cells), np.int16); done = 0
        if os.path.exists(f):
            d = np.load(f); done = int(d["done"]); C[:done] = d["C"][:done]
        for k in range(done, len(allc)):
            C[k] = S.features(np.repeat(allc[k], ra.m), inhib=a.inhib)
            if (k + 1) % a.save_every == 0 or k + 1 == len(allc):
                np.savez(f + ".tmp.npz", C=C, done=k + 1); os.replace(f + ".tmp.npz", f)
                print(f"recorded {name} {k + 1}/{len(allc)} at inhibition x{a.inhib:g} ({time.time() - t0:.0f} s)", flush=True)
    open(os.path.join(a.out, f"DONE_i{a.inhib:g}"), "w").write("ok\n")


# -------------------------------------------------------------------------------------- analysis
def predictor_inputs(name, run_dir, toks, E, a, kind):
    """what the prediction synapses see: 'cells' = sequence-layer spike counts; 'context' = the context
    groups directly; 'both' = both pathways (each normalised to sum 1)"""
    ctx = lambda: np.array([c for tok in toks for c in contexts(evidence_of(tok, E), a.decay)])
    suffix = "" if getattr(a, "inhib", 1.0) == 1.0 else f"_i{a.inhib:g}"
    cells = lambda: np.load(os.path.join(run_dir, f"counts_{name}{suffix}.npz"))["C"].astype(float)
    if kind == "cells":
        return cells()
    if kind == "context":
        return ctx()
    norm = lambda X: X / np.maximum(X.sum(1, keepdims=True), 1e-12)
    return np.hstack([norm(ctx()), norm(cells())])


def train_predictor(X, toks, E, eta, teacher="percept"):
    """one pass of the prediction rule over the stream.  teacher = what the prediction is compared with
    when the letter arrives:  'percept' (default) - the winning letter unit after combining evidence and
    prediction (what was seen; no labels);  'evidence' - the raw bottom-up evidence (Stage 36 diagnosis:
    learns predictions smeared over look-alike letters);  'label' - the true letter (reference only)."""
    P = Predictor(X.shape[1], eta=eta)
    Ev = np.concatenate([evidence_of(t, E) for t in toks])
    Y = np.concatenate([t["let"] for t in toks])
    for x, e, y in zip(X, Ev, Y):
        if teacher == "percept":
            tgt = np.eye(L)[combine(e, P.predict(x), 1.0).argmax()]
        elif teacher == "evidence":
            tgt = e
        else:
            tgt = np.eye(L)[y]
        P.learn(x, tgt)
    return P


def read(toks, E, priors, beta):
    """letter and word accuracy with prediction weight beta; priors: (steps x 26) predictions"""
    k = 0; lc = lt = wc = 0; pos = {}
    for tok in toks:
        ev = evidence_of(tok, E); ok_all = True
        for t in range(len(ev)):
            p = combine(ev[t], priors[k], beta) if beta > 0 else ev[t]
            ok = int(p.argmax() == tok["let"][t]); lc += ok; lt += 1; ok_all &= bool(ok)
            pos.setdefault(t, [0, 0]); pos[t][0] += ok; pos[t][1] += 1
            k += 1
        wc += ok_all
    return dict(letter=lc / lt, word=wc / len(toks), by_position=[round(pos[t][0] / pos[t][1], 3) for t in sorted(pos)])


def next_letter(toks, priors):
    y = np.concatenate([t["let"] for t in toks])
    return dict(top1=float((priors.argmax(1) == y).mean()),
                bits=float(-np.mean(np.log2(np.maximum(priors[np.arange(len(y)), y], 1e-12)))))


def analyse_run(run_dir, a, betas=(0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):
    ra = argparse.Namespace(**json.load(open(os.path.join(run_dir, "args.json"))))
    ra.inhib = a.inhib
    ev = np.load(EVID)
    E = {"train": ev["E_train"], "test": ev["E_test"]}
    words, freqs, tr, ho, sets = build_sets(ra.seed, bool(ra.shuffle), ra)
    res = dict(run=run_dir, seed=ra.seed, shuffle=bool(ra.shuffle), n_cells=ra.n_cells, inhib=a.inhib)
    kinds = ["cells", "context"] if os.path.exists(os.path.join(run_dir, "counts_pred.npz")) else ["context"]
    plan = [(k, "percept") for k in kinds] + [("context", "evidence"), ("context", "label")]
    if "cells" in kinds:
        plan += [("both", "percept"), ("cells", "label"), ("both", "label")]
    for kind, teacher in plan:
        Xp = predictor_inputs("pred", run_dir, sets["pred"], E["train"], ra, kind)
        best = None
        for eta in (0.03, 0.1, 0.3):
            P = train_predictor(Xp, sets["pred"], E["train"], eta, teacher)
            Xv = predictor_inputs("val", run_dir, sets["val"], E["train"], ra, kind)
            pv = np.array([P.predict(x) for x in Xv])
            for b in betas:
                acc = read(sets["val"], E["train"], pv, b)["letter"]
                if best is None or acc > best[0]:
                    best = (acc, eta, b, P)
        _, eta, beta, P = best
        out = dict(eta=eta, beta=beta)
        for name in ("known", "held"):
            X = predictor_inputs(name, run_dir, sets[name], E["test"], ra, kind)
            pr = np.array([P.predict(x) for x in X])
            out[name] = dict(alone=read(sets[name], E["test"], pr, 0), with_prediction=read(sets[name], E["test"], pr, beta),
                             next_letter=next_letter(sets[name], pr))
        key = {"cells": "sequence_layer", "context": "context_only", "both": "context_and_sequence_layer"}[kind] + ("" if teacher == "percept" else f"_{teacher}_teacher")
        res[key] = out
    # ideal readers: n-gram statistics of the training words (shuffled if the run is the control)
    tw = [words[i] for i in tr]
    if ra.shuffle:
        r = np.random.default_rng(7); tw = ["".join(r.permutation(list(w))) for w in tw]
    for n in (1, 2, 3):
        G = NGram(tw, freqs[tr], n=n)
        best = None
        for b in betas[1:]:
            acc = np.mean([(G.filter(evidence_of(t, E["train"]), b)[0].argmax(1) == t["let"]).mean() for t in sets["val"]])
            if best is None or acc > best[0]:
                best = (acc, b)
        out = dict(beta=best[1])
        for name in ("known", "held"):
            post, pred = zip(*[G.filter(evidence_of(t, E["test"]), best[1]) for t in sets[name]])
            lc = np.mean(np.concatenate([p.argmax(1) == t["let"] for p, t in zip(post, sets[name])]))
            wc = np.mean([np.all(p.argmax(1) == t["let"]) for p, t in zip(post, sets[name])])
            out[name] = dict(letter=float(lc), word=float(wc), next_letter=next_letter(sets[name], np.concatenate(pred)))
        res[f"ideal_{n}gram"] = out
    return res


def analyse(a):
    rows = [analyse_run(d, a) for d in a.runs]
    print(json.dumps(rows, indent=1))
    if a.save:
        json.dump(rows, open(a.save, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prepare", "run", "record", "analyse"))
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shuffle", type=int, default=0)
    ap.add_argument("--n-words", type=int, default=5000)
    ap.add_argument("--held-frac", type=float, default=0.2)
    ap.add_argument("--learn-tokens", type=int, default=2000)
    ap.add_argument("--pred-tokens", type=int, default=2000)
    ap.add_argument("--val-words", type=int, default=200)
    ap.add_argument("--test-words", type=int, default=400)
    ap.add_argument("--n-cells", type=int, default=400)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--layer", choices=("spiking", "none"), default="spiking")
    ap.add_argument("--inhib", type=float, default=1.0, help="record/analyse: lateral inhibition factor of the frozen sequence layer")
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    {"prepare": prepare, "run": run, "record": record, "analyse": analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
