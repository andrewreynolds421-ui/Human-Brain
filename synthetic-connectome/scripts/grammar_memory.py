#!/usr/bin/env python3
"""Stage 51: a memory that carries through the sentence.

Stage 50's model predicted each word from the two before it, and learned word classes and local order but
none of the grammar that spans a sentence. Here the model keeps a running state - a recurrent layer whose
activity after each word depends on that word AND its own previous activity - and learns by predicting
the next word and correcting by the prediction error (as reading brains are thought to do: N400, mismatch
signals). Same 32 books, same split, text only.

  trigram  Stage 50's model (word classes + category / word transitions), for comparison
  srn      simple recurrent network (Elman 1990): one recurrent layer, 256 units
  gru      the same with GATES (Cho et al. 2014): learned signals decide, word by word, how much of the
           state to keep and how much to overwrite - the kind of gating of working memory attributed to
           prefrontal cortex and basal ganglia (Frank, Loughry & O'Reilly 2001; O'Reilly & Frank 2006)

MODEL: learning by backpropagation through time is an engineering stand-in for how cortex assigns credit
over time; it is not claimed to be biological. Sizes (128 input, 256 recurrent units) are ASSUMED; the
number of passes is chosen on the validation books (early stopping).

Tests (never-read books unless said):
  C  real sentence vs two neighbours swapped / shuffled (as Stage 50)
  D  BLiMP (all pairs; pairs whose words are all known)
  E  AGREEMENT across words: sentences from the test books with a noun subject and its present-tense verb
     (or was / were); the verb's number flipped ("the dogs in the yard *barks*"); does the model prefer
     the real one?  By the number of words between subject and verb, and with an "attractor" (a noun of the
     other number in between). Subjects / verbs found by spaCy's parser (grading only).
  F  SURPRISE at a word that does not fit (N400-like): a noun in a test sentence replaced by another noun of
     the same frequency; is the model more surprised by the substitute than by the original word, at that
     word, given only what came before?

    python scripts/grammar_memory.py --model gru --seed 0  ->  results/grammar_memory/gru_s0.json
"""
import argparse, glob, json, math, os, re, sys, time
from collections import Counter, defaultdict
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grammar as GR                                                            # noqa: E402
OUT = os.path.join(ROOT, "results", "grammar_memory")
MODELS = "/home/claude/models/grammar_memory"
FLIP = {"is": "are", "are": "is", "was": "were", "were": "was", "has": "have", "have": "has", "does": "do", "do": "does"}


def to_vbz(w):
    if re.search(r"(s|sh|ch|x|z|o)$", w): return w + "es"
    if re.search(r"[^aeiou]y$", w): return w[:-1] + "ies"
    return w + "s"


MORE_BOOKS = "/home/claude/corpora/books_more"


def corpus_big(C):
    """Stage 52: 141 more children's / young-adult books added to the TRAINING books only. Any sentence sharing a
    6-word run with a test or validation sentence is dropped (anthologies and readers reprint passages)."""
    def sh(x): return {tuple(x[i:i + 6]) for i in range(len(x) - 5)}
    held = set()
    for x in C["test"] + C["val"]: held |= sh(x)
    extra = []
    for d in sorted(glob.glob(os.path.join(MORE_BOOKS, "*"))):
        extra += [x for x in GR.sentences(GR.book_text(d)) if not (sh(x) & held)]
    return dict(C, train=C["train"] + extra)


# ---------------------------------------------------------------- test items (built once, cached)
def items(C, vocab):
    path = os.path.join(OUT, "test_items.json")
    if os.path.exists(path):
        return json.load(open(path))
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    test = C["test"]
    agree, nouns_at = [], []
    for si, doc in enumerate(nlp.pipe((" ".join(s) for s in test), batch_size=256)):
        s = test[si]
        if [t.text for t in doc] != s:                 # tokenisation must match ("don't" etc.): skip otherwise
            continue
        for c in doc:
            if c.dep_ != "nsubj" or c.tag_ not in ("NN", "NNS"):
                continue
            h = c.head
            auxs = [a for a in h.children if a.dep_ in ("aux", "auxpass") and a.i > c.i]
            v = auxs[0] if auxs else h
            if v.i <= c.i:
                continue
            w = v.text
            if w in FLIP:
                bad = FLIP[w]
            elif v.tag_ == "VBZ" and v.dep_ not in ("aux", "auxpass") and w.endswith("s"):
                bad = h.lemma_ if v is h else None
            elif v.tag_ == "VBP" and v.dep_ not in ("aux", "auxpass"):
                bad = to_vbz(w)
            else:
                bad = None
            plural = c.tag_ == "NNS"
            sing_verb = w in ("is", "was", "has", "does") or (v.tag_ == "VBZ" and w not in FLIP)
            if bad is None or bad not in vocab or plural == sing_verb:
                continue
            between = doc[c.i + 1:v.i]
            attr = any(t.tag_ in ("NN", "NNS") and (t.tag_ == "NNS") != plural for t in between)
            badsent = list(s); badsent[v.i] = bad
            agree.append(dict(good=s, bad=badsent, dist=v.i - c.i - 1, attractor=bool(attr), subj=c.text, verb=w))
        for t in doc:
            if t.tag_ == "NN" and t.i >= 3 and t.text in vocab:
                nouns_at.append((si, t.i))
    json.dump(dict(agree=agree, nouns_at=nouns_at), open(path, "w"))
    return dict(agree=agree, nouns_at=nouns_at)


def substitutions(C, nouns_at, tok, vocab, seed):
    """F: replace a noun by another noun of about the same frequency (rank within +-25 among nouns)"""
    rng = np.random.default_rng(50 + seed)
    nounset = sorted({C["test"][si][i] for si, i in nouns_at} | set(), key=lambda w: -tok[w])
    rank = {w: r for r, w in enumerate(nounset)}
    out = []
    for k in rng.choice(len(nouns_at), min(4000, len(nouns_at)), replace=False):
        si, i = nouns_at[k]; s = C["test"][si]; w = s[i]
        r = rank[w]; lo, hi = max(0, r - 25), min(len(nounset), r + 26)
        cand = [x for x in nounset[lo:hi] if x != w and x not in s]
        if cand:
            out.append((s, i, str(rng.choice(cand))))
    return out


def controlled(tok, vocab, seed=0):
    """E2: built sentences with the books' own words - 'the king of the birds was very angry', 'the boys that
    the girl saw were happy'. Subject singular or plural, the noun in between the same or the other number
    (attractor), PP or relative clause. The ungrammatical twin flips the verb."""
    rng = np.random.default_rng(900 + seed)
    pairs = [(w, w + "s") for w in sorted(vocab) if len(w) > 2 and w + "s" in vocab and tok[w] >= 30 and tok[w + "s"] >= 10]
    animals = {"dog", "cat", "bird", "girl", "boy", "king", "queen", "man", "rabbit", "fox", "wolf", "horse", "friend", "child", "soldier",
               "sister", "brother", "princess", "prince", "mouse", "witch", "frog", "duck", "owl", "bear", "monkey", "squirrel", "farmer", "teacher", "sailor", "neighbour", "pig", "cow", "goat"}
    an = [p for p in pairs if p[0] in animals]
    things = [p for p in pairs if p[0] not in animals]
    preps = [w for w in ("near", "behind", "beside", "with", "of", "by") if w in vocab]
    rel_v = [w for w in ("saw", "loved", "liked", "met", "followed", "found", "helped", "knew") if w in vocab]
    ends = [["very", "happy"], ["here"], ["very", "old"], ["in", "the", "garden"], ["quite", "still"], ["afraid"], ["there"]]
    out = []
    for _ in range(3000):
        kind = rng.choice(["pp", "rel"]); sp = bool(rng.integers(2)); ap_ = bool(rng.integers(2)); past = bool(rng.integers(2))
        a1 = an[rng.integers(len(an))]
        if kind == "pp":
            a2 = (things + an)[rng.integers(len(things) + len(an))]; mid = [str(rng.choice(preps)), "the", a2[ap_]]
        else:
            a2 = an[rng.integers(len(an))]; mid = ["that", "the", a2[ap_], str(rng.choice(rel_v))]
        if a2[0] == a1[0]:
            continue
        v = ("were" if sp else "was") if past else ("are" if sp else "is")
        good = ["the", a1[sp]] + mid + [v] + list(ends[rng.integers(len(ends))])
        bad = list(good); bad[2 + len(mid)] = FLIP[v]
        out.append(dict(good=good, bad=bad, kind=str(kind), subj_plural=sp, attractor=(ap_ != sp)))
    return out


# ---------------------------------------------------------------- trigram (Stage 50) scorer
class Trigram:
    def __init__(self, C, seed, K=80, mix=0.3):
        self.G = GR.Grammar(C["train"], K, seed=seed); self.mix = mix; self.vocab = self.G.vocab

    def logp(self, sents):
        return [self.G.score(s, self.mix) for s in sents]

    def word_logp(self, s, i):
        G = self.G
        cs = [G.cs, G.cs] + [G.c(w) for w in s]; ws = [GR.S, GR.S] + [w if w in G.vocab else GR.UNK for w in s]
        j = i + 2
        pc = G.p_cat(cs[j - 2], cs[j - 1], cs[j]); pe = (G.em[cs[j]][ws[j]] + 0.1) / (G.cn[cs[j]] + 0.1 * G.V)
        return math.log(self.mix * pc * pe + (1 - self.mix) * G.p_word(ws[j - 2], ws[j - 1], ws[j]))


# ---------------------------------------------------------------- recurrent networks
class Net:
    def __init__(self, kind, C, seed, epochs=8, hidden=256, emb=128, frac=1.0, tag="", min_count=3):
        import torch, torch.nn as nn
        torch.manual_seed(seed); torch.set_num_threads(int(os.environ.get("THREADS", "2")))
        self.torch = torch
        cnt = Counter(w for s in C["train"] for w in s)
        self.itos = ["<pad>", "<s>", "</s>", "<unk>"] + sorted(w for w, c in cnt.items() if c >= min_count)
        self.stoi = {w: i for i, w in enumerate(self.itos)}; self.vocab = set(self.itos[4:])
        V = len(self.itos)
        rnn = nn.RNN if kind == "srn" else nn.GRU

        class M(nn.Module):
            def __init__(s):
                super().__init__(); s.e = nn.Embedding(V, emb, padding_idx=0); s.r = rnn(emb, hidden, batch_first=True)
                s.d = nn.Dropout(0.25); s.o = nn.Linear(hidden, V)
            def forward(s, x):
                h, _ = s.r(s.d(s.e(x))); return s.o(s.d(h))
        self.m = M()
        os.makedirs(MODELS, exist_ok=True)
        path = os.path.join(MODELS, f"{kind}{tag}{'' if frac == 1.0 else f'_f{frac}'}_s{seed}.pt")
        self.log = []
        if os.path.exists(path) and os.path.exists(path + ".json"):
            self.m.load_state_dict(torch.load(path)); self.log = json.load(open(path + ".json"))
        else:
            if frac < 1.0:                                   # read only part of the books (random sentences)
                keep = np.random.default_rng(77 + seed).choice(len(C["train"]), int(frac * len(C["train"])), replace=False)
                C = dict(C, train=[C["train"][i] for i in sorted(keep)])
            self.train(C, seed, epochs, path)
        self.m.eval()

    def enc(self, s):
        return [1] + [self.stoi.get(w, 3) for w in s] + [2]

    def batches(self, sents, bs, rng=None):
        idx = np.argsort([len(s) for s in sents], kind="stable")
        groups = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        if rng is not None: rng.shuffle(groups)
        for g in groups:
            seqs = [self.enc(sents[i]) for i in g]; L = max(map(len, seqs))
            x = np.zeros((len(g), L), np.int64)
            for k, q in enumerate(seqs): x[k, :len(q)] = q
            yield g, self.torch.from_numpy(x)

    def train(self, C, seed, epochs, path):
        """resumable: a checkpoint every SAVE_MIN minutes, inside a pass too (the machine restarts about hourly);
        each pass's batch order is fixed by (seed, pass), so a resumed pass skips the batches already done"""
        torch = self.torch; F = torch.nn.functional
        opt = torch.optim.Adam(self.m.parameters(), lr=2e-3)
        best, bad, start, done_b, tot, n = 1e9, 0, 0, 0, 0.0, 0
        ck = path + ".ckpt"; save_s = 60 * float(os.environ.get("SAVE_MIN", "10"))
        if os.path.exists(ck):
            z = torch.load(ck, weights_only=False)
            self.m.load_state_dict(z["model"]); opt.load_state_dict(z["opt"])
            best, bad, start, self.log = z["best"], z["bad"], z["epoch"], z["log"]
            done_b, tot, n = z.get("batch", 0), z.get("tot", 0.0), z.get("n", 0)
            print("resumed at pass", start + 1, "batch", done_b, flush=True)

        def save(ep, b, tot, n):
            torch.save(dict(model=self.m.state_dict(), opt=opt.state_dict(), best=best, bad=bad, epoch=ep, log=self.log, batch=b, tot=tot, n=n), ck + ".tmp")
            os.replace(ck + ".tmp", ck)

        for ep in range(start, epochs):
            if bad >= 2: break
            self.m.train(); t0 = time.time(); last = time.time()
            for b, (_, x) in enumerate(self.batches(C["train"], 64, np.random.default_rng(1000 * seed + ep))):
                if b < done_b: continue
                out = self.m(x[:, :-1]); y = x[:, 1:]
                loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1), ignore_index=0)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0); opt.step()
                k = int((y > 0).sum()); tot += float(loss.detach()) * k; n += k
                if time.time() - last > save_s:
                    self.m.eval(); save(ep, b + 1, tot, n); self.m.train(); last = time.time()
            done_b = 0
            self.m.eval()
            val = -np.sum(self.logp(C["val"])) / sum(len(s) + 1 for s in C["val"])
            self.log.append(dict(epoch=ep + 1, train=round(tot / max(n, 1), 3), val=round(float(val), 3), minutes=round((time.time() - t0) / 60, 1)))
            print(self.log[-1], flush=True); tot, n = 0.0, 0
            if val < best - 1e-3:
                best, bad = val, 0; torch.save(self.m.state_dict(), path + ".best")
            else:
                bad += 1
            for g in opt.param_groups: g["lr"] *= 0.7
            save(ep + 1, 0, 0.0, 0)
        os.replace(path + ".best", path); json.dump(self.log, open(path + ".json", "w"))
        if os.path.exists(ck): os.remove(ck)
        self.m.load_state_dict(torch.load(path))

    def token_logp(self, sents):
        """per sentence, the list of log P(token_t | tokens before) for the words and </s>"""
        torch = self.torch; out = [None] * len(sents)
        with torch.no_grad():
            for g, x in self.batches(sents, 256):
                lp = torch.log_softmax(self.m(x[:, :-1]), -1).gather(2, x[:, 1:].unsqueeze(-1)).squeeze(-1).numpy()
                for k, i in enumerate(g):
                    out[i] = lp[k, :len(sents[i]) + 1]
        return out

    def logp(self, sents):
        return [float(v.sum()) for v in self.token_logp(sents)]


# ---------------------------------------------------------------- tests
def evaluate(M, C, tok, seed, res):
    """the Stage 51 test battery (C word order, D BLiMP, E / E2 agreement, F surprise) for any model with logp,
    token_logp and vocab"""
    vocab = {w for w, c in tok.items() if c >= 3}
    it = items(C, vocab)

    def pref(pairs):
        lg = M.logp([p[0] for p in pairs]); lb = M.logp([p[1] for p in pairs])
        return np.array(lg) > np.array(lb)

    # validation: log-prob per word
    res["val_logp_per_word"] = round(float(np.sum(M.logp(C["val"])) / sum(len(s) + 1 for s in C["val"])), 3)
    # C
    def swapped(s, r):
        i = int(r.integers(0, len(s) - 1)); t = list(s); t[i], t[i + 1] = t[i + 1], t[i]; return t
    tr_ = np.random.default_rng(2 + 10 * seed); test = [s for s in C["test"] if len(set(s)) > 3][:6000]; Cr = {}
    for name, make in (("two_neighbours_swapped", lambda s: swapped(s, tr_)), ("shuffled", lambda s: list(tr_.permutation(s)))):
        pairs = [(s, make(s)) for s in test]; pairs = [p for p in pairs if p[0] != p[1]]
        Cr[name] = round(float(pref(pairs).mean()), 4)
    res["C"] = Cr; print("C", Cr, flush=True)
    # D
    D = {}; allk = []; allk_small = []; small_vocab = {w for w, c in tok.items() if c >= 3}
    for f in sorted(glob.glob(os.path.join(GR.BLIMP, "*.jsonl"))):
        rows = [json.loads(l) for l in open(f)]
        P = [(re.findall(r"[a-z]+(?:'[a-z]+)?", x["sentence_good"].lower()), re.findall(r"[a-z]+(?:'[a-z]+)?", x["sentence_bad"].lower())) for x in rows]
        ok = pref(P); kn = np.array([all(w in M.vocab for w in g + b) for g, b in P])
        D[os.path.basename(f)[:-6]] = dict(field=rows[0].get("linguistics_term", "?"), all=round(float(ok.mean()), 3),
                                           known_pairs=int(kn.sum()), known=round(float(ok[kn].mean()), 3) if kn.sum() else None)
        allk += list(ok[kn]); ks = np.array([all(w in small_vocab for w in g + b) for g, b in P]); allk_small += list(ok[ks])
    res["D"] = dict(all=round(float(np.mean([v["all"] for v in D.values()])), 3), known_words_only=round(float(np.mean(allk)), 3),
                  known_to_stage51_vocab=round(float(np.mean(allk_small)), 3), by_phenomenon=D)
    print("D", res["D"]["all"], res["D"]["known_words_only"], flush=True)
    def at_verb(X):
        """log P(right verb | words before) - log P(wrong verb | words before), and the plural-minus-singular
        log-odds (positive = the model expects a plural verb)"""
        idx = [next(i for i, (g, b) in enumerate(zip(x["good"], x["bad"])) if g != b) for x in X]
        if isinstance(M, Trigram):
            lg = np.array([M.word_logp(x["good"], i) for x, i in zip(X, idx)]); lb = np.array([M.word_logp(x["bad"], i) for x, i in zip(X, idx)])
        else:
            tg = M.token_logp([x["good"] for x in X]); tb = M.token_logp([x["bad"] for x in X])
            lg = np.array([t[i] for t, i in zip(tg, idx)]); lb = np.array([t[i] for t, i in zip(tb, idx)])
        d = lg - lb
        plural_verb = np.array([x["good"][i] in ("are", "were", "have", "do") or (x["good"][i] not in FLIP and not x["good"][i].endswith("s")) for x, i in zip(X, idx)])
        return d > 0, np.where(plural_verb, d, -d), plural_verb

    def auc(pos, neg):
        if len(pos) == 0 or len(neg) == 0: return None
        r = np.argsort(np.argsort(np.r_[pos, neg])) + 1
        return round(float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)

    # E
    A = [x for x in it["agree"] if x["good"] != x["bad"]]; ok, lo_pl, plv = at_verb(A)
    dist = np.array([x["dist"] for x in A]); att = np.array([x["attractor"] for x in A])
    bins = {"0 (next word)": dist == 0, "1-2": (dist >= 1) & (dist <= 2), "3-5": (dist >= 3) & (dist <= 5), "6+": dist >= 6}
    res["E"] = dict(n=len(A), all=round(float(ok.mean()), 4),
                    by_distance={k: [round(float(ok[m].mean()), 4), int(m.sum())] for k, m in bins.items() if m.sum()},
                    with_attractor=[round(float(ok[att].mean()), 4), int(att.sum())], no_attractor_distance_1plus=[round(float(ok[~att & (dist >= 1)].mean()), 4), int((~att & (dist >= 1)).sum())],
                    sensitivity_auc=auc(lo_pl[plv], lo_pl[~plv]),
                    sensitivity_auc_distance_1plus=auc(lo_pl[plv & (dist >= 1)], lo_pl[~plv & (dist >= 1)]))
    print("E", res["E"], flush=True)
    E2 = controlled(tok, vocab); ok, lo_pl, plv = at_verb(E2); r2 = {}
    sp = np.array([x["subj_plural"] for x in E2])
    for kind in ("pp", "rel"):
        for at in (False, True):
            m = np.array([x["kind"] == kind and x["attractor"] == at for x in E2])
            r2[f"{kind}_{'attractor' if at else 'same_number'}"] = dict(right=round(float(ok[m].mean()), 4), singular_subj=round(float(ok[m & ~sp].mean()), 4),
                                                                      plural_subj=round(float(ok[m & sp].mean()), 4), sensitivity_auc=auc(lo_pl[m & sp], lo_pl[m & ~sp]), n=int(m.sum()))
    res["E2_controlled"] = dict(n=len(E2), all=round(float(ok.mean()), 4), sensitivity_auc=auc(lo_pl[sp], lo_pl[~sp]), **r2, examples=[" ".join(x["good"]) for x in E2[:6]])
    print("E2", {k: v for k, v in res["E2_controlled"].items() if k != "examples"}, res["E2_controlled"]["examples"][:3], flush=True)
    # F
    subs = substitutions(C, it["nouns_at"], tok, vocab, seed)
    if isinstance(M, Trigram):
        so = np.array([-M.word_logp(s, i) for s, i, _ in subs])
        sw = np.array([-M.word_logp(s[:i] + [w] + s[i + 1:], i) for s, i, w in subs])
    else:
        lo = M.token_logp([s for s, _, _ in subs]); lw = M.token_logp([s[:i] + [w] + s[i + 1:] for s, i, w in subs])
        so = np.array([-lo[k][i] for k, (s, i, w) in enumerate(subs)]); sw = np.array([-lw[k][i] for k, (s, i, w) in enumerate(subs)])
    pos = np.array([i for _, i, _ in subs])
    res["F"] = dict(n=len(subs), more_surprised_by_substitute=round(float(np.mean(sw > so)), 4),
                    mean_extra_surprise_nats=round(float(np.mean(sw - so)), 3),
                    late_in_sentence_pos8plus=round(float(np.mean((sw > so)[pos >= 8])), 4) if (pos >= 8).sum() else None,
                    early_pos3to5=round(float(np.mean((sw > so)[pos <= 5])), 4))
    print("F", res["F"], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=("trigram", "srn", "gru"), required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--corpus", choices=("books", "big"), default="books"); a = ap.parse_args()
    tag = "" if a.corpus == "books" else "_big"
    os.makedirs(OUT, exist_ok=True)
    C = GR.corpus(); tok = Counter(w for s in C["train"] for w in s)       # test items always from the original 32 books
    Ct = C if a.corpus == "books" else corpus_big(C)
    t0 = time.time()
    M = Trigram(Ct, a.seed) if a.model == "trigram" else Net(a.model, Ct, a.seed, a.epochs, frac=a.frac, tag=tag, min_count=3 if a.corpus == "books" else 5)
    res = dict(model=a.model, seed=a.seed, frac=a.frac, corpus=a.corpus, train_words=sum(map(len, Ct["train"])), train_minutes=round((time.time() - t0) / 60, 1), training=getattr(M, "log", None))
    res = evaluate(M, C, tok, a.seed, res)
    json.dump(res, open(os.path.join(OUT, f"{a.model}{tag}{'' if a.frac == 1.0 else f'_f{a.frac}'}_s{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
