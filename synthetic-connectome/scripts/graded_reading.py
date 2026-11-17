#!/usr/bin/env python3
"""Stage 53: teaching reading grade by grade.

Books (Project Gutenberg, 3 folders in /home/claude/corpora): 38 school readers labelled by grade (McGuffey,
New National, Aldine, Summers, Elson, Cyr ...) and ~240 other children's books and classics. Each other book's
grade is its Flesch-Kincaid grade CALIBRATED on the readers (straight line, FITTED). Bands: 1, 2, 3, 4, 5, 6,
7+ (the calibrated scale puts even the adult classics at 6-8, so one top band).

READING-LEVEL TEST (per band, books never read): multiple-choice CLOZE - a content word (noun, verb,
adjective, adverb) of a sentence is blanked and 4 words offered: the original and 3 of the same part of
speech and about the same frequency; the model picks the one that makes the sentence most probable
(chance 25 %). Cloze is a standard measure of children's reading comprehension (Taylor 1953; Bormuth 1967).
The model "reads at" a band when its cloze score there is >= PASS (75 %, ASSUMED).

TEACHING
  curriculum - starts at band 1; reads 200,000-word lessons (80 % the current band, 20 % review of easier
               bands); after each lesson it is tested on held-out sentences of its current band's books
               (validation: every 25th sentence, never read); promoted when it passes, or when 4 lessons
               bring no gain of 1 point (promoted "not mastered"). After the top band: all bands mixed.
  random     - CONTROL: the same books, sentences drawn at random from all bands, same number of words.
Both are tested on every band every 2 M words.

Model: the gated recurrent memory of Stages 51-52 (GRU, 128 input / 256 memory units) with an adaptive
softmax output (Grave et al. 2017) for speed; vocabulary = words met 5+ times in the training books.
MODEL: learning by backpropagation through time.

    python scripts/graded_reading.py prepare                      # catalogue, splits, cloze items
    python scripts/graded_reading.py run --regime curriculum --seed 0 --budget 20
"""
import argparse, glob, json, math, os, pickle, re, sys, time
from collections import Counter, defaultdict
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar as GR                                                         # noqa: E402
from synth.readability import fk_grade                                       # noqa: E402

DATA = "/home/claude/models/graded"                                          # big caches, not in the repository
OUT = os.path.join(ROOT, "results", "graded")
CAT = os.path.join(ROOT, "data", "grades", "catalogue.json")
BANDS = ["1", "2", "3", "4", "5", "6", "7+"]
PASS, LESSON, REVIEW, PLATEAU = 0.75, 200_000, 0.2, 4
TEST_READERS = ("New-National-First", "Summers-Readers-First", "Beacon-Second", "Third-Eclectic", "Child-s-WorldThird",
                "New-National-Fourth", "De-La-Salle-Fifth", "Silent-ReadersSixth", "Literary-World-Seventh", "Eighth-Reader")
TEST_OTHER = ("Treasure-Island", "The-Secret-Garden", "Just-So-Stories", "Black-Beauty", "Ozma-of-Oz",
              "The-Scarlet-Letter", "The-House-of-the-Seven-Gables", "Silas-Marner")


def band_of(g):
    g = max(1, round(g))
    return BANDS[g - 1] if g <= 6 else "7+"


# ---------------------------------------------------------------- catalogue and splits
def prepare():
    raw = json.load(open(os.path.join(ROOT, "data", "grades", "catalogue_raw.json")))
    seen, cat = set(), []
    for c in raw:                                                            # the same reader in two folders
        key = re.sub(r"_\d+$", "", c["name"])
        if key in seen: continue
        seen.add(key); cat.append(c)
    r = [c for c in cat if c["reader_grade"]]
    b, a = np.polyfit([c["fk"] for c in r], [c["reader_grade"] for c in r], 1)
    for c in cat:
        c["grade"] = c["reader_grade"] if c["reader_grade"] else round(float(a + b * c["fk"]), 1)
        c["band"] = band_of(c["grade"])
        c["split"] = "test" if any(t in c["name"] for t in TEST_READERS + TEST_OTHER) else "train"
    json.dump(dict(calibration=dict(slope=round(float(b), 3), intercept=round(float(a), 3), r=round(float(np.corrcoef([c["fk"] for c in r], [c["reader_grade"] for c in r])[0, 1]), 3)),
                   books=cat), open(CAT, "w"), indent=1)
    # sentences
    S = {sp: defaultdict(list) for sp in ("train", "val", "test")}
    for c in cat:
        ss = GR.sentences(GR.book_text(c["dir"]))
        if c["split"] == "test":
            S["test"][c["band"]] += ss
        else:
            for i, s in enumerate(ss):
                (S["val"] if i % 25 == 12 else S["train"])[c["band"]].append(s)
    # test sentences must not appear in training text (readers reprint passages): drop training / validation
    # sentences sharing a 6-word run with any test sentence
    def sh(x): return {tuple(x[i:i + 6]) for i in range(len(x) - 5)}
    held = set()
    for v in S["test"].values():
        for s in v: held |= sh(s)
    dropped = 0
    for sp in ("train", "val"):
        for bnd in S[sp]:
            k = [s for s in S[sp][bnd] if not (sh(s) & held)]; dropped += len(S[sp][bnd]) - len(k); S[sp][bnd] = k
    cnt = Counter(w for v in S["train"].values() for s in v for w in s)
    vocab = sorted(w for w, n in cnt.items() if n >= 5)
    items = {sp: {bnd: cloze_items(S[sp][bnd], cnt, set(vocab), 600 if sp == "test" else 300, seed=7) for bnd in BANDS} for sp in ("val", "test")}
    os.makedirs(DATA, exist_ok=True)
    pickle.dump(dict(S=S, vocab=vocab, items=items), open(os.path.join(DATA, "data.pkl"), "wb"))
    summ = dict(calibration=dict(slope=round(float(b), 3), intercept=round(float(a), 3)), dropped_overlap=dropped, vocab=len(vocab),
                words={sp: {bnd: sum(map(len, S[sp][bnd])) for bnd in BANDS} for sp in S},
                books={bnd: dict(train=[c["name"] for c in cat if c["band"] == bnd and c["split"] == "train"].__len__(),
                                 test=[c["name"][:40] for c in cat if c["band"] == bnd and c["split"] == "test"]) for bnd in BANDS},
                cloze_items={sp: {bnd: len(v) for bnd, v in items[sp].items()} for sp in items},
                cloze_examples={bnd: [" ".join(x["options"][x["answer"]] if i == x["pos"] else w for i, w in enumerate(x["sent"])) + "   [" + " / ".join(x["options"]) + "]"
                                      for x in items["test"][bnd][:3]] for bnd in BANDS})
    os.makedirs(OUT, exist_ok=True); json.dump(summ, open(os.path.join(OUT, "prepare.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "cloze_examples"}, indent=1))
    for bnd, ex in summ["cloze_examples"].items():
        print(bnd, ex[0][:200] if ex else "-")


def cloze_items(sents, cnt, vocab, n, seed):
    """blank a content word; 3 distractors of the same part of speech and frequency (rank +-40), not in the sentence"""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    rng = np.random.default_rng(seed)
    pool = [s for s in sents if 6 <= len(s) <= 25]
    pick = [pool[i] for i in rng.choice(len(pool), min(len(pool), 3 * n), replace=False)]
    tagged, bypos = [], defaultdict(Counter)
    for doc, s in zip(nlp.pipe((" ".join(s) for s in pick), batch_size=256), pick):
        toks = [t for t in doc]
        if [t.text for t in toks] != s: continue
        tagged.append((s, [t.pos_ for t in toks]))
        for t in toks:
            if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and t.text in vocab: bypos[t.pos_][t.text] += 1
    ranked = {p: sorted({w for w in c} | set(), key=lambda w: -cnt[w]) for p, c in bypos.items()}
    rank = {p: {w: i for i, w in enumerate(v)} for p, v in ranked.items()}
    out = []
    for s, tags in tagged:
        cand = [i for i, (w, p) in enumerate(zip(s, tags)) if i >= 2 and p in ranked and w in vocab and len(ranked[p]) > 100]
        if not cand: continue
        i = int(rng.choice(cand)); p = tags[i]; w = s[i]; r = rank[p][w]
        near = [x for x in ranked[p][max(0, r - 40): r + 41] if x != w and x not in s]
        if len(near) < 3: continue
        opts = [w] + [str(x) for x in rng.choice(near, 3, replace=False)]; order = rng.permutation(4)
        out.append(dict(sent=s, pos=i, options=[opts[k] for k in order], answer=int(np.argmax(order == 0)), tag=p))
        if len(out) >= n: break
    return out


# ---------------------------------------------------------------- model
class Model:
    def __init__(self, vocab, seed):
        import torch, torch.nn as nn
        torch.manual_seed(seed); torch.set_num_threads(int(os.environ.get("THREADS", "1")))
        self.torch = torch
        self.itos = ["<pad>", "<s>", "</s>", "<unk>"] + list(vocab); self.stoi = {w: i for i, w in enumerate(self.itos)}
        V = len(self.itos)

        class M(nn.Module):
            def __init__(s):
                super().__init__(); s.e = nn.Embedding(V, 128, padding_idx=0); s.r = nn.GRU(128, 256, batch_first=True); s.d = nn.Dropout(0.25)
                s.o = nn.AdaptiveLogSoftmaxWithLoss(256, V, cutoffs=[2000, 10000], div_value=4.0)
            def hidden(s, x):
                h, _ = s.r(s.d(s.e(x))); return s.d(h)
        self.m = M(); self.opt = torch.optim.Adam(self.m.parameters(), lr=2e-3)

    def enc(self, s):
        return [1] + [self.stoi.get(w, 3) for w in s] + [2]

    def pad(self, seqs):
        L = max(map(len, seqs)); x = np.zeros((len(seqs), L), np.int64)
        for k, q in enumerate(seqs): x[k, :len(q)] = q
        return self.torch.from_numpy(x)

    def learn(self, sents, bs=64):
        """one pass over these sentences (length-sorted batches, shuffled)"""
        torch = self.torch; self.m.train()
        idx = np.argsort([len(s) for s in sents], kind="stable"); groups = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        np.random.default_rng(len(sents)).shuffle(groups)
        for g in groups:
            x = self.pad([self.enc(sents[i]) for i in g]); h = self.m.hidden(x[:, :-1]); y = x[:, 1:]
            keep = (y > 0).reshape(-1)
            loss = self.m.o(h.reshape(-1, h.shape[-1])[keep], y.reshape(-1)[keep]).loss
            self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0); self.opt.step()

    def logp(self, sents, bs=128):
        torch = self.torch; self.m.eval(); out = np.zeros(len(sents))
        with torch.no_grad():
            idx = np.argsort([len(s) for s in sents], kind="stable")
            for k in range(0, len(idx), bs):
                g = idx[k:k + bs]; x = self.pad([self.enc(sents[i]) for i in g]); h = self.m.hidden(x[:, :-1]); y = x[:, 1:]
                keep = (y > 0).reshape(-1)                         # only the actual next word's probability is needed
                tok = self.m.o(h.reshape(-1, h.shape[-1])[keep], y.reshape(-1)[keep]).output
                full = torch.zeros(y.numel()); full[keep] = tok
                out[g] = full.reshape(y.shape).sum(1).numpy()
        return out

    def cloze(self, items):
        if not items: return None
        sents = [x["sent"][:x["pos"]] + [o] + x["sent"][x["pos"] + 1:] for x in items for o in x["options"]]
        lp = self.logp(sents).reshape(len(items), 4)
        return float(np.mean(lp.argmax(1) == np.array([x["answer"] for x in items])))

    def state(self):
        return dict(model=self.m.state_dict(), opt=self.opt.state_dict())

    def load(self, z):
        self.m.load_state_dict(z["model"]); self.opt.load_state_dict(z["opt"])


# ---------------------------------------------------------------- teaching
def run(a):
    D = pickle.load(open(os.path.join(DATA, "data.pkl"), "rb")); S = D["S"]
    M = Model(D["vocab"], a.seed)
    tag = f"{a.regime}_s{a.seed}"; ck = os.path.join(DATA, tag + ".ckpt")
    st = dict(words=0, band=0, best=0.0, stale=0, lessons=[], promotions=[], tests=[], next_test=0, rng=None, done=False)
    if os.path.exists(ck):
        z = pickle.load(open(ck, "rb")); M.load(z["m"]); st = z["st"]; print("resumed at", st["words"], "words", flush=True)
    rng = np.random.default_rng(a.seed)
    if st["rng"]: rng.bit_generator.state = st["rng"]
    budget = int(a.budget * 1e6); last_save = time.time()
    allsent = [s for b in BANDS for s in S["train"][b]]

    def save():
        st["rng"] = rng.bit_generator.state
        pickle.dump(dict(m=M.state(), st=st), open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)

    def draw(pool, nwords):
        out, n = [], 0
        while n < nwords:
            s = pool[int(rng.integers(len(pool)))]; out.append(s); n += len(s)
        return out

    def test_all():
        r = dict(words=st["words"], band=BANDS[min(st["band"], len(BANDS) - 1)], cloze={b: round(M.cloze(D["items"]["test"][b]), 4) for b in BANDS},
                 logp_per_word={b: round(float(M.logp(S["test"][b][:1500]).sum() / sum(len(s) + 1 for s in S["test"][b][:1500])), 3) for b in BANDS})
        lvl = [b for b in BANDS if r["cloze"][b] >= PASS]
        r["reads_at"] = lvl[-1] if lvl else "below 1"
        r["passes_all_up_to"] = next((BANDS[i - 1] for i, b in enumerate(BANDS) if r["cloze"][b] < PASS), BANDS[-1]) if r["cloze"][BANDS[0]] >= PASS else "below 1"
        st["tests"].append(r); print("TEST", r["words"], r["band"], r["cloze"], "reads at", r["reads_at"], flush=True)

    while st["words"] < budget:
        if st["words"] >= st["next_test"]:
            test_all(); st["next_test"] += 2_000_000; save()
        if a.regime == "random":
            lesson = draw(allsent, LESSON)
        else:
            b = min(st["band"], len(BANDS) - 1)
            if st["band"] >= len(BANDS):                                     # after the top band: everything mixed
                lesson = draw(allsent, LESSON)
            else:
                easier = [s for k in BANDS[:b] for s in S["train"][k]]
                n_rev = int(REVIEW * LESSON) if easier else 0
                lesson = draw(S["train"][BANDS[b]], LESSON - n_rev) + (draw(easier, n_rev) if n_rev else [])
        M.learn(lesson); st["words"] += sum(map(len, lesson))
        b = min(st["band"], len(BANDS) - 1)
        v = M.cloze(D["items"]["val"][BANDS[b]])
        st["lessons"].append(dict(words=st["words"], band=BANDS[b], val_cloze=round(v, 4)))
        if a.regime == "curriculum" and st["band"] < len(BANDS):
            if v >= st["best"] + 0.01: st["best"], st["stale"] = v, 0
            else: st["stale"] += 1
            if v >= PASS or st["stale"] >= PLATEAU:
                st["promotions"].append(dict(band=BANDS[b], words=st["words"], val_cloze=round(v, 4), mastered=bool(v >= PASS)))
                print("PROMOTED from", BANDS[b], "at", st["words"], "words, cloze", round(v, 3), "mastered" if v >= PASS else "NOT mastered", flush=True)
                st["band"] += 1; st["best"], st["stale"] = 0.0, 0
        if time.time() - last_save > 600:
            save(); last_save = time.time()
    test_all(); st["done"] = True; save()
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(regime=a.regime, seed=a.seed, budget_words=budget, pass_mark=PASS, promotions=st["promotions"], tests=st["tests"], lessons=st["lessons"]),
              open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("prepare", "run"))
    ap.add_argument("--regime", choices=("curriculum", "random"), default="curriculum"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=20.0, help="millions of words read"); a = ap.parse_args()
    prepare() if a.cmd == "prepare" else run(a)


if __name__ == "__main__":
    main()
