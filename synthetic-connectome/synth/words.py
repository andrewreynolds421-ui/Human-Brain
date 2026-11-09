"""Curriculum step 2: reading words letter by letter, with a prediction rule (Stage 36).

Pieces (each MEASURED / FITTED / ASSUMED as labelled):

* Words: the most frequent English words of 3-7 lowercase letters with their measured frequencies
  (wordfreq 3.1, Speer 2022: aggregated subtitles, Wikipedia, books, web, news).  A fraction of word
  types is held out; training sequences are sampled in proportion to frequency (what a reader meets).
* Letters: the frozen 1,600-cell letter layer (Stage 23, l1600v) - its spike counts to real images,
  one font per word.  26 letter units read those counts (logistic readout trained on labelled images
  of *readout fonts* only); their output is the bottom-up evidence for each letter.
* Context: what the sequence layer sees at each step - the current letter (evidence, population code of
  m inputs per letter), a fading trace of the earlier letters (ASSUMED decay 0.5 per letter; serial
  order as a recency gradient, Page & Norris 1998), and a word-start group.
* Sequence layer: the same self-organising spiking layer as the letter layer (synth/sensory.py,
  Diehl & Cook 2015), learning without labels, so its cells become detectors of letter chunks.
* Prediction rule: synapses from the sequence cells to the 26 letter units learn to predict the NEXT
  letter, taught by that letter's own bottom-up evidence when it arrives (a delta rule on a local
  error, as in dendritic prediction of somatic activity: Urbanczik & Senn 2014).  The prediction is
  added top-down to the letter units in the log domain with gain beta (FITTED on validation fonts):
  posterior ~ evidence x prediction^beta (interactive activation: McClelland & Rumelhart 1981).
* Bounds: an ideal reader that knows the training words' exact bigram / trigram letter statistics and
  combines them with the same evidence by forward filtering (left to right, as the model reads).

The prediction rule is applied per letter to spike counts (event level), not in continuous time.
"""
import os
import numpy as np

LETTERS = "abcdefghijklmnopqrstuvwxyz"
L = 26
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PER_FONT = 6                       # renderings per font and letter in data/letters.npz


# ----------------------------------------------------------------------------------------- words
def load_words(n_top=5000, min_len=3, max_len=7):
    """the n_top most frequent English words with min_len..max_len lowercase ASCII letters;
    returns (words, frequencies) - frequencies MEASURED (wordfreq)"""
    from wordfreq import top_n_list, word_frequency
    out = []
    for w in top_n_list("en", 100000):
        if w.isascii() and w.isalpha() and w.islower() and min_len <= len(w) <= max_len:
            out.append(w)
            if len(out) == n_top:
                break
    f = np.array([word_frequency(w, "en") for w in out])
    return out, f


def split_words(words, held_frac=0.2, seed=0):
    """indices of trained and held-out word types (random split)"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(words))
    n_held = int(round(held_frac * len(words)))
    return np.sort(perm[n_held:]), np.sort(perm[:n_held])


def encode(word):
    return np.array([LETTERS.index(c) for c in word], int)


# ------------------------------------------------------------------------------ letter responses
class LetterBank:
    """letter-layer responses (spike counts, 1600 cells) to every image, indexed by font and letter.
    Image k of a split is font k // (26*6), letter (k // 6) % 26 (scripts/make_letters.py order)."""

    def __init__(self, feat_train, feat_test):
        self.F = {"train": feat_train, "test": feat_test}
        self.n_fonts = {s: len(F) // (L * PER_FONT) for s, F in self.F.items()}

    @classmethod
    def load(cls, folder=os.path.join(ROOT, "results", "words")):
        def merge(prefix, shards):
            parts = [np.load(os.path.join(folder, f"{prefix}{k}.npz")) for k in range(shards)]
            n = sum(len(p["idx"]) for p in parts)
            F = np.zeros((n, parts[0]["F"].shape[1]), np.float32)
            for p in parts:
                assert int(p["done"]) == len(p["idx"]), "feature file not finished"
                F[p["idx"]] = p["F"]
            return F
        return cls(merge("feat_train", 2), merge("feat_test", 2))

    def rows(self, split, fonts=None, letters=None):
        n = len(self.F[split])
        font = np.arange(n) // (L * PER_FONT); let = (np.arange(n) // PER_FONT) % L
        m = np.ones(n, bool)
        if fonts is not None:
            m &= np.isin(font, fonts)
        if letters is not None:
            m &= np.isin(let, letters)
        return np.nonzero(m)[0], let[m], font[m]

    def word(self, split, font, word_idx, rng):
        """responses for one word printed in one font: a random rendering of each letter"""
        r = rng.integers(0, PER_FONT, len(word_idx))
        k = font * L * PER_FONT + word_idx * PER_FONT + r
        return self.F[split][k]


def readout_features(F):
    return np.log1p(F)


class LetterReadout:
    """26 letter units reading the letter layer (multinomial logistic regression; FITTED on labelled
    images of the readout fonts only).  evidence(F) = probability per letter, class-balanced."""

    def __init__(self, C=0.05):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=C, max_iter=3000)

    def fit(self, F, y):
        self.m.fit(readout_features(F), y)
        return self

    def evidence(self, F):
        return self.m.predict_proba(readout_features(np.atleast_2d(F)))


# -------------------------------------------------------------------------------------- context
def context_vector(cur, trace, start, m=20):
    """input rates (0..1) to the sequence layer: [current letter (26 groups) | trace (26 groups) | start],
    m inputs per group (population code)"""
    g = np.concatenate([cur, np.clip(trace, 0, 1), [1.0 if start else 0.0]])
    return np.repeat(g, m)


def context_groups(cur, trace, start):
    return np.concatenate([cur, np.clip(trace, 0, 1), [1.0 if start else 0.0]])


def letter_pairs(groups):
    """Stage 37 conjunctive code: (current letter or start) x (earlier letters or start), 27 x 27 = 729
    cells, each the product of its two inputs - letter-pair ('open bigram') detectors"""
    cur, tr, st = groups[:L], groups[L:2 * L], groups[2 * L]
    return np.outer(np.r_[cur, st], np.r_[tr, st]).ravel()


# ----------------------------------------------------------------------------- prediction rule
class Predictor:
    """synapses from context cells (spike counts c) to the 26 letter units; the prediction of the next
    letter is softmax(W c + b).  Learning: delta rule on the local error (teacher - prediction), the
    teacher being the next letter's bottom-up evidence (no labels)."""

    def __init__(self, n_in, eta=0.05, decay=0.0, seed=0, scale="sum"):
        """scale: 'sum' = inputs divided by their total (Stage 36); 'none' = inputs as they are
        (Stage 37: better, +0.7 / +0.6 points on known / new words over 4 networks)"""
        self.W = np.zeros((L, n_in)); self.b = np.zeros(L)
        self.eta, self.decay, self.scale = eta, decay, scale

    def _scale(self, c):
        c = np.asarray(c, float)
        if self.scale == "none":
            return c
        s = c.sum()
        return c / s if s > 0 else c

    def predict(self, c):
        z = self.W @ self._scale(c) + self.b
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def learn(self, c, teacher):
        x = self._scale(c)
        err = teacher - self.predict(c)
        self.W += self.eta * np.outer(err, x) - self.eta * self.decay * self.W
        self.b += self.eta * 0.1 * err
        return err


def combine(evidence, prior, beta):
    """letter units with bottom-up evidence and top-down prediction: posterior ~ ev * prior^beta"""
    lp = np.log(np.maximum(evidence, 1e-12)) + beta * np.log(np.maximum(prior, 1e-12))
    lp -= lp.max()
    p = np.exp(lp)
    return p / p.sum()


# ------------------------------------------------------------------------------- ideal readers
class NGram:
    """ideal reader: exact letter n-gram statistics (n = 1, 2, 3; '^' = word start) of the training
    word tokens (frequency-weighted, as counts per million tokens), with add-alpha smoothing"""

    def __init__(self, words, freqs, n=2, alpha=0.5):
        self.n = n
        K = L + 1                                   # index L = start symbol
        shape = (K,) * (n - 1) + (L,)
        C = np.full(shape, alpha)
        freqs = np.asarray(freqs, float) / np.sum(freqs) * 1e6
        for w, f in zip(words, freqs):
            seq = [L] * (n - 1) + list(encode(w))
            for t in range(n - 1, len(seq)):
                C[tuple(seq[t - n + 1:t + 1])] += f
        self.P = C / C.sum(-1, keepdims=True)

    def filter(self, E, beta=1.0):
        """forward filtering of the evidence sequence E (T x 26), left to right.  beta weights the
        statistics against the evidence (applied as evidence ** (1/beta), which gives the same best
        letter as prior ** beta x evidence and keeps the belief update exact).  Returns posteriors and
        the predictions made before each letter (both T x 26)."""
        T, K = len(E), L + 1
        post, pred = np.zeros((T, L)), np.zeros((T, L))
        belief = np.zeros((K,) * (self.n - 1)) if self.n > 1 else None
        if belief is not None:
            belief[(L,) * (self.n - 1)] = 1.0
        for t in range(T):
            lik = np.maximum(E[t], 1e-12) ** (1.0 / max(beta, 1e-9))
            if self.n == 1:
                pred[t] = self.P
                p = self.P * lik; post[t] = p / p.sum()
                continue
            joint = belief[..., None] * self.P          # (history..., next)
            pr = joint.reshape(-1, L).sum(0); pred[t] = pr / pr.sum()
            jl = joint * lik
            p = jl.reshape(-1, L).sum(0); post[t] = p / p.sum()
            if self.n == 2:
                nb = np.zeros(K); nb[:L] = p
            else:                                       # history (a, b) -> (b, next)
                nb = np.zeros((K, K)); nb[:, :L] = jl.sum(0)
            belief = nb / nb.sum()
        return post, pred
