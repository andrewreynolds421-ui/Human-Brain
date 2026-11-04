"""Fixed text -> spike-pattern encoder for the learner's cue cells.

Each word (lower-cased, letters/digits only, length >= 2, common function words
dropped) activates a fixed pseudo-random subset of the cue cells, chosen by a
hash of the word (sparse distributed code, as in hashing-based "random
indexing").  A message activates the union of its words' subsets.  The code
is not learned: the network has to learn which patterns lead to which action.

Nothing here understands language; similar messages overlap only in the words
they share.
"""
import hashlib
import re

import numpy as np

STOP = set("""a an and are as at be but by can could do does for from had has have how i if in into is it its
me my no not of on or our please she so than that the their them then there these they this to was we
were what when where which who why will with would you your just about any some get got""".split())


def words(text):
    return [w for w in re.findall(r"[a-z0-9&$]+", text.lower()) if len(w) >= 2 and w not in STOP]


def word_cells(word, n_cells, frac=0.05, seed=0):
    """fixed subset of cue-cell indices (0..n_cells-1) for one word"""
    h = int.from_bytes(hashlib.blake2b(f"{seed}:{word}".encode(), digest_size=8).digest(), "little")
    k = max(1, int(round(frac * n_cells)))
    return np.random.default_rng(h).choice(n_cells, k, replace=False)


def word_vector(word, n_cells, seed=0):
    """fixed random input weights from one word to every cue cell"""
    h = int.from_bytes(hashlib.blake2b(f"v{seed}:{word}".encode(), digest_size=8).digest(), "little")
    return np.random.default_rng(h).standard_normal(n_cells)


def encode_kwta(text, n_cells, frac=0.33, max_words=12, seed=0):
    """fly-olfaction-style code (random projection + k-winners-take-all; Dasgupta et al. 2017):
    each word sends fixed random input to every cue cell, and only the frac most-driven cells
    fire.  Every message activates the same number of cells; similar messages share most."""
    seen = list(dict.fromkeys(words(text)))[:max_words]
    act = np.zeros(n_cells)
    if seen:
        drive = sum(word_vector(w, n_cells, seed) for w in seen)
        act[np.argsort(drive)[-max(1, int(round(frac * n_cells))):]] = 1.0
    return act, seen


def encode(text, n_cells, frac=0.05, max_words=12, seed=0, mode="union"):
    if mode == "kwta":
        return encode_kwta(text, n_cells, frac, max_words, seed)
    """activity level per cue cell in [0, 1]: 1 for cells of any word in the message
    (first max_words distinct content words); all zero if no content words"""
    act = np.zeros(n_cells)
    seen = []
    for w in words(text):
        if w not in seen:
            seen.append(w)
        if len(seen) >= max_words:
            break
    for w in seen:
        act[word_cells(w, n_cells, frac, seed)] = 1.0
    return act, seen
