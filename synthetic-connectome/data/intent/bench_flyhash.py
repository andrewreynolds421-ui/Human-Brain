#!/usr/bin/env python3
"""
Does FlyHash actually beat the alternatives at the data sizes Warmind has?

The claim under test: sparse random expansion needs fewer examples per class
than conventional approaches. That claim is worth checking rather than
repeating, because the paper measured it on odor and image data, not on
short voice commands.

Baselines:
  - TF-IDF + logistic regression  (the honest strong baseline; if this wins,
    use it and skip the biology)
  - TF-IDF + k-NN                 (nearest neighbour without the fly hash,
    isolating whether the EXPANSION is doing anything)
  - Random LSH                    (dense Gaussian projection DOWN, i.e.
    conventional LSH, to test the fly's up-projection specifically)
  - Majority class                (the floor)

Run:  python bench_flyhash.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fly_hash import CharNgramEncoder, FlyHash, FlyHashConfig, FlyRouter  # noqa: E402


# ---------------------------------------------------------------------------
# Data: paraphrase families per intent, in the register a wake word gets
# ---------------------------------------------------------------------------

DATA = {
    "LOCAL_LAUNCH_APP": [
        "open spotify", "launch spotify", "start up spotify", "open the spotify app",
        "fire up spotify", "put on spotify", "open google maps", "launch maps",
        "open chrome", "start chrome", "launch the camera", "open camera app",
        "open whatsapp", "launch telegram", "start the calculator",
        "open my email app", "launch youtube", "open netflix",
    ],
    "LOCAL_TRACK": [
        "log 80 kilos", "record my weight 80 kilograms", "log weight eighty kilos",
        "note my weight at 80kg", "log blood pressure 120 over 80",
        "record bp 118 over 76", "log my resting heart rate 62",
        "note heart rate 58", "log 7 hours of sleep", "record sleep 6 and a half hours",
        "log my mood as good today", "note mood low", "record 8500 steps",
        "log steps 12000", "log my weight as 79.4",
    ],
    "LOCAL_STATUS": [
        "status report", "what are the subminds doing", "submind status",
        "give me a status update", "what is everyone working on",
        "report status", "how are the subminds", "current status",
    ],
    "LOCAL_CANCEL": [
        "never mind", "cancel", "stop", "forget it", "cancel that",
        "nevermind", "abort", "dismiss", "stop listening",
    ],
    "REMOTE_HEALTH": [
        "how are my steps trending", "what did my sleep look like this week",
        "is my resting heart rate normal", "what does a bp of 140 over 90 mean",
        "am i sleeping enough", "how has my weight changed this month",
        "what should i ask my doctor about this", "explain what hrv means",
        "why am i so tired lately", "chart my weight over the last year",
    ],
    "REMOTE_FINANCE": [
        "what is apple trading at", "how is nvidia doing",
        "show me congress trades for tesla", "am i overspending this month",
        "what is my runway", "chart my net worth", "how much did i spend on food",
        "what would 10k at 7 percent be in 20 years", "is the market up today",
        "show me my holdings",
    ],
    "REMOTE_GEO": [
        "where is kyiv", "show me berlin on the map", "route from berlin to prague",
        "how do i get to the airport", "find coffee shops near me",
        "what is around here", "draw the border between france and spain",
        "directions home", "where am i", "find a petrol station on my route",
    ],
    "REMOTE_DEVICE": [
        "list the files in my workspace", "run this python script",
        "what is in my sandbox", "open a browser tab for me",
        "delete that file from the workspace", "execute this code",
        "save this to a file", "browse to that site and summarise it",
    ],
    "REMOTE_DASHBOARD": [
        "add a weight widget to the dashboard", "make the dashboard green",
        "change the title to command deck", "hide the projects panel",
        "always answer in metric from now on", "remember to call me commander",
        "put a progress bar on the dashboard", "restyle the interface darker",
    ],
    "REMOTE_CHAT": [
        "what is happening in the news", "explain quantum entanglement",
        "who won the election", "tell me a joke", "what do you think about this",
        "summarise what happened today", "why is the sky blue",
        "what is the capital of mongolia", "how does photosynthesis work",
        "what should i cook tonight",
    ],
}


def make_splits(n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label, items in DATA.items():
        items = list(items)
        rng.shuffle(items)
        n = min(n_per_class, len(items) - 3)
        train += [(t, label) for t in items[:n]]
        test += [(t, label) for t in items[n:]]
    return train, test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def eval_fly(train, test, seed):
    r = FlyRouter(FlyHashConfig(seed=seed), k=5)
    r.remember_many(train)
    return np.mean([r.predict(t)[0] == y for t, y in test])


def eval_conventional_lsh(train, test, seed):
    """Dense Gaussian projection DOWN — textbook LSH, for contrast."""
    enc = CharNgramEncoder(dim=512)
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(512, 128))

    def sig(texts):
        x = enc.encode_batch(texts)
        x = x / np.maximum(x.mean(axis=1, keepdims=True), 1e-8)
        return (x @ proj > 0).astype(np.uint8)

    tr = sig([t for t, _ in train])
    te = sig([t for t, _ in test])
    labels = [y for _, y in train]
    correct = 0
    for i, (_, y) in enumerate(test):
        sim = (tr == te[i]).sum(axis=1)
        top = np.argpartition(sim, -5)[-5:]
        votes = Counter(labels[j] for j in top)
        correct += votes.most_common(1)[0][0] == y
    return correct / len(test)


def eval_tfidf_logreg(train, test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    pipe.fit([t for t, _ in train], [y for _, y in train])
    return np.mean(pipe.predict([t for t, _ in test]) == [y for _, y in test])


def eval_tfidf_knn(train, test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    k = min(5, len(train))
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
        KNeighborsClassifier(n_neighbors=k, metric="cosine"),
    )
    pipe.fit([t for t, _ in train], [y for _, y in train])
    return np.mean(pipe.predict([t for t, _ in test]) == [y for _, y in test])


def eval_majority(train, test):
    top = Counter(y for _, y in train).most_common(1)[0][0]
    return np.mean([y == top for _, y in test])


# ---------------------------------------------------------------------------

def main():
    seeds = range(5)
    sizes = [1, 2, 3, 5, 8]

    print(f"{len(DATA)} classes, "
          f"{sum(len(v) for v in DATA.values())} utterances total")
    print("Accuracy, mean over 5 seeds. n = training examples PER CLASS.\n")
    header = f"{'n/class':>8} " + "".join(f"{m:>12}" for m in
             ["FlyHash", "TFIDF+kNN", "TFIDF+LR", "conv.LSH", "majority"])
    print(header)
    print("-" * len(header))

    results = {}
    for n in sizes:
        row = {m: [] for m in ["fly", "knn", "lr", "lsh", "maj"]}
        for s in seeds:
            train, test = make_splits(n, s)
            row["fly"].append(eval_fly(train, test, s))
            row["knn"].append(eval_tfidf_knn(train, test))
            row["lr"].append(eval_tfidf_logreg(train, test))
            row["lsh"].append(eval_conventional_lsh(train, test, s))
            row["maj"].append(eval_majority(train, test))
        means = {k: float(np.mean(v)) for k, v in row.items()}
        results[n] = means
        print(f"{n:>8} " + "".join(f"{means[k]:>12.3f}" for k in
              ["fly", "knn", "lr", "lsh", "maj"]))

    print("\nOne-shot detail (n=1), the case that matters for cold start:")
    m = results[1]
    print(f"  FlyHash beats conventional LSH by {m['fly'] - m['lsh']:+.3f}")
    print(f"  FlyHash vs TF-IDF k-NN:            {m['fly'] - m['knn']:+.3f}")
    print(f"  FlyHash vs logistic regression:    {m['fly'] - m['lr']:+.3f}")

    print("\nCrossover — where a trained model overtakes retrieval:")
    for n in sizes:
        if results[n]["lr"] > results[n]["fly"]:
            print(f"  logistic regression passes FlyHash at n={n}")
            break
    else:
        print("  logistic regression never passes FlyHash in this range")

    # Latency, since the whole point is that it is cheap.
    import time
    train, test = make_splits(8, 0)
    r = FlyRouter(FlyHashConfig(seed=0))
    r.remember_many(train)
    t0 = time.perf_counter()
    for t, _ in test * 20:
        r.predict(t)
    per = (time.perf_counter() - t0) / (len(test) * 20) * 1000
    print(f"\nFlyHash inference: {per:.2f} ms/utterance "
          f"({r.fly.n_kc} Kenyon cells, {len(r.labels)} memories)")


if __name__ == "__main__":
    main()
