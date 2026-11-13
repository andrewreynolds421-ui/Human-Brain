"""
Cold-start intent router: TF-IDF character n-grams + calibrated logistic
regression.

Why this and not the fly circuit or DistilBERT
----------------------------------------------
Benchmarked against FlyHash, conventional LSH, TF-IDF k-NN and a majority
baseline at 1-8 training examples per class (see bench_flyhash.py). Logistic
regression won at every size, including one-shot. It also trains in under a
second, needs no PyTorch, and exports to a few hundred KB of JSON that
Kotlin can evaluate directly — no onnxruntime dependency.

DistilBERT should overtake it once tool_calls has real volume. This is the
thing to run until then, and the baseline to beat afterwards.

The calibration problem this exists to solve
--------------------------------------------
labels.py routes on absolute confidence: 0.60 to pick a submind, 0.85 to
act on the phone without asking. Those numbers are only meaningful if the
model's probabilities correspond to actual correctness rates.

Out of the box they do not. With 10 classes, char n-grams and default
regularisation on a small dataset, EVERY prediction scored below 0.40 — the
router would have sent everything to RASPUTIN and never acted at all. Not a
threshold that needed lowering: a model whose outputs did not mean what the
thresholds assumed.

The fix is temperature scaling (Guo et al. 2017): fit one scalar T on
held-out data to minimise negative log-likelihood, and divide the logits by
it. It cannot change which class wins, so accuracy is untouched — it only
makes the confidence honest. After fitting, a prediction reported at 0.85
should be right about 85% of the time, which is what makes the asymmetric
thresholds mean something.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RouterConfig:
    ngram_range: tuple[int, int] = (3, 5)
    min_df: int = 1
    C: float = 10.0
    max_features: int = 20000


class LogRegRouter:
    """Fit, calibrate, predict, export."""

    def __init__(self, config: RouterConfig | None = None):
        self.cfg = config or RouterConfig()
        self.vec = None
        self.clf = None
        self.temperature = 1.0
        self.classes_: list[str] = []

    # --- training ----------------------------------------------------------

    def fit(self, texts: list[str], labels: list[str],
            calibrate: bool = True, seed: int = 0) -> "LogRegRouter":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        # char_wb rather than word n-grams: the input is ASR output after a
        # wake word, so "log eighty killos" has to land near "log 80 kilos".
        # Character n-grams degrade gracefully on transcription errors; word
        # tokens do not.
        self.vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=self.cfg.ngram_range,
            min_df=self.cfg.min_df, max_features=self.cfg.max_features,
            sublinear_tf=True,
        )
        X = self.vec.fit_transform(texts)

        self.clf = LogisticRegression(
            max_iter=3000, C=self.cfg.C,
            # Real usage is dominated by REMOTE_CHAT. Unweighted, the model
            # learns to say that and reports good accuracy while being
            # useless for every class that makes the router worth having.
            class_weight="balanced",
        )
        self.clf.fit(X, labels)
        self.classes_ = list(self.clf.classes_)

        if calibrate and len(set(labels)) > 1:
            self._fit_temperature(texts, labels, seed)
        return self

    def _fit_temperature(self, texts, labels, seed) -> None:
        """
        Fit T by cross-validated NLL.

        Uses CV rather than a holdout because at this data size a holdout
        big enough to fit T on is a holdout too big to spare. Falls back to
        T=1 if there are too few examples per class to split at all.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from scipy.optimize import minimize_scalar

        counts = {l: labels.count(l) for l in set(labels)}
        n_splits = min(3, min(counts.values()))
        if n_splits < 2:
            print("[calibrate] too few examples per class; leaving T=1")
            self.temperature = 1.0
            return

        y = np.array(labels)
        logits_all, y_all = [], []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_i, te_i in skf.split(np.zeros(len(y)), y):
            vec = type(self.vec)(**self.vec.get_params())
            Xtr = vec.fit_transform([texts[i] for i in tr_i])
            clf = LogisticRegression(max_iter=3000, C=self.cfg.C,
                                     class_weight="balanced")
            clf.fit(Xtr, y[tr_i])
            Xte = vec.transform([texts[i] for i in te_i])
            logits_all.append(clf.decision_function(Xte))
            y_all.append(np.array([list(clf.classes_).index(v) for v in y[te_i]]))

        logits = np.vstack(logits_all)
        target = np.concatenate(y_all)

        def nll(log_t):
            t = np.exp(log_t)
            z = logits / t
            z = z - z.max(axis=1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            return -logp[np.arange(len(target)), target].mean()

        res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
        self.temperature = float(np.exp(res.x))
        print(f"[calibrate] T = {self.temperature:.3f} "
              f"({'sharpening' if self.temperature < 1 else 'softening'})")

    # --- inference ---------------------------------------------------------

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        z = self.clf.decision_function(self.vec.transform(texts))
        if z.ndim == 1:                      # binary case
            z = np.column_stack([-z, z])
        z = z / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, text: str) -> tuple[str, float]:
        # Out-of-vocabulary guard, mirroring IntentRouter.kt. An input with no
        # known n-grams is an all-zero vector, and softmax over the intercepts
        # alone produces a confident-looking number that means nothing.
        if self.vec.transform([text]).nnz == 0:
            return "REMOTE_CHAT", 0.0
        p = self.predict_proba([text])[0]
        i = int(p.argmax())
        return self.classes_[i], float(p[i])

    def route(self, text: str) -> dict:
        """Prediction plus the threshold policy from labels.py."""
        from labels import route as apply_policy
        label, conf = self.predict(text)
        return apply_policy(label, conf)

    # --- export ------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Export as JSON so Kotlin can evaluate it without a runtime.

        A multinomial logistic regression on TF-IDF features is a sparse
        lookup plus a matrix multiply. Shipping the vocabulary, idf weights,
        coefficients and temperature is enough to reproduce it exactly in
        about 60 lines of Kotlin — no onnxruntime, no 65MB model file.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        from labels import policy

        coef = self.clf.coef_
        payload = {
            "format": "logreg-tfidf-v2",      # v2: carries the routing policy
            "policy": policy(),
            "classes": self.classes_,
            "temperature": self.temperature,
            "analyzer": "char_wb",
            "ngram_range": list(self.cfg.ngram_range),
            "sublinear_tf": True,
            "vocabulary": {k: int(v) for k, v in self.vec.vocabulary_.items()},
            "idf": self.vec.idf_.tolist(),
            "coef": coef.tolist(),
            "intercept": self.clf.intercept_.tolist(),
        }
        f = path / "router.json"
        f.write_text(json.dumps(payload))
        size_kb = f.stat().st_size // 1024
        print(f"wrote {f} ({size_kb} KB, {len(payload['vocabulary'])} features, "
              f"{len(self.classes_)} classes)")

        if size_kb > 2000:
            print("  NOTE: over 2MB. Lower max_features or raise min_df — "
                  "most of this is vocabulary, and rare n-grams contribute "
                  "little.")

    @classmethod
    def load_json(cls, path: str | Path) -> dict:
        return json.loads((Path(path) / "router.json").read_text())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def expected_calibration_error(conf: np.ndarray, correct: np.ndarray,
                               bins: int = 10) -> float:
    """
    Mean gap between stated confidence and actual accuracy, weighted by bin
    population. 0.0 is perfect. Above ~0.15 means the thresholds in
    labels.py are not measuring what they claim to.
    """
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def report(router: "LogRegRouter", texts: list[str], labels: list[str]) -> dict:
    """Accuracy, calibration, and the operational safety metric."""
    from sklearn.metrics import classification_report

    proba = router.predict_proba(texts)
    pred = np.array([router.classes_[i] for i in proba.argmax(1)])
    conf = proba.max(1)
    truth = np.array(labels)
    correct = pred == truth

    print(classification_report(truth, pred, zero_division=0))

    ece = expected_calibration_error(conf, correct)
    print(f"Accuracy: {correct.mean():.3f}   ECE: {ece:.3f}")
    print("\nReliability (stated confidence vs actual accuracy):")
    for lo, hi in [(0.0, 0.4), (0.4, 0.6), (0.6, 0.85), (0.85, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            print(f"  [{lo:.2f},{hi:.2f})  n={m.sum():4d}  "
                  f"stated {conf[m].mean():.3f}  actual {correct[m].mean():.3f}")

    # The number that decides whether this is safe to ship.
    from labels import LOCAL_ACTION_THRESHOLD, LOCAL_LABELS
    acting = np.array([p in LOCAL_LABELS for p in pred]) & (conf >= LOCAL_ACTION_THRESHOLD)
    silent_errors = int((acting & ~correct).sum())
    print(f"\nLocal actions at >={LOCAL_ACTION_THRESHOLD}: {int(acting.sum())}, "
          f"wrong: {silent_errors}")
    if acting.sum():
        print(f"  silent error rate: {silent_errors / acting.sum():.1%} "
              f"— each one is the phone acting on something unsaid")

    return {"accuracy": float(correct.mean()), "ece": ece,
            "local_actions": int(acting.sum()), "silent_errors": silent_errors}
