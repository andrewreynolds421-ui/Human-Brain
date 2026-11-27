#!/usr/bin/env python3
"""Stage 98: VEGA's reading network packaged for RASPUTIN (the Warmind backend), shadow mode.

Writes, into the backend's vega/ folder:
  reader.npz   the Stage 94 large reading network (gru1024x2), half precision
  vocab.json   its words and word endings
  head.json    a decision layer on top of VEGA's reading of a request: which of RASPUTIN's ten destinations
               (intent/labels.py) the request belongs to.  Trained on the router's own seed phrases.

    READER_BOOKS=bundle python scripts/export_rasputin.py /home/claude/warmind-backend
"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("READER_BOOKS", "bundle")
import big_reader as BR                    # noqa: E402
from reader_adapter import BigM            # noqa: E402

READER = os.environ.get("VEGA_READER", "/home/claude/models/big_reader/bigreader_gru1024x2_s0.pt")


def toks(t):
    return re.findall(r"[a-z]+|\d+", t.lower())


def main(backend):
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from scipy.optimize import minimize_scalar
    sys.path.insert(0, os.path.join(backend, "intent"))
    from bench_flyhash import DATA
    from labels import LABELS
    M = BigM(torch, READER); out = os.path.join(backend, "vega"); os.makedirs(out, exist_ok=True)
    sd = torch.load(READER)
    np.savez(os.path.join(out, "reader.npz"), **{k.replace(".", "_"): v.numpy() for k, v in sd.items()})
    json.dump({"words": M.VB.itos, "ends": M.VB.ends, "eos": BR.EOS, "reader": os.path.basename(READER)}, open(os.path.join(out, "vocab.json"), "w"))

    def feat(t):
        w = toks(t) or ["a"]; x, en = M._tensors([[BR.EOS] + w])
        with torch.no_grad(): h = M.net.states(x, en)[0, 1:]; e = M.net.embed(x, en)[0, 1:]
        return torch.cat([h.mean(0), h[-1], e.mean(0)]).numpy()
    rows = [(t, l) for l in LABELS for t in DATA[l]]
    X = np.stack([feat(t) for t, _ in rows]); y = np.array([LABELS.index(l) for _, l in rows])
    mu, sd_ = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd_ / np.sqrt(X.shape[1])
    # temperature from held-out folds, so a stated confidence means what it says (as the letters router does)
    logits = np.zeros((len(y), len(LABELS)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y):
        c = LogisticRegression(max_iter=3000, class_weight="balanced", C=10).fit(Z[tr], y[tr]); logits[te] = c.decision_function(Z[te])
    def nll(lt):
        z = logits / np.exp(lt); z = z - z.max(1, keepdims=True); p = np.exp(z) / np.exp(z).sum(1, keepdims=True); return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    T = float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x)); cv = float((logits.argmax(1) == y).mean())
    c = LogisticRegression(max_iter=3000, class_weight="balanced", C=10).fit(Z, y)
    json.dump({"labels": LABELS, "mean": mu.round(5).tolist(), "std": sd_.round(5).tolist(), "coef": c.coef_.round(5).tolist(),
               "intercept": c.intercept_.round(5).tolist(), "temperature": T, "trained_on": len(y), "held_out_accuracy": round(cv, 3)},
              open(os.path.join(out, "head.json"), "w"))
    probes = ["open spotify", "where is kyiv", "what is apple trading at", "never mind", "how are my steps trending", "why is the sky blue"]
    json.dump([{"text": t, "feature_head": feat(t)[:8].round(4).tolist(), "feature_sum": round(float(feat(t).sum()), 3)} for t in probes],
              open(os.path.join(out, "parity.json"), "w"), indent=1)
    print("held-out accuracy", cv, "temperature", T, {f: os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)})


if __name__ == "__main__":
    main(sys.argv[1])
