"""Stage 38: letter-unit evidence for every image of the family-split letter set (data/letters_fam.npz), from
the V1-like complex cells (synth/v1.py) read by 26 letter units (logistic readout trained on the 209
training families, C chosen on the 30 validation families).  Layout for scripts/word_reading.py with
WORDS_LETTER_SET=families: E_train = training then validation families (239 x 156 images), E_test = 60 test
families.  Also writes the evidence of letter units on the pixels, for comparison.

    python scripts/analysis/stage38_evidence.py [--source v1|pixels]
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("v1", "pixels"), default="v1")
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    d = np.load(os.path.join(ROOT, "data", "letters_fam.npz"))
    if a.source == "v1":
        v = np.load(os.path.join(ROOT, "data", "letters_fam_v1.npz"))
        F = {k: v[f"V_{k}"] for k in ("train", "val", "test")}
    else:
        F = {k: d[f"X_{k}"].reshape(len(d[f"X_{k}"]), -1) / 255.0 for k in ("train", "val", "test")}
    y = {k: d[f"y_{k}"] for k in F}
    best = None
    for C in (0.03, 0.1, 0.3):
        m = LogisticRegression(C=C, max_iter=4000).fit(F["train"], y["train"])
        acc = m.score(F["val"], y["val"])
        print(f"C={C}: validation families {acc:.4f}", flush=True)
        if best is None or acc > best[0]:
            best = (acc, C, m)
    acc, C, m = best
    E_train = np.vstack([m.predict_proba(F["train"]), m.predict_proba(F["val"])])
    E_test = m.predict_proba(F["test"])
    out = os.path.join(ROOT, "results", "words", "evidence_fam.npz" if a.source == "v1" else "evidence_fam_pixels.npz")
    te = float((E_test.argmax(1) == y["test"]).mean())
    np.savez(out, E_train=E_train, E_test=E_test, C=C, val_acc=acc, test_acc=te)
    print(json.dumps(dict(source=a.source, C=C, validation=round(acc, 4), test_new_families=round(te, 4),
                          own_training_families=round(float((E_train[:len(y['train'])].argmax(1) == y['train']).mean()), 4))))


if __name__ == "__main__":
    main()
