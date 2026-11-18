"""Stage 60 summary: base vs phenomena pages (results/phenomena/*_s*.json -> stage60_summary.json).
BLiMP fields the pages teach vs fields they do not (control); built held-out pairs; key BLiMP sub-sets."""
import glob, json, os
from collections import defaultdict
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "phenomena")
TAUGHT = ("determiner_noun_agreement", "anaphor_agreement", "binding", "npi_licensing", "filler_gap_dependency", "subject_verb_agreement")


def ms(v):
    v = 100 * np.asarray(v, float); return [round(float(v.mean()), 1), round(float(v.std(ddof=1)), 1) if len(v) > 1 else None, len(v)]


def main():
    out = {}
    for c in ("base", "phenomena"):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, f"{c}_s[0-9].json")))]
        if not runs: continue
        ph = runs[0]["D"]["by_phenomenon"]
        field = defaultdict(list)
        for r in runs:
            per = defaultdict(list)
            for k, v in r["D"]["by_phenomenon"].items(): per[v["field"]].append(v["all"])
            for f, v in per.items(): field[f].append(np.mean(v))
        o = dict(networks=len(runs), blimp_all=ms([r["D"]["all"] for r in runs]),
                 blimp_taught_fields=ms([np.mean([v["all"] for v in r["D"]["by_phenomenon"].values() if v["field"] in TAUGHT]) for r in runs]),
                 blimp_other_fields=ms([np.mean([v["all"] for v in r["D"]["by_phenomenon"].values() if v["field"] not in TAUGHT]) for r in runs]),
                 by_field={f: ms(v) for f, v in sorted(field.items())},
                 by_phenomenon={k: ms([r["D"]["by_phenomenon"][k]["all"] for r in runs]) for k in ph},
                 built={k: ms([r["built"][k] for r in runs]) for k in runs[0]["built"]},
                 books_agreement=ms([r["E"]["sensitivity_auc"] for r in runs]), word_order=ms([r["C"]["two_neighbours_swapped"] for r in runs]),
                 val_logp=round(float(np.mean([r["val_logp_per_word"] for r in runs])), 3))
        out[c] = o
    json.dump(out, open(os.path.join(R, "stage60_summary.json"), "w"), indent=1)
    for k in ("blimp_all", "blimp_taught_fields", "blimp_other_fields", "books_agreement", "word_order", "val_logp"):
        print(f"{k:50s}", " | ".join(f"{c}: {o[k]}" for c, o in out.items()))
    print("-- fields")
    for f in next(iter(out.values()))["by_field"]:
        print(f"{f:50s}", " | ".join(f"{c}: {o['by_field'][f]}" for c, o in out.items()), "TAUGHT" if f in TAUGHT else "")
    print("-- built (held-out verbs / adjectives / prepositions)")
    for k in next(iter(out.values()))["built"]:
        print(f"{k:50s}", " | ".join(f"{c}: {o['built'][k]}" for c, o in out.items()))
    print("-- BLiMP sub-sets")
    first = next(iter(out.values()))["by_phenomenon"]
    for k in sorted(first, key=lambda k: first[k][0]):
        print(f"{k:50s}", " | ".join(f"{c}: {o['by_phenomenon'][k][0]}" for c, o in out.items()))


if __name__ == "__main__":
    main()
