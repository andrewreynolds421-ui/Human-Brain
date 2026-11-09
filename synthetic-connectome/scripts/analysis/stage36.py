"""Stage 36 summary: letters and words read with and without the prediction, over networks, with
paired differences (same test words) against letters alone and against the shuffled-order control.

    python scripts/analysis/stage36.py results/words/proto_all.json [--out results/words/stage36_summary.json]
"""
import argparse, json
import numpy as np


def ms(v):
    v = np.asarray(v, float)
    return dict(mean=round(100 * v.mean(), 1), sd=round(100 * v.std(ddof=1), 1) if len(v) > 1 else None, n=len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R = [r for f in a.files for r in json.load(open(f))]
    norm = {r["seed"]: r for r in R if not r["shuffle"]}
    ctrl = {r["seed"]: r for r in R if r["shuffle"]}
    seeds = sorted(norm)
    models = [k for k in norm[seeds[0]] if k.startswith(("context_only", "sequence_layer", "context_and"))]
    out = {}
    for n in ("known", "held"):
        o = {}
        for metric in ("letter", "word"):
            alone = [norm[s][models[0]][n]["alone"][metric] for s in seeds]
            o[f"{metric}_alone"] = ms(alone)
            for m in models:
                w = [norm[s][m][n]["with_prediction"][metric] for s in seeds]
                d = dict(**ms(w), gain_vs_alone=ms(np.array(w) - alone))
                cs = [s for s in seeds if s in ctrl and m in ctrl[s]]
                if cs:
                    c = [ctrl[s][m][n]["with_prediction"][metric] for s in cs]
                    d["shuffled_control"] = ms(c)
                    d["gain_vs_control"] = ms(np.array([norm[s][m][n]["with_prediction"][metric] for s in cs]) - c)
                o[f"{metric}_{m}"] = d
            for g in ("ideal_1gram", "ideal_2gram", "ideal_3gram"):
                o[f"{metric}_{g}"] = ms([norm[s][g][n][metric] for s in seeds])
        for m in models:
            o[f"next_letter_{m}"] = dict(top1=ms([norm[s][m][n]["next_letter"]["top1"] for s in seeds]),
                                         bits=round(float(np.mean([norm[s][m][n]["next_letter"]["bits"] for s in seeds])), 2))
        for g in ("ideal_1gram", "ideal_2gram", "ideal_3gram"):
            o[f"next_letter_{g}"] = dict(top1=ms([norm[s][g][n]["next_letter"]["top1"] for s in seeds]),
                                         bits=round(float(np.mean([norm[s][g][n]["next_letter"]["bits"] for s in seeds])), 2))
        out[n] = o
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
