"""Stage 40 summary over networks: capitals learned without labels (results/capitals/learn_s*.json).

    python scripts/analysis/stage40.py  ->  results/capitals/stage40_summary.json
"""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "capitals")


def ms(v):
    v = 100 * np.asarray(v, float)
    return [round(float(v.mean()), 1), round(float(v.std(ddof=1)), 1) if len(v) > 1 else None]


def main():
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else ""          # "" = Stage 40; "_replay" = Stage 41
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, f"learn{tag}_s[0-9].json")))]
    conds = [k for k in runs[0] if isinstance(runs[0][k], dict) and "capitals_isolated" in runs[0][k]]
    out = dict(networks=[r["seed"] for r in runs])
    for c in conds:
        g = lambda f: [f(r[c]) for r in runs]
        o = dict(capital_first_letter=ms(g(lambda x: np.mean([x["words_Capital_known"]["capitals"], x["words_Capital_new"]["capitals"]]))),
                 all_caps_words=ms(g(lambda x: np.mean([x["words_ALLCAPS_known"]["capitals"], x["words_ALLCAPS_new"]["capitals"]]))),
                 each_capital_alone=ms(g(lambda x: x["capitals_isolated"])), lowercase_alone=ms(g(lambda x: x["lowercase_isolated"])))
        if "teacher_correct_overall" in runs[0][c]:
            o["teacher_correct"] = ms(g(lambda x: x["teacher_correct_overall"]))
            o["settings"] = [(r[c]["eta_w"], r[c].get("confidence_needed")) for r in runs]
        pc = np.mean([[r[c]["per_capital"][ch] for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] for r in runs], 0)
        o["worst_capitals"] = [("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i], round(float(pc[i]), 2)) for i in np.argsort(pc)[:6]]
        if "per_lowercase" in runs[0][c]:
            pl = np.mean([[r[c]["per_lowercase"][ch] for ch in "abcdefghijklmnopqrstuvwxyz"] for r in runs], 0)
            o["worst_lowercase"] = [("abcdefghijklmnopqrstuvwxyz"[i], round(float(pl[i]), 2)) for i in np.argsort(pl)[:4]]
        out[c] = o
        print(f"{c:30s} The-type {o['capital_first_letter']} | ALL CAPS {o['all_caps_words']} | alone {o['each_capital_alone']} | lowercase {o['lowercase_alone']} | worst {o['worst_capitals'][:4]}")
    out["lowercase_words_both_side_prediction"] = {n: dict(alone=ms([r["lowercase_words"][n]["letters_alone"] for r in runs]),
                                                           with_prediction=ms([r["lowercase_words"][n]["with_both_side_prediction"] for r in runs])) for n in ("known", "new")}
    print(out["lowercase_words_both_side_prediction"])
    json.dump(out, open(os.path.join(R, "stage40_summary.json" if not tag else f"stage41{tag}_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
