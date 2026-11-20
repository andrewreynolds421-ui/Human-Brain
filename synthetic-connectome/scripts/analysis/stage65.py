"""Stage 65 summary: output words ordered by frequency against the same groups with alphabetical order
(results/output_order -> stage65_summary.json). books = Stage 61 awake; both_* = Stage 64 under each marker."""
import glob, json, os
from collections import defaultdict
import numpy as np
RR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
R = os.path.join(RR, "output_order")
PATHS = {"books_alpha": os.path.join(RR, "sleep_grammar", "awake_s[0-9].json"), "books_freq": os.path.join(R, "books", "awake_s[0-9].json"),
         "both_book_alpha": os.path.join(RR, "register", "both_book", "awake_s[0-9].json"), "both_book_freq": os.path.join(R, "both_book", "awake_s[0-9].json"),
         "both_talk_alpha": os.path.join(RR, "register", "both_talk", "awake_s[0-9].json"), "both_talk_freq": os.path.join(R, "both_talk", "awake_s[0-9].json")}


def ms(v):
    v = 100 * np.asarray(v, float); return [round(float(v.mean()), 1), round(float(v.std(ddof=1)), 1) if len(v) > 1 else None, len(v)]


def main():
    out = {}
    for c, p in PATHS.items():
        runs = [json.load(open(f)) for f in sorted(glob.glob(p))]
        if not runs: continue
        field = defaultdict(list)
        for r in runs:
            per = defaultdict(list)
            for k, v in r["D"]["by_phenomenon"].items(): per[v["field"]].append(v["all"])
            for f, v in per.items(): field[f].append(np.mean(v))
        out[c] = dict(networks=len(runs), blimp_all=ms([r["D"]["all"] for r in runs]), by_field={f: ms(v) for f, v in sorted(field.items())},
                      by_phenomenon={k: ms([r["D"]["by_phenomenon"][k]["all"] for r in runs]) for k in runs[0]["D"]["by_phenomenon"]},
                      built={k: ms([r["built"][k] for r in runs]) for k in runs[0]["built"]},
                      books_agreement=ms([r["E"]["sensitivity_auc"] for r in runs]), word_order=ms([r["C"]["two_neighbours_swapped"] for r in runs]),
                      val_logp=round(float(np.mean([r["val_logp_per_word"] for r in runs])), 3),
                      minutes_per_pass=round(float(np.mean([t["minutes"] for r in runs for t in r["training"]])), 1))
    json.dump(out, open(os.path.join(R, "stage65_summary.json"), "w"), indent=1)
    cs = list(out)
    for k in ("blimp_all", "books_agreement", "word_order", "val_logp", "minutes_per_pass"):
        print(f"{k:40s}", " | ".join(f"{c}: {out[c][k]}" for c in cs))
    print("-- fields")
    for f in out[cs[0]]["by_field"]: print(f"{f:40s}", " | ".join(f"{c}: {out[c]['by_field'][f][:2]}" for c in cs))
    print("-- built")
    for k in out[cs[0]]["built"]: print(f"{k:40s}", " | ".join(f"{c}: {out[c]['built'][k][:2]}" for c in cs))


if __name__ == "__main__":
    main()
