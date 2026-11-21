"""Stage 70 summary: days only vs days + nights, from Stage 65's books networks (results/day_night -> stage70_summary.json)."""
import glob, json, os
from collections import defaultdict
import numpy as np
RR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
R = os.path.join(RR, "day_night")
PATHS = {"start": os.path.join(RR, "output_order", "books", "awake_s[0-9].json"), "awake": os.path.join(R, "awake", "books", "awake_s[0-9].json"),
         "cycle": os.path.join(R, "cycle", "books", "awake_s[0-9].json")}
GRADE0 = os.path.join(RR, "grade_test", "books_s[0-9].json")


def ms(v, k=100):
    v = k * np.asarray(v, float); return [round(float(v.mean()), 1 if k == 100 else 3), round(float(v.std(ddof=1)), 1 if k == 100 else 3) if len(v) > 1 else None, len(v)]


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
        out[c] = dict(blimp_all=ms([r["D"]["all"] for r in runs]), by_field={f: ms(v) for f, v in sorted(field.items())}, built={k: ms([r["built"][k] for r in runs]) for k in runs[0]["built"]},
                      books_agreement=ms([r["E"]["sensitivity_auc"] for r in runs]), word_order=ms([r["C"]["two_neighbours_swapped"] for r in runs]))
        life = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, c, "life_s[0-9].json")))]
        if life:
            out[c]["grade"] = {b: ms([l["grade"][b] for l in life]) for b in life[0]["grade"]}; out[c]["grade_mean"] = ms([l["grade_mean"] for l in life])
            out[c]["curve"] = [dict(when=e["when"], speech=ms([l["curve"][i]["speech"] for l in life], 1), books=ms([l["curve"][i]["books"] for l in life], 1)) for i, e in enumerate(life[0]["curve"])]
    g0 = [json.load(open(f)) for f in sorted(glob.glob(GRADE0))]
    if g0 and "start" in out:
        out["start"]["grade"] = {b: ms([g["book"][b] for g in g0]) for b in g0[0]["book"]}; out["start"]["grade_mean"] = ms([np.mean(list(g["book"].values())) for g in g0])
    json.dump(out, open(os.path.join(R, "stage70_summary.json"), "w"), indent=1)
    cs = list(out)
    for c in cs:
        if "curve" in out[c]: print(c, [(e["when"], e["speech"][0], e["books"][0]) for e in out[c]["curve"]])
    for c in cs: print(c, "grade", {b: v[0] for b, v in out[c].get("grade", {}).items()}, out[c].get("grade_mean"))
    for k in ("blimp_all", "books_agreement", "word_order"): print(f"{k:38s}", " | ".join(f"{c}: {out[c][k][:2]}" for c in cs))
    for f in out[cs[0]]["by_field"]: print(f"{f:38s}", " | ".join(f"{c}: {out[c]['by_field'][f][:2]}" for c in cs))
    for k in out[cs[0]]["built"]: print(f"{k:38s}", " | ".join(f"{c}: {out[c]['built'][k][:2]}" for c in cs))


if __name__ == "__main__":
    main()
