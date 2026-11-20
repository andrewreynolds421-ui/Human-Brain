"""Stage 63 summary: the books + speech mix with a register marker, tested under each marker, against Stage 62's
unmarked mix, books only (Stage 61 awake) and speech only (results/register -> stage63_summary.json)."""
import glob, json, os
from collections import defaultdict
import numpy as np
RR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
R = os.path.join(RR, "register")
PATHS = {"books": os.path.join(RR, "sleep_grammar", "awake_s[0-9].json"), "speech": os.path.join(RR, "childes", "speech", "awake_s[0-9].json"),
         "mix": os.path.join(RR, "childes", "books_speech", "awake_s[0-9].json"),
         "marked_book": os.path.join(R, "book", "awake_s[0-9].json"), "marked_talk": os.path.join(R, "talk", "awake_s[0-9].json"),
         "both_book": os.path.join(R, "both_book", "awake_s[0-9].json"), "both_talk": os.path.join(R, "both_talk", "awake_s[0-9].json")}   # Stage 64: pages under both markers
TAUGHT = ("determiner_noun_agreement", "anaphor_agreement", "binding", "npi_licensing", "filler_gap_dependency", "subject_verb_agreement")


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
        out[c] = dict(networks=len(runs), blimp_all=ms([r["D"]["all"] for r in runs]),
                      blimp_taught_fields=ms([np.mean([v["all"] for v in r["D"]["by_phenomenon"].values() if v["field"] in TAUGHT]) for r in runs]),
                      blimp_other_fields=ms([np.mean([v["all"] for v in r["D"]["by_phenomenon"].values() if v["field"] not in TAUGHT]) for r in runs]),
                      by_field={f: ms(v) for f, v in sorted(field.items())},
                      by_phenomenon={k: ms([r["D"]["by_phenomenon"][k]["all"] for r in runs]) for k in runs[0]["D"]["by_phenomenon"]},
                      built={k: ms([r["built"][k] for r in runs]) for k in runs[0]["built"]},
                      books_agreement=ms([r["E"]["sensitivity_auc"] for r in runs]), word_order=ms([r["C"]["two_neighbours_swapped"] for r in runs]),
                      val_logp=round(float(np.mean([r["val_logp_per_word"] for r in runs])), 3))
    for kb, kt, name in (("marked_book", "marked_talk", "talk_minus_book"), ("both_book", "both_talk", "both_talk_minus_book")):   # per network, talk - book marker (same weights)
        if kb not in out or kt not in out: continue
        b = {os.path.basename(f): json.load(open(f)) for f in glob.glob(PATHS[kb])}; t = {os.path.basename(f): json.load(open(f)) for f in glob.glob(PATHS[kt])}
        both = sorted(set(b) & set(t))
        if both:
            out[name] = dict(blimp_all=ms([t[k]["D"]["all"] - b[k]["D"]["all"] for k in both]),
                                          built={q: ms([t[k]["built"][q] - b[k]["built"][q] for k in both]) for q in b[both[0]]["built"]})
    json.dump(out, open(os.path.join(R, "stage63_summary.json"), "w"), indent=1)
    cs = [c for c in PATHS if c in out]
    for k in ("blimp_all", "blimp_taught_fields", "blimp_other_fields", "books_agreement", "word_order", "val_logp"):
        print(f"{k:42s}", " | ".join(f"{c}: {out[c][k]}" for c in cs))
    print("-- fields")
    for f in out[cs[0]]["by_field"]:
        print(f"{f:42s}", " | ".join(f"{c}: {out[c]['by_field'][f][:2]}" for c in cs))
    print("-- built")
    for k in out[cs[0]]["built"]:
        print(f"{k:42s}", " | ".join(f"{c}: {out[c]['built'][k][:2]}" for c in cs), "| diff", out.get("talk_minus_book", {}).get("built", {}).get(k, [None])[:2], out.get("both_talk_minus_book", {}).get("built", {}).get(k, [None])[:2])


if __name__ == "__main__":
    main()
