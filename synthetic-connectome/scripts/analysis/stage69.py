"""Stage 69 summary: Stage 65's books networks before and after a night of replay (speech only / mixed), against
Stage 65's books + speech networks trained on the mix from the start (results/sleep_replay -> stage69_summary.json)."""
import glob, json, os
from collections import defaultdict
import numpy as np
RR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
R = os.path.join(RR, "sleep_replay")
PATHS = {"no_sleep": os.path.join(RR, "output_order", "books", "awake_s[0-9].json"), "sleep_speech": os.path.join(R, "speech", "books", "awake_s[0-9].json"),
         "sleep_mixed": os.path.join(R, "mixed", "books", "awake_s[0-9].json"), "trained_on_mix_book": os.path.join(RR, "output_order", "both_book", "awake_s[0-9].json"),
         "trained_on_mix_talk": os.path.join(RR, "output_order", "both_talk", "awake_s[0-9].json")}


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
                      books_agreement=ms([r["E"]["sensitivity_auc"] for r in runs]), word_order=ms([r["C"]["two_neighbours_swapped"] for r in runs]),
                      val_logp=round(float(np.mean([r["val_logp_per_word"] for r in runs])), 3))
    for c in ("speech", "mixed"):
        rep = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, c, "replay_s[0-9].json")))]
        if rep:
            out["replay_" + c] = dict(networks=len(rep), lessons=rep[0]["lessons"], sentences_replayed=rep[0]["sentences_replayed"],
                                      **{f"{w}_{d}_{k}": ms([r[w][d][k] for r in rep], 1) for w in ("before", "after") for d in ("books", "speech") for k in ("none", "old_segment")})
    json.dump(out, open(os.path.join(R, "stage69_summary.json"), "w"), indent=1)
    cs = [c for c in PATHS if c in out]
    for c in ("speech", "mixed"):
        if "replay_" + c in out: print("replay", c, {k: v[:2] if isinstance(v, list) else v for k, v in out["replay_" + c].items()})
    for k in ("blimp_all", "books_agreement", "word_order", "val_logp"): print(f"{k:38s}", " | ".join(f"{c}: {out[c][k][:2] if isinstance(out[c][k], list) else out[c][k]}" for c in cs))
    for f in out[cs[0]]["by_field"]: print(f"{f:38s}", " | ".join(f"{c}: {out[c]['by_field'][f][:2]}" for c in cs))
    for k in out[cs[0]]["built"]: print(f"{k:38s}", " | ".join(f"{c}: {out[c]['built'][k][:2]}" for c in cs))


if __name__ == "__main__":
    main()
