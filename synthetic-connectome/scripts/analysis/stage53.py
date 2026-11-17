"""Stage 53 summary: graded teaching vs random order (results/graded/{curriculum,random}_s*.json -> stage53_summary.json)."""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "graded")
BANDS = ["1", "2", "3", "4", "5", "6", "7+"]


def main():
    out = {}
    for reg in ("curriculum", "random"):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, f"{reg}_s[0-9].json")))]
        if not runs: continue
        steps = sorted({t["words"] // 1_000_000 for t in runs[0]["tests"]})
        curve = {}
        for m in steps:
            v = [np.mean(list(next(t for t in r["tests"] if t["words"] // 1_000_000 == m)["cloze"].values())) for r in runs]
            curve[m] = [round(100 * float(np.mean(v)), 1), round(100 * float(np.std(v, ddof=1)), 1) if len(v) > 1 else None]
        final = {b: [round(100 * float(np.mean([r["tests"][-1]["cloze"][b] for r in runs])), 1), round(100 * float(np.std([r["tests"][-1]["cloze"][b] for r in runs], ddof=1)), 1)] for b in BANDS}
        o = dict(networks=len(runs), curve_mean_cloze_by_millions=curve, final_cloze_by_band=final,
                 final_mean=[round(100 * float(np.mean([np.mean(list(r["tests"][-1]["cloze"].values())) for r in runs])), 1),
                             round(100 * float(np.std([np.mean(list(r["tests"][-1]["cloze"].values())) for r in runs], ddof=1)), 1)],
                 final_logp_per_word={b: round(float(np.mean([r["tests"][-1]["logp_per_word"][b] for r in runs])), 3) for b in BANDS})
        if reg == "curriculum":
            o["promotions"] = {b: dict(millions_of_words=round(float(np.mean([next(p["words"] for p in r["promotions"] if p["band"] == b) for r in runs])) / 1e6, 2),
                                       mastered=sum(next(p["mastered"] for p in r["promotions"] if p["band"] == b) for r in runs))
                               for b in BANDS if all(any(p["band"] == b for p in r["promotions"]) for r in runs)}
        out[reg] = o
    json.dump(out, open(os.path.join(R, "stage53_summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
