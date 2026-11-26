"""Stage 86 summary: the two best hippocampus variants trained in days (joint / awake / sleep / reread), tested on
bAbI, VEGA's world stories and real-book passages by grade (memory, reading network alone, both).

    python scripts/analysis/stage86.py [runs_dir]      # default: the vega-runs checkout
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
RUNS = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/runs_branch/runs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "stage86")
DAYS = {1: [1, 6, 9, 10, 11, 12, 13], 2: [2, 3, 4, 5, 7, 8, 14], 3: [15, 16, 17, 18, 19, 20]}
BANDS = ["1", "2", "3", "4", "5", "6", "7+"]


def load():
    R = defaultdict(dict)
    for f in glob.glob(os.path.join(RUNS, "r86*_*", "*.json")):
        d = json.load(open(f)); c = d.get("config", {})
        if not c.get("ROUTINE"): continue
        var = "loop+controller" if c.get("CONTROLLER") else "loop+allcues"
        R[(var, c["ROUTINE"])][d["seed"]] = d                     # r86b_ (reread fix) and r86_ share names; later file wins
    return R


def m(v): v = [x for x in v if x is not None]; return round(float(np.mean(v)), 1) if v else None


def main():
    R = load(); os.makedirs(OUT, exist_ok=True); summ = {}
    for (var, r), runs in sorted(R.items()):
        ds = [runs[s] for s in sorted(runs)]
        e = dict(networks=sorted(runs), passed=[d["passed_95"] for d in ds], mean_error=m([d["mean_error"] for d in ds]),
                 world=m([100 * d["world"]["same"] for d in ds]))
        for key in ("passages", "passages_reader", "passages_both"):
            ok = [d for d in ds if isinstance(d.get(key), dict)]
            if ok: e[key] = {b: m([100 * d[key][b] for d in ok]) for b in BANDS}
        curve = defaultdict(lambda: defaultdict(list))                # after each day: mean accuracy on each day's tasks
        for d in ds:
            for c in d.get("curve", []):
                for k, ts in DAYS.items(): curve[c["after"]][f"day{k} tasks"].append(100 * np.mean([c[f"qa{t}"] for t in ts]))
                curve[c["after"]]["passages"].append(100 * np.mean([c[f"p{b}"] for b in BANDS if f"p{b}" in c]))
                curve[c["after"]]["world"].append(100 * c["world"])
        e["curve"] = {a: {k: m(v) for k, v in x.items()} for a, x in curve.items()}
        summ[f"{var} / {r}"] = e
    json.dump(summ, open(os.path.join(OUT, "stage86_summary.json"), "w"), indent=1)
    for k, e in summ.items():
        print(f"\n{k}: nets {e['networks']} passed {e['passed']} bAbI error {e['mean_error']} world {e['world']}")
        for key in ("passages", "passages_reader", "passages_both"):
            if key in e: print(f"   {key:16s}", " ".join(f"{b}:{e[key][b]}" for b in BANDS))
        for a, x in e["curve"].items(): print(f"   after {a:16s}", x)


if __name__ == "__main__":
    main()
