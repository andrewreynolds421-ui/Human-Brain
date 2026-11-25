"""Stage 80/82 summary: bAbI (20 tasks, joint 10k) per network, against Stage 79 on the same reading networks (paired by seed).

    python scripts/analysis/stage80.py            # Stage 80 (episodes_) vs Stage 79
    python scripts/analysis/stage80.py concepts_  # Stage 82 vs Stage 80
"""
import json, os, sys
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "babi_all")
NEW = sys.argv[1] if len(sys.argv) > 1 else "episodes_"
OLD = {"episodes_": "", "concepts_": "episodes_"}[NEW]


def load(tag):
    out = {}
    for s in range(4):
        f = os.path.join(R, f"{tag}s{s}.json")
        if os.path.exists(f): out[s] = json.load(open(f))
    return out


def pct(v): return round(100 * v, 1) if v <= 1 else round(v, 1)


def main():
    new, old = load(NEW), load(OLD); seeds = sorted(new)
    tasks = list(new[seeds[0]]["tasks"])
    summ = dict(new=NEW or "stage79", old=OLD or "stage79", seeds=seeds,
                passed=[new[s]["passed_95"] for s in seeds], passed_old=[old[s]["passed_95"] for s in seeds if s in old],
                mean_error=[round(new[s]["mean_error"], 2) for s in seeds], mean_error_old=[round(old[s]["mean_error"], 2) for s in seeds if s in old],
                tasks={t: dict(new=[pct(new[s]["tasks"][t]) for s in seeds], old=[pct(old[s]["tasks"][t]) for s in seeds if s in old]) for t in tasks},
                world={k: [pct(new[s]["world"][k]) for s in seeds] for k in new[seeds[0]]["world"]})
    paired = [s for s in seeds if s in old]
    d = [old[s]["mean_error"] - new[s]["mean_error"] for s in paired]
    summ["error_drop_paired"] = dict(mean=round(float(np.mean(d)), 2), each=[round(x, 2) for x in d], better_in=f"{sum(x > 0 for x in d)}/{len(d)}")
    for t in tasks:
        n = [pct(new[s]["tasks"][t]) for s in paired]; o = [pct(old[s]["tasks"][t]) for s in paired]
        summ["tasks"][t]["change"] = round(float(np.mean(np.subtract(n, o))), 1)
    json.dump(summ, open(os.path.join(R, f"stage{80 if NEW == 'episodes_' else 82}_summary.json"), "w"), indent=1)
    print("passed", summ["passed"], "was", summ["passed_old"]); print("mean error", summ["mean_error"], "was", summ["mean_error_old"], summ["error_drop_paired"])
    for t, v in summ["tasks"].items(): print(f"{t:32s} {np.mean(v['new']):6.1f}  was {np.mean(v['old']) if v['old'] else float('nan'):6.1f}  change {v['change']:+5.1f}  {v['new']}")
    print("world", summ["world"])


if __name__ == "__main__":
    main()
