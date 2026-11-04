"""Summarise 2-area reward-learning sweeps in results/speed by config (accuracy per 50 trials, trials to 75%).
    python scripts/analysis/score.py <tag>
(kept from the session's scratch files for reproducibility; paths may need adjusting)"""
import json, glob, sys, re
import numpy as np
tag = sys.argv[1]
rows = {}
for f in sorted(glob.glob(f"/home/claude/synthetic-connectome/results/speed/{tag}_*.json")):
    name = re.match(rf".*/{tag}_(.*)_s\d+\.json", f).group(1)
    rows.setdefault(name, []).append(json.load(open(f)))
def ttc(c, crit=0.75, win=30):
    r = np.convolve(c, np.ones(win) / win, "valid")
    hit = np.flatnonzero(r >= crit)
    return int(hit[0] + win) if hit.size else None
print(f"{'config':10s} n  1-50  51-100 101-150  last50  trials->75%(30-trial avg)   L5 9/46d Hz")
for name, rs in rows.items():
    C = np.array([r["curve"] for r in rs], float)
    blocks = [C[:, i:i + 50].mean() * 100 for i in range(0, C.shape[1], 50)]
    t = [ttc(c) for c in C]
    hz = np.mean([r["rates"].get("9/46d:L5E", np.nan) for r in rs])
    print(f"{name:10s} {len(rs)}  " + "  ".join(f"{b:5.1f}" for b in blocks) + f"   {np.mean(C[:, -50:])*100:5.1f}   {t}   {hz:.1f}")
