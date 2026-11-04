"""Job sweep used for the learning-rule search: runs reward_learning.py over configs x seeds, two at a time (reads /tmp/configs.json, /tmp/g8.json).
(kept from the session's scratch files for reproducibility; paths may need adjusting)"""
"""run configs x seeds, 2 at a time; usage: sweep.py TAG TRIALS seeds... (configs from /tmp/configs.json)"""
import json, subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
tag, trials, seeds = sys.argv[1], sys.argv[2], sys.argv[3:]
base = json.load(open("/tmp/g8.json"))
cfgs = json.load(open("/tmp/configs.json"))
jobs = []
for name in cfgs:
    for s in seeds:
        out = f"/home/claude/synthetic-connectome/results/speed/{tag}_{name}_s{s}.json"
        if not os.path.exists(out):
            jobs.append((name, s, out))
os.makedirs("/home/claude/synthetic-connectome/results/speed", exist_ok=True)
def go(j):
    name, s, out = j
    g = dict(base, **cfgs[name]); ns = str(g.pop("_n_scale", 0.015))
    subprocess.run([sys.executable, "scripts/reward_learning.py", "--trials", trials, "--seed", s, "--fast", "--n-scale", ns,
                    "--genome", json.dumps(g), "--out", out], cwd="/home/claude/synthetic-connectome",
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with ThreadPoolExecutor(2) as ex:
    list(ex.map(go, jobs))
