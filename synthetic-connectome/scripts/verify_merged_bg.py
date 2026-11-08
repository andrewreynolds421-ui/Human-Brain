"""Stage 35 check: the loop's six populations merged into one NeuronGroup (BasalGanglia(merged=True))
must give the SAME learner run as the separate groups - every choice, reward, loop rate, dopamine
response and plastic weight, trial for trial - not merely similar statistics.

Each variant runs in its own process (Brian2 names count up within a process, and names set the
schedule order, which sets the random stream).  Compares the learner's whole JSON record except wall time.

    python scripts/verify_merged_bg.py                 # 4 configurations x 2 networks x 20 trials
    python scripts/verify_merged_bg.py --trials 40 --seeds 31,9001
"""
import argparse, json, os, subprocess, sys, time
import concurrent.futures as cf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENOME = {"eta": 13.1, "tau_elig_ms": 263, "a_minus": 0.73, "baseline_trials": 10, "w_max_factor": 30,
          "istdp_eta": 0.0163, "local_plastic": 0, "cue_fibres": 1000, "gate": 1, "normalize": 0, "wta": 8.0,
          "wta_in_mV": 3.0, "sp": 1, "sp_pool": 1.0, "sp_prune": 0.15, "sp_new": 0.5, "sp_every": 10,
          "surprise_gain": 3.0, "surprise_slow_trials": 50.0, "uncert_gain": 2.0}
CASES = [("r25_hom", "46d,9/46d"), ("r25_gated", "46d,9/46d"), ("td_hom2", "46d,9/46d"), ("r25_hom", "all")]


def run(cfg_name, areas, seed, merged, trials, outdir):
    cfg = json.load(open(os.path.join(ROOT, "results", "hom", "configs.json")))[cfg_name]
    bg = dict(cfg["bg"], merged=merged)
    tag = f"{cfg_name}_{'8' if areas == 'all' else '2'}_{seed}_{'m' if merged else 's'}"
    out = os.path.join(outdir, tag + ".json")
    cmd = [sys.executable, "scripts/reward_learning.py", "--trials", str(trials), "--seed", str(seed), "--fast",
           "--preset", "human_h01_dendritic_v2", "--areas", areas, "--genome", json.dumps(GENOME),
           "--selection", "bg", "--bg", json.dumps(bg), "--da", json.dumps(cfg["da"]), "--out", out]
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env=dict(os.environ, PYTHONPATH=ROOT))
    return tag, out, time.time() - t0


def strip(r):
    r = json.loads(json.dumps(r)); r.pop("wall_s", None)
    r.get("bg", {}).get("params", {}).pop("merged", None)        # the option itself is the one intended difference
    return r


def first_diff(a, b, path=""):
    if type(a) != type(b):
        return path or "/"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}"
            d = first_diff(a[k], b[k], f"{path}/{k}")
            if d: return d
        return None
    if isinstance(a, list):
        if len(a) != len(b): return f"{path}[len]"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d: return d
        return None
    return None if a == b else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", default="31,9001")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--outdir", default="/tmp/verify_merged_bg")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    jobs = [(c, ar, int(s), m) for c, ar in CASES for s in a.seeds.split(",") for m in (False, True)]
    done = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run, c, ar, s, m, a.trials, a.outdir): (c, ar, s, m) for c, ar, s, m in jobs}
        for f in cf.as_completed(futs):
            done[futs[f]] = f.result()
    rows, ok = [], True
    for c, ar, s in sorted({(c, ar, s) for c, ar, s, _ in jobs}):
        (_, fa, wa), (_, fb, wb) = done[(c, ar, s, False)], done[(c, ar, s, True)]
        ra, rb = json.load(open(fa)), json.load(open(fb))
        d = first_diff(strip(ra), strip(rb))
        ok &= d is None
        rows.append(dict(config=c, areas=8 if ar == "all" else 2, seed=s, identical=d is None, first_difference=d,
                         wall_separate_s=ra.get("wall_s"), wall_merged_s=rb.get("wall_s")))
    res = dict(trials=a.trials, rows=rows, all_identical=ok)
    print(json.dumps(res, indent=1))
    os.makedirs(os.path.join(ROOT, "results", "speed"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "speed", "verify_merged_bg.json"), "w"), indent=1)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
