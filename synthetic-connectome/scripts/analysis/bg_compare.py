"""Stage 26 comparison: action selection by the basal-ganglia-thalamus loop vs the hand-made
WTA groups vs no selection circuit, on the v2 column with the quick rule (150 trials).

Reads results/bg/learn_<cfg>_seed*.json (loop; cfg from results/bg/learn_configs.json),
results/bg/learn_none_seed*.json (no circuit), results/v2/quick_2area_seed*.json (WTA),
and the eight-area equivalents (learn8_*, results/v2/quick_8area_seed*.json).
Writes results/bg/learning_comparison.json and prints the table."""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
os.chdir(ROOT)


def summarise(files):
    rows = []
    for f in sorted(files):
        if f.endswith(".partial.json"):
            continue
        d = json.load(open(f))
        c = np.array(d["curve"], float)
        first75 = next((i + 20 for i in range(len(c) - 19) if c[i:i + 20].mean() >= 0.75), None)
        row = dict(seed=d["seed"], acc_last50=float(c[-50:].mean()), acc_last40=float(c[-40:].mean()),
                   reach75=first75, dominance=d.get("dominance"), wall_s=d.get("wall_s"))
        if d.get("bg"):
            log = d["bg"]["log"]
            snr = np.array([e["snr"] for e in log]); th = np.array([e["thal"] for e in log])
            ch = np.array(d["choices"]) - 1
            cues = np.array([0 if x == "A" else 1 for x in d["cues"]])
            n = len(ch)
            row.update(bg_argmin_snr_matches_choice=float((snr.argmin(1) == ch).mean()),
                       bg_argmin_snr_correct_last50=float((snr.argmin(1)[-50:] == cues[-50:]).mean()),
                       bg_below_5hz_frac=d["bg"]["selected_frac"],
                       snr_winner_hz=float(snr[np.arange(n), ch].mean()), snr_loser_hz=float(snr[np.arange(n), 1 - ch].mean()),
                       thal_winner_hz=float(th[np.arange(n), ch].mean()), thal_loser_hz=float(th[np.arange(n), 1 - ch].mean()))
        rows.append(row)
    if not rows:
        return None
    keys = [k for k in rows[0] if k not in ("seed",) and all(r.get(k) is not None for r in rows)]
    agg = {k: dict(mean=float(np.mean([r[k] for r in rows])), sd=float(np.std([r[k] for r in rows])),
                   min=float(np.min([r[k] for r in rows])), max=float(np.max([r[k] for r in rows]))) for k in keys}
    return dict(n=len(rows), seeds=[r["seed"] for r in rows], per_network=rows, **agg)


def main():
    cfgs = json.load(open("results/bg/learn_configs.json"))
    out = {"about": "Stage 26: the measured basal-ganglia-thalamus loop vs the WTA stand-in vs no selection circuit; "
                    "v2 column, quick rule, 150 trials; accuracy over the last 50 trials; same network seeds per row block",
           "two_areas": {}, "eight_areas": {}}
    out["two_areas"]["wta (stage 25)"] = summarise(glob.glob("results/v2/quick_2area_seed*.json"))
    out["two_areas"]["none"] = summarise(glob.glob("results/bg/learn_none_seed*.json"))
    for name, kw in cfgs.items():
        s = summarise(glob.glob(f"results/bg/learn_{name}_seed*.json"))
        if s:
            s["bg_options"] = kw
            out["two_areas"][f"bg {name}"] = s
    out["eight_areas"]["wta (stage 25)"] = summarise(glob.glob("results/v2/quick_8area_seed*.json"))
    out["eight_areas"]["none"] = summarise(glob.glob("results/bg/learn8_none_seed*.json"))
    for name, kw in cfgs.items():
        s = summarise(glob.glob(f"results/bg/learn8_{name}_seed*.json"))
        if s:
            s["bg_options"] = kw
            out["eight_areas"][f"bg {name}"] = s
    json.dump(out, open("results/bg/learning_comparison.json", "w"), indent=1)
    for block in ("two_areas", "eight_areas"):
        print(f"\n{block}")
        for k, s in out[block].items():
            if not s:
                continue
            extra = ""
            if "snr_winner_hz" in s:
                extra = (f"  SNr win/lose {s['snr_winner_hz']['mean']:.1f}/{s['snr_loser_hz']['mean']:.1f} Hz"
                         f"  thal {s['thal_winner_hz']['mean']:.1f}/{s['thal_loser_hz']['mean']:.1f}"
                         f"  argmin-SNr correct (last 50) {s['bg_argmin_snr_correct_last50']['mean']:.2f}")
            print(f"  {k:16s} n={s['n']}  last50 {100*s['acc_last50']['mean']:.1f} ± {100*s['acc_last50']['sd']:.1f} %"
                  f"  75% at {s['reach75']['min']:.0f}-{s['reach75']['max']:.0f}  {s['wall_s']['mean']/150:.1f} s/trial" + extra)


if __name__ == "__main__":
    main()
