"""Stage 28 summary: the relay as a share of the pools' drive.  Reads results/thal/*_seed*.json and the
Stage 26/27 references; writes results/thal/summary.json and prints the table.
"Reaches 75 %" = first trial at which the previous 20 average >= 75 %; "longest early run" = longest
run of one action in the first 40 choices (perseveration)."""
import glob, json, os, re
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def one(f):
    d = json.load(open(f)); c = np.array(d["curve"], float)
    t75 = next((i + 20 for i in range(len(c) - 19) if c[i:i + 20].mean() >= 0.75), None)
    ch = d["choices"][:40]; run = best = 1
    for a, b in zip(ch, ch[1:]):
        run = run + 1 if a == b else 1; best = max(best, run)
    row = dict(seed=d["seed"], acc_last50=float(c[-50:].mean()), reach75=t75, early_run=best,
               blocks30=[round(float(c[i:i + 30].mean()), 2) for i in range(0, len(c), 30)],
               l5e_rest_hz=d["rates"].get("9/46d:L5E"), s_per_trial=d["wall_s"] / d["trials"])
    if d.get("bg") and d["bg"].get("log"):
        log = d["bg"]["log"]; th = np.array([e["thal"] for e in log]); snr = np.array([e["snr"] for e in log])
        chs = np.array(d["choices"]) - 1; n = len(chs)
        cues = np.array([0 if x == "A" else 1 for x in d["cues"]])
        row.update(thal_win=float(th[np.arange(n), chs].mean()), thal_lose=float(th[np.arange(n), 1 - chs].mean()),
                   snr_correct_last50=float((snr.argmin(1)[-50:] == cues[-50:]).mean()))
    return row


def agg(files):
    rows = [one(f) for f in sorted(files) if not f.endswith(".partial.json")]
    if not rows:
        return None
    out = dict(n=len(rows), per_network=rows)
    for k in ("acc_last50", "reach75", "early_run", "l5e_rest_hz", "s_per_trial", "thal_win", "thal_lose", "snr_correct_last50"):
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            out[k] = dict(mean=float(np.mean(v)), sd=float(np.std(v)), min=float(min(v)), max=float(max(v)))
    return out


def main():
    res = dict(about=__doc__, configs=json.load(open("results/thal/configs.json")), rows={})
    names = sorted({re.sub(r"_seed\d+\.json$", "", os.path.basename(f)) for f in glob.glob("results/thal/*_seed*.json")
                    if not f.endswith(".partial.json")})
    for n in names:
        res["rows"][n] = agg(glob.glob(f"results/thal/{n}_seed*.json"))
    res["rows"]["ref: loop only, no routing (Stage 27)"] = agg(glob.glob("results/bg/plastic_dl_g1.5_e0.05_seed*.json"))
    res["rows"]["ref: full system, no routing (Stage 27)"] = agg(glob.glob("results/bg/plastic_dasys_seed*.json"))
    res["rows"]["ref: WTA groups (Stage 25)"] = agg(glob.glob("results/v2/quick_2area_seed*.json"))
    json.dump(res, open("results/thal/summary.json", "w"), indent=1)
    for n, s in res["rows"].items():
        if not s:
            continue
        r75 = s.get("reach75"); th = (f"  thal win/lose {s['thal_win']['mean']:.0f}/{s['thal_lose']['mean']:.0f}"
                                     if "thal_win" in s else "")
        print(f"{n:42s} n={s['n']}  {100 * s['acc_last50']['mean']:.1f} ± {100 * s['acc_last50']['sd']:.1f} %"
              f"  75% at {('%d-%d' % (r75['min'], r75['max'])) if r75 else 'never'}"
              f"  early run {s['early_run']['mean']:.0f}  L5E {s['l5e_rest_hz']['mean']:.1f} Hz  {s['s_per_trial']['mean']:.1f} s/trial{th}")


if __name__ == "__main__":
    main()
