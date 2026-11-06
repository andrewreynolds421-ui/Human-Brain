"""Stage 30 summary: intrinsic homeostasis in the loop's populations (relay; critic and value cells).
Writes results/hom/stage30_summary.json and prints the tables.  Late relay rates = mean relay rate of the
chosen / unchosen channel in the last 30 trials (the drift Stage 28/29 found)."""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage29 import load, learning, dopamine, ms  # noqa: E402

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def relays(runs):
    out = []
    for d in runs:
        log = d.get("bg", {}).get("log") if d.get("bg") else None
        if not log:
            continue
        th = np.array([e["thal"] for e in log]); ch = np.array(d["choices"]) - 1; n = len(ch)
        win, lose = th[np.arange(n), ch], th[np.arange(n), 1 - ch]
        out.append(dict(seed=d["seed"], win_first30=float(win[:30].mean()), lose_first30=float(lose[:30].mean()),
                        win_last30=float(win[-30:].mean()), lose_last30=float(lose[-30:].mean())))
    if not out:
        return None
    return {k: ms([o[k] for o in out]) for k in out[0] if k != "seed"}


def block(pattern):
    r = load(pattern)
    return dict(learning=learning(r), relay=relays(r), dopamine=dopamine(r))


def main():
    H, T = "results/hom/", "results/thal/"
    res = {"about": __doc__}
    res["scan_31_32"] = {n: block(H + f"{n}_seed3[12].json") for n in ("hom0.1", "hom0.5")}
    res["scan_31_32"]["no homeostasis (Stage 28)"] = block(T + "route15_full_seed3[12].json")
    res["heldout_35_38"] = {"relay homeostasis": block(H + "heldout_hom_seed*.json"),
                            "no homeostasis (Stage 28)": block(T + "heldout_route15_full_seed*.json"),
                            "WTA groups": block(T + "heldout_wta_seed*.json")}
    res["eight_areas"] = {"relay homeostasis": block(H + "areas8_hom_seed*.json"),
                          "no homeostasis (Stage 29b)": block(T + "areas8_bal_eta05_seed*.json"),
                          "WTA groups (Stage 25)": block("results/v2/quick_8area_seed*.json")}
    res["td"] = {"value x4 + critic x0.5 + homeostasis": block(H + "td_hom_seed*.json"),
                 "value x4 + critic x0.5, no homeostasis (Stage 29c)": block(T + "td3_seed*.json")}
    json.dump(res, open(H + "stage30_summary.json", "w"), indent=1)
    for part, rows in res.items():
        if part == "about":
            continue
        print(part)
        for name, b in rows.items():
            L = b["learning"]
            if not L:
                continue
            r = L["reach75"]
            line = (f"  {name:48s} n={L['acc_last50']['n']}  {100 * L['acc_last50']['mean']:.1f} ± {100 * L['acc_last50']['sd']:.1f} %"
                    f"  75% at {('%d-%d' % (r['min'], r['max'])) if r else '-'}  early {L['early_run']['mean']:.0f}")
            if b["relay"]:
                R = b["relay"]
                line += (f"  relay win/lose first {R['win_first30']['mean']:.0f}/{R['lose_first30']['mean']:.0f}"
                         f" last {R['win_last30']['mean']:.0f}/{R['lose_last30']['mean']:.0f}")
            if part == "td" and b["dopamine"]:
                D = b["dopamine"]; g = lambda k: (f"{D[k]['mean']:.1f}" if D.get(k) else "-")
                line += (f"  DA cue {g('da_cue_first30')}->{g('da_cue_last30')} reward {g('da_reward_first30')}->"
                         f"{g('da_reward_last30')} error {g('da_error_last30')} value {g('value_last30')} critic {g('critic_last30')}")
            print(line)


if __name__ == "__main__":
    main()
