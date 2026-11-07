"""Stage 31 summary: how the relay homeostat responds — slower average (30 s) vs rest-gated (between
trials only), in eight areas (scan 9001-9002; held-out 9005-9008) and two areas (held-out 35-38).
Writes results/hom/stage31_summary.json."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage30 import block  # noqa: E402
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
H, T = "results/hom/", "results/thal/"
res = {"about": __doc__}
res["scan_8areas_9001_9002"] = {
    "slow average (30 s)": block(H + "areas8_slow30_seed900[12].json"),
    "rest-gated": block(H + "areas8_gated_seed900[12].json"),
    "5 s homeostat (Stage 30)": block(H + "areas8_hom_seed900[12].json"),
    "no homeostat, cortical x0.5 (29b)": block(T + "areas8_bal_eta05_seed900[12].json"),
    "no homeostat, full cortical rate": block(H + "areas8_nohom_fulleta_seed900[12].json"),
    "5 s homeostat, cortical x0.5": block(H + "areas8_hom_halfeta_seed900[12].json"),
    "25 % routed + 5 s homeostat": block(H + "areas8_r25_hom_seed900[12].json"),
    "25 % routed + rest-gated": block(H + "areas8_r25_gated_seed900[12].json")}
res["heldout_8areas_9005_9008"] = {
    "chosen homeostat": block(H + "areas8_chosen_seed900[5-8].json"),
    "no homeostat, routed": block(H + "areas8_nohom_seed900[5-8].json"),
    "WTA groups": block(H + "areas8_wta_seed900[5-8].json")}
res["heldout_2areas_35_38"] = {
    "chosen homeostat": block(H + "heldout_chosen_seed*.json"),
    "5 s homeostat (Stage 30)": block(H + "heldout_hom_seed*.json"),
    "WTA groups": block(T + "heldout_wta_seed*.json")}
json.dump(res, open(H + "stage31_summary.json", "w"), indent=1)
for part, rows in res.items():
    if part == "about":
        continue
    print(part)
    for name, b in rows.items():
        L = b["learning"]
        if not L:
            continue
        r = L["reach75"]; R = b["relay"]
        print(f"  {name:28s} n={L['acc_last50']['n']}  {100 * L['acc_last50']['mean']:.1f} ± {100 * L['acc_last50']['sd']:.1f} %"
              f"  75% at {('%d-%d' % (r['min'], r['max'])) if r else '-'}  early {L['early_run']['mean']:.0f}"
              + (f"  relay {R['win_first30']['mean']:.0f}/{R['lose_first30']['mean']:.0f} -> {R['win_last30']['mean']:.0f}/{R['lose_last30']['mean']:.0f}" if R else ""))
