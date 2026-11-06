"""Stage 29 summary: (a) loop/cortex balance confirmed on held-out networks 35-38, (b) eight areas and
rule reversal with the routed loop, (c) temporal-difference dopamine (burst at the cue).
Writes results/thal/stage29_summary.json and prints the tables."""
import glob, json, os
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def load(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(pat)) if not f.endswith(".partial.json")]


def t75(c, start=0):
    c = np.asarray(c[start:], float)
    return next((i + 20 for i in range(len(c) - 19) if c[i:i + 20].mean() >= 0.75), None)


def early_run(ch):
    run = best = 1
    for a, b in zip(ch[:40], ch[1:40]):
        run = run + 1 if a == b else 1; best = max(best, run)
    return best


def ms(v):
    v = [x for x in v if x is not None]
    return None if not v else dict(mean=round(float(np.mean(v)), 3), sd=round(float(np.std(v)), 3),
                                   min=float(min(v)), max=float(max(v)), n=len(v))


def learning(runs):
    if not runs:
        return None
    return dict(seeds=[d["seed"] for d in runs], acc_last50=ms([float(np.mean(d["curve"][-50:])) for d in runs]),
                reach75=ms([t75(d["curve"]) for d in runs]), early_run=ms([early_run(d["choices"]) for d in runs]),
                s_per_trial=ms([d["wall_s"] / d["trials"] for d in runs]),
                per_network=[dict(seed=d["seed"], acc_last50=float(np.mean(d["curve"][-50:])), reach75=t75(d["curve"]),
                                  early_run=early_run(d["choices"])) for d in runs])


def reversal(runs, at=100):
    if not runs:
        return None
    rows = []
    for d in runs:
        c = np.asarray(d["curve"], float)
        rows.append(dict(seed=d["seed"], before=float(c[at - 30:at].mean()), first30_after=float(c[at:at + 30].mean()),
                         end=float(c[-30:].mean()), relearn_to75=t75(c, at)))
    return dict(seeds=[r["seed"] for r in rows], before=ms([r["before"] for r in rows]),
                first30_after=ms([r["first30_after"] for r in rows]), end=ms([r["end"] for r in rows]),
                relearn_to75=ms([r["relearn_to75"] for r in rows]),
                never_relearned=sum(r["relearn_to75"] is None for r in rows), per_network=rows)


def dopamine(runs):
    if not runs or not runs[0].get("dopamine"):
        return None
    out = []
    for d in runs:
        dl = d["dopamine"]["log"]; R = np.array([e["R"] for e in dl])
        def blk(key, sl, cond=None):
            v = np.array([e.get(key, np.nan) for e in dl], float)[sl]
            m = R[sl] > 0 if cond == "rew" else (R[sl] < 0 if cond == "err" else np.ones(v.size, bool))
            return float(np.nanmean(v[m])) if m.any() else None
        f_, l_ = slice(0, 30), slice(-30, None)
        out.append(dict(seed=d["seed"],
                        da_cue_first30=blk("da_cue_hz", f_), da_cue_last30=blk("da_cue_hz", l_),
                        da_reward_first30=blk("da_hz", f_, "rew"), da_reward_last30=blk("da_hz", l_, "rew"),
                        da_error_last30=blk("da_hz", l_, "err"),
                        value_first30=blk("value_hz", f_), value_last30=blk("value_hz", l_),
                        critic_first30=blk("critic_hz", f_), critic_last30=blk("critic_hz", l_)))
    keys = [k for k in out[0] if k != "seed"]
    return dict(per_network=out, **{k: ms([o[k] for o in out]) for k in keys})


def main():
    T = "results/thal/"
    res = {"about": __doc__}
    res["a_heldout_35_38"] = {k: learning(load(T + f"heldout_{k}_seed*.json")) for k in ("bal_eta05", "route15_full", "wta")}
    res["a_scan_31_32"] = {k: learning(load(T + f"{k}_seed3[12].json")) for k in ("bal_eta05", "bal_eta025", "bal_route10", "route15_full")}
    res["b_reversal_31_34"] = {"loop (balanced, routed)": reversal(load(T + "reversal_bal_eta05_seed*.json")),
                               "WTA groups": reversal(load(T + "reversal_wta_seed*.json"))}
    res["b_eight_areas_9001_9004"] = {"loop (balanced, routed)": learning(load(T + "areas8_bal_eta05_seed*.json")),
                                      "loop, no routing (Stage 26)": learning(load("results/bg/learn8_A_seed*.json")),
                                      "WTA groups (Stage 25)": learning(load("results/v2/quick_8area_seed*.json"))}
    res["c_td"] = {"with value pathway": dict(learning=learning(load(T + "td_seed*.json")), dopamine=dopamine(load(T + "td_seed*.json"))),
                   "without (balanced, held-out nets)": dict(dopamine=dopamine(load(T + "heldout_bal_eta05_seed*.json"))),
                   "without (Stage 28, nets 31-34)": dict(dopamine=dopamine(load(T + "route15_full_seed*.json")))}
    json.dump(res, open(T + "stage29_summary.json", "w"), indent=1)

    def line(name, s):
        if not s:
            return
        r = s["reach75"]
        print(f"  {name:34s} n={s['acc_last50']['n']}  {100 * s['acc_last50']['mean']:.1f} ± {100 * s['acc_last50']['sd']:.1f} %"
              f"  75% at {('%d-%d' % (r['min'], r['max'])) if r else '-'}  early run {s['early_run']['mean']:.0f}"
              f"  {s['s_per_trial']['mean']:.1f} s/trial")
    print("(a) held-out 35-38"); [line(k, v) for k, v in res["a_heldout_35_38"].items()]
    print("(a) scan 31-32"); [line(k, v) for k, v in res["a_scan_31_32"].items()]
    print("(b) eight areas"); [line(k, v) for k, v in res["b_eight_areas_9001_9004"].items()]
    print("(b) reversal at 100")
    for k, v in res["b_reversal_31_34"].items():
        if v:
            print(f"  {k:34s} before {100 * v['before']['mean']:.0f}%  first30 after {100 * v['first30_after']['mean']:.0f}%"
                  f"  end {100 * v['end']['mean']:.0f} ± {100 * v['end']['sd']:.0f}%  relearn to 75% "
                  f"{('%d-%d' % (v['relearn_to75']['min'], v['relearn_to75']['max'])) if v['relearn_to75'] else '-'}"
                  f"  never {v['never_relearned']}")
    print("(c) dopamine")
    for k, v in res["c_td"].items():
        d = v["dopamine"]
        if d:
            g = lambda x: (f"{d[x]['mean']:.1f}" if d.get(x) else "-")
            print(f"  {k:34s} DA at cue {g('da_cue_first30')} -> {g('da_cue_last30')} Hz; at reward {g('da_reward_first30')} -> "
                  f"{g('da_reward_last30')} Hz; error {g('da_error_last30')}; value {g('value_first30')} -> {g('value_last30')}; "
                  f"critic {g('critic_first30')} -> {g('critic_last30')}")
        if v.get("learning"):
            line("  learning", v["learning"])


if __name__ == "__main__":
    main()
