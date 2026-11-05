"""Test the basal-ganglia–thalamus loop on its own (Stage 26, task "loop alone").

Stand-in cortex: K Poisson pools (40 cells each) at a 3 spikes/s background (Humphries 2006
awake background), raised to a "salience" rate per channel.  Three parts, per seed:

1. rest      — all pools at background; per-population rates vs results/bg_parameters.json ranges
2. selection — the paper's protocol: channel 0 gets r1, later channel 1 gets r2, for all pairs
               (r1, r2) in {grid, 2 grid, ..., 40} spikes/s; a channel is selected when its SNr
               mean rate is below 5 spikes/s.  Scored: single selection of channel 0 in phase 1,
               switch to channel 1 in phase 2 when r2 > r1, channel 0 kept when r2 < r1,
               dual selections counted
3. thalamus  — relay rates of selected vs unselected channels (what the cortex would receive)

usage: python scripts/bg_selection_test.py --ctx-gain 2 --stn-gain 0.3 --seeds 1,2 --grid 8
           --out results/bg/scan_g2_s0.3.json   [--thal-drive 24 --da 0.3 --k 3 --quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

BACKGROUND_HZ = 3.0        # Humphries 2006: awake background cortical rate
N_POOL = 40                # stand-in pool size (real pools: ~30-60 L5 pyramids)


def run_seed(seed, K, grid, kw, phase_ms=1000.0, quick=False, smax=40):
    import brian2 as b2
    from brian2 import ms, Hz
    from synth.basal_ganglia import BasalGanglia, SELECT_HZ
    b2.start_scope()
    b2.prefs.codegen.target = "cython"
    b2.defaultclock.dt = 0.1 * ms
    b2.seed(seed)
    pools = [b2.PoissonGroup(N_POOL, rates=BACKGROUND_HZ * Hz, name=f"pool{k}") for k in range(K)]
    bg = BasalGanglia([(p, np.arange(N_POOL)) for p in pools], seed=seed, monitor="all", **kw)
    net = b2.Network(*pools, *bg.objects)
    # 1. rest
    net.run(2000 * ms)
    rest = {pop: bg.rates(pop, 500, 2000).tolist() for pop in bg.monitors}
    bg.clear()
    net.store("rest")
    # 2. selection grid
    sal = list(range(grid, smax + 1, grid))
    if quick:
        sal = sal[::2] or sal
    pairs = []
    meas = (0.4 * phase_ms, phase_ms)
    for r1 in sal:
        for r2 in sal:
            net.restore("rest")
            bg.clear()
            t0 = float(net.t / ms)
            pools[0].rates = r1 * Hz
            net.run(phase_ms * ms)
            snr1 = bg.rates("SNr", t0 + meas[0], t0 + meas[1])
            th1 = bg.rates("Thal", t0 + meas[0], t0 + meas[1])
            pools[1].rates = r2 * Hz
            net.run(phase_ms * ms)
            snr2 = bg.rates("SNr", t0 + phase_ms + meas[0], t0 + phase_ms + meas[1])
            th2 = bg.rates("Thal", t0 + phase_ms + meas[0], t0 + phase_ms + meas[1])
            d1 = bg.rates("D1", t0 + phase_ms + meas[0], t0 + phase_ms + meas[1])
            stn = bg.rates("STN", t0 + phase_ms + meas[0], t0 + phase_ms + meas[1])
            sel1 = [int(k) for k in np.flatnonzero(snr1 < SELECT_HZ)]
            sel2 = [int(k) for k in np.flatnonzero(snr2 < SELECT_HZ)]
            pairs.append(dict(r1=r1, r2=r2, snr_phase1=snr1.round(2).tolist(), snr_phase2=snr2.round(2).tolist(),
                              thal_phase1=th1.round(2).tolist(), thal_phase2=th2.round(2).tolist(),
                              d1_phase2=d1.round(2).tolist(), stn_phase2=stn.round(2).tolist(),
                              selected_phase1=sel1, selected_phase2=sel2))
            pools[0].rates = BACKGROUND_HZ * Hz; pools[1].rates = BACKGROUND_HZ * Hz
    return rest, pairs


def score(pairs):
    p1_single = [p for p in pairs if p["selected_phase1"] == [0]]
    p1_none = [p for p in pairs if p["selected_phase1"] == []]
    p1_multi = [p for p in pairs if len(p["selected_phase1"]) > 1]
    up = [p for p in pairs if p["r2"] > p["r1"]]
    down = [p for p in pairs if p["r2"] < p["r1"]]
    eq = [p for p in pairs if p["r2"] == p["r1"]]
    switch = [p for p in up if p["selected_phase2"] == [1]]
    keep = [p for p in down if p["selected_phase2"] == [0]]
    dual2 = [p for p in pairs if len(p["selected_phase2"]) > 1]
    none2 = [p for p in pairs if p["selected_phase2"] == []]
    eq_single = [p for p in eq if len(p["selected_phase2"]) == 1]
    thal_sel, thal_un = [], []
    for p in pairs:
        for k, th in enumerate(p["thal_phase2"]):
            (thal_sel if k in p["selected_phase2"] else thal_un).append(th)
    n = len(pairs)
    return dict(n_pairs=n,
                phase1_single_frac=len(p1_single) / n, phase1_none_frac=len(p1_none) / n,
                phase1_multi_frac=len(p1_multi) / n,
                phase1_min_salience_selected=min([p["r1"] for p in p1_single], default=None),
                switch_frac=len(switch) / max(1, len(up)), keep_frac=len(keep) / max(1, len(down)),
                equal_single_frac=len(eq_single) / max(1, len(eq)),
                phase2_dual_frac=len(dual2) / n, phase2_none_frac=len(none2) / n,
                thal_selected_hz=float(np.mean(thal_sel)) if thal_sel else None,
                thal_unselected_hz=float(np.mean(thal_un)) if thal_un else None)


def in_range(rest, targets):
    out = {}
    for pop, vals in rest.items():
        rng = targets.get(pop, {}).get("range")
        m = float(np.mean(vals))
        out[pop] = dict(mean_hz=round(m, 2), range=rng, ok=(None if rng is None else bool(rng[0] <= m <= rng[1])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-gain", type=float, default=1.0)
    ap.add_argument("--stn-gain", type=float, default=1.0)
    ap.add_argument("--thal-drive", type=float, default=24.0)
    ap.add_argument("--thal-n-in", type=int, default=16)
    ap.add_argument("--da", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--max", type=int, default=40, help="highest salience (spikes/s)")
    ap.add_argument("--stn-spon", type=float, default=None, help="tonic current nA (paper 1.1)")
    ap.add_argument("--gp-spon", type=float, default=None, help="tonic current nA (paper 0.38)")
    ap.add_argument("--snr-spon", type=float, default=None, help="tonic current nA (paper 0.39)")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--quick", action="store_true", help="every other salience")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kw = dict(ctx_gain=a.ctx_gain, ctx_gain_stn=a.stn_gain, thal_drive_mV=a.thal_drive, da=a.da, thal_n_in=a.thal_n_in)
    spon = {k: v for k, v in (("STN", a.stn_spon), ("GP", a.gp_spon), ("SNr", a.snr_spon)) if v is not None}
    if spon:
        kw["spon_nA"] = spon
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "bg_parameters.json")) as f:
        targets = json.load(f)["rate_targets_hz"]
    seeds = [int(s) for s in a.seeds.split(",")]
    per_seed, t_start = [], time.time()
    for s in seeds:
        rest, pairs = run_seed(s, a.k, a.grid, kw, quick=a.quick, smax=a.max)
        sc = score(pairs)
        per_seed.append(dict(seed=s, rest_hz=rest, rest_check=in_range(rest, targets), score=sc, pairs=pairs))
        print(f"seed {s}: rest", {k: v["mean_hz"] for k, v in per_seed[-1]["rest_check"].items()},
              "ok", [k for k, v in per_seed[-1]["rest_check"].items() if v["ok"]],
              "| single %.2f switch %.2f keep %.2f dual %.2f thal sel/unsel %s/%s" % (
                  sc["phase1_single_frac"], sc["switch_frac"], sc["keep_frac"], sc["phase2_dual_frac"],
                  None if sc["thal_selected_hz"] is None else round(sc["thal_selected_hz"], 1),
                  None if sc["thal_unselected_hz"] is None else round(sc["thal_unselected_hz"], 1)), flush=True)
    keys = [k for k in per_seed[0]["score"] if isinstance(per_seed[0]["score"][k], (int, float))]
    summary = {k: dict(mean=float(np.mean([p["score"][k] for p in per_seed if p["score"][k] is not None])),
                       sd=float(np.std([p["score"][k] for p in per_seed if p["score"][k] is not None])))
               for k in keys if any(p["score"][k] is not None for p in per_seed)}
    rest_ok = {pop: sum(p["rest_check"][pop]["ok"] is True for p in per_seed) for pop in per_seed[0]["rest_check"]}
    out = dict(params=kw, k=a.k, grid=a.grid, max=a.max, quick=a.quick, seeds=seeds, n_pool=N_POOL, background_hz=BACKGROUND_HZ,
               summary=summary, rest_in_range_seeds=rest_ok, per_seed=per_seed, wall_s=round(time.time() - t_start, 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("summary", {k: round(v["mean"], 3) for k, v in summary.items()}, "rest in range (seeds):", rest_ok)


if __name__ == "__main__":
    main()
