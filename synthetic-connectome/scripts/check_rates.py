#!/usr/bin/env python3
"""Compare the column's spontaneous firing rates with the reference ranges in
results/rate_targets.json (human where human data exist, rodent otherwise).

    python scripts/check_rates.py                     # dendritic preset, 2 s, seed 3
    python scripts/check_rates.py --istdp             # with inhibitory plasticity on
    python scripts/check_rates.py --n-scale 0.03 --t 3000 --seeds 1 2 3

Prints one line per population with IN / LOW / HIGH and writes
results/rate_check.json.  Exit code 1 if any population is out of range,
so it can serve as a check.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import brian2 as b2                                   # noqa: E402
from brian2 import ms                                 # noqa: E402
from synth import synthesize, PRESETS                 # noqa: E402
from synth.network import build_network               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = json.load(open(os.path.join(HERE, "..", "results", "rate_targets.json")))["populations"]
ISTDP_TARGET = {"L23E": 1.6, "L4E": 5.8, "L5E": 4.8, "L6E": 3.2}


def run(n_scale, t_ms, seed, istdp, preset_kw, preset="human_h01_dendritic"):
    p = PRESETS[preset](verbose=False, **preset_kw)
    col = synthesize(p, N_scaling=n_scale, K_scaling=1.0, seed=seed)
    b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms
    nw = build_network(col, seed=seed, bg_ratio=preset_kw.get("bg_ratio", 2.0), merged_inputs=True, shared_stp=True,
                       istdp=dict(eta=0.0163, tau_ms=20.0, target_hz=ISTDP_TARGET) if istdp else None)
    nw["net"].run(300 * ms)
    n0 = nw["spikes"].num_spikes
    nw["net"].run(t_ms * ms)
    i = np.asarray(nw["spikes"].i)[n0:]
    counts = np.bincount(col.pop_of[i], minlength=len(col.pops))
    n_per = np.bincount(col.pop_of, minlength=len(col.pops))
    return col.pops, counts / np.maximum(n_per, 1) / (t_ms / 1000.0), n_per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="human_h01_dendritic")
    ap.add_argument("--n-scale", type=float, default=0.03)
    ap.add_argument("--t", type=float, default=2000.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[3])
    ap.add_argument("--istdp", action="store_true")
    ap.add_argument("--ext-gain", type=float, default=None, help="override the preset's external gain")
    ap.add_argument("--bg-ratio", type=float, default=2.0)
    ap.add_argument("--nfs-ext", type=float, default=None, help="external-fibre factor for non-FS interneurons (assumption test)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "rate_check.json"))
    a = ap.parse_args()
    pk = dict(bg_ratio=a.bg_ratio)
    if a.ext_gain is not None: pk["ext_gain"] = a.ext_gain
    elif a.preset == "human_h01_dendritic": pk["ext_gain"] = 2.0
    if a.nfs_ext is not None: pk["nfs_ext_factor"] = a.nfs_ext
    runs = [run(a.n_scale, a.t, s, a.istdp, pk, a.preset) for s in a.seeds]
    pops = runs[0][0]
    rates = np.mean([r[1] for r in runs], 0)
    spread = np.std([r[1] for r in runs], 0) if len(runs) > 1 else np.zeros(len(pops))
    rows, bad = [], 0
    print(f"{'pop':5s} {'cells':>5s} {'rate':>7s}   {'range':>12s}  species   verdict")
    for k, p in enumerate(pops):
        t = TARGETS[p]; r = float(rates[k])
        verdict = "IN" if t["min"] <= r <= t["max"] else ("LOW" if r < t["min"] else "HIGH")
        bad += verdict != "IN"
        print(f"{p:5s} {int(runs[0][2][k]):5d} {r:7.2f}   {t['min']:5.1f}-{t['max']:<5.1f}  {t['species']:7s}  {verdict}"
              + (f"  (±{spread[k]:.2f} over seeds)" if len(runs) > 1 else ""))
        rows.append(dict(pop=p, rate=round(r, 3), spread=round(float(spread[k]), 3), min=t["min"], max=t["max"],
                         species=t["species"], verdict=verdict))
    json.dump(dict(preset=a.preset, n_scale=a.n_scale, t_ms=a.t, seeds=a.seeds, istdp=a.istdp, ext_gain=pk.get("ext_gain"),
                   bg_ratio=a.bg_ratio, nfs_ext=a.nfs_ext, rows=rows, out_of_range=bad), open(a.out, "w"), indent=1)
    print(f"{bad} of {len(pops)} populations out of range")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
