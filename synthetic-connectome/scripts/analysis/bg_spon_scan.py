"""Scan the tonic currents of STN / GP / SNr for resting rates inside the recorded ranges
(results/bg_parameters.json), with the stand-in cortex at background.  Paper values
(Humphries 2006): STN 1.1, GP 0.38, SNr 0.39 nA.  Writes results/bg/spon_scan.json."""
import itertools, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import brian2 as b2
from brian2 import ms, Hz
from synth.basal_ganglia import BasalGanglia

ctx_gain, stn_gain = float(sys.argv[1]), float(sys.argv[2])
seeds = [1, 2]
targets = json.load(open("results/bg_parameters.json"))["rate_targets_hz"]
grid = dict(STN=[1.1, 1.14, 1.18], GP=[0.38, 0.42, 0.46], SNr=[0.39, 0.42, 0.45])
rows = []
t0 = time.time()
for stn, gp, snr in itertools.product(grid["STN"], grid["GP"], grid["SNr"]):
    rates = {}
    for seed in seeds:
        b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms; b2.seed(seed)
        pools = [b2.PoissonGroup(40, rates=3 * Hz) for _ in range(3)]
        bg = BasalGanglia([(p, np.arange(40)) for p in pools], seed=seed, monitor="all", ctx_gain=ctx_gain,
                          ctx_gain_stn=stn_gain, spon_nA={"STN": stn, "GP": gp, "SNr": snr})
        net = b2.Network(*pools, *bg.objects); net.run(1500 * ms)
        for pop in bg.monitors:
            rates.setdefault(pop, []).append(float(np.mean(bg.rates(pop, 500, 1500))))
    mean = {k: round(float(np.mean(v)), 2) for k, v in rates.items()}
    ok = {k: (targets[k]["range"] is not None and targets[k]["range"][0] <= mean[k] <= targets[k]["range"][1]) for k in mean}
    # distance to the middle of each range, in units of half the range width (0 = centre, 1 = edge)
    dist = {k: abs(mean[k] - np.mean(targets[k]["range"])) / (np.ptp(targets[k]["range"]) / 2) for k in ("STN", "GP", "SNr")}
    rows.append(dict(spon_nA=dict(STN=stn, GP=gp, SNr=snr), rates_hz=mean, ok=ok, n_ok=sum(ok.values()),
                     score=round(float(np.mean(list(dist.values()))), 3)))
    print(rows[-1]["spon_nA"], mean, "ok", [k for k, v in ok.items() if v], "score %.2f" % rows[-1]["score"], flush=True)
rows.sort(key=lambda r: (-r["n_ok"], r["score"]))
json.dump(dict(ctx_gain=ctx_gain, ctx_gain_stn=stn_gain, seeds=seeds, grid=grid, best=rows[0], rows=rows,
               wall_s=round(time.time() - t0)), open("results/bg/spon_scan.json", "w"), indent=1)
print("best", rows[0])
