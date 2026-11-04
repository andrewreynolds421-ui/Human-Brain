"""Where does the column's rhythm come from?  Measures burst timing (interval\nmean, CV, population-rate autocorrelation) under input and parameter changes.\n\n    python scripts/rhythm_diagnostics.py baseline constant rate4 gaba:3 taumE:0.5\n"""
import sys, json, numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from synth import PRESETS, synthesize
from synth.simulate import simulate
from synth.analyze import population_rates, synchrony

def bursts(res, t0=300.0, bin_ms=2.0):
    t = res["spike_t_ms"]; t = t[t >= t0]
    edges = np.arange(t0, res["t_sim_ms"] + bin_ms, bin_ms)
    h, _ = np.histogram(t, edges)
    k = np.ones(3) / 3; hs = np.convolve(h, k, "same")
    thr = hs.mean() + 2 * hs.std()
    above = hs > thr
    on = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    times = edges[on]
    ibi = np.diff(times)
    # autocorrelation of the population rate
    x = h - h.mean(); ac = np.correlate(x, x, "full")[x.size - 1:]; ac /= ac[0]
    lags = np.arange(ac.size) * bin_ms
    sel = (lags > 20) & (lags < 400)
    peak_lag = float(lags[sel][np.argmax(ac[sel])]); peak = float(ac[sel].max())
    return dict(n_bursts=int(times.size), ibi_mean=float(ibi.mean()) if ibi.size else None,
                ibi_cv=float(ibi.std() / ibi.mean()) if ibi.size > 1 else None,
                ibi=[round(float(v), 1) for v in ibi[:12]],
                ac_peak_lag_ms=peak_lag, ac_peak=round(peak, 3),
                fano=float(h.var() / h.mean()) if h.mean() > 0 else None)



PRESET_KW = dict(verbose=False, ext_gain=0.5)


def run(case="baseline", seed=1, t_ms=2500.0):
    """case: baseline | no_recurrent | constant | rate<f> (fibres f x faster, same mean) |
    taumE:<x> | gaba:<ms> | stp:<target>,<source>:<field>:<value> (steady input for the last three)"""
    p = PRESETS["human_h01_12pop"](**PRESET_KW)
    kw = {}
    if case == "no_recurrent": kw = dict(recurrent=False)
    elif case == "constant": kw = dict(ext_mode="constant")
    elif case.startswith("rate"): kw = dict(ext_rate_factor=float(case[4:]))
    elif case.startswith("taumE:"):
        x = float(case.split(":")[1]); t = p.neuron.tau_m_pop.copy(); C = p.neuron.C_m_pop.copy()
        t[p.is_exc] *= x; C[p.is_exc] *= x; p.neuron.tau_m_pop, p.neuron.C_m_pop = t, C
        kw = dict(ext_mode="constant")
    elif case.startswith("gaba:"):
        p.tau_inh_ms = float(case.split(":")[1]); kw = dict(ext_mode="constant")
    elif case.startswith("stp:"):
        _, pair, field, val = case.split(":"); k = tuple(pair.split(","))
        d = dict(p.stp[k]); d[field] = float(val); p.stp[k] = d; kw = dict(ext_mode="constant")
    col = synthesize(p, N_scaling=0.1, K_scaling=0.25, seed=seed)
    res = simulate(col, t_sim_ms=t_ms, seed=seed, verbose=False, **kw)
    return dict(case=case, seed=seed, **bursts(res))


if __name__ == "__main__":
    for c in sys.argv[1:] or ["baseline"]:
        print(json.dumps(run(c)), flush=True)
