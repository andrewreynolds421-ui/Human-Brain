"""Stage 33 (roadmap Stage 7, first step): the resting column's field-potential spectrum against human
intracranial EEG.

Model signal: LFP proxy of Mazzoni et al. 2015 (PLoS Comput Biol 11:e1004584), the reference weighted sum
    LFP(t) = sum |I_exc(t - 6 ms)| - 1.65 * sum |I_inh(t)|
over pyramidal cells (L2/3 and L5 of both areas), with each cell's synaptic currents taken on both
compartments: I_exc = g_e (E_e - v) + g_ed (E_e - v_d), I_inh = g_i (E_i - v) + g_id (E_i - v_d).
Spectrum: Welch, 2 s windows.  Measures:
  * slope of log10 power vs log10 frequency over 30-45 Hz (Lendner et al. 2020 eLife 9:e55092: awake
    human iEEG -2.75 +/- 0.15 and -2.99 +/- 0.32 (mean +/- SEM, n = 12 and 10); N3 sleep -3.69, REM -4.15,
    propofol -4.34);
  * aperiodic exponent and periodic peaks, 2-80 Hz, by spectral parameterisation (FOOOF; Donoghue et al.
    2020 Nat Neurosci 23:1655), to compare with the atlas of normal human iEEG (Frauscher et al. 2018 Brain
    141:1130: frontal lobe beta 20-24 Hz; alpha mainly occipital).
Conditions: "awake" (the v2 column at rest) and "propofol" (GABA_A decay x2 in every cell — propofol's main
synaptic action is to prolong GABA_A currents; the factor is ASSUMED) — a check of direction: Gao et al. 2017
(NeuroImage 158:70) showed more inhibition steepens the 30-50 Hz slope.

    python scripts/spectrum_validation.py --seeds 31,32,33,34 --t 10 --out results/spectra/stage33.json
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


V_REST_MV = -65.0     # ASSUMED fixed driving-force reference for the "fixed" proxy


def record(seed, cond, t_s, n_rec, preset, ext_mode="poisson", ext_mod=None, proxy="fixed"):
    import brian2 as b2
    from brian2 import ms, mV, nS
    from synth.learner import build_learner_network
    from synth.network import build_network
    b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms; b2.seed(seed)
    col = build_learner_network(n_scale=0.015, seed=seed, preset=preset)
    nw = build_network(col, seed=seed, merged_inputs=(ext_mode == "poisson"), shared_stp=True, ext_mode=ext_mode,
                       **({"ext_mod": ext_mod} if ext_mod else {}))
    G, net = nw["G"], nw["net"]
    if cond == "propofol":
        G.tau_i = np.asarray(G.tau_i) * 2.0 * b2.second
    pyr = np.flatnonzero(np.isin(col.pop_of, [col.pops.index("L23E"), col.pops.index("L5E")]))
    rng = np.random.default_rng(seed)
    cells = np.sort(rng.choice(pyr, min(n_rec, pyr.size), replace=False))
    # the currents are summed inside the simulator at every 0.1 ms step (sampling the conductances
    # sparsely aliases their fast input fluctuations into the band), then low-passed and decimated to 1 kHz
    nrn = col.neuron
    probe = b2.NeuronGroup(1, "Ie : amp\nIi : amp", name="lfp_probe")
    if proxy == "fixed":
        # driving force fixed at the resting potential (V_REST_MV): the proxy follows the synaptic conductances
        # only; with the actual membrane potential every spike and reset injects broadband power
        model = """Ie_post = (g_e_pre + g_ed_pre) * (E_e - V_rest) : amp (summed)
    Ii_post = (g_i_pre + g_id_pre) * (V_rest - E_i) : amp (summed)"""
    else:
        model = """Ie_post = abs(g_e_pre * (E_e - v_pre) + g_ed_pre * (E_e - v_d_pre)) : amp (summed)
    Ii_post = abs(g_i_pre * (E_i - v_pre) + g_id_pre * (E_i - v_d_pre)) : amp (summed)"""
    Sx = b2.Synapses(G, probe, model=model, name="lfp_sum",
                     namespace=dict(E_e=nrn["E_exc_mV"] * mV, E_i=nrn["E_inh_mV"] * mV, V_rest=V_REST_MV * mV))
    Sx.connect(i=cells, j=0)
    mon = b2.StateMonitor(probe, ["Ie", "Ii"], record=[0], dt=0.1 * ms)
    net.add(probe, Sx)
    net.run(1000 * ms)                                         # settle
    net.add(mon)
    t0 = time.time(); net.run(t_s * 1000 * ms); wall = time.time() - t0
    from scipy.signal import decimate
    pA = b2.pA
    I_exc = decimate(decimate(np.asarray(mon.Ie[0] / pA), 5, ftype="fir"), 2, ftype="fir")   # 10 kHz -> 1 kHz
    I_inh = decimate(decimate(np.asarray(mon.Ii[0] / pA), 5, ftype="fir"), 2, ftype="fir")
    lfp = I_exc[:-6] - 1.65 * I_inh[6:]                       # excitatory term delayed 6 ms (1 ms samples)
    # firing rates per population over the recorded period (the fix must keep Stage 25's rates)
    sp = nw["spikes"]; ti = np.asarray(sp.t / ms); ii = np.asarray(sp.i)
    keep = ti >= 1000.0
    cnt = np.bincount(ii[keep], minlength=col.N)
    rates = {p: round(float(cnt[col.pop_of == k].mean() / t_s), 2) for k, p in enumerate(col.pops)}
    return lfp, dict(n_cells=int(cells.size), wall_s=round(wall, 1), rates=rates)


def analyse(lfp, fs=1000.0):
    from scipy.signal import welch
    f, P = welch(lfp - lfp.mean(), fs=fs, nperseg=int(2 * fs))
    sel = (f >= 30) & (f <= 45)
    slope = float(np.polyfit(np.log10(f[sel]), np.log10(P[sel]), 1)[0])
    low = (f >= 1) & (f <= 4)
    out = dict(slope_30_45=slope, low_vs_30_45_db=float(10 * np.log10(P[low].mean() / P[sel].mean())))
    try:
        from fooof import FOOOF
        fm = FOOOF(peak_width_limits=(2, 12), max_n_peaks=4, min_peak_height=0.15, aperiodic_mode="fixed", verbose=False)
        fm.fit(f, P, [2, 80])
        out.update(exponent_2_80=float(fm.aperiodic_params_[1]), r2=float(fm.r_squared_),
                   peaks=[dict(freq=round(float(c), 2), power=round(float(p), 3), width=round(float(w), 2))
                          for c, p, w in fm.peak_params_])
    except Exception as e:
        out["fooof_error"] = str(e)
    keep = f <= 100
    return out, f[keep].tolist(), P[keep].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="31,32,33,34")
    ap.add_argument("--t", type=float, default=10.0, help="seconds recorded per condition")
    ap.add_argument("--n-rec", type=int, default=300)
    ap.add_argument("--preset", default="human_h01_dendritic_v2")
    ap.add_argument("--conds", default="awake,propofol")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext-mode", default="poisson", help="poisson (default) or constant (external drive as its mean conductance)")
    ap.add_argument("--proxy", default="fixed", help="fixed (driving force at rest) or actual (uses the membrane potential, Stage 33)")
    ap.add_argument("--ext-mod", default=None, help="JSON: shared slow modulation of the external rate (Stage 34)")
    a = ap.parse_args()
    res = dict(about=__doc__, preset=a.preset, t_s=a.t, ext_mode=a.ext_mode, ext_mod=a.ext_mod, proxy=a.proxy, runs=[])
    if os.path.exists(a.out):                                  # resume: keep finished runs
        res = json.load(open(a.out))
    done = {(r["seed"], r["cond"]) for r in res["runs"]}
    for s in [int(x) for x in a.seeds.split(",")]:
        for cond in a.conds.split(","):
            if (s, cond) in done:
                continue
            lfp, info = record(s, cond, a.t, a.n_rec, a.preset, a.ext_mode, json.loads(a.ext_mod) if a.ext_mod else None, a.proxy)
            m, f, P = analyse(lfp)
            res["runs"].append(dict(seed=s, cond=cond, **info, **m, f=f, P=P))
            print(s, cond, {k: v for k, v in m.items()}, info, flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump(res, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
    for cond in a.conds.split(","):
        rs = [r for r in res["runs"] if r["cond"] == cond]
        sl = [r["slope_30_45"] for r in rs]; ex = [r.get("exponent_2_80") for r in rs if r.get("exponent_2_80") is not None]
        print(f"{cond}: slope 30-45 Hz {np.mean(sl):.2f} ± {np.std(sl):.2f} (n={len(sl)}); exponent 2-80 {np.mean(ex):.2f}; "
              f"peaks {[[p['freq'] for p in r.get('peaks', [])] for r in rs]}")


if __name__ == "__main__":
    main()
