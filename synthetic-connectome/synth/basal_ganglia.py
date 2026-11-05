"""A spiking basal-ganglia–thalamus loop for action selection (Stage 26).

Circuit, constants and protocol from Humphries, Stewart & Gurney (2006) J Neurosci
26:12921–12942 (rat; ModelDB 83559), written out with labels in
results/bg_parameters.json.  K channels, one per cortical action pool:

    cortex pool k ──► D1_k ──┤ SNr_k ──┤ Thal_k ──► cortex pool k      (direct)
    cortex pool k ──► D2_k ──┤ GP_k ──┤ STN_k, GP, SNr_k                (indirect)
    cortex pool k ──► STN_k ──► GP, SNr (all channels, diffuse)         (hyperdirect)
    GP ──┤ GP, SNr ──┤ SNr (diffuse collaterals)

Neurons are leaky integrate-and-fire with the paper's membranes, tonic currents,
reset/refractory/noise and chloride floor; synapses are exponential currents scaled to
the paper's peak PSPs (AMPA 3 mV / 2 ms, NMDA 0.1 mV / 100 ms, GABA_A 3 mV / 3 ms),
delays per pathway.  A channel is *selected* when its SNr falls below 5 spikes/s,
which releases its thalamic relay and so feeds the pool that drove it.

Added here and labelled ASSUMED: the thalamic relay and its two pathways (SNr → Thal,
Thal → cortex), because the paper stops at SNr.  FITTED (scans in results/bg/):
ctx_gain (cortical PSP factor), thal_drive_mV, thal_gain.  Left out, stated in the
parameter file: dopamine modulation, striatal up/down states beyond the constant
down-state current, STN rebound bursts, shunting inhibition, FSIs, TRN.  Reward is
still applied by the learner — nothing in this circuit predicts it yet.

Usage::

    bg = BasalGanglia(sources=[(G, pool1_idx), (G, pool2_idx)], seed=1,
                      target=(G, "g_ed", w_thal_nS))        # target optional
    net.add(*bg.objects)
    ...
    bg.rates("SNr", t0_ms, t1_ms)     -> per-channel mean rate (Hz) in the window
    bg.selected(t0_ms, t1_ms)         -> channel index with SNr < 5 Hz, or None
    bg.clear()                        -> drop recorded spikes (long runs)
"""
from __future__ import annotations

import json
import os

import numpy as np
import brian2 as b2
from brian2 import ms, mV, nS

PARAM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "bg_parameters.json")

# Humphries 2006, Table 2 / Methods: R (MOhm), tau_m (ms), theta (mV above rest), I_spon (nA)
MEMBRANE = {
    "D1": dict(R=42.0, tau=25.0, theta=30.0, I_spon=-0.25),     # I_down: constant down-state current
    "D2": dict(R=42.0, tau=25.0, theta=30.0, I_spon=-0.25),
    "STN": dict(R=18.0, tau=6.0, theta=20.0, I_spon=1.1),
    "GP": dict(R=88.0, tau=14.0, theta=30.0, I_spon=0.38),
    "SNr": dict(R=112.0, tau=8.0, theta=30.0, I_spon=0.39),
}
THAL_MEMBRANE = dict(tau=10.0, theta=20.0)                      # ASSUMED generic relay cell
PSP = {"ampa": (3.0, 2.0), "nmda": (0.1, 100.0), "gaba": (3.0, 3.0)}   # peak mV, tau ms (Table 1)
RESET_MV, REFRACTORY_MS, NOISE_SD_MV, V_LIM_MV = 0.0, 2.0, 0.3, 20.0
C_WITHIN = 0.25                                                 # within-channel connection probability
STRIATAL_W = 4.0                                                # D1->SNr, D2->GP weight (Oorschot 1996)
SELECT_HZ = 5.0                                                 # selection: SNr channel mean rate below this
# (pre, post, receptor, scope, delay ms, weight factor); scope "within" = same channel, "diffuse" = all
PATHWAYS = [
    ("D1", "SNr", "gaba", "within", 4.0, STRIATAL_W),
    ("D2", "GP", "gaba", "within", 5.0, STRIATAL_W),
    ("STN", "SNr", "glut", "diffuse", 1.5, 1.0),
    ("STN", "GP", "glut", "diffuse", 2.0, 1.0),
    ("GP", "STN", "gaba", "within", 4.0, 1.0),
    ("GP", "GP", "gaba", "diffuse", 1.0, 1.0),
    ("GP", "SNr", "gaba", "within", 3.0, 1.0),
    ("SNr", "SNr", "gaba", "diffuse", 1.0, 1.0),
    ("SNr", "Thal", "gaba", "within", 2.0, 1.0),                # ASSUMED nigro-thalamic
]
CTX_DELAY_MS = {"D1": 10.0, "D2": 10.0, "STN": 2.5}
THAL_CTX_DELAY_MS = 2.0                                         # ASSUMED

DEFAULTS = dict(
    n_per=64,            # neurons per population per channel (paper: 64)
    ctx_n_in=16,         # cortical inputs per striatal/STN cell, as for a 64-cell channel at p 0.25
    ctx_gain=1.0,        # FITTED: factor on the cortical PSP onto D1/D2 (results/bg/ctx_scan.json)
    ctx_gain_stn=1.0,    # FITTED: factor on the cortical PSP onto STN (hyperdirect)
    da=0.3,              # tonic dopamine lambda (paper: 0.3 normal, 0 depleted, 0.8 high); cortical input
                         # onto D1 x (1 + lambda), onto D2 x (1 - lambda) (Gurney et al. 2001 convention)
    thal_drive_mV=24.0,  # FITTED: tonic depolarisation of the relay (results/bg/thal_scan.json)
    thal_gain=1.0,       # FITTED: factor on the thalamo-cortical conductance
    thal_n_in=16,        # thalamic inputs per cortical pool cell
    noise_sd_mV=NOISE_SD_MV,
    v_lim_mV=V_LIM_MV,
    # tonic currents (nA): FITTED on the stand-in so the resting rates sit inside the recorded
    # ranges with this cortical background (results/bg/spon_scan.json); the paper's own values
    # (STN 1.1, GP 0.38, SNr 0.39, tuned to its own cortical input) are in MEMBRANE
    spon_nA={"STN": 1.22, "GP": 0.38, "SNr": 0.42},
)


def psp_scale(tau_m, tau_s):
    """Jump (mV) in an exponential synaptic current (time constant tau_s) that gives a 1 mV
    peak PSP on a leaky membrane with time constant tau_m (both ms)."""
    tau_m, tau_s = float(tau_m), float(tau_s)
    if abs(tau_m - tau_s) < 1e-9:
        return float(np.e)
    a = tau_s / (tau_m - tau_s)
    tp = np.log(tau_m / tau_s) * tau_m * tau_s / (tau_m - tau_s)
    peak = a * (np.exp(-tp / tau_m) - np.exp(-tp / tau_s))
    return float(1.0 / peak)


def load_params():
    with open(PARAM_FILE) as f:
        return json.load(f)


class BasalGanglia:
    """K-channel basal-ganglia–thalamus loop attached to K cortical pools.

    sources: list of (NeuronGroup/PoissonGroup, index array) — the cortical cells of each channel
    target:  None, or (NeuronGroup, conductance variable name, weight nS) for Thal -> cortex;
             the thalamic relay of channel k then projects onto sources[k]'s cells
    monitor: population names to record ("all" or a list); SNr and Thal are always recorded
    """

    POPS = ("D1", "D2", "STN", "GP", "SNr", "Thal")

    def __init__(self, sources, seed=0, target=None, monitor=("SNr", "Thal"), name="bg", **kw):
        unknown = set(kw) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown BasalGanglia options: {sorted(unknown)}")
        self.p = dict(DEFAULTS, **kw)
        self.K = len(sources)
        if self.K < 1:
            raise ValueError("need at least one channel")
        self.sources = [(g, np.asarray(idx, int)) for g, idx in sources]
        self.name = name
        self.rng = np.random.default_rng(seed)
        n = int(self.p["n_per"])
        self.n_per = n
        self.ch = np.repeat(np.arange(self.K), n)           # channel of each neuron (same in every pop)
        self.groups, self.synapses, self.monitors, self._floors = {}, [], {}, []
        self._build_populations()
        self._wire_internal()
        self._wire_cortex()
        if target is not None:
            self._wire_thalamocortical(*target)
        mon = self.POPS if monitor == "all" else tuple(set(tuple(monitor)) | {"SNr", "Thal"})
        for pop in mon:
            self.monitors[pop] = b2.SpikeMonitor(self.groups[pop], name=f"{name}_mon_{pop}")

    # ------------------------------------------------------------ construction
    def _build_populations(self):
        n, K = self.n_per, self.K
        eqs = """
        dv/dt = (-v + ampa + nmda - gaba + v_spon) / tau_m + sigma_step * xi / sqrt(dt_noise) : volt (unless refractory)
        dampa/dt = -ampa / tau_ampa : volt
        dnmda/dt = -nmda / tau_nmda : volt
        dgaba/dt = -gaba / tau_gaba : volt
        ch : integer (constant)
        """
        dt = b2.defaultclock.dt
        for pop in self.POPS:
            m = THAL_MEMBRANE if pop == "Thal" else MEMBRANE[pop]
            spon = (self.p["spon_nA"] or {}).get(pop, m.get("I_spon"))
            v_spon = self.p["thal_drive_mV"] if pop == "Thal" else m["R"] * spon           # MOhm * nA = mV
            ns = dict(tau_m=m["tau"] * ms, v_spon=v_spon * mV, sigma_step=self.p["noise_sd_mV"] * mV,
                      dt_noise=dt, tau_ampa=PSP["ampa"][1] * ms, tau_nmda=PSP["nmda"][1] * ms,
                      tau_gaba=PSP["gaba"][1] * ms, theta=m["theta"] * mV, v_lim=self.p["v_lim_mV"] * mV)
            G = b2.NeuronGroup(n * K, eqs, threshold="v > theta", reset=f"v = {RESET_MV}*mV",
                               refractory=REFRACTORY_MS * ms, method="euler", namespace=ns,
                               name=f"{self.name}_{pop.lower()}")
            G.ch = self.ch
            G.v = np.clip(self.rng.normal(0.5 * m["theta"], 3.0, n * K), -self.p["v_lim_mV"], 0.95 * m["theta"]) * mV
            if pop != "D1" and pop != "D2":          # chloride floor (striatum gets no GABA here)
                self._floors.append(G.run_regularly("v = clip(v, -v_lim, 1*volt)", dt=dt, when="end",
                                                    name=f"{self.name}_{pop.lower()}_floor"))
            self.groups[pop] = G

    def _onpre(self, receptor, post, factor=1.0):
        """on_pre code giving the paper's peak PSP(s) on population `post`, times factor."""
        tau_m = (THAL_MEMBRANE if post == "Thal" else MEMBRANE[post])["tau"]
        parts = []
        kinds = ("ampa", "nmda") if receptor == "glut" else ("gaba",)
        for kind in kinds:
            peak, tau_s = PSP[kind]
            parts.append(f"{kind}_post += {peak * psp_scale(tau_m, tau_s) * factor:.6g}*mV")
        return "; ".join(parts)

    def _wire_internal(self):
        for pre, post, rec, scope, delay, factor in PATHWAYS:
            S = b2.Synapses(self.groups[pre], self.groups[post], on_pre=self._onpre(rec, post, factor),
                            delay=delay * ms, name=f"{self.name}_{pre.lower()}_{post.lower()}")
            if scope == "within":
                S.connect(condition="ch_pre == ch_post", p=C_WITHIN)
            else:                                 # diffuse: b = c / K (paper: c / 3 = 0.083)
                S.connect(p=C_WITHIN / self.K)
            self.synapses.append(S)

    def _ctx_pairs(self, k, pre_idx, n_in):
        """pre/post index arrays from the pool of channel k onto its 64 cells of one population,
        n_in inputs per cell on average (p = n_in / pool size, capped at 1); returns also the
        weight factor that keeps the expected total drive when the pool is smaller than n_in."""
        n = self.n_per
        post = np.arange(k * n, (k + 1) * n)
        p = min(1.0, n_in / max(1, pre_idx.size))
        pre_i = np.repeat(pre_idx, n)
        post_j = np.tile(post, pre_idx.size)
        keep = self.rng.random(pre_i.size) < p
        return pre_i[keep], post_j[keep], max(1.0, n_in / max(1, pre_idx.size))

    def _wire_cortex(self):
        n_in, lam = int(self.p["ctx_n_in"]), float(self.p["da"])
        gains = {"D1": self.p["ctx_gain"] * (1.0 + lam), "D2": self.p["ctx_gain"] * (1.0 - lam),
                 "STN": self.p["ctx_gain_stn"]}
        for pop in ("D1", "D2", "STN"):
            gain = float(gains[pop])
            by_src = {}
            for k, (g, idx) in enumerate(self.sources):
                i, j, wf = self._ctx_pairs(k, idx, n_in)
                by_src.setdefault(id(g), (g, [], [], []))
                by_src[id(g)][1].append(i); by_src[id(g)][2].append(j); by_src[id(g)][3].append(np.full(i.size, wf))
            for m_, (g, ii, jj, ww) in enumerate(by_src.values()):
                S = b2.Synapses(g, self.groups[pop], model="wf : 1",
                                on_pre=self._onpre("glut", pop, gain).replace("*mV", "*mV*wf"),
                                delay=CTX_DELAY_MS[pop] * ms, name=f"{self.name}_ctx{m_}_{pop.lower()}")
                S.connect(i=np.concatenate(ii), j=np.concatenate(jj))
                S.wf = np.concatenate(ww)
                self.synapses.append(S)

    def _wire_thalamocortical(self, G, var, w_nS):
        n_in, gain = int(self.p["thal_n_in"]), float(self.p["thal_gain"])
        ii, jj = [], []
        n = self.n_per
        for k, (g, idx) in enumerate(self.sources):
            if g is not G:
                continue
            pre = np.arange(k * n, (k + 1) * n)
            p = min(1.0, n_in / n)
            pre_i = np.repeat(pre, idx.size); post_j = np.tile(idx, n)
            keep = self.rng.random(pre_i.size) < p
            ii.append(pre_i[keep]); jj.append(post_j[keep])
        if not ii:
            return
        S = b2.Synapses(self.groups["Thal"], G, model="w : siemens", on_pre=f"{var}_post += w",
                        delay=THAL_CTX_DELAY_MS * ms, name=f"{self.name}_thal_ctx")
        S.connect(i=np.concatenate(ii), j=np.concatenate(jj))
        S.w = float(w_nS) * gain * nS
        self.synapses.append(S)
        self.thal_ctx = S

    # --------------------------------------------------------------- readout
    @property
    def objects(self):
        # (the floor operations are contained in their groups and travel with them)
        return list(self.groups.values()) + self.synapses + list(self.monitors.values())

    def counts(self, pop, t0_ms, t1_ms):
        """spikes per channel of population `pop` in [t0, t1) ms"""
        mon = self.monitors[pop]
        t = np.asarray(mon.t / ms); i = np.asarray(mon.i)
        win = (t >= t0_ms) & (t < t1_ms)
        return np.bincount(self.ch[i[win]], minlength=self.K)

    def rates(self, pop, t0_ms, t1_ms):
        """mean rate (Hz) per channel of population `pop` in [t0, t1) ms"""
        return self.counts(pop, t0_ms, t1_ms) / self.n_per / ((t1_ms - t0_ms) / 1000.0)

    def selected(self, t0_ms, t1_ms, hz=SELECT_HZ):
        """channel whose SNr is below `hz` (lowest if several), else None"""
        r = self.rates("SNr", t0_ms, t1_ms)
        k = int(np.argmin(r))
        return k if r[k] < hz else None

    def clear(self):
        for mon in self.monitors.values():
            mon.resize(0)
            mon.variables["N"].set_value(0)

    def summary(self, t0_ms, t1_ms):
        return {pop: [round(float(x), 2) for x in self.rates(pop, t0_ms, t1_ms)] for pop in self.monitors}
