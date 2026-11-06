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
from brian2 import ms, mV, nS, Hz

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
THAL_STR_DELAY_MS = 5.0                                         # ASSUMED (thalamostriatal)
CTX_DELAY_MS = {"D1": 10.0, "D2": 10.0, "STN": 2.5}
THAL_CTX_DELAY_MS = 2.0                                         # ASSUMED

DEFAULTS = dict(
    n_per=64,            # neurons per population per channel (paper: 64)
    ctx_n_in=16,         # cortical inputs per striatal/STN cell, as for a 64-cell channel at p 0.25
    ctx_gain=1.0,        # FITTED: factor on the cortical PSP onto D1/D2 (results/bg/ctx_scan.json)
    ctx_gain_stn=1.0,    # FITTED: factor on the cortical PSP onto STN (hyperdirect)
    da=0.3,              # tonic dopamine lambda (paper: 0.3 normal, 0 depleted, 0.8 high); cortical input
                         # onto D1 x (1 + lambda), onto D2 x (1 - lambda) (Gurney et al. 2001 convention)
    thal_drive_mV=24.0,  # tonic depolarisation of the relay; 31 mV gives the measured resting rate of motor
                         # thalamus (~13 Hz; Vitek et al. 1994) under tonic SNr (results/bg/thal_drive_scan.json)
    thal_gain=1.0,       # FITTED: factor on the thalamo-cortical conductance
    thal_n_in=16,        # thalamic inputs per cortical pool cell
    thal_str_gain=0.0,   # thalamostriatal input: Thal_k -> D1_k, D2_k (within channel, p 0.25, glutamate, 5 ms;
                         # Smith et al. 2004 "The thalamostriatal system"); the gain is FITTED, 0 = pathway absent
    noise_sd_mV=NOISE_SD_MV,
    v_lim_mV=V_LIM_MV,
    # tonic currents (nA): FITTED on the stand-in so the resting rates sit inside the recorded
    # ranges with this cortical background (results/bg/spon_scan.json); the paper's own values
    # (STN 1.1, GP 0.38, SNr 0.39, tuned to its own cortical input) are in MEMBRANE
    spon_nA={"STN": 1.22, "GP": 0.38, "SNr": 0.42},
    # ---- context (cue-area) input to the striatum and its plasticity (Stage 27) ----
    cue_n_in=16,         # context inputs per striatal cell (same convergence as the pool input)
    ctx_gain_cue=None,   # FITTED: PSP factor for context inputs (None = ctx_gain)
    plastic=False,       # context -> D1/D2 synapses learn from dopamine (three-factor rule)
    tau_x_ms=20.0,       # pre/post coincidence window (Pawlak & Kerr 2008: ~20-30 ms STDP windows in MSNs)
    tau_e_ms=1000.0,     # eligibility decay: dopamine must arrive 0.3-2 s after the input (Yagishita et al. 2014)
    a_minus=0.0,         # post-before-pre term (0 = pure pre->post coincidence; ASSUMED)
    post_lin=1.0,        # eligibility added per post spike: xpre * (post_lin + post_gain * xpost); post_gain > 0
    post_gain=0.0,       # makes it supralinear in the post cell's recent firing (ASSUMED stand-in for the
                         # Ca/NMDA dependence of striatal LTP: Calabresi et al. 1992; Shen et al. 2008)
    eta_bg=0.0,          # FITTED: cortico-striatal learning rate (results/bg/eta_scan.json)
    w_max_factor=3.0,    # ASSUMED: bound on the plastic weight (x initial)
    w_init_spread=0.25,  # ASSUMED: initial plastic weights uniform in 1 +/- spread
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


THAL_REST_HZ = 13.0      # MEASURED: awake monkey VLo 13 +/- 8, VA 15 +/- 8 spikes/s (Vitek et al. 1994 J Neurophysiol 71:1498)


def route_external_through_thalamus(G, pools, frac, bg, w_ext_nS, rest_hz=THAL_REST_HZ):
    """Stage 28: make the relay a *share of the pools' drive* instead of an addition to it.
    For each pool cell, a fraction `frac` (ASSUMED) of its external Poisson input is removed (merged-input
    mode: lam_ext scaled), and the thalamo-cortical weight is set so that the relay, firing at its resting
    rate, delivers the same mean input as was removed.  At rest the pools get what they got before; a
    selected channel (SNr down, relay up) gets more, an unselected one (relay silenced) gets less.
    Returns a dict with what was done (for the run record)."""
    if not hasattr(G, "lam_ext"):
        raise RuntimeError("routing through the thalamus needs merged inputs (fast mode)")
    S = getattr(bg, "thal_ctx", None)
    if S is None:
        raise RuntimeError("the loop has no thalamo-cortical pathway (give BasalGanglia a target)")
    dt_s = float(b2.defaultclock.dt / b2.second)
    cells = np.unique(np.concatenate([np.asarray(p, int) for p in pools]))
    lam = np.asarray(G.lam_ext)[cells].copy()
    removed_hz = frac * lam / dt_s                                   # input spikes/s removed per cell
    lam_new = (1.0 - frac) * lam
    lam_all = np.asarray(G.lam_ext).copy(); lam_all[cells] = lam_new; G.lam_ext = lam_all
    e_all = np.asarray(G.e_ext).copy(); e_all[cells] = np.exp(-lam_new); G.e_ext = e_all
    j = np.asarray(S.j)
    n_in = np.bincount(j, minlength=len(G))[cells].astype(float)       # relay inputs per pool cell
    # per-synapse weight: removed rate of its target / (its inputs x relay resting rate), in external units
    per_cell = np.zeros(len(G)); per_cell[cells] = removed_hz / np.maximum(n_in, 1.0) / rest_hz
    S.w = w_ext_nS * per_cell[j] * nS
    bg.routed = dict(frac=float(frac), rest_hz=rest_hz, removed_input_hz_per_cell=float(removed_hz.mean()),
                     relay_inputs_per_cell=float(n_in.mean()), weight_x_external=float(per_cell[cells].mean()))
    return bg.routed


class BasalGanglia:
    """K-channel basal-ganglia–thalamus loop attached to K cortical pools.

    sources: list of (NeuronGroup/PoissonGroup, index array) — the cortical cells of each channel
    target:  None, or (NeuronGroup, conductance variable name, weight nS) for Thal -> cortex;
             the thalamic relay of channel k then projects onto sources[k]'s cells
    monitor: population names to record ("all" or a list); SNr and Thal are always recorded
    """

    POPS = ("D1", "D2", "STN", "GP", "SNr", "Thal")

    def __init__(self, sources, seed=0, target=None, monitor=("SNr", "Thal"), name="bg", context=None, **kw):
        """context: None, or (group, index array) of cue/context cells that project to the striatum of
        *every* channel (the cortico-striatal convergence the task needs); with plastic=True these
        synapses learn by the dopamine-gated rule (reward(delta))."""
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
        self.plastic = []                                     # (Synapses, sign, population)
        self.context = None if context is None else (context[0], np.asarray(context[1], int))
        self._build_populations()
        self._wire_internal()
        self._wire_cortex()
        if self.context is not None:
            self._wire_context()
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
        if self.p["thal_str_gain"] > 0:
            for pop in ("D1", "D2"):
                S = b2.Synapses(self.groups["Thal"], self.groups[pop], on_pre=self._onpre("glut", pop, float(self.p["thal_str_gain"])),
                                delay=THAL_STR_DELAY_MS * ms, name=f"{self.name}_thal_{pop.lower()}")
                S.connect(condition="ch_pre == ch_post", p=C_WITHIN)
                self.synapses.append(S)
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

    def _wire_context(self):
        """Cue/context cells -> D1 and D2 of every channel, cue_n_in inputs per striatal cell,
        D1 x (1 + lambda) / D2 x (1 - lambda) as for the pool input.  With plastic=True each synapse
        carries a weight wf, an eligibility e (pre->post coincidences within tau_x, decaying with
        tau_e) and is changed by reward(delta): D1 += eta delta e, D2 -= eta delta e (Shen et al. 2008:
        dopamine potentiates D1-MSN inputs and depresses D2-MSN inputs; dips do the reverse)."""
        g, idx = self.context
        lam = float(self.p["da"])
        gain = self.p["ctx_gain"] if self.p["ctx_gain_cue"] is None else self.p["ctx_gain_cue"]
        gains = {"D1": gain * (1.0 + lam), "D2": gain * (1.0 - lam)}
        n_in = int(self.p["cue_n_in"])
        n = self.n_per
        p_conn = min(1.0, n_in / max(1, idx.size))
        wf_scale = max(1.0, n_in / max(1, idx.size))
        for pop in ("D1", "D2"):
            post = np.arange(n * self.K)
            pre_i = np.repeat(idx, post.size); post_j = np.tile(post, idx.size)
            keep = self.rng.random(pre_i.size) < p_conn
            pre_i, post_j = pre_i[keep], post_j[keep]
            psp = self._onpre("glut", pop, float(gains[pop])).replace("*mV", "*mV*wf")
            if self.p["plastic"]:
                model = """wf : 1
                elig : 1
                t_e : second
                dxpre/dt = -xpre / tau_x : 1 (event-driven)
                dxpost/dt = -xpost / tau_x : 1 (event-driven)"""
                on_pre = psp + "; elig = elig * exp(-(t - t_e) / tau_e) - a_minus * xpost; t_e = t; xpre += 1"
                on_post = "elig = elig * exp(-(t - t_e) / tau_e) + xpre * (post_lin + post_gain * xpost); t_e = t; xpost += 1"
                S = b2.Synapses(g, self.groups[pop], model=model, on_pre=on_pre, on_post=on_post,
                                delay=CTX_DELAY_MS[pop] * ms, name=f"{self.name}_cue_{pop.lower()}",
                                namespace=dict(tau_x=self.p["tau_x_ms"] * ms, tau_e=self.p["tau_e_ms"] * ms,
                                               a_minus=float(self.p["a_minus"]), post_lin=float(self.p["post_lin"]),
                                               post_gain=float(self.p["post_gain"])))
            else:
                S = b2.Synapses(g, self.groups[pop], model="wf : 1", on_pre=psp, delay=CTX_DELAY_MS[pop] * ms,
                                name=f"{self.name}_cue_{pop.lower()}")
            S.connect(i=pre_i, j=post_j)
            spread = float(self.p["w_init_spread"])
            S.wf = wf_scale * self.rng.uniform(1.0 - spread, 1.0 + spread, pre_i.size)
            if self.p["plastic"]:
                S.elig = 0.0; S.t_e = 0 * ms
                self.plastic.append((S, +1.0 if pop == "D1" else -1.0, pop))
            self.synapses.append(S)
        self.w_init = {pop: np.asarray(S.wf).copy() for S, _, pop in self.plastic} if self.p["plastic"] else {}
        self.w_max = {pop: self.p["w_max_factor"] * w for pop, w in self.w_init.items()}

    def reward(self, delta, t_ms=None):
        """Dopamine step: delta = reward minus its prediction (positive = burst, negative = dip).
        Applies the three-factor rule to the plastic context -> striatum synapses using each
        synapse's eligibility decayed to now.  Returns the mean absolute weight change (x initial)
        per population; nothing happens when plastic=False or eta_bg = 0."""
        if not self.plastic or self.p["eta_bg"] == 0.0:
            return {}
        now = float(b2.defaultclock.t / ms) if t_ms is None else float(t_ms)
        eta, tau_e, out = float(self.p["eta_bg"]), float(self.p["tau_e_ms"]), {}
        for S, sign, pop in self.plastic:
            e = np.asarray(S.elig) * np.exp(-np.clip(now - np.asarray(S.t_e / ms), 0, None) / tau_e)
            w = np.asarray(S.wf)
            w_new = np.clip(w + sign * eta * float(delta) * e, 0.0, self.w_max[pop])
            S.wf = w_new
            out[pop] = float(np.mean(np.abs(w_new - w) / np.maximum(self.w_init[pop], 1e-9)))
        return out

    def weight_summary(self):
        """mean plastic weight per population and channel, relative to the initial mean"""
        out = {}
        for S, _, pop in self.plastic:
            w = np.asarray(S.wf); j = np.asarray(S.j)
            out[pop] = [round(float(w[self.ch[j] == k].mean() / self.w_init[pop].mean()), 3) for k in range(self.K)]
        return out

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


# ----------------------------------------------------------------------------------------------
# Stage 27b: reward prediction inside the circuit — a dopamine population and a striosomal critic
# ----------------------------------------------------------------------------------------------
DA_DEFAULTS = dict(
    n=64,                # ASSUMED population size (as the loop's channels)
    tau_ms=20.0,         # ASSUMED LIF membrane for SNc/VTA cells
    theta_mV=20.0,
    refractory_ms=10.0,  # caps the rate near 100 Hz; bursts in vivo reach 15-30 Hz (Schultz 1998)
    tonic_hz=4.5,        # MEASURED target: 4-5 spikes/s in vivo (Grace & Bunney 1984; Schultz 1998)
    drive_mV=14.5,       # FITTED: tonic depolarisation giving tonic_hz with the model noise
    rew_fibres=100,      # primary-reward input (PPTg / lateral hypothalamus glutamate), Poisson fibres
    rew_hz=10.0,         # FITTED: fibre rate per unit reward during the reward event (burst ~25 spikes/s)
    rew_ms=150.0,        # ASSUMED event length (phasic bursts last 100-200 ms; Schultz 1998)
    inh_fibres=100,      # negative-outcome input (lateral habenula -> RMTg GABA; Matsumoto & Hikosaka 2007; Jhou 2009)
    inh_hz=10.0,         # FITTED: fibre rate per unit negative reward (pause to ~0)
    p_in=0.25,           # input connection probability (as the loop)
    crit_n=64,           # striosomal D1 critic cells (membrane as D1)
    crit_n_in=16,        # context inputs per critic cell
    crit_gain=2.5,       # FITTED in the learner: striosomes fire 3-8 spikes/s to a cue at the start (a floor of
                         # activity, so the prediction can grow or shrink); 1.5 left them silent and unable to learn
    crit_to_da_mV=0.5,   # ASSUMED: slow inhibition from one critic spike onto a DA cell (peak, mV)
    crit_tau_ms=500.0,   # ASSUMED stand-in for sustained striosomal inhibition reaching the reward time
    crit_eta=0.05,       # FITTED: critic learning rate (0.02 too slow to follow the learner's accuracy)
    crit_w_max_factor=10.0,
    crit_spread_mV=15.0, # ASSUMED: per-cell offset of the critic's resting level, uniform in [-spread, 0], so the
                         # population's rate rises gradually with its input instead of all cells crossing threshold at once
    burst_scale_hz=20.0, # delta = (rate - tonic) / burst_scale: about +1 for a full burst, -0.2 for a pause
    dip_gain=4.0,        # ASSUMED: factor on negative delta, so a full pause weighs about as much as a full burst
                         # (pauses are rate-floored and code negative errors by duration; Bayer & Glimcher 2005, 2007)
    window_ms=200.0,     # DA rate read over this window from the reward event
    # ---- Stage 29c: fast excitatory value pathway, so the burst moves to the cue (temporal difference) ----
    value_n=0,           # PPTg-type excitatory cells driven by the cue cells through learned synapses (0 = off);
                         # PPTg neurons carry the reward predicted by a cue at cue time (Okada et al. 2009;
                         # Kobayashi & Okada 2007) and excite dopamine cells (Lokwan et al. 1999).  With the slow
                         # striosomal inhibition this is the two-pathway account of the TD error (Houk et al. 1995;
                         # Vitay & Hamker 2014): fast excitation at the cue, slow cancellation at the reward.
    value_tau_ms=10.0, value_theta_mV=20.0,  # ASSUMED generic LIF
    value_offset_mV=(-24.0, -12.0),          # ASSUMED: resting offsets, so the cells are near-silent until learned
    value_n_in=16,
    value_gain=1.0,      # FITTED: PSP factor of cue -> value cells
    value_to_da_mV=1.0,  # FITTED: AMPA peak of one value-cell spike on a DA cell
    value_eta=0.05,      # learning rate (as the critic)
    value_w_max_factor=10.0,
)


class DopamineSystem:
    """SNc/VTA dopamine population with a primary-reward input, a negative-outcome input and a
    striosomal critic that learns to predict the reward from the cue cells.  The dopamine rate
    deviation after an outcome is the reward-prediction error that gates the loop's plasticity.

    What is measured: tonic 4-5 spikes/s, phasic bursts of 100-200 ms, pauses on omission, the
    three input pathways' existence and signs.  What is a stand-in, stated: LIF constants, the
    sustained (500 ms) critic inhibition that carries the prediction to the reward time (the real
    mechanism is a temporal-difference circuit; the burst does not transfer to the cue here),
    and the fitted input rates."""

    def __init__(self, bg, context, seed=0, name="da", **kw):
        unknown = set(kw) - set(DA_DEFAULTS)
        if unknown:
            raise TypeError(f"unknown DopamineSystem options: {sorted(unknown)}")
        self.p = dict(DA_DEFAULTS, **kw)
        self.bg, self.name = bg, name
        self.rng = np.random.default_rng(seed + 101)
        g, idx = context[0], np.asarray(context[1], int)
        p = self.p
        dt = b2.defaultclock.dt
        eqs = """
        dv/dt = (-v + ampa - gaba - gslow + v_spon) / tau_m + sigma_step * xi / sqrt(dt_noise) : volt (unless refractory)
        dampa/dt = -ampa / tau_ampa : volt
        dgaba/dt = -gaba / tau_gaba : volt
        dgslow/dt = -gslow / tau_slow : volt
        """
        ns = dict(tau_m=p["tau_ms"] * ms, v_spon=p["drive_mV"] * mV, sigma_step=NOISE_SD_MV * mV, dt_noise=dt,
                  tau_ampa=PSP["ampa"][1] * ms, tau_gaba=PSP["gaba"][1] * ms, tau_slow=p["crit_tau_ms"] * ms,
                  theta=p["theta_mV"] * mV)
        self.DA = b2.NeuronGroup(int(p["n"]), eqs, threshold="v > theta", reset="v = 0*mV",
                                 refractory=p["refractory_ms"] * ms, method="euler", namespace=ns, name=f"{name}_snc")
        self.DA.v = self.rng.uniform(0, 0.9 * p["theta_mV"], int(p["n"])) * mV
        self.rew = b2.PoissonGroup(int(p["rew_fibres"]), rates=0 * Hz, name=f"{name}_reward_in")
        self.inh = b2.PoissonGroup(int(p["inh_fibres"]), rates=0 * Hz, name=f"{name}_habenula_in")
        a_e = PSP["ampa"][0] * psp_scale(p["tau_ms"], PSP["ampa"][1])
        a_i = PSP["gaba"][0] * psp_scale(p["tau_ms"], PSP["gaba"][1])
        S1 = b2.Synapses(self.rew, self.DA, on_pre=f"ampa_post += {a_e:.6g}*mV", name=f"{name}_rew_syn")
        S1.connect(p=p["p_in"])
        S2 = b2.Synapses(self.inh, self.DA, on_pre=f"gaba_post += {a_i:.6g}*mV", name=f"{name}_inh_syn")
        S2.connect(p=p["p_in"])
        # critic: striosomal D1 cells with the loop's D1 membrane, driven by the context cells
        m = MEMBRANE["D1"]
        ceqs = """
        dv/dt = (-v + ampa + nmda + v_spon + v_off) / tau_m + sigma_step * xi / sqrt(dt_noise) : volt (unless refractory)
        dampa/dt = -ampa / tau_ampa : volt
        dnmda/dt = -nmda / tau_nmda : volt
        v_off : volt (constant)
        """
        cns = dict(tau_m=m["tau"] * ms, v_spon=m["R"] * m["I_spon"] * mV, sigma_step=NOISE_SD_MV * mV, dt_noise=dt,
                   tau_ampa=PSP["ampa"][1] * ms, tau_nmda=PSP["nmda"][1] * ms, theta=m["theta"] * mV)
        self.critic = b2.NeuronGroup(int(p["crit_n"]), ceqs, threshold="v > theta", reset="v = 0*mV",
                                     refractory=REFRACTORY_MS * ms, method="euler", namespace=cns, name=f"{name}_striosome")
        self.critic.v = self.rng.uniform(-10, 10, int(p["crit_n"])) * mV
        self.critic.v_off = self.rng.uniform(-float(p["crit_spread_mV"]), 0.0, int(p["crit_n"])) * mV
        lam = float(bg.p["da"])
        gain = float(p["crit_gain"]) * (1.0 + lam)
        psp = bg._onpre("glut", "D1", gain).replace("*mV", "*mV*wf")
        model = """wf : 1
        elig : 1
        t_e : second
        dxpre/dt = -xpre / tau_x : 1 (event-driven)
        dxpost/dt = -xpost / tau_x : 1 (event-driven)"""
        on_pre = psp + "; elig = elig * exp(-(t - t_e) / tau_e) - a_minus * xpost; t_e = t; xpre += 1"
        on_post = "elig = elig * exp(-(t - t_e) / tau_e) + xpre; t_e = t; xpost += 1"
        self.crit_syn = b2.Synapses(g, self.critic, model=model, on_pre=on_pre, on_post=on_post,
                                    delay=CTX_DELAY_MS["D1"] * ms, name=f"{name}_cue_critic",
                                    namespace=dict(tau_x=bg.p["tau_x_ms"] * ms, tau_e=bg.p["tau_e_ms"] * ms,
                                                   a_minus=float(bg.p["a_minus"])))
        n_in = int(p["crit_n_in"])
        p_conn = min(1.0, n_in / max(1, idx.size)); wf_scale = max(1.0, n_in / max(1, idx.size))
        pre_i = np.repeat(idx, int(p["crit_n"])); post_j = np.tile(np.arange(int(p["crit_n"])), idx.size)
        keep = self.rng.random(pre_i.size) < p_conn
        self.crit_syn.connect(i=pre_i[keep], j=post_j[keep])
        spread = float(bg.p["w_init_spread"])
        self.crit_syn.wf = wf_scale * self.rng.uniform(1.0 - spread, 1.0 + spread, int(keep.sum()))
        self.crit_syn.elig = 0.0; self.crit_syn.t_e = 0 * ms
        self.crit_w0 = np.asarray(self.crit_syn.wf).copy()
        # critic -> DA: slow inhibition (the prediction), scaled so one spike gives crit_to_da_mV peak
        a_s = float(p["crit_to_da_mV"]) * psp_scale(p["tau_ms"], p["crit_tau_ms"])
        S3 = b2.Synapses(self.critic, self.DA, on_pre=f"gslow_post += {a_s:.6g}*mV", name=f"{name}_critic_da")
        S3.connect(p=p["p_in"])
        self.mon = b2.SpikeMonitor(self.DA, name=f"{name}_mon")
        self.mon_c = b2.SpikeMonitor(self.critic, name=f"{name}_mon_critic")
        self.objects = [self.DA, self.rew, self.inh, S1, S2, self.critic, self.crit_syn, S3, self.mon, self.mon_c]
        self.value = self.value_syn = self.mon_v = None
        if int(p["value_n"]) > 0:
            nv = int(p["value_n"])
            veqs = """
            dv/dt = (-v + ampa + nmda + v_off) / tau_m + sigma_step * xi / sqrt(dt_noise) : volt (unless refractory)
            dampa/dt = -ampa / tau_ampa : volt
            dnmda/dt = -nmda / tau_nmda : volt
            v_off : volt (constant)
            """
            vns = dict(tau_m=p["value_tau_ms"] * ms, sigma_step=NOISE_SD_MV * mV, dt_noise=dt, tau_ampa=PSP["ampa"][1] * ms,
                       tau_nmda=PSP["nmda"][1] * ms, theta=p["value_theta_mV"] * mV)
            self.value = b2.NeuronGroup(nv, veqs, threshold="v > theta", reset="v = 0*mV", refractory=REFRACTORY_MS * ms,
                                        method="euler", namespace=vns, name=f"{name}_pptg")
            lo, hi = p["value_offset_mV"]
            self.value.v_off = self.rng.uniform(lo, hi, nv) * mV
            vpsp = PSP["ampa"][0] * psp_scale(p["value_tau_ms"], PSP["ampa"][1]) * float(p["value_gain"])
            vnm = PSP["nmda"][0] * psp_scale(p["value_tau_ms"], PSP["nmda"][1]) * float(p["value_gain"])
            von_pre = (f"ampa_post += {vpsp:.6g}*mV*wf; nmda_post += {vnm:.6g}*mV*wf; "
                       "elig = elig * exp(-(t - t_e) / tau_e) - a_minus * xpost; t_e = t; xpre += 1")
            self.value_syn = b2.Synapses(g, self.value, model=model, on_pre=von_pre, on_post=on_post, delay=5 * ms,
                                         name=f"{name}_cue_pptg",
                                         namespace=dict(tau_x=bg.p["tau_x_ms"] * ms, tau_e=bg.p["tau_e_ms"] * ms,
                                                        a_minus=float(bg.p["a_minus"])))
            n_in = int(p["value_n_in"])
            p_c = min(1.0, n_in / max(1, idx.size)); wfs = max(1.0, n_in / max(1, idx.size))
            pi_ = np.repeat(idx, nv); pj_ = np.tile(np.arange(nv), idx.size)
            kp = self.rng.random(pi_.size) < p_c
            self.value_syn.connect(i=pi_[kp], j=pj_[kp])
            self.value_syn.wf = wfs * self.rng.uniform(1.0 - spread, 1.0 + spread, int(kp.sum()))
            self.value_syn.elig = 0.0; self.value_syn.t_e = 0 * ms
            self.value_w0 = np.asarray(self.value_syn.wf).copy()
            a_v = float(p["value_to_da_mV"]) * psp_scale(p["tau_ms"], PSP["ampa"][1])
            S4 = b2.Synapses(self.value, self.DA, on_pre=f"ampa_post += {a_v:.6g}*mV", delay=3 * ms, name=f"{name}_pptg_da")
            S4.connect(p=p["p_in"])
            self.mon_v = b2.SpikeMonitor(self.value, name=f"{name}_mon_pptg")
            self.objects += [self.value, self.value_syn, S4, self.mon_v]
        self._event = None

    # ------------------------------------------------------------------ events
    def deliver(self, value):
        """start the primary-outcome input: value > 0 excites (reward), value < 0 inhibits (LHb/RMTg)"""
        v = float(value)
        self.rew.rates = (max(v, 0.0) * self.p["rew_hz"]) * Hz
        self.inh.rates = (max(-v, 0.0) * self.p["inh_hz"]) * Hz
        self._event = float(b2.defaultclock.t / ms)

    def end_event(self):
        self.rew.rates = 0 * Hz; self.inh.rates = 0 * Hz

    def rate(self, t0_ms, t1_ms, which="da"):
        mon = {"da": self.mon, "critic": self.mon_c, "value": self.mon_v}[which]
        if mon is None:
            return 0.0
        t = np.asarray(mon.t / ms)
        n = int(((t >= t0_ms) & (t < t1_ms)).sum())
        N = {"da": int(self.p["n"]), "critic": int(self.p["crit_n"]), "value": int(self.p["value_n"])}[which]
        return n / N / ((t1_ms - t0_ms) / 1000.0)

    def delta(self, t0_ms, t1_ms):
        """reward-prediction error read from the dopamine rate: (rate - tonic) / burst_scale,
        negative values x dip_gain"""
        d = (self.rate(t0_ms, t1_ms) - self.p["tonic_hz"]) / self.p["burst_scale_hz"]
        return float(d if d >= 0 else d * self.p["dip_gain"])

    def learn(self, delta, t_ms=None):
        """critic update: cue -> striosome weights += crit_eta * delta * eligibility"""
        now = float(b2.defaultclock.t / ms) if t_ms is None else float(t_ms)
        S = self.crit_syn
        e = np.asarray(S.elig) * np.exp(-np.clip(now - np.asarray(S.t_e / ms), 0, None) / self.bg.p["tau_e_ms"])
        w = np.clip(np.asarray(S.wf) + self.p["crit_eta"] * float(delta) * e, 0.0, self.p["crit_w_max_factor"] * self.crit_w0)
        S.wf = w
        if self.value_syn is not None:              # the fast value pathway learns from the same error
            V = self.value_syn
            ev = np.asarray(V.elig) * np.exp(-np.clip(now - np.asarray(V.t_e / ms), 0, None) / self.bg.p["tau_e_ms"])
            V.wf = np.clip(np.asarray(V.wf) + self.p["value_eta"] * float(delta) * ev, 0.0,
                           self.p["value_w_max_factor"] * self.value_w0)
        return float(np.mean(w / self.crit_w0))

    def clear(self):
        for mon in (self.mon, self.mon_c) + ((self.mon_v,) if self.mon_v is not None else ()):
            mon.resize(0)
            mon.variables["N"].set_value(0)
