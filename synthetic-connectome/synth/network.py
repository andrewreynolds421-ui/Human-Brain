"""Build the Brian2 objects for a synthesized column (shared by simulate.py
and live.py), for either synapse model:

cuba  dv/dt = (E_L - v)/tau_m + (I_syn + I_dc + I_stim)/C_m
      dI_syn/dt = -I_syn/tau_syn ;  spike: I_syn += w (pA)

coba  dv/dt = (E_L - v)/tau_m + (g_e (E_e - v) + g_i (E_i - v) + I_dc + I_stim)/C_m
      dg_e/dt = -g_e/tau_e ; dg_i/dt = -g_i/tau_i ;  spike: g_e += w or g_i += w (nS)

Returns a dict with the neuron group, synapse groups (E-presynaptic and
I-presynaptic in coba, one group in cuba), external inputs, per-layer
stimulus generators, the spike monitor and the Network.
"""
from __future__ import annotations

import numpy as np
import brian2 as b2
from brian2 import ms, mV, pA, pF, nS, Hz

from .params import layers_of

STP_MODEL = """
w : {unit}
U : 1
tau_rec : second
tau_fac : second
x : 1
u : 1
t_last : second
"""
STP_PRE = """
x = 1 + (x - 1) * exp(-(t - t_last) / tau_rec)
u = u * exp(-(t - t_last) / tau_fac)
u = u + U * (1 - u)
{target}_post += w * u * x
x = x - u * x
t_last = t
"""


def _poisson_inv_numpy(lam, e, u):
    lam, e, u = np.broadcast_arrays(np.asarray(lam, float), np.asarray(e, float), np.asarray(u, float))
    k = np.zeros(lam.shape, int); p = e.copy(); s_ = e.copy()
    todo = u > s_
    while todo.any() and k.max() < 1000:
        k[todo] += 1
        p[todo] *= lam[todo] / k[todo]
        s_[todo] += p[todo]
        todo = u > s_
    return k


POISSON_INV = b2.Function(_poisson_inv_numpy, arg_units=[1, 1, 1], return_unit=1, return_type="integer",
                          arg_names=["lam", "e", "u"], stateless=True)
# Poisson sample by inversion of one uniform u, with e = exp(-lam) precomputed per
# neuron: exact, and cheaper than a generic Poisson draw for the small lam here
POISSON_INV.implementations.add_implementation("cython", """
cdef int poisson_inv(double lam, double e, double u):
    cdef int k = 0
    cdef double p = e
    cdef double s = e
    while u > s and k < 1000:
        k += 1
        p *= lam / k
        s += p
    return k
""")


def _max_delay(col):
    """longest outgoing axonal delay of each neuron (ms)"""
    m = np.zeros(col.N)
    np.maximum.at(m, col.pre, np.asarray(col.delay_ms, float))
    return m


def stp_slot_overflows(col, spike_i, spike_t_ms, slots=3):
    """shared_stp check: number of spikes whose STP value was no longer held when
    it arrived at the neuron's slowest synapse (a later spike count >= slots
    within that delay).  0 = the shared state was exact for this run."""
    spike_i = np.asarray(spike_i); spike_t_ms = np.asarray(spike_t_ms, float)
    dmax = _max_delay(col)
    order = np.lexsort((spike_t_ms, spike_i))
    i_s, t_s = spike_i[order], spike_t_ms[order]
    bad = 0
    if i_s.size > slots:
        same = i_s[slots:] == i_s[:-slots]
        gap = t_s[slots:] - t_s[:-slots]
        bad = int(np.sum(same & (gap < dmax[i_s[:-slots]])))
    return bad


def build_network(col, seed=0, with_stimulus=False, ext_mode="poisson", ext_rate_factor=1.0, recurrent=True,
                  bg_ratio=None, bg_rate_hz=5.0, bg_w_exc_nS=0.5, bg_w_inh_nS=1.0, bg_balance_mV=-60.0,
                  bg_by_membrane=None, istdp=None, rstdp=None, merged_inputs=False, shared_stp=False,
                  stp_slots=3):
    """ext_mode: "poisson" (default, independent Poisson fibres per neuron),
    "constant" (the same mean drive as a steady conductance / current, no
    fluctuations), or "none".  ext_rate_factor > 1 uses that many times
    fewer fibres firing that many times faster (same mean drive).
    recurrent=False drops all within-column synapses (diagnostics).
    istdp: dict(eta=..., tau_ms=20, target_hz={pop: rate}, w_max_factor=10) makes
    inhibitory -> pyramidal synapses plastic with the Vogels et al. (2011,
    Science) rule: each pre spike w += eta (x_post - alpha), each post spike
    w += eta x_pre, alpha = 2 * target * tau.  Inhibition onto a pyramidal cell
    grows while it fires above its target rate and shrinks below it.
    rstdp: dict(mask=bool array over synapses, tau_ms=20, tau_elig_ms=500,
    a_plus=1.0, a_minus=1.05) gives the masked excitatory synapses a reward
    eligibility trace (three-factor rule): pre-before-post pairings add to
    ``elig``, post-before-pre subtract; the trace decays with tau_elig
    (stored lazily with its last update time ``t_e``).  Weights are changed by
    the caller from reward: w += eta * (R - baseline) * elig.
    bg_ratio (coba only): add a balanced synaptic background, as in vivo
    (high-conductance state): mean background conductance = bg_ratio x the
    neuron's leak conductance, split between excitation and inhibition so its
    mean current is zero at bg_balance_mV.  None/0 = off (default)."""
    nrn = col.neuron
    coba = nrn.get("synapse_model", "cuba") == "coba"
    rng = np.random.default_rng(seed)
    b2.seed(seed)
    pop = col.pop_of
    pop16 = np.asarray(pop, np.int16)            # per-synapse lookups at 2 bytes instead of 8
    POPS = col.pops
    N_POP = len(POPS)
    IS_EXC = col.is_exc

    two_comp = coba and bool(nrn.get("two_compartment"))
    if two_comp:
        eqs = """
        dv/dt = (gL_s * (E_L - v) + g_c * (v_d - v) + (g_e + g_ext) * (E_e - v) + g_i * (E_i - v) + I_dc + I_stim - w_a)/C_s : volt (unless refractory)
        dv_d/dt = (gL_d * (E_L - v_d) + g_c * (v - v_d) + (g_ed + g_extd) * (E_e - v_d) + g_id * (E_i - v_d))/C_d : volt
        g_ext : siemens
        g_extd : siemens
        dw_a/dt = -w_a/tau_w : amp
        b_a : amp (constant)
        dg_e/dt = -g_e/tau_e : siemens
        dg_i/dt = -g_i/tau_i : siemens
        dg_ed/dt = -g_ed/tau_e : siemens
        dg_id/dt = -g_id/tau_i : siemens
        tau_e : second (constant)
        tau_i : second (constant)
        dI_stim/dt = -I_stim/tau_stim : amp
        I_dc : amp
        tau_m : second (constant)
        C_m : farad (constant)
        C_s : farad (constant)
        C_d : farad (constant)
        gL_s : siemens (constant)
        gL_d : siemens (constant)
        g_c : siemens (constant)
        E_L : volt (constant)
        V_th : volt (constant)
        V_reset : volt (constant)
        """
        ns = dict(E_e=nrn["E_exc_mV"] * mV, E_i=nrn["E_inh_mV"] * mV, tau_stim=nrn["tau_exc_ms"] * ms)
    elif coba:
        eqs = """
        dv/dt = (E_L - v)/tau_m + ((g_e + g_ext) * (E_e - v) + g_i * (E_i - v) + I_dc + I_stim - w_a)/C_m : volt (unless refractory)
        g_ext : siemens
        dw_a/dt = -w_a/tau_w : amp
        b_a : amp (constant)
        dg_e/dt = -g_e/tau_e : siemens
        dg_i/dt = -g_i/tau_i : siemens
        tau_e : second (constant)
        tau_i : second (constant)
        dI_stim/dt = -I_stim/tau_stim : amp
        I_dc : amp
        tau_m : second (constant)
        C_m : farad (constant)
        E_L : volt (constant)
        V_th : volt (constant)
        V_reset : volt (constant)
        """
        ns = dict(E_e=nrn["E_exc_mV"] * mV, E_i=nrn["E_inh_mV"] * mV,
                  tau_stim=nrn["tau_exc_ms"] * ms)
    else:
        eqs = """
        dv/dt = (E_L - v)/tau_m + (I_syn + I_ext + I_dc + I_stim - w_a)/C_m : volt (unless refractory)
        I_ext : amp
        dw_a/dt = -w_a/tau_w : amp
        b_a : amp (constant)
        dI_syn/dt = -I_syn/tau_syn : amp
        dI_stim/dt = -I_stim/tau_syn : amp
        I_dc : amp
        tau_m : second (constant)
        C_m : farad (constant)
        E_L : volt (constant)
        V_th : volt (constant)
        V_reset : volt (constant)
        """
        ns = dict(tau_syn=nrn["tau_syn"] * ms)

    # merged_inputs: every Poisson input (external drive + background) becomes a
    # per-neuron channel drawn in ONE vectorised update per time step, instead of
    # one PoissonInput object per population/area/compartment.  Each channel adds
    # w * Poisson(N * rate * dt): the same statistics as PoissonInput's
    # Binomial(N, rate * dt) to within (rate * dt) ~ 1e-3.
    MERGED = bool(merged_inputs) and coba
    # channels: one external drive per neuron (onto g_e or g_ed) + one background
    # stream per conductance.  lam = expected events per step, e_* = exp(-lam).
    chan_vars = (["g_e", "g_i"] + (["g_ed", "g_id"] if two_comp else [])) if MERGED else []
    ext_vars = (["g_e"] + (["g_ed"] if two_comp else [])) if MERGED else []
    if MERGED:
        ns["poisson_inv"] = POISSON_INV
        eqs += "\n        lam_ext : 1 (constant)\n        e_ext : 1 (constant)"
        eqs += "".join(f"\n        w_ext_{v_} : siemens (constant)" for v_ in ext_vars)
        eqs += "".join(f"\n        lam_bg_{v_} : 1 (constant)\n        e_bg_{v_} : 1 (constant)\n        w_bg_{v_} : siemens (constant)"
                       for v_ in chan_vars) + "\n"
    # shared_stp: STP parameters are fixed per (presynaptic, postsynaptic population)
    # pair, so every synapse from one neuron in the same STP class carries an
    # identical depression/facilitation state.  It is computed once per neuron and
    # class at each spike (reset) and kept for the last ``stp_slots`` spikes; a
    # synapse reads the value for the spike that is arriving (emission time =
    # t - delay).  Exact unless a neuron fires ``stp_slots`` more spikes within one
    # axonal delay (check with ``stp_slot_overflows``).  The iSTDP postsynaptic
    # trace depends only on the postsynaptic neuron, so it moves there too (exact).
    SHARED = bool(shared_stp) and coba and recurrent and col.has_stp
    reset = "v = V_reset; w_a += b_a"
    regular = []
    syn_cls, n_cls, K = None, 0, max(int(stp_slots), 2)
    if SHARED:
        # STP settings are fixed per (pre, post) population pair: find each pair's settings from
        # its first synapse, check every synapse agrees (1 byte/synapse temporaries only), then
        # number the distinct settings per presynaptic population -> class of each synapse
        pair = (pop16[col.pre] * N_POP + pop16[col.post]).astype(np.int16)
        present = np.flatnonzero(np.bincount(pair, minlength=N_POP * N_POP))
        pars32 = [np.asarray(a_, np.float32) for a_ in (col.stp_U, col.stp_tau_rec_ms, col.stp_tau_fac_ms)]
        pair_par = np.zeros((N_POP * N_POP, 3), np.float32)
        for code in present:                                 # first synapse of each population pair
            f_ = int(np.argmax(pair == code))
            pair_par[code] = [a_[f_] for a_ in pars32]
        for j_, a_ in enumerate(pars32):
            if not np.array_equal(a_, pair_par[pair, j_]):
                raise ValueError("shared_stp needs STP settings fixed per population pair")
        cls_of_pair = np.zeros(N_POP * N_POP, np.int8)
        cls_tab = []
        for sp in range(N_POP):
            pp = present[present // N_POP == sp]
            uniq, inv = (np.unique(pair_par[pp].astype(float), axis=0, return_inverse=True) if pp.size
                         else (np.zeros((0, 3)), None))
            if pp.size:
                cls_of_pair[pp] = inv.ravel()
            cls_tab.append(uniq)
        syn_cls = cls_of_pair[pair]
        del pair
        n_cls = max(len(u_) for u_ in cls_tab)
        for c in range(n_cls):
            eqs += f"\n        U_c{c} : 1 (constant)\n        trr_c{c} : second (constant)\n        tff_c{c} : second (constant)"
            eqs += f"\n        xs_c{c} : 1\n        us_c{c} : 1"
            eqs += "".join(f"\n        ux_c{c}_{k} : 1" for k in range(K))
        eqs += "".join(f"\n        ts_{k} : second" for k in range(K)) + "\n"
        for c in range(n_cls):
            reset += (f"\nxs_c{c} = 1 + (xs_c{c} - 1) * exp(-(t - ts_0) / trr_c{c})"
                      f"\nus_c{c} = us_c{c} * exp(-(t - ts_0) / tff_c{c})"
                      f"\nus_c{c} = us_c{c} + U_c{c} * (1 - us_c{c})")
            reset += "".join(f"\nux_c{c}_{k} = ux_c{c}_{k - 1}" for k in range(K - 1, 0, -1))
            reset += f"\nux_c{c}_0 = us_c{c} * xs_c{c}\nxs_c{c} = xs_c{c} - us_c{c} * xs_c{c}"
        if istdp:
            # presynaptic iSTDP trace right after each of the last K spikes
            eqs += "\n        itrace : 1\n        ialpha : 1 (constant)" + "".join(f"\n        ptr_{k} : 1" for k in range(K)) + "\n"
            reset += "".join(f"\nptr_{k} = ptr_{k - 1}" for k in range(K - 1, 0, -1))
            reset += "\nptr_0 = ptr_1 * exp(-(t - ts_0) / tau_istdp) + 1"
            reset += "\nitrace += 1"
            regular.append("itrace *= tr_decay")
            ns["tau_istdp"] = float(istdp.get("tau_ms", 20.0)) * ms
            ns["tr_decay"] = float(np.exp(-float(b2.defaultclock.dt / ms) / float(istdp.get("tau_ms", 20.0))))
        reset += "".join(f"\nts_{k} = ts_{k - 1}" for k in range(K - 1, 0, -1)) + "\nts_0 = t"
        ns["stp_eps"] = 0.6 * b2.defaultclock.dt
    ns["tau_w"] = float(nrn.get("adapt_tau_ms", 150.0)) * ms
    adapt_b = np.asarray(nrn.get("adapt_b_pop") or [0.0] * N_POP, float)
    G = b2.NeuronGroup(col.N, eqs, threshold="v > V_th", reset=reset,
                       refractory=nrn["t_ref"] * ms, method="euler", namespace=ns)
    G.b_a = adapt_b[pop] * pA
    G.w_a = 0 * pA

    def per_pop(key, scalar_key):
        arr = nrn.get(key)
        return np.asarray(arr if arr is not None else [nrn[scalar_key]] * N_POP, float)[pop]
    G.tau_m = per_pop("tau_m_pop", "tau_m") * ms
    G.C_m = per_pop("C_m_pop", "C_m") * pF
    G.E_L = per_pop("E_L_pop", "E_L") * mV
    G.V_th = per_pop("V_th_pop", "V_th") * mV
    G.V_reset = per_pop("V_reset_pop", "V_reset") * mV
    G.v = rng.normal(np.asarray(nrn["V0_mean"])[pop], np.asarray(nrn["V0_std"])[pop]) * mV
    G.I_dc = col.DC_pA[pop] * pA
    if two_comp:
        cp = {k: np.asarray(v, float)[pop] for k, v in nrn["comp"].items()}
        G.C_s = cp["C_s"] * pF; G.C_d = cp["C_d"] * pF
        G.gL_s = cp["gL_s"] * nS; G.gL_d = cp["gL_d"] * nS; G.g_c = cp["g_c"] * nS
        G.v_d = G.v[:]
    if coba:
        te = np.asarray(nrn.get("tau_exc_pop") or [nrn["tau_exc_ms"]] * N_POP, float)
        ti = np.asarray(nrn.get("tau_inh_pop") or [nrn["tau_inh_ms"]] * N_POP, float)
        G.tau_e = te[pop] * ms
        G.tau_i = ti[pop] * ms
    G.I_stim = 0 * pA
    if MERGED:
        # exp(-0) = 1 for channels a neuron does not use (else the sampler spins)
        G.e_ext = 1.0
        for v_ in chan_vars:
            setattr(G, f"e_bg_{v_}", 1.0)
    if SHARED:
        for c in range(n_cls):
            tab = np.array([cls_tab[sp][c] if c < len(cls_tab[sp]) else (1.0, 1e-3, 1e-3) for sp in range(N_POP)])
            setattr(G, f"U_c{c}", tab[pop, 0])
            setattr(G, f"trr_c{c}", np.maximum(tab[pop, 1], 1e-3) * ms)
            setattr(G, f"tff_c{c}", np.maximum(tab[pop, 2], 1e-3) * ms)
            setattr(G, f"xs_c{c}", 1.0); setattr(G, f"us_c{c}", 0.0)
        for k in range(K):
            setattr(G, f"ts_{k}", -1e9 * ms)
        if istdp:
            G.ialpha = (2.0 * np.array([istdp["target_hz"].get(POPS[k_], 2.0) for k_ in range(N_POP)])
                        * float(istdp.get("tau_ms", 20.0)) * 1e-3)[pop]
        if b2.defaultclock.dt * 1.0 > float(np.min(col.delay_ms)) * ms:
            raise ValueError("shared_stp needs every delay >= dt")
    w_ext_pop = np.asarray(nrn.get("w_ext_pop") or [col.w_ext_pA] * N_POP, float)
    unit, wunit = ("siemens", nS) if coba else ("amp", pA)

    def shared_pre(c, target):
        code = ["te = t - delay + stp_eps"]
        for k in range(K - 1):
            prev = "".join(f"(1 - s{j}) * " for j in range(k))
            code.append(f"s{k} = {prev}int(ts_{k}_pre <= te)")
        code.append(f"s{K - 1} = 1" + "".join(f" - s{j}" for j in range(K - 1)))
        code.append(f"{target}_post += w * (" + " + ".join(f"s{k} * ux_c{c}_{k}_pre" for k in range(K)) + ")")
        return "\n".join(code) + "\n"

    def make_syn(mask, target_var, plastic=False, cls=None, eta_w=None):
        pre, post = col.pre[mask], col.post[mask]
        if cls is not None and plastic != "reward":
            if plastic:
                # Vogels iSTDP without per-synapse traces: the postsynaptic trace is
                # the neuron's (itrace); the presynaptic trace *as it arrived at this
                # synapse* is rebuilt at a postsynaptic spike from the slot holding
                # the last spike that had arrived (ptr_k decayed from its arrival time)
                tau_p = float(istdp.get("tau_ms", 20.0)) * ms
                model = f"w : {unit}\neta : siemens (shared, constant)\nw_max : siemens\ndax : second (constant)\n"
                on_pre = shared_pre(cls, target_var) + "w = clip(w + eta * (itrace_post - ialpha_post), 0 * nS, w_max)\n"
                # ta: time t mapped back to emission time with the delay as Brian rounds it
                # (inside on_post `delay` is the post pathway's, i.e. 0: use dax, the axonal delay)
                post_code = ["ta = t - floor(dax / dt + 0.5) * dt", "te = ta + 0.5 * dt"]
                for k in range(K):
                    prev = "".join(f"(1 - q{j}) * " for j in range(k))
                    post_code.append(f"q{k} = {prev}int(ts_{k}_pre <= te)")
                post_code.append("trace_in = " + " + ".join(
                    f"q{k} * ptr_{k}_pre * exp(-(ta - ts_{k}_pre) / tau_istdp)" for k in range(K)))
                post_code.append("w = clip(w + eta * trace_in, 0 * nS, w_max)")
                S = b2.Synapses(G, G, model=model, on_pre=on_pre, on_post="\n".join(post_code) + "\n",
                                namespace=dict(tau_istdp=tau_p, stp_eps=ns["stp_eps"]))
                S.connect(i=pre, j=post)
                w0 = col.weight_pA[mask]
                S.w = w0 * wunit
                S.eta = float(istdp.get("eta", 1e-3)) * eta_w * nS
                S.dax = col.delay_ms[mask] * ms
                S.w_max = float(istdp.get("w_max_factor", 10.0)) * w0 * nS + 1e-12 * nS
            else:
                S = b2.Synapses(G, G, model=f"w : {unit}", on_pre=shared_pre(cls, target_var),
                                namespace=dict(stp_eps=ns["stp_eps"]))
                S.connect(i=pre, j=post)
                S.w = col.weight_pA[mask] * wunit
            S.delay = col.delay_ms[mask] * ms
            return S
        if plastic == "reward":
            tau_r = float(rstdp.get("tau_ms", 20.0)) * ms
            model = STP_MODEL.format(unit=unit) + """
            elig : 1
            t_e : second
            tau_el : second (shared, constant)
            a_plus : 1 (shared, constant)
            a_minus : 1 (shared, constant)
            dtr_a/dt = -tr_a / tau_rs : 1 (event-driven)
            dtr_b/dt = -tr_b / tau_rs : 1 (event-driven)
            """
            on_pre = STP_PRE.format(target=target_var) + """
            elig = elig * exp(-(t - t_e) / tau_el) - a_minus * tr_b
            t_e = t
            tr_a += 1
            """
            on_post = """
            elig = elig * exp(-(t - t_e) / tau_el) + a_plus * tr_a
            t_e = t
            tr_b += 1
            """
            S = b2.Synapses(G, G, model=model, on_pre=on_pre, on_post=on_post, namespace=dict(tau_rs=tau_r))
            S.connect(i=pre, j=post)
            S.tau_el = float(rstdp.get("tau_elig_ms", 500.0)) * ms
            S.a_plus = float(rstdp.get("a_plus", 1.0)); S.a_minus = float(rstdp.get("a_minus", 1.05))
            S.w = col.weight_pA[mask] * wunit
            S.U = col.stp_U[mask]; S.tau_rec = col.stp_tau_rec_ms[mask] * ms; S.tau_fac = col.stp_tau_fac_ms[mask] * ms
            S.x = 1.0; S.u = 0.0; S.t_last = -1e9 * ms; S.elig = 0.0; S.t_e = 0 * ms
            S.delay = col.delay_ms[mask] * ms
            return S
        if plastic:
            tau_p = float(istdp.get("tau_ms", 20.0)) * ms
            model = STP_MODEL.format(unit=unit) + """
            alpha : 1
            eta : siemens
            w_max : siemens
            dtr_a/dt = -tr_a / tau_istdp : 1 (event-driven)
            dtr_b/dt = -tr_b / tau_istdp : 1 (event-driven)
            """
            on_pre = STP_PRE.format(target=target_var) + """
            tr_a += 1
            w = clip(w + eta * (tr_b - alpha), 0 * nS, w_max)
            """
            on_post = """
            tr_b += 1
            w = clip(w + eta * tr_a, 0 * nS, w_max)
            """
            S = b2.Synapses(G, G, model=model, on_pre=on_pre, on_post=on_post, namespace=dict(tau_istdp=tau_p))
            S.connect(i=pre, j=post)
            w0 = col.weight_pA[mask]
            S.w = w0 * wunit
            S.U = col.stp_U[mask]; S.tau_rec = col.stp_tau_rec_ms[mask] * ms; S.tau_fac = col.stp_tau_fac_ms[mask] * ms
            S.x = 1.0; S.u = 0.0; S.t_last = -1e9 * ms
            tgt_hz = istdp["target_hz"]
            post_rate = np.array([tgt_hz.get(POPS[k], 2.0) for k in range(N_POP)])[pop[post]]
            S.alpha = 2.0 * post_rate * float(istdp.get("tau_ms", 20.0)) * 1e-3
            S.eta = float(istdp.get("eta", 1e-3)) * np.mean(w0) * nS
            S.w_max = float(istdp.get("w_max_factor", 10.0)) * w0 * nS + 1e-12 * nS
            S.delay = col.delay_ms[mask] * ms
            return S
        if col.has_stp:
            S = b2.Synapses(G, G, model=STP_MODEL.format(unit=unit), on_pre=STP_PRE.format(target=target_var))
            S.connect(i=pre, j=post)
            S.w = col.weight_pA[mask] * wunit
            S.U = col.stp_U[mask]; S.tau_rec = col.stp_tau_rec_ms[mask] * ms; S.tau_fac = col.stp_tau_fac_ms[mask] * ms
            S.x = 1.0; S.u = 0.0; S.t_last = -1e9 * ms
        else:
            S = b2.Synapses(G, G, model=f"w : {unit}", on_pre=f"{target_var}_post += w")
            S.connect(i=pre, j=post)
            S.w = col.weight_pA[mask] * wunit
        S.delay = col.delay_ms[mask] * ms
        return S

    syn_index, syn_plastic = [], []
    if not recurrent:
        synapses = []
    elif two_comp:
        from .dendrite import lands_on_dendrite
        types = np.array([p_[-1] for p_ in POPS])
        on_d_pop = np.array([[lands_on_dendrite(types[t], types[s_]) for s_ in range(N_POP)] for t in range(N_POP)])
        pre_exc = IS_EXC[pop16[col.pre]]
        on_d = (np.asarray(col.syn_on_dend, bool) if getattr(col, "syn_on_dend", None) is not None
                else on_d_pop[pop16[col.post], pop16[col.pre]])
        synapses = []
        post_exc = IS_EXC[pop16[col.post]]
        if rstdp is not None:
            rm = np.asarray(rstdp["mask"], bool)
            groups = [(pre_exc & ~on_d & ~rm, "g_e", False), (pre_exc & on_d & ~rm, "g_ed", False),
                      (pre_exc & ~on_d & rm, "g_e", "reward"), (pre_exc & on_d & rm, "g_ed", "reward")]
        else:
            groups = [(pre_exc & ~on_d, "g_e", False), (pre_exc & on_d, "g_ed", False)]
        if istdp:
            for base, var in ((~pre_exc & ~on_d, "g_i"), (~pre_exc & on_d, "g_id")):
                groups += [(base & ~post_exc, var, False), (base & post_exc, var, True)]
        else:
            groups += [(~pre_exc & ~on_d, "g_i", False), (~pre_exc & on_d, "g_id", False)]
        for mask, var, plastic in groups:
            if not mask.any():
                continue
            if SHARED and plastic != "reward":
                # one synapse group per STP class (each reads its class's shared state)
                eta_w = float(np.mean(col.weight_pA[mask]))
                for c in range(n_cls):
                    sub = mask & (syn_cls == c)
                    if sub.any():
                        synapses.append(make_syn(sub, var, plastic, cls=c, eta_w=eta_w))
                        syn_index.append(np.flatnonzero(sub)); syn_plastic.append(plastic)
            else:
                synapses.append(make_syn(mask, var, plastic)); syn_index.append(np.flatnonzero(mask))
                syn_plastic.append(plastic)
    elif coba:
        pre_exc = IS_EXC[pop[col.pre]]
        synapses = [make_syn(pre_exc, "g_e"), make_syn(~pre_exc, "g_i")]
        syn_index = [np.flatnonzero(pre_exc), np.flatnonzero(~pre_exc)]
        syn_plastic = [False, False]
    else:
        synapses = [make_syn(np.ones(col.S, bool), "I_syn")]
        syn_index = [np.arange(col.S)]
        syn_plastic = [False]

    slices = col.pop_slices()
    ext = []
    used = set()
    dt_s = float(b2.defaultclock.dt / b2.second)

    def poisson_in(lo, hi, var, N, rate, weight, kind):
        """PoissonInput onto neurons lo:hi, or a merged per-neuron channel"""
        if not MERGED:
            ext.append(b2.PoissonInput(G[lo:hi], var, N=N, rate=rate * Hz, weight=weight * wunit))
            return
        lam_n, e_n = (("lam_ext", "e_ext") if kind == "ext" else (f"lam_bg_{var}", f"e_bg_{var}"))
        lam = getattr(G, lam_n)
        assert not np.any(lam[lo:hi] > 0), f"channel {kind}/{var} used twice on the same neurons"
        lam[lo:hi] = N * rate * dt_s
        getattr(G, e_n)[lo:hi] = np.exp(-N * rate * dt_s)
        getattr(G, f"w_{kind}_{var}")[lo:hi] = weight * wunit
        used.add((var, kind))
    tau_ext_pop = (np.asarray(nrn.get("tau_exc_pop") or [nrn["tau_exc_ms"]] * N_POP, float) if coba
                   else np.full(N_POP, nrn["tau_syn"]))
    for p_i, p in enumerate(POPS):
        sl = slices[p]
        if sl.stop - sl.start == 0:
            continue
        n_fib = max(col.K_ext[p_i], 1.0)
        if ext_mode == "poisson":
            n_eff = max(int(round(n_fib / ext_rate_factor)), 1)
            rate = col.bg_rate if ext_rate_factor == 1.0 else col.bg_rate * n_fib / n_eff
            ext_var = ("g_ed" if (two_comp and IS_EXC[p_i]) else "g_e") if coba else "I_syn"
            scale = getattr(col, "ext_scale_per_neuron", None)
            if scale is None:
                poisson_in(sl.start, sl.stop, ext_var, n_eff, rate, w_ext_pop[p_i], "ext")
            else:
                # external input reduced per neuron (e.g. per area): one input per contiguous run
                sc = np.asarray(scale)[sl.start:sl.stop]
                cuts = np.flatnonzero(np.diff(sc) != 0) + 1
                for a, b in zip(np.r_[0, cuts], np.r_[cuts, sc.size]):
                    poisson_in(sl.start + a, sl.start + b, ext_var, max(int(round(n_eff * sc[a])), 1),
                               rate, w_ext_pop[p_i], "ext")
        elif ext_mode == "constant":
            mean = n_fib * col.bg_rate * w_ext_pop[p_i] * tau_ext_pop[p_i] * 1e-3   # nS or pA
            if two_comp and IS_EXC[p_i]:
                G.g_extd[sl] = mean * nS
            elif coba:
                G.g_ext[sl] = mean * nS
            else:
                G.I_ext[sl] = mean * pA
    bg_ratio = bg_ratio if bg_ratio is not None else nrn.get("bg_ratio")
    if coba and bg_ratio:
        E_e, E_i = nrn["E_exc_mV"], nrn["E_inh_mV"]
        frac_e = (bg_balance_mV - E_i) / ((E_e - bg_balance_mV) + (bg_balance_mV - E_i))   # share of conductance that is excitatory
        tau_m_p = np.asarray(nrn.get("tau_m_pop") or [nrn["tau_m"]] * N_POP, float)
        C_p = np.asarray(nrn.get("C_m_pop") or [nrn["C_m"]] * N_POP, float)
        te = np.asarray(nrn.get("tau_exc_pop") or [nrn["tau_exc_ms"]] * N_POP, float)
        ti = np.asarray(nrn.get("tau_inh_pop") or [nrn["tau_inh_ms"]] * N_POP, float)
        for p_i, p in enumerate(POPS):
            sl = slices[p]
            if sl.stop - sl.start == 0:
                continue
            gL = C_p[p_i] / tau_m_p[p_i]                     # nS (pF / ms)
            Gbg = bg_ratio * gL
            # with two compartments the background is spread over the membrane:
            # dendrite share = its share of membrane (bg_by_membrane, default on)
            by_mem = (bg_by_membrane if bg_by_membrane is not None else nrn.get("bg_by_membrane", True))
            if two_comp and IS_EXC[p_i] and by_mem:
                cpp = nrn["comp"]; cd = cpp["C_d"][p_i] / (cpp["C_s"][p_i] + cpp["C_d"][p_i])
                parts = (("", 1 - cd), ("d", cd))
            else:
                parts = (("", 1.0),)
            for suffix, share in parts:
                for var, Gx, w, tau in (("g_e" + suffix, Gbg * frac_e * share, bg_w_exc_nS, te[p_i]),
                                        ("g_i" + suffix, Gbg * (1 - frac_e) * share, bg_w_inh_nS, ti[p_i])):
                    n = max(int(round(Gx / (w * bg_rate_hz * tau * 1e-3))), 1)
                    poisson_in(sl.start, sl.stop, var, n, bg_rate_hz, w, "bg")
    if MERGED and used:
        code = []
        if any(k_ == "ext" for _, k_ in used):
            code.append("n_ext = poisson_inv(lam_ext, e_ext, rand())")
            code += [f"{v_} += w_ext_{v_} * n_ext" for v_ in ext_vars if (v_, "ext") in used]
        code += [f"{v_} += w_bg_{v_} * poisson_inv(lam_bg_{v_}, e_bg_{v_}, rand())" for v_ in chan_vars if (v_, "bg") in used]
        ext.append(G.run_regularly("\n".join(code), dt=b2.defaultclock.dt, when="synapses"))
    if regular:
        ext.append(G.run_regularly("\n".join(regular), dt=b2.defaultclock.dt, when="before_synapses"))
    stim_groups = {}
    if with_stimulus:
        for layer, pops in layers_of(POPS).items():
            a, bnd = slices[POPS[pops[0]]].start, slices[POPS[pops[-1]]].stop
            sub = G[a:bnd]
            pg = b2.PoissonGroup(len(sub), rates=0 * Hz, name=f"stim_{layer}")
            Sst = b2.Synapses(pg, sub, model="w_stim : amp", on_pre="I_stim_post += w_stim")
            Sst.connect(j="i")
            Sst.w_stim = 0 * pA
            stim_groups[layer] = (pg, Sst, len(sub))
            ext += [pg, Sst]

    spikes = b2.SpikeMonitor(G)
    net = b2.Network(G, *synapses, spikes, *ext)
    reward_groups = [(S_, idx) for S_, idx, pl_ in zip(synapses, syn_index, syn_plastic) if pl_ == "reward"]
    return dict(G=G, synapses=synapses, syn_index=syn_index, reward_groups=reward_groups, shared_stp=SHARED,
                stp_slots=K if SHARED else None, ext=ext, stim_groups=stim_groups, spikes=spikes, net=net,
                coba=coba, w_ext_pop=w_ext_pop, wunit=wunit)
