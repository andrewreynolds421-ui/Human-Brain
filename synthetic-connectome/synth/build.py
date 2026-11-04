"""Synthesize an explicit cortical column from layer statistics.

This is the "statistical connectomics" step: the inputs are population
sizes, connection probabilities, weight / delay distributions and a spatial
profile; the output is a concrete list of neurons with 3D positions and a
concrete list of synapses (pre, post, weight, delay).

Scaling
-------
A full 1 mm^2 column has ~77k neurons and ~300M synapses, too many for a
laptop-class machine.  Following Potjans & Diesmann / the NEST reference
implementation, the network can be shrunk with two factors:

* ``N_scaling`` shrinks the number of neurons,
* ``K_scaling`` shrinks the number of inputs per neuron (in-degree).

When in-degrees are reduced, synaptic weights are increased by
1/sqrt(K_scaling) to preserve input *variance*, and a DC current is added
to preserve input *mean*, given the known full-scale firing rates.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from .params import ColumnParams, POPULATIONS, is_exc_of, pop_layer


# ---------------------------------------------------------------------------
# helpers (formulas from Potjans & Diesmann 2014 / NEST helpers.py)
# ---------------------------------------------------------------------------

def num_synapses_from_conn_probs(conn_probs, n_target, n_source):
    """Total synapse count per population pair, allowing multapses.

    Inverts  p = 1 - (1 - 1/(Nt*Ns))**K  for the synapse count K.
    """
    prod = np.outer(n_target, n_source).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.log(1.0 - conn_probs) / np.log((prod - 1.0) / prod)
    k[~np.isfinite(k)] = 0.0
    return k


def psc_over_psp(C_m, tau_m, tau_syn):
    """Factor converting a PSP amplitude (mV) to a PSC amplitude (pA)."""
    sub = 1.0 / (tau_syn - tau_m)
    pre = tau_m * tau_syn / C_m * sub
    frac = (tau_m / tau_syn) ** sub
    return 1.0 / (pre * (frac ** tau_m - frac ** tau_syn))


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------

@dataclass
class SynthesizedColumn:
    name: str
    N_scaling: float
    K_scaling: float
    # neurons
    n_per_pop: np.ndarray          # (8,)
    pop_of: np.ndarray             # (N,) population index per neuron
    pos_um: np.ndarray             # (N, 3) x, y, depth
    # synapses
    pre: np.ndarray                # (S,) int32
    post: np.ndarray               # (S,) int32
    weight_pA: np.ndarray          # (S,) float32
    delay_ms: np.ndarray           # (S,) float32
    # external drive per population
    K_ext: np.ndarray              # (8,) scaled external in-degree
    w_ext_pA: float
    bg_rate: float
    DC_pA: np.ndarray              # (8,) compensating DC current
    # neuron model
    neuron: dict
    # optional short-term plasticity per synapse (None = static synapses)
    stp_U: np.ndarray = None       # (S,) float32, release fraction
    stp_tau_rec_ms: np.ndarray = None
    stp_tau_fac_ms: np.ndarray = None
    populations: list = None       # population names (None = the 8 PD2014 populations)
    syn_on_dend: np.ndarray = None # (S,) bool, two-compartment placement per synapse (None = by population pair)

    @property
    def pops(self) -> list:
        return list(self.populations) if self.populations is not None else list(POPULATIONS)

    @property
    def is_exc(self) -> np.ndarray:
        return is_exc_of(self.pops)

    @property
    def has_stp(self) -> bool:
        return self.stp_U is not None and self.stp_U.size == self.pre.size

    @property
    def N(self) -> int:
        return int(self.pop_of.size)

    @property
    def S(self) -> int:
        return int(self.pre.size)

    def pop_slices(self):
        starts = np.concatenate([[0], np.cumsum(self.n_per_pop)[:-1]])
        return {p: slice(int(s), int(s + n))
                for p, s, n in zip(self.pops, starts, self.n_per_pop)}

    def describe(self) -> str:
        lines = [f"Synthesized column '{self.name}'  "
                 f"(N_scaling={self.N_scaling}, K_scaling={self.K_scaling})",
                 f"  neurons : {self.N:,}",
                 f"  synapses: {self.S:,}  "
                 f"(mean in-degree {self.S / self.N:.0f})"]
        indeg = np.bincount(self.post, minlength=self.N)
        exc_syn = self.is_exc[self.pop_of[self.pre]]
        for i, p in enumerate(self.pops):
            sel = self.pop_of == i
            fe = exc_syn[np.isin(self.post, np.flatnonzero(sel))].mean() if sel.any() else 0
            lines.append(f"  {p:5s} n={sel.sum():6d}  in-degree={indeg[sel].mean():7.1f}"
                         f"  exc fraction of inputs={fe:.2f}  DC={self.DC_pA[i]:.1f} pA")
        d = np.linalg.norm(self.pos_um[self.pre, :2] - self.pos_um[self.post, :2], axis=1)
        lines.append(f"  mean lateral distance of a synapse: {d.mean():.0f} um")
        return "\n".join(lines)

    def save(self, path):
        np.savez_compressed(path, **{k: v for k, v in asdict(self).items()
                                     if not isinstance(v, dict) and v is not None},
                            neuron=np.array([self.neuron], dtype=object))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d["neuron"] = d["neuron"][0]
        for k in ("name",):
            d[k] = str(d[k])
        if "populations" in d:
            d["populations"] = [str(x) for x in d["populations"]]
        for k in ("N_scaling", "K_scaling", "w_ext_pA", "bg_rate"):
            d[k] = float(d[k])
        return cls(**d)


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------

def _layer_of_pop(pop: str) -> str:
    return pop_layer(pop)


def place_neurons(params: ColumnParams, n_per_pop, rng):
    """Uniform random positions within each population's layer."""
    bounds = params.layer_bounds_um
    pos, pop_of = [], []
    for i, (pop, n) in enumerate(zip(params.populations, n_per_pop)):
        top, bottom = bounds[_layer_of_pop(pop)]
        xy = rng.uniform(0, params.footprint_um, size=(n, 2))
        z = rng.uniform(top, bottom, size=(n, 1))
        pos.append(np.hstack([xy, z]))
        pop_of.append(np.full(n, i, dtype=np.int16))
    return np.vstack(pos), np.concatenate(pop_of)


def _sample_sources_spatial(target_xy, source_xy, counts, sigma, rng):
    """For each target, draw `counts[i]` sources with Gaussian lateral bias.

    Normalising per target keeps every neuron's in-degree exactly as the
    population statistics dictate; only *which* sources are chosen
    depends on distance.
    """
    out = np.empty(int(counts.sum()), dtype=np.int32)
    k = 0
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    chunk = 256
    for a in range(0, target_xy.shape[0], chunk):
        b = min(a + chunk, target_xy.shape[0])
        d2 = ((target_xy[a:b, None, :] - source_xy[None, :, :]) ** 2).sum(-1)
        w = np.exp(-d2 * inv2s2)
        cdf = np.cumsum(w, axis=1)
        cdf /= cdf[:, -1:]
        for r in range(b - a):
            c = int(counts[a + r])
            if c == 0:
                continue
            u = rng.random(c)
            out[k:k + c] = np.searchsorted(cdf[r], u)
            k += c
    np.clip(out, 0, source_xy.shape[0] - 1, out=out)
    return out


def synthesize(params: ColumnParams, N_scaling=0.1, K_scaling=0.1,
               spatial=True, seed=0, dt_ms=0.1) -> SynthesizedColumn:
    rng = np.random.default_rng(seed)
    nrn = params.neuron
    POPS = list(params.populations)
    N_POP = len(POPS)
    IS_EXC = params.is_exc

    # ---- neurons -----------------------------------------------------------
    n_full = params.full_num_neurons.astype(float)
    n_per_pop = np.round(n_full * N_scaling).astype(int)
    pos, pop_of = place_neurons(params, n_per_pop, rng)
    offsets = np.concatenate([[0], np.cumsum(n_per_pop)[:-1]])

    # ---- synapse numbers ----------------------------------------------------
    syn_full = num_synapses_from_conn_probs(params.conn_probs, n_full, n_full)
    indeg_full = syn_full / n_full[:, None]              # per target neuron
    indeg = indeg_full * K_scaling
    n_syn = np.round(indeg * n_per_pop[:, None]).astype(int)   # (target, source)

    # ---- weights ------------------------------------------------------------
    # PSP -> PSC conversion depends on the *target* neuron's membrane parameters
    tau_m_pop, C_m_pop, V_th_pop, E_L_pop, V_reset_pop = params.neuron_pop_arrays()
    coba = params.synapse_model == "coba"
    if coba:
        # separate shape factors for the two conductance time constants, and
        # amplitude -> conductance at the measurement's own driving force
        te_pop, ti_pop = params.syn_tau_arrays()
        conv_e = np.array([psc_over_psp(C_m_pop[t], tau_m_pop[t], te_pop[t]) for t in range(N_POP)])
        conv_i = np.array([psc_over_psp(C_m_pop[t], tau_m_pop[t], ti_pop[t]) for t in range(N_POP)])
        drive_e = params.E_exc_mV - params.V_hold_exc_mV          # > 0
        drive_i = params.E_inh_mV - params.V_hold_inh_mV          # < 0
        conv = np.where(IS_EXC[None, :], conv_e[:, None] / drive_e, conv_i[:, None] / drive_i)  # pA/mV = nS per mV of PSP
        conv_pop = conv_e / drive_e
        comp = None
        if params.two_compartment:
            from .dendrite import compartments, peak_response, lands_on_dendrite
            from .params import pop_type
            A_pop = (np.asarray(params.dend_attenuation_pop, float) if params.dend_attenuation_pop is not None
                     else np.full(N_POP, 0.5))
            comp = {k: np.zeros(N_POP) for k in ("C_s", "C_d", "gL_s", "gL_d", "g_c")}
            for t in range(N_POP):
                if pop_type(POPS[t]) == "E":
                    cs, cd, gls, gld, gc = compartments(C_m_pop[t], tau_m_pop[t], params.dend_cap_frac, A_pop[t])
                else:
                    cs, cd, gls, gld, gc = C_m_pop[t], C_m_pop[t], C_m_pop[t] / tau_m_pop[t], C_m_pop[t] / tau_m_pop[t], 0.0
                for k, v in zip(("C_s", "C_d", "gL_s", "gL_d", "g_c"), (cs, cd, gls, gld, gc)):
                    comp[k][t] = v
                if gc == 0:
                    continue
                cells = (cs, cd, gls, gld, gc)
                pk = {}
                for where in ("s", "d"):
                    for name, tau in (("e", te_pop[t]), ("i", ti_pop[t])):
                        pk[(where, name)] = peak_response(*cells, tau, where)[0]
                for s_ in range(N_POP):
                    where = "d" if lands_on_dendrite("E", pop_type(POPS[s_])) else "s"
                    if IS_EXC[s_]:
                        conv[t, s_] = (1.0 / pk[(where, "e")]) / drive_e
                    else:
                        conv[t, s_] = (1.0 / pk[(where, "i")]) / drive_i
                conv_pop[t] = (1.0 / pk[("d", "e")]) / drive_e       # external input lands on the dendrite
    else:
        conv_pop = np.array([psc_over_psp(C_m_pop[t], tau_m_pop[t], nrn.tau_syn) for t in range(N_POP)])
        conv = conv_pop[:, None]
    PSC_mean = params.PSP_matrix_mean * conv / np.sqrt(K_scaling)   # pA (cuba) or nS (coba), >0 for both signs in coba
    PSC_ext = params.PSP_exc_mean * conv_pop            # per target population
    PSC_ext_scaled = PSC_ext / np.sqrt(K_scaling)
    delay_mean = params.delay_matrix_mean

    pre_l, post_l, w_l, d_l = [], [], [], []
    U_l, trec_l, tfac_l = [], [], []
    any_stp = params.stp is not None
    for t in range(N_POP):
        for s in range(N_POP):
            m = n_syn[t, s]
            if m == 0:
                continue
            if any_stp:
                sp = params.stp_for(POPS[t], POPS[s])
                if sp is None:                       # static: U=1, no recovery needed
                    U_l.append(np.ones(m, np.float32)); trec_l.append(np.full(m, 1e-3, np.float32)); tfac_l.append(np.full(m, 1e-3, np.float32))
                else:
                    U_l.append(np.full(m, sp["U"], np.float32))
                    trec_l.append(np.full(m, max(sp.get("tau_rec_ms", 1e-3), 1e-3), np.float32))
                    tfac_l.append(np.full(m, max(sp.get("tau_fac_ms", 1e-3), 1e-3), np.float32))
            # targets drawn uniformly (multapses allowed, as in PD2014)
            tgt_local = rng.integers(0, n_per_pop[t], m)
            if spatial:
                counts = np.bincount(tgt_local, minlength=n_per_pop[t])
                order = np.argsort(tgt_local, kind="stable")
                tgt_local = tgt_local[order]
                src_local = _sample_sources_spatial(
                    pos[offsets[t]:offsets[t] + n_per_pop[t], :2],
                    pos[offsets[s]:offsets[s] + n_per_pop[s], :2],
                    counts, params.lateral_sigma_um, rng)
            else:
                src_local = rng.integers(0, n_per_pop[s], m)
            mu = PSC_mean[t, s]
            if coba:
                mu = abs(mu)          # conductances are positive; the reversal potential carries the sign
            rel = params.weight_rel_std_for(POPS[t], POPS[s])
            if rel > 0.5:
                # broad, skewed amplitude distributions (as measured in slices): lognormal with the same mean and CV
                sig2 = np.log(1.0 + rel ** 2)
                w = np.sign(mu) * rng.lognormal(np.log(abs(mu)) - 0.5 * sig2, np.sqrt(sig2), m)
            else:
                w = rng.normal(mu, abs(mu) * rel, m)
                w = np.maximum(w, 0) if mu > 0 else np.minimum(w, 0)
            dmu = delay_mean[t, s]
            d = rng.normal(dmu, dmu * params.delay_rel_std, m)
            d = np.maximum(d, dt_ms)
            pre_l.append(src_local + offsets[s])
            post_l.append(tgt_local + offsets[t])
            w_l.append(w)
            d_l.append(d)

    syn_on_dend = None
    if coba and params.two_compartment:
        from .dendrite import lands_on_dendrite
        from .params import pop_type
        tps = [pop_type(q) for q in POPS]
        on_pop = np.array([[lands_on_dendrite(tps[t], tps[s_]) for s_ in range(N_POP)] for t in range(N_POP)])
        syn_on_dend_l = []
        for t in range(N_POP):
            for s_ in range(N_POP):
                m = n_syn[t, s_]
                if m == 0:
                    continue
                d = np.full(m, on_pop[t, s_])
                if tps[t] == "E" and tps[s_] == "F" and params.fs_soma_fraction < 1.0:
                    d = rng.random(m) >= params.fs_soma_fraction
                syn_on_dend_l.append(d)
        syn_on_dend = np.concatenate(syn_on_dend_l)
        # weights of F->E synapses moved to the dendrite: rescale so the somatic PSP still matches the data
        if params.fs_soma_fraction < 1.0:
            w_all = np.concatenate(w_l)
            post_all = np.concatenate(post_l); pre_all = np.concatenate(pre_l)
            tp = pop_of[post_all]; sp = pop_of[pre_all]
            for t in range(N_POP):
                if tps[t] != "E" or comp["g_c"][t] == 0:
                    continue
                cells = tuple(comp[k][t] for k in ("C_s", "C_d", "gL_s", "gL_d", "g_c"))
                ratio = peak_response(*cells, ti_pop[t], "s")[0] / peak_response(*cells, ti_pop[t], "d")[0]
                sel = (tp == t) & np.isin(sp, [i for i, q in enumerate(tps) if q == "F"]) & syn_on_dend
                w_all[sel] *= ratio
            w_l = [w_all]
    pre = np.concatenate(pre_l).astype(np.int32)
    post = np.concatenate(post_l).astype(np.int32)
    weight = np.concatenate(w_l).astype(np.float32)
    delay = np.concatenate(d_l).astype(np.float32)
    if any_stp:
        stp_U, stp_rec, stp_fac = (np.concatenate(U_l), np.concatenate(trec_l), np.concatenate(tfac_l))
        # with STP the effective first-spike weight is w*U; rescale so that the
        # *resting-state* PSP matches the specified amplitude (as measured in slices)
        weight = (weight / stp_U).astype(np.float32)
    else:
        stp_U = stp_rec = stp_fac = None

    # ---- external drive + DC compensation for the K scaling ----------------
    K_ext = params.K_ext * K_scaling
    DC = np.zeros(N_POP)
    if coba:
        # conductance drive: mean current at rest = g * (E_syn - E_L) * rate * tau
        drive_rest_e = params.E_exc_mV - E_L_pop
        drive_rest_i = params.E_inh_mV - E_L_pop
        g_rec = np.abs(params.PSP_matrix_mean * conv) * indeg_full
        if params.full_mean_rates is not None:
            cur_e = np.sum(g_rec[:, IS_EXC] * params.full_mean_rates[IS_EXC], axis=1) * te_pop * drive_rest_e
            cur_i = np.sum(g_rec[:, ~IS_EXC] * params.full_mean_rates[~IS_EXC], axis=1) * ti_pop * drive_rest_i
            input_rec = (cur_e + cur_i) * 0.001
        else:
            input_rec = 0.0
        input_ext = PSC_ext * params.K_ext * params.bg_rate * te_pop * drive_rest_e * 0.001
        DC = (1.0 - np.sqrt(K_scaling)) * (input_rec + input_ext)
        if comp is not None:
            # DC is injected at the soma; external drive it stands in for is dendritic
            A_ss = np.where(comp["g_c"] > 0, comp["g_c"] / (comp["g_c"] + comp["gL_d"] + 1e-12), 1.0)
            DC = DC * A_ss
    elif params.full_mean_rates is not None:
        input_rec = np.sum(params.PSP_matrix_mean * conv * indeg_full
                           * params.full_mean_rates, axis=1)
        input_ext = PSC_ext * params.K_ext * params.bg_rate
        DC = 0.001 * nrn.tau_syn * (1.0 - np.sqrt(K_scaling)) * (input_rec + input_ext)
    else:
        # no reference rates: compensate external drive only (rough)
        input_ext = PSC_ext * params.K_ext * params.bg_rate
        DC = 0.001 * nrn.tau_syn * (1.0 - np.sqrt(K_scaling)) * input_ext
    neuron_dict = asdict(nrn) | {"V0_mean": nrn.V0_mean.tolist(), "V0_std": nrn.V0_std.tolist(),
                                 "tau_m_pop": tau_m_pop.tolist(), "C_m_pop": C_m_pop.tolist(),
                                 "V_th_pop": V_th_pop.tolist(), "E_L_pop": E_L_pop.tolist(),
                                 "V_reset_pop": V_reset_pop.tolist(), "w_ext_pop": PSC_ext_scaled.tolist(),
                                 "adapt_b_pop": params.adapt_b_array().tolist(),
                                 "adapt_tau_ms": params.adapt_tau_ms,
                                 "synapse_model": params.synapse_model, "E_exc_mV": params.E_exc_mV,
                                 "E_inh_mV": params.E_inh_mV, "tau_exc_ms": params.tau_exc_ms,
                                 "tau_inh_ms": params.tau_inh_ms,
                                 "bg_ratio": getattr(params, "bg_ratio", None),
                                 "two_compartment": bool(coba and params.two_compartment),
                                 "comp": ({k: v.tolist() for k, v in comp.items()} if (coba and params.two_compartment) else None),
                                 "tau_exc_pop": params.syn_tau_arrays()[0].tolist(),
                                 "tau_inh_pop": params.syn_tau_arrays()[1].tolist()}

    return SynthesizedColumn(
        name=params.name, N_scaling=N_scaling, K_scaling=K_scaling,
        n_per_pop=n_per_pop, pop_of=pop_of, pos_um=pos.astype(np.float32),
        pre=pre, post=post, weight_pA=weight, delay_ms=delay,
        stp_U=stp_U, stp_tau_rec_ms=stp_rec, stp_tau_fac_ms=stp_fac,
        K_ext=K_ext, w_ext_pA=float(np.mean(PSC_ext_scaled)), bg_rate=params.bg_rate,
        DC_pA=DC, neuron=neuron_dict,
        populations=None if POPS == POPULATIONS else POPS,
        syn_on_dend=syn_on_dend,
    )
