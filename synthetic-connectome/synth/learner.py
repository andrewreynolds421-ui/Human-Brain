"""A live two-area network that learns cue -> action mappings from reward.

``LearnerColumn`` extends ``LiveColumn`` with the task and learning rule of
stage 6b (scripts/reward_learning.py, validated: chance -> 73% over 300
trials on two new networks):

* areas 46d -> 9/46d (strongest measured inter-areal link)
* two cues = halves of 46d's L2/3 pyramids; two actions = halves of
  9/46d's L5 pyramids, made to compete by two selection groups (stand-in
  for basal-ganglia action selection)
* three-factor learning on the 46d -> 9/46d L5 synapses, updates gated to
  the chosen action, evolved settings (EVOLVED_GENOME)

Interaction:
* ``trial(cue)``   present cue "A" or "B" for 200 ms, read the choice (1/2)
                   from the 200 ms response window, run 300 ms more, and
                   tag the synapses: their eligibility at the end of the trial
                   is stored until a reward arrives (synaptic tagging), so the
                   reward may come any time later, as from a human or agent
* ``reward(v)``    v in [-1, 1]: if a trial is waiting, apply
                   w += eta (v - running mean) * tag, onto the chosen pool only;
                   otherwise fall back to the column's Hebbian reward
* ``step``, ``stimulate``, ``state``, ``reset`` as in LiveColumn
"""
from __future__ import annotations

import numpy as np
import os
import brian2 as b2
from brian2 import ms, Hz, nS

from .live import LiveColumn

EVOLVED_GENOME = dict(eta=6.55, tau_elig_ms=263.0, a_minus=0.73, baseline_trials=10.0, w_max_factor=30.0,
                      istdp_eta=0.0163, wta=8.0, wta_in_mV=3.0, cue_fibres=1000)
# faster learning (README Stage 13): higher base rate + expected-uncertainty boost
QUICK_SETTINGS = dict(eta=13.1, uncert_gain=2.0)
# action selection scaled to pool size (README Stage 15): selection input per spike x
# WTA_REF_POOL / pool size, so bigger networks keep the same competition
WTA_REF_POOL = 30.5
ISTDP_TARGET = {"L23E": 1.6, "L4E": 5.8, "L5E": 4.8, "L6E": 3.2}            # spontaneous rates of the v1 column
ISTDP_TARGET_BY_PRESET = {"human_h01_dendritic": ISTDP_TARGET,
                          "human_h01_dendritic_v2": {"L23E": 1.3, "L4E": 4.0, "L5E": 3.9, "L6E": 1.1}}   # results/rate_check_v2.json
CUE_MS, TRIAL_MS = 200.0, 500.0
PATTERN_FIBRES = 100
SENS_RELAY_HZ = 300.0


def build_learner_network(n_scale=0.015, seed=1, ext_gain=None, areas=None, preset="human_h01_dendritic"):
    """areas: list of area names (default the two-area learner); "all" = the eight areas.
    preset: "human_h01_dendritic" (stage 3d column, ext_gain 2.0) or "human_h01_dendritic_v2" (stage 25)."""
    from . import PRESETS
    from .areas import build_areas, add_interareal, load_interareal
    data = load_interareal()
    areas = data["areas"] if areas == "all" else list(areas or ["46d", "9/46d"])
    kw = dict(verbose=False, bg_ratio=2.0)
    if ext_gain is not None or preset == "human_h01_dendritic":
        kw["ext_gain"] = 2.0 if ext_gain is None else ext_gain
    p = PRESETS[preset](**kw)
    col = build_areas(p, areas, n_scale=n_scale, seed=seed)
    col = add_interareal(col, dict(data, areas=areas), seed=seed, verbose=False)
    col.name = "two_area_learner_46d_to_9-46d" if len(areas) == 2 else f"{len(areas)}_area_learner_46d_to_9-46d"
    col.preset = preset
    _trim_memory()                     # return wiring-build scratch (~0.6 GB for 8 areas) to the system
    return col


def _trim_memory():
    import gc, ctypes
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _plastic_mask(col, action_area="9/46d"):
    """every inter-areal excitatory synapse onto the action area's L5 pyramids
    (two areas: exactly the 46d -> 9/46d pathway)"""
    dst = col.areas.index(action_area)
    # 1-byte lookup tables: target cells first (rare), then the rest only where needed
    is_target = (np.asarray(col.area_of) == dst) & (np.asarray(col.pop_of) == col.pops.index("L5E"))
    exc_other = col.is_exc[col.pop_of] & (np.asarray(col.area_of) != dst)
    return is_target[col.post] & exc_other[col.pre]


class LearnerColumn(LiveColumn):
    """structural=True adds synapses that form and disappear (synth/structural.py):
    a pool of absent potential synapses equal in number to the plastic ones; every
    ``sp_every`` rewards, synapses below 15% of their starting weight are removed
    and as many form at random sites at 50% (README Stage 8)."""

    def __init__(self, col, genome=None, seed=1, structural=False, sp_every=10, surprise_gain=0.0,
                 surprise_slow_trials=50.0, fast=False, shared_stp=None, merged_inputs=None, uncert_gain=0.0, lean=False, n_actions=2, pattern_cues=False,
                 cue_frac=0.05, cue_gain=1.0, cue_code="union", cue_ms=CUE_MS, sensory=None, sens_gain=1.0, sens_lr=1.0,
                 selection="wta", bg_kw=None, da_kw=None, **kw):
        # selection: "wta" = the stand-in selection groups (genome wta / wta_in_mV); "bg" = the
        # basal-ganglia-thalamus loop built from Humphries et al. 2006 (synth/basal_ganglia.py, Stage 26),
        # one channel per action pool, options in bg_kw.  The choice is read from the pool spike
        # counts either way; with "bg" the SNr rates of the trial are reported too.
        if selection not in ("wta", "bg"):
            raise ValueError("selection must be 'wta' or 'bg'")
        self.selection, self.bg_kw = selection, dict(bg_kw or {})
        # da_kw: not None = reward prediction inside the loop (Stage 27b): a dopamine population and a
        # striosomal critic; the outcome given to reward() is delivered as an input event and the loop's
        # cortico-striatal plasticity is gated by the dopamine response, not by value - rbar
        self.da_kw = None if da_kw is None else dict(da_kw)
        if self.da_kw is not None and selection != "bg":
            raise ValueError("the dopamine system needs selection='bg'")
        self.bg = self.da = None
        self.sens_lr = float(sens_lr)     # learning-rate factor for the sensory -> action synapses
        # sensory: a trained synth.sensory.SensoryLayer (or path to one).  Text trials then go through it
        # and its cells' spikes are replayed onto the action pools through plastic synapses that learn
        # by the same reward rule (a sensory area feeding the action area; not part of the H01 model)
        if isinstance(sensory, str):
            from .sensory import SensoryLayer
            sensory = SensoryLayer.load(sensory)
        self.sensory, self.sens_gain = sensory, float(sens_gain)
        self.cue_ms = float(cue_ms)       # cue / decision window length (evidence accumulation time)
        self.cue_code = cue_code          # text code: "union" (each word its own cells) or "kwta" 
        self.cue_gain = float(cue_gain)   # pattern cues: drive per active cell x cue_gain
        # n_actions: action pools in 9/46d L5 (one selection group each, inhibiting all other pools).
        # pattern_cues: trials may also take text (synth/textcode.py) or an activity array over the
        # cue cells instead of the named cues "A"/"B".  Defaults reproduce the two-action learner.
        self.n_actions, self.pattern_cues, self.cue_frac = int(n_actions), bool(pattern_cues), float(cue_frac)
        # lean: same dynamics, less RAM — no reset snapshot (reset = rebuild from the seed), wiring
        # tables the running learner no longer reads are dropped, build scratch returned to the system
        self.lean = bool(lean)
        # uncert_gain: expected uncertainty — learning rate x (1 + uncert_gain * (1 - max(0, slow mean
        # reward))): high while rewards are at chance, easing off as the rule is learned
        self.uncert_gain = float(uncert_gain)
        # fast: merged Poisson inputs + per-neuron STP/iSTDP state (synth/network.py);
        # same dynamics, see README "Speed"
        self.fast = bool(fast)
        self._shared = self.fast if shared_stp is None else bool(shared_stp)
        self._merged = self.fast if merged_inputs is None else bool(merged_inputs)
        self.surprise_gain, self.surprise_slow = float(surprise_gain), float(surprise_slow_trials)
        self.genome = dict(EVOLVED_GENOME, **(genome or {}))
        self._seed = seed
        self.structural, self.sp_every = bool(structural), int(sp_every)
        mask = _plastic_mask(col)
        self._w_init_all = np.asarray(col.weight_pA, float)
        if self.structural:
            from .structural import add_potential_synapses, StructuralPlasticity
            orig = col
            col, _pot, self._w_init_all, mask = add_potential_synapses(col, mask, 1.0, seed)
            if self.lean:     # the caller's column object would otherwise keep a second full copy
                for k_ in ("pre", "post", "weight_pA", "delay_ms", "stp_U", "stp_tau_rec_ms", "stp_tau_fac_ms",
                           "syn_on_dend"):
                    if hasattr(orig, k_):
                        setattr(orig, k_, None)
                del orig
            _trim_memory()
            self.sp_rule = StructuralPlasticity(0.15, 0.5, seed)
        self._mask = mask
        super().__init__(col, seed=seed, **kw)
        if self.lean:
            self._release_build_memory()

    # ------------------------------------------------------------------ build
    def _build(self, seed, target, verbose):
        col, g = self.col, self.genome
        b2.start_scope()
        b2.prefs.codegen.target = target
        b2.defaultclock.dt = self.dt_ms * ms
        from .network import build_network
        pops = col.pops
        src, dst = col.areas.index("46d"), col.areas.index("9/46d")
        l5, l23 = pops.index("L5E"), pops.index("L23E")
        mask = self._mask
        nw = build_network(col, seed=seed, with_stimulus=True, merged_inputs=self._merged, shared_stp=self._shared,
                           rstdp=dict(mask=mask, tau_elig_ms=g["tau_elig_ms"], a_minus=g["a_minus"]),
                           istdp=dict(eta=g["istdp_eta"], tau_ms=20.0,
                                      target_hz=ISTDP_TARGET_BY_PRESET.get(getattr(col, "preset", None), ISTDP_TARGET))
                           if g["istdp_eta"] > 0 else None)
        self.G, self.spikes, self.net = nw["G"], nw["spikes"], nw["net"]
        self._synapses, self._syn_index = nw["synapses"], nw["syn_index"]
        self._stim_groups, self.w_ext_pop = nw["stim_groups"], nw["w_ext_pop"]
        self.coba, self._wunit = nw["coba"], nw["wunit"]
        self.rgroups = nw["reward_groups"]
        self.w0 = [self._w_init_all[idx].astype(float) for _, idx in self.rgroups]
        self.present = [np.asarray(S_.w / nS) > 0 for S_, _ in self.rgroups]
        self.w_scale = float(np.mean(np.concatenate([w[p_] for w, p_ in zip(self.w0, self.present)])))
        self.w_max = [g["w_max_factor"] * w + 1e-9 for w in self.w0]
        rng = np.random.default_rng(seed + 31)
        cue_cells = np.flatnonzero((col.area_of == src) & (col.pop_of == l23))
        perm = rng.permutation(cue_cells)
        self.cues = {"A": perm[: perm.size // 2], "B": perm[perm.size // 2:]}
        act = np.flatnonzero((col.area_of == dst) & (col.pop_of == l5))
        perm = rng.permutation(act)
        K = self.n_actions
        cuts = [round(k * perm.size / K) for k in range(K + 1)]
        self.pools = {k + 1: np.sort(perm[cuts[k]:cuts[k + 1]]) for k in range(K)}
        self.in_pool = {k: np.isin(np.arange(col.N), v) for k, v in self.pools.items()}
        w_cue = float(np.asarray(col.neuron["w_ext_pop"])[l23])
        self.cue_gen = {}
        extra = []
        for name, cells in self.cues.items():
            pg = b2.PoissonGroup(2000, rates=0 * Hz, name=f"cue_{name}")
            S = b2.Synapses(pg, self.G, model="w : siemens", on_pre="g_ed_post += w", name=f"cue_syn_{name}")
            S.connect(i=rng.integers(0, 2000, cells.size * int(g["cue_fibres"])), j=np.repeat(cells, int(g["cue_fibres"])))
            S.w = w_cue * nS
            extra += [pg, S]; self.cue_gen[name] = pg
        if self.selection == "bg":
            from .basal_ganglia import BasalGanglia
            w_thal = float(np.asarray(col.neuron["w_ext_pop"])[l5])   # the column's external (thalamic) synapse on L5E
            ctx = (self.G, np.sort(cue_cells)) if (self.bg_kw.get("plastic") or self.da_kw is not None) else None
            bkw = dict(self.bg_kw); thal_route = float(bkw.pop("thal_route", 0.0))
            self.bg = BasalGanglia([(self.G, self.pools[k_ + 1]) for k_ in range(K)], seed=seed + 7,
                                   target=(self.G, "g_ed", w_thal), context=ctx, **bkw)
            extra += self.bg.objects
            if thal_route > 0:                    # Stage 28: the relay carries a share of the pools' drive
                from .basal_ganglia import route_external_through_thalamus
                route_external_through_thalamus(self.G, [self.pools[k_ + 1] for k_ in range(K)], thal_route, self.bg, w_thal)
            if self.da_kw is not None:
                from .basal_ganglia import DopamineSystem
                self.da = DopamineSystem(self.bg, ctx, seed=seed + 7, **self.da_kw)
                extra += self.da.objects
        for k_ in (range(K) if self.selection == "wta" else ()):
            own = self.pools[k_ + 1]
            other = np.concatenate([self.pools[j + 1] for j in range(K) if j != k_])
            Sg = b2.NeuronGroup(40, "dv/dt = (-(v - -56*mV)) / (10*ms) + sigma_s * sqrt(2 / (10*ms)) * xi : volt",
                                threshold="v > -50*mV", reset="v = -60*mV", refractory=2 * ms, method="euler",
                                namespace=dict(sigma_s=1.5 * b2.mV), name=f"select{k_}")
            Sg.v = -56 * b2.mV
            keep = rng.random(own.size * 40) < 0.3
            w_in = float(g["wta_in_mV"]) * (g["wta_ref_pool"] / own.size if g.get("wta_ref_pool") else 1.0)
            ein = b2.Synapses(self.G, Sg, on_pre=f"v_post += {w_in}*mV", name=f"sel_in{k_}")
            ein.connect(i=np.repeat(own, 40)[keep], j=np.tile(np.arange(40), own.size)[keep]); ein.delay = 1 * ms
            pre_i = np.repeat(np.arange(40), other.size); post_j = np.tile(other, 40)
            keep2 = rng.random(pre_i.size) < 0.5
            iout = b2.Synapses(Sg, self.G, model="w : siemens", on_pre="g_i_post += w", name=f"sel_out{k_}")
            iout.connect(i=pre_i[keep2], j=post_j[keep2]); iout.w = float(g["wta"]) * nS; iout.delay = 1 * ms
            extra += [Sg, ein, iout]
        if self.pattern_cues:
            # one Poisson source per cue cell; an active cell gets the same mean drive as a named cue
            self.cue_cells_all = np.sort(cue_cells)
            # (a Poisson source fires at most once per time step, so the drive is spread over
            # PATTERN_FIBRES sources per cell, each well below 1 spike per step)
            nc = self.cue_cells_all.size
            self.pattern_gen = b2.PoissonGroup(nc * PATTERN_FIBRES, rates=0 * Hz, name="cue_pattern")
            Sp = b2.Synapses(self.pattern_gen, self.G, model="w : siemens", on_pre="g_ed_post += w", name="cue_syn_pattern")
            Sp.connect(i=np.arange(nc * PATTERN_FIBRES), j=np.repeat(self.cue_cells_all, PATTERN_FIBRES))
            Sp.w = w_cue * nS
            extra += [self.pattern_gen, Sp]
        if self.sensory is not None:
            ns_e = self.sensory.n_e
            # each sensory cell stands for a small population: while the cue lasts it drives its fibres
            # at SENS_RELAY_HZ x (its spike count / the top count) — a reading of which cells won
            self.sens_relay = b2.PoissonGroup(ns_e, rates=0 * Hz, name="sens_relay")
            pool_cells = np.concatenate([self.pools[k] for k in sorted(self.pools)])
            tau_r = 20.0 * ms
            Ss = b2.Synapses(self.sens_relay, self.G, model="""w : siemens
            elig : 1
            t_e : second
            dtr_a/dt = -tr_a / tau_rs : 1 (event-driven)
            dtr_b/dt = -tr_b / tau_rs : 1 (event-driven)""",
                             on_pre="""g_ed_post += w
            elig = elig * exp(-(t - t_e) / tau_el) - a_minus * tr_b
            t_e = t
            tr_a += 1""",
                             on_post="""elig = elig * exp(-(t - t_e) / tau_el) + tr_a
            t_e = t
            tr_b += 1""", name="sens_syn",
                             namespace=dict(tau_rs=tau_r, tau_el=float(g["tau_elig_ms"]) * ms, a_minus=float(g["a_minus"])))
            Ss.connect(i=np.repeat(np.arange(ns_e), pool_cells.size), j=np.tile(pool_cells, ns_e))
            self.s_w0 = np.full(len(Ss), self.w_scale * self.sens_gain)
            Ss.w = self.s_w0 * nS; Ss.elig = 0.0; Ss.t_e = 0 * ms
            Ss.delay = 2 * ms
            self.s_syn, self.s_post = Ss, np.tile(pool_cells, ns_e)
            self.s_wmax = g["w_max_factor"] * self.s_w0
            extra += [self.sens_relay, Ss]
        self.net.add(*extra)
        # lean: no reset snapshot (Brian2 assembles a full in-memory copy of the state to make one,
        # ~1-2 GB at double size); reset = rebuild from the same seed, which gives the same state
        self.reset_needs_rebuild = self.lean
        if not self.lean:
            self._store("init")
        # learner bookkeeping
        self.rbar = 0.0
        self.rslow = 0.0
        self.last_surprise = 0.0
        self.pending = None
        self.history = []                       # dicts: cue, choice, reward
        self.n_rewards = 0
        self.cum_weight_change = 0.0
        self.ee_idx = np.empty(0, int)          # LiveColumn's Hebbian reward is not used here
        self._warm = False

    def _clear_spikes(self):
        super()._clear_spikes()
        if self.bg is not None:
            self.bg.clear()
        if self.da is not None:
            self.da.clear()

    # ---------------------------------------------------------------- trials
    def trial(self, cue="A"):
        """cue: "A"/"B", or (pattern_cues) a text string or an activity array over the cue cells"""
        pattern, label = None, cue
        sens_spikes = None
        if self.sensory is not None and isinstance(cue, str) and cue not in self.cues:
            counts = self.sensory.features(cue)
            from .textcode import words as _w
            sens_spikes, label = counts, " ".join(list(dict.fromkeys(_w(cue)))[:8])[:60] or "(no content words)"
        if sens_spikes is not None:
            pass
        elif self.pattern_cues and not (isinstance(cue, str) and cue in self.cues):
            if isinstance(cue, str):
                from .textcode import encode
                pattern, kept = encode(cue, self.cue_cells_all.size, self.cue_frac, mode=self.cue_code)
                label = " ".join(kept)[:60] or "(no content words)"
            else:
                pattern, label = np.clip(np.asarray(cue, float), 0, 1), "pattern"
        else:
            cue = cue if cue in self.cues else "A"; label = cue
        if self.pending is not None:            # an unrewarded trial is replaced by the new one
            self.events.append((round(self.t_ms), "trial", "previous trial left unrewarded"))
        self._in_trial = True
        n0 = self.spikes.num_spikes
        t_start = float(self.net.t / ms)
        if sens_spikes is not None:
            c_ = np.asarray(sens_spikes, float)
            self.sens_relay.rates = (SENS_RELAY_HZ * c_ / max(c_.max(), 1.0)) * Hz
        elif pattern is None:
            self.cue_gen[cue].rates = 30 * Hz
        else:
            # named cue: each cue cell gets cue_fibres fibres x 30 Hz; same per active cell here
            self.pattern_gen.rates = (np.repeat(pattern, PATTERN_FIBRES) * 30.0 * int(self.genome["cue_fibres"])
                                      * self.cue_gain / PATTERN_FIBRES * Hz)
        if self.bg is not None:
            self.bg.set_homeostat(False)          # gated relay homeostasis: closed during cue and decision
        self.step(self.cue_ms)
        if sens_spikes is not None:
            self.sens_relay.rates = 0 * Hz
        elif pattern is None:
            self.cue_gen[cue].rates = 0 * Hz
        else:
            self.pattern_gen.rates = 0 * Hz
        self.step(100.0)                           # same 300 ms after the cue whatever its length
        if self.bg is not None:
            self.bg.set_homeostat(True)
        self.step(TRIAL_MS - CUE_MS - 100.0)
        t = np.asarray(self.spikes.t / ms)[n0:]; i = np.asarray(self.spikes.i)[n0:]
        snr = None
        if self.bg is not None:               # read the loop's output before the monitors are emptied
            snr = [round(float(x), 1) for x in self.bg.rates("SNr", t_start + 20, t_start + self.cue_ms + 20)]
            self.last_bg = dict(snr=snr, selected=self.bg.selected(t_start + 20, t_start + self.cue_ms + 20))
        self._in_trial = False
        self._clear_spikes()
        win = (t >= t_start + 20) & (t < t_start + self.cue_ms + 20)
        counts = [int(self.in_pool[k][i[win]].sum()) for k in range(1, self.n_actions + 1)]
        top = np.flatnonzero(np.array(counts) == max(counts)) + 1
        choice = int(top[0]) if top.size == 1 else int(np.random.default_rng().choice(top))
        cue = label
        c1, c2 = counts[0], counts[1]
        t_end = float(self.net.t / ms)
        tags = []
        for (S_, idx) in self.rgroups:          # synaptic tags: eligibility frozen at trial end
            el = np.asarray(S_.elig) * np.exp(-(t_end - np.asarray(S_.t_e / ms)) / self.genome["tau_elig_ms"])
            tags.append(el * self.in_pool[choice][self.col.post[idx]])
        s_tag = None
        if self.sensory is not None:
            el = np.asarray(self.s_syn.elig) * np.exp(-(t_end - np.asarray(self.s_syn.t_e / ms)) / self.genome["tau_elig_ms"])
            s_tag = el * self.in_pool[choice][self.s_post]
        self.pending = dict(cue=cue, choice=choice, counts=tuple(counts), tags=tags, s_tag=s_tag)
        self.events.append((round(self.t_ms), "trial", f"cue {cue} -> action {choice} (spikes {counts}"
                            + (f", SNr Hz {snr})" if snr is not None else ")")))
        out = dict(cue=cue, choice=choice, pool_spikes=counts)
        if snr is not None:
            out["snr_hz"] = snr
        return out

    def reward(self, value=1.0, note=""):
        value = float(np.clip(value, -1.0, 1.0))
        if self.pending is None:
            self.events.append((round(self.t_ms), "reward", f"{value:+.2f} ignored: no trial waiting {note}".strip()))
            return dict(value=value, applied=False, reason="no trial waiting for feedback",
                        eligible_synapses=0, active_neurons=0, relative_weight_change=0.0)
        g = self.genome
        before = after = 0.0
        # surprise (unexpected uncertainty): recent reward below the long-run average
        surprise = max(0.0, self.rslow - self.rbar) if self.surprise_gain > 0 else 0.0
        lr = g["eta"] * (1.0 + self.surprise_gain * surprise) * (1.0 + self.uncert_gain * (1.0 - max(0.0, self.rslow)))
        self.last_lr_gain = lr / g["eta"]
        do_sp = self.structural and (self.n_rewards + 1) % self.sp_every == 0
        n_pr = n_fo = 0
        for gi, ((S_, idx), wm, tag) in enumerate(zip(self.rgroups, self.w_max, self.pending["tags"])):
            w = np.asarray(S_.w / nS)
            new = np.clip(w + lr * (value - self.rbar) * tag * self.w_scale, 0, wm)
            if self.structural:
                new = np.where(self.present[gi], new, 0.0)
                if do_sp:
                    a_, b_, formed = self.sp_rule.update(new, self.w0[gi], self.present[gi])
                    n_pr += a_; n_fo += b_
                    if formed.size:
                        el = np.asarray(S_.elig); el[formed] = 0.0; S_.elig = el
            before += np.abs(w).sum(); after += np.abs(new - w).sum()
            S_.w = new * nS
        if self.sensory is not None and self.pending.get("s_tag") is not None:
            w = np.asarray(self.s_syn.w / nS)
            self.s_syn.w = np.clip(w + self.sens_lr * lr * (value - self.rbar) * self.pending["s_tag"] * self.w_scale,
                                   0, self.s_wmax) * nS
        if do_sp:
            self.events.append((round(self.t_ms), "rewiring", f"{n_pr} synapses removed, {n_fo} formed"))
        rpe = value - self.rbar
        self.last_dopamine = None
        if self.da is not None:
            # the outcome as an input event; the dopamine rate deviation over the next window is the
            # prediction error that changes the loop's synapses (and the critic's)
            t0 = float(self.net.t / ms)
            self._in_trial = True                 # keep the monitors until the window is read
            self.da.deliver(value); self.step(self.da.p["rew_ms"]); self.da.end_event()
            self.step(self.da.p["window_ms"] - self.da.p["rew_ms"])
            self._in_trial = False
            delta = self.da.delta(t0, t0 + self.da.p["window_ms"])
            self.last_dopamine = dict(da_hz=round(self.da.rate(t0, t0 + self.da.p["window_ms"]), 1), delta=round(delta, 3))
            self.bg.reward(delta)
            self.da.learn(delta)
            self._clear_spikes()
        elif self.bg is not None and self.bg.plastic:
            self.bg.reward(rpe)
        self.rbar += (value - self.rbar) / g["baseline_trials"]
        self.rslow += (value - self.rslow) / self.surprise_slow
        self.last_surprise = surprise
        change = float(after / max(before, 1e-12))
        self.cum_weight_change += change
        self.n_rewards += 1
        p = self.pending
        self.history.append(dict(cue=p["cue"], choice=p["choice"], reward=value))
        self.pending = None
        n_tagged = int(sum((t != 0).sum() for t in p["tags"]))
        self.events.append((round(self.t_ms), "reward", f"{value:+.2f} on trial cue {p['cue']} -> action {p['choice']} "
                                                        f"(prediction error {rpe:+.2f}) {note}".strip()))
        out = dict(value=value, applied=True, cue=p["cue"], choice=p["choice"], prediction_error=round(rpe, 3),
                   eligible_synapses=n_tagged, active_neurons=int(self.pools[p["choice"]].size),
                   relative_weight_change=change)
        if self.last_dopamine is not None:
            out["dopamine"] = self.last_dopamine
        return out

    def selectivity(self):
        """Mean learned weight cue A -> pool 1 and B -> pool 2, minus the crossed
        mapping, in units of the starting weight (0 = no preference)."""
        s = 0.0
        for (S_, idx) in self.rgroups:
            w = np.asarray(S_.w / nS); pre = self.col.pre[idx]; post = self.col.post[idx]
            a = np.isin(pre, self.cues["A"]); b = np.isin(pre, self.cues["B"])
            p1, p2 = self.in_pool[1][post], self.in_pool[2][post]
            m = lambda x: float(w[x].mean()) if x.any() else 0.0
            s += (m(a & p1) - m(a & p2)) + (m(b & p2) - m(b & p1))
        return s / self.w_scale

    def _release_build_memory(self):
        """lean mode: weights, delays and STP settings now live in the simulator; keep only the
        wiring indices the learner reads (pre/post) and give freed memory back to the system"""
        c = self.col
        for k in ("weight_pA", "delay_ms", "stp_U", "stp_tau_rec_ms", "stp_tau_fac_ms", "syn_on_dend"):
            if hasattr(c, k):
                setattr(c, k, None)
        self._w_init_all = None
        _trim_memory()

    def reset(self):
        if getattr(self, "reset_needs_rebuild", False):
            raise RuntimeError("lean learner: reset by rebuilding it (same seed gives the same start state)")
        super().reset()
        if self.lean:
            _trim_memory()
        self.rbar, self.rslow, self.pending, self.history = 0.0, 0.0, None, []

    def state(self, *a, **kw):
        st = super().state(*a, **kw)
        last = self.history[-20:]
        st["learner"] = dict(
            trials_rewarded=len(self.history),
            waiting_for_feedback=self.pending is not None,
            last_trial=(dict(cue=self.pending["cue"], choice=self.pending["choice"]) if self.pending else
                        (self.history[-1] if self.history else None)),
            mean_reward_last20=round(float(np.mean([h["reward"] for h in last])), 2) if last else None,
            reward_baseline=round(self.rbar, 3),
            surprise=round(self.last_surprise, 3),
            learning_rate_x=round(getattr(self, "last_lr_gain", 1.0), 2),
        )
        if self.structural:
            st["learner"]["synapses_present"] = int(sum(p_.sum() for p_ in self.present))
            st["learner"]["synapses_removed_total"] = self.sp_rule.pruned
            st["learner"]["synapses_formed_total"] = self.sp_rule.formed
        return st
