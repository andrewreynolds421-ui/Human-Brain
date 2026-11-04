#!/usr/bin/env python3
"""Reward learning between two areas (46d -> 9/46d) of the multi-area model.

Task: each trial one of two cues (half A / half B of 46d's L2/3 pyramids,
200 ms of extra input) is shown; 9/46d's L5 pyramids are split into two
action pools, and the pool with more spikes in [cue+20, cue+220] ms is the
choice.  Correct (A -> pool 1, B -> pool 2) gives reward +1, wrong -1.

Learning: three-factor rule on the excitatory synapses from 46d onto 9/46d's
L5 pyramids (optionally also 9/46d's local L2/3 -> L5): pre/post pairings
build an eligibility trace; after each trial
    w += eta * (R - Rbar) * elig * mean(w0),   clipped to [0, w_max_factor * w0]
with Rbar the running mean reward (time constant ``baseline_trials``).
Optional inhibitory plasticity (Vogels rule, ``istdp_eta``) keeps rates
near target.

The measured network is fixed; only learning-rule settings (the "genome")
vary.  ``run(genome, trials, seed)`` returns the learning curve and rates.

    python scripts/reward_learning.py --trials 200
"""
import argparse, json, os, sys, time, copy
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import brian2 as b2                                                   # noqa: E402
from brian2 import ms, Hz, nS                                         # noqa: E402
from synth import PRESETS                                             # noqa: E402
from synth.areas import build_areas, add_interareal, load_interareal  # noqa: E402
from synth.network import build_network                               # noqa: E402

DEFAULT_GENOME = dict(eta=0.05, tau_elig_ms=500.0, a_minus=1.05, baseline_trials=20.0,
                      w_max_factor=5.0, istdp_eta=0.0, local_plastic=0, cue_fibres=1000,
                      gate=0, normalize=0, wta=0.0, wta_in_mV=3.0,
                      sp=0, sp_pool=1.0, sp_prune=0.15, sp_new=0.5, sp_every=10,
                      surprise_gain=0.0, surprise_slow_trials=50.0, uncert_gain=0.0, wta_ref_pool=0)
# wta_ref_pool > 0: selection input per spike scaled by wta_ref_pool / (pool size), so the
#       selection circuit gets the same total drive whatever the network size (0 = off)
# uncert_gain > 0: expected uncertainty (Yu & Dayan 2005; attributed to
#       acetylcholine).  While the network is still unsure how to earn reward the
#       learning rate is raised, and it falls as reward comes in:
#       lr x (1 + uncert_gain * (1 - max(0, slow mean reward))).  Slow mean
#       reward starts at 0 (chance) and is 1 when every trial is rewarded.
# surprise_gain > 0: unexpected uncertainty (Yu & Dayan 2005; attributed to
#       noradrenaline).  surprise = max(0, slow mean reward - fast mean reward),
#       slow over surprise_slow_trials, fast = the reward baseline; the
#       learning rate is multiplied by (1 + surprise_gain * surprise).
# sp=1: structural plasticity (synth/structural.py): a pool of sp_pool x the
#       plastic synapses as absent potential sites; synapses below sp_prune x
#       their starting weight are removed and the same number form at random
#       absent sites with sp_new x starting weight.
# wta > 0: action selection.  Two selection groups (40 LIF cells each, a
#         stand-in for basal-ganglia action selection, not measured): action
#         pool k excites selection group k (p 0.3, wta_in_mV per spike); group k
#         inhibits the *other* pool's somata (p 0.5, wta nS per spike).
# gate=1: attention-gated reinforcement (Roelfsema & van Ooyen 2005): only
#         synapses onto the chosen action pool are updated (feedback from the
#         selected action tags them).  A modelling assumption about circuitry
#         outside the two modelled areas.
# normalize=1: each postsynaptic cell's total plastic input weight is held at
#         its initial value after every update (synaptic scaling), so learning
#         redistributes strength instead of inflating it.
CUE_MS, TRIAL_MS, SETTLE_MS = 200.0, 500.0, 500.0
ISTDP_TARGET = {"L23E": 1.6, "L4E": 5.8, "L5E": 4.8, "L6E": 3.2}


def build(seed=1, n_scale=0.015, areas=("46d", "9/46d"), cache={}):
    key = (seed, n_scale, tuple(areas))
    if key not in cache:
        data = load_interareal()
        areas = list(areas)
        sub = dict(data, areas=areas)
        p = PRESETS["human_h01_dendritic"](verbose=False, ext_gain=2.0)
        col = build_areas(p, areas, n_scale=n_scale, seed=seed)
        col = add_interareal(col, sub, seed=seed, verbose=False)
        cache[key] = col
    return cache[key]


def run(genome, trials=200, seed=1, n_scale=0.015, verbose=False, reverse_at=None,
        areas=("46d", "9/46d"), cue_area="46d", action_area="9/46d", fast=False, partial_out=None):
    """reverse_at: trial index from which the rewarded mapping flips (A -> 2, B -> 1).
    fast: merged Poisson inputs + per-neuron STP/iSTDP state (same dynamics, ~2x faster)"""
    g = dict(DEFAULT_GENOME, **genome)
    col = build(seed, n_scale, areas)
    pops = col.pops
    src_area, dst_area = col.areas.index(cue_area), col.areas.index(action_area)
    pre_a, post_a = col.area_of[col.pre], col.area_of[col.post]
    pre_p, post_p = col.pop_of[col.pre], col.pop_of[col.post]
    E = col.is_exc
    l5 = pops.index("L5E"); l23 = pops.index("L23E")
    mask = E[pre_p] & (pre_a != dst_area) & (post_a == dst_area) & (post_p == l5)
    if g["local_plastic"]:
        mask |= (pre_p == l23) & (pre_a == dst_area) & (post_a == dst_area) & (post_p == l5)
    w_init_all = np.asarray(col.weight_pA, float)
    if g["sp"]:
        from synth.structural import add_potential_synapses, StructuralPlasticity
        col, _potential, w_init_all, mask = add_potential_synapses(col, mask, g["sp_pool"], seed)
        sp_rule = StructuralPlasticity(g["sp_prune"], g["sp_new"], seed)
    b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms
    istdp = dict(eta=g["istdp_eta"], tau_ms=20.0, target_hz=ISTDP_TARGET) if g["istdp_eta"] > 0 else None
    nw = build_network(col, seed=seed, bg_ratio=2.0, rstdp=dict(mask=mask, tau_elig_ms=g["tau_elig_ms"], a_minus=g["a_minus"]),
                       istdp=istdp, merged_inputs=fast, shared_stp=fast)
    G, net, spk = nw["G"], nw["net"], nw["spikes"]
    # free wiring tables the simulator now holds (weights, delays, STP) and return build scratch
    for k_ in ("weight_pA", "delay_ms", "stp_U", "stp_tau_rec_ms", "stp_tau_fac_ms", "syn_on_dend"):
        if hasattr(col, k_):
            setattr(col, k_, None)
    from synth.learner import _trim_memory
    _trim_memory()
    rgroups = nw["reward_groups"]
    w0 = [w_init_all[idx].astype(float) for _, idx in rgroups]               # starting (or template) weights
    present = [np.asarray(S_.w / nS) > 0 for S_, _ in rgroups]
    w_scale = float(np.mean(np.concatenate([w[p_] for w, p_ in zip(w0, present)])))
    w_max = [g["w_max_factor"] * w + 1e-9 for w in w0]
    sp_log = []
    snaps = []

    def snapshot(trial_no):
        """weight selectivity (+ = original mapping) and, with turnover, share of
        present synapses that follow the original mapping"""
        sel_w, cnt = 0.0, dict(A1=0, A2=0, B1=0, B2=0)
        for gi, (S_, idx) in enumerate(rgroups):
            w = np.asarray(S_.w / nS); pre = col.pre[idx]; post = col.post[idx]
            a_pre = np.isin(pre, cueA); b_pre = np.isin(pre, cueB); p1 = in1[post]; p2 = in2[post]
            m = lambda x: float(w[x].mean()) if x.any() else 0.0
            sel_w += (m(a_pre & p1) - m(a_pre & p2)) + (m(b_pre & p2) - m(b_pre & p1))
            pr = present[gi]
            for key, sel_ in (("A1", a_pre & p1), ("A2", a_pre & p2), ("B1", b_pre & p1), ("B2", b_pre & p2)):
                cnt[key] += int((sel_ & pr).sum())
        orig = cnt["A1"] + cnt["B2"]; other = cnt["A2"] + cnt["B1"]
        return dict(trial=trial_no, selectivity=round(sel_w / w_scale, 3),
                    structure_original_share=round(orig / max(orig + other, 1), 4) if g["sp"] else None,
                    accuracy_last50=round(float(np.mean(curve[-50:])), 3))
    # cues: halves of 46d L2/3 pyramids; action pools: halves of 9/46d L5 pyramids
    cue_cells = np.flatnonzero((col.area_of == src_area) & (col.pop_of == l23))
    rng = np.random.default_rng(seed + 31)
    perm = rng.permutation(cue_cells); cueA, cueB = perm[: perm.size // 2], perm[perm.size // 2:]
    act = np.flatnonzero((col.area_of == dst_area) & (col.pop_of == l5))
    perm = rng.permutation(act); pool1, pool2 = np.sort(perm[: perm.size // 2]), np.sort(perm[perm.size // 2:])
    w_cue = float(np.asarray(col.neuron["w_ext_pop"])[l23])
    if g["wta"] > 0:
        sel_groups = []
        for k_, (own, other) in enumerate(((pool1, pool2), (pool2, pool1))):
            Sg = b2.NeuronGroup(40, "dv/dt = (-(v - -56*mV)) / (10*ms) + sigma_s * sqrt(2 / (10*ms)) * xi : volt",
                                threshold="v > -50*mV", reset="v = -60*mV", refractory=2 * ms, method="euler",
                                namespace=dict(sigma_s=1.5 * b2.mV), name=f"select{k_}")
            Sg.v = -56 * b2.mV
            keep = rng.random(own.size * 40) < 0.3
            w_in = float(g['wta_in_mV']) * (g['wta_ref_pool'] / own.size if g.get('wta_ref_pool') else 1.0)
            ein = b2.Synapses(G, Sg, on_pre=f"v_post += {w_in}*mV", name=f"sel_in{k_}b")
            ii = np.repeat(own, 40)[keep]; jj = np.tile(np.arange(40), own.size)[keep]
            ein.connect(i=ii, j=jj); ein.delay = 1 * ms
            iout = b2.Synapses(Sg, G, model="w : siemens", on_pre="g_i_post += w", name=f"sel_out{k_}")
            pre_i = np.repeat(np.arange(40), other.size); post_j = np.tile(other, 40)
            keep2 = rng.random(pre_i.size) < 0.5
            iout.connect(i=pre_i[keep2], j=post_j[keep2]); iout.w = float(g["wta"]) * nS; iout.delay = 1 * ms
            sel_groups += [Sg, ein, iout]
        net.add(*sel_groups)
    stim = {}
    for name, cells in (("A", cueA), ("B", cueB)):
        pg = b2.PoissonGroup(2000, rates=0 * Hz)
        S = b2.Synapses(pg, G, model="w : siemens", on_pre="g_ed_post += w")
        S.connect(i=rng.integers(0, 2000, cells.size * int(g["cue_fibres"])), j=np.repeat(cells, int(g["cue_fibres"])))
        S.w = w_cue * nS
        net.add(pg, S); stim[name] = pg
    net.run(SETTLE_MS * ms)
    in1 = np.zeros(col.N, bool); in1[pool1] = True
    curve = []
    in2 = np.zeros(col.N, bool); in2[pool2] = True
    rbar, choices, wsum, dominance, counts = 0.0, [], [], [], []
    rslow, surprises = 0.0, []
    cue_seq = rng.integers(0, 2, trials)
    t_now = SETTLE_MS
    for k in range(trials):
        cue = "AB"[cue_seq[k]]
        n0 = spk.num_spikes
        stim[cue].rates = 30 * Hz
        net.run(CUE_MS * ms)
        stim[cue].rates = 0 * Hz
        net.run((TRIAL_MS - CUE_MS) * ms)
        t_spk = np.asarray(spk.t[n0:] / ms); i_spk = np.asarray(spk.i[n0:])
        win = (t_spk >= t_now + 20) & (t_spk < t_now + CUE_MS + 20)
        c1, c2 = in1[i_spk[win]].sum(), in2[i_spk[win]].sum()
        choice = 1 if c1 > c2 else 2 if c2 > c1 else int(rng.integers(1, 3))
        dominance.append(abs(int(c1) - int(c2)) / max(int(c1) + int(c2), 1))
        cue_cells_now = cueA if cue == "A" else cueB
        counts.append((int(c1), int(c2), int(np.isin(i_spk[win], cue_cells_now).sum())))
        reversed_now = reverse_at is not None and k >= reverse_at
        correct = ((choice == 1) == (cue == "A")) != reversed_now
        R = 1.0 if correct else -1.0
        t_now += TRIAL_MS
        # reward-gated update
        chosen = in1 if choice == 1 else in2
        surprise = max(0.0, rslow - rbar) if g["surprise_gain"] > 0 else 0.0
        lr_gain = (1.0 + g["surprise_gain"] * surprise) * (1.0 + g["uncert_gain"] * (1.0 - max(0.0, rslow)))
        surprises.append(round(surprise, 3))
        n_pr = n_fo = 0
        for gi, ((S_, idx), wm, w_init) in enumerate(zip(rgroups, w_max, w0)):
            el = np.asarray(S_.elig) * np.exp(-(t_now - np.asarray(S_.t_e / ms)) / g["tau_elig_ms"])
            if g["gate"]:
                el = el * chosen[col.post[idx]]
            w = np.clip(np.asarray(S_.w / nS) + g["eta"] * lr_gain * (R - rbar) * el * w_scale, 0, wm)
            if g["normalize"]:
                post = col.post[idx]
                tot0 = np.bincount(post, weights=w_init, minlength=col.N)
                tot = np.bincount(post, weights=w, minlength=col.N)
                f = np.where(tot > 0, tot0 / np.maximum(tot, 1e-12), 1.0)
                w = np.clip(w * f[post], 0, wm)
            if g["sp"]:
                w = np.where(present[gi], w, 0.0)                  # absent sites carry no weight
            if g["sp"] and (k + 1) % int(g["sp_every"]) == 0:      # pruning reflects sustained weakness
                a_, b_, new = sp_rule.update(w, w_init, present[gi])
                n_pr += a_; n_fo += b_
                if new.size:
                    el_all = np.asarray(S_.elig); el_all[new] = 0.0; S_.elig = el_all
            S_.w = w * nS
        rbar += (R - rbar) / g["baseline_trials"]
        rslow += (R - rslow) / g["surprise_slow_trials"]
        curve.append(int(correct)); choices.append(choice)
        if g["sp"]:
            sp_log.append((n_pr, n_fo))
        if k % 50 == 49:
            snaps.append(snapshot(k + 1))
        if k % 20 == 19:
            wsum.append(float(np.mean(np.concatenate([np.asarray(S_.w / nS) for S_, _ in rgroups])) / w_scale))
            if partial_out:                     # a run killed by a restart still leaves its curve so far
                json.dump(dict(partial=True, trials_done=k + 1, curve=curve, weight_trend=wsum, seed=seed,
                               genome=g), open(partial_out, "w"))
            if verbose:
                print(f"trial {k + 1:4d}  accuracy (last 20) {np.mean(curve[-20:]):.2f}  mean plastic weight x{wsum[-1]:.2f}", flush=True)
    # rates over the last 10 trials' inter-trial intervals
    t = np.asarray(spk.t / ms); i = np.asarray(spk.i)
    rest = np.zeros(t.size, bool)
    for k in range(max(0, trials - 10), trials):
        on = SETTLE_MS + k * TRIAL_MS
        rest |= (t >= on + CUE_MS + 50) & (t < on + TRIAL_MS)
    dur = 10 * (TRIAL_MS - CUE_MS - 50) / 1000.0
    rates = {}
    for a_i, a_n in enumerate(col.areas):
        for k, q in enumerate(pops):
            cells = (col.area_of == a_i) & (col.pop_of == k)
            rates[f"{a_n}:{q}"] = round(float(np.isin(i[rest], np.flatnonzero(cells)).sum() / cells.sum() / dur), 2)
    # selectivity learned: weight from cue-A cells onto pool 1 minus onto pool 2 (and B)
    sel = 0.0
    try:
        for (S_, idx) in rgroups:
            w = np.asarray(S_.w / nS); pre = col.pre[idx]; post = col.post[idx]
            a_pre = np.isin(pre, cueA); b_pre = np.isin(pre, cueB)
            p1 = in1[post]; p2 = in2[post]
            def m(x): return float(w[x].mean()) if x.any() else 0.0
            sel += (m(a_pre & p1) - m(a_pre & p2)) + (m(b_pre & p2) - m(b_pre & p1))
        sel /= w_scale
    except Exception:
        pass
    # structural selectivity: share of present synapses that follow the rewarded mapping
    struct = None
    if g["sp"]:
        cnt = dict(A1=0, A2=0, B1=0, B2=0)
        for (S_, idx), p_ in zip(rgroups, present):
            pre = col.pre[idx][p_]; post = col.post[idx][p_]
            a_pre = np.isin(pre, cueA); b_pre = np.isin(pre, cueB)
            cnt["A1"] += int((a_pre & in1[post]).sum()); cnt["A2"] += int((a_pre & in2[post]).sum())
            cnt["B1"] += int((b_pre & in1[post]).sum()); cnt["B2"] += int((b_pre & in2[post]).sum())
        match = cnt["A1"] + cnt["B2"]; cross = cnt["A2"] + cnt["B1"]
        struct = dict(counts=cnt, matching_share=round(match / max(match + cross, 1), 4),
                      pruned=sp_rule.pruned, formed=sp_rule.formed,
                      present=int(sum(p_.sum() for p_ in present)),
                      turnover_by_20=[int(sum(a for a, _ in sp_log[i:i + 20])) for i in range(0, len(sp_log), 20)])
    return dict(genome=g, trials=trials, seed=seed, curve=curve, structural=struct, snapshots=snaps, reverse_at=reverse_at,
                surprise=surprises,
                acc_first=float(np.mean(curve[:40])), acc_last=float(np.mean(curve[-40:])),
                weight_trend=wsum, selectivity=round(sel, 4), rates=rates,
                dominance=round(float(np.mean(dominance)), 3) if dominance else None,
                counts=counts, cues="".join("AB"[c] for c in cue_seq), choices=[int(c) for c in choices],
                n_plastic=int(sum(idx.size for _, idx in rgroups)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--genome", default="{}")
    ap.add_argument("--reverse-at", type=int, default=None, help="flip the rewarded mapping from this trial")
    ap.add_argument("--areas", default="46d,9/46d", help="comma list, or 'all' for the eight areas")
    ap.add_argument("--cue-area", default="46d")
    ap.add_argument("--action-area", default="9/46d")
    ap.add_argument("--n-scale", type=float, default=0.015)
    ap.add_argument("--fast", action="store_true", help="merged inputs + per-neuron synaptic state (README: Speed)")
    ap.add_argument("--out", default=None, help="also write the JSON result here")
    a = ap.parse_args()
    t0 = time.time()
    areas = load_interareal()["areas"] if a.areas == "all" else a.areas.split(",")
    r = run(json.loads(a.genome), a.trials, a.seed, n_scale=a.n_scale, verbose=True, reverse_at=a.reverse_at,
            areas=areas, cue_area=a.cue_area, action_area=a.action_area, fast=a.fast,
            partial_out=(a.out + ".partial.json") if a.out else None)
    r["wall_s"] = round(time.time() - t0)
    r["fast"] = bool(a.fast)
    print(json.dumps(r))
    if a.out:
        json.dump(r, open(a.out, "w"))
        if os.path.exists(a.out + ".partial.json"):
            os.remove(a.out + ".partial.json")


if __name__ == "__main__":
    main()
