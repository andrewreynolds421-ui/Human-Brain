"""Tests for the simulation package.

Fast tests (seconds) run by default; the ones that build Brian2 networks are
marked ``slow`` (minutes, need a C compiler) and run with
``pytest -m slow`` or ``pytest -m ""``.

    python -m pytest tests -q            # fast only
    python -m pytest tests -q -m slow    # the network checks
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

REF = json.load(open(os.path.join(HERE, "reference", "presets.json")))


# ------------------------------------------------------------------ synthesis
@pytest.mark.parametrize("name", sorted(REF))
def test_preset_regression(name):
    """Every preset must synthesise exactly the column it did when the reference
    was recorded (same wiring, weights, delays and DC). Any change to
    synthesis that is meant to alter a column must regenerate the reference
    on purpose (tests/reference/presets.json) and say so in the README."""
    from synth import synthesize, PRESETS
    r = REF[name]
    c = synthesize(PRESETS[name](**r["kwargs"]), N_scaling=r["N_scaling"], K_scaling=r["K_scaling"], seed=r["seed"])
    assert c.N == r["N"] and c.S == r["S"]
    assert int(c.pre.sum()) == r["pre_sum"] and int(c.post.sum()) == r["post_sum"]
    assert float(c.weight_pA.astype(float).sum()) == pytest.approx(r["weight_sum"], rel=1e-9)
    assert float(c.delay_ms.astype(float).sum()) == pytest.approx(r["delay_sum"], rel=1e-9)
    assert np.allclose(c.DC_pA, r["DC"], atol=1e-5)


def test_provenance_is_reported():
    """The dendritic preset labels every calibrated / assumed value."""
    from synth import PRESETS
    p = PRESETS["human_h01_dendritic"](verbose=False, ext_gain=2.0, bg_ratio=2.0)
    prov = getattr(p, "provenance", {})
    assert prov, "preset has no provenance record"
    text = json.dumps(prov).lower()
    assert "calibrat" in text or "fitted" in text


# ---------------------------------------------------------- structural plasticity
def test_structural_plasticity_keeps_the_synapse_count():
    from synth.structural import StructuralPlasticity
    rng = np.random.default_rng(0)
    n = 2000
    w_init = rng.uniform(0.5, 2.0, n)
    present = rng.random(n) < 0.5
    w = np.where(present, w_init * rng.uniform(0.0, 1.5, n), 0.0)
    rule = StructuralPlasticity(prune_frac=0.15, new_frac=0.5, seed=1)
    n_before = int(present.sum())
    pruned, formed, new = rule.update(w, w_init, present)
    assert pruned == formed == new.size
    assert int(present.sum()) == n_before, "turnover must keep the number of present synapses"
    assert np.all(present[new])
    assert np.allclose(w[new], 0.5 * w_init[new]), "new synapses start at new_frac of the template weight"
    weak = (w > 0) & (w < 0.15 * w_init) & present
    assert not weak.any(), "no present synapse below the pruning threshold may remain"


# -------------------------------------------------------------------- text code
def test_text_code_is_deterministic_and_sparse():
    from synth.textcode import encode, words
    assert words("How is the S&P doing today?") == ["s&p", "doing", "today"]
    a, kept = encode("what is my resting heart rate", 1000, 0.05)
    b, _ = encode("what is my resting heart rate", 1000, 0.05)
    assert np.array_equal(a, b) and kept == ["resting", "heart", "rate"]
    assert 50 <= a.sum() <= 150                      # 3 words x 5 % of 1000 cells, minus overlaps
    c, _ = encode("chart my heart rate this week", 1000, 0.05)
    assert (a * c).sum() >= 90                        # the two shared words share their cells
    k1, _ = encode("what is my resting heart rate", 1000, 0.33, mode="kwta")
    k2, _ = encode("how is apple stock doing", 1000, 0.33, mode="kwta")
    assert k1.sum() == k2.sum() == 330                # k-winners: every message lights the same number


# ---------------------------------------------------------------- networks (slow)
@pytest.mark.slow
def test_shared_stp_is_exact_and_merged_inputs_keep_rates():
    """Fast mode must reproduce the per-synapse code spike for spike; merged
    Poisson inputs must leave population rates within noise."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import verify_fast
    assert verify_fast.main(T=600.0)


@pytest.mark.slow
def test_learner_default_path_is_unchanged():
    """Two-area learner: four trials with the default settings give the stored
    choices and pool spike counts (same seed). Guards the reward rule, the
    cue drive and the selection circuit against silent changes."""
    import brian2 as b2
    from brian2 import ms
    from synth.learner import LearnerColumn, build_learner_network
    col = build_learner_network(seed=3)
    L = LearnerColumn(col, seed=3, structural=True, surprise_gain=3.0, fast=True, lean=True,
                      genome=dict(eta=13.1), uncert_gain=2.0)
    L.net.run(300 * ms)
    out = []
    for k in range(4):
        r = L.trial("AB"[k % 2]); out.append((r["choice"], tuple(r["pool_spikes"])))
        L.reward(1.0 if r["choice"] == 1 + k % 2 else -1.0)
    ref_path = os.path.join(HERE, "reference", "learner_default.json")
    if not os.path.exists(ref_path):
        json.dump(out, open(ref_path, "w"))
        pytest.skip("reference recorded; rerun to compare")
    ref = [(c, tuple(p)) for c, p in json.load(open(ref_path))]
    assert out == ref


@pytest.mark.slow
def test_sensory_layer_save_load_roundtrip(tmp_path):
    from synth.sensory import SensoryLayer
    S = SensoryLayer(n_in=100, n_e=20, seed=0)
    x = np.zeros(100); x[:30] = 1.0
    for _ in range(5):
        S.present(x, learn=True)
    w = np.asarray(S.S.w).copy(); th = np.asarray(S.E.theta).copy()
    p = str(tmp_path / "s.npz"); S.save(p)
    T = SensoryLayer.load(p)
    assert np.allclose(np.asarray(T.S.w), w) and np.allclose(np.asarray(T.E.theta), th)
    assert T.trained_examples == 5


# ---------------------------------------------------------------- basal ganglia (stage 26)
def test_psp_scale_gives_the_stated_peak():
    """A spike must produce the paper's peak PSP on each membrane: integrate the LIF +
    exponential-current response numerically and compare with the analytic scale."""
    from synth.basal_ganglia import psp_scale, PSP, MEMBRANE
    for pop, m in MEMBRANE.items():
        for kind, (peak, tau_s) in PSP.items():
            dt, tau_m = 0.001, m["tau"]
            a = peak * psp_scale(tau_m, tau_s)
            g, v, vmax = a, 0.0, 0.0
            for _ in range(int(8 * max(tau_m, tau_s) / dt)):
                v += dt * (-v + g) / tau_m
                g += dt * (-g / tau_s)
                vmax = max(vmax, v)
            assert vmax == pytest.approx(peak, rel=2e-3), (pop, kind, vmax)


def test_basal_ganglia_wiring_follows_the_parameter_file():
    """K channels x 64 cells per population; within-channel pathways at p 0.25 inside
    the channel only, diffuse pathways at 0.25 / K across all channels, cortical inputs
    ~ctx_n_in per cell; every number labelled in results/bg_parameters.json."""
    import brian2 as b2
    from brian2 import ms, Hz
    from synth.basal_ganglia import BasalGanglia, load_params, C_WITHIN
    b2.start_scope(); b2.defaultclock.dt = 0.1 * ms
    K, n_pool = 3, 40
    pools = [b2.PoissonGroup(n_pool, rates=3 * Hz) for _ in range(K)]
    bg = BasalGanglia([(p, np.arange(n_pool)) for p in pools], seed=3, monitor="all")
    par = load_params()
    assert par["populations"]["per_channel_n"]["value"] == bg.n_per == 64
    for pop in bg.POPS:
        assert len(bg.groups[pop]) == K * 64
    by_name = {S.name: S for S in bg.synapses}
    d1_snr = by_name["bg_d1_snr"]
    assert np.all(bg.ch[np.asarray(d1_snr.i)] == bg.ch[np.asarray(d1_snr.j)])      # within channel only
    assert len(d1_snr) == pytest.approx(K * 64 * 64 * C_WITHIN, rel=0.1)
    stn_snr = by_name["bg_stn_snr"]
    same = np.mean(bg.ch[np.asarray(stn_snr.i)] == bg.ch[np.asarray(stn_snr.j)])
    assert same == pytest.approx(1.0 / K, abs=0.05)                                   # diffuse: all channels
    assert len(stn_snr) == pytest.approx((K * 64) ** 2 * C_WITHIN / K, rel=0.1)
    ctx = by_name["bg_ctx0_d1"]
    assert len(ctx) == pytest.approx(64 * bg.p["ctx_n_in"], rel=0.15)                # channel 0's pool -> D1_0
    assert set(bg.ch[np.asarray(ctx.j)]) == {0}
    assert all(k in par["rate_targets_hz"] for k in bg.POPS)


@pytest.mark.slow
def test_basal_ganglia_selects_the_driven_channel():
    """Stand-in cortex: a channel driven at 24 spikes/s must bring its SNr below 5 Hz while
    the resting channels stay tonic and its thalamic relay is released (Humphries 2006
    selection criterion; fitted gains from results/bg/)."""
    import brian2 as b2
    from brian2 import ms, Hz
    from synth.basal_ganglia import BasalGanglia, SELECT_HZ
    b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms; b2.seed(5)
    pools = [b2.PoissonGroup(40, rates=3 * Hz) for _ in range(3)]
    bg = BasalGanglia([(p, np.arange(40)) for p in pools], seed=5, ctx_gain=2.0, ctx_gain_stn=0.3)
    net = b2.Network(*pools, *bg.objects)
    net.run(1000 * ms)
    rest = bg.rates("SNr", 300, 1000)
    assert np.all(rest > SELECT_HZ)
    pools[1].rates = 24 * Hz
    net.run(800 * ms)
    snr = bg.rates("SNr", 1300, 1800); thal = bg.rates("Thal", 1300, 1800)
    assert bg.selected(1300, 1800) == 1 and snr[1] < SELECT_HZ and min(snr[0], snr[2]) > SELECT_HZ
    assert thal[1] > 5 * max(thal[0], thal[2])


# ------------------------------------------------------- plasticity in the loop (stage 27)
def test_plastic_context_synapses_follow_the_rule():
    """Context -> D1/D2 synapses carry weight and eligibility; reward(delta) moves D1 weights with
    delta and D2 against it, in proportion to eligibility, within [0, w_max]; nothing moves at
    eta_bg = 0."""
    import brian2 as b2
    from brian2 import ms, Hz
    from synth.basal_ganglia import BasalGanglia
    b2.start_scope(); b2.defaultclock.dt = 0.1 * ms
    pools = [b2.PoissonGroup(40, rates=3 * Hz) for _ in range(2)]
    cue = b2.PoissonGroup(100, rates=1 * Hz)
    bg = BasalGanglia([(p, np.arange(40)) for p in pools], seed=2, context=(cue, np.arange(100)),
                      plastic=True, eta_bg=0.1, ctx_gain_cue=1.0)
    assert [pop for _, _, pop in bg.plastic] == ["D1", "D2"]
    for S, sign, pop in bg.plastic:
        assert len(S) == pytest.approx(2 * 64 * bg.p["cue_n_in"], rel=0.15)
        S.elig = np.linspace(0, 1, len(S)); S.t_e = 0 * ms           # hand-set eligibilities
    w_before = {pop: np.asarray(S.wf).copy() for S, _, pop in bg.plastic}
    bg.reward(+1.0, t_ms=0.0)
    for S, sign, pop in bg.plastic:
        dw = np.asarray(S.wf) - w_before[pop]
        e = np.linspace(0, 1, len(S))
        assert np.allclose(dw, np.clip(w_before[pop] + sign * 0.1 * e, 0, bg.w_max[pop]) - w_before[pop])
        assert (dw >= 0).all() if pop == "D1" else (dw <= 0).all()
    bg.p["eta_bg"] = 0.0
    assert bg.reward(+1.0) == {}


@pytest.mark.slow
def test_dopamine_cells_burst_to_reward_and_pause_to_punishment():
    """The dopamine population sits near its tonic rate, bursts to a reward event and pauses to a
    negative one; the read-out error has the right sign in each case (stand-in cortex)."""
    import brian2 as b2
    from brian2 import ms, Hz
    from synth.basal_ganglia import BasalGanglia, DopamineSystem
    b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms; b2.seed(7)
    pools = [b2.PoissonGroup(40, rates=3 * Hz) for _ in range(2)]
    cue = b2.PoissonGroup(100, rates=1 * Hz)
    bg = BasalGanglia([(p, np.arange(40)) for p in pools], seed=7, context=(cue, np.arange(100)), plastic=True, eta_bg=0.05)
    da = DopamineSystem(bg, (cue, np.arange(100)), seed=7)
    net = b2.Network(*pools, cue, *bg.objects, *da.objects)
    net.run(1000 * ms)
    tonic = da.rate(300, 1000)
    assert 2.5 < tonic < 7.0, tonic
    t = float(net.t / ms); da.deliver(+1.0); net.run(150 * ms); da.end_event(); net.run(50 * ms)
    assert da.rate(t, t + 200) > 15 and da.delta(t, t + 200) > 0.5
    net.run(500 * ms)
    t = float(net.t / ms); da.deliver(-1.0); net.run(150 * ms); da.end_event(); net.run(50 * ms)
    assert da.rate(t, t + 200) < 2 and da.delta(t, t + 200) < -0.5


# ------------------------------------------------------ thalamus as the pools' drive (stage 28)
def test_routing_through_the_thalamus_preserves_the_resting_input():
    """route_external_through_thalamus removes `frac` of each pool cell's external Poisson input and
    sets the relay weights so that the relay at its resting rate delivers exactly what was removed."""
    import brian2 as b2
    from brian2 import ms, Hz
    from synth.basal_ganglia import BasalGanglia, route_external_through_thalamus, THAL_REST_HZ
    b2.start_scope(); b2.defaultclock.dt = 0.1 * ms
    G = b2.NeuronGroup(100, "v : 1\ng_ed : siemens\nlam_ext : 1\ne_ext : 1", threshold="v > 1")
    dt_s = 1e-4
    lam0 = np.linspace(2.0, 3.0, 100); G.lam_ext = lam0; G.e_ext = np.exp(-lam0)
    pools = [np.arange(0, 30), np.arange(30, 60)]
    bg = BasalGanglia([(G, p) for p in pools], seed=4, target=(G, "g_ed", 0.33), thal_n_in=64)
    info = route_external_through_thalamus(G, pools, 0.15, bg, 0.33)
    cells = np.arange(60)
    assert np.allclose(np.asarray(G.lam_ext)[cells], 0.85 * lam0[cells])
    assert np.allclose(np.asarray(G.lam_ext)[60:], lam0[60:])                     # non-pool cells untouched
    assert np.allclose(np.asarray(G.e_ext)[cells], np.exp(-0.85 * lam0[cells]))
    # relay at rest: sum over a cell's relay inputs of rate x weight == removed rate x external weight
    w = np.asarray(bg.thal_ctx.w / b2.nS); j = np.asarray(bg.thal_ctx.j)
    delivered = np.bincount(j, weights=w * THAL_REST_HZ, minlength=100)[cells]
    removed = 0.15 * lam0[cells] / dt_s * 0.33
    assert np.allclose(delivered, removed, rtol=1e-9)
    assert info["frac"] == 0.15 and info["relay_inputs_per_cell"] == 64


@pytest.mark.slow
def test_relay_homeostasis_pulls_a_held_channel_back_toward_its_target():
    """Stage 30: with thal_hom > 0 a relay held selected for seconds is pulled back toward 13 Hz,
    while with thal_hom = 0 it stays high; v_hom stays inside its bound."""
    import brian2 as b2
    from brian2 import ms, Hz
    from synth.basal_ganglia import BasalGanglia
    out = {}
    for k in (0.0, 0.5):
        b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms; b2.seed(1)
        pools = [b2.PoissonGroup(40, rates=3 * Hz) for _ in range(2)]
        bg = BasalGanglia([(p, np.arange(40)) for p in pools], seed=1, ctx_gain=2.0, ctx_gain_stn=0.3,
                          thal_drive_mV=31, thal_hom=k)
        net = b2.Network(*pools, *bg.objects); net.run(1000 * ms)
        pools[0].rates = 14 * Hz; net.run(6000 * ms)
        out[k] = bg.rates("Thal", 6000, 7000)[0]
        if k > 0:
            assert np.all(np.abs(np.asarray(bg.groups["Thal"].v_hom / b2.mV)) <= bg.p["hom_max_mV"] + 0.1)   # clipped every 10 ms
    assert out[0.0] > 40 and out[0.5] < 0.5 * out[0.0], out


@pytest.mark.slow
def test_merged_loop_gives_identical_spikes():
    """Stage 35: the six populations as slices of one NeuronGroup (merged=True, the default) must give
    the same spikes, spike for spike, as six separate groups - with the relay homeostat, its gate, and
    the thalamostriatal pathway all in use (the learner-level check is scripts/verify_merged_bg.py)."""
    import brian2 as b2
    from brian2 import ms, Hz
    from synth.basal_ganglia import BasalGanglia
    out = {}
    for merged in (False, True):
        b2.start_scope(); b2.prefs.codegen.target = "cython"; b2.defaultclock.dt = 0.1 * ms; b2.seed(5)
        src = b2.PoissonGroup(60, 8 * Hz, name="src")
        bg = BasalGanglia([(src, np.arange(k * 20, (k + 1) * 20)) for k in range(3)], seed=3, ctx_gain=2.0,
                          ctx_gain_stn=0.3, thal_drive_mV=31, thal_hom=0.5, thal_hom_gated=True, thal_str_gain=0.5,
                          monitor="all", merged=merged)
        net = b2.Network(src, *bg.objects); net.run(600 * ms)
        bg.set_homeostat(False); net.run(300 * ms)
        out[merged] = {p: (np.asarray(m.i).copy(), np.asarray(m.t / ms).copy()) for p, m in bg.monitors.items()}
    assert sum(len(v[0]) for v in out[True].values()) > 1000
    for pop in BasalGanglia.POPS:
        assert np.array_equal(out[False][pop][0], out[True][pop][0]), pop
        assert np.array_equal(out[False][pop][1], out[True][pop][1]), pop


def test_word_ideal_reader_matches_brute_force():
    """Stage 36: the bigram forward filter gives the exact posterior of the last letter (brute force over
    every letter string of the same length), and an uninformative reader predicts 't' after a word start."""
    from synth.words import NGram, L
    rng = np.random.default_rng(0)
    words = ["the", "then", "than", "ten", "net", "eat", "tea", "ant"]
    G = NGram(words, np.ones(len(words)), n=2, alpha=0.1)
    E = rng.dirichlet(np.ones(L) * 0.5, 3)
    post, _ = G.filter(E)
    import itertools
    sub = np.argsort(-E, 1)[:, :6]                       # brute force over the 6 likeliest letters per slot
    num = np.zeros(L)
    for s in itertools.product(*sub):
        p, prev = 1.0, L
        for t, c in enumerate(s):
            p *= G.P[prev, c] * E[t, c]; prev = c
        num[s[-1]] += p
    bf = num / num.sum()
    m = num > 0
    E2 = E.copy()
    for t in range(3):                                   # same restriction for the filter
        keep = np.zeros(L, bool); keep[sub[t]] = True; E2[t, ~keep] = 0
    post2, _ = G.filter(np.maximum(E2, 0))
    assert np.allclose(post2[-1][m], bf[m], rtol=1e-4, atol=0)       # residue: the 1e-12 evidence floor
    _, pred = G.filter(np.full((1, L), 1.0 / L))
    assert pred[0].argmax() == 19                        # 't'


def test_prediction_rule_learns_a_sequence_from_its_own_input():
    """Stage 36: the delta rule, taught only by the arriving letter's evidence, learns a deterministic
    successor (one-hot context of the current letter -> next letter)."""
    from synth.words import Predictor, L, letter_pairs
    nxt = np.roll(np.arange(L), -1)
    for scale in ("sum", "none"):                         # Stage 36 and Stage 37 input scaling
        P = Predictor(L, eta=0.5, scale=scale)
        for _ in range(40):
            for a in range(L):
                c = np.zeros(L); c[a] = 1.0
                y = np.full(L, 0.02 / (L - 1)); y[nxt[a]] = 0.98       # noisy evidence of the true next letter
                P.learn(c, y)
        assert np.mean([P.predict(np.eye(L)[a]).argmax() == nxt[a] for a in range(L)]) == 1.0
    g = np.zeros(2 * L + 1); g[3] = 1.0; g[L + 7] = 0.5               # current 'd', earlier 'h' at 0.5
    pc = letter_pairs(g).reshape(L + 1, L + 1)
    assert pc[3, 7] == 0.5 and pc.sum() == 0.5


@pytest.mark.slow
def test_frozen_reads_do_not_change_the_layer():
    """Stage 38 fix: reading a trained self-organising layer (features / spike_train) leaves its adaptive
    thresholds exactly as they were, however long the reading goes on (they used to decay with tau_theta,
    so a readout trained on early recordings was tested on a different network)."""
    import brian2 as b2
    from synth.sensory import SensoryLayer
    b2.start_scope()
    rng = np.random.default_rng(0)
    S = SensoryLayer(n_in=64, n_e=20, seed=0, dt_ms=0.5)
    X = (rng.random((30, 64)) < 0.2).astype(float)
    for x in X[:10]:
        S.present(x, learn=True)
    th0 = np.asarray(S.E.theta).copy(); w0 = np.asarray(S.S.w).copy()
    for x in X:
        S.features(x)
    assert np.array_equal(th0, np.asarray(S.E.theta)) and np.array_equal(w0, np.asarray(S.S.w))
    th_before = np.asarray(S.E.theta).copy()
    S.present(X[0], learn=True)                                # learning still adapts the thresholds
    assert not np.array_equal(th_before, np.asarray(S.E.theta)) or np.asarray(S.mon.count).sum() == 0


def test_word_context_from_both_sides():
    """Stage 40: the both-sides context of a letter holds its neighbours, fading traces of the letters
    further out on each side, and first/last flags - and nothing of the letter itself."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from capitals import context2
    ev = np.eye(26)[[19, 7, 4, 13]]                       # t h e n
    c = context2(ev, 1)
    assert c[19] == 1 and c[26 + 4] == 1                  # previous t, next e
    assert c[52:78].sum() == 0 and c[78 + 13] == 0.5      # nothing further left; n fades on the right
    assert c[104] == 0 and c[105] == 0 and c[7] == 0 and c[26 + 7] == 0
    assert context2(ev, 0)[104] == 1 and context2(ev, 3)[105] == 1


def test_word_units_learn_from_one_showing_and_reject_unknown_words():
    """Stage 42: a word unit taught from one noisy showing recognises its word from other noisy letter
    evidence; a word that was never taught gets a clearly lower match."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from word_units import Lexicon
    from synth.words import encode
    rng = np.random.default_rng(0)
    words = ["the", "and", "for", "that", "with", "this"]
    def noisy(w):                                          # letter evidence: 70 % on the right letter, rest spread
        e = rng.dirichlet(np.ones(26) * 0.3, len(w)) * 0.3
        e[np.arange(len(w)), encode(w)] += 0.7
        return e
    lex = Lexicon(words, np.ones(len(words)))
    for i, w in enumerate(words):
        lex.teach(i, noisy(w))
    lex.finish()
    hits = [lex.read(noisy(w))[0] == i for i, w in enumerate(words) for _ in range(20)]
    assert np.mean(hits) == 1.0
    known = np.mean([lex.read(noisy("that"))[1] for _ in range(20)]); unknown = np.mean([lex.read(noisy("when"))[1] for _ in range(20)])
    assert known > unknown + 0.5


def test_primer_parses_into_lessons_with_pictures():
    """Stage 43: the First Reader parses into ordered lessons; lesson 1 is the running dog."""
    from synth.primer import lessons
    b = lessons()
    assert len(b) >= 50 and b[0]["pictures"] == ["Running dog."]
    assert b[0]["sentences"] == [["The", "dog"], ["The", "dog", "ran"]]
    words = {t.lower() for l in b for s in l["sentences"] for t in s}
    assert 600 < len(words) < 750 and all(t.isalpha() for t in words)


def test_picture_word_links_prefer_the_word_that_comes_with_the_picture():
    """Stage 45: delta P links a picture to the word that comes with it more often than without it - not to
    a word met everywhere ("the") and not to a word met once."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from picture_book import Links
    Lk = Links()
    for i in range(20):
        pics = [0] if i < 8 else [1]
        words = ["the", "ran"] + (["dog"] if i < 6 else []) + (["sat"] if i == 3 else [])
        Lk.page(pics, words)
    assert Lk.name(0, None) == "dog"
    assert Lk.strength(0, "the") == 0.0 and Lk.strength(0, "sat") < Lk.strength(0, "dog")


def test_code_reading_splits_identifiers_into_shared_pieces():
    """Stage 46: tokens and pieces - read_file and readFile share the units for 'read' and 'file'."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from read_code import tokens, pieces, TokenUnits
    t = tokens("def read_file(path):\n    return open(path).read()")
    assert t[:3] == ["def", "read_file", "("] and "<indent>" in t and t.count("<nl>") == 2
    assert pieces("read_file") == pieces("readFile") == ["read", "file"] and pieces("(") == []
    TU = TokenUnits(); a = TU.read(["read_file"]); b = TU.read(["readFile"])
    assert len(set(a) & set(b)) == 2
