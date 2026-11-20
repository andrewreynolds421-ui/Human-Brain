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


def test_sleep_merges_a_duplicate_word_unit_and_keeps_different_words():
    """Stage 48: replaying a word unit's own pattern merges it into a stronger unit that recognises it as the
    same word (a duplicate made from a misread letter); a different word of the same length stays."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from sleep_primer import merge
    from read_primer import Words
    from synth.words import encode
    def ev(w, wrong_at=None):
        e = np.full((len(w), 26), 0.004); e[np.arange(len(w)), encode(w)] = 0.9
        if wrong_at is not None:
            e[wrong_at] = 0.004; e[wrong_at, 4] = 0.5; e[wrong_at, encode(w)[wrong_at]] = 0.45   # "can" half-read as "cen"
        return e / e.sum(1, keepdims=True)
    W = Words(-0.76, 0.0)
    a = W.recruit(ev("can"), "can"); W.n[a] = 10
    W.recruit(ev("can", wrong_at=1), "can")
    W.recruit(ev("dog"), "dog")
    N, removed, wrong, right = merge(W)
    assert removed == 1 and right == 1 and wrong == 0 and sorted(N.truth) == ["can", "dog"]


def test_grammar_learns_word_classes_and_prefers_word_order_seen():
    """Stage 50: from a tiny text, nouns and verbs fall into different categories, and a sentence in the
    order seen scores higher than the same words with two neighbours swapped."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from grammar import Grammar
    rng = np.random.default_rng(0)
    nouns, verbs = ["dog", "cat", "bird", "girl", "boy", "fox"], ["ran", "sat", "sang", "hid", "fell", "slept"]
    sents = [["the", str(rng.choice(nouns)), str(rng.choice(verbs)), "away"] for _ in range(400)]
    G = Grammar(sents, K=3, n_ctx=20, min_count=3, seed=0)
    assert len({G.cat[n] for n in nouns}) == 1 and len({G.cat[v] for v in verbs}) == 1 and G.cat["dog"] != G.cat["ran"]
    assert G.score(["the", "dog", "ran", "away"]) > G.score(["the", "ran", "dog", "away"])


def test_grammar_memory_controlled_agreement_items():
    """Stage 51: built agreement pairs differ only at the verb, and the right verb agrees with the SUBJECT
    (not with the noun in between)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from collections import Counter
    from grammar_memory import controlled, FLIP
    words = ["dog", "dogs", "cat", "cats", "bird", "birds", "tree", "trees", "near", "of", "saw", "met", "here", "very", "happy"]
    tok = Counter({w: 50 for w in words}); E2 = controlled(tok, set(words))
    assert len(E2) > 100
    for x in E2:
        diff = [i for i, (g, b) in enumerate(zip(x["good"], x["bad"])) if g != b]
        assert len(diff) == 1 and FLIP[x["good"][diff[0]]] == x["bad"][diff[0]]
        assert (x["good"][diff[0]] in ("are", "were")) == x["subj_plural"] == x["good"][1].endswith("s")


def test_graded_reading_bands_and_readability():
    """Stage 53: harder text gets a higher readability grade; grades map to bands with 7 and up merged."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from synth.readability import fk_grade
    from graded_reading import band_of
    easy = "The cat sat. The dog ran. I see the sun. We can play."
    hard = ("Notwithstanding the considerable difficulties occasioned by the unprecedented circumstances, "
            "the administration endeavoured to accommodate the extraordinary requirements of the population.")
    assert fk_grade(easy) < 2 < 12 < fk_grade(hard)
    assert [band_of(g) for g in (0.4, 1.4, 3.6, 6.2, 7.0, 11.0)] == ["1", "1", "4", "6", "7+", "7+"]


def test_subject_slot_gates():
    """Stage 54: the always-open control's gates are 1 everywhere; a learned slot's gates lie in (0, 1); one gate
    value per word per slot."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from subject_slot import SlotNet, K
    sents = [["the", "dog", "ran", "home"], ["the", "cats", "sat", "there"]] * 20
    C = dict(train=sents, val=sents[:2])
    G_open = SlotNet("open", C, 0).gates_of([sents[0]])[0]
    G_learn = SlotNet("learned", C, 0).gates_of([sents[0]])[0]
    assert G_open.shape == (4, K) and np.allclose(G_open, 1.0)
    assert G_learn.shape == (4, K) and ((G_learn > 0) & (G_learn < 1)).all()


def test_picture_pages_agree_with_the_actor():
    """Stage 55: the verb on a generated page agrees with the thing that acts, whose number the scene shows; the
    actor is marked as acting; the misleading-noun flag is right."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import picture_pages as PP
    bank = PP.PictureBank(); rng = np.random.default_rng(3)
    for _ in range(200):
        p = PP.make_page(rng, bank, [0, 1, 2])
        actor = p["objs"][p["actor"]]; v = p["words"][p["verb"]]
        several = bool(actor[401] == 1.0)
        assert several == p["plural"] == (v in ("are", "were"))
        assert actor[402 + len(PP.ACTIONS)] == 0.0 and sum(o[402 + len(PP.ACTIONS)] for o in p["objs"]) == 2.0
        others = [o for k, o in enumerate(p["objs"]) if k != p["actor"]]
        assert p["words"][p["verb"] + 1] in PP.ACTIONS


def test_picture_check_nouns_held_out_and_pages_agree():
    """Stage 56: held-out nouns never appear on pages; a page's verb agrees with the acting noun's number;
    every noun's stand-in look is the same each time (stable across runs)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import picture_check as PC
    tr, held = PC.nouns(); assert not set(tr) & set(held) and len(held) > 150
    look = PC.Look(); rng = np.random.default_rng(0); heldw = {w for p in held for w in p}
    for _ in range(300):
        p = PC.make_page(rng, tr, look)
        assert not set(p["words"]) & heldw
        assert (p["words"][p["verb"]] in ("are", "were")) == bool(p["number"])
    assert np.allclose(look("acquaintance", rng), PC.Look()("acquaintance", rng))


def test_frames_pages_use_no_held_out_frames():
    """Stage 57: pages never use a held-out preposition, relative-clause verb or ending in that role; the verb agrees
    with the subject's number; the built test uses only held-out frames."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import frames as F, picture_check as PC
    tr, held = PC.nouns(); look = PC.Look(); rng = np.random.default_rng(2)
    nounw = {w for p in tr for w in p}
    for _ in range(500):
        p = F.make_page(rng, tr, look); w = p["words"]
        frame_words = [x for x in w if x not in nounw]
        assert not set(frame_words) & (set(F.HELD_PREPS) | set(F.HELD_REL))
        assert (w[p["verb"]] in ("are", "were")) == bool(p["number"])
    T = F.built_tests(tr, held, np.random.default_rng(3), n=50)
    for x in T["page_nouns"] + T["held_out_nouns"]:
        assert x["good"][2] in F.HELD_PREPS + ["that"] and (x["kind"] == "pp" or x["good"][5] in F.HELD_REL)


def test_spelling_endings_shared():
    """Stage 58: words with the same ending share its units; the reader still runs."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import spelling as SP, picture_check as PC
    M = PC.CheckNet(["the", "dog", "dogs", "cat", "cats", "box", "boxes", "is", "are"], 0, True); SP.spelled(M)
    ix = {w: i for i, w in enumerate(M.itos)}; ids = M.m.e.ids
    assert int(ids[0][ix["dogs"]]) == int(ids[0][ix["cats"]]) != int(ids[0][ix["dog"]])
    assert int(ids[1][ix["boxes"]]) != int(ids[1][ix["dogs"]])            # "es" vs "gs"
    assert np.isfinite(M.logp([["the", "boxes", "are"]])[0])


def test_phenomena_pages_and_pairs():
    """Stage 60: training pages are grammatical by construction (reflexive agrees with its binder; 'any' only after a
    licensor in the same clause); test pairs differ only where the phenomenon is."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import phenomena as PH, picture_check as PC, frames as FR
    tr, held = PC.nouns(); rng = np.random.default_rng(4)
    for _ in range(400):
        p = PH.phenomenon_page(rng, tr, FR.REL); w = p["words"]
        if "any" in w:
            i = w.index("any"); assert set(w[:i]) & {"didn't", "never", "no", "not"}
            assert "that" not in w[:i] or w[w.index("that") + 1] != "didn't" or w.index("didn't") > w.index("that") + 2
    T = PH.built_tests(tr, FR.HELD_REL, PH.HELD_ADJ, np.random.default_rng(5), n=50)
    for k, prs in T.items():
        for g, b in prs:
            assert g != b and len(g) == len(b) and (k == "npi_scope" or sum(x != y for x, y in zip(g, b)) <= 2)


def test_adult_pages_agree():
    """Stage 61: in adult-construction pages, has / have and plain present verbs agree with a bare or quantified
    subject; be-negations are followed by -ing or 'here', do-negations by a base verb."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import sleep_grammar as SG, picture_check as PC, frames as FR
    tr, held = PC.nouns(); rng = np.random.default_rng(6); plural = {p[1] for p in tr}; sing = {p[0] for p in tr}
    for _ in range(600):
        w = SG.adult_page(rng, tr, FR.REL, SG.TRAIN_F, SG.TRAIN_M)["words"]
        for i, x in enumerate(w[:-1]):
            if x in ("isn't", "aren't", "wasn't", "weren't"): assert w[i + 1] == "here" or w[i + 1].endswith("ing")
            if x in ("don't", "doesn't"): assert w[i + 1] in SG.INTR
        if len(w) > 2 and w[0] in ("every", "each") and w[2] in ("has", "have"): assert w[2] == "has"
        if len(w) > 2 and w[0] in ("all", "most", "many", "few") and w[2] in ("has", "have"): assert w[2] == "have"


def test_register_markers():
    """Stage 63: speech sentences start with the talk marker, others with the book marker; the test marker switches unmarked ones"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import register as RG
    M = RG.Tagged(["the", "dog", "dogs", "runs", "run", "ever", "have", "you"], 0)
    RG.Tagged.EVAL = "book"
    assert M.enc(["the", "dog"])[0] == 1 and M.enc(RG.Talk(["have", "you"]))[0] == M.T != 1
    a = M.logp([["the", "dog", "runs"]])[0]; M.step([dict(words=RG.Talk(["have", "you", "ever"])), dict(words=["the", "dogs", "run"])])
    RG.Tagged.EVAL = "talk"
    assert M.enc(["the", "dog"])[0] == M.T and len(M.enc(["the", "dog"])) == 4
    assert M.logp([["the", "dog", "runs"]])[0] != a
    RG.Tagged.EVAL = "book"


def test_register_pages_both():
    """Stage 64: a wrapped page maker gives about half its pages the talk marker, the same one for the same words"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import register as RG
    words = [["the", "dog", str(i)] for i in range(400)]; it = iter(words * 2)
    mk = RG.half_talk(lambda: dict(words=next(it)))
    a = [isinstance(mk()["words"], RG.Talk) for _ in range(400)]; b = [isinstance(mk()["words"], RG.Talk) for _ in range(400)]
    assert a == b and 0.35 < sum(a) / 400 < 0.65
