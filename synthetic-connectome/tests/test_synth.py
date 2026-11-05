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
