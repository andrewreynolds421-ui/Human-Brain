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
