"""Reward-modulated STDP: the "dopamine-gated" three-factor learning rule.

This is the learning mechanism discussed for the mushroom body / basal
ganglia: synapses keep a short-lived *eligibility trace* of recent
pre-before-post spike pairings, and the trace is only converted into a
lasting weight change when a global reward signal arrives:

    de_ij/dt = -e_ij / tau_e  + (pairing of pre_i and post_j)
    dw_ij/dt = eta * DA(t) * e_ij

DA > 0 is reward, DA < 0 punishment.  No gradient, no backprop: every
synapse only ever sees its own two neurons plus one broadcast scalar.
A homeostatic rule (synaptic scaling: each neuron's total input weight is
held constant) keeps the weights bounded, as in cortex.

The testbed task: two input patterns (A, B) must be mapped to two output
"action" neurons.  The network is told only whether its action was right
(+1) or wrong (-1) at the end of each trial.

References: Izhikevich EM (2007) Cereb Cortex 17:2443; Fremaux & Gerstner
(2016) Front Neural Circuits 9:85 (review of three-factor rules).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RSTDPConfig:
    n_in: int = 200              # input neurons (Poisson)
    n_out: int = 2               # action neurons (LIF)
    dt: float = 1.0              # ms
    trial_ms: float = 300.0
    rate_on: float = 40.0        # Hz, inputs belonging to the shown pattern
    rate_off: float = 4.0        # Hz, other inputs (noise)
    # LIF output neurons (dimensionless voltage, threshold 1)
    tau_m: float = 20.0
    v_th: float = 1.0
    v_reset: float = 0.0
    t_ref: float = 3.0
    tau_syn: float = 5.0
    w_inh: float = 0.8           # mutual inhibition between action neurons
    # eligibility trace
    tau_plus: float = 20.0       # pre trace: how long a pre spike "waits" for a post spike
    tau_minus: float = 20.0      # post trace: used for the (optional) depression term
    a_plus: float = 1.0
    a_minus: float = 0.0         # 0 = potentiation-only eligibility (robust default)
    tau_e: float = 500.0         # eligibility decay, ms
    # reward
    eta: float = 0.0005
    w_max: float = 0.3
    w_init: float = 0.05
    noise_sigma: float = 0.30     # exploration: Gaussian current noise on action neurons
    use_baseline: bool = True    # DA = reward - running mean reward
    da_clip: float = 1.0         # |DA| never exceeds this (prevents over-punishment after streaks)
    baseline_tau_trials: float = 20.0
    seed: int = 0


class RSTDPNetwork:
    def __init__(self, cfg: RSTDPConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.w = np.clip(self.rng.normal(cfg.w_init, cfg.w_init * 0.2,
                                         (cfg.n_in, cfg.n_out)), 0, cfg.w_max)
        self.w_sum_target = self.w.sum(0).copy()   # homeostatic set-point
        self.e = np.zeros_like(self.w)
        self.x_pre = np.zeros(cfg.n_in)
        self.x_post = np.zeros(cfg.n_out)
        self.v = np.zeros(cfg.n_out)
        self.I = np.zeros(cfg.n_out)
        self.ref = np.zeros(cfg.n_out)
        self.r_bar = 0.0

    # ----------------------------------------------------------------- trial
    def run_trial(self, input_rates_hz, record=False):
        """Run one trial; return output spike counts (and voltage trace)."""
        c, dt = self.cfg, self.cfg.dt
        n_steps = int(c.trial_ms / dt)
        p_spike = input_rates_hz * dt / 1000.0
        counts = np.zeros(c.n_out, int)
        self.v[:] = c.v_reset
        self.I[:] = 0
        self.ref[:] = 0
        self.x_pre[:] = 0
        self.x_post[:] = 0
        self.e[:] = 0
        vs = []
        d_pre, d_post = np.exp(-dt / c.tau_plus), np.exp(-dt / c.tau_minus)
        d_e, d_syn = np.exp(-dt / c.tau_e), np.exp(-dt / c.tau_syn)
        for _ in range(n_steps):
            pre = self.rng.random(c.n_in) < p_spike
            self.I = self.I * d_syn + pre @ self.w
            I_eff = self.I + self.rng.normal(0, c.noise_sigma, c.n_out)
            active = self.ref <= 0
            self.v[active] += dt / c.tau_m * (-self.v[active] + I_eff[active])
            self.ref = np.maximum(self.ref - dt, 0)
            post = (self.v >= c.v_th) & active
            if post.any():
                self.v[post] = c.v_reset
                self.ref[post] = c.t_ref
                counts += post
                # winner-take-all: a spike inhibits the other action neuron(s)
                self.I -= c.w_inh * (post.sum() - post)
            # eligibility: pre-before-post pairings (and optional reverse)
            self.x_pre = self.x_pre * d_pre + pre
            self.x_post = self.x_post * d_post + post
            self.e *= d_e
            if post.any():
                self.e += c.a_plus * np.outer(self.x_pre, post)
            if c.a_minus and pre.any():
                self.e -= c.a_minus * np.outer(pre, self.x_post)
            if record:
                vs.append(self.v.copy())
        return counts, (np.array(vs) if record else None)

    # ---------------------------------------------------------------- reward
    def apply_reward(self, reward: float):
        """Dopamine pulse: eligibility -> lasting weight change, then scaling."""
        c = self.cfg
        da = reward - self.r_bar if c.use_baseline else reward
        da = float(np.clip(da, -c.da_clip, c.da_clip))
        if c.use_baseline:
            self.r_bar += (reward - self.r_bar) / c.baseline_tau_trials
        self.w += c.eta * da * self.e
        np.clip(self.w, 0, c.w_max, out=self.w)
        # homeostatic synaptic scaling: keep each neuron's total input fixed
        s = self.w.sum(0)
        s[s == 0] = 1.0
        self.w *= self.w_sum_target / s
        np.clip(self.w, 0, c.w_max, out=self.w)
        return da


# -------------------------------------------------------------------- task
def make_patterns(cfg: RSTDPConfig):
    """Pattern A drives the first half of the inputs, B the second half."""
    half = cfg.n_in // 2
    A = np.full(cfg.n_in, cfg.rate_off); A[:half] = cfg.rate_on
    B = np.full(cfg.n_in, cfg.rate_off); B[half:] = cfg.rate_on
    return {"A": A, "B": B}


def train(cfg: RSTDPConfig, n_trials=300, verbose=True):
    net = RSTDPNetwork(cfg)
    patterns = make_patterns(cfg)
    target = {"A": 0, "B": 1}
    half = cfg.n_in // 2
    log = []
    for k in range(n_trials):
        name = "A" if net.rng.random() < 0.5 else "B"
        counts, _ = net.run_trial(patterns[name])
        if counts[0] == counts[1]:
            action = int(net.rng.random() < 0.5)
        else:
            action = int(np.argmax(counts))
        correct = action == target[name]
        net.apply_reward(+1.0 if correct else -1.0)
        log.append(dict(pattern=name, action=action, correct=correct, counts=counts.copy(),
                        wA=net.w[:half].mean(0).copy(), wB=net.w[half:].mean(0).copy()))
        if verbose and (k + 1) % 50 == 0:
            acc = np.mean([r["correct"] for r in log[-50:]])
            L = log[-1]
            print(f"trial {k+1:4d}  accuracy(last 50) = {acc:.2f}   "
                  f"A->out0 {L['wA'][0]:.3f}  A->out1 {L['wA'][1]:.3f}  "
                  f"B->out0 {L['wB'][0]:.3f}  B->out1 {L['wB'][1]:.3f}   "
                  f"spikes {counts}")
    return net, log
