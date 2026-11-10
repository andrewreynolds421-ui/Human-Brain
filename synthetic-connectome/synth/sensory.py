"""Self-organising spiking sensory layer (after Diehl & Cook 2015, Front. Comput. Neurosci.).

Unlabeled inputs (here: messages through the fixed word code of synth/textcode.py)
drive Poisson input fibres onto excitatory LIF cells.  Learning needs no labels:

* input -> E synapses learn by spike-timing plasticity (triplet rule as in
  Diehl & Cook: depression on presynaptic spikes, potentiation on postsynaptic
  spikes scaled by a slow postsynaptic trace),
* each E cell drives its own inhibitory cell, which inhibits every *other* E
  cell: the cells compete, so each learns a different family of inputs,
* an adaptive threshold (theta rises with every spike) keeps any cell from
  winning everything, and each cell's input weights are renormalised to a
  fixed sum after every example.

After training the weights are frozen; a message is then represented by which
few E cells fire.  The parameters are Diehl & Cook's (MNIST); not fitted to
cortex.  Between examples the state is reset directly instead of simulating
their 150 ms rest (same effect, less time).
"""
import json
import time

import numpy as np
import brian2 as b2
from brian2 import ms, mV, Hz

from .textcode import encode

E_EQS = """
dv/dt = ((v_rest - v) + ge * (-v) + gi * (-100*mV - v)) / (100*ms) + noise_mV * mV * sqrt(2 / (100*ms)) * xi : volt (unless refractory)
dge/dt = -ge / (1*ms) : 1
dgi/dt = -gi / (2*ms) : 1
dtheta/dt = -theta / tau_theta : volt
theta_plus : volt (shared)
"""
I_EQS = """
dv/dt = ((-60*mV - v) + ge * (-v) + gi * (-85*mV - v)) / (10*ms) : volt (unless refractory)
dge/dt = -ge / (1*ms) : 1
dgi/dt = -gi / (2*ms) : 1
"""
STDP_MODEL = """
w : 1
plastic : 1 (shared)
dpre/dt = -pre / (20*ms) : 1 (event-driven)
dpost1/dt = -post1 / (20*ms) : 1 (event-driven)
dpost2/dt = -post2 / (40*ms) : 1 (event-driven)
"""
STDP_PRE = """
ge_post += w
pre = 1.
w = clip(w - plastic * 0.0001 * post1, 0, 1.0)
"""
STDP_POST = """
post2before = post2
w = clip(w + plastic * 0.01 * pre * post2before, 0, 1.0)
post1 = 1.
post2 = 1.
"""


FAST_E_EXTRA = """
post1 : 1
post2 : 1
"""
FAST_IN_EQS = """
rate : Hz
pre : 1
"""


class SensoryLayer:
    def __init__(self, n_in=1000, n_e=1000, word_frac=0.02, rate_hz=63.75, present_ms=350.0,
                 seed=0, dt_ms=0.5, target="cython", fast=False, noise_mV=0.0):
        """fast=True keeps the plasticity traces on the neurons instead of the synapses. Tested: no
        faster (0.8x; the cost is delivering each spike to every synapse), so off by default."""
        self.fast = bool(fast)
        self.n_in, self.n_e, self.word_frac = n_in, n_e, word_frac
        self.rate_hz, self.present_ms = rate_hz, present_ms
        self.seed = seed
        b2.prefs.codegen.target = target
        self.dt_ms = dt_ms
        rng = np.random.default_rng(seed)
        clock = b2.Clock(dt=dt_ms * ms, name="sensory_clock")
        # noise_mV: membrane noise (SD); breaks ties when many cells near threshold would fire together
        self.noise_mV = float(noise_mV)
        ns = dict(v_rest=-65 * mV, tau_theta=1e7 * ms, noise_mV=self.noise_mV)
        if self.fast:
            self.inp = b2.NeuronGroup(n_in, FAST_IN_EQS, threshold="rand() < rate * dt", reset="pre = 1.",
                                      clock=clock, name="sens_input")
            self.E = b2.NeuronGroup(n_e, E_EQS + FAST_E_EXTRA, threshold="v > -52*mV - 20*mV + theta",
                                    reset="v = -65*mV; theta += theta_plus; post1 = 1.; post2 = 1.",
                                    refractory=5 * ms, method="euler", namespace=ns, clock=clock, name="sens_E")
            d = float(dt_ms)
            self.inp.run_regularly(f"pre *= {float(np.exp(-d / 20.0))!r}", when="before_synapses", name="sens_in_decay")
            self.E.run_regularly(f"post1 *= {float(np.exp(-d / 20.0))!r}\npost2 *= {float(np.exp(-d / 40.0))!r}",
                                 when="before_synapses", name="sens_E_decay")
        else:
            self.inp = b2.PoissonGroup(n_in, rates=0 * Hz, clock=clock, name="sens_input")
            self.E = b2.NeuronGroup(n_e, E_EQS, threshold="v > -52*mV - 20*mV + theta", reset="v = -65*mV; theta += theta_plus",
                                    refractory=5 * ms, method="euler", namespace=ns, clock=clock, name="sens_E")
        self.I = b2.NeuronGroup(n_e, I_EQS, threshold="v > -40*mV", reset="v = -45*mV", refractory=2 * ms,
                                method="euler", clock=clock, name="sens_I")
        if self.fast:
            self.S = b2.Synapses(self.inp, self.E, "w : 1\nplastic : 1 (shared)",
                                 on_pre="ge_post += w\nw = clip(w - plastic * 0.0001 * post1_post, 0, 1.0)",
                                 on_post="w = clip(w + plastic * 0.01 * pre_pre * post2_post, 0, 1.0)",
                                 clock=clock, name="sens_in_E")
        else:
            self.S = b2.Synapses(self.inp, self.E, STDP_MODEL, on_pre=STDP_PRE, on_post=STDP_POST, clock=clock,
                                 name="sens_in_E")
        self.S.connect()
        self.S.w = rng.random(n_in * n_e) * 0.3
        self.S.plastic = 1.0
        self.EI = b2.Synapses(self.E, self.I, on_pre="ge_post += 10.4", clock=clock, name="sens_E_I")
        self.EI.connect(j="i")
        self.IE = b2.Synapses(self.I, self.E, "w_ie : 1 (shared)", on_pre="gi_post += w_ie", clock=clock, name="sens_I_E")
        self.IE.connect(condition="i != j")
        self.IE.w_ie = 17.0
        self.mon = b2.SpikeMonitor(self.E, record=False, name="sens_counts")   # spike counts only
        self.net = b2.Network(self.inp, self.E, self.I, self.S, self.EI, self.IE, self.mon)
        self.E.v = -65 * mV; self.I.v = -60 * mV
        self.E.theta = 20 * mV                  # Diehl & Cook start: threshold -52 mV + (theta - 20 mV)
        self.norm_sum = 0.1 * n_in              # Diehl & Cook: 78 for 784 inputs
        self.normalize()
        self.trained_examples = 0

    # ----------------------------------------------------------------------------------
    def code(self, text):
        return encode(text, self.n_in, self.word_frac)[0]

    def normalize(self):
        if not hasattr(self, "_post"):
            self._post = np.asarray(self.S.j)
        w = np.asarray(self.S.w)
        tot = np.bincount(self._post, weights=w, minlength=self.n_e)
        self.S.w = w * (self.norm_sum / np.maximum(tot, 1e-12))[self._post]

    def _set_rates(self, hz):
        if self.fast:
            self.inp.rate = hz * Hz
        else:
            self.inp.rates = hz * Hz

    def _reset_state(self):
        self.E.v = -65 * mV; self.E.ge = 0; self.E.gi = 0
        self.I.v = -60 * mV; self.I.ge = 0; self.I.gi = 0
        if self.fast:                           # as after Diehl & Cook's 150 ms rest
            self.inp.pre = 0; self.E.post1 = 0; self.E.post2 = 0

    def present(self, x, learn=False, min_spikes=5, max_tries=5, max_cells=50):
        """drive with activity vector x (0/1 per input); returns spike count per E cell.
        Safety valve (learning only): if more than max_cells cells fire for one example — a
        synchrony burst that normal examples never cause (3-20 cells) and that would make many
        cells learn the same thing and collapse into one template — that example's learning is
        undone (weights and thresholds restored)."""
        if learn and max_cells:
            w_backup = np.asarray(self.S.w).copy(); th_backup = np.asarray(self.E.theta).copy()
        if not learn:
            # Stage 38 fix: a frozen layer must not change while it is read.  theta decays with
            # tau_theta (1e4 s) whether or not the layer learns; recording ~24,000 images (~8,000 s
            # simulated) let the thresholds sag by ~half between the first and last recorded image, so
            # a readout trained on early recordings was tested on a different network.  Diehl & Cook
            # hold theta fixed in test mode; so do we (restored after every frozen presentation).
            th_frozen = np.asarray(self.E.theta).copy()
        self.S.plastic = 1.0 if learn else 0.0
        self.E.theta_plus = (0.05 if learn else 0.0) * mV       # threshold adapts only while learning
        boost = 0.0
        for _ in range(max_tries):
            c0 = np.asarray(self.mon.count).copy()
            self._set_rates(x * (self.rate_hz + boost))
            self.net.run(self.present_ms * ms)
            counts = np.asarray(self.mon.count) - c0
            self._set_rates(0)
            self._reset_state()
            if counts.sum() >= min_spikes or not x.any():
                break
            boost += 32.0                       # Diehl & Cook: raise input intensity and repeat
        if not learn:
            self.E.theta = th_frozen * b2.volt
        if learn and max_cells and (counts > 0).sum() > max_cells:
            self.S.w = w_backup; self.E.theta = th_backup * b2.volt
            self.skipped_bursts = getattr(self, "skipped_bursts", 0) + 1
        elif learn:
            self.normalize()
        if learn:
            self.trained_examples += 1
        return counts

    def train(self, texts, log_every=500, verbose=True):
        t0 = time.time()
        for n, t in enumerate(texts):
            self.present(self.code(t), learn=True)
            if verbose and (n + 1) % log_every == 0:
                used = int((np.asarray(self.E.theta) > 0.0201).sum())
                print(f"  sensory training {n + 1}/{len(texts)}: {time.time() - t0:.0f}s, cells used {used}/{self.n_e}",
                      flush=True)

    def features(self, text, inhib=1.0):
        """spike count per cell for a message (str) or an input vector in [0, 1] (e.g. image pixels);
        inhib < 1 weakens lateral inhibition (more winners) for readout"""
        self.IE.w_ie = 17.0 * inhib
        c = self.present(self.code(text) if isinstance(text, str) else np.asarray(text, float), learn=False)
        self.IE.w_ie = 17.0
        return c

    def spike_train(self, text, duration_ms=None):
        """frozen layer's response to one message: (cell indices, times in ms from onset)"""
        if not hasattr(self, "_tmon"):
            self._tmon = b2.SpikeMonitor(self.E, name="sens_times")
            self.net.add(self._tmon)
        self.S.plastic = 0.0; self.E.theta_plus = 0 * mV
        th_frozen = np.asarray(self.E.theta).copy()
        n0, t0 = self._tmon.num_spikes, float(self.net.t / ms)
        self._set_rates(self.code(text) * self.rate_hz)
        self.net.run((duration_ms or self.present_ms) * ms)
        self._set_rates(0)
        i = np.asarray(self._tmon.i)[n0:].copy(); t = np.asarray(self._tmon.t / ms)[n0:] - t0
        self.E.theta = th_frozen * b2.volt                     # frozen read: thresholds unchanged (Stage 38)
        self._reset_state()
        self._tmon.resize(0); self._tmon.variables["N"].set_value(0)
        return i, t

    # ----------------------------------------------------------------------------------
    def save(self, path):
        np.savez_compressed(path, w=np.asarray(self.S.w, np.float32), theta=np.asarray(self.E.theta / mV, np.float32),
                            meta=json.dumps(dict(n_in=self.n_in, n_e=self.n_e, word_frac=self.word_frac,
                                                 rate_hz=self.rate_hz, present_ms=self.present_ms, seed=self.seed,
                                                 trained_examples=self.trained_examples, fast=self.fast,
                                                 noise_mV=self.noise_mV, dt_ms=float(self.dt_ms))))

    @classmethod
    def load(cls, path, **kw):
        d = np.load(path)
        meta = json.loads(str(d["meta"]))
        kw.setdefault("fast", meta.get("fast", False))
        kw.setdefault("noise_mV", meta.get("noise_mV", 0.0))
        kw.setdefault("dt_ms", meta.get("dt_ms", 0.5))
        s = cls(n_in=meta["n_in"], n_e=meta["n_e"], word_frac=meta["word_frac"], rate_hz=meta["rate_hz"],
                present_ms=meta["present_ms"], seed=meta["seed"], **kw)
        s.S.w = d["w"].astype(float); s.E.theta = d["theta"].astype(float) * mV
        s.trained_examples = meta["trained_examples"]
        return s
