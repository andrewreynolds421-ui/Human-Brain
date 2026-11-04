"""Simulate a synthesized column with Brian2 (leaky integrate-and-fire).

Neuron model (current-based, exponential synapses)::

    dv/dt     = (E_L - v)/tau_m + (I_syn + I_dc)/C_m
    dI_syn/dt = -I_syn/tau_syn
    spike at v > V_th, reset to V_reset, refractory t_ref

Each synapse adds its weight (pA) to the target's I_syn after its own
delay.  Background drive is a Poisson process of K_ext fibres at bg_rate
per neuron plus the DC compensation computed at synthesis time.
"""
from __future__ import annotations

import time
import numpy as np
import brian2 as b2
from brian2 import ms, mV, pA, pF, Hz

from .build import SynthesizedColumn
from .params import POPULATIONS, N_POP


def simulate(col: SynthesizedColumn, t_sim_ms=1000.0, dt_ms=0.1, seed=0,
             target="cython", record_v_from=0, verbose=True, **net_kw):
    """Run the column; return dict with spike times / indices and rates.

    Returns
    -------
    dict with keys ``spike_t_ms``, ``spike_i``, ``pop_of``, ``n_per_pop``,
    ``t_sim_ms``, ``wall_s``
    """
    b2.start_scope()
    b2.prefs.codegen.target = target
    b2.defaultclock.dt = dt_ms * ms
    from .network import build_network
    nw = build_network(col, seed=seed, **net_kw)
    spikes, net = nw["spikes"], nw["net"]

    t0 = time.time()
    if verbose:
        print(f"simulating {col.N:,} neurons / {col.S:,} synapses for {t_sim_ms:.0f} ms ...",
              flush=True)
    net.run(t_sim_ms * ms, report="text" if verbose else None, report_period=10 * b2.second)
    wall = time.time() - t0
    if verbose:
        print(f"done in {wall:.1f} s wall")

    return dict(spike_t_ms=np.asarray(spikes.t / ms), spike_i=np.asarray(spikes.i),
                pop_of=col.pop_of, n_per_pop=col.n_per_pop, populations=col.pops,
                t_sim_ms=t_sim_ms, wall_s=wall)
