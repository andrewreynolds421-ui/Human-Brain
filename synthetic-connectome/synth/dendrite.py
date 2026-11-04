"""Two-compartment (soma + dendrite) pyramidal neurons.

Each excitatory neuron gets a passive dendritic compartment coupled to its
integrate-and-fire soma:

    C_s dv/dt   = g_Ls (E_L - v)   + g_c (v_d - v) + somatic synaptic, DC, adaptation currents
    C_d dv_d/dt = g_Ld (E_L - v_d) + g_c (v - v_d) + dendritic synaptic currents

Spikes are detected and reset at the soma.  Interneurons (compact, aspiny)
stay single-compartment (g_c = 0, every input on the soma).

Where inputs land (anatomy):
    onto pyramidal cells   excitatory (local + external)  -> dendrite
                           fast-spiking (basket) inhibition -> soma
                           non-fast-spiking inhibition      -> dendrite
    onto interneurons      everything                       -> soma

Compartment split.  Two numbers describe the reduction: ``dend_cap_frac``
(share of membrane in the dendrite, default 0.7) and ``dend_attenuation``
(steady-state voltage ratio soma/dendrite for current injected in the
dendrite, default 0.5).  Both compartments keep the measured membrane time
constant, and all conductances are scaled so the *somatic input resistance*
equals the measured one.  These two numbers are generic reduced-model
choices, not measurements; scripts/dendrite_scan.py tests their effect.

Synaptic strengths are still the somatic PSPs measured in slices: each
synapse's conductance is computed so that, delivered to its compartment, it
produces the measured somatic PSP peak.  So the data are reproduced at the
soma, and dendritic synapses are stronger locally and slower at the soma,
as in real cells.
"""
from __future__ import annotations

import numpy as np


def compartments(C_pF, tau_ms, cap_frac=0.7, attenuation=0.5):
    """(C_s, C_d, gL_s, gL_d, g_c) in pF / nS for a cell with measured total
    capacitance C and membrane time constant tau (somatic input resistance
    preserved)."""
    gL = C_pF / tau_ms
    p, A = cap_frac, attenuation
    s = 1.0 / ((1 - p) + p * A)
    gLs, gLd = (1 - p) * gL * s, p * gL * s
    gc = gLd * A / (1 - A)
    return (1 - p) * C_pF * s, p * C_pF * s, gLs, gLd, gc


def peak_response(C_s, C_d, gLs, gLd, gc, tau_syn_ms, where="d", dt=0.02, T=400.0):
    """Somatic peak (mV) and somatic 1/e decay time (ms) for a unit current
    kernel exp(-t/tau_syn) (pA) injected into the dendrite ('d') or soma ('s')."""
    n = int(T / dt)
    vs = vd = 0.0
    peak, t_peak = 0.0, 0.0
    trace = np.empty(n)
    for k in range(n):
        I = np.exp(-k * dt / tau_syn_ms)
        Id, Is = (I, 0.0) if where == "d" else (0.0, I)
        dvs = (-gLs * vs + gc * (vd - vs) + Is) / C_s
        dvd = (-gLd * vd + gc * (vs - vd) + Id) / C_d
        vs += dt * dvs
        vd += dt * dvd
        trace[k] = vs
        if vs > peak:
            peak, t_peak = vs, k * dt
    k0 = int(t_peak / dt)
    below = np.flatnonzero(trace[k0:] < peak / np.e)
    decay = below[0] * dt if below.size else np.nan
    return peak, decay


def lands_on_dendrite(target_type: str, source_type: str) -> bool:
    """target/source are population type letters (E, F, N, I)."""
    if target_type != "E":
        return False
    return source_type in ("E", "N")
