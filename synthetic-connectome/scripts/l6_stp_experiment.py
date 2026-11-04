#!/usr/bin/env python3
"""Does short-term depression contain the L6E runaway?  (open problem 5)

Lowers the inhibitory multiplier onto L6E step by step with static synapses
and with textbook depressing Tsodyks-Markram synapses (rodent literature
values; the human values come from scripts/extract_synphys.py).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synth import synthesize                      # noqa: E402
from synth.h01_fit import human_h01_column       # noqa: E402
from synth.simulate import simulate              # noqa: E402
from synth.analyze import population_rates       # noqa: E402

STP_GENERIC = {                                   # key = (target class, source class)
    ("E", "E"): dict(U=0.5,  tau_rec_ms=800, tau_fac_ms=0),
    ("I", "E"): dict(U=0.35, tau_rec_ms=400, tau_fac_ms=0),
    ("E", "I"): dict(U=0.25, tau_rec_ms=700, tau_fac_ms=0),
    ("I", "I"): dict(U=0.3,  tau_rec_ms=500, tau_fac_ms=0),
}
print("inhibition onto L6E | static: L6E  L6I  L5E  L4I  | depressing: L6E  L6I  L5E  L4I   (spikes/s)")
for g6 in (2.47, 1.5, 1.0, 0.6):
    row = f"x{g6:<5}             |"
    for stp in (None, STP_GENERIC):
        params = human_h01_column(use_calibration=True)
        params.g_per_target = np.array(params.g_per_target, float); params.g_per_target[6] = g6
        params.stp = stp
        col = synthesize(params, N_scaling=0.1, K_scaling=0.1, seed=1)
        r = population_rates(simulate(col, t_sim_ms=1500, seed=1, verbose=False), t_start_ms=500)
        row += f"  {r[6]:6.1f} {r[7]:5.1f} {r[4]:5.1f} {r[3]:5.1f}   |"
    print(row)
