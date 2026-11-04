"""Regression check: synthesize the old presets and print sums that must stay identical after code changes.
    python scripts/analysis/regress.py /path/to/repo
(kept from the session's scratch files for reproducibility; paths may need adjusting)"""
import sys, numpy as np
root = sys.argv[1]; sys.path.insert(0, root)
from synth import synthesize, PRESETS
out = {}
for name, kw in (("rodent", {}), ("human_h01_synphys", dict(verbose=False)), ("human_h01_coba", dict(verbose=False, ext_gain=0.5, adapt_b_exc_pA=60, adapt_b_inh_pA=10))):
    c = synthesize(PRESETS[name](**kw), N_scaling=0.02, K_scaling=0.05, seed=3)
    out[name] = (c.pre.sum(), c.post.sum(), float(c.weight_pA.astype(float).sum()), float(c.delay_ms.astype(float).sum()), c.DC_pA.round(6).tolist(), c.neuron["adapt_b_pop"])
print(out)
