"""Statistical synthesis and simulation of cortical columns."""
from .params import ColumnParams, rodent_column, human_column, PRESETS, POPULATIONS
from .build import synthesize, SynthesizedColumn
from .h01_fit import human_h01_column
from .synphys_fit import human_synphys_column

PRESETS["human_h01"] = human_h01_column
PRESETS["human_h01_synphys"] = human_synphys_column
PRESETS["human_h01_membranes"] = lambda **kw: human_synphys_column(mode="membranes", **kw)
PRESETS["human_h01_coba"] = lambda **kw: human_synphys_column(mode="full", coba=True, **kw)
from .subtypes_fit import human_12pop_column       # noqa: E402
PRESETS["human_h01_12pop"] = human_12pop_column
def _dendritic(**kw):
    # finished configuration (stage 3d): H01 external budget, 70% of fast-spiking
    # synapses on the dendrite (H01: 2-7% of inhibitory synapses on the soma proper;
    # the soma compartment stands for soma + proximal dendrite)
    kw.setdefault("ext_budget", "h01")
    kw.setdefault("fs_soma_fraction", 0.3)
    kw.setdefault("untrusted", "rodent")
    return human_12pop_column(two_compartment=True, **kw)


PRESETS["human_h01_dendritic"] = _dendritic


def _dendritic_v2(**kw):
    """Stage 25 recalibration (4 Nov 2026), on top of the dendritic column:
    * measured subtype synapse values only from >= 10 pairs (the layer-pair rule),
      else class values — this restores excitation onto non-FS interneurons;
    * non-FS interneurons receive 0.1x the external (thalamic / long-range) fibres
      of FS cells (ASSUMED from Ji et al. 2016: thalamic input onto SST and VIP
      cells ~1/10 of that onto PV cells; Cruikshank 2007, 2010);
    * external gain 1.5 (was 2.0), deep external input x1.2 / x0.7 (was 1.6 / 1.0),
      E->FS outside L2/3 x0.3 (was 0.35): refitted so every population is within
      its reference range (results/rate_targets.json; scripts/check_rates.py).
    Results published before Stage 25 use ``human_h01_dendritic`` (unchanged)."""
    import json as _json, os as _os
    cal = _json.load(open(_os.path.join(_os.path.dirname(__file__), "..", "results", "deep_ext_calibration_v2.json")))
    kw.setdefault("ext_gain", cal["ext_gain"])
    kw.setdefault("nfs_ext_factor", cal["nfs_ext_factor"])
    kw.setdefault("subtype_min_n", cal["subtype_min_n"])
    kw.setdefault("ef_deep_factor", cal["ef_deep_factor"])
    kw.setdefault("deep_ext_gain", cal["deep_ext_gain"])
    return _dendritic(**kw)


PRESETS["human_h01_dendritic_v2"] = _dendritic_v2

__all__ = ["ColumnParams", "rodent_column", "human_column", "human_h01_column", "PRESETS",
           "POPULATIONS", "synthesize", "SynthesizedColumn"]
