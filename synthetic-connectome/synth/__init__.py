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

__all__ = ["ColumnParams", "rodent_column", "human_column", "human_h01_column", "PRESETS",
           "POPULATIONS", "synthesize", "SynthesizedColumn"]
