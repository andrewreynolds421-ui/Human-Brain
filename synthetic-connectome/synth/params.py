"""Layer-resolved statistics used to synthesize a cortical column.

Everything here is a *statistical description* of cortex, not a wiring
diagram: how many neurons of each type per layer, how likely each type is
to connect to each other type, how strong and how delayed those
connections are, and how much background drive each population receives.
The synthesizer (build.py) turns these numbers into an explicit network.

Two presets are provided:

* ``rodent``  - the Potjans & Diesmann (2014) 1 mm^2 microcircuit, whose
                parameters were compiled from cat/rat anatomy and
                electrophysiology.  It has published reference firing rates,
                so it is the validation case for the pipeline.
* ``human``   - the same layer structure rescaled to human cortical
                thickness and neuron density.  These are PLACEHOLDER priors
                to be replaced by measurements from the H01 dataset
                (Shapson-Coe et al. 2024) once the data is accessible.
                It runs, but there is no reference activity to validate
                against yet.

Population order everywhere: L23E, L23I, L4E, L4I, L5E, L5I, L6E, L6I.

Reference
---------
Potjans TC, Diesmann M (2014) The cell-type specific cortical microcircuit:
relating structure and activity in a full-scale spiking network model.
Cerebral Cortex 24:785-806.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

POPULATIONS = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
LAYERS = ["L23", "L4", "L5", "L6"]
N_POP = len(POPULATIONS)
IS_EXC = np.array([True, False] * 4)

# Population names are <layer><type>.  Type letters: E excitatory, I
# inhibitory (all interneurons), F fast-spiking interneuron, N non-fast-
# spiking interneuron.  A column may use any list of such names (the
# 12-population human column uses E, F, N per layer).
INHIBITORY_TYPES = ("I", "F", "N")


def pop_type(name: str) -> str:
    return name[-1]


def pop_class(name: str) -> str:
    """'E' or 'I' for any population name."""
    return "I" if name[-1] in INHIBITORY_TYPES else "E"


def pop_layer(name: str) -> str:
    return name[:-1]


def is_exc_of(pops) -> np.ndarray:
    return np.array([pop_class(p) == "E" for p in pops])


def layers_of(pops) -> dict:
    """Ordered {layer: [population indices]} for a population list."""
    out = {}
    for i, p in enumerate(pops):
        out.setdefault(pop_layer(p), []).append(i)
    return out


def exc_inh_matrix(val_exc: float, val_inh: float, is_exc=IS_EXC) -> np.ndarray:
    """Matrix whose columns (sources) take the excitatory / inhibitory value."""
    is_exc = np.asarray(is_exc, bool)
    m = np.zeros((is_exc.size, is_exc.size))
    m[:, is_exc] = val_exc
    m[:, ~is_exc] = val_inh
    return m


# Connection probabilities, rows = target population, cols = source
# population.  Potjans & Diesmann 2014, Table 5.
CONN_PROBS = np.array(
    [[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],
     [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],
     [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],
     [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],
     [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],
     [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],
     [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
     [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443]])


@dataclass
class NeuronParams:
    """Leaky integrate-and-fire neuron with exponential current synapses."""
    E_L: float = -65.0      # mV, leak / resting potential
    V_th: float = -50.0     # mV, spike threshold
    V_reset: float = -65.0  # mV
    C_m: float = 250.0      # pF
    tau_m: float = 10.0     # ms
    tau_syn: float = 0.5    # ms
    t_ref: float = 2.0      # ms
    # optional per-population overrides (length 8 arrays); None = use scalars
    tau_m_pop: np.ndarray | None = None
    C_m_pop: np.ndarray | None = None
    V_th_pop: np.ndarray | None = None
    E_L_pop: np.ndarray | None = None
    # population-specific initial membrane potentials (reduce start-up burst)
    V0_mean: np.ndarray = field(default_factory=lambda: np.array(
        [-68.28, -63.16, -63.33, -63.45, -63.11, -61.66, -66.72, -61.43]))
    V0_std: np.ndarray = field(default_factory=lambda: np.array(
        [5.36, 4.57, 4.74, 4.94, 4.94, 4.55, 5.46, 4.48]))


@dataclass
class ColumnParams:
    name: str
    # --- cell counts -----------------------------------------------------
    full_num_neurons: np.ndarray          # per population, full scale
    # --- geometry (um) ---------------------------------------------------
    footprint_um: float                   # side of the square column
    layer_thickness_um: dict              # layer -> thickness
    # lateral spread of connection probability (Gaussian sigma, um).
    # Potjans-Diesmann probabilities are column averages; we distribute
    # them in space with this profile without changing population-level
    # in-degrees.
    lateral_sigma_um: float = 300.0
    # --- connectivity ----------------------------------------------------
    conn_probs: np.ndarray = field(default_factory=lambda: CONN_PROBS.copy())
    PSP_exc_mean: float = 0.15            # mV
    g: float = -4.0                       # inhibitory / excitatory weight ratio
    # optional per-target-population multiplier on inhibitory weights
    # (1.0 = use g as is); lets a preset rebalance E/I input per population
    g_per_target: np.ndarray | None = None
    # full PSP matrix (mV) measured directly; overrides PSP_exc_mean / g when set
    psp_matrix_override: np.ndarray | None = None
    # relative weight spread per class pair {("E","I"): 0.4, ...}; overrides weight_rel_std
    weight_rel_std_class: dict | None = None
    weight_rel_std: float = 0.1
    delay_exc_mean: float = 1.5           # ms
    delay_inh_mean: float = 0.75          # ms
    delay_rel_std: float = 0.5
    # --- external drive --------------------------------------------------
    K_ext: np.ndarray = field(default_factory=lambda: np.array(
        [1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100]))
    bg_rate: float = 8.0                  # spikes/s per external fibre
    # --- synapse model ------------------------------------------------------
    # "cuba": current-based (reference model).  "coba": conductance-based with
    # reversal potentials; a measured PSP is converted to a conductance at the
    # holding potential it was measured at, so inhibition grows with
    # depolarisation and excitation shrinks, as in real neurons.
    synapse_model: str = "cuba"
    E_exc_mV: float = 0.0
    E_inh_mV: float = -75.0
    tau_exc_ms: float = 2.0      # AMPA-like conductance decay
    tau_inh_ms: float = 6.0      # GABA-A-like conductance decay
    # optional per-target-population decay times (ms); override the two scalars
    tau_exc_pop: np.ndarray | None = None
    tau_inh_pop: np.ndarray | None = None
    V_hold_exc_mV: float = -70.0  # holding potentials of the PSP measurements
    V_hold_inh_mV: float = -55.0
    # --- spike-frequency adaptation (adaptive LIF), optional -----------------
    # per-spike increment of an outward adaptation current (pA) and its decay
    # (ms); 0 = no adaptation.  Separate values for E and I populations.
    adapt_b_exc_pA: float = 0.0
    adapt_b_inh_pA: float = 0.0
    adapt_tau_ms: float = 150.0
    # --- two-compartment pyramidal cells (coba only), see synth/dendrite.py ---
    two_compartment: bool = False
    dend_cap_frac: float = 0.7
    dend_attenuation_pop: np.ndarray | None = None   # per population (used for E only); None -> 0.5
    fs_soma_fraction: float = 1.0     # share of fast-spiking synapses onto pyramidal cells placed on the soma
    # optional per-population adaptation increments (pA); overrides the two above
    adapt_b_pop: np.ndarray | None = None
    # --- short-term plasticity (Tsodyks-Markram), optional ---------------
    # dict keyed by (target_pop, source_pop) names or by ("E"|"I", "E"|"I")
    # class pairs -> dict(U=..., tau_rec_ms=..., tau_fac_ms=...).  None = static.
    stp: dict | None = None
    # --- reference activity (None if unknown) ----------------------------
    full_mean_rates: np.ndarray | None = None
    neuron: NeuronParams = field(default_factory=NeuronParams)
    # population layout (names <layer><type>, see pop_class); default: the 8 PD2014 populations
    populations: list = field(default_factory=lambda: list(POPULATIONS))

    # derived ---------------------------------------------------------------
    @property
    def n_pop(self) -> int:
        return len(self.populations)

    @property
    def is_exc(self) -> np.ndarray:
        return is_exc_of(self.populations)

    def _lookup(self, table, target_pop, source_pop):
        """Most specific entry: exact names, then type letters (E/F/N/I), then E/I class."""
        for key in ((target_pop, source_pop), (pop_type(target_pop), pop_type(source_pop)),
                    (pop_class(target_pop), pop_class(source_pop))):
            if key in table:
                return table[key]
        return None

    def weight_rel_std_for(self, target_pop, source_pop):
        if self.weight_rel_std_class:
            v = self._lookup(self.weight_rel_std_class, target_pop, source_pop)
            if v is not None:
                return v
        return self.weight_rel_std

    @property
    def PSP_matrix_mean(self) -> np.ndarray:
        if self.psp_matrix_override is not None:
            return np.asarray(self.psp_matrix_override, float).copy()
        exc = self.is_exc
        m = exc_inh_matrix(self.PSP_exc_mean, self.PSP_exc_mean * self.g, exc)
        if self.g_per_target is not None:
            m[:, ~exc] *= np.asarray(self.g_per_target)[:, None]
        if list(self.populations) == POPULATIONS:
            m[0, 2] *= 2.0        # L4E -> L23E is twice as strong (PD2014)
        return m

    @property
    def delay_matrix_mean(self) -> np.ndarray:
        return exc_inh_matrix(self.delay_exc_mean, self.delay_inh_mean, self.is_exc)

    def stp_for(self, target_pop: str, source_pop: str):
        """STP parameters for a population pair, or None (static synapse)."""
        if not self.stp:
            return None
        return self._lookup(self.stp, target_pop, source_pop)

    def syn_tau_arrays(self):
        """Per-target (tau_exc, tau_inh) conductance decay arrays in ms."""
        te = np.asarray(self.tau_exc_pop, float) if self.tau_exc_pop is not None else np.full(self.n_pop, self.tau_exc_ms)
        ti = np.asarray(self.tau_inh_pop, float) if self.tau_inh_pop is not None else np.full(self.n_pop, self.tau_inh_ms)
        return te, ti

    def adapt_b_array(self) -> np.ndarray:
        if self.adapt_b_pop is not None:
            return np.asarray(self.adapt_b_pop, float)
        return np.where(self.is_exc, self.adapt_b_exc_pA, self.adapt_b_inh_pA).astype(float)

    def neuron_pop_arrays(self):
        """Per-population (tau_m, C_m, V_th, E_L, V_reset) arrays, scalars broadcast."""
        n = self.neuron
        def arr(v, default):
            return np.asarray(v, float) if v is not None else np.full(self.n_pop, default, float)
        E_L = arr(n.E_L_pop, n.E_L)
        return (arr(n.tau_m_pop, n.tau_m), arr(n.C_m_pop, n.C_m), arr(n.V_th_pop, n.V_th), E_L, E_L.copy())

    @property
    def layer_bounds_um(self) -> dict:
        """Depth interval (top, bottom) of each layer, pia at depth 0."""
        bounds, depth = {}, 0.0
        for layer in ["L1"] + list(layers_of(self.populations)):
            t = self.layer_thickness_um[layer]
            bounds[layer] = (depth, depth + t)
            depth += t
        return bounds

    @property
    def total_thickness_um(self) -> float:
        return float(sum(self.layer_thickness_um.values()))


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def rodent_column() -> ColumnParams:
    """Potjans & Diesmann 2014 as published (validation case)."""
    return ColumnParams(
        name="rodent_PD2014",
        full_num_neurons=np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948]),
        footprint_um=1000.0,
        # approximate rodent/cat layer thicknesses; only used for 3D placement
        layer_thickness_um={"L1": 100, "L23": 400, "L4": 300, "L5": 300, "L6": 400},
        full_mean_rates=np.array([0.943, 3.026, 4.368, 5.882, 7.733, 8.664, 1.096, 7.851]),
    )


def human_column(neurons_per_mm3: float = 16_000.0) -> ColumnParams:
    """Human-cortex preset: PLACEHOLDER until fit to H01.

    What is changed relative to the rodent preset
    ---------------------------------------------
    * total thickness ~2.8 mm (human temporal cortex) instead of ~1.5 mm
    * neuron density ~16,000 / mm^3, the order reported for the H01 sample,
      instead of ~70,000 / mm^3 implied by the rodent column
    * per-layer *proportions* and all connection probabilities are carried
      over unchanged from the rodent preset.  This is the main thing H01
      should replace: per-layer counts, E/I ratio per layer, and
      type-by-type connection probabilities.

    The slower, larger human neurons (tau_m, C_m) are also left at rodent
    values; H01 morphologies + Allen Cell Types human electrophysiology are
    the sources for those.
    """
    thickness = {"L1": 250, "L23": 1000, "L4": 300, "L5": 550, "L6": 700}
    footprint = 1000.0
    volume_mm3 = (footprint / 1000.0) ** 2 * sum(thickness.values()) / 1000.0
    total = neurons_per_mm3 * volume_mm3
    rodent = rodent_column()
    proportions = rodent.full_num_neurons / rodent.full_num_neurons.sum()
    return ColumnParams(
        name="human_placeholder",
        full_num_neurons=np.round(total * proportions).astype(int),
        footprint_um=footprint,
        layer_thickness_um=thickness,
        lateral_sigma_um=400.0,
        full_mean_rates=None,
    )


PRESETS = {"rodent": rodent_column, "human": human_column}
