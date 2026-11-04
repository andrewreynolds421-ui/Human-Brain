"""Biologically-local learning rules (three-factor / reward-modulated STDP)."""
from .rstdp import RSTDPConfig, RSTDPNetwork, train, make_patterns

__all__ = ["RSTDPConfig", "RSTDPNetwork", "train", "make_patterns"]
