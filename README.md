# Human-Brain

A statistically faithful model of human cortex built from the data that exists
(H01 synapse-level cortex, Allen human physiology, macaque long-range tracing),
learning with local, reward-gated plasticity, and wired into the Warmind /
RASPUTIN assistant as a live "neural core".

| folder | what it is |
|---|---|
| `synthetic-connectome/` | the simulation package (`synth/`), experiment scripts, results and the stage-by-stage log (`README.md`), the audit (`AUDIT.md`) and working standards (`METHODS.md`) |
| `warmind-neural-core/` | the service that runs the model inside the Warmind FastAPI backend (`neural_core.py`), its tool, tests, panel and docs |

Data releases (large source files such as the Allen Synaptic Physiology
database) are attached to this repository's GitHub releases, not committed.

Start with `synthetic-connectome/AUDIT.md` for where the project stands.
