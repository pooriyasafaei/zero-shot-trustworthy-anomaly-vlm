"""CERM-AD Stage I: causal visual evidence verification.

Existing trust signals in this repository answer "is the model confident?" or "do
two branches agree?". Neither answers the question that matters for a defect
detector: **does the verdict causally depend on the actual defect**, or is the
model merely confident, or reacting to something that happens to co-occur with it?

Stage I answers it by intervention. For an anomalous image we neutralise the true
defect region with normal content and re-ask the model; we then do the same edit at
matched, non-defective locations as placebos. If the model's anomaly log-odds move
much more for the real defect than for comparable irrelevant edits, the decision was
grounded in the defect.

See ``docs/CERM_AD.md`` for the method, the equations and the limitations.
"""
from .effects import CausalEffect, compute_effects
from .donor import DonorBank, DonorMatch
from .placebo import PlaceboMask, PlaceboPlan, generate_placebos
from .reconstruction import Intervention, apply_intervention, reconstruction_diagnostics

__all__ = [
    "CausalEffect", "compute_effects",
    "DonorBank", "DonorMatch",
    "PlaceboMask", "PlaceboPlan", "generate_placebos",
    "Intervention", "apply_intervention", "reconstruction_diagnostics",
]
