"""Tracel: reference-guided transcript-to-cell assignment with two-stage optimal transport.

OT1 (:func:`tracel.run_ot1`) aligns spatial cells with a scRNA-seq reference by low-rank fused
Gromov-Wasserstein transport and estimates a soft expression profile for every spatial cell.
OT2 (:func:`tracel.run_ot2_windows`, :func:`tracel.merge_ot2`) reassigns transcripts to nearby cells
with gene-wise unbalanced Sinkhorn transport and keeps assignments with sufficient relative row mass.
"""
from .ot1 import OT1Params, OT1Result, log_normalize, run_ot1
from .ot2 import OT2Params, make_windows, merge_ot2, prepare_ot2_inputs, run_ot2_windows, solve_window

__version__ = "0.1.0"

__all__ = [
    "OT1Params", "OT1Result", "log_normalize", "run_ot1",
    "OT2Params", "make_windows", "prepare_ot2_inputs", "run_ot2_windows", "solve_window", "merge_ot2",
]
