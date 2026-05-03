"""
BBT-Quant: Influence-adaptive LLM weight quantization bridging the BBT paper
(Banach-Butterfly Transform) with Intel auto-round (W2A16 SignRound) and
OpenVINO for deployment on Intel NPU / Arc GPU.

Pipeline:
    calibration → per-channel BBT influence (WHT-basis activation energy)
    → WHT rotation of weights
    → AWQ-style pre-scaling with scales s_l = (1 + Inf_l)^alpha
    → auto-round W2A16 quantization (unchanged internals)
    → OpenVINO IR export
    → Windows-side NPU/GPU inference

The BBT paper defines Inf_l(f) for Boolean f; here we use a real-valued analog:
the fraction of calibration activation energy carried by spectral coordinate l.
This extension is research-grade and documented in influence.py.
"""
from .influence import compute_layer_influences, bbt_channel_scales
from .wht_rotation import (
    WHTRotatedLinear,
    apply_wht_rotation,
    hadamard_matrix,
)

__all__ = [
    "compute_layer_influences",
    "bbt_channel_scales",
    "WHTRotatedLinear",
    "apply_wht_rotation",
    "hadamard_matrix",
]
