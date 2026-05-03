"""
BBT influence for real-valued LLM weights (research extension of the paper).

Paper definition (Boolean):
    Inf_l(f) = sum over S containing l of f_hat(S)^2,
    where f: {-1,+1}^n -> {-1,+1} and f_hat is the Walsh-Hadamard transform.

Real-valued analog used here (per-spectral-coordinate activation energy):
    Given Linear layer L with input x in R^{B*T*d_in}, compute spectral
    activations h = x @ H_{d_in} / sqrt(d_in). Define the BBT influence of
    coordinate l as
        Inf_l(L) = E[h_l^2] / sum_k E[h_k^2]
    with the expectation taken over a calibration dataset. Clamped to [0,1].

This is NOT the Boolean-function influence of the paper. It is a natural
real-valued analog that preserves the paper's core intuition: "coordinates
that carry more spectral energy are more important to preserve under
quantization." See also QuaRot/AWQ per-channel importance heuristics.

The downstream consumer is autoround_bbt.py, which turns the influence
vector into AWQ-style per-channel scales s_l = (1 + Inf_l)^alpha.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Hadamard matrix (cached, normalised)
# -----------------------------------------------------------------------------

_HADAMARD_CACHE: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}


def _next_pow2(n: int) -> int:
    return 1 if n == 0 else 2 ** math.ceil(math.log2(n))


def _hadamard(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return H_d / sqrt(d) for d a power of two. Cached per (d, device, dtype)."""
    key = (d, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    H = torch.ones((1, 1), dtype=dtype, device=device)
    while H.shape[0] < d:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    H = H / math.sqrt(d)
    _HADAMARD_CACHE[key] = H
    return H


# -----------------------------------------------------------------------------
# Hook-based activation energy collector
# -----------------------------------------------------------------------------

class _SpectralEnergyHook:
    """
    Accumulates mean(h^2) over calibration batches for a single Linear layer,
    where h is the WHT of the input activations.
    """

    def __init__(self, layer_name: str, d_in: int, device: torch.device):
        self.layer_name = layer_name
        self.d_in = d_in
        self.d_pad = _next_pow2(d_in)
        self.device = device
        self.energy = torch.zeros(self.d_pad, dtype=torch.float64, device=device)
        self.count = 0

    def __call__(self, module: nn.Module, inputs, output) -> None:
        x = inputs[0]  # (..., d_in)
        if x.dim() < 2:
            return
        x_flat = x.detach().reshape(-1, x.shape[-1]).to(self.device, torch.float32)
        # Zero-pad last dim to next power of two
        if self.d_in < self.d_pad:
            pad = torch.zeros(
                x_flat.shape[0],
                self.d_pad - self.d_in,
                dtype=x_flat.dtype,
                device=x_flat.device,
            )
            x_flat = torch.cat([x_flat, pad], dim=1)
        H = _hadamard(self.d_pad, x_flat.device, x_flat.dtype)
        h = x_flat @ H  # (N, d_pad)
        self.energy += (h.double() ** 2).mean(dim=0)
        self.count += 1


@torch.no_grad()
def compute_layer_influences(
    model: nn.Module,
    calib_iter: Iterable,
    target_module_types: Tuple[type, ...] = (nn.Linear,),
    skip_substrings: Tuple[str, ...] = ("lm_head", "embed_tokens", "embed_positions"),
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run the model over calibration batches; return per-layer influence vectors.

    Args:
        model: an nn.Module (e.g. HF CausalLM). Must be in eval() and on device.
        calib_iter: yields dicts with at least "input_ids" (and optional "attention_mask").
        target_module_types: which module types to instrument (default: nn.Linear).
        skip_substrings: skip layer names containing any of these (embeddings,
            lm_head — per QuaRot/SpinQuant convention).
        device: where to accumulate energy statistics.
        max_batches: cap calibration at this many batches.

    Returns:
        dict: layer_name -> Tensor of shape (d_pad,), values in [0,1], summing to 1.
              The returned vector is *already sliced* to d_in if d_in < d_pad
              (i.e., the padded tail is dropped and the result is renormalised).
    """
    device = device or next(model.parameters()).device
    hooks: Dict[str, _SpectralEnergyHook] = {}
    handles = []

    for name, module in model.named_modules():
        if not isinstance(module, target_module_types):
            continue
        if any(s in name for s in skip_substrings):
            continue
        d_in = module.in_features if hasattr(module, "in_features") else None
        if d_in is None:
            continue
        hook = _SpectralEnergyHook(name, d_in, device)
        handles.append(module.register_forward_hook(hook))
        hooks[name] = hook

    try:
        model.eval()
        for batch_i, batch in enumerate(calib_iter):
            if max_batches is not None and batch_i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            kwargs = {}
            if "attention_mask" in batch:
                kwargs["attention_mask"] = batch["attention_mask"].to(device)
            model(input_ids, **kwargs)
    finally:
        for h in handles:
            h.remove()

    influences: Dict[str, torch.Tensor] = {}
    for name, h in hooks.items():
        if h.count == 0:
            continue
        e = (h.energy / h.count).to(torch.float32)
        e = e[: h.d_in]  # drop padded tail
        total = e.sum().clamp_min(1e-12)
        influences[name] = (e / total).clamp(0.0, 1.0)
    return influences


# -----------------------------------------------------------------------------
# AWQ-style scale derivation
# -----------------------------------------------------------------------------

def bbt_channel_scales(
    influence: torch.Tensor, alpha: float = 0.5, min_scale: float = 1e-2
) -> torch.Tensor:
    """
    Turn a per-coordinate influence vector into per-channel quantization scales.

    s_l = clamp((d_in * Inf_l + 1)^alpha, min=min_scale)

    Rationale:
        - Coordinates with Inf_l ~ 1/d (uniform) get s_l ~ 2^alpha (mild weighting)
        - Coordinates with Inf_l >> 1/d (high-energy) get amplified
        - alpha = 0 collapses to uniform scaling (= no BBT)
        - alpha = 1 matches SmoothQuant / AWQ-like aggressive scaling
        - alpha = 0.5 is the default (square-root energy scaling, matches the
          BBT paper's p_l = 1 + Inf_l form up to a constant when Inf in [0,1])

    Args:
        influence: Tensor of shape (d_in,), non-negative, roughly summing to 1.
        alpha: scaling exponent in [0, 1].
        min_scale: floor on per-channel scale to avoid zero-division.

    Returns:
        Tensor (d_in,) of positive scales, to be used as:
            W' = W * s[None, :]         (column-wise scale on weight)
            x' = x / s[None, ...]       (inverse scale on input activation)
        which leaves W @ x invariant at full precision but biases the
        quantization error budget toward low-s (= low-influence) channels.
    """
    d = influence.numel()
    s = (d * influence.clamp_min(0.0) + 1.0).pow(alpha)
    return s.clamp_min(min_scale)


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------

def save_influences(influences: Dict[str, torch.Tensor], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in influences.items()}, str(path))


def load_influences(path: Path) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in torch.load(str(path), map_location="cpu").items()}


if __name__ == "__main__":
    # Sanity check: Hadamard matrix is orthogonal, scales are positive.
    d = 64
    H = _hadamard(d, torch.device("cpu"), torch.float32)
    assert torch.allclose(H @ H.t(), torch.eye(d), atol=1e-5), "H not orthogonal"
    print(f"Hadamard H_{d}/sqrt({d}) is orthogonal: OK")

    inf = torch.full((d,), 1.0 / d)  # uniform influence
    s = bbt_channel_scales(inf, alpha=0.5)
    print(f"Uniform influence -> scales: min={s.min():.4f}, max={s.max():.4f}, "
          f"all close to sqrt(2): {torch.allclose(s, torch.full_like(s, math.sqrt(2.0)), atol=1e-4)}")

    # Spike influence at coord 0
    spike = torch.zeros(d)
    spike[0] = 1.0
    s = bbt_channel_scales(spike, alpha=0.5)
    print(f"Spike influence -> scales: [0]={s[0]:.4f} (expect sqrt(d+1)={math.sqrt(d+1):.4f}), "
          f"[1]={s[1]:.4f} (expect 1.0)")
