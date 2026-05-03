"""
Walsh-Hadamard rotation of weight matrices, in the QuaRot / SpinQuant sense.

Rotating a Linear's input by an orthogonal matrix Q:
    y = W @ x  ==  W @ Q^T @ (Q @ x)  ==  W' @ x'
with W' = W @ Q^T and x' = Q @ x. For Q = H / sqrt(d) (normalised Hadamard),
the columns of W' live in the WHT spectral basis, which is where the BBT
paper's per-coordinate influence is defined.

We do NOT quantize here. This module only applies the mathematically
equivalent rotation. Quantization happens downstream in autoround_bbt.py,
which sees the rotated model as its input.

Inputs with d_in not a power of two are zero-padded to the next pow2.
The rotation is fused into the weight matrix itself; the forward pass
applies the inverse rotation to the input activations so the overall
linear map is unchanged (to full precision).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .influence import _hadamard, _next_pow2


# Public re-export for the package surface
def hadamard_matrix(d: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return the normalised Hadamard H_{pow2(d)} / sqrt(pow2(d))."""
    device = device or torch.device("cpu")
    return _hadamard(_next_pow2(d), device, dtype)


class WHTRotatedLinear(nn.Linear):
    """
    nn.Linear subclass: weight is stored rotated in the WHT spectral basis,
    with self.in_features = d_pad (the padded power-of-two input dim) so
    that auto-round and other walkers see a quantizable Linear of the right
    shape. The public/original input dim is exposed as self.public_in_features.

        self.weight := W_orig_padded @ H / sqrt(d_pad)     # (out, d_pad)
        forward(x)   : pad x -> d_pad; rotate by H; (optional 1/scale); F.linear
                     equivalent to W_orig @ x_orig + bias at full precision.

    Subclassing nn.Linear is what makes auto-round walk into this module:
    SUPPORTED_LAYER_TYPES = (torch.nn.Linear, transformers.pytorch_utils.Conv1D).
    """

    input_scale: torch.Tensor
    public_in_features: int
    d_pad: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        d_pad = _next_pow2(in_features)
        # nn.Linear.__init__ creates self.weight of shape (out_features, d_pad)
        # and sets self.in_features = d_pad, self.out_features = out_features.
        super().__init__(in_features=d_pad, out_features=out_features, bias=bias,
                         device=device, dtype=dtype)
        # Public/original input dim, used by forward() to pad correctly.
        self.public_in_features = in_features
        self.d_pad = d_pad
        if bias:
            # nn.Linear initialises bias from uniform; zero it to match the
            # original Module-based version's behaviour.
            with torch.no_grad():
                self.bias.zero_()
        # Optional per-spectral-coord input scale (applied AFTER the WHT
        # rotation so the scale lives in the same basis as the weight columns).
        # Set via set_input_scale(); None means "no scaling".
        self.register_buffer(
            "input_scale",
            torch.ones(self.d_pad, device=device, dtype=dtype or torch.float32),
            persistent=True,
        )
        self._has_input_scale = False
        # Flag consumed by autoround_bbt so it knows this layer is already rotated.
        self._bbt_rotated = True

    def set_input_scale(self, scale: torch.Tensor) -> None:
        """
        Set the per-spectral-coordinate input scale (length d_pad or d_in).
        Forward will divide x_spec by this before matmul. Pair with a
        column-wise multiply of the weight by the same scale to keep math
        invariant at full precision.
        """
        s = scale.to(self.weight.device, self.weight.dtype)
        if s.numel() < self.d_pad:
            pad = torch.ones(self.d_pad - s.numel(), device=s.device, dtype=s.dtype)
            s = torch.cat([s, pad], dim=0)
        elif s.numel() > self.d_pad:
            s = s[: self.d_pad]
        self.input_scale.copy_(s)
        self._has_input_scale = True

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "WHTRotatedLinear":
        W = linear.weight.detach()                                # (out, in)
        out_features, in_features = W.shape
        d_pad = _next_pow2(in_features)
        device, dtype = W.device, W.dtype

        H = _hadamard(d_pad, device, dtype)                       # (d_pad, d_pad)

        if in_features < d_pad:
            pad = torch.zeros(
                out_features, d_pad - in_features, device=device, dtype=dtype
            )
            W = torch.cat([W, pad], dim=1)

        # Rotated weight: W_rot = W_padded @ H^T   (since H is symmetric, = W_padded @ H)
        W_rot = W @ H

        new = cls(
            in_features=in_features,
            out_features=out_features,
            bias=linear.bias is not None,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            new.weight.copy_(W_rot)
            if linear.bias is not None and new.bias is not None:
                new.bias.copy_(linear.bias.detach())
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either the public_in_features (the upstream activation
        # dim, e.g. 576 for SmolLM hidden_size) or the already-padded d_pad.
        *batch, d_in = x.shape
        if d_in == self.public_in_features and self.public_in_features != self.d_pad:
            pad = torch.zeros(
                *batch,
                self.d_pad - self.public_in_features,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([x, pad], dim=-1)
        elif d_in != self.d_pad:
            raise ValueError(
                f"Expected last dim {self.public_in_features} or {self.d_pad}, got {d_in}"
            )
        H = _hadamard(self.d_pad, x.device, x.dtype)
        x_spec = x @ H
        if self._has_input_scale:
            x_spec = x_spec / self.input_scale
        return F.linear(x_spec, self.weight, self.bias)


# -----------------------------------------------------------------------------
# Model surgery
# -----------------------------------------------------------------------------

def _set_submodule(root: nn.Module, dotted: str, new_module: nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


@torch.no_grad()
def apply_wht_rotation(
    model: nn.Module,
    skip_substrings: Tuple[str, ...] = ("lm_head", "embed_tokens", "embed_positions"),
    only_substrings: Optional[Tuple[str, ...]] = None,
    verbose: bool = False,
) -> int:
    """
    Walk the model, replace every nn.Linear with WHTRotatedLinear.

    Args:
        model: HF CausalLM or any nn.Module.
        skip_substrings: do not rotate layers whose name contains these.
        only_substrings: if given, only rotate layers whose name contains
            one of these (takes precedence over skip_substrings for filtering
            positives). Typical use: ("mlp.", "self_attn.") to cover decoder blocks.
        verbose: print each replacement.

    Returns:
        Number of Linear layers replaced.
    """
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if only_substrings and not any(s in name for s in only_substrings):
            continue
        if any(s in name for s in skip_substrings):
            continue
        targets.append((name, module))

    count = 0
    for name, linear in targets:
        rotated = WHTRotatedLinear.from_linear(linear)
        _set_submodule(model, name, rotated)
        count += 1
        if verbose:
            print(f"  rotated {name}: ({linear.in_features} -> {rotated.d_pad}) x {linear.out_features}")
    return count


# -----------------------------------------------------------------------------
# Round-trip sanity
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    d_in, d_out = 192, 96  # 192 is not a power of 2 -> pads to 256
    lin = nn.Linear(d_in, d_out)
    lin.eval()
    rotated = WHTRotatedLinear.from_linear(lin)
    rotated.eval()

    x = torch.randn(4, 7, d_in)
    with torch.no_grad():
        y_orig = lin(x)
        y_rot = rotated(x)
    err = (y_orig - y_rot).abs().mean() / y_orig.abs().mean().clamp_min(1e-9)
    print(f"Round-trip relative error: {err.item():.2e} (expect < 1e-5)")
    assert err < 1e-4, "WHTRotatedLinear is not math-equivalent to the original Linear"
    print("OK")
