"""
BBT-informed wrapper around Intel auto-round.

Strategy (no monkey-patching of auto-round internals):

  1. Build a WHT-rotated copy of the model (weights live in spectral basis).
  2. Compute per-layer BBT influence from a calibration loader.
  3. Derive per-channel scales s_l = (d * Inf_l + 1)^alpha.
  4. Apply AWQ-style equivalent pre-scaling:
         W_rot <- W_rot * diag(s)       (column-wise)
     and register a forward pre-hook that divides the input by s.
     Mathematically identity; operationally biases auto-round's L2
     reconstruction error toward low-s (low-influence) channels.
  5. Call auto_round.AutoRound(model=..., scheme='W2A16', enable_alg_ext=True,
     ...).quantize_and_save(...).
  6. Save the rotation + scale metadata so the OpenVINO export step can
     fold them into the IR graph as a pre-rotation of the block input.

If auto-round is not installed, the wrapper raises a clear import error
pointing at the install instructions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from .influence import bbt_channel_scales, compute_layer_influences, save_influences
from .wht_rotation import WHTRotatedLinear, apply_wht_rotation


# -----------------------------------------------------------------------------
# Per-channel spectral-basis scaling
# -----------------------------------------------------------------------------

def _pad_scales(scales: torch.Tensor, d_pad: int, device, dtype) -> torch.Tensor:
    s = scales.to(device, dtype)
    if s.numel() < d_pad:
        pad = torch.ones(d_pad - s.numel(), device=s.device, dtype=s.dtype)
        return torch.cat([s, pad], dim=0)
    return s[:d_pad]


@torch.no_grad()
def _apply_column_scales(module: WHTRotatedLinear, scales: torch.Tensor) -> None:
    """Multiply each weight column by s (spectral-basis AWQ-style scaling)."""
    s_full = _pad_scales(scales, module.d_pad, module.weight.device, module.weight.dtype)
    module.weight.mul_(s_full[None, :])


def _set_input_inverse_scale(module: WHTRotatedLinear, scales: torch.Tensor) -> None:
    """
    Set the module's input_scale buffer so that forward() divides the
    post-WHT activation by these scales. Paired with _apply_column_scales,
    this leaves W @ x invariant at full precision.
    """
    s_full = _pad_scales(scales, module.d_pad, module.weight.device, module.weight.dtype)
    module.set_input_scale(s_full)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def _import_auto_round():
    try:
        from auto_round import AutoRound  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "auto-round is not installed. On Intel XPU:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/xpu\n"
            "  pip install auto-round\n"
            f"Underlying error: {exc}"
        )
    return AutoRound


def prepare_model_for_autoround(
    model: nn.Module,
    calib_iter: Iterable,
    alpha: float = 0.5,
    skip_substrings: Tuple[str, ...] = ("lm_head", "embed_tokens", "embed_positions"),
    only_substrings: Optional[Tuple[str, ...]] = None,
    max_calib_batches: Optional[int] = 32,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Mutate `model` in place so that auto-round sees a WHT-rotated,
    BBT-scaled copy. Returns a dict `layer_name -> scales` for provenance.

    This function does NOT call auto-round. Call it, then pass `model`
    into `auto_round.AutoRound(model=model, ...)`.

    The forward pre-hooks installed here cancel out the column scaling so
    the math is unchanged. Quantization, however, is now biased.
    """
    # 1. Influences (on unrotated model — the definition uses raw activations
    #    rotated by H inside the hook, so we don't need a rotated model yet).
    if verbose:
        print("[bbt-quant] computing per-layer influences...")
    influences = compute_layer_influences(
        model,
        calib_iter,
        skip_substrings=skip_substrings,
        max_batches=max_calib_batches,
    )
    if verbose:
        print(f"[bbt-quant] got influences for {len(influences)} layers")

    # 2. Rotate every targeted Linear in-place.
    if verbose:
        print("[bbt-quant] applying WHT rotation...")
    n_rotated = apply_wht_rotation(
        model,
        skip_substrings=skip_substrings,
        only_substrings=only_substrings,
        verbose=False,
    )
    if verbose:
        print(f"[bbt-quant] rotated {n_rotated} linear layers")

    # 3. Apply per-channel scales on the rotated weights + install input hooks.
    scales_map: Dict[str, torch.Tensor] = {}
    rotated_modules: Dict[str, WHTRotatedLinear] = {
        n: m for n, m in model.named_modules() if isinstance(m, WHTRotatedLinear)
    }
    for name, module in rotated_modules.items():
        inf = influences.get(name)
        if inf is None:
            continue
        s = bbt_channel_scales(inf, alpha=alpha).to(module.weight.device)
        _apply_column_scales(module, s)
        _set_input_inverse_scale(module, s)
        scales_map[name] = s.detach().cpu()

    if verbose:
        print(f"[bbt-quant] scaled {len(scales_map)} layers (alpha={alpha})")
    return scales_map


def quantize_with_autoround(
    model: nn.Module,
    tokenizer,
    output_dir: Path,
    bits: int = 2,
    group_size: int = 128,
    sym: bool = True,
    enable_alg_ext: bool = True,
    device: str = "auto",
    n_samples: int = 128,
    seqlen: int = 2048,
    format: str = "auto_round",
    extra_kwargs: Optional[Dict] = None,
) -> Path:
    """
    Thin wrapper around auto_round.AutoRound(...).quantize_and_save(...).

    Args:
        model: already rotated+scaled by prepare_model_for_autoround.
        tokenizer: matching HuggingFace tokenizer.
        output_dir: where the quantized model will be written.
        bits: 2 for ternary (W2A16), 4 for W4A16.
        enable_alg_ext: turn on auto-round's improved INT2 algorithm (2025/08).
        format: one of {"auto_round", "auto_gptq", "auto_awq", "gguf"}.

    Returns:
        Path to `output_dir`.
    """
    AutoRound = _import_auto_round()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kw = dict(
        bits=bits,
        group_size=group_size,
        sym=sym,
        nsamples=n_samples,
        seqlen=seqlen,
        device=device,
    )
    iters_override = getattr(run_bbt_autoround, "_iters_override", None)
    if iters_override is not None:
        kw["iters"] = int(iters_override)
    # auto-round's public CLI flag is --enable_alg_ext; the Python API
    # exposes it as a constructor kwarg in recent versions. Best-effort.
    if enable_alg_ext:
        kw["enable_alg_ext"] = True
    if extra_kwargs:
        kw.update(extra_kwargs)

    ar = AutoRound(model=model, tokenizer=tokenizer, **kw)
    ar.quantize_and_save(output_dir=str(output_dir), format=format)
    return output_dir


# -----------------------------------------------------------------------------
# BBT prep that keeps layers as plain nn.Linear (so auto-round walks them)
# -----------------------------------------------------------------------------

def _bbt_prep_inplace_plain_linear(
    model: nn.Module,
    influences: Dict[str, torch.Tensor],
    *,
    alpha: float = 0.5,
    skip_substrings: Tuple[str, ...] = ("lm_head", "embed_tokens", "embed_positions"),
    only_substrings: Optional[Tuple[str, ...]] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    For each target nn.Linear, replace its weight with the WHT-rotated and
    BBT-column-scaled version (shape (out_features, d_pad)), set in_features
    to d_pad, and install a forward_pre_hook that pads the input to d_pad,
    rotates it through the normalized Hadamard, and divides by the BBT input
    scale. The module remains a plain nn.Linear (type-equal), so auto-round's
    strict ``type(m) == nn.Linear`` checks pass at both quantize and export.

    Math: y = (pad(x) @ H * 1/s) @ W_scaled^T where W_scaled = (W_padded @ H) * s.
    At full precision this equals the original W @ x; quantizing W_scaled
    biases the rounding error toward low-influence channels.
    """
    from .influence import bbt_channel_scales, _hadamard, _next_pow2

    scales_map: Dict[str, torch.Tensor] = {}
    targets: list[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(s in name for s in skip_substrings):
            continue
        if only_substrings is not None and not any(s in name for s in only_substrings):
            continue
        targets.append((name, mod))

    for name, lin in targets:
        inf = influences.get(name)
        if inf is None:
            continue
        device, dtype = lin.weight.device, lin.weight.dtype
        in_features = lin.in_features
        out_features = lin.out_features
        d_pad = _next_pow2(in_features)
        H = _hadamard(d_pad, device, dtype)  # normalized
        # Pad weight to d_pad columns
        with torch.no_grad():
            W = lin.weight.detach()
            if in_features < d_pad:
                pad = torch.zeros(out_features, d_pad - in_features, device=device, dtype=dtype)
                W = torch.cat([W, pad], dim=1)
            W_rot = W @ H  # (out, d_pad), in spectral basis
            # Scales s in spectral basis (use d_pad-wide scaling vector)
            inf_pad = torch.zeros(d_pad, dtype=inf.dtype, device=inf.device)
            inf_pad[: inf.numel()] = inf
            s = bbt_channel_scales(inf_pad, alpha=alpha).to(device, dtype)
            W_scaled = W_rot * s[None, :]  # column-wise multiply
            # Replace weight in-place
            new_weight = nn.Parameter(W_scaled, requires_grad=lin.weight.requires_grad)
            lin.weight = new_weight
            lin.in_features = d_pad

        # forward_pre_hook: pad input to d_pad, apply WHT, divide by scale
        public_in = in_features
        # Capture scale and d_pad in closure (avoid relying on nn.Linear attrs)
        s_buf = s.detach().clone()

        def _make_hook(public_in, d_pad, s_buf, H_cache):
            def hook(_mod, inputs):
                (x,) = inputs
                if x.shape[-1] == public_in:
                    if d_pad != public_in:
                        zeros = torch.zeros(*x.shape[:-1], d_pad - public_in,
                                            dtype=x.dtype, device=x.device)
                        x = torch.cat([x, zeros], dim=-1)
                elif x.shape[-1] != d_pad:
                    raise ValueError(
                        f"BBT layer expected last dim {public_in} or {d_pad}, "
                        f"got {x.shape[-1]}"
                    )
                Hf = H_cache.to(x.device, x.dtype)
                x = (x @ Hf) / s_buf.to(x.device, x.dtype)
                return (x,)
            return hook

        lin.register_forward_pre_hook(_make_hook(public_in, d_pad, s_buf, H))
        scales_map[name] = s_buf.cpu()

    if verbose:
        print(f"[bbt-quant-v2] prepped {len(scales_map)} plain nn.Linear "
              f"layers with rotation+scale (alpha={alpha})")
    return scales_map


# ---------------------------------------------------------------------------
# Module-graph helpers (used by no_rotation + MoE-aware variants)
# ---------------------------------------------------------------------------

# Common naming variants across architectures we want to support:
#   Llama / Qwen2.5 / Qwen3:  attn.q_proj/k_proj/v_proj/o_proj, mlp.gate_proj/up_proj/down_proj
#   Qwen1.5-MoE / Qwen2-MoE:  mlp.{gate,experts[i].{gate_proj,up_proj,down_proj},shared_expert.{...},shared_expert_gate}
#   DeepSeek-V4:              attn.{wq_a,wq_b,wkv,wo_a,wo_b}, ffn.{gate (Parameter),experts[i].{w1,w2,w3},shared_experts.{...}}
_MLP_GATE_NAMES = ("gate_proj", "w1")        # silu-gated branch input projection
_MLP_UP_NAMES   = ("up_proj", "w3")          # multiplicative branch input projection
_MLP_DOWN_NAMES = ("down_proj", "w2")        # output projection from intermediate
_ATTN_INPUT_NAMES = ("q_proj", "k_proj", "v_proj", "wq_a", "wkv")
_ATTN_OUTPUT_NAMES = ("o_proj", "wo_a", "wo_b")


def _first_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return n, v
    return None, None


def _module_name(parent: nn.Module, child: nn.Module) -> Optional[str]:
    """Dotted path from `parent` down to `child` (first match), e.g.
    'mlp.experts.5.down_proj'. Returns None if `child` is not a descendant."""
    for name, m in parent.named_modules():
        if m is child:
            return name
    return None


def _get_block_mlp(block):
    """Return the block's MLP/FFN sub-module (mlp or ffn naming)."""
    return getattr(block, "mlp", None) or getattr(block, "ffn", None)


class _ScaleTarget:
    """Adapter unifying nn.Linear and modern HF fused MoE Parameters under one
    `mul_input_columns(s)` interface so the absorption logic can treat both
    uniformly. The underlying tensor's storage is mutated in place."""
    def __init__(self, label: str, mod_or_param, kind: str, in_dim: int):
        self.label = label
        self.target = mod_or_param   # nn.Linear | nn.Parameter | the experts module holding it
        self.kind = kind             # "linear" | "fused_3d_lastdim_in"
        self.in_features = in_dim
    @property
    def device(self):
        if self.kind == "linear":
            return self.target.weight.device
        return self.target.device
    @property
    def dtype(self):
        if self.kind == "linear":
            return self.target.weight.dtype
        return self.target.dtype
    @torch.no_grad()
    def mul_input_columns(self, s: torch.Tensor) -> None:
        s = s.to(self.device, self.dtype)
        if self.kind == "linear":
            self.target.weight.mul_(s[None, :])
        elif self.kind == "linear_weight":
            # Raw 2D Parameter used as the W in F.linear(x, W). Layout (out, in).
            self.target.mul_(s[None, :])
        elif self.kind == "fused_3d_lastdim_in":
            # Shape (n_experts, out_features, in_features); input dim is last.
            self.target.mul_(s.view(1, 1, -1))


def _collect_mlp_input_consumers(mlp):
    """
    All weight tensors that consume the pre-MLP norm output. Returns a list
    of `_ScaleTarget`. Covers:
      - standard SwiGLU on mlp itself (gate_proj/up_proj or w1/w3)
      - routed experts as ModuleList of Expert(Linear) — older HF / V4-style
      - routed experts as fused 3D Parameter (modern HF Qwen2MoE: gate_up_proj
        shape (n_experts, 2*moe_inter, hidden))
      - shared expert (single Module or ModuleList)
      - router gate (single Linear)
      - shared_expert_gate
    """
    if mlp is None:
        return []
    out: list[_ScaleTarget] = []

    def _add_branches(prefix, container):
        for nm in _MLP_GATE_NAMES + _MLP_UP_NAMES:
            t = getattr(container, nm, None)
            if isinstance(t, nn.Linear):
                lbl = f"{prefix}.{nm}" if prefix else nm
                out.append(_ScaleTarget(lbl, t, "linear", t.in_features))

    # Standard SwiGLU directly on mlp
    _add_branches("", mlp)
    # Routed experts: handle BOTH layouts.
    experts = getattr(mlp, "experts", None)
    if isinstance(experts, nn.ModuleList):
        for i, exp in enumerate(experts):
            if exp is None:
                continue
            _add_branches(f"experts.{i}", exp)
    elif experts is not None and not isinstance(experts, nn.ModuleList):
        # Modern fused HF MoE (Qwen2MoeExperts, etc.). The fused gate+up has
        # shape (n_experts, 2*moe_inter, hidden); we scale along the trailing
        # `hidden` axis. A separate per-expert down_proj Parameter exists too,
        # but that's an OUTPUT-side absorption target, handled in `_collect_mlp_down_pairs`.
        for fused_attr in ("gate_up_proj", "gate_proj", "w1"):
            p = getattr(experts, fused_attr, None)
            if isinstance(p, nn.Parameter) and p.dim() == 3:
                out.append(_ScaleTarget(
                    f"experts.{fused_attr}", p, "fused_3d_lastdim_in", p.shape[-1]))
    # Shared expert (single Module or ModuleList)
    for shared_attr in ("shared_expert", "shared_experts"):
        shared = getattr(mlp, shared_attr, None)
        if shared is None:
            continue
        if isinstance(shared, nn.ModuleList):
            for j, sh in enumerate(shared):
                if sh is not None:
                    _add_branches(f"{shared_attr}.{j}", sh)
        else:
            _add_branches(shared_attr, shared)
    # MoE router gate. nn.Linear (older HF), or a custom router Module with a
    # raw 2D `.weight` Parameter used via F.linear (modern HF Qwen2MoE's
    # Qwen2MoeTopKRouter, DeepSeek-V4 Gate).
    gate = getattr(mlp, "gate", None)
    if isinstance(gate, nn.Linear):
        out.append(_ScaleTarget("gate", gate, "linear", gate.in_features))
    elif gate is not None:
        gw = getattr(gate, "weight", None)
        if isinstance(gw, nn.Parameter) and gw.dim() == 2:
            out.append(_ScaleTarget("gate", gw, "linear_weight", gw.shape[-1]))
    sge = getattr(mlp, "shared_expert_gate", None)
    if isinstance(sge, nn.Linear):
        out.append(_ScaleTarget("shared_expert_gate", sge, "linear", sge.in_features))
    return out


def _collect_mlp_down_pairs(mlp):
    """All (up_proj-equivalent, down_proj-equivalent) linear pairs (one per
    expert + shared expert + base mlp) for the down_proj absorption trick."""
    if mlp is None:
        return []
    pairs: list[tuple[str, nn.Linear, nn.Linear]] = []

    def _try_pair(prefix, container):
        _, up = _first_attr(container, _MLP_UP_NAMES)
        _, down = _first_attr(container, _MLP_DOWN_NAMES)
        if isinstance(up, nn.Linear) and isinstance(down, nn.Linear):
            pairs.append((prefix or "mlp", up, down))

    _try_pair("", mlp)
    experts = getattr(mlp, "experts", None)
    if isinstance(experts, nn.ModuleList):
        for i, exp in enumerate(experts):
            if exp is not None:
                _try_pair(f"experts.{i}", exp)
    for shared_attr in ("shared_expert", "shared_experts"):
        shared = getattr(mlp, shared_attr, None)
        if shared is None:
            continue
        if isinstance(shared, nn.ModuleList):
            for j, sh in enumerate(shared):
                if sh is not None:
                    _try_pair(f"{shared_attr}.{j}", sh)
        else:
            _try_pair(shared_attr, shared)
    return pairs


def _bbt_prep_no_rotation(
    model: nn.Module,
    influences: Dict[str, torch.Tensor],
    *,
    alpha: float = 0.5,
    skip_substrings: Tuple[str, ...] = ("lm_head", "embed_tokens", "embed_positions"),
    only_substrings: Optional[Tuple[str, ...]] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    AWQ-style BBT scaling without WHT rotation. For Llama-family models:
      * input_layernorm feeds q_proj / k_proj / v_proj (shared input)
      * post_attention_layernorm feeds gate_proj / up_proj (shared input)
      * o_proj and down_proj have no upstream LayerNorm to absorb into;
        they're left at vanilla quant.

    For each absorption group:
      W_group_member.weight *= diag(s)
      upstream_norm.weight /= s
    Math is invariant. Auto-round then quantizes the scaled W's; quant noise
    is biased toward small-s (low-influence) columns.
    """
    from .influence import bbt_channel_scales

    scales_map: Dict[str, torch.Tensor] = {}

    # Find Llama-style transformer blocks
    layers_container = None
    for name, mod in model.named_modules():
        if name.endswith(".layers") and isinstance(mod, nn.ModuleList):
            layers_container = mod
            break
    if layers_container is None:
        if verbose:
            print("[bbt-quant-no-rot] no .layers ModuleList found; falling back to "
                  "weight-only column scaling without norm absorption")
        return _bbt_prep_no_rot_fallback(model, influences, alpha,
                                         skip_substrings, only_substrings, verbose)

    n_blocks = 0
    for block_idx, block in enumerate(layers_container):
        n_blocks += 1
        # Group 1: input_layernorm -> input-side attention linears. Standard
        # set is {q,k,v}_proj. Laguna adds `g_proj` (per-head softplus gate on
        # input_layernorm output, multiplying attn_output before o_proj) —
        # without scaling g_proj's input columns, the softplus(W·x/s) leaks
        # the per-channel scaling through a non-linearity and breaks invariance.
        # Exclude o_proj/dense/out_proj/wo_* explicitly — those consume the
        # post-attention output, not input_layernorm.
        _INPUT_ATTN_NAMES = ("q_proj", "k_proj", "v_proj", "g_proj")
        norm = getattr(block, "input_layernorm", None)
        attn = getattr(block, "self_attn", None)
        targets_attn = []
        if attn is not None and norm is not None and hasattr(norm, "weight"):
            hidden = norm.weight.numel()
            for nm in _INPUT_ATTN_NAMES:
                t = getattr(attn, nm, None)
                if isinstance(t, nn.Linear) and t.in_features == hidden:
                    targets_attn.append((nm, t))
        if norm is not None and hasattr(norm, "weight") and targets_attn:
            # Use influence from q_proj as the shared input influence
            base_name = f"model.layers.{block_idx}.self_attn.q_proj"
            inf = influences.get(base_name)
            if inf is not None and inf.numel() == norm.weight.numel():
                device, dtype = norm.weight.device, norm.weight.dtype
                s = bbt_channel_scales(inf, alpha=alpha).to(device, dtype)
                with torch.no_grad():
                    norm.weight.div_(s)  # absorb 1/s into norm
                    for nm, t in targets_attn:
                        t.weight.mul_(s[None, :])  # column-wise * s
                        scales_map[f"model.layers.{block_idx}.self_attn.{nm}"] = s.detach().cpu()

        # Group 2: post_attention_layernorm -> gate/up branches of every MLP/MoE
        # consumer. Standard MLP yields (gate_proj, up_proj) directly on `mlp`;
        # MoE yields one pair per expert + shared expert + the router gate.
        norm2 = getattr(block, "post_attention_layernorm", None) or \
                getattr(block, "ffn_norm", None)
        mlp = _get_block_mlp(block)
        consumers = _collect_mlp_input_consumers(mlp)
        if norm2 is not None and hasattr(norm2, "weight") and consumers:
            # Pick a representative influence. Prefer a Linear-based consumer
            # whose name we can look up in `influences`. Skip "gate" labels
            # (router/shared_expert_gate) — their narrow output dimension makes
            # them unrepresentative of MLP-input statistics. For fused-3D
            # Parameter experts (modern HF MoE) we can't compute influence
            # directly via forward hook, so we rely on the shared_expert's
            # gate_proj influence as a proxy (same upstream input distribution).
            inf = None
            for tgt in consumers:
                if tgt.kind != "linear" or tgt.label.endswith("gate"):
                    continue
                mname = _module_name(block, tgt.target)
                if mname is None:
                    continue
                key = f"model.layers.{block_idx}.{mname}"
                inf = influences.get(key)
                if inf is not None:
                    break
            if inf is not None and inf.numel() == norm2.weight.numel():
                device, dtype = norm2.weight.device, norm2.weight.dtype
                s = bbt_channel_scales(inf, alpha=alpha).to(device, dtype)
                with torch.no_grad():
                    norm2.weight.div_(s)
                    for tgt in consumers:
                        tgt.mul_input_columns(s)
                        if tgt.kind == "linear":
                            mname = _module_name(block, tgt.target)
                            if mname is not None:
                                key = f"model.layers.{block_idx}.{mname}"
                                scales_map[key] = s.detach().cpu()
                        else:
                            scales_map[f"model.layers.{block_idx}.mlp.{tgt.label}"] = (
                                s.detach().cpu())

        # Group 3: down_proj absorption via up_proj output rows. One pair per
        # expert + shared expert + base mlp (whichever exists).
        for label, up, down in _collect_mlp_down_pairs(mlp):
            base_name = f"model.layers.{block_idx}.{_module_name(block, down)}"
            inf = influences.get(base_name)
            if inf is None or inf.numel() != down.in_features:
                continue
            device, dtype = down.weight.device, down.weight.dtype
            s = bbt_channel_scales(inf, alpha=alpha).to(device, dtype)
            with torch.no_grad():
                up.weight.div_(s[:, None])
                if up.bias is not None:
                    up.bias.div_(s)
                down.weight.mul_(s[None, :])
                scales_map[base_name] = s.detach().cpu()

        # Group 4: o_proj absorption via v_proj output rows.
        # Only sound when num_attention_heads == num_key_value_heads (no GQA),
        # because GQA replicates V across heads and the per-channel scale on
        # v_proj output gets repeated 1:N, which o_proj's per-channel column
        # scale doesn't match cleanly.
        if attn is not None:
            v = getattr(attn, "v_proj", None)
            o = getattr(attn, "o_proj", None)
            cfg = getattr(model, "config", None)
            non_gqa = (
                cfg is not None
                and getattr(cfg, "num_attention_heads", None) ==
                    getattr(cfg, "num_key_value_heads", None)
            )
            if isinstance(v, nn.Linear) and isinstance(o, nn.Linear) and non_gqa:
                base_name = f"model.layers.{block_idx}.self_attn.o_proj"
                inf = influences.get(base_name)
                if inf is not None and inf.numel() == o.in_features:
                    device, dtype = o.weight.device, o.weight.dtype
                    s = bbt_channel_scales(inf, alpha=alpha).to(device, dtype)
                    with torch.no_grad():
                        v.weight.div_(s[:, None])
                        if v.bias is not None:
                            v.bias.div_(s)
                        o.weight.mul_(s[None, :])
                        scales_map[base_name] = s.detach().cpu()

    if verbose:
        print(f"[bbt-quant-norot-absorb] BBT-scaled {len(scales_map)} layers across "
              f"{n_blocks} blocks with norm absorption (alpha={alpha})")
    return scales_map


def _bbt_prep_spectral(
    model: nn.Module,
    influences: Dict[str, torch.Tensor],
    *,
    alpha: float = 0.5,
    skip_substrings: Tuple[str, ...] = ("lm_head", "embed_tokens", "embed_positions"),
    only_substrings: Optional[Tuple[str, ...]] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Spectral BBT: WHT-rotate weights into spectral basis, column-scale by
    influence-derived s, tag the linear so the (monkey-patched) auto-round
    wrapper's linear_forward can do the matching pad+rotate+scale on the
    input before F.linear.

    Must be paired with `patch_autoround_wrapper_for_bbt()` before
    AutoRound(...) is instantiated.
    """
    from .influence import bbt_channel_scales, _hadamard, _next_pow2

    scales_map: Dict[str, torch.Tensor] = {}

    # Detect post-projection norms (e.g. Qwen3's q_norm/k_norm). If present,
    # applying spectral BBT to q_proj/k_proj distorts the magnitude in a way
    # that the subsequent RMSNorm amplifies (the RMSNorm divisor depends on
    # mean(y^2) which BBT scaling redistributes asymmetrically). Skip those
    # layers when post-projection norms are detected.
    skip_qk = False
    for _, blk in model.named_modules():
        attn = getattr(blk, "self_attn", None)
        if attn is not None and (hasattr(attn, "q_norm") or hasattr(attn, "k_norm")):
            skip_qk = True
            break
    if skip_qk:
        # Skip the entire attention block in spectral mode for Qwen3-style
        # architectures: q/k go through Q_norm/K_norm which amplify the
        # asymmetric per-channel quant noise BBT redistribution introduces,
        # and o_proj's input depends on V multiplied by an attention pattern
        # that's hard to keep math-invariant under spectral rotation.
        skip_substrings = tuple(skip_substrings) + ("q_proj", "k_proj", "v_proj", "o_proj")
        if verbose:
            print("[bbt-quant-spectral] detected Q_norm/K_norm in attention; "
                  "skipping spectral BBT for q/k/v/o_proj (MLP-only BBT)")

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(s in name for s in skip_substrings):
            continue
        if only_substrings is not None and not any(s in name for s in only_substrings):
            continue
        inf = influences.get(name)
        if inf is None:
            continue
        device, dtype = mod.weight.device, mod.weight.dtype
        in_features = mod.in_features
        out_features = mod.out_features
        d_pad = _next_pow2(in_features)
        H = _hadamard(d_pad, device, dtype)
        # Pad influence to d_pad
        inf_pad = torch.zeros(d_pad, dtype=inf.dtype, device=inf.device)
        inf_pad[: inf.numel()] = inf
        s = bbt_channel_scales(inf_pad, alpha=alpha).to(device, dtype)
        with torch.no_grad():
            W = mod.weight.detach()
            if in_features < d_pad:
                pad = torch.zeros(out_features, d_pad - in_features, device=device, dtype=dtype)
                W = torch.cat([W, pad], dim=1)
            W_rot = W @ H
            W_scaled = W_rot * s[None, :]
            mod.weight = nn.Parameter(W_scaled, requires_grad=mod.weight.requires_grad)
            mod.in_features = d_pad
        # Tag for the patched wrapper (post-wrap forwards)
        mod._bbt_pad_dim = d_pad
        mod._bbt_public_in = in_features
        mod._bbt_input_scale = s.detach()
        mod._bbt_hadamard = H.detach()
        # Also install a forward_pre_hook for the calibration pass that
        # auto-round runs BEFORE wrapping the layer.
        s_buf = s.detach()
        H_buf = H.detach()
        public_in_local = in_features
        d_pad_local = d_pad

        def _make_calib_hook(public_in, d_pad, s_buf, H_buf):
            def hook(_mod, inputs):
                (x,) = inputs
                if x.shape[-1] == public_in and d_pad != public_in:
                    zeros = torch.zeros(*x.shape[:-1], d_pad - public_in,
                                        dtype=x.dtype, device=x.device)
                    x = torch.cat([x, zeros], dim=-1)
                elif x.shape[-1] != d_pad:
                    raise ValueError(
                        f"BBT spectral expected last dim {public_in} or {d_pad}, "
                        f"got {x.shape[-1]}"
                    )
                Hf = H_buf.to(x.device, x.dtype)
                sf = s_buf.to(x.device, x.dtype)
                x = (x @ Hf) / sf
                return (x,)
            return hook

        mod.register_forward_pre_hook(
            _make_calib_hook(public_in_local, d_pad_local, s_buf, H_buf)
        )
        scales_map[name] = s.detach().cpu()

    if verbose:
        print(f"[bbt-quant-spectral] prepped {len(scales_map)} layers (alpha={alpha})")
    return scales_map


def patch_autoround_for_xpu_memory():
    """
    Inner-loop port of MegaTrain's host-streaming pattern
    (https://arxiv.org/abs/2604.05091, github.com/DLYuanGod/MegaTrain)
    applied to auto-round's per-block tuning loop on Arc B580 (12 GB VRAM).

    What we tried and what actually moves the needle:

    1. **Boundary CPU eviction** (`block.to("cpu")` + `clear_memory()` after
       each `_quantize_block`): keeps already-tuned blocks off the
       accelerator. Verified working but insufficient on its own — the
       working set inside _quantize_block alone exceeds the SYCL spill
       threshold for SmolLM-360M-shaped blocks.

    2. **Stateless GPU layer template** (one GPU shell reused across blocks):
       structurally clean, but auto-round writes quantization metadata
       (`.scale`, `.qweight`, etc.) directly onto the block's submodules and
       its `_immediate_pack` step looks them up by reference identity from
       `model.layers[i]`. A template substitution breaks that contract;
       upstream changes to auto-round would be needed.

    3. **Reduced inner-iter batch** (`batch_size=2`, `gradient_accumulate_steps=4`):
       the actually-effective MegaTrain-aligned trick for our setup. Auto-round
       processes `batch_size` calibration samples per backward pass; with
       batch_size=2 the per-iter activation working set is 4x smaller and
       fits below the spill threshold while keeping the same effective
       calibration via gradient accumulation. This is exactly MegaTrain's
       "stream the activations through smaller compute slices" idea, just
       expressed via auto-round's existing knobs.

    What this patch does today:
      - Sets `batch_size=2, gradient_accumulate_steps=4` on the BaseCompressor
        instance at __init__ (only when it would otherwise default high).
      - Adds `block.to("cpu")` + `clear_memory()` after each block.

    Idempotent.
    """
    from auto_round.compressors import base as ar_base
    from auto_round.utils.device import clear_memory

    if getattr(ar_base.BaseCompressor, "_bbt_mem_patched", False):
        return
    orig_qb = ar_base.BaseCompressor._quantize_block
    orig_init = ar_base.BaseCompressor.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # Reduce inner-iter activation pressure. For spectral BBT mode the
        # rotation+scale work doubles the activation footprint, so we go
        # smaller (batch_size=1). Vanilla and norot stay at batch_size=4.
        try:
            mode = getattr(run_bbt_autoround, "_rotation_mode", "no_rotation")
            # Any rotation mode (spectral, spectral_pca, spectral_pca_2d) doubles
            # the per-iter activation footprint via the rotation+scale ops, so we
            # halve batch_size to stay below the SYCL spill threshold on 12 GB
            # consumer GPUs. Vanilla / norot stay at batch_size=4.
            rotation_modes = {"spectral", "spectral_pca", "spectral_pca_2d"}
            target_bs = 1 if mode in rotation_modes else 4
            if getattr(self, "batch_size", 8) > target_bs:
                self.batch_size = target_bs
        except Exception:
            pass

    def patched_qb(self, block, input_ids, input_others, *args, **kwargs):
        result = orig_qb(self, block, input_ids, input_others, *args, **kwargs)
        try:
            block.to("cpu")
        except Exception:
            pass
        try:
            clear_memory(device_list=self.device_list)
        except Exception:
            pass
        return result

    ar_base.BaseCompressor.__init__ = patched_init
    ar_base.BaseCompressor._quantize_block = patched_qb
    ar_base.BaseCompressor._bbt_mem_patched = True


def patch_autoround_wrapper_for_bbt():
    """
    Monkey-patch auto_round.wrapper.WrapperLinear.linear_forward so that
    BBT-tagged linears do their pad+rotate+inverse_scale on the input before
    the F.linear call. Caches device/dtype-cast H and scale tensors on each
    layer to avoid the per-iter allocation churn that fragments the SYCL
    allocator (the cause of the Qwen2.5-1.5B spectral spill we hit).
    Idempotent.
    """
    from auto_round import wrapper as ar_wrapper
    if getattr(ar_wrapper.WrapperLinear, "_bbt_patched", False):
        return
    import torch.nn.functional as F

    def patched_linear_forward(self, x, weight, bias):
        d_pad = getattr(self.orig_layer, "_bbt_pad_dim", None)
        if d_pad is not None:
            public_in = self.orig_layer._bbt_public_in
            if x.shape[-1] == public_in and d_pad != public_in:
                zeros = torch.zeros(*x.shape[:-1], d_pad - public_in,
                                    dtype=x.dtype, device=x.device)
                x = torch.cat([x, zeros], dim=-1)
            elif x.shape[-1] != d_pad:
                raise ValueError(
                    f"BBT layer expected last dim {public_in} or {d_pad}, "
                    f"got {x.shape[-1]}"
                )
            # Cache H/s casts to (device, dtype) ONCE per layer to avoid
            # 200 iters * 7 linears * N blocks worth of fresh allocations.
            cache_key = (str(x.device), x.dtype)
            cache = getattr(self.orig_layer, "_bbt_dev_cache", None)
            if cache is None or cache.get("key") != cache_key:
                cache = {
                    "key": cache_key,
                    "H": self.orig_layer._bbt_hadamard.to(x.device, x.dtype),
                    "s": self.orig_layer._bbt_input_scale.to(x.device, x.dtype),
                }
                self.orig_layer._bbt_dev_cache = cache
            x = (x @ cache["H"]) / cache["s"]
        return F.linear(x, weight, bias)

    ar_wrapper.WrapperLinear.linear_forward = patched_linear_forward
    ar_wrapper.WrapperLinear._bbt_patched = True


class MatrixGammaRMSNorm(nn.Module):
    """
    RMSNorm with matrix-valued gamma per head, replacing q_norm/k_norm in
    Qwen3-style attention. Construction:
        original: out[h] = gamma * (y[h] / RMS(y[h]))
        prepped : weights are W_q_rot = U[h]^T W_q[h], so y'[h] = U[h]^T y[h];
                  with Gamma[h] = diag(gamma) @ U[h], forward gives:
                      out[h] = Gamma[h] @ (y'[h] / RMS(y'[h]))
                             = diag(gamma) U[h] @ U[h]^T (y[h] / RMS)
                             = gamma * y[h] / RMS
                             = original out[h]

    The rotation U[h] cancels out of the RMSNorm output. RoPE + attention then
    receive the SAME q/k vectors as the unmodified model, so math is preserved
    end-to-end (incl. position-dependent RoPE rotations that don't commute with
    arbitrary U[h]). The benefit is purely in the basis where auto-round
    rounds W_q rows: rotated rows have a different per-row dynamic range, so
    INT2 quant noise distributes through different output channels than vanilla.

    Stores `gamma_matrix` of shape (n_groups, head_dim, head_dim). For non-GQA,
    n_groups == n_heads. For GQA, q-side uses one Gamma per q-head (replicated
    within a kv-group), k-side uses one Gamma per kv-head.
    """

    def __init__(self, gamma_matrix: torch.Tensor, gamma_vector_orig: torch.Tensor,
                 eps: float = 1e-6):
        super().__init__()
        # gamma_matrix: (n_groups, head_dim, head_dim). Used by forward.
        # gamma_vector_orig: (head_dim,). Carried on `.weight` so it serializes
        # to the same param name a vanilla Qwen3RMSNorm would write — at dequant
        # we drop gamma_matrix and the resulting state_dict loads back as a plain
        # Qwen3RMSNorm with the correct (original) gamma vector.
        self.gamma_matrix = nn.Parameter(gamma_matrix.contiguous(),
                                          requires_grad=False)
        self.weight = nn.Parameter(gamma_vector_orig.detach().contiguous(),
                                    requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, n_heads, head_dim) per Qwen3 attention layout.
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # out[b, t, h, d] = sum_e gamma_matrix[h, d, e] * normed[b, t, h, e]
        gamma = self.gamma_matrix.to(normed.dtype)
        out = torch.einsum("bthe,hde->bthd", normed, gamma)
        return out.to(input_dtype)


@torch.no_grad()
def _calibrate_qk_pca(
    model: nn.Module,
    calib_iter: Iterable,
    *,
    max_batches: Optional[int] = 32,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Hook q_proj / k_proj outputs for every Qwen3-style attention layer and
    accumulate per-head covariance C_h = E[y_h y_h^T] over calibration. Returns
    a dict mapping the attention module's name to a tuple-of-tensors carried
    in a flat dict:
        {f"{attn_name}.q_pca_U": (n_q_heads, head_dim, head_dim),
         f"{attn_name}.k_pca_U": (n_kv_heads, head_dim, head_dim)}

    The U tensors are the eigenvectors of C_h (largest eigenvalue first).
    """
    device = device or next(model.parameters()).device
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {}
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)

    handles = []
    accs: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, mod in model.named_modules():
        if not (hasattr(mod, "q_norm") and hasattr(mod, "k_norm")
                and hasattr(mod, "q_proj") and hasattr(mod, "k_proj")):
            continue
        accs[name] = {
            "q_cov": torch.zeros(n_heads, head_dim, head_dim,
                                  dtype=torch.float64, device=device),
            "k_cov": torch.zeros(n_kv_heads, head_dim, head_dim,
                                  dtype=torch.float64, device=device),
            "q_count": 0,
            "k_count": 0,
        }

        def _make_hook(attn_name, kind, n_h):
            def hook(_m, _inp, out):
                # out: (B, T, n_h*head_dim)
                B, T, _ = out.shape
                y = out.detach().reshape(B, T, n_h, head_dim).to(
                    torch.float32).reshape(-1, n_h, head_dim)  # (BT, n_h, head_dim)
                # Per-head covariance: (n_h, head_dim, head_dim)
                cov = torch.einsum("nhd,nhe->hde", y, y).to(torch.float64)
                if kind == "q":
                    accs[attn_name]["q_cov"] += cov
                    accs[attn_name]["q_count"] += y.shape[0]
                else:
                    accs[attn_name]["k_cov"] += cov
                    accs[attn_name]["k_count"] += y.shape[0]
            return hook

        handles.append(mod.q_proj.register_forward_hook(
            _make_hook(name, "q", n_heads)))
        handles.append(mod.k_proj.register_forward_hook(
            _make_hook(name, "k", n_kv_heads)))

    if verbose:
        print(f"[bbt-pca] hooked {len(accs)} Qwen3-style attention layers")

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

    pca: Dict[str, torch.Tensor] = {}
    for attn_name, acc in accs.items():
        if acc["q_count"] == 0 or acc["k_count"] == 0:
            continue
        Cq = acc["q_cov"] / max(1, acc["q_count"])
        Ck = acc["k_cov"] / max(1, acc["k_count"])
        # Symmetrize then eigendecompose. eigvals_h ascending; we don't reorder
        # because the math is invariant under any ordering of U columns as long
        # as the same U is used for rotation and Gamma reconstruction.
        Cq_sym = 0.5 * (Cq + Cq.transpose(-1, -2))
        Ck_sym = 0.5 * (Ck + Ck.transpose(-1, -2))
        # torch.linalg.eigh: returns (n_groups, head_dim) eigvals + (n_groups, head_dim, head_dim) eigvecs
        Uq = torch.linalg.eigh(Cq_sym.cpu())[1].to(torch.float32)  # eigh on cpu (stable)
        Uk = torch.linalg.eigh(Ck_sym.cpu())[1].to(torch.float32)
        pca[f"{attn_name}.q_pca_U"] = Uq
        pca[f"{attn_name}.k_pca_U"] = Uk

    if verbose:
        print(f"[bbt-pca] computed PCA for {len(pca)//2} attention layers "
              f"(n_heads={n_heads}, n_kv={n_kv_heads}, head_dim={head_dim})")
    return pca


def _bbt_prep_spectral_pca(
    model: nn.Module,
    influences: Dict[str, torch.Tensor],
    calib_iter: Iterable,
    *,
    alpha: float = 0.5,
    max_calib_batches: Optional[int] = 32,
    skip_substrings: Tuple[str, ...] = ("lm_head", "embed_tokens", "embed_positions"),
    only_substrings: Optional[Tuple[str, ...]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Route A: PCA-per-head matrix-Gamma BBT for Qwen3-style architectures.

    Per attention block:
      1. Calibrate per-head covariance of q_proj/k_proj outputs.
      2. Eigendecompose to U_h per head (per kv-head for k).
      3. Rotate q_proj rows by U_h^T (block-diagonal across heads).
         For GQA, q-heads in the same kv-group share U_kv (so attention scores
         are preserved: (U q_h) . (U k_kv)^T = q_h . k_kv).
      4. Replace q_norm with MatrixGammaRMSNorm(Gamma_h = U_h^T diag(gamma) U_h)
         and same for k_norm.

    Then run the standard norm-absorbed BBT on remaining layers (MLP, plus the
    INPUT-side scaling on q/k/v_proj which is independent of the PCA rotation).

    Returns:
        scales_map (channel scales for all BBT-scaled layers)
        pca_map (per-attention-layer U matrices, for dequant inversion)
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        if verbose:
            print("[bbt-spectral-pca] no model.config; falling back to no_rotation")
        return _bbt_prep_no_rotation(
            model, influences, alpha=alpha,
            skip_substrings=skip_substrings, only_substrings=only_substrings,
            verbose=verbose), {}
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    gqa_factor = n_heads // n_kv_heads

    pca = _calibrate_qk_pca(model, calib_iter,
                             max_batches=max_calib_batches, verbose=verbose)
    if not pca:
        if verbose:
            print("[bbt-spectral-pca] no Qwen3-style attention layers detected; "
                  "falling back to no_rotation")
        return _bbt_prep_no_rotation(
            model, influences, alpha=alpha,
            skip_substrings=skip_substrings, only_substrings=only_substrings,
            verbose=verbose), {}

    pca_map: Dict[str, torch.Tensor] = {}
    n_attn_blocks = 0
    for attn_name, attn in model.named_modules():
        if not (hasattr(attn, "q_norm") and hasattr(attn, "k_norm")
                and hasattr(attn, "q_proj") and hasattr(attn, "k_proj")):
            continue
        Uq = pca.get(f"{attn_name}.q_pca_U")
        Uk = pca.get(f"{attn_name}.k_pca_U")
        if Uq is None or Uk is None:
            continue
        n_attn_blocks += 1

        # GQA collapse: replace each q-group's Uq[a..a+gqa-1] with the
        # corresponding Uk[a // gqa] so the attention dot product stays
        # identical pre-quantization. Sub-optimal vs per-q-head PCA but
        # essential for math invariance under GQA.
        if gqa_factor > 1:
            Uq_collapsed = torch.zeros_like(Uq)
            for kv_h in range(n_kv_heads):
                for off in range(gqa_factor):
                    Uq_collapsed[kv_h * gqa_factor + off] = Uk[kv_h]
            Uq = Uq_collapsed

        device = attn.q_proj.weight.device
        dtype = attn.q_proj.weight.dtype

        # 1. Rotate q_proj rows: W_q[h_block] <- U_q[h]^T @ W_q[h_block]
        with torch.no_grad():
            Wq = attn.q_proj.weight.detach()  # (n_heads*head_dim, hidden)
            Wq_rot = Wq.clone()
            Uq_dev = Uq.to(device, dtype)
            for h in range(n_heads):
                row_lo = h * head_dim
                row_hi = (h + 1) * head_dim
                Wq_rot[row_lo:row_hi, :] = Uq_dev[h].t() @ Wq[row_lo:row_hi, :]
            attn.q_proj.weight = nn.Parameter(Wq_rot, requires_grad=False)
            if attn.q_proj.bias is not None:
                bq = attn.q_proj.bias.detach()
                bq_rot = bq.clone()
                for h in range(n_heads):
                    row_lo = h * head_dim
                    row_hi = (h + 1) * head_dim
                    bq_rot[row_lo:row_hi] = Uq_dev[h].t() @ bq[row_lo:row_hi]
                attn.q_proj.bias = nn.Parameter(bq_rot, requires_grad=False)

            # 2. Same for k_proj.
            Wk = attn.k_proj.weight.detach()
            Wk_rot = Wk.clone()
            Uk_dev = Uk.to(device, dtype)
            for h in range(n_kv_heads):
                row_lo = h * head_dim
                row_hi = (h + 1) * head_dim
                Wk_rot[row_lo:row_hi, :] = Uk_dev[h].t() @ Wk[row_lo:row_hi, :]
            attn.k_proj.weight = nn.Parameter(Wk_rot, requires_grad=False)
            if attn.k_proj.bias is not None:
                bk = attn.k_proj.bias.detach()
                bk_rot = bk.clone()
                for h in range(n_kv_heads):
                    row_lo = h * head_dim
                    row_hi = (h + 1) * head_dim
                    bk_rot[row_lo:row_hi] = Uk_dev[h].t() @ bk[row_lo:row_hi]
                attn.k_proj.bias = nn.Parameter(bk_rot, requires_grad=False)

            # 3. Replace q_norm with MatrixGammaRMSNorm:
            #     Gamma_h = diag(gamma_orig) @ U_q[h]
            # so that Gamma_h @ (rotated y / RMS) = gamma_orig * (y_orig / RMS),
            # i.e. the q_norm output equals vanilla. Crucially this preserves
            # math invariance through RoPE (which doesn't commute with U_h).
            gamma_q = attn.q_norm.weight.detach().to(torch.float32).cpu()
            eps_q = getattr(attn.q_norm, "variance_epsilon", 1e-6)
            Gamma_q = torch.zeros(n_heads, head_dim, head_dim, dtype=dtype)
            Uq_cpu = Uq.to(torch.float32)
            for h in range(n_heads):
                # diag(gamma) @ U[h]: row i = gamma[i] * U[h, i, :]
                Gamma_q[h] = gamma_q.unsqueeze(1) * Uq_cpu[h]
            attn.q_norm = MatrixGammaRMSNorm(
                Gamma_q.to(device, dtype),
                gamma_q.to(device, dtype),
                eps=eps_q,
            )

            gamma_k = attn.k_norm.weight.detach().to(torch.float32).cpu()
            eps_k = getattr(attn.k_norm, "variance_epsilon", 1e-6)
            Gamma_k = torch.zeros(n_kv_heads, head_dim, head_dim, dtype=dtype)
            Uk_cpu = Uk.to(torch.float32)
            for h in range(n_kv_heads):
                Gamma_k[h] = gamma_k.unsqueeze(1) * Uk_cpu[h]
            attn.k_norm = MatrixGammaRMSNorm(
                Gamma_k.to(device, dtype),
                gamma_k.to(device, dtype),
                eps=eps_k,
            )

        # Save U for dequant inversion (cpu, fp32). Index by attention name.
        pca_map[f"{attn_name}.q_pca_U"] = Uq.to(torch.float32).cpu()
        pca_map[f"{attn_name}.k_pca_U"] = Uk.to(torch.float32).cpu()

    if verbose:
        print(f"[bbt-spectral-pca] PCA-rotated q/k projections in "
              f"{n_attn_blocks} attention blocks; replaced q_norm/k_norm with "
              f"MatrixGammaRMSNorm")

    # Now apply standard norm-absorbed BBT on top. q/k_proj will get input-side
    # column scaling (independent of the PCA output rotation). The PCA-collapsed
    # math remains invariant.
    scales_map = _bbt_prep_no_rotation(
        model, influences, alpha=alpha,
        skip_substrings=skip_substrings, only_substrings=only_substrings,
        verbose=verbose,
    )
    return scales_map, pca_map


@torch.no_grad()
def _calibrate_qk_pair_pca(
    model: nn.Module,
    calib_iter: Iterable,
    *,
    max_batches: Optional[int] = 32,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    For each q_proj / k_proj, accumulate per-(head, pair) 2x2 covariance over
    the (i, i+head_dim/2) coordinate pairs (RoPE's natural symmetry plane), then
    eigendecompose to a 2x2 rotation matrix per (layer, head, pair).

    Returns flat dict:
        {f"{base}.q_pair_U": (n_heads, head_dim/2, 2, 2),
         f"{base}.k_pair_U": (n_kv_heads, head_dim/2, 2, 2)}
    """
    device = device or next(model.parameters()).device
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {}
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    if head_dim % 2 != 0:
        if verbose:
            print(f"[bbt-pair-pca] head_dim={head_dim} is odd; skipping pair-PCA")
        return {}
    half = head_dim // 2

    def _hook_factory(name: str, n_h: int, accs: dict, kind: str):
        def hook(_m, _inp, out):
            B, T, D_total = out.shape
            y = out.detach().reshape(B * T, n_h, head_dim).to(torch.float32)
            # Pair view: (BT, n_h, 2, half) where dim 2 indexes pair-half (lo/hi).
            # For each pair p, the "lo" coord is y[..., p] and "hi" is y[..., p+half].
            y_lo = y[..., :half]   # (BT, n_h, half)
            y_hi = y[..., half:]   # (BT, n_h, half)
            # 2x2 covariance per (head, pair):
            #   C00 = sum y_lo^2, C01 = sum y_lo*y_hi, C11 = sum y_hi^2
            c00 = (y_lo * y_lo).sum(dim=0)   # (n_h, half)
            c01 = (y_lo * y_hi).sum(dim=0)
            c11 = (y_hi * y_hi).sum(dim=0)
            cov = torch.stack([
                torch.stack([c00, c01], dim=-1),
                torch.stack([c01, c11], dim=-1),
            ], dim=-2)  # (n_h, half, 2, 2)
            if kind == "q":
                accs[name + "_q"] = accs.get(name + "_q", torch.zeros_like(cov)) + cov.to(torch.float64)
                accs[name + "_q_count"] = accs.get(name + "_q_count", 0) + y.shape[0]
            else:
                accs[name + "_k"] = accs.get(name + "_k", torch.zeros_like(cov)) + cov.to(torch.float64)
                accs[name + "_k_count"] = accs.get(name + "_k_count", 0) + y.shape[0]
        return hook

    accs: dict = {}
    handles = []
    bases: list[str] = []
    for name, mod in model.named_modules():
        if not (hasattr(mod, "q_proj") and hasattr(mod, "k_proj")):
            continue
        if not isinstance(mod.q_proj, nn.Linear):
            continue
        bases.append(name)
        handles.append(mod.q_proj.register_forward_hook(
            _hook_factory(name, n_heads, accs, "q")))
        handles.append(mod.k_proj.register_forward_hook(
            _hook_factory(name, n_kv_heads, accs, "k")))

    if verbose:
        print(f"[bbt-pair-pca] hooked {len(bases)} attention layers "
              f"(n_heads={n_heads}, n_kv={n_kv_heads}, head_dim={head_dim})")

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

    def _force_so2(U: torch.Tensor) -> torch.Tensor:
        """eigh returns U in O(2); flip second column if det(U) = -1 so that
        U is a pure rotation in SO(2) and commutes with RoPE's 2x2 rotations."""
        det = U[..., 0, 0] * U[..., 1, 1] - U[..., 0, 1] * U[..., 1, 0]
        flip = (det < 0).to(U.dtype) * -2.0 + 1.0   # +1 if det>=0, -1 if det<0
        U2 = U.clone()
        U2[..., :, 1] = U[..., :, 1] * flip.unsqueeze(-1)
        return U2

    pca: Dict[str, torch.Tensor] = {}
    for base in bases:
        if (base + "_q_count") not in accs or (base + "_k_count") not in accs:
            continue
        Cq = accs[base + "_q"] / max(1, accs[base + "_q_count"])  # (n_h, half, 2, 2)
        Ck = accs[base + "_k"] / max(1, accs[base + "_k_count"])
        # Symmetrize. eigh is stable for symmetric 2x2.
        Cq = 0.5 * (Cq + Cq.transpose(-1, -2))
        Ck = 0.5 * (Ck + Ck.transpose(-1, -2))
        Uq = torch.linalg.eigh(Cq.cpu())[1].to(torch.float32)  # (n_heads, half, 2, 2)
        Uk = torch.linalg.eigh(Ck.cpu())[1].to(torch.float32)
        Uq = _force_so2(Uq)
        Uk = _force_so2(Uk)
        pca[f"{base}.q_pair_U"] = Uq
        pca[f"{base}.k_pair_U"] = Uk
    if verbose:
        print(f"[bbt-pair-pca] computed pair-PCA for {len(pca)//2} attention layers")
    return pca


def _apply_pair_rotation(W: torch.Tensor, U: torch.Tensor,
                          n_h: int, head_dim: int) -> torch.Tensor:
    """
    Rotate each (h*head_dim+p, h*head_dim+p+head_dim/2) row pair of W by U[h, p].
    Specifically: new_lo = u00*lo + u10*hi, new_hi = u01*lo + u11*hi
    (i.e., U^T applied as a left-multiplier to the column vector (lo, hi)).
    W shape: (n_h*head_dim, in_features). U shape: (n_h, head_dim/2, 2, 2).
    """
    half = head_dim // 2
    in_features = W.shape[1]
    Wph = W.view(n_h, head_dim, in_features)             # (n_h, head_dim, in)
    W_lo = Wph[:, :half, :]                              # (n_h, half, in)
    W_hi = Wph[:, half:, :]
    u00 = U[..., 0, 0].unsqueeze(-1)  # (n_h, half, 1)
    u10 = U[..., 1, 0].unsqueeze(-1)
    u01 = U[..., 0, 1].unsqueeze(-1)
    u11 = U[..., 1, 1].unsqueeze(-1)
    new_lo = u00 * W_lo + u10 * W_hi
    new_hi = u01 * W_lo + u11 * W_hi
    out = torch.cat([new_lo, new_hi], dim=1).reshape(n_h * head_dim, in_features)
    return out.contiguous()


def _bbt_prep_spectral_pca_2d(
    model: nn.Module,
    influences: Dict[str, torch.Tensor],
    calib_iter: Iterable,
    *,
    alpha: float = 0.5,
    max_calib_batches: Optional[int] = 32,
    skip_substrings: Tuple[str, ...] = ("lm_head", "embed_tokens", "embed_positions"),
    only_substrings: Optional[Tuple[str, ...]] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    RoPE-compatible spectral_pca for non-q_norm architectures (SmolLM, Qwen2.5).

    Each (i, i+head_dim/2) coordinate pair is rotated by an independent 2x2
    matrix U[h, p]. Because both U and RoPE's R_t live in SO(2) on the same
    plane, they commute, so the post-RoPE attention dot product is invariant
    when q-heads sharing a kv-head use the same rotation as the kv-head:
        (R_t U q_orig) · (R_s U k_orig) = q_orig · k_orig    (after GQA collapse)
    No matrix-Gamma replacement needed: q_norm doesn't exist on these models.

    On top of the pair-PCA rotation, the standard norm-absorbed BBT scaling is
    still applied (input-side, independent).

    Returns scales_map and pca_map (the latter holds {f"{name}.q_pair_U", "...k_pair_U"}).
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return _bbt_prep_no_rotation(
            model, influences, alpha=alpha,
            skip_substrings=skip_substrings, only_substrings=only_substrings,
            verbose=verbose), {}
    n_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    gqa_factor = n_heads // n_kv_heads
    pca = _calibrate_qk_pair_pca(model, calib_iter,
                                  max_batches=max_calib_batches, verbose=verbose)
    if not pca:
        if verbose:
            print("[bbt-pair-pca] no usable attention layers; falling back to no_rotation")
        return _bbt_prep_no_rotation(
            model, influences, alpha=alpha,
            skip_substrings=skip_substrings, only_substrings=only_substrings,
            verbose=verbose), {}

    pca_map: Dict[str, torch.Tensor] = {}
    n_attn = 0
    for attn_name, attn in model.named_modules():
        if not (hasattr(attn, "q_proj") and hasattr(attn, "k_proj")
                and isinstance(attn.q_proj, nn.Linear)):
            continue
        Uq = pca.get(f"{attn_name}.q_pair_U")
        Uk = pca.get(f"{attn_name}.k_pair_U")
        if Uq is None or Uk is None:
            continue
        n_attn += 1
        # GQA collapse: q-pair rotation = corresponding kv-pair rotation, so
        # q · k post-RoPE stays invariant.
        if gqa_factor > 1:
            Uq_collapsed = torch.zeros_like(Uq)
            for kv_h in range(n_kv_heads):
                for off in range(gqa_factor):
                    Uq_collapsed[kv_h * gqa_factor + off] = Uk[kv_h]
            Uq = Uq_collapsed

        device = attn.q_proj.weight.device
        dtype = attn.q_proj.weight.dtype
        with torch.no_grad():
            Uq_dev = Uq.to(device, dtype)
            Uk_dev = Uk.to(device, dtype)
            Wq_rot = _apply_pair_rotation(attn.q_proj.weight.detach(),
                                           Uq_dev, n_heads, head_dim)
            attn.q_proj.weight = nn.Parameter(Wq_rot, requires_grad=False)
            if attn.q_proj.bias is not None:
                # Bias is a vector of length n_heads*head_dim; rotate per (h, p)
                # via the 2x2 U[h, p] applied to (b[h*D+p], b[h*D+p+half]).
                bq = attn.q_proj.bias.detach()
                bq_rot = _apply_pair_rotation(bq.view(-1, 1), Uq_dev,
                                                n_heads, head_dim).view(-1)
                attn.q_proj.bias = nn.Parameter(bq_rot, requires_grad=False)
            Wk_rot = _apply_pair_rotation(attn.k_proj.weight.detach(),
                                           Uk_dev, n_kv_heads, head_dim)
            attn.k_proj.weight = nn.Parameter(Wk_rot, requires_grad=False)
            if attn.k_proj.bias is not None:
                bk = attn.k_proj.bias.detach()
                bk_rot = _apply_pair_rotation(bk.view(-1, 1), Uk_dev,
                                                n_kv_heads, head_dim).view(-1)
                attn.k_proj.bias = nn.Parameter(bk_rot, requires_grad=False)

        pca_map[f"{attn_name}.q_pair_U"] = Uq.to(torch.float32).cpu()
        pca_map[f"{attn_name}.k_pair_U"] = Uk.to(torch.float32).cpu()

    if verbose:
        print(f"[bbt-pair-pca] pair-rotated q/k in {n_attn} attention layers; "
              f"layering norm-absorbed BBT on top")

    scales_map = _bbt_prep_no_rotation(
        model, influences, alpha=alpha,
        skip_substrings=skip_substrings, only_substrings=only_substrings,
        verbose=verbose,
    )
    return scales_map, pca_map


def _bbt_prep_no_rot_fallback(model, influences, alpha, skip_substrings,
                              only_substrings, verbose):
    """Original simple weight-only path (works on any architecture but
    yields biased noise toward low-influence channels at dequant time)."""
    from .influence import bbt_channel_scales

    scales_map: Dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear): continue
        if any(s in name for s in skip_substrings): continue
        if only_substrings is not None and not any(s in name for s in only_substrings):
            continue
        inf = influences.get(name)
        if inf is None: continue
        device, dtype = mod.weight.device, mod.weight.dtype
        s = bbt_channel_scales(inf, alpha=alpha).to(device, dtype)
        with torch.no_grad():
            mod.weight.mul_(s[None, :])
        scales_map[name] = s.detach().cpu()
    if verbose:
        print(f"[bbt-quant-norot-fallback] BBT-scaled {len(scales_map)} layers (alpha={alpha})")
    return scales_map


# -----------------------------------------------------------------------------
# End-to-end helper
# -----------------------------------------------------------------------------

def run_bbt_autoround(
    model: nn.Module,
    tokenizer,
    calib_iter: Iterable,
    output_dir: Path,
    *,
    bits: int = 2,
    alpha: float = 0.5,
    group_size: int = 128,
    enable_alg_ext: bool = True,
    device: str = "auto",
    max_calib_batches: Optional[int] = 32,
    only_substrings: Optional[Tuple[str, ...]] = None,
    skip_substrings: Tuple[str, ...] = ("lm_head", "embed_tokens", "embed_positions"),
    save_metadata: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Full BBT x auto-round pipeline. Returns a summary dict.

    After this call, `output_dir` contains the auto-round quantized model
    plus `bbt_metadata.pt` (influences + scales) and `bbt_summary.json`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect influences once *before* rotation so we can also save them raw.
    raw_influences = compute_layer_influences(
        model,
        calib_iter,
        skip_substrings=skip_substrings,
        max_batches=max_calib_batches,
    )
    if save_metadata:
        save_influences(raw_influences, output_dir / "bbt_influences.pt")

    # BBT prep. Two modes:
    #   - rotation_mode == "no_rotation": bake column scales into each
    #     plain nn.Linear's weight directly (no WHT, no padding). Simplest
    #     and auto-round-compatible. Inverse scales are recovered at dequant
    #     time -- we never run inference on the quantized model directly.
    #   - rotation_mode == "wht_pow2_rotation": legacy spectral-basis BBT
    #     (currently blocked by auto-round's wrapper bypassing pre-hooks).
    rotation_mode = getattr(run_bbt_autoround, "_rotation_mode", "no_rotation")
    pca_map: Dict[str, torch.Tensor] = {}
    if rotation_mode == "no_rotation":
        scales_map = _bbt_prep_no_rotation(
            model,
            raw_influences,
            alpha=alpha,
            skip_substrings=skip_substrings,
            only_substrings=only_substrings,
            verbose=verbose,
        )
    elif rotation_mode == "spectral":
        # Patch wrapper BEFORE auto-round wraps anything
        patch_autoround_wrapper_for_bbt()
        scales_map = _bbt_prep_spectral(
            model,
            raw_influences,
            alpha=alpha,
            skip_substrings=skip_substrings,
            only_substrings=only_substrings,
            verbose=verbose,
        )
    elif rotation_mode == "spectral_pca":
        scales_map, pca_map = _bbt_prep_spectral_pca(
            model,
            raw_influences,
            calib_iter,
            alpha=alpha,
            max_calib_batches=max_calib_batches,
            skip_substrings=skip_substrings,
            only_substrings=only_substrings,
            verbose=verbose,
        )
    elif rotation_mode == "spectral_pca_2d":
        scales_map, pca_map = _bbt_prep_spectral_pca_2d(
            model,
            raw_influences,
            calib_iter,
            alpha=alpha,
            max_calib_batches=max_calib_batches,
            skip_substrings=skip_substrings,
            only_substrings=only_substrings,
            verbose=verbose,
        )
    else:
        scales_map = _bbt_prep_inplace_plain_linear(
            model,
            raw_influences,
            alpha=alpha,
            skip_substrings=skip_substrings,
            only_substrings=only_substrings,
            verbose=verbose,
        )
    if save_metadata:
        torch.save(scales_map, str(output_dir / "bbt_scales.pt"))
        if pca_map:
            torch.save(pca_map, str(output_dir / "bbt_pca.pt"))

    # Always patch auto-round's per-block memory hygiene; the upstream loop
    # only calls clear_memory() once at the start, which causes cumulative
    # XPU memory pressure on bigger models (see BBT_RESULTS.md).
    patch_autoround_for_xpu_memory()

    if verbose:
        print(f"[bbt-quant] launching auto-round: bits={bits} alpha={alpha}")
    quantize_with_autoround(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        bits=bits,
        group_size=group_size,
        enable_alg_ext=enable_alg_ext,
        device=device,
    )

    summary = {
        "bits": bits,
        "alpha": alpha,
        "group_size": group_size,
        "enable_alg_ext": enable_alg_ext,
        "n_scaled_layers": len(scales_map),
        "n_influence_layers": len(raw_influences),
        "skip_substrings": list(skip_substrings),
        "only_substrings": list(only_substrings) if only_substrings else None,
    }
    if save_metadata:
        with open(output_dir / "bbt_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    if verbose:
        print(f"[bbt-quant] done. Written to {output_dir}")
    return summary
