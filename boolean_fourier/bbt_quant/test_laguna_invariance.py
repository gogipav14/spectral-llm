"""
Math invariance test for our BBT toolkit on a tiny synthetic Laguna built
directly from the poolside modeling source. Laguna's attention has:

  - q_norm/k_norm (LagunaRMSNorm on head_dim, post-projection — Qwen3-style)
  - g_proj: nn.Linear(hidden_size, num_heads) — a per-head softplus gate
    multiplied into the attention output before o_proj. CRITICAL: g_proj
    consumes input_layernorm output, so input-side BBT absorption MUST
    include it; otherwise softplus(W·x/s) leaks per-channel scaling through
    a non-linearity and breaks invariance.
  - sparse MoE blocks with fused 3D experts.gate_up_proj Parameter +
    LagunaTopKRouter (raw 2D Parameter on a custom Module) + standalone
    shared_experts (LagunaMLP SwiGLU) — same structure as Qwen2-MoE.
  - alternating sliding/full attention via layer_types config.

This test was the canary that surfaced the g_proj bug: without scaling
g_proj's input columns, no_rotation gave rel_err ~0.014 on Laguna's
attention path even though q_proj output was provably unchanged (q_norm
stayed correct, but the head-gate g_proj multiplier didn't).

Run: ``python -m boolean_fourier.bbt_quant.test_laguna_invariance``
"""
from __future__ import annotations
import os
import sys

# Locate the poolside modeling source. We expect it to have been pulled to
# /tmp/laguna (configuration_laguna.py + modeling_laguna.py with the relative
# import patched). Skip the test gracefully if not present.
LAGUNA_SRC = os.environ.get("LAGUNA_SRC", "/tmp/laguna")
if not (os.path.exists(os.path.join(LAGUNA_SRC, "modeling_laguna.py"))
        and os.path.exists(os.path.join(LAGUNA_SRC, "configuration_laguna.py"))):
    print(f"[laguna-invariance] SKIPPED — modeling source not at {LAGUNA_SRC}. "
          f"Set $LAGUNA_SRC or pull from huggingface.co/poolside/Laguna-XS.2.")
    sys.exit(0)

sys.path.insert(0, LAGUNA_SRC)

import torch
from configuration_laguna import LagunaConfig
from modeling_laguna import LagunaForCausalLM

from .autoround_bbt import (
    _bbt_prep_spectral_pca, _bbt_prep_spectral_pca_2d,
    _collect_mlp_input_consumers,
)
from .influence import compute_layer_influences


def _build_tiny_laguna():
    cfg = LagunaConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
        max_position_embeddings=64, rms_norm_eps=1e-6, sliding_window=8,
        rope_parameters={
            "full_attention": {"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 1.0},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 1.0},
            "original_max_position_embeddings": 64,
        },
        layer_types=["full_attention", "sliding_attention"],
        mlp_layer_types=["sparse", "dense"],
        moe_router_logit_softcapping=0.0,
        bos_token_id=0, eos_token_id=1, pad_token_id=0, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return cfg, LagunaForCausalLM(cfg).eval().to(torch.float32)


def main() -> None:
    cfg, m = _build_tiny_laguna()
    inp = torch.randint(0, cfg.vocab_size, (1, 16))

    # Sanity: confirm structure
    print(f"[laguna] block-0 mlp: {type(m.model.layers[0].mlp).__name__}")
    print(f"[laguna] block-1 mlp: {type(m.model.layers[1].mlp).__name__}")
    print(f"[laguna] block-0 attn has q_norm: {hasattr(m.model.layers[0].self_attn, 'q_norm')}")
    print(f"[laguna] block-0 attn has g_proj (head gate): "
          f"{hasattr(m.model.layers[0].self_attn, 'g_proj')}")
    cons0 = _collect_mlp_input_consumers(m.model.layers[0].mlp)
    print(f"[laguna] block-0 SPARSE consumers: {len(cons0)} "
          f"({[(c.label, c.kind) for c in cons0]})")

    # spectral_pca path
    with torch.no_grad():
        ref = m(inp).logits
    calib = [{"input_ids": inp}]
    infl = compute_layer_influences(m, calib, max_batches=1)
    _, pca = _bbt_prep_spectral_pca(m, infl, calib_iter=calib, alpha=0.5,
                                      max_calib_batches=1, verbose=True)
    with torch.no_grad():
        prep = m(inp).logits
    rel = (ref - prep).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    print(f"\n[laguna-spectral-pca] relative = {rel:.4e}")
    assert rel < 1e-3, f"spectral_pca INVARIANCE BROKEN ({rel})"
    print("[laguna-spectral-pca] OK")

    # spectral_pca_2d path on a fresh model
    cfg2, m2 = _build_tiny_laguna()
    with torch.no_grad():
        ref2 = m2(inp).logits
    calib2 = [{"input_ids": inp}]
    infl2 = compute_layer_influences(m2, calib2, max_batches=1)
    _bbt_prep_spectral_pca_2d(m2, infl2, calib_iter=calib2, alpha=0.5,
                                max_calib_batches=1, verbose=True)
    with torch.no_grad():
        prep2 = m2(inp).logits
    rel2 = (ref2 - prep2).abs().max().item() / max(ref2.abs().max().item(), 1e-9)
    print(f"\n[laguna-pair-pca] relative = {rel2:.4e}")
    assert rel2 < 1e-3, f"spectral_pca_2d INVARIANCE BROKEN ({rel2})"
    print("[laguna-pair-pca] OK")


if __name__ == "__main__":
    main()
