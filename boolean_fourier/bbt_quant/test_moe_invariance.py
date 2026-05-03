"""
Math invariance test for the MoE-aware ``_bbt_prep_no_rotation`` prep on a
tiny synthetic Qwen2-MoE built from the actual HF modeling classes (just with
reduced dimensions so it fits in memory). Exercises:

  * standard SwiGLU on ``mlp`` directly
  * routed experts as ``Qwen2MoeExperts`` (modern HF — fused 3D ``gate_up_proj``
    Parameter, not a ``ModuleList``)
  * shared expert as a standalone Module (not a ``ModuleList``)
  * router gate as a custom ``Qwen2MoeTopKRouter`` Module (raw 2D Parameter,
    not ``nn.Linear``)
  * shared_expert_gate as nn.Linear

After the prep mutates the model in-place (norm.weight /= s, all consumers'
input columns *= s), the full-precision forward pass must reproduce the
original logits to numerical precision. If this test fails, downstream
quantization will silently produce a model that disagrees with vanilla
auto-round at full precision (= mathematically broken, regardless of any
INT2 quant-noise improvement).

Run: ``python -m boolean_fourier.bbt_quant.test_moe_invariance``
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from .autoround_bbt import (
    _bbt_prep_no_rotation, _collect_mlp_input_consumers, _collect_mlp_down_pairs,
)
from .influence import compute_layer_influences


def main() -> None:
    cfg = AutoConfig.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", trust_remote_code=True)
    cfg.hidden_size = 128
    cfg.intermediate_size = 256
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    cfg.num_hidden_layers = 2
    cfg.num_experts = 8
    cfg.moe_intermediate_size = 64
    cfg.shared_expert_intermediate_size = 96
    cfg.num_experts_per_tok = 2
    cfg.vocab_size = 1024
    cfg.max_position_embeddings = 256

    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()

    mlp = m.model.layers[0].mlp
    consumers = _collect_mlp_input_consumers(mlp)
    pairs = _collect_mlp_down_pairs(mlp)
    print(f"[moe-invariance] Block-0 MLP consumers ({len(consumers)}):")
    for tgt in consumers:
        print(f"  {tgt.label} ({tgt.kind}): in_dim={tgt.in_features}")
    print(f"[moe-invariance] Block-0 down pairs ({len(pairs)}):")
    for label, up, down in pairs:
        print(f"  {label}: up out={up.out_features}, down in={down.in_features}")

    inp = torch.randint(0, cfg.vocab_size, (1, 32))
    with torch.no_grad():
        ref = m(inp).logits

    calib = [{"input_ids": inp}]
    infl = compute_layer_influences(m, calib, max_batches=1)
    scales = _bbt_prep_no_rotation(m, infl, alpha=0.5, verbose=True)

    with torch.no_grad():
        prep = m(inp).logits

    diff = (ref - prep).abs().max().item()
    rel = diff / max(ref.abs().max().item(), 1e-9)
    print(f"\n[moe-invariance] max|ref - prep| = {diff:.4e}")
    print(f"[moe-invariance] relative        = {rel:.4e}")
    assert rel < 1e-3, f"INVARIANCE BROKEN ({rel})"
    print("[moe-invariance] OK — math invariance preserved through MoE prep")

    n_per_block = len(scales) // cfg.num_hidden_layers
    print(f"[moe-invariance] avg layers absorbed per block = {n_per_block} "
          f"(vanilla MLP would be 6; full MoE coverage ~10+)")
    assert n_per_block >= 10, f"Expected MoE-level absorption count, got {n_per_block}"


if __name__ == "__main__":
    main()
