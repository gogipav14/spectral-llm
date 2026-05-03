"""
Fallback dequantizer for auto-round W2/W4 checkpoints that no installed
inference backend can load (e.g. group_size that doesn't divide hidden_size,
missing C kernels on Windows Python 3.13, etc.).

Reads an auto_round:auto_gptq-packed checkpoint, unpacks qweight/qzeros/scales
to FP16 nn.Linear weights, strips quantization_config, writes a plain HF dir.

Usage:
    python -m bbt_quant.dequantize_autoround \
        --src C:\\Users\\gogip\\bbt_quant_out\\smollm_w2_vanilla\\autoround \
        --dst C:\\Users\\gogip\\bbt_quant_out\\smollm_w2_vanilla\\fp16
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open


def _unpack_qweight(qweight: torch.Tensor, bits: int, sym: bool) -> torch.Tensor:
    """
    qweight packed along in_features (row) axis in int32, (in_feat/(32/bits), out_feat).
    Returns int tensor of shape (in_features, out_features) with values in
    [0, 2^bits) (if asym) or centered (if sym, shift later when applying scales).
    """
    pack = 32 // bits
    n_rows, n_cols = qweight.shape
    in_features = n_rows * pack
    mask = (1 << bits) - 1
    out = torch.empty((in_features, n_cols), dtype=torch.int32)
    for i in range(pack):
        out[i::pack] = (qweight >> (i * bits)) & mask
    return out


def _unpack_qzeros(qzeros: torch.Tensor, bits: int) -> torch.Tensor:
    """
    qzeros packed along out_features (col) axis in int32, (n_groups, out_feat/(32/bits)).
    Returns (n_groups, out_features) int tensor.
    """
    pack = 32 // bits
    n_groups, n_cols = qzeros.shape
    out_features = n_cols * pack
    mask = (1 << bits) - 1
    out = torch.empty((n_groups, out_features), dtype=torch.int32)
    for i in range(pack):
        out[:, i::pack] = (qzeros >> (i * bits)) & mask
    return out


def _dequantize_layer(qweight: torch.Tensor, scales: torch.Tensor,
                      qzeros: torch.Tensor | None, bits: int, group_size: int,
                      sym: bool, target_in_features: int | None = None) -> torch.Tensor:
    """
    Returns FP16 weight of shape (out_features, target_in_features or padded_in),
    suitable for nn.Linear. auto-round pads in_features up to a multiple of
    group_size when it doesn't divide evenly; we truncate back here.
    """
    q = _unpack_qweight(qweight, bits, sym)  # (stored_in, out) int32
    stored_in, out_features = q.shape
    n_groups = scales.shape[0]
    # auto-round may write more group-slots than there are rows when
    # in_features < n_groups * group_size (partial final group).
    if sym:
        zero_point = 1 << (bits - 1)
        w = q.to(torch.float32) - zero_point
    else:
        z = _unpack_qzeros(qzeros, bits).to(torch.float32) + 1
        z_ext = z.repeat_interleave(group_size, dim=0)[:stored_in]
        w = q.to(torch.float32) - z_ext
    s_ext = scales.to(torch.float32).repeat_interleave(group_size, dim=0)[:stored_in]  # (stored_in, out)
    w = w * s_ext
    w = w.t().contiguous()  # (out, stored_in)
    if target_in_features is not None and target_in_features < stored_in:
        w = w[:, :target_in_features].contiguous()
    return w.to(torch.float16)


def _next_pow2(n: int) -> int:
    import math as _math
    return 1 if n == 0 else 2 ** _math.ceil(_math.log2(n))


def _hadamard_normalized(d: int) -> torch.Tensor:
    """Normalized Hadamard H_d / sqrt(d), assuming d is a power of two."""
    import math as _math
    H = torch.ones((1, 1), dtype=torch.float32)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / _math.sqrt(d)


def _bbt_unrotate(W_padded: torch.Tensor, input_scale: torch.Tensor,
                  public_in: int) -> torch.Tensor:
    """
    Convert a BBT-rotated, BBT-scaled weight back to a plain Linear weight.

    Forward equation in the rotated layer:
        y = (pad(x) @ H @ diag(1/s)) @ W^T
    Define W_eff = W @ diag(1/s) @ H so that y = pad(x) @ W_eff^T.
    Padded columns of x are zero, so we can truncate W_eff[:, :public_in].

    Args:
        W_padded: shape (out_features, d_pad), the dequantized (W2) weight
        input_scale: shape (d_pad,), the BBT input scale vector
        public_in: original (pre-padding) in_features

    Returns:
        W_unrotated: shape (out_features, public_in), plain Linear weight.
    """
    d_pad = W_padded.shape[1]
    assert input_scale.numel() == d_pad, (
        f"scale {input_scale.numel()} != d_pad {d_pad}"
    )
    H = _hadamard_normalized(d_pad).to(W_padded.dtype)
    s_inv = (1.0 / input_scale.to(W_padded.dtype)).reshape(1, -1)
    W_eff = (W_padded * s_inv) @ H  # (out, d_pad)
    return W_eff[:, :public_in].contiguous()


def dequantize_checkpoint(src: Path, dst: Path) -> dict:
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    with open(src / "quantization_config.json") as f:
        qcfg = json.load(f)
    bits = int(qcfg["bits"])
    group_size = int(qcfg["group_size"])
    sym = bool(qcfg.get("sym", True))
    print(f"[dequant] bits={bits} group_size={group_size} sym={sym}")

    with open(src / "config.json") as f:
        mcfg = json.load(f)
    hidden_size = int(mcfg["hidden_size"])
    intermediate_size = int(mcfg["intermediate_size"])
    n_heads = int(mcfg["num_attention_heads"])
    head_dim = int(mcfg.get("head_dim", hidden_size // n_heads))
    n_kv_heads = int(mcfg.get("num_key_value_heads", n_heads))
    attn_hidden = n_heads * head_dim     # o_proj in dim (may != hidden_size on Qwen3)
    kv_hidden = n_kv_heads * head_dim
    in_feats_by_suffix = {
        "q_proj": hidden_size, "k_proj": hidden_size, "v_proj": hidden_size,
        "o_proj": attn_hidden,
        "gate_proj": hidden_size, "up_proj": hidden_size,
        "down_proj": intermediate_size,
    }

    # BBT detection: presence of bbt_scales.pt means some layers were
    # WHT-rotated and column-scaled; we need to unfold that at dequant time.
    bbt_scales: dict[str, torch.Tensor] = {}
    bbt_scales_path = src / "bbt_scales.pt"
    if bbt_scales_path.exists():
        bbt_scales = torch.load(str(bbt_scales_path), map_location="cpu",
                                weights_only=True)
        print(f"[dequant] BBT mode: {len(bbt_scales)} rotated/scaled layers")

    # spectral_pca mode: per-head PCA rotation on q_proj/k_proj output rows.
    # Keys are `f"{attn_name}.q_pca_U"` / `.k_pca_U`, values are
    # (n_groups, head_dim, head_dim) eigenvector tensors.
    # spectral_pca_2d uses `f"{attn_name}.q_pair_U"` / `.k_pair_U` of shape
    # (n_groups, head_dim/2, 2, 2) — per-(head, pair) 2x2 rotation.
    bbt_pca: dict[str, torch.Tensor] = {}
    bbt_pca_path = src / "bbt_pca.pt"
    if bbt_pca_path.exists():
        bbt_pca = torch.load(str(bbt_pca_path), map_location="cpu",
                             weights_only=True)
        n_full = sum(1 for k in bbt_pca if k.endswith("pca_U"))
        n_pair = sum(1 for k in bbt_pca if k.endswith("pair_U"))
        if n_full:
            print(f"[dequant] spectral_pca mode: {n_full//2} attention layers "
                  f"with per-head PCA")
        if n_pair:
            print(f"[dequant] spectral_pca_2d mode: {n_pair//2} attention layers "
                  f"with per-(head,pair) 2x2 PCA")

    # Load all tensors to dict. Auto-round shards models >2GB across multiple
    # safetensors files (e.g. W4/W3 quants of larger models); single-file W2
    # quants of small models stay in `model.safetensors`. Detect both.
    tensors: dict[str, torch.Tensor] = {}
    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index.get("weight_map", {}).values()))
        print(f"[dequant] sharded checkpoint: {len(shard_files)} shards")
        for shard in shard_files:
            with safe_open(str(src / shard), framework="pt") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
    else:
        with safe_open(str(src / "model.safetensors"), framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

    # Group quantized layers by base name
    layer_bases = set()
    for k in tensors:
        if k.endswith(".qweight"):
            layer_bases.add(k[: -len(".qweight")])

    # input_scale buffers persist in the checkpoint; we read them when
    # they exist (BBT path) and drop them from pass-through.
    print(f"[dequant] found {len(layer_bases)} quantized layers")
    new_tensors: dict[str, torch.Tensor] = {}
    pass_through = 0
    for k, v in tensors.items():
        base = k.rsplit(".", 1)[0]
        suffix = k.rsplit(".", 1)[1]
        if base in layer_bases and suffix in {"qweight", "qzeros", "scales",
                                              "g_idx", "input_scale"}:
            continue
        # spectral_pca: drop the matrix-Gamma buffer; the per-head Gamma is
        # equivalent to the original vector-gamma once we un-rotate the
        # corresponding q_proj/k_proj rows below. The `q_norm.weight` /
        # `k_norm.weight` saved alongside is the original scalar gamma.
        if suffix == "gamma_matrix":
            continue
        # spectral_pca / spectral_pca_2d: bias of q_proj/k_proj was rotated
        # alongside the weight at prep time. Un-rotate it here so the
        # dequantized model has plain (unrotated) biases.
        if suffix == "bias" and bbt_pca:
            base = k.rsplit(".", 1)[0]
            attn_name = base.rsplit(".", 1)[0]
            sub = base.rsplit(".", 1)[-1]
            if sub in {"q_proj", "k_proj"}:
                pair_U_b = bbt_pca.get(
                    f"{attn_name}.{'q' if sub == 'q_proj' else 'k'}_pair_U")
                pca_U_b = bbt_pca.get(
                    f"{attn_name}.{'q' if sub == 'q_proj' else 'k'}_pca_U")
                if pca_U_b is not None:
                    n_groups, hd, _ = pca_U_b.shape
                    b = v.to(torch.float32).view(n_groups, hd)
                    U = pca_U_b.to(torch.float32)
                    b_unrot = torch.einsum("hde,he->hd", U, b)
                    v = b_unrot.reshape(-1).to(v.dtype)
                elif pair_U_b is not None:
                    n_groups, half, _, _ = pair_U_b.shape
                    head_dim = 2 * half
                    b = v.to(torch.float32).view(n_groups, head_dim)
                    b_lo = b[:, :half]
                    b_hi = b[:, half:]
                    U = pair_U_b.to(torch.float32)
                    u00 = U[..., 0, 0]; u01 = U[..., 0, 1]
                    u10 = U[..., 1, 0]; u11 = U[..., 1, 1]
                    new_lo = u00 * b_lo + u01 * b_hi
                    new_hi = u10 * b_lo + u11 * b_hi
                    v = torch.cat([new_lo, new_hi], dim=1).reshape(-1).to(v.dtype)
        new_tensors[k] = v
        pass_through += 1

    n_bbt = 0
    n_pca = 0
    for base in sorted(layer_bases):
        qw = tensors[base + ".qweight"]
        sc = tensors[base + ".scales"]
        qz = tensors.get(base + ".qzeros")
        suffix = base.rsplit(".", 1)[-1]
        target_in = in_feats_by_suffix.get(suffix)

        # spectral_pca per-head un-rotation. Applied AFTER all other dequant
        # transforms below by saving the U key here and de-rotating w later.
        pca_U = None        # full per-head (head_dim, head_dim) — spectral_pca
        pair_U = None       # per-(head, pair) (head_dim/2, 2, 2) — spectral_pca_2d
        if bbt_pca and suffix in {"q_proj", "k_proj"}:
            attn_name = base[: -len(f".{suffix}")]
            pca_U = bbt_pca.get(f"{attn_name}.{'q' if suffix == 'q_proj' else 'k'}_pca_U")
            pair_U = bbt_pca.get(f"{attn_name}.{'q' if suffix == 'q_proj' else 'k'}_pair_U")

        if base in bbt_scales:
            # BBT path. Three sub-modes detected from scale length:
            #   - len(s) == in_features and bbt_summary says "norm_absorbed":
            #       norm.weight has been pre-divided by s; W has been pre-mul'd
            #       by s. Just dequantize as-is; the modified state is what we
            #       want at inference (math invariant w.r.t. original).
            #   - len(s) == in_features (no norm absorb): legacy fallback;
            #       column-divide by s (yields biased noise toward low-inf).
            #   - len(s) == d_pad: full WHT-rotated BBT.
            input_scale = tensors.get(base + ".input_scale")
            if input_scale is None:
                input_scale = bbt_scales[base]
            input_scale = input_scale.to(torch.float32)
            scale_len = input_scale.numel()
            assert target_in is not None, f"unknown public_in for {base}"
            if scale_len == target_in:
                # No-rotation. Always treat as norm-absorbed: keep the scaled
                # weights as-is. (For legacy fallback mode users would manually
                # divide; that path produced bad PPL.)
                w = _dequantize_layer(qw, sc, qz, bits, group_size, sym,
                                      target_in_features=target_in).to(torch.float16)
            else:
                # WHT-rotated: do NOT truncate during dequant. Unrotate first.
                W_padded = _dequantize_layer(qw, sc, qz, bits, group_size, sym,
                                             target_in_features=None).to(torch.float32)
                w = _bbt_unrotate(W_padded, input_scale, target_in).to(torch.float16)
            n_bbt += 1
        else:
            w = _dequantize_layer(qw, sc, qz, bits, group_size, sym,
                                  target_in_features=target_in)

        # spectral_pca un-rotation: W_orig[h] = U[h] @ W_quant_rotated[h]
        # per-head row block. Applied AFTER any BBT input-scale handling so
        # the per-head row rotation acts on the (out_features, target_in)
        # weight in its plain orientation.
        if pca_U is not None:
            w_f32 = w.to(torch.float32)
            n_groups, head_dim, _ = pca_U.shape
            assert w_f32.shape[0] == n_groups * head_dim, (
                f"{base}: out_features={w_f32.shape[0]} != "
                f"n_groups({n_groups}) * head_dim({head_dim})"
            )
            U = pca_U.to(torch.float32)
            w_unrot = w_f32.clone()
            for h in range(n_groups):
                lo, hi = h * head_dim, (h + 1) * head_dim
                w_unrot[lo:hi, :] = U[h] @ w_f32[lo:hi, :]
            w = w_unrot.to(torch.float16)
            n_pca += 1

        # spectral_pca_2d un-rotation: invert the per-(head, pair) 2x2 rotation.
        # Forward applied U^T to (lo, hi); to invert, apply U:
        #   old_lo = u00*new_lo + u01*new_hi
        #   old_hi = u10*new_lo + u11*new_hi
        if pair_U is not None:
            w_f32 = w.to(torch.float32)
            n_groups, half, _, _ = pair_U.shape
            head_dim = 2 * half
            assert w_f32.shape[0] == n_groups * head_dim, (
                f"{base}: out_features={w_f32.shape[0]} != "
                f"n_groups({n_groups}) * head_dim({head_dim})"
            )
            U = pair_U.to(torch.float32)  # (n_groups, half, 2, 2)
            in_features = w_f32.shape[1]
            Wph = w_f32.view(n_groups, head_dim, in_features)
            W_lo = Wph[:, :half, :]
            W_hi = Wph[:, half:, :]
            u00 = U[..., 0, 0].unsqueeze(-1)
            u01 = U[..., 0, 1].unsqueeze(-1)
            u10 = U[..., 1, 0].unsqueeze(-1)
            u11 = U[..., 1, 1].unsqueeze(-1)
            old_lo = u00 * W_lo + u01 * W_hi
            old_hi = u10 * W_lo + u11 * W_hi
            w_unrot = torch.cat([old_lo, old_hi], dim=1).reshape(
                n_groups * head_dim, in_features)
            w = w_unrot.to(torch.float16)
            n_pca += 1
        new_tensors[base + ".weight"] = w

    print(f"[dequant] wrote {len(new_tensors)} tensors ({pass_through} pass-through + "
          f"{len(layer_bases)} dequantized; {n_bbt} BBT-unrotated; "
          f"{n_pca} per-head PCA-unrotated)")

    # Save as safetensors. If the un-quantized FP16 footprint would exceed
    # ~5 GB, shard. HF default is ~5 GB / shard; we follow the same.
    from safetensors.torch import save_file
    total_bytes = sum(t.element_size() * t.numel() for t in new_tensors.values())
    SHARD_THRESHOLD = 5 * (1024 ** 3)
    if total_bytes <= SHARD_THRESHOLD:
        save_file(new_tensors, str(dst / "model.safetensors"))
    else:
        # Sharded save: bin tensors greedily into shards under the threshold
        # and write a HuggingFace-style index json so AutoModel loads cleanly.
        n_shards = (total_bytes + SHARD_THRESHOLD - 1) // SHARD_THRESHOLD
        target = total_bytes // n_shards + 1
        shards: list[dict[str, torch.Tensor]] = [{}]
        cur = 0
        for k in sorted(new_tensors.keys()):
            v = new_tensors[k]
            sz = v.element_size() * v.numel()
            if cur + sz > target and shards[-1]:
                shards.append({})
                cur = 0
            shards[-1][k] = v
            cur += sz
        weight_map = {}
        for i, shard in enumerate(shards, 1):
            fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            save_file(shard, str(dst / fname))
            for k in shard:
                weight_map[k] = fname
        index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
        with open(dst / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
        print(f"[dequant] sharded output: {len(shards)} shards "
              f"(total {total_bytes / (1024**3):.2f} GB)")

    # Copy non-weight files; strip quantization_config from config.json
    for src_file in src.iterdir():
        # Skip ALL safetensors and their index — we rewrote them above.
        if src_file.name == "model.safetensors" or \
           src_file.name == "model.safetensors.index.json" or \
           (src_file.name.startswith("model-") and src_file.name.endswith(".safetensors")):
            continue
        if src_file.name == "config.json":
            with open(src_file) as f:
                cfg = json.load(f)
            cfg.pop("quantization_config", None)
            with open(dst / "config.json", "w") as f:
                json.dump(cfg, f, indent=2)
            continue
        if src_file.name == "quantization_config.json":
            continue
        if src_file.is_file():
            shutil.copy2(src_file, dst / src_file.name)

    return {
        "src": str(src), "dst": str(dst), "bits": bits, "group_size": group_size,
        "n_dequantized": len(layer_bases), "n_total_tensors": len(new_tensors),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    args = p.parse_args()
    summary = dequantize_checkpoint(args.src, args.dst)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
