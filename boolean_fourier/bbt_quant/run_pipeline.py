"""
End-to-end BBT-quant CLI:

    calibration -> WHT rotation -> BBT scaling -> auto-round W2A16
                -> (optional) OpenVINO IR export

Designed to be runnable from WSL for the quantization half; the OpenVINO
export is skipped gracefully if the tools are missing (so you can finish it
on Windows later with `python -m bbt_quant.export_openvino`).

Typical WSL call (quantize only):
    python -m bbt_quant.run_pipeline \
        --model HuggingFaceTB/SmolLM-135M \
        --output /mnt/c/Users/gogip/bbt_quant_out/smollm_w2_bbt \
        --bits 2 --alpha 0.5 --calib-samples 128 --skip-ov

Windows follow-up (export + eval on NPU):
    python -m bbt_quant.export_openvino \
        --autoround-dir C:\\Users\\gogip\\bbt_quant_out\\smollm_w2_bbt\\autoround \
        --output-dir   C:\\Users\\gogip\\bbt_quant_out\\smollm_w2_bbt\\openvino \
        --bits 2
    python -m bbt_quant.eval_windows \
        --model-dir C:\\Users\\gogip\\bbt_quant_out\\smollm_w2_bbt\\openvino \
        --backend openvino --device NPU \
        --max-tokens 8192
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

from .autoround_bbt import run_bbt_autoround


# -----------------------------------------------------------------------------
# Calibration data loader (wikitext-2, streaming-friendly)
# -----------------------------------------------------------------------------

def _build_calib_iter(tokenizer, n_samples: int, seqlen: int) -> Iterable:
    """Yield {'input_ids': LongTensor[1, seqlen]} batches from wikitext-2 train."""
    from datasets import load_dataset  # type: ignore
    import torch

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]  # (T,)

    yielded = 0
    offset = 0
    total = input_ids.numel()
    while yielded < n_samples and offset + seqlen + 1 < total:
        chunk = input_ids[offset : offset + seqlen].unsqueeze(0)  # (1, seqlen)
        offset += seqlen
        yielded += 1
        yield {"input_ids": chunk}


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(description="BBT x auto-round x OpenVINO pipeline")
    p.add_argument("--model", required=True,
                   help="HF model id or local path (e.g. HuggingFaceTB/SmolLM-135M)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output dir (writes <output>/autoround/ and <output>/openvino/)")
    p.add_argument("--bits", type=int, default=2, choices=[2, 3, 4])
    p.add_argument("--alpha", type=float, default=0.5,
                   help="BBT scale exponent: 0 disables BBT, 0.5 is default, 1 aggressive")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--calib-samples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--device", default="auto",
                   help="torch device for calibration & auto-round: auto, cpu, cuda, xpu")
    p.add_argument("--only-substrings", default=None,
                   help="comma-separated substrings; if set, rotate only matching linears "
                        "(e.g. 'mlp.,self_attn.')")
    p.add_argument("--skip-substrings",
                   default="lm_head,embed_tokens,embed_positions")
    p.add_argument("--no-alg-ext", action="store_true",
                   help="disable auto-round --enable_alg_ext (improved INT2)")
    p.add_argument("--iters", type=int, default=None,
                   help="auto-round inner-iter cap per block (default: auto-round's 200)")
    p.add_argument("--bbt-mode", default="no_rotation",
                   choices=["no_rotation", "spectral", "spectral_pca",
                            "spectral_pca_2d", "v1_pad_pre_hook"],
                   help="BBT prep flavour: no_rotation (norm-absorbed, default), "
                        "spectral (WHT-rotated, requires wrapper patch), "
                        "spectral_pca (per-head PCA + matrix-Gamma q_norm/k_norm "
                        "for Qwen3-style architectures), "
                        "spectral_pca_2d (per-(head,pair) 2x2 PCA, RoPE-compatible, "
                        "for non-q_norm architectures like SmolLM/Qwen2.5), "
                        "v1_pad_pre_hook (legacy, broken)")
    p.add_argument("--skip-ov", action="store_true",
                   help="skip the OpenVINO export step (run on Windows later)")
    p.add_argument("--dry-run", action="store_true",
                   help="load model + tokenizer, print plan, exit before quantization")
    args = p.parse_args(argv)

    out_root = Path(args.output)
    autoround_dir = out_root / "autoround"
    openvino_dir = out_root / "openvino"
    out_root.mkdir(parents=True, exist_ok=True)

    # Lazy imports so --dry-run works even if torch isn't in the env.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"
    print(f"[pipeline] device resolved to: {device}")

    print(f"[pipeline] loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    # Plan
    n_linears = sum(1 for _ in model.modules() if _.__class__.__name__ == "Linear")
    print(f"[pipeline] model has ~{n_linears} Linear layers")
    print(f"[pipeline] will quantize to W{args.bits}A16 with alpha={args.alpha}")
    print(f"[pipeline] output: {out_root}")

    if args.dry_run:
        print("[pipeline] --dry-run: stopping before quantization.")
        return 0

    skip = tuple(s.strip() for s in args.skip_substrings.split(",") if s.strip())
    only = (
        tuple(s.strip() for s in args.only_substrings.split(",") if s.strip())
        if args.only_substrings
        else None
    )

    calib_iter = list(_build_calib_iter(tokenizer, args.calib_samples, args.seqlen))

    # Pass BBT mode to run_bbt_autoround via module-level attribute
    from .autoround_bbt import run_bbt_autoround
    run_bbt_autoround._rotation_mode = args.bbt_mode
    if args.iters is not None:
        run_bbt_autoround._iters_override = args.iters
    print(f"[pipeline] BBT mode: {args.bbt_mode}; iters={args.iters or 'default'}")

    summary = run_bbt_autoround(
        model=model,
        tokenizer=tokenizer,
        calib_iter=calib_iter,
        output_dir=autoround_dir,
        bits=args.bits,
        alpha=args.alpha,
        group_size=args.group_size,
        enable_alg_ext=not args.no_alg_ext,
        device=device,
        max_calib_batches=args.calib_samples,
        skip_substrings=skip,
        only_substrings=only,
        save_metadata=True,
        verbose=True,
    )
    print(f"[pipeline] auto-round done. Summary:\n{json.dumps(summary, indent=2)}")

    if args.skip_ov:
        print("[pipeline] --skip-ov set; OpenVINO export skipped.")
        print(f"[pipeline] run later on Windows:")
        print(f"  python -m bbt_quant.export_openvino --autoround-dir {autoround_dir} "
              f"--output-dir {openvino_dir} --bits {args.bits}")
        return 0

    # OV export (best-effort)
    try:
        from .export_openvino import convert_to_openvino
        ov_summary = convert_to_openvino(autoround_dir, openvino_dir, bits=args.bits)
        print(f"[pipeline] OV export done:\n{json.dumps(ov_summary, indent=2)}")
    except Exception as e:
        print(f"[pipeline] OV export skipped ({e}). "
              f"Run on Windows later with `python -m bbt_quant.export_openvino`.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
