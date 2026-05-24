"""Compute Gini and HHI concentration indices of the BBT influence vectors
(rho_l) for every Linear layer in a model on a calibration set, then report
mean / max / per-layer summary.

This answers: "do wider-MLP models actually have more concentrated rho_l
profiles, in support of the width-correlation hypothesis in the paper?"

Usage:
    python compute_gini_hhi.py --model Qwen/Qwen2.5-0.5B --device xpu \
        --calib-samples 128 --out qwen25_05B_gini.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the existing collector. Works both as
# `python -m boolean_fourier.bbt_quant.compute_gini_hhi` and as a direct script.
import sys
from pathlib import Path as _P
_REPO_ROOT = _P(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from boolean_fourier.bbt_quant.influence import (
    _SpectralEnergyHook,
    _next_pow2,
)


def gini(x: np.ndarray) -> float:
    """Standard Gini index on a non-negative vector. Returns 0 if all equal."""
    x = np.asarray(x, dtype=np.float64).flatten()
    if (x < 0).any():
        x = x - x.min()
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def hhi(x: np.ndarray) -> float:
    """Herfindahl-Hirschman Index on a non-negative vector (sums to 1)."""
    x = np.asarray(x, dtype=np.float64).flatten()
    if x.sum() == 0:
        return 0.0
    p = x / x.sum()
    return float((p * p).sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="xpu")
    p.add_argument("--calib-samples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--skip-substrings",
                   default="lm_head,embed_tokens,embed_positions")
    args = p.parse_args()

    print(f"[gini] model={args.model} device={args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(args.device).eval()

    skip = tuple(s.strip() for s in args.skip_substrings.split(",") if s.strip())
    hooks: Dict[str, _SpectralEnergyHook] = {}
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in skip):
            continue
        d_in = module.in_features
        h = _SpectralEnergyHook(name, d_in, torch.device(args.device))
        handles.append(module.register_forward_hook(h))
        hooks[name] = h

    print(f"[gini] hooked {len(hooks)} Linear layers")

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"][0]
    seqlen = args.seqlen
    n = args.calib_samples
    with torch.no_grad():
        for i in range(n):
            off = i * seqlen
            if off + seqlen + 1 >= ids.numel():
                break
            win = ids[off : off + seqlen].unsqueeze(0).to(args.device)
            _ = model(win)
            if (i + 1) % 16 == 0:
                print(f"[gini] calibration batch {i+1}/{n}", flush=True)

    for hh in handles:
        hh.remove()

    rows: List[dict] = []
    g_all = []
    h_all = []
    for name, hook in hooks.items():
        if hook.count == 0:
            continue
        rho = (hook.energy / hook.count).cpu().numpy()
        if rho.sum() == 0:
            continue
        rho = rho / rho.sum()
        gi = gini(rho)
        hi = hhi(rho)
        rows.append({
            "name": name,
            "d_in": hook.d_in,
            "d_pad": hook.d_pad,
            "gini": gi,
            "hhi": hi,
            "uniform_hhi": 1.0 / hook.d_pad,
            "hhi_over_uniform": hi * hook.d_pad,
        })
        g_all.append(gi)
        h_all.append(hi)

    summary = {
        "model": args.model,
        "n_layers": len(rows),
        "calib_samples": args.calib_samples,
        "seqlen": args.seqlen,
        "gini_mean": float(np.mean(g_all)),
        "gini_median": float(np.median(g_all)),
        "gini_max": float(np.max(g_all)),
        "gini_min": float(np.min(g_all)),
        "hhi_mean": float(np.mean(h_all)),
        "hhi_median": float(np.median(h_all)),
        "hhi_max": float(np.max(h_all)),
        "hhi_min": float(np.min(h_all)),
        "per_layer": rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_layer"}, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
