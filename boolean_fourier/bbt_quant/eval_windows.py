"""
Evaluation harness for BBT-quantized LLMs — designed to run from Windows so
OpenVINO can target the NPU plugin (WSL does not expose the NPU device).

Supports three backends:
  --backend hf        : vanilla HuggingFace PyTorch (CPU or any torch-device)
  --backend hf_xpu    : HuggingFace PyTorch + Intel XPU (torch-xpu)
  --backend openvino  : OpenVINO runtime, device ∈ {NPU, GPU, GPU.1, CPU, AUTO}

Metrics:
  wikitext-2 perplexity (sliding-window NLL over the full test split by default;
                         can be capped with --max-tokens for smoke tests)
  tokens/sec on a fixed decode workload

All paths accept either POSIX-style (/mnt/c/Users/gogip/...) or Windows-style
(C:\\Users\\gogip\\...) strings; pathlib.Path handles both.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Dict, Iterable, Optional


# -----------------------------------------------------------------------------
# Dataset loading (wikitext-2)
# -----------------------------------------------------------------------------

def _load_wikitext(tokenizer, seqlen: int = 2048) -> "torch.Tensor":
    import torch
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    return enc["input_ids"]  # (1, T)


# -----------------------------------------------------------------------------
# Backend: HuggingFace PyTorch
# -----------------------------------------------------------------------------

def _eval_hf(model_dir: Path, device: str, max_tokens: Optional[int]) -> Dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    ids = _load_wikitext(tokenizer).to(device)
    if max_tokens is not None:
        ids = ids[:, :max_tokens]
    seqlen = 2048
    nlls = []
    n_windows = max(1, ids.shape[1] // seqlen)
    with torch.no_grad():
        for i in range(n_windows):
            win = ids[:, i * seqlen : (i + 1) * seqlen]
            if win.shape[1] < 2:
                continue
            logits = model(win).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = win[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
            nlls.append(loss.item() * shift_labels.numel())
    total_tokens = n_windows * (seqlen - 1)
    ppl = math.exp(sum(nlls) / max(1, total_tokens))

    # Throughput: short decode
    prompt = tokenizer("The quick brown fox ", return_tensors="pt").input_ids.to(device)
    for _ in range(2):  # warmup
        model.generate(prompt, max_new_tokens=16, do_sample=False)
    t0 = time.perf_counter()
    out = model.generate(prompt, max_new_tokens=64, do_sample=False)
    elapsed = time.perf_counter() - t0
    new_toks = out.shape[1] - prompt.shape[1]
    tok_per_s = new_toks / max(elapsed, 1e-6)

    return {
        "ppl_wikitext2": ppl,
        "tokens_per_sec": tok_per_s,
        "total_eval_tokens": total_tokens,
        "n_windows": n_windows,
    }


# -----------------------------------------------------------------------------
# Backend: OpenVINO
# -----------------------------------------------------------------------------

def _eval_openvino(model_dir: Path, device: str, max_tokens: Optional[int]) -> Dict:
    # optimum-intel gives us a HF-API-compatible wrapper around OV.
    from optimum.intel import OVModelForCausalLM  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    ov_model = OVModelForCausalLM.from_pretrained(
        str(model_dir), device=device, trust_remote_code=True
    )

    # PPL via teacher-forced logits. optimum-intel's OV model exposes .forward.
    import torch
    ids = _load_wikitext(tokenizer)
    if max_tokens is not None:
        ids = ids[:, :max_tokens]
    seqlen = 1024  # NPU memory-friendly
    nlls = []
    n_windows = max(1, ids.shape[1] // seqlen)
    for i in range(n_windows):
        win = ids[:, i * seqlen : (i + 1) * seqlen]
        if win.shape[1] < 2:
            continue
        logits = ov_model(win).logits
        if not torch.is_tensor(logits):
            logits = torch.as_tensor(logits)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = win[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
        nlls.append(loss.item() * shift_labels.numel())
    total_tokens = n_windows * (seqlen - 1)
    ppl = math.exp(sum(nlls) / max(1, total_tokens))

    # Throughput
    prompt = tokenizer("The quick brown fox ", return_tensors="pt").input_ids
    for _ in range(2):
        ov_model.generate(prompt, max_new_tokens=16, do_sample=False)
    t0 = time.perf_counter()
    out = ov_model.generate(prompt, max_new_tokens=64, do_sample=False)
    elapsed = time.perf_counter() - t0
    new_toks = out.shape[1] - prompt.shape[1]
    tok_per_s = new_toks / max(elapsed, 1e-6)

    return {
        "ppl_wikitext2": ppl,
        "tokens_per_sec": tok_per_s,
        "total_eval_tokens": total_tokens,
        "n_windows": n_windows,
    }


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------

def run_eval(
    model_dir: Path,
    backend: str,
    device: str,
    max_tokens: Optional[int] = None,
) -> Dict:
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    if backend == "hf":
        return _eval_hf(model_dir, device or "cpu", max_tokens)
    if backend == "hf_xpu":
        import torch  # noqa: F401
        return _eval_hf(model_dir, device or "xpu", max_tokens)
    if backend == "openvino":
        return _eval_openvino(model_dir, device or "AUTO", max_tokens)
    raise ValueError(f"unknown backend: {backend}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="BBT-quant evaluation harness")
    p.add_argument("--model-dir", type=Path, required=True,
                   help="HF directory (backend=hf*) or OV IR directory (backend=openvino).")
    p.add_argument("--backend", choices=["hf", "hf_xpu", "openvino"], default="openvino")
    p.add_argument("--device", default=None,
                   help="cpu|cuda|xpu for hf*; NPU|GPU|GPU.1|CPU|AUTO for openvino")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="cap eval token count (for smoke tests)")
    p.add_argument("--out", type=Path, default=None,
                   help="write JSON results here")
    args = p.parse_args()

    summary = {
        "model_dir": str(args.model_dir),
        "backend": args.backend,
        "device": args.device,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    result = run_eval(args.model_dir, args.backend, args.device or "", args.max_tokens)
    summary.update(result)
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
