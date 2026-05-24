"""
Canonical wikitext-2 perplexity evaluator for BBT-quantized (and reference) LLMs.

This is the *single source of truth* for the perplexity numbers in the
companion paper. It replaces the ad-hoc inline `python -c` snippets that were
previously used and never saved -- which led to a reproduction scare when the
same fp16 weights gave 119.31 PPL under transformers 4.57.6 but 72.97 under
transformers 5.6 (the Qwen2 fp16 forward path changed between versions, and W2
models are numerically fragile enough to amplify that into a ~60% swing).

LESSON BAKED IN HERE: every result JSON records the full environment
(transformers version, torch version, device, dtype, window, token cap, and the
model's own quantization_config). If a number ever looks surprising again,
the JSON tells you exactly which stack produced it.

--------------------------------------------------------------------------------
Reproducing the paper's headline W2A16 numbers
--------------------------------------------------------------------------------
The paper's table was produced with:
    transformers 4.57.6, torch fp16, auto-round 0.12.2 (enable_alg_ext=False),
    8192-token cap, sliding window 1024.

Run (from the Windows ovvenv, which has transformers 4.57.6):
    python -m boolean_fourier.bbt_quant.eval_ppl \
        --model C:\\Users\\gogip\\bbt_quant_out\\qwen25_0_5b_vanilla\\fp16 \
        --paper-repro --label "Qwen2.5-0.5B vanilla" \
        --out C:\\Users\\gogip\\bbt_quant_out\\qwen25_0_5b_vanilla\\eval.json

`--paper-repro` is exactly `--max-tokens 8192 --seqlen 1024`.

Full-test-set evaluation (the more standard metric, used for v2):
    python -m boolean_fourier.bbt_quant.eval_ppl --model <dir> --out <json>
    (no token cap; sliding window defaults to 2048)

The script accepts either an HF model id (e.g. Qwen/Qwen2.5-0.5B for an FP16
reference) or a local directory (auto-round or dequantized fp16 checkpoint).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import time
from pathlib import Path
from typing import Dict, Optional


def _collect_env(model_path: str, device: str, dtype_name: str) -> Dict:
    """Record everything needed to reproduce the number later."""
    import torch
    import transformers

    env: Dict = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "model": model_path,
        "device": device,
        "dtype": dtype_name,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import auto_round  # type: ignore
        env["auto_round_version"] = getattr(auto_round, "__version__", "unknown")
    except Exception:
        env["auto_round_version"] = None

    # If the model dir carries a quantization_config, surface it -- this is the
    # other axis that silently changes numbers (bits / group_size / alg_ext).
    cfg_path = Path(model_path) / "config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
            if "quantization_config" in cfg:
                env["quantization_config"] = cfg["quantization_config"]
        except Exception:
            pass
    # BBT metadata sidecar, if present.
    bbt_path = Path(model_path) / "bbt_summary.json"
    if bbt_path.is_file():
        try:
            env["bbt_summary"] = json.loads(bbt_path.read_text())
        except Exception:
            pass
    return env


def evaluate(
    model_path: str,
    device: str = "xpu",
    dtype_name: str = "float16",
    max_tokens: Optional[int] = None,
    seqlen: int = 2048,
    label: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> Dict:
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]

    result = _collect_env(model_path, device, dtype_name)
    result["label"] = label
    result["seqlen"] = seqlen
    result["max_tokens"] = max_tokens
    # A tokenizer saved by a newer transformers can be unreadable by an older
    # one (e.g. list- vs dict-format special tokens between 5.x and 4.57).
    # Allow pointing the tokenizer at the base HF id while weights load locally.
    tok_src = tokenizer_path or model_path
    result["tokenizer"] = tok_src

    print(f"[eval_ppl] {label or model_path}", flush=True)
    print(f"[eval_ppl] transformers={result['transformers_version']} "
          f"torch={result['torch_version']} device={device} dtype={dtype_name} "
          f"seqlen={seqlen} max_tokens={max_tokens}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    if max_tokens is not None:
        ids = ids[:, :max_tokens]
    ids = ids.to(device)

    nlls, total = [], 0
    n_windows = max(1, ids.shape[1] // seqlen)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(n_windows):
            win = ids[:, i * seqlen : (i + 1) * seqlen]
            if win.shape[1] < 2:
                continue
            logits = model(win).logits
            sl = logits[:, :-1, :].contiguous()
            lb = win[:, 1:].contiguous()
            loss = F.cross_entropy(
                sl.view(-1, sl.size(-1)), lb.view(-1), reduction="mean"
            )
            nlls.append(loss.item() * lb.numel())
            total += lb.numel()
    elapsed = time.perf_counter() - t0
    ppl = math.exp(sum(nlls) / max(1, total))

    result.update(
        {
            "ppl_wikitext2": ppl,
            "n_windows": n_windows,
            "total_eval_tokens": total,
            "elapsed_sec": round(elapsed, 3),
        }
    )
    print(f"[eval_ppl] PPL={ppl:.4f}  ({total} tokens, {n_windows} windows, "
          f"{elapsed:.1f}s)", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="HF model id (FP16 reference) or local dir (quantized / dequantized)")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer source (HF id or dir); defaults to --model. "
                        "Use the base HF id if the checkpoint's tokenizer was "
                        "saved by an incompatible transformers version.")
    p.add_argument("--device", default="xpu", choices=["cpu", "cuda", "xpu"])
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-tokens", type=int, default=None,
                   help="cap eval token count (None = full wikitext-2 test split)")
    p.add_argument("--seqlen", type=int, default=2048,
                   help="sliding-window length (paper headline used 1024)")
    p.add_argument("--paper-repro", action="store_true",
                   help="shortcut for the paper's headline setup: "
                        "--max-tokens 8192 --seqlen 1024")
    p.add_argument("--label", default=None, help="human label stored in the JSON")
    p.add_argument("--out", type=Path, default=None, help="write result JSON here")
    args = p.parse_args()

    max_tokens = args.max_tokens
    seqlen = args.seqlen
    if args.paper_repro:
        max_tokens = 8192
        seqlen = 1024

    result = evaluate(
        model_path=str(args.model),
        device=args.device,
        dtype_name=args.dtype,
        max_tokens=max_tokens,
        seqlen=seqlen,
        label=args.label,
        tokenizer_path=args.tokenizer,
    )
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("quantization_config", "bbt_summary")}, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[eval_ppl] wrote {args.out}")


if __name__ == "__main__":
    main()
