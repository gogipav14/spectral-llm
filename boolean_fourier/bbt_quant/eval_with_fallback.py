"""
Thin wrapper around eval_windows.run_eval that tries OpenVINO NPU first
and falls back to GPU, then CPU. Writes the final JSON to --out and prints
which device actually ran.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from bbt_quant.eval_windows import run_eval


def try_device(model_dir: Path, backend: str, device: str,
               max_tokens: Optional[int]) -> tuple[Optional[dict], Optional[str]]:
    t0 = time.perf_counter()
    try:
        result = run_eval(model_dir, backend, device, max_tokens)
        result["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        return result, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--backend", default="openvino")
    p.add_argument("--devices", default="NPU,GPU,CPU",
                   help="comma-separated fallback order")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    summary = {
        "model_dir": str(args.model_dir),
        "backend": args.backend,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "attempts": [],
    }

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    for device in devices:
        print(f"[eval] trying {args.backend}/{device} ...", flush=True)
        result, err = try_device(args.model_dir, args.backend, device, args.max_tokens)
        attempt = {"device": device, "ok": result is not None}
        if result:
            summary.update(result)
            summary["device"] = device
            attempt["ppl_wikitext2"] = result.get("ppl_wikitext2")
            attempt["tokens_per_sec"] = result.get("tokens_per_sec")
            attempt["elapsed_sec"] = result.get("elapsed_sec")
        else:
            attempt["error"] = err
            print(f"[eval] {device} failed: {err.splitlines()[0]}", flush=True)
        summary["attempts"].append(attempt)
        if result:
            print(f"[eval] {device} OK: PPL={result['ppl_wikitext2']:.4f} "
                  f"tok/s={result['tokens_per_sec']:.2f}", flush=True)
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "attempts"}, indent=2))
    return 0 if "device" in summary else 1


if __name__ == "__main__":
    sys.exit(main())
