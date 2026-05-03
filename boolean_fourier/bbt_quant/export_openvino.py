"""
Export a BBT-auto-round quantized directory to OpenVINO IR.

This module is designed to be importable from both WSL and native Windows
Python. It only hard-depends on torch for loading; optimum-intel /
openvino / nncf are imported inside the functions so that a WSL-only
environment can still run the calibration + quantization step and then
hand off to a Windows environment for the export.

Two export paths are tried in order:
  1. `optimum.intel.OVModelForCausalLM.from_pretrained(autoround_dir, export=True)`
     This is the cleanest route if optimum-intel supports the auto_round
     format directly.
  2. Fallback: load auto-round model as an AutoModelForCausalLM, run
     `nncf.compress_weights(mode=...)`, serialize with openvino.save_model.
     If bits=2 is unsupported, this path silently falls back to INT4
     and flags it in the returned summary.

The returned dict `{"ir_dir": Path, "mode": str, "bits": int, "fallback": bool}`
is also written to `bbt_export_summary.json` next to the IR files.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional


def _have_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Path 1: optimum-intel direct export
# -----------------------------------------------------------------------------

def _try_optimum_intel(autoround_dir: Path, output_dir: Path, bits: int) -> Optional[Dict]:
    if not _have_module("optimum.intel") or not _have_module("openvino"):
        return None
    try:
        from optimum.intel import OVModelForCausalLM  # type: ignore

        # The auto_round format stores its config under quantization_config.
        # optimum-intel >= 1.20 has a loader that detects this and routes to
        # the auto-round-specific OV path.
        model = OVModelForCausalLM.from_pretrained(
            str(autoround_dir),
            export=True,
            trust_remote_code=True,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        return {
            "ir_dir": str(output_dir),
            "mode": f"optimum-intel direct export (W{bits})",
            "bits": bits,
            "fallback": False,
        }
    except Exception as exc:  # pragma: no cover
        print(f"[bbt-export] optimum-intel direct export failed: {exc}")
        return None


# -----------------------------------------------------------------------------
# Path 2: NNCF compress_weights fallback
# -----------------------------------------------------------------------------

def _try_nncf_fallback(
    autoround_dir: Path, output_dir: Path, bits: int
) -> Optional[Dict]:
    if not _have_module("nncf") or not _have_module("openvino"):
        return None
    try:
        import nncf  # type: ignore
        import openvino as ov  # type: ignore
        from optimum.exporters.openvino import export_from_model  # type: ignore
        from transformers import AutoModelForCausalLM  # type: ignore

        model = AutoModelForCausalLM.from_pretrained(
            str(autoround_dir), trust_remote_code=True
        )

        # Export to OV first (FP32), then compress weights.
        tmp_ov_dir = output_dir.with_suffix(".fp32_tmp")
        tmp_ov_dir.mkdir(parents=True, exist_ok=True)
        export_from_model(model=model, output=str(tmp_ov_dir))

        core = ov.Core()
        # Heuristic: openvino model is 'openvino_model.xml' after optimum export.
        xml_paths = list(tmp_ov_dir.rglob("openvino_model.xml"))
        if not xml_paths:
            xml_paths = list(tmp_ov_dir.rglob("*.xml"))
        if not xml_paths:
            raise RuntimeError(f"No .xml found under {tmp_ov_dir}")
        ov_model = core.read_model(str(xml_paths[0]))

        # Pick NNCF mode. INT4_SYM is the most aggressive widely-supported option.
        # INT2 is not a public NNCF mode at the time of writing -> fall back.
        effective_bits = bits
        fallback = False
        try:
            mode = nncf.CompressWeightsMode.INT4_SYM if bits <= 4 else nncf.CompressWeightsMode.INT8_SYM
            # If a public INT2 mode appears upstream, prefer it:
            int2_mode = getattr(nncf.CompressWeightsMode, "INT2_SYM", None)
            if bits == 2 and int2_mode is not None:
                mode = int2_mode
            elif bits == 2:
                fallback = True
                effective_bits = 4
        except Exception:
            mode = nncf.CompressWeightsMode.INT4_SYM
            fallback = bits != 4
            effective_bits = 4

        compressed = nncf.compress_weights(ov_model, mode=mode)
        output_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(compressed, str(output_dir / "openvino_model.xml"))

        # Copy tokenizer / config files alongside
        for src in autoround_dir.iterdir():
            if src.suffix in {".json", ".txt", ".model"} or src.name.startswith("tokenizer"):
                try:
                    shutil.copy2(src, output_dir / src.name)
                except Exception:
                    pass

        # Clean temp
        shutil.rmtree(tmp_ov_dir, ignore_errors=True)

        return {
            "ir_dir": str(output_dir),
            "mode": f"nncf compress_weights (effective W{effective_bits})",
            "bits": effective_bits,
            "fallback": fallback,
        }
    except Exception as exc:  # pragma: no cover
        print(f"[bbt-export] nncf fallback failed: {exc}")
        return None


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def convert_to_openvino(
    autoround_dir: Path,
    output_dir: Path,
    bits: int = 2,
) -> Dict:
    """
    Convert an auto-round-quantized directory to OpenVINO IR.

    Args:
        autoround_dir: path produced by BBTAutoRound / quantize_with_autoround.
        output_dir: where to write the OV IR (xml + bin + configs).
        bits: requested weight bit-width. Falls back to 4 if the selected
              OpenVINO/NNCF stack does not support lower.

    Returns:
        Summary dict, also written as `bbt_export_summary.json` in output_dir.

    Raises:
        RuntimeError if neither export path is available.
    """
    autoround_dir = Path(autoround_dir)
    output_dir = Path(output_dir)

    if not autoround_dir.exists():
        raise FileNotFoundError(autoround_dir)

    summary: Optional[Dict] = None
    for attempt in (_try_optimum_intel, _try_nncf_fallback):
        summary = attempt(autoround_dir, output_dir, bits)
        if summary is not None:
            break
    if summary is None:
        raise RuntimeError(
            "No export path available. Install one of:\n"
            "  pip install optimum-intel[openvino,nncf]\n"
            "  or a newer openvino + nncf with INT4 weight compression."
        )

    # Copy BBT metadata next to the IR so the deploy script can find it.
    for meta in ("bbt_influences.pt", "bbt_scales.pt", "bbt_summary.json"):
        src = autoround_dir / meta
        if src.exists():
            try:
                shutil.copy2(src, output_dir / meta)
            except Exception:
                pass

    with open(output_dir / "bbt_export_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--autoround-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--bits", type=int, default=2)
    args = p.parse_args()
    try:
        result = convert_to_openvino(args.autoround_dir, args.output_dir, args.bits)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
