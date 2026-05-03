# bbt_quant

Influence-adaptive LLM weight quantization that bridges the Banach-Butterfly
Transform paper (`paper/bbt_paper.tex`) with three industrial tools:

| Tool | Role |
|---|---|
| **Intel auto-round** | W2A16 SignRound quantization (unchanged internals) |
| **OpenXLA / JAX** | Backend for the BBT spectral analytics already used in `bbt/` |
| **OpenVINO** | Deployment on Intel NPU / Arc GPU from Windows |

## What it does

The BBT paper proves that, for Boolean functions, a WHT per-layer geometry
with exponents `p_l = 1 + Inf_l(f)` exposes a finer difficulty structure
than total influence alone. We port that intuition to real-valued weight
matrices with three ingredients:

1. **WHT rotation** of every `nn.Linear`'s input basis (QuaRot/SpinQuant
   convention). Math is unchanged at full precision; the weight matrix
   now lives in the WHT spectral basis.
2. **Real-valued influence** per spectral coordinate: activation energy
   `E[h_l^2]` on a calibration set, normalised to sum to 1.
3. **AWQ-style pre-scaling**: scales `s_l = (d*Inf_l + 1)^alpha` applied
   column-wise on the (rotated) weight, inverse scales on the input. This
   biases auto-round's block-wise L2 reconstruction loss toward preserving
   high-influence channels — no internal loss patching required.

auto-round then runs unchanged, producing a standard W2A16 checkpoint that
OpenVINO knows how to consume via `optimum-intel`.

## Deployment topology (WSL / Windows)

WSL does **not** expose the Intel NPU device to OpenVINO's NPU plugin.
This package is structured so the heavy compute (calibration + auto-round)
can run in WSL on the Intel XPU (B280) and the IR export + NPU inference
run on native Windows Python.

```
┌──────────────── WSL (Ubuntu) ───────────────┐    ┌─── Windows ───┐
│                                             │    │               │
│  python -m bbt_quant.run_pipeline           │    │               │
│      --model HuggingFaceTB/SmolLM-135M      │    │               │
│      --output /mnt/c/Users/gogip/bbt_out    │───▶│  bbt_out/     │
│      --bits 2 --skip-ov                     │    │   autoround/  │
│                                             │    │               │
│  (calibration, WHT rotation, BBT scaling,   │    │               │
│   auto-round W2A16 — XPU)                   │    │               │
└─────────────────────────────────────────────┘    │               │
                                                   │               │
                                                   │  python -m bbt_quant.export_openvino │
                                                   │      --autoround-dir bbt_out/autoround │
                                                   │      --output-dir   bbt_out/openvino   │
                                                   │      --bits 2                           │
                                                   │                                         │
                                                   │  python -m bbt_quant.eval_windows       │
                                                   │      --model-dir bbt_out/openvino       │
                                                   │      --backend openvino --device NPU    │
                                                   └─────────────────────────────────────────┘
```

The default cross-boundary path is `/mnt/c/Users/gogip/bbt_quant_out/` in
WSL, which is `C:\Users\gogip\bbt_quant_out\` in Windows.

## Installation

### WSL side (quantization)

Use the existing Intel XPU venv:

```bash
source /home/gogip/xpu_venv/bin/activate
pip install "auto-round"                        # SignRound, W2A16, alg_ext
pip install "transformers>=4.40" datasets
pip install "torch" --index-url https://download.pytorch.org/whl/xpu  # if not already
```

### Windows side (export + deploy)

```powershell
# Python 3.11 recommended
python -m venv %USERPROFILE%\ovvenv
%USERPROFILE%\ovvenv\Scripts\activate
pip install "optimum-intel[openvino,nncf]>=1.20"
pip install openvino openvino-tokenizers
pip install "transformers>=4.40" datasets torch
```

The NPU plugin is bundled with OpenVINO on Intel Core Ultra (Meteor Lake /
Lunar Lake) systems.

## Quick reference

### WSL: quantize (BBT + auto-round W2A16)

```bash
python -m bbt_quant.run_pipeline \
  --model HuggingFaceTB/SmolLM-135M \
  --output /mnt/c/Users/gogip/bbt_quant_out/smollm_w2_bbt \
  --bits 2 --alpha 0.5 --calib-samples 128 --device xpu --skip-ov
```

### Windows: export + eval on NPU

```powershell
python -m bbt_quant.export_openvino ^
  --autoround-dir C:\Users\gogip\bbt_quant_out\smollm_w2_bbt\autoround ^
  --output-dir   C:\Users\gogip\bbt_quant_out\smollm_w2_bbt\openvino ^
  --bits 2

python -m bbt_quant.eval_windows ^
  --model-dir C:\Users\gogip\bbt_quant_out\smollm_w2_bbt\openvino ^
  --backend openvino --device NPU ^
  --out       C:\Users\gogip\bbt_quant_out\smollm_w2_bbt\eval_npu.json
```

### Baseline comparison (vanilla auto-round W2, no BBT)

Set `--alpha 0` to disable BBT scaling while still running the same
pipeline (the WHT rotation is still applied because it does not change
full-precision math — it changes only the basis auto-round quantizes in;
if you want a pure auto-round baseline, see `--only-substrings ''` plus
`--alpha 0` and skip the rotation pass by patching `only_substrings` to
an empty tuple in `autoround_bbt.prepare_model_for_autoround`).

## Package layout

```
bbt_quant/
├── __init__.py
├── influence.py          # BBT influence on real weights (WHT activation energy)
├── wht_rotation.py       # WHTRotatedLinear + model walker (math-preserving)
├── autoround_bbt.py      # Pre-scaling + auto-round glue (no internal patching)
├── export_openvino.py    # optimum-intel primary, NNCF fallback
├── eval_windows.py       # hf / hf_xpu / openvino backends; NPU-capable
├── run_pipeline.py       # CLI orchestrator
└── README.md             # this file
```

## Relation to the BBT paper

The paper's `Inf_l(f)` is defined only for Boolean `f: {-1,+1}^n -> {-1,+1}`.
Here we use an analog on real-valued weights:

    Inf_l(L) := E[h_l^2] / sum_k E[h_k^2],   h = H_{d_in} x / sqrt(d_in)

which agrees with the Boolean definition up to normalization when the
"truth table" happens to be Boolean. This real-valued extension is
research-grade and not derived from the paper's proofs — it is the
natural mapping documented in `influence.py`.

## Relation to existing llm_wht/

The sibling package `boolean_fourier/llm_wht/` implements a direct
WHT+ternary QAT pipeline (hand-rolled rounding, tier-based curriculum).
`bbt_quant/` is deliberately independent so the in-flight 6-model
BBT-QAT benchmark is not perturbed. Nothing in `llm_wht/` is modified.
