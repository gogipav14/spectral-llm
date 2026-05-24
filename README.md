# Spectral-LLM: Boolean-Fourier theory, ternary synthesis, and extreme low-bit LLM quantization

Three connected research artifacts at the intersection of Boolean Fourier
analysis, mechanistic interpretability, and edge-deployable LLM quantization.

```
Spectral-LLM/
├── paper/                         # arXiv-ready manuscripts
│   ├── bbt_paper.tex              # The Banach-Butterfly Invariant (theory paper)
│   ├── bbt_quant_paper.tex        # Influence-Inspired Spectral Rotations (LLM application)
│   └── draft_nai.tex              # Differentiable Logic Synthesis (under review)
├── boolean_fourier/               # All Python source
│   ├── bbt_quant/                 # ★ The LLM-quantization toolkit
│   ├── phase1..phase5/            # Boolean-PTF synthesis at n=2..5
│   └── inference/                 # NPU inference code
├── experiments/                   # Reproducible experiment scripts + outputs
│   ├── n4_min_support_mu.py       # MILP minimum-support sweep at n=4
│   ├── n5_min_support_mu.py       # n=5 uniform-sample sweep
│   ├── npn_enumerate.py           # 616,126 NPN-canonical reps at n=5
│   ├── n5_npn_min_support.py      # MILP on NPN-uniform sample
│   └── *.csv, *.npy               # Saved results (see experiments/README.md)
└── README.md                      # this file
```

## The trilogy

### 1. The Banach-Butterfly Invariant — theory paper [`paper/bbt_paper.tex`]

An **influence-adaptive Banach geometry** on the Walsh-Hadamard butterfly
factorization (a function-dependent profile, not a new linear transform). For
a Boolean function $f:\{-1,+1\}^n \to \{-1,+1\}$ with coordinate influences
$\mathrm{Inf}_\ell(f)$, assign $p_\ell = 1 + \mathrm{Inf}_\ell(f)$ to butterfly
layer $\ell$, yielding the contraction invariant
$\mu(f) = \prod_\ell 2^{-\mathrm{Inf}_\ell/(1+\mathrm{Inf}_\ell)}$.

**What's in it:**
- Exact butterfly $\ell_p$ operator norms via Riesz-Thorin + duality
- Strict Schur-convexity of $\mu$ (modulo permutation) on the influence vector
- Algebraic / non-polynomial position of $\mu$ in the Fourier-data ring
- **MILP minimum-support certificates for all 65,536 Boolean functions at n=4**:
  mean 6.42, max 9, all-odd by a parity argument
- **NPN-canonical enumeration of all 616,126 representatives at n=5** (matching
  OEIS A000370) and minimum-support MILP on a 10,000-sample
- Empirical $\mu$-vs-support correlation: strong conditional Spearman
  $\rho = +0.571$ at $n=4$ in the largest fixed-influence stratum, **reverses to
  $\rho \approx -0.38$ at $n=5$** under both function-uniform and NPN-canonical
  sampling. $\mu$ is a valid Schur-convex concentration invariant but not a
  universal monotone predictor of minimum support across $n$.

### 2. Influence-Inspired Spectral Rotations for Extreme Low-Bit LLM Quantization — application paper [`paper/bbt_quant_paper.tex`]

A math-invariant pre-quantization transformation: WHT-rotate each linear
layer's weight matrix and per-channel-scale by spectral activation energy
before handing off to Intel `auto-round`.

**Headline empirical results (W2A16, group size 64):**

| Model | Vanilla PPL | BBT-spectral PPL | Δ |
|---|---:|---:|---:|
| SmolLM-135M | 81.03 | **45.11** | −44.3% |
| SmolLM-360M | 43.21 | **36.55** | −15.4% |
| Qwen2.5-0.5B | 119.31 | **50.22** | **−57.9%** |
| Qwen2.5-1.5B | 36.93 | **28.16** | −23.7% |

**Three architectural extensions** for model families the basic recipe
initially failed on:

- **Spectral-PCA (Route A)**: per-head PCA matrix-Γ replacement of `q_norm` /
  `k_norm` for Qwen3-style attention. Qwen3-0.6B drops vanilla 136.76 → **88.99**
  (−35%). The asymmetric construction $\Gamma_h = \mathrm{diag}(\gamma) U_h$ is
  required for math invariance through position-dependent RoPE.
- **Pair-PCA (Route A-2D)**: per-pair SO(2) rotations that commute with RoPE,
  for non-`q_norm` architectures. Qwen2.5-1.5B drops 36.93 → **21.84** (−41%).
- **MoE-aware ScaleTarget adapter**: handles fused 3D experts (Qwen2-MoE,
  DeepSeek-V4, poolside Laguna). The Laguna-fuzzing experiment surfaced a
  previously-undetected `g_proj` input-side attention-gate bug.

**Bit-width ablation** on Qwen2.5-1.5B (W2 vs W4): redistribution payoff scales
with the per-channel quantization-noise budget (−41% at W2, +0.06 at W4 within
noise) — consistent with the theory paper's Schur-convexity intuition.

**Cross-device deployment** through OpenVINO IR on the same machine
(Core Ultra 5 225F + Arc B580 + AI Boost NPU): PPL invariant to ±0.1 across
NPU + dGPU + CPU; full throughput / first-token-latency table for 11
model/variant combinations in the paper.

### 3. Differentiable Logic Synthesis — under-review companion [`paper/draft_nai.tex`]

Preprint: [arXiv:2601.13953](https://arxiv.org/abs/2601.13953). The original
synthesis architecture: differentiable spectral coefficient selection with
Sinkhorn-constrained composition, establishing certified ternary representability
for all Boolean functions through $n=4$. The present trilogy uses that certified
universe as a finite testbed for the Boolean-theory diagnostics in Paper 1.

## Quickstart: BBT-spectral on a small LLM

The toolkit at [`boolean_fourier/bbt_quant/`](boolean_fourier/bbt_quant/) drives
a calibration → WHT rotation → influence scaling → auto-round → dequant →
OpenVINO IR pipeline.

```bash
# WSL/Linux side: quantize
python -m boolean_fourier.bbt_quant.run_pipeline \
    --model HuggingFaceTB/SmolLM-135M \
    --output ./out/smollm135 \
    --bits 2 --alpha 0.5 --group-size 64 \
    --calib-samples 128 --seqlen 2048 \
    --bbt-mode spectral_pca_2d --skip-ov

# Dequantize for portability
python -m boolean_fourier.bbt_quant.dequantize_autoround \
    --src ./out/smollm135/autoround \
    --dst ./out/smollm135/fp16

# Evaluate wikitext-2 PPL (canonical, provenance-recording evaluator).
# Every result JSON records the live transformers/torch version, because
# absolute W2 perplexity is sensitive to the modeling-code version.
python -m boolean_fourier.bbt_quant.eval_ppl \
    --model ./out/smollm135/fp16 --device xpu \
    --out ./out/smollm135/eval.json
# To reproduce the paper's headline numbers exactly, pin transformers==4.57.6
# (see boolean_fourier/bbt_quant/requirements.txt) and add --paper-repro
# (= --max-tokens 8192 --seqlen 1024).
```

Five BBT modes are wired through `--bbt-mode`:
- `no_rotation` — input-side BBT scaling, norm-absorbed (every architecture)
- `spectral` — input-side WHT rotation + scaling
- `spectral_pca` — Route A: per-head PCA + matrix-Γ q_norm/k_norm replacement
  (Qwen3, DeepSeek-V4, Laguna)
- `spectral_pca_2d` — pair-PCA for RoPE architectures without q_norm/k_norm
  (SmolLM, Qwen2.5, Llama)
- `v1_pad_pre_hook` — legacy, broken (kept for diff-archaeology)

Sanity tests live alongside the toolkit
([`test_pca_invariance.py`](boolean_fourier/bbt_quant/test_pca_invariance.py),
[`test_moe_invariance.py`](boolean_fourier/bbt_quant/test_moe_invariance.py),
[`test_laguna_invariance.py`](boolean_fourier/bbt_quant/test_laguna_invariance.py))
verifying math invariance to fp32 numerical floor (rel err $\sim 10^{-7}$) on
synthetic fixtures pulled directly from each architecture's modeling source.

## Reproducibility

- **n=4 MILP** (all 65,536 functions): 9.9 minutes single-machine, 7 workers,
  `scipy.optimize.milp` (HiGHS). Output: `experiments/n4_min_support_mu.csv`.
- **n=5 NPN enumeration** (616,126 reps): 65 seconds with numba JIT.
  Output: `experiments/npn5_canonical.npy`.
- **n=5 NPN-uniform 10k sample MILP**: 11 minutes single-machine.
  Output: `experiments/n5_npn_min_support.csv`.
- **LLM quant runs**: ~10–60 minutes per model on Intel Arc B580 (12 GB) with
  the documented `batch_size` and per-block CPU eviction patches; see
  `boolean_fourier/bbt_quant/autoround_bbt.py:patch_autoround_for_xpu_memory`.

## Hardware tested

- Intel Core Ultra 5 225F (Lunar Lake) with AI Boost NPU
- Intel Arc B580 (12 GB)
- NVIDIA RTX 5060 (8 GB, via llama.cpp Vulkan backend for some smoke tests)
- WSL2 Ubuntu 24.04 (xpu_venv) for quantization, native Windows 11
  (ovvenv) for OpenVINO export and NPU/GPU/CPU evaluation

## Citing

```bibtex
@misc{pavlov2026bbt,
  title  = {The Banach-Butterfly Invariant: Influence-Adaptive Walsh Geometry
            for Ternary Polynomial Threshold Functions},
  author = {Gorgi Pavlov},
  year   = {2026},
  eprint = {arXiv:TBD},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@misc{pavlov2026bbtquant,
  title  = {Influence-Inspired Spectral Rotations for Extreme Low-Bit
            LLM Quantization},
  author = {Gorgi Pavlov},
  year   = {2026},
  eprint = {arXiv:TBD},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

MIT. See [`LICENSE`](LICENSE).

## Contact

Gorgi Pavlov, Ph.D. — `gorgipavlov@gmail.com` · `gop214@lehigh.edu`
